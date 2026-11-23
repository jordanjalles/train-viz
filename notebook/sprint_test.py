"""CPU test of sprint_cells.py against the user's notebook model code.

    python sprint_test.py NOTEBOOK.ipynb sprint_cells.py

Checks behavior, not only that it runs: token counts per block, zeros at skipped tokens,
equality with the notebook's FlowModel, and that the on/off switch reaches a compiled step.
"""
import json
import math
import re
import sys
import textwrap

import torch
import torch.nn as nn
import torch.nn.functional as F

nb_path, cells_path = sys.argv[1], sys.argv[2]
cells = ["".join(c["source"]) for c in json.load(open(nb_path))["cells"]]
text = open(cells_path).read()
blocks = dict(zip(re.findall(r"^# === (BLOCK \d)", text, re.M), re.split(r"^# === BLOCK \d.*\n", text, flags=re.M)[1:]))
assert len(blocks) == 6, list(blocks)
body = text[text.index("# === BLOCK 1"):]
long = [(i, l) for i, l in enumerate(body.splitlines(), 1) if len(l) > 72 and not l.startswith("# === ")]
assert not long, f"lines over 72 chars: {long}"

def cell(prefix):
    hits = [c for c in cells if c.lstrip().startswith(prefix)]
    assert len(hits) == 1, (prefix, len(hits))
    return hits[0]

# ---- the notebook's globals and model pieces ----
g = dict(torch=torch, nn=nn, F=F, math=math, device="cpu")
exec("num_blocks=10; n_heads=4; head_size=8; flow_emb_dim=32; n_latent_tokens=48; emb_dim=48", g)
for prefix in ("#multi-head attention", "class FlowSwiGLU", "class TimestepEmbedder", "#Flow block", "#define latent flow model"):
    exec(cell(prefix), g)
FlowModel = g["FlowModel"]
exec(blocks["BLOCK 1"], g)
assert (g["use_sparse_middle"], g["sparse_keep_ratio"]) == (True, 0.25) and "sparse_until_epoch" not in g
assert len(blocks["BLOCK 1"].strip().splitlines()) == 5

# 0. block 2 only defines: one class and two functions, nothing else
before = set(g)
exec(blocks["BLOCK 2"], g)
assert set(g) - before == {"SparseMiddleFlowModel", "set_sparse_middle", "check_sparse_middle"}, set(g) - before
print("0 ok: block 2 defines SparseMiddleFlowModel, set_sparse_middle, check_sparse_middle and nothing else")

# block 3 goes into the notebook's real init_models cell: the lines it replaces exist exactly once
init_src = cell("# -------------------- Init Models + Optimizer")
assert init_src.count("flow_model = FlowModel(") == 1 and init_src.count("FlowModel(") == 1
new_init = init_src.replace("flow_model = FlowModel(", "flow_model = SparseMiddleFlowModel(")
compile(new_init, "init_models", "exec")
call = re.search(r"    flow_model = SparseMiddleFlowModel\(\n(?:        .*\n)*?    \)\.to\(device\)\n", new_init).group(0)

torch.manual_seed(0)
for flag in (False, True):
    g["use_sparse_middle"] = flag
    exec(textwrap.dedent(call), g)   # the notebook's own call, with the one word changed
    assert type(g["flow_model"]).__name__ == "SparseMiddleFlowModel" and g["flow_model"].sparse_middle is flag
new = g["flow_model"]
assert (new.shallow_blocks, new.middle_end, new.keep_ratio) == (2, 8, 0.25)
g["sparse_keep_ratio"], g["sparse_shallow_blocks"] = 0.5, 1
other = g["SparseMiddleFlowModel"](g["emb_dim"], 32, 10, 4, 8)
assert (other.shallow_blocks, other.keep_ratio) == (1, 0.5), "first-cell values are not picked up"
g["sparse_keep_ratio"], g["sparse_shallow_blocks"] = 0.25, 2
print("3 ok: the one-word edit in init_models builds the sparse model from the first-cell values")

# give the zero-initialized parts real values, as in a trained model
with torch.no_grad():
    for p in new.parameters():
        p.add_(0.05 * torch.randn_like(p))
set_sparse_middle = g["set_sparse_middle"]
B, T = 6, 48
z, t = torch.randn(B, T, 48), torch.rand(B)

def run(model, **kw):
    seen, fused = [], []
    hooks = [b.register_forward_hook(lambda m, a, o: seen.append(a[0].shape[1])) for b in model.blocks]
    hooks.append(model.fusion.register_forward_hook(lambda m, a, o: fused.append(a[0].detach())))
    out = model(z, t, **kw)
    for h in hooks:
        h.remove()
    return out, seen, fused[0]

# 1. dense path = the notebook's FlowModel with the fusion layer between block 7 and block 8
new.train(); set_sparse_middle(False)
dense_out, seen, fused_in = run(new)
assert seen == [48] * 10
C = 32
plain = FlowModel(g["emb_dim"], C, 10, 4, 8)
res = plain.load_state_dict(new.state_dict(), strict=False)
assert res.unexpected_keys == ["fusion.weight", "fusion.bias"] and not res.missing_keys, res
with torch.no_grad():   # fusion that passes the deep path through -> must equal the plain FlowModel
    saved = new.fusion.weight.clone(), new.fusion.bias.clone()
    new.fusion.weight.copy_(torch.cat((torch.zeros(C, C), torch.eye(C)), dim=1)); new.fusion.bias.zero_()
    assert torch.equal(new(z, t), plain.train()(z, t)), "dense path differs from the notebook's FlowModel"
    new.fusion.weight.copy_(saved[0]); new.fusion.bias.copy_(saved[1])
print("1 ok: dense path with a pass-through fusion equals the notebook's FlowModel exactly")

# 2. sparse training step: shallow 2 and final 2 see 48 tokens, the 6 middle blocks see 12
set_sparse_middle(True)
out, seen, fused_in = run(new)
assert seen == [48, 48] + [12] * 6 + [48, 48], seen
shallow, deep = fused_in[..., :C], fused_in[..., C:]
kept = deep.abs().sum(-1) > 0
assert kept.sum(1).tolist() == [12] * B, kept.sum(1)
assert (deep[~kept] == 0).all() and (shallow.abs().sum(-1) > 0).all()
assert not torch.equal(kept[0], kept[1]), "every sample got the same subset"
out.square().mean().backward()
for name in ("fusion.weight", "blocks.5.MHA.QKV.weight", "blocks.0.MHA.QKV.weight", "blocks.9.MHA.QKV.weight"):
    grad = dict(new.named_parameters())[name].grad
    assert grad is not None and grad.abs().sum() > 0, f"no gradient in {name}"
_, _, fused2 = run(new)
assert not torch.equal(fused2[..., C:].abs().sum(-1) > 0, kept), "the subset does not change between steps"
print("2 ok: token counts", seen, "| 12 kept per sample, zeros elsewhere, new subset per sample and step")

# 3. no gradients (sampling, split losses) -> dense, also when the model is left in train mode
with torch.no_grad():
    out, seen, _ = run(new)
assert seen == [48] * 10 and torch.equal(out, dense_out.detach())
with torch.inference_mode():
    _, seen, _ = run(new)
assert seen == [48] * 10
new.eval(); _, seen, _ = run(new); new.train()
assert seen == [48] * 10
print("3 ok: no_grad / inference_mode / eval are dense")

# 4. the sparse_until_epoch lines in the training loop
states = []
for epoch in range(5):
    exec(textwrap.dedent(blocks["BLOCK 4"]), dict(g, epoch=epoch, sparse_until_epoch=3))
    states.append(new.sparse_middle)
assert states == [True, True, True, False, False], states
set_sparse_middle(True)
exec(textwrap.dedent(blocks["BLOCK 4"]), dict(g, epoch=9, sparse_until_epoch=None))
ns_b3 = {}
exec(blocks["BLOCK 3"], ns_b3)
assert ns_b3["sparse_until_epoch"] == 180
assert new.sparse_middle is True
print("4 ok: sparse_until_epoch=3 ->", states, "| None leaves the switch alone")

# 5. the switch reaches a compiled train step (the attribute is not frozen into the graph)
opt = torch.optim.SGD(new.parameters(), lr=0.0)
def train_step(z):
    opt.zero_grad(set_to_none=True)
    pred = new(z, t)
    pred.square().mean().backward()
    opt.step()
    return pred.detach()
compiled = torch.compile(train_step, dynamic=False)
p_sparse = compiled(z)
set_sparse_middle(False)
p_dense = compiled(z)
set_sparse_middle(True)
p_sparse2 = compiled(z)
assert torch.allclose(p_dense, dense_out.detach(), atol=1e-5), (p_dense - dense_out).abs().max()
assert not torch.allclose(p_sparse, p_dense, atol=1e-3) and not torch.allclose(p_sparse2, p_dense, atol=1e-3)
print("5 ok: compiled step is sparse, dense after set_sparse_middle(False), sparse again after (True)")

# 6. bfloat16 autocast path (as in the notebook's train step)
with torch.autocast("cpu", dtype=torch.bfloat16):
    out = new(z, t)
out.float().square().mean().backward()
print("6 ok: bfloat16 autocast forward + backward, output", out.dtype)

# 7. the check the guide's last step runs
exec("check_sparse_middle()", g)
print("ALL OK")
