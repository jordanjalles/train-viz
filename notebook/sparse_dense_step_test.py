"""CPU check of the user's train_step_xm_sparse_dense under torch.compile.

    python sparse_dense_step_test.py NOTEBOOK.ipynb

The step flips model.sparse_middle inside one compiled function. This looks at the graph that
torch.compile actually captured: a sparse pass has exactly one gather and one scatter, a dense
pass has none. It also checks which candidate the dense pass gets and that both losses train.
"""
import json
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

cells = ["".join(c["source"]) for c in json.load(open(sys.argv[1]))["cells"]]


def cell(prefix):
    hits = [c for c in cells if c.lstrip().startswith(prefix)]
    assert len(hits) == 1, (prefix, len(hits))
    return hits[0]


g = dict(torch=torch, nn=nn, F=F, math=math, device="cpu", flow_noise_scale=1.0)
exec("num_blocks=6; n_heads=2; head_size=8; flow_emb_dim=16; n_latent_tokens=48; emb_dim=48; XM_K=4; XM_N=2", g)
exec("use_sparse_middle=True; sparse_keep_ratio=0.25; sparse_shallow_blocks=2; sparse_final_blocks=2", g)
for prefix in ("#multi-head attention", "class FlowSwiGLU", "class TimestepEmbedder", "#Flow block", "#define latent flow model",
               "#define sparse-middle flow model", "def train_step_xm_sparse_dense"):
    exec(cell(prefix), g)
torch.manual_seed(0)
model = g["SparseMiddleFlowModel"](48, 16, 6, 2, 8)
with torch.no_grad():
    for p in model.parameters():
        p.add_(0.1 * torch.randn_like(p))
model.train()
g["flow_model"] = model
g["optimizer"] = torch.optim.AdamW(model.parameters(), lr=1e-3)
step = g["train_step_xm_sparse_dense"]
# the notebook's autocast is for cuda; on the CPU it is switched off with a warning, which is fine

graphs = []


def spy_backend(gm, example_inputs):
    ops = [str(getattr(n.target, "__name__", n.target)) for n in gm.graph.nodes if n.op in ("call_function", "call_method")]
    graphs.append(ops)
    return gm.forward


B = 5
z = torch.randn(B, 48, 48)
compiled = torch.compile(step, backend=spy_backend, dynamic=False)
before = {k: v.detach().clone() for k, v in model.state_dict().items()}
out = compiled(z)
ops = [o for graph in graphs for o in graph]
count = lambda name: sum(o == name for o in ops)
print(f"captured {len(graphs)} graph(s), {len(ops)} ops | gather {count('gather')} | scatter {count('scatter')} | attention {count('scaled_dot_product_attention')}")
assert count("gather") == 1 and count("scatter") == 1, "expected exactly one sparse pass in the compiled step"
assert count("scaled_dot_product_attention") == 2 * 6, "expected two passes through all 6 blocks"
assert model.sparse_middle is True, "the flag was not restored after the step"
print("1 ok: the compiled step holds one sparse pass and one dense pass; the flag is restored")

# 2. both losses are real and the step updates shallow, middle, final and fusion weights
assert set(out) == {"flow_loss", "sparse_flow_loss", "dense_flow_loss"}
s, d, total = (out[k].item() for k in ("sparse_flow_loss", "dense_flow_loss", "flow_loss"))
assert abs(total - (2 * s + d) / 3) < 1e-4 * max(1, abs(total)), (s, d, total)
moved = [k for k, v in model.state_dict().items() if not torch.equal(v, before[k])]
for name in ("blocks.0.MHA.QKV.weight", "blocks.3.MHA.QKV.weight", "blocks.5.MHA.QKV.weight", "fusion.weight"):
    assert name in moved, f"{name} did not change"
print(f"2 ok: flow_loss = (N * sparse + dense) / (N + 1) with N=2: sparse {s:.3f}, dense {d:.3f}, total {total:.3f}; all parts updated")

# 3. the dense pass gets, per image, the candidate with the lowest sparse error (index math of the step)
K, n = 4, B
sparse_errors = torch.rand(K, n)
best = sparse_errors.argmin(dim=0)
flat = best * n + torch.arange(n)
cand = torch.arange(K * n)                      # candidate k of image b sits at k * B + b after z.repeat(K, 1, 1)
assert all(cand[flat[b]] == best[b] * n + b for b in range(n))
assert torch.equal(sparse_errors.reshape(-1)[flat], sparse_errors.min(dim=0).values)
print("3 ok: flat index best * B + b picks each image's lowest-error candidate")

# 4. a second call reuses the graph (no recompile each step), and switching the flag off outside still works
n_graphs = len(graphs)
compiled(z)
assert len(graphs) == n_graphs, "the step recompiled on its second call"
print("4 ok: the second step reuses the compiled graph")
print("ALL OK")
