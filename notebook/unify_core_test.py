"""CPU test of unify_core_cells.py against the user's current classes.

    python unify_core_test.py AE_NOTEBOOK.ipynb unify_core_cells.py

Reference = CoreTransformer and SparseMiddleTransformer of ImageNet-AE-selfteach.ipynb, with
that notebook's working MHA / SwiGLU / Block (same Block(x, rope) interface as the Unify file).
"""
import builtins
import json
import math
import re
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

cells = ["".join(c["source"]) for c in json.load(open(sys.argv[1]))["cells"]]
text = open(sys.argv[2]).read()
blocks = dict(zip(re.findall(r"^# === (BLOCK \d)", text, re.M), re.split(r"^# === BLOCK \d.*\n", text, flags=re.M)[1:]))
body = text[text.index("# === BLOCK 1"):]
long = [l for l in body.splitlines() if len(l) > 72 and not l.startswith("# === ")]
assert not long, long


def cell(*needles):
    hits = [c for c in cells if all(n in c for n in needles)]
    assert len(hits) == 1, (needles, len(hits))
    return hits[0]


g = dict(torch=torch, nn=nn, F=F, math=math)
for needles in (("def apply_2d_rope",), ("class MHA(nn.Module)",), ("class SwiGLU(nn.Module)",), ("class Block(nn.Module)",),
                ("class CoreTransformer(nn.Module)",), ("class SparseMiddleTransformer(CoreTransformer)",)):
    exec(cell(*needles), g)
OldCore, OldSparse = g["CoreTransformer"], g["SparseMiddleTransformer"]
before = set(g)
exec(blocks["BLOCK 1"], g)
New = g["CoreTransformer"]
assert New is not OldCore and set(g) - before <= {"CoreTransformer"}, set(g) - before

# no global reads: every name the class looks up is a builtin, a module, or one of the model parts
allowed = set(dir(builtins)) | {"torch", "nn", "math", "Block", "CoreTransformer", "F"}
used = set()
for fn in vars(New).values():
    code = getattr(getattr(fn, "__func__", fn), "__code__", None)
    if code:
        used |= {n for n in code.co_names if n in g or n in dir(builtins) or n.islower()} & (set(g) | set(dir(builtins)))
stray = {n for n in used if n not in allowed and not hasattr(torch.Tensor, n) and not hasattr(nn.Module, n)}
assert not stray, f"names read from the notebook: {stray}"
print("0 ok: block 1 defines only CoreTransformer; it reads no notebook globals:", sorted(used & (set(g) - set(dir(builtins)))))

E, NB, NH, HS, NL = 64, 6, 4, 16, 8
torch.manual_seed(0)
old = OldCore(E, NB, NH, HS, NL)
with torch.no_grad():
    for p in old.parameters():
        p.add_(0.05 * torch.randn_like(p))
new = New(E, NB, NH, HS, NL)
assert sorted(new.state_dict()) == sorted(old.state_dict()), "dense: weight names differ from the old CoreTransformer"
new.load_state_dict(old.state_dict())
assert not any("cache" in k for k, _ in new.named_buffers()), "a RoPE cache buffer is left"
x4, x8 = torch.randn(3, NL + 16, E), torch.randn(3, NL + 64, E)

# 1. dense: same output as the old CoreTransformer, in eval and in train (same jitter draw), with and without masking
old.eval(); new.eval()
with torch.no_grad():
    assert torch.equal(new(x4, 4, 4), old(x4, 4, 4)) and torch.equal(new(x8, 8, 8), old(x8, 8, 8))
    keep = torch.rand(3, 16).argsort(1)[:, :10]
    xm = torch.cat((x4[:, :NL], x4[:, NL:].gather(1, keep[:, :, None].expand(-1, -1, E))), dim=1)
    assert torch.equal(new(xm, 4, 4, keep_indices=keep), old(xm, 4, 4, keep_indices=keep))
old.train(); new.train()
torch.manual_seed(5); a = old(x4, 4, 4)
torch.manual_seed(5); b = new(x4, 4, 4)
assert torch.equal(a, b)
print("1 ok: keep_ratio=1 equals the old CoreTransformer (eval, train, input masking, two grids); same weight names")

# 2. sparse: same output as the old SparseMiddleTransformer for the same weights and random draws
torch.manual_seed(1)
olds = OldSparse(E, NB, NH, HS, NL, keep_ratio=0.25, shallow_blocks=1, final_blocks=1)
with torch.no_grad():
    for p in olds.parameters():
        p.add_(0.05 * torch.randn_like(p))
news = New(E, NB, NH, HS, NL, keep_ratio=0.25, shallow_blocks=1, final_blocks=1)
assert sorted(news.state_dict()) == sorted(olds.state_dict())
news.load_state_dict(olds.state_dict())
olds.train(); news.train()
for xx, grid in ((x4, 4), (x8, 8)):
    torch.manual_seed(9); a = olds(xx, grid, grid)
    torch.manual_seed(9); b = news(xx, grid, grid)
    assert torch.equal(a, b), grid
torch.manual_seed(9); a = olds(xm, 4, 4, keep_indices=keep)
torch.manual_seed(9); b = news(xm, 4, 4, keep_indices=keep)
assert torch.equal(a, b)
seen = []
hooks = [blk.register_forward_hook(lambda m, args, out: seen.append(args[0].shape[1])) for blk in news.blocks]
news(x8, 8, 8)
assert seen == [72, 18, 18, 18, 18, 72], seen          # 8 latents + 64 patches -> 2 + 16 kept
seen.clear(); news.eval()
with torch.no_grad():
    assert torch.equal(news(x8, 8, 8), olds.eval()(x8, 8, 8))
assert seen == [72] * 6
seen.clear(); news.train(); news.sparse_middle = False
news(x8, 8, 8)
assert seen == [72] * 6
for h in hooks:
    h.remove()
news.sparse_middle = True
print("2 ok: keep_ratio=0.25 equals the old SparseMiddleTransformer (train with the same draws, eval, masking);"
      " middle blocks see 18 of 72 tokens; sparse_middle=False is dense")

# 3. gradients reach every part in a sparse step
news(x8, 8, 8).square().mean().backward()
for name in ("fusion.weight", "blocks.0.MHA.QKV.weight", "blocks.3.MHA.QKV.weight", "blocks.5.MHA.QKV.weight"):
    assert dict(news.named_parameters())[name].grad.abs().sum() > 0, name
print("3 ok: gradients reach fusion, shallow, middle and final blocks")

# 4. the case that broke the multi-resolution run: a compiled eval-mode forward at alternating grids
graphs = {"new": 0, "old": 0}


def spy(which):
    def backend(gm, example_inputs):
        graphs[which] += 1
        return gm.forward
    return backend


new.eval(); old.eval()
cn = torch.compile(lambda x, r: new(x, r, r), backend=spy("new"), dynamic=False)
co = torch.compile(lambda x, r: old(x, r, r), backend=spy("old"), dynamic=False)
with torch.no_grad():
    for xx, grid in ((x4, 4), (x8, 8), (x4, 4), (x8, 8), (x4, 4), (x8, 8)):
        assert torch.allclose(cn(xx, grid), co(xx, grid), atol=1e-5)
assert graphs["new"] == 2, graphs
print(f"4 ok: alternating 4x4 and 8x8 grids three times: new core compiled {graphs['new']} graphs, old core {graphs['old']}")

# 5. no patches at all (a latent-only model): every token is latent
flow_like = New(E, NB, NH, HS, NL, keep_ratio=0.5, shallow_blocks=1, final_blocks=1).train()
out = flow_like(torch.randn(2, 12, E), 0, 0)
assert out.shape == (2, 12, E)
print("5 ok: patch_rows = patch_cols = 0 works (latent tokens only)")

# 6. the check cell
exec(blocks["BLOCK 2"], dict(g, n_heads=NH, head_size=HS, num_blocks=NB))
print("ALL OK")
