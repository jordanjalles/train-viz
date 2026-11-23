"""CPU test of the path-drop guidance blocks (5 and 6) of sprint_cells.py.

    python pdg_test.py NOTEBOOK.ipynb sprint_cells.py

Uses the notebook's own model and sampler cells with a tiny model and a fake decoder.
"""
import json
import math
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

nb_path, cells_path = sys.argv[1], sys.argv[2]
cells = ["".join(c["source"]) for c in json.load(open(nb_path))["cells"]]
text = open(cells_path).read()
blocks = dict(zip(re.findall(r"^# === (BLOCK \d)", text, re.M), re.split(r"^# === BLOCK \d.*\n", text, flags=re.M)[1:]))
for b in ("BLOCK 5", "BLOCK 6"):
    assert all(len(l) <= 72 for l in blocks[b].splitlines()), b + " has a line over 72 characters"
assert len(blocks["BLOCK 6"].strip().splitlines()) <= 12


def cell(prefix):
    hits = [c for c in cells if c.lstrip().startswith(prefix)]
    assert len(hits) == 1, (prefix, len(hits))
    return hits[0]


g = dict(torch=torch, nn=nn, F=F, math=math, plt=plt, device="cpu", flow_noise_scale=1.0)
exec("num_blocks=6; n_heads=2; head_size=8; flow_emb_dim=16; n_latent_tokens=48; emb_dim=48", g)
for prefix in ("#multi-head attention", "class FlowSwiGLU", "class TimestepEmbedder", "#Flow block", "#define latent flow model",
               '"""Paste this file into the notebook containing your model'):
    exec(cell(prefix), g)
exec(blocks["BLOCK 1"], g)
exec(blocks["BLOCK 2"], g)
torch.manual_seed(0)
model = g["SparseMiddleFlowModel"](48, 16, 6, 2, 8)
with torch.no_grad():
    for p in model.parameters():
        p.add_(0.1 * torch.randn_like(p))
g["flow_model"] = model
g["decoder"] = lambda z: torch.sigmoid(z.reshape(z.shape[0], 3, 24, 32))   # fake decoder: z -> image

before = set(g)
exec(blocks["BLOCK 5"], g)
assert set(g) - before == {"flow_velocity_weak", "denoise_seed_pdg", "path_drop_gap", "plot_path_drop_guidance"}, set(g) - before
print("0 ok: block 5 defines four functions and nothing else")

# an untrained model (the notebook's zero inits) gives no gap at all: the probe must say so
fresh = g["SparseMiddleFlowModel"](48, 16, 6, 2, 8)
g["flow_model"] = fresh
gaps = g["path_drop_gap"](n=4)
assert max(gaps) == 0.0, gaps
a, b = g["denoise_seed_pdg"](5, 0.0, 4), g["denoise_seed_pdg"](5, 3.0, 4)
assert torch.equal(a, b), "untrained model: guidance should change nothing"
g["flow_model"] = model
gaps = g["path_drop_gap"](n=4)
assert min(gaps) > 0.01, gaps
print("0b ok: untrained model -> gap 0% and identical samples at every guidance; trained-like model -> gap", [round(x, 3) for x in gaps])

z, t = torch.randn(3, 48, 48), torch.rand(3)
model.eval()

# 1. the weak pass runs the shallow and final blocks only
seen = []
hooks = [b.register_forward_hook(lambda m, a, o, i=i: seen.append(i)) for i, b in enumerate(model.blocks)]
weak = g["flow_velocity_weak"](z, t)
for h in hooks:
    h.remove()
assert seen == [0, 1, 4, 5], seen

# 2. it equals the model's own forward when the middle blocks output zeros
class Zero(nn.Module):
    def forward(self, x, time_emb):
        return torch.zeros_like(x)
saved = [model.blocks[i] for i in (2, 3)]
model.blocks[2], model.blocks[3] = Zero(), Zero()
with torch.no_grad():
    want = model(z, t)
model.blocks[2], model.blocks[3] = saved
assert torch.allclose(weak, want, atol=1e-6), (weak - want).abs().max()
with torch.no_grad():
    full = model(z, t)
assert (full - weak).abs().max() > 1e-2, "weak and full are the same: the test model is degenerate"
print("1 ok: weak pass = blocks", seen, "| equals the model with its middle blocks zeroed | differs from the full model")

# 3. guidance 1 reproduces the notebook's denoise_seed exactly; 0 and 2 differ from it
for sampler in ("euler", "heun", "midpoint"):
    ref = g["denoise_seed"](7, 6, sampler)
    assert torch.allclose(g["denoise_seed_pdg"](7, 1.0, 6, sampler), ref, atol=1e-5), sampler
out0, out2 = g["denoise_seed_pdg"](7, 0.0, 6), g["denoise_seed_pdg"](7, 2.0, 6)
ref = g["denoise_seed"](7, 6, "euler")
assert not torch.allclose(out0, ref, atol=1e-3) and not torch.allclose(out2, ref, atol=1e-3) and not torch.allclose(out0, out2, atol=1e-3)
print("2 ok: guidance 1 equals denoise_seed for euler, heun, midpoint; guidance 0 and 2 give other samples")

# 4. one guided Euler step is weak + s * (full - weak)
ts = g["make_timesteps"](1, "linear", device="cpu")
rng = torch.Generator().manual_seed(3)
z0 = torch.randn((1, 48, 48), generator=rng)
with torch.no_grad():
    t1 = ts[0].expand(1)
    w, f = g["flow_velocity_weak"](z0, t1), model(z0, t1)
    want = g["decoder"](z0 + (ts[1] - ts[0]) * (w + 2.5 * (f - w)))
assert torch.allclose(g["denoise_seed_pdg"](3, 2.5, 1), want, atol=1e-6)
print("3 ok: one step at guidance 2.5 equals weak + 2.5 * (full - weak)")

# 5. the model's mode is restored, and the plot cell runs. It must not need the notebook's _ode_step.
del g["_ode_step"]
model.train()
g["denoise_seed_pdg"](1, 2.0, 2)
assert model.training
exec(blocks["BLOCK 6"].replace("range(4)", "range(2)").replace("steps=20", "steps=3"), g)
assert len(plt.get_fignums()) == 1 and len(plt.gcf().axes) == 2 * 5
titles = [a.get_title() for a in plt.gcf().axes[:5]]
assert titles[0].startswith("weak model only") and titles[1].startswith("no guidance") and titles[3].startswith("guidance 2"), titles
print("4 ok: plot cell draws seeds x guidances:", titles)
print("ALL OK")
