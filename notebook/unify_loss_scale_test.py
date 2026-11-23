"""CPU test: the split loss is on the scale of the train-step loss once both draw t the same way.

    python unify_loss_scale_test.py UNIFY_NOTEBOOK.ipynb unify_eval_cells.py

Uses the notebook's model and train step. Shows the gap with uniform t (the old split loss),
then checks the new split loss and the page's three edits.
"""
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
assert not [l for l in blocks["BLOCK 2"].splitlines() if len(l) > 72]


def cell(*needles):
    hits = [c for c in cells if all(n in c for n in needles)]
    assert len(hits) == 1, (needles, len(hits))
    return hits[0]


g = dict(torch=torch, nn=nn, F=F, math=math, device="cpu")
exec("num_blocks=4; n_heads=4; head_size=16; emb_dim=64; patch_size=16; keep_ratio=0.25; shallow_blocks=1; final_blocks=1; c=3", g)
exec("t_logit_mean = -0.8\nt_logit_std = 0.8", g)                 # edit 1: two lines in the first cell
for n in ("def apply_2d_rope", "class ChannelGroupResize", "class LinSkipPatchEmbed", "class LinSkipUnEmbed", "class MHA(nn.Module)", "class SwiGLU(nn.Module)",
          "class Block(nn.Module)", "class TimestepEmbedder", "class CoreTransformer(nn.Module)", "class ImageTransformer(nn.Module)"):
    exec(cell(n), g)
step_src = cell("def make_train_step")
OLD, NEW = ") * 0.8 - 0.8", ") * t_logit_std + t_logit_mean"   # edit 2: one line in the train step
assert step_src.count(OLD) == 1, step_src.count(OLD)
exec(step_src.replace(OLD, NEW), g)
torch.manual_seed(0)
model = g["ImageTransformer"](3, 16, 64, 4, 4, 16, 0.25, 1, 1)
with torch.no_grad():
    for p in model.parameters():
        p.add_(0.02 * torch.randn_like(p))
model.core_transformer.sparse_middle = False        # compare like with like: dense in both
g["model"] = model
torch.manual_seed(1)
g["train_dataloader"] = [(torch.rand(16, 3, 32, 32), torch.zeros(16)) for _ in range(20)]
g["test_dataloader"] = [(torch.rand(16, 3, 32, 32), torch.zeros(16)) for _ in range(20)]
exec(blocks["BLOCK 2"], g)                          # edit 3: the split-loss cell

# the train step's own loss, averaged over many steps (learning rate 0, so the weights do not move)
zero_lr = torch.optim.SGD(model.parameters(), lr=0.0)
step = g["make_train_step"]()
torch.manual_seed(3)
step_losses = torch.stack([step(images, zero_lr)["loss"] for images, _ in g["train_dataloader"] * 10])
step_mean = step_losses.mean().item()

# the old split loss: the same formula with uniform t
rng = torch.Generator().manual_seed(0)
old = []
with torch.no_grad():
    for images, _ in g["train_dataloader"]:
        t = torch.rand(16, generator=rng)
        old.append(g["x_pred_v_loss"](images, t, torch.randn(images.shape, generator=rng)))
old = torch.cat(old)
uniform_mean = old.mean().item()
new_mean = g["get_split_loss"]("train").item()

print(f"train-step loss, mean of 200 steps:        {step_mean:9.4f}   (single steps from {step_losses.min():.3f} to {step_losses.max():.3f})")
print(f"OLD split loss, uniform t:                 {uniform_mean:9.4f}   = {uniform_mean / step_mean:6.1f} x the train-step loss")
share = old.sort(descending=True).values[:len(old) // 20].sum().item() / old.sum().item()
print(f"   the 5% of images with the largest loss carry {share:.0%} of it (t near 1, where the loss divides by (1 - t))")
print(f"NEW split loss, t drawn as in train_step:  {new_mean:9.4f}   = {new_mean / step_mean:6.2f} x the train-step loss")
assert uniform_mean > 5 * step_mean, "expected the uniform-t loss to be far above the train-step loss"
assert 0.7 < new_mean / step_mean < 1.4, new_mean / step_mean

# sample_train_t is the train step's distribution, and the fixed-seed loss still repeats exactly
t = g["sample_train_t"](200_000)
assert abs(torch.logit(t).mean().item() + 0.8) < 0.01 and abs(torch.logit(t).std().item() - 0.8) < 0.01
assert torch.equal(g["get_split_loss"]("test"), g["get_split_loss"]("test"))
torch.manual_seed(7)
a = step(g["test_dataloader"][0][0], zero_lr)["loss"]
torch.manual_seed(7)
tt = torch.sigmoid(torch.randn(16) * 0.8 - 0.8); nn_ = torch.randn(16, 3, 32, 32)
with torch.no_grad():
    b = g["x_pred_v_loss"](g["test_dataloader"][0][0], tt, nn_).mean()
assert torch.allclose(a, b, rtol=1e-5), (a, b)
print(f"ok: sample_train_t has logit mean {torch.logit(t).mean():.2f} and std {torch.logit(t).std():.2f}; share of t above 0.95: {(t > 0.95).float().mean():.5f}"
      f" (uniform: 0.05)")
print("ok: the split loss repeats exactly, and the edited train step still equals x_pred_v_loss for the same draws")
print("ALL OK")
