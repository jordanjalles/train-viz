"""CPU test of unify_eval_cells.py with the model cells of the user's ImageNet-Unify.ipynb.

    python unify_eval_test.py UNIFY_NOTEBOOK.ipynb unify_eval_cells.py

The notebook's model cannot run as pushed (three call mismatches). FIXES below are applied to
a copy here, and the test prints them: the user has to make the same edits.
"""
import json
import math
import re
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

cells = ["".join(c["source"]) for c in json.load(open(sys.argv[1]))["cells"]]
text = open(sys.argv[2]).read()
blocks = dict(zip(re.findall(r"^# === (BLOCK \d)", text, re.M), re.split(r"^# === BLOCK \d.*\n", text, flags=re.M)[1:]))
assert len(blocks) == 5
body = text[text.index("# === BLOCK 1"):]
long = [l for l in body.splitlines() if len(l) > 72 and not l.startswith("# === ")]
assert not long, long
for b in ("BLOCK 3", "BLOCK 5"):
    assert len(blocks[b].strip().splitlines()) <= 12, b + " (action cell) is too long"


def cell(*needles):
    hits = [c for c in cells if all(n in c for n in needles)]
    assert len(hits) == 1, (needles, len(hits))
    return hits[0]


FIXES = [   # (cell needle, old text, new text, what)
    ("class Block(nn.Module)", "def forward(self, x, time_emb):", "def forward(self, x, rope, time_emb):",
     "Block.forward takes (x, time_emb) but CoreTransformer calls block(x, rope, time_emb)"),
    ("class Block(nn.Module)", "self.MHA( self.pre_norm1(x) * (1 + scale_MHA) + shift_MHA )", "self.MHA( self.pre_norm1(x) * (1 + scale_MHA) + shift_MHA, rope )",
     "Block calls self.MHA(x) but MHA.forward needs (x, rope)"),
    ("class ImageTransformer(nn.Module)", "def forward(self, images):", "def forward(self, images, time):",
     "ImageTransformer.forward takes (images) but uses `time`, and train_step calls model(noised_images, t)"),
]
g = dict(torch=torch, nn=nn, F=F, math=math, time=time, plt=plt, device="cpu", t_logit_mean=-0.8, t_logit_std=0.8)
exec("num_blocks=4; n_heads=4; head_size=16; emb_dim=64; patch_size=16; keep_ratio=0.25; shallow_blocks=1; final_blocks=1; c=3; lr=2e-3; beta1=0.9; beta2=0.99", g)
for needles in (("def apply_2d_rope",), ("class ChannelGroupResize",), ("class LinSkipPatchEmbed",), ("class LinSkipUnEmbed",), ("class MHA(nn.Module)",),
                ("class SwiGLU(nn.Module)",), ("class Block(nn.Module)",), ("class TimestepEmbedder",), ("class CoreTransformer(nn.Module)",),
                ("class ImageTransformer(nn.Module)",), ("def make_train_step",)):
    src = cell(*needles)
    for needle, old, new, what in FIXES:
        if needle in src and src.count(old) == 1:      # the user has since made these edits in the notebook
            src = src.replace(old, new)
    exec(src, g)
print("fixes applied to a copy of the notebook's model (the pushed model cannot run without them):")
for _, _, _, what in FIXES:
    print("  -", what)

torch.manual_seed(0)
model = g["ImageTransformer"](3, 16, 64, 4, 4, 16, 0.25, 1, 1)
g["model"] = model
g["optimizer"] = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(0.9, 0.99))
g["train_step"] = g["make_train_step"]()          # uncompiled: same code path, fast on the CPU


class CountingLoader(list):
    passes = 0
    def __iter__(self):
        type(self).passes += 1
        return super().__iter__()


torch.manual_seed(1)
g["train_dataloader"] = CountingLoader([(torch.rand(8, 3, 32, 32), torch.zeros(8)) for _ in range(6)])
g["test_dataloader"] = CountingLoader([(torch.rand(8, 3, 32, 32), torch.zeros(8)) for _ in range(6)])

before = set(g)
exec(blocks["BLOCK 2"], g)
assert set(g) - before == {"sample_train_t", "x_pred_v_loss", "get_eval_batches", "get_split_loss"}, set(g) - before
get_split_loss = g["get_split_loss"]

# 1. same images, same t, same noise -> the same number on every call; the model's mode is restored
model.train()
a, b = get_split_loss("test", n_batches=4), get_split_loss("test", n_batches=4)
assert torch.equal(a, b) and a.ndim == 0 and model.training
assert not torch.equal(a, get_split_loss("test", n_batches=4, seed=1))
assert not torch.equal(a, get_split_loss("train", n_batches=4))
print(f"1 ok: get_split_loss is repeatable ({a:.5f} twice), restores train mode, differs by seed and by split")

# 2. eval batches are loaded once per split
passes = CountingLoader.passes
get_split_loss("test", n_batches=4); get_split_loss("train", n_batches=4); get_split_loss("test", n_batches=2)
assert CountingLoader.passes == passes, "the loaders were iterated again"
print("2 ok: the eval images are cached; later calls do not touch the loaders")

# 3. it is the train step's loss: same t and noise draws -> same value (sparse middle off for the comparison)
model.core_transformer.sparse_middle = False
images = g["test_dataloader"][0][0]
zero_lr = torch.optim.SGD(model.parameters(), lr=0.0)
torch.manual_seed(7)
step_loss = g["train_step"](images, zero_lr)["loss"]
torch.manual_seed(7)
t = torch.sigmoid(torch.randn(8) * 0.8 - 0.8) if "* 0.8 - 0.8" in cell("def make_train_step") else torch.rand(8)
noise = torch.randn_like(images)
with torch.no_grad():
    mine = g["x_pred_v_loss"](images, t, noise).mean()
assert torch.allclose(step_loss, mine, rtol=1e-5), (step_loss, mine)
model.core_transformer.sparse_middle = True
print(f"3 ok: x_pred_v_loss equals the train step's loss for the same draws ({step_loss:.5f})")

# 4. the split loss runs dense even though training is sparse
seen = []
hooks = [blk.register_forward_hook(lambda m, a, o: seen.append(a[0].shape[1])) for blk in model.core_transformer.blocks]
get_split_loss("test", n_batches=1)
assert seen == [4, 4, 4, 4], seen
seen.clear(); model.train(); g["train_step"](images, zero_lr)
assert seen == [4, 1, 1, 4], seen
for h in hooks:
    h.remove()
print("4 ok: split loss sees all 4 patch tokens in every block; a train step's middle blocks see 1")

# 5. micro-batch cells: fix, overfit, plots
exec(blocks["BLOCK 3"], g)
assert g["fixed_micro_batch"].shape == (6, 3, 32, 32)
before = set(g)
exec(blocks["BLOCK 4"], g)
assert set(g) - before == {"plot_fixed_micro_batch", "plot_micro_batch_predictions", "overfit_micro_batch"}, set(g) - before
test_before = get_split_loss("test", n_batches=4)
small = blocks["BLOCK 5"].replace("steps=2000, report_every=200", "steps=400, report_every=100")
assert small != blocks["BLOCK 5"]
exec(small, g)
losses = g["micro_losses"]
micro_after, test_after = get_split_loss("micro"), get_split_loss("test", n_batches=4)
first, last = losses[:20].mean().item(), losses[-20:].mean().item()
assert losses.shape == (400,) and last < 0.6 * first, (first, last)
assert micro_after < test_after, (micro_after, test_after)
assert model.training
figs = [len(plt.figure(n).axes) for n in plt.get_fignums()]
assert figs == [6, 6 * 9], figs            # 6 targets; then 1 + 2 * 4 rows of 6
print(f"5 ok: 400 micro-batch steps: train-step loss {first:.4f} -> {last:.4f}; micro-loss {micro_after:.4f} is below test-loss {test_after:.4f}"
      f" (test was {test_before:.4f}); plots drawn: {figs} panels")
print("ALL OK")
