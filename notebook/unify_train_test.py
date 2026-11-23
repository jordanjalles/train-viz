"""CPU test of unify_train_cells.py with the cells of the user's ImageNet-Unify.ipynb.

    python unify_train_test.py UNIFY_NOTEBOOK.ipynb unify_train_cells.py

Uses the notebook's own model, train step and split-loss cells with a tiny configuration and
fake loaders.
"""
import contextlib
import io
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
parts = re.split(r"^# === BLOCK (\w+): .*? ===\n", text, flags=re.M)[1:]
blocks = {parts[i]: parts[i + 1] for i in range(0, len(parts), 2)}
assert list(blocks) == ["1", "2", "3", "4", "5", "6A", "6B", "6C", "6D", "6E", "6F", "6G"], list(blocks)
body = text[text.index("# === BLOCK 1"):]
long = [l for l in body.splitlines() if len(l) > 72 and not l.startswith("# === ")]
assert not long, long
for b in ("6A", "6B", "6C", "6D", "6E", "6F", "6G"):
    assert len(blocks[b].strip().splitlines()) <= 6, b


def cell(*needles):
    hits = [c for c in cells if all(n in c for n in needles)]
    assert len(hits) == 1, (needles, len(hits))
    return hits[0]


g = dict(torch=torch, nn=nn, F=F, math=math, time=time, plt=plt, device="cpu", IMAGE_SIZE=32)
exec("num_blocks=4; n_heads=4; head_size=16; emb_dim=64; patch_size=16; keep_ratio=0.25; shallow_blocks=1; final_blocks=1; c=3", g)
for needles in (("def apply_2d_rope",), ("class ChannelGroupResize",), ("class LinSkipPatchEmbed",), ("class LinSkipUnEmbed",), ("class MHA(nn.Module)",),
                ("class SwiGLU(nn.Module)",), ("class Block(nn.Module)",), ("class TimestepEmbedder",), ("class CoreTransformer(nn.Module)",),
                ("class ImageTransformer(nn.Module)",), ("def get_split_loss",), ("def make_train_step",)):
    exec(cell(*needles), g)
torch.manual_seed(0)
model = g["ImageTransformer"](3, 16, 64, 4, 4, 16, 0.25, 1, 1)
g["model"] = model
g["optimizer"] = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(0.9, 0.99))
g["train_step"] = g["make_train_step"]()
torch.manual_seed(1)
g["train_dataloader"] = [(torch.rand(8, 3, 32, 32), torch.zeros(8)) for _ in range(12)]

for bid, names in (("1", {"defaultdict", "logs", "eval_logs", "clear_logs", "log_metrics", "get_metrics_lists"}),
                   ("2", {"make_timesteps", "denoise_seeds", "denoise_seed", "_show", "plot_seeds", "plot_seed_trajectory", "compare_samplers"}),
                   ("3", {"TrainReadout", "ensure_loader_alive", "evaluate", "preview"})):
    before = set(g)
    exec(blocks[bid], g)
    assert set(g) - before == names, (bid, set(g) - before)
g["logs"]["loss"].append(1.0)
exec(blocks["1"], g)
assert g["logs"]["loss"] == [1.0], "running the log cell again cleared the logs"
g["clear_logs"]()
print("0 ok: blocks 1, 2 and 3 only define; the log cell keeps the logs when it is run again")

# 1. time grid
ts = g["make_timesteps"](20)
assert len(ts) == 21 and ts[0] == 0 and ts[-1] == 1 and (ts[1:] > ts[:-1]).all()
shifted = g["make_timesteps"](20, shift=3.0)
assert (shifted[1:] > shifted[:-1]).all() and shifted[10] < ts[10], "shift > 1 must spend more steps at high noise"
print("1 ok: 21 rising times from 0 to 1; shift=3 puts the midpoint at t=%.2f" % shifted[10])

# 2. a model that always predicts one fixed image must be sampled to exactly that image
target = torch.rand(3, 32, 32)


class Perfect(nn.Module):
    training = False
    def __init__(self):
        super().__init__()
        self.un_embed = model.un_embed
    def forward(self, x, t):
        return target.expand(x.shape[0], -1, -1, -1).clone()


g["model"] = Perfect()
for sampler in ("euler", "heun"):
    for steps in (1, 7, 20):
        out = g["denoise_seeds"]([3, 4], steps, sampler)
        assert torch.allclose(out, target.expand(2, -1, -1, -1), atol=1e-5), (sampler, steps, (out - target).abs().max())
g["model"] = model
print("2 ok: with a model that predicts one fixed image, euler and heun land on it exactly (1, 7 and 20 steps)")

# 3. one Euler step is the model's prediction from pure noise at t=0
model.eval()
noise = torch.randn((3, 32, 32), generator=torch.Generator().manual_seed(5))
with torch.no_grad():
    want = model(noise[None], torch.zeros(1)).float().clamp(0, 1)
assert torch.allclose(g["denoise_seed"](5, steps=1), want, atol=1e-5)
print("3 ok: denoise_seed(seed, steps=1) equals model(noise of that seed, t=0)")

# 4. seeds: repeatable, different per seed, independent of the other seeds in the call; mode restored; other sizes
model.train()
a = g["denoise_seeds"]([1, 2, 3], 6)
assert model.training
assert torch.equal(a, g["denoise_seeds"]([1, 2, 3], 6))
assert torch.allclose(a[1:2], g["denoise_seed"](2, 6), atol=1e-5) and not torch.allclose(a[0], a[1], atol=1e-3)
assert g["denoise_seeds"]([1], 3, image_size=64).shape == (1, 3, 64, 64)
seen = []
hooks = [blk.register_forward_hook(lambda m, x, o: seen.append(x[0].shape[1])) for blk in model.core_transformer.blocks]
g["denoise_seeds"]([1], 2)
for h in hooks:
    h.remove()
assert set(seen) == {4}, seen
print("4 ok: same seed -> same image; a seed's image does not depend on its batch; 64 px works; sampling is dense")

# 5. the training loop cell, two epochs of 12 steps, with and without a test loader
def run_loop():
    src = blocks["4"]
    for old, new in (("num_epochs = 20", "num_epochs = 2"), ("report_every = 500", "report_every = 5"),
                     ("eval_every = 2_000", "eval_every = 8"), ("preview_every = 10_000", "preview_every = 12")):
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(src, g)
    return out.getvalue()


plt.close("all")
text1 = run_loop()
lines = text1.splitlines()
assert lines[0].startswith("START step 0 | train-loss: ") and "test-loss: n/a (no test_dataloader)" in lines[0], lines[0]
reports = [l for l in lines if l.startswith("step ")]
assert [int(l.split()[1].replace(",", "")) for l in reports] == [5, 10, 15, 20], reports
assert all(k in reports[0] for k in ("epoch 1", "ms/step", "img/s", "min/epoch", "loss ", "lr 2.0e-03")), reports[0]
assert sum(l.startswith("EVAL step") for l in lines) == 3 and sum(l.startswith("EPOCH ") for l in lines) == 2
assert len(g["logs"]["loss"]) == 24 and g["eval_logs"]["step"] == [0, 8, 12, 16, 24, 24], g["eval_logs"]["step"]
assert len(plt.get_fignums()) == 2, "expected a preview at steps 12 and 24"
assert model.training
torch.manual_seed(2)
g["test_dataloader"] = [(torch.rand(8, 3, 32, 32), torch.zeros(8)) for _ in range(4)]
text2 = run_loop()
first = text2.splitlines()[0]
assert first.startswith("START step 24 |") and "best test" in first, first
assert len(g["logs"]["loss"]) == 48
print("5 ok: loop prints START, 4 speed lines per run, EVAL and EPOCH lines; logs 24 steps per run; previews at steps 12 and 24;")
print("      a second run continues the step count (START step 24); without test_dataloader it prints 'n/a'")
print("      sample line:", reports[0])

# 5b. the user's resume error: a persistent-worker loader whose workers were killed (as a kernel interrupt does)
import os
import signal
from torch.utils.data import DataLoader, TensorDataset

loader = DataLoader(TensorDataset(torch.rand(64, 3, 8, 8), torch.zeros(64)), batch_size=8, num_workers=2,
                    persistent_workers=True, multiprocessing_context="fork")
assert len([b for b in loader]) == 8
try:
    for w in loader._iterator._workers:
        os.kill(w.pid, signal.SIGKILL)
    time.sleep(1.0)
except RuntimeError:
    pass            # torch's child-signal handler may report the kill right here
try:
    [b for b in loader]
    raise AssertionError("expected the dead-worker error")
except RuntimeError as e:
    assert "exited unexpectedly" in str(e), e
out = io.StringIO()
with contextlib.redirect_stdout(out):
    g["ensure_loader_alive"](loader)
assert "starting new ones" in out.getvalue()
assert len([b for b in loader]) == 8
out = io.StringIO()
with contextlib.redirect_stdout(out):
    g["ensure_loader_alive"](loader)             # healthy loader: nothing to do, workers kept
assert out.getvalue() == "" and loader._iterator is not None
print("5b ok: killed workers give 'DataLoader worker ... exited unexpectedly'; ensure_loader_alive restarts them; a healthy loader is left alone")

# 6. history and seed-plot cells
plt.close("all")
out = io.StringIO()
with contextlib.redirect_stdout(out):
    exec(blocks["5"], g)
    exec(blocks["6A"], g)
assert "steps:           48" in out.getvalue() and len(plt.get_fignums()) == 1
ax = plt.gcf().axes[0]
assert len(ax.lines) == 3 and ax.get_yscale() == "log"       # train-step mean, train loss, test loss
plt.close("all")
for bid, panels in (("6B", 6), ("6C", 6), ("6D", 6), ("6E", 30), ("6F", 10), ("6G", 18)):   # 6F: 4 test steps -> 4 picks + final, two rows
    src = blocks[bid].replace("step_counts=(10, 20, 50)", "step_counts=(2, 3, 4)").replace("steps=20", "steps=4")
    exec(src, g)
    assert len(plt.gcf().axes) == panels, (bid, len(plt.gcf().axes))
    plt.close("all")
print("6 ok: history prints and plots 3 lines on a log axis; the six seed-plot cells draw 6, 6, 6, 30, 10 and 18 panels")
print("ALL OK")
