"""Dry run of the checklist on public/unify_next.html against the user's pushed notebook (CPU).

    python unify_checklist_test.py UNIFY_NOTEBOOK.ipynb

Applies steps 1 to 6 to a copy of the notebook with the same code blocks the page shows, then
does steps 7 to 9: a fresh namespace ("restart"), every cell in order down to the readouts
cell, then the training loop. Tiny settings and a fake ImageNet; nothing real is read.
"""
import contextlib
import io
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def block(file, bid):
    text = (HERE / file).read_text()
    m = re.search(rf"^# === BLOCK {bid}[: ].*?===\n(.*?)(?=^# === BLOCK |\Z)", text, re.M | re.S)
    return m.group(1).rstrip("\n") + "\n"


cells = ["".join(c["source"]) for c in json.load(open(sys.argv[1]))["cells"] if c["cell_type"] == "code"]


def find(prefix):
    hits = [i for i, c in enumerate(cells) if c.lstrip().startswith(prefix)]
    assert len(hits) == 1, (prefix, hits)
    return hits[0]


# ---- Part A of the page: steps 1 to 6 ----
i = find("class ChannelGroupResize(nn.Module):")
assert cells[i + 1].lstrip().startswith("class LinSkipPatchEmbed") and cells[i + 2].lstrip().startswith("class LinSkipUnEmbed")
cells[i] = block("unify_patch_cells.py", 1)                    # step 1
del cells[i + 1:i + 3]                                          # step 2
cells.insert(i + 1, block("unify_patch_cells.py", 2))          # step 3
cells.insert(find("train_ds = BufferedImageNet(") + 1, block("unify_eval_cells.py", 1))   # step 4
cells[find("# training readouts (definitions only)")] = block("unify_train_cells.py", 3)  # step 5
cells[find("# MAIN TRAINING LOOP")] = block("unify_train_cells.py", 4)                    # step 6
print("steps 1-6 applied: every cell the page names exists exactly once in the pushed notebook")

# ---- fake world and tiny settings ----
tmp = Path(tempfile.mkdtemp())
os.environ.update(ML_DATASETS=str(tmp / "datasets"), ML_MODELS=str(tmp / "models"), ML_CHECKPOINTS=str(tmp / "ckpt"))
os.chdir(tmp)
N_TRAIN, N_VAL, IMG = 192, 48, 32


def make_fake_imagenet():
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image
    root = tmp / "datasets" / "imagenet"
    root.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for split, shards, per in (("train", 4, N_TRAIN // 4), ("validation", 2, N_VAL // 2)):
        for s in range(shards):
            images, labels = [], []
            for _ in range(per):
                buf = io.BytesIO()
                Image.fromarray(rng.integers(0, 255, (40, 48, 3), dtype=np.uint8)).save(buf, format="PNG")
                images.append({"bytes": buf.getvalue(), "path": None})
                labels.append(int(rng.integers(0, 1000)))
            pq.write_table(pa.table({"image": images, "label": labels}), root / f"{split}-{s:05d}-of-{shards:05d}.parquet")


make_fake_imagenet()
first = find("from pathlib import Path")
cells[first] += ("\n# tiny test settings\nnum_blocks = 4; n_heads = 4; head_size = 16; emb_dim = 64\n"
                 f"IMAGE_SIZE = {IMG}; CACHE_BATCH_SIZE = 16; ALLOW_PARTIAL_IMAGENET = True; lr = 1e-3\n")
SUBS = [("num_images=1_281_167", f"num_images={N_TRAIN}"), ("num_images=50_000", f"num_images={N_VAL}"),
        ("num_epochs = 20", "num_epochs = 1"), ("report_every = 500", "report_every = 3"),
        ("eval_every = 2_000", "eval_every = 6"), ("preview_every = 10_000", "preview_every = 12")]
for a, b in SUBS:
    assert sum(c.count(a) for c in cells) == 1, (a, sum(c.count(a) for c in cells))
    cells = [c.replace(a, b) for c in cells]
SKIP = ("print(torch.__version__)",)          # prints the CUDA device name


def run(ns, indices):
    out = {}
    for k in indices:
        src = cells[k]
        if not src.strip() or src.lstrip().startswith(SKIP) or all(l.strip().startswith("#") or not l.strip() for l in src.splitlines()):
            continue
        buf = io.StringIO()
        t0 = time.time()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            exec(compile(src, f"<cell {k}>", "exec"), ns)
        out[k] = buf.getvalue()
        import matplotlib.pyplot as plt
        plt.close("all")
        first_line = next(l for l in src.splitlines() if l.strip())[:58]
        print(f"  ran cell {k:>2} in {time.time() - t0:5.1f}s | {first_line}", flush=True)
    return out


# ---- Part B: step 7 (restart = fresh namespace), step 8 (top down to the readouts cell) ----
ns = {"__name__": "__main__"}
readouts = find("# training readouts (definitions only)")
out = run(ns, range(0, readouts + 1))
check = out[find("# Does the fixed skip path keep colour?")]
assert "red   in -> out RGB [1.0, 0.0, 0.0]" in check and "blue  in -> out RGB [0.0, 0.0, 1.0]" in check, check
init_out = out[next(k for k, c in enumerate(cells) if ") = init_models()" in c)]
assert "ae parameters" in init_out, init_out
assert type(ns["model"].embed.skip).__name__ == "PatchPoolSkip" and "test_dataloader" in ns
print("step 8 ok: all cells down to the readouts cell run; the color check prints the three expected lines;", init_out.strip())

# ---- step 9: the training loop ----
loop = find("# MAIN TRAINING LOOP")
text = run(ns, [loop])[loop]
lines = text.splitlines()
assert lines[0].startswith("START step 0 | train-loss: ") and "best test" in lines[0], lines[0]
n_steps = len(ns["logs"]["loss"])
counts = (sum(l.startswith("step ") for l in lines), sum(l.startswith("EVAL step") for l in lines), sum(l.startswith("EPOCH 1") for l in lines))
cached = len(ns["get_eval_batches"].cache["train"])
print(f"           loop steps in the epoch: {n_steps}; train eval batches cached by the START line: {cached}")
assert n_steps >= 6 and counts == (n_steps // 3, n_steps // 6, 1), (n_steps, counts, lines)
print("step 9 ok:", lines[0])
print("          ", next(l for l in lines if l.startswith("step ")))

# ---- later: stop and go on. Kill the loader workers as a kernel stop does, then run the loop cell again ----
import signal
try:
    for w in ns["train_dataloader"]._iterator._workers:
        os.kill(w.pid, signal.SIGKILL)
    time.sleep(1.0)
except RuntimeError:
    pass
text = run(ns, [loop])[loop]
assert "data loader: workers had stopped, starting new ones" in text and f"START step {n_steps} |" in text, text[:300]
assert len(ns["logs"]["loss"]) == 2 * n_steps
print(f"later ok: with the workers killed, running the loop cell again prints the restart line and continues at step {n_steps}")

# ---- the generated samples are no longer grey by construction: the model's untrained output has color ----
with torch.no_grad():
    sample = ns["denoise_seeds"]([0, 1], steps=4)
chroma = (sample - sample.mean(1, keepdim=True)).abs().mean().item()
assert chroma > 1e-3, chroma
print(f"samples: mean distance of R, G, B from grey = {chroma:.3f} after {2 * n_steps} tiny training steps (old patch cells: about 0.0001 at init)")
print("ALL OK")
