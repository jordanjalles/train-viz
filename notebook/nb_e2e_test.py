"""End-to-end dry run of the user's flow notebook with the page edits applied (CPU, tiny, fake data).

    python nb_e2e_test.py NOTEBOOK.ipynb baseline|patched

baseline: the notebook as it is.  patched: flow_resume.html + sprint.html edits applied by the
same block files the pages are built from. Both use the same tiny settings, a fake ImageNet
(parquet shards of random images), a fake exported autoencoder and temp folders, so nothing
real is read or written and the GPU is not used.

Every code cell runs in notebook order in one namespace, like "run cells from the top". A cell
that fails is recorded and the run goes on. Compare the two modes: a cell that fails in both
fails because of the fake setup, not because of the edits.
"""
import contextlib
import io
import json
import os
import re
import signal
import sys
import tempfile
import textwrap
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
nb_path, MODE = sys.argv[1], sys.argv[2]
CELL_TIMEOUT = 240

# ------------------------------------------------------------------ fake world
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


def make_fake_ae():
    class Enc(nn.Module):
        def __init__(self):
            super().__init__(); self.lin = nn.Linear(48, 48 * 48)
        def forward(self, x):
            return self.lin(F.adaptive_avg_pool2d(x, 4).flatten(1)).view(-1, 48, 48)
    class Dec(nn.Module):
        def __init__(self):
            super().__init__(); self.lin = nn.Linear(48 * 48, 3 * IMG * IMG)
        def forward(self, z):
            return torch.sigmoid(self.lin(z.flatten(1))).view(-1, 3, IMG, IMG)
    d = tmp / "models" / "fake_ae"
    d.mkdir(parents=True)
    b = torch.export.Dim("b", min=1, max=4096)
    torch.manual_seed(0)
    torch.export.save(torch.export.export(Enc(), (torch.rand(2, 3, IMG, IMG),), dynamic_shapes=({0: b},)), d / "encoder.pt2")
    torch.export.save(torch.export.export(Dec(), (torch.rand(2, 48, 48),), dynamic_shapes=({0: b},)), d / "decoder.pt2")


make_fake_imagenet()
make_fake_ae()
torch.cuda.synchronize = lambda *a, **k: None   # the report branch of the loop calls it; no GPU here

# ------------------------------------------------------------------ tiny settings
# Appended to the cell that sets them, so they win over the notebook's values.
FIRST_CELL = "from pathlib import Path\nimport os\nimport time"
TINY = {
    FIRST_CELL:'num_blocks = 6; n_heads = 2; head_size = 8; flow_emb_dim = 16; XM_K = 2\nae_name = "fake_ae"; image_size = %d' % IMG,
    "# ------------------------------------------------------------------\n# ImageNet loader config": "IMAGE_SIZE = %d; CACHE_BATCH_SIZE = 16; ALLOW_PARTIAL_IMAGENET = True" % IMG,
}
SUBS = [("num_images=1_281_167", f"num_images={N_TRAIN}"), ("num_images=50_000", f"num_images={N_VAL}"),
        ("num_epochs = 200", "num_epochs = NUM_EPOCHS_TEST"), ("report_every = 500", "report_every = 3")]
SKIP = {"print(torch.__version__)": "prints CUDA device info", "sync_models()": "rsync to the PC",
        "#def sync models": "rsync helper"}

cells = ["".join(c["source"]) for c in json.load(open(nb_path))["cells"] if c["cell_type"] == "code"]


def find(prefix, src=None):
    hits = [i for i, c in enumerate(src or cells) if c.lstrip().startswith(prefix)]
    assert len(hits) == 1, (prefix, hits)
    return hits[0]


STOP_AT = find("import os\nimport torch\n\nname =")     # first cell of the bottom save/load zone
applied = []
if MODE == "patched":
    from make_flow_resume_page import apply_edit, parse_blocks
    R = parse_blocks()
    stext = (HERE / "sprint_cells.py").read_text()
    S = dict(zip(re.findall(r"^# === (BLOCK \d)", stext, re.M), re.split(r"^# === BLOCK \d.*\n", stext, flags=re.M)[1:]))

    def insert_below(prefix, code, tag):
        cells.insert(find(prefix) + 1, code); applied.append(tag)

    i0 = find(FIRST_CELL)
    cells[i0] = apply_edit(cells[i0], R["1"]) + "\n\n" + S["BLOCK 1"] + "\n" + S["BLOCK 3"]
    applied += ["resume 1", "sprint 1", "sprint 3 (optional)"]
    insert_below("train_ds = BufferedImageNet(", R["2"]["code"], "resume 2")
    insert_below("#log metrics", R["3"]["code"], "resume 3")
    insert_below("#define latent flow model", S["BLOCK 2"], "sprint 2")
    ii = find("# -------------------- Init Models + Optimizer")
    assert cells[ii].count("flow_model = FlowModel(") == 1
    cells[ii] = cells[ii].replace("flow_model = FlowModel(", "flow_model = SparseMiddleFlowModel(")
    applied.append("sprint one-word edit")
    init_call = [i for i, c in enumerate(cells) if ") = init_models()" in c]
    assert len(init_call) == 1
    cells.insert(init_call[0] + 1, "check_sparse_middle()"); applied.append("sprint check call")
    cells[find(R["4A"]["find"])] = R["4A"]["code"]; applied.append("resume 4A")
    cells[find(R["4B"]["find"])] = R["4B"]["code"]; applied.append("resume 4B")
    il = find("# MAIN TRAINING LOOP")
    for bid in ("5A", "5B", "5C"):
        cells[il] = apply_edit(cells[il], R[bid]); applied.append("resume " + bid)
    lines = cells[il].split("\n")
    at = [k for k, l in enumerate(lines) if l.startswith("for epoch in range(")]
    assert len(at) == 1
    lines[at[0] + 1:at[0] + 1] = S["BLOCK 4"].rstrip("\n").split("\n")
    cells[il] = "\n".join(lines); applied.append("sprint 4 (optional)")
    ib = find("# load flow model")
    bottom = [R[b]["code"] for b in ("6A", "6B", "6C", "6D")]
    cells[ib:ib] = bottom; applied += ["resume 6A-6D"]
    STOP_AT = find("import os\nimport torch\n\nname =")

for key, extra in TINY.items():
    k = find(key); cells[k] = cells[k] + "\n\n# tiny test settings\n" + extra
for a, b in SUBS:
    n = sum(c.count(a) for c in cells)
    assert n >= 1, (a, n)
    cells = [c.replace(a, b) for c in cells]


class CellTimeout(Exception):
    pass


def _alarm(*_):
    raise CellTimeout()


signal.signal(signal.SIGALRM, _alarm)


def run_cells(ns, indices, label):
    """Run cells in order in ns. Returns [(index, first line, status, seconds, output tail)]."""
    rows = []
    for i in indices:
        src = cells[i]
        first = next((l.strip() for l in src.splitlines() if l.strip() and not set(l.strip()) <= set("#")), "")[:60]
        if not src.strip() or all(l.strip().startswith("#") or not l.strip() for l in src.splitlines()):
            continue
        skip = next((why for p, why in SKIP.items() if src.lstrip().startswith(p)), None)
        if skip:
            rows.append((i, first, "skipped: " + skip, 0.0, "")); continue
        out, t0 = io.StringIO(), time.time()
        signal.alarm(CELL_TIMEOUT)
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                exec(compile(src, f"<cell {i}>", "exec"), ns)
            status = "ok"
        except CellTimeout:
            status = f"timeout after {CELL_TIMEOUT} s"
        except BaseException as e:  # noqa: BLE001
            tb = traceback.extract_tb(e.__traceback__)
            where = next((f"line {f.lineno}" for f in reversed(tb) if f.filename.startswith("<cell")), "")
            status = f"FAIL {type(e).__name__}: {str(e)[:150]} ({where})"
        finally:
            signal.alarm(0)
        import matplotlib.pyplot as plt
        plt.close("all")
        rows.append((i, first, status, time.time() - t0, out.getvalue()))
        print(f"[{label}] cell {i:>3} {time.time() - t0:6.1f}s {status[:110]:<110} | {first}", flush=True)
    return rows


def out_of(rows, prefix):
    return next(r[4] for r in rows if cells[r[0]].lstrip().startswith(prefix))


checks = []


def check(cond, what):
    checks.append((bool(cond), what))
    print(("  CHECK ok:   " if cond else "  CHECK FAIL: ") + what, flush=True)


# ------------------------------------------------------------------ phase 1: fresh kernel, top to the STOP zone
print(f"mode {MODE} | {len(cells)} code cells | edits applied: {', '.join(applied) or 'none'}", flush=True)
ns1 = {"__name__": "__main__", "NUM_EPOCHS_TEST": 2}
rows1 = run_cells(ns1, range(0, STOP_AT), "run 1")
n_fail = sum(r[2].startswith(("FAIL", "timeout")) for r in rows1)
print(f"run 1: {sum(r[2] == 'ok' for r in rows1)} ok, {n_fail} failed, {sum(r[2].startswith('skipped') for r in rows1)} skipped", flush=True)
summary = {"mode": MODE, "run1": [(i, first, status) for i, first, status, _, _ in rows1]}

steps_per_epoch = N_TRAIN // 16
loop_ok = any(cells[r[0]].lstrip().startswith("# MAIN TRAINING LOOP") and r[2] == "ok" for r in rows1)
check(loop_ok, "the training loop cell ran to the end")
check(len(ns1.get("logs", {}).get("flow_loss", [])) == 2 * steps_per_epoch, f"2 epochs x {steps_per_epoch} steps were logged (got {len(ns1.get('logs', {}).get('flow_loss', []))})")

if MODE == "patched":
    ckpt = tmp / "ckpt"
    smoke = out_of(rows1, "check_sparse_middle()")
    check("training step, tokens per block: [48, 48, 12, 12, 48, 48]" in smoke and "sampling / eval, tokens per block: [48, 48, 48, 48, 48, 48]" in smoke,
          "smoke test: 6 tiny blocks -> shallow 2 and final 2 see 48 tokens, middle 2 see 12; sampling sees 48")
    c_train = out_of(rows1, R["4A"]["find"])
    check("no file at" in c_train and "saved 192" in c_train, "train cache cell: encoded 192 images and saved the file")
    check((ckpt / "z_cache" / "fake_ae_32px" / "train.pt").exists() and (ckpt / "z_cache" / "fake_ae_32px" / "val.pt").exists(), "both z cache files exist")
    loop_out = out_of(rows1, "# MAIN TRAINING LOOP")
    check(loop_out.count("saved lf_imagenet_xm12 | epochs done") == 2 and "EPOCH 2" in loop_out, "the loop saved after epoch 1 and epoch 2")
    check((ckpt / "flow" / "lf_imagenet_xm12" / "latest.pt").exists(), "latest.pt exists")
    m1 = ns1["flow_model"]
    check(type(m1).__name__ == "SparseMiddleFlowModel" and m1.sparse_middle is True and getattr(m1, "epochs_done", None) == 2, "model is the sparse-middle class, still sparse, epochs_done = 2")
    w1 = {k: v.detach().clone() for k, v in m1.state_dict().items()}
    logs1 = list(ns1["logs"]["flow_loss"])
    z1 = ns1["cached_z"].clone()
    enc_before = dict(ns1)  # keep a reference so run 1 objects stay alive

    # -------------------------------------------------------------- phase 2: "kernel restart" and resume
    print("\n--- kernel restart: new namespace, run from the top through the compile cell ---", flush=True)
    ns2 = {"__name__": "__main__", "NUM_EPOCHS_TEST": 4}
    cells[find(FIRST_CELL)] += "\nsparse_until_epoch = 3"    # epochs 0-2 sparse, epoch 3 dense
    compile_cell = find("#compile and speed")
    rows2 = run_cells(ns2, range(0, compile_cell + 1), "run 2")
    c2 = out_of(rows2, R["4A"]["find"])
    check("loaded 192 latents" in c2 and "encoding" not in c2, "after the restart the train cache cell loads from disk and does not encode")
    check(torch.equal(ns2["cached_z"].float(), z1.float()), "the loaded latents equal the ones encoded in run 1")
    check(not any(torch.equal(v, w1[k_]) for k_, v in ns2["flow_model"].state_dict().items() if "blocks.0.MHA.QKV.weight" in k_), "before the resume cell the model is a fresh one")
    rows_r = run_cells(ns2, [find("# RESUME A RUN")], "run 2")
    r_out = rows_r[0][4]
    check("epochs done: 2" in r_out and "optimizer restored" in r_out, "resume cell: epochs done 2, optimizer restored")
    check(all(torch.equal(v, w1[k_]) for k_, v in ns2["flow_model"].state_dict().items()), "after the resume cell every weight equals the end of run 1")
    check(ns2["logs"]["flow_loss"] == [float(np.float32(x)) for x in logs1], "the loss history is restored")
    rows_l = run_cells(ns2, [find("# MAIN TRAINING LOOP")], "run 2")
    l_out = rows_l[0][4]
    check(rows_l[0][2] == "ok" and "EPOCH 3" in l_out and "EPOCH 4" in l_out and "EPOCH 1 " not in l_out and "EPOCH 2 " not in l_out, "the loop continued with epochs 3 and 4 only")
    check(len(ns2["logs"]["flow_loss"]) == 4 * steps_per_epoch, f"history now holds 4 x {steps_per_epoch} steps")
    check("sparse middle OFF" in l_out and ns2["flow_model"].sparse_middle is False, "sparse_until_epoch = 3: the loop switched to dense at the 4th epoch")
    rows_s = run_cells(ns2, [find("# SAVE NOW")], "run 2")
    check(rows_s[0][2] == "ok" and "epochs done 4" in rows_s[0][4], "save-now cell wrote latest.pt at epoch 4")
    rows_p = run_cells(ns2, [find("plot_seeds(sampler=\"heun\")")], "run 2")
    check(rows_p[0][2] == "ok", "sampling (plot_seeds, heun) runs on the resumed sparse-middle model")
    summary["run2"] = [(i, first, status) for i, first, status, _, _ in rows2 + rows_r + rows_l + rows_s + rows_p]

    # -------------------------------------------------------------- phase 3: another notebook, latents only
    ns3 = {}
    exec(R["E1"]["code"], ns3)
    zt, yt = ns3["load_z_cache"]("fake_ae", "train", image_size=IMG)
    check(torch.equal(zt.float(), z1.float()) and len(ns3["latent_loader"](zt, yt, train=True, batch_size=16)) == steps_per_epoch, "block E1 in an empty namespace loads the same latents")

summary["checks"] = checks
print("\nRESULT", MODE, "| cells failed in run 1:", n_fail, "| checks failed:", sum(not c for c, _ in checks), "of", len(checks))
print("SUMMARY_JSON " + json.dumps(summary))
