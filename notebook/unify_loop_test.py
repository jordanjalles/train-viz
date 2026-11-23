"""Test of the looped-middle cells (unify_loop_cells.py) on the user's pushed Unify notebook.

    python unify_loop_test.py UNIFY_NOTEBOOK.ipynb

Applies the blocks to a copy of the notebook as public/unify_loop.html tells the user to, runs
every cell from the top through the training loop with tiny settings and a fake ImageNet, and
checks each invariant of the spec. Runs on the CPU, or on the GPU when one is visible (then the
compiled train step and its recompiles are checked too).
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
GPU = torch.cuda.is_available()


def block(bid):
    text = (HERE / "unify_loop_cells.py").read_text()
    m = re.search(rf"^# === BLOCK {bid}[: ].*?===\n(.*?)(?=^# === BLOCK |\Z)", text, re.M | re.S)
    body = m.group(1).rstrip("\n") + "\n"
    assert max(len(l) for l in body.splitlines()) <= 76, (bid, max(body.splitlines(), key=len))
    return body


cells = ["".join(c["source"]) for c in json.load(open(sys.argv[1]))["cells"] if c["cell_type"] == "code"]


def find(prefix):
    hits = [i for i, c in enumerate(cells) if c.lstrip().startswith(prefix)]
    assert len(hits) == 1, (prefix, hits)
    return hits[0]


# ---- the page's steps ----
first = find("from pathlib import Path")
assert cells[first].count("keep_ratio = 1.0\n") == 1
cells[first] = cells[first].replace("keep_ratio = 1.0\n", "keep_ratio = 0.25\n")                 # step 1
anchor = "final_blocks = 2\n"
assert cells[first].count(anchor) == 1
cells[first] = cells[first].replace(anchor, anchor + "\n" + block(1))                               # step 2
cells.insert(find("#define unified model") + 1, block(2))                                           # step 3
cells[find("# --------------------Def Init Models")] = block(3)                                     # step 4
cells.insert(find("#log metrics (definitions only)") + 1, block(4))                                 # step 5
init = next(k for k, c in enumerate(cells) if ") = init_models()" in c)
cells.insert(init + 1, block(5))                                                                    # step 6
cells.insert(find("################## Training Step Methods") + 1, block(6))                        # step 7
cells[find("#make and compile the training steps")] = block(7)                                      # step 8
cells[find("# MAIN TRAINING LOOP")] = block(8)                                                      # step 9
cells.append(block(9))                                                                              # step 10
print("steps applied: every cell the page names exists exactly once in the pushed notebook")

# ---- fake world and tiny settings ----
tmp = Path(tempfile.mkdtemp())
os.environ.update(ML_DATASETS=str(tmp / "datasets"), ML_MODELS=str(tmp / "models"), ML_CHECKPOINTS=str(tmp / "ckpt"))
os.chdir(tmp)
N_TRAIN, N_VAL, IMG = 480, 48, 64


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
                base = rng.integers(0, 255, (5, 6, 3), dtype=np.uint8)          # smooth blobs: something to learn
                Image.fromarray(base).resize((80, 72), Image.BICUBIC).save(buf, format="PNG")
                images.append({"bytes": buf.getvalue(), "path": None})
                labels.append(int(rng.integers(0, 1000)))
            pq.write_table(pa.table({"image": images, "label": labels}), root / f"{split}-{s:05d}-of-{shards:05d}.parquet")


make_fake_imagenet()
cells[first] += ("\n# tiny test settings\nnum_blocks = 5; n_heads = 4; head_size = 16; emb_dim = 64\n"
                 "shallow_blocks = 1; final_blocks = 1\n"
                 f"IMAGE_SIZE = {IMG}; CACHE_BATCH_SIZE = 16; ALLOW_PARTIAL_IMAGENET = True; lr = 1e-3\n")
SUBS = [("num_images=1_281_167", f"num_images={N_TRAIN}"), ("num_images=50_000", f"num_images={N_VAL}"),
        ("num_epochs = 20", "num_epochs = 1"), ("report_every = 500", "report_every = 5"),
        ("eval_every = 2_000", "eval_every = 10"), ("preview_every = 10_000", "preview_every = 20")]
if not GPU:
    SUBS.append(("train_step = torch.compile(make_looped_train_step())", "train_step = make_looped_train_step()  # CPU test: no compiler"))
for a, b in SUBS:
    assert sum(c.count(a) for c in cells) == 1, (a, sum(c.count(a) for c in cells))
    cells = [c.replace(a, b) for c in cells]
SKIP = ("print(torch.__version__)",)


def run(ns, indices):
    out = {}
    for k in indices:
        src = cells[k]
        if not src.strip() or src.lstrip().startswith(SKIP) or all(l.strip().startswith("#") or not l.strip() for l in src.splitlines()):
            continue
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            exec(compile(src, f"<cell {k}>", "exec"), ns)
        out[k] = buf.getvalue()
        import matplotlib.pyplot as plt
        plt.close("all")
    return out


import types
mod = types.ModuleType("unify_nb")        # a real module, as a notebook kernel has: torch.compile needs one
sys.modules["unify_nb"] = mod
ns = mod.__dict__
loop_cell = find("# MAIN TRAINING LOOP")
out = run(ns, range(0, loop_cell))
model, device = ns["model"], ns["device"]
core = model.core_transformer
F = torch.nn.functional
print("cells above the loop run. Output of check_looped_model():")
print("   " + out[init + 1].strip().replace("\n", "\n   "))
T = (IMG // 16) ** 2
kept = int(T * 0.25)
assert f"middle blocks see {kept} tokens, 4 times each" in out[init + 1] and f"middle blocks see {T} tokens, 4 times each" in out[init + 1]
assert f"final blocks see {T} tokens, 4 times each" in out[init + 1]

# ---- 1. shared weights: the parameter count does not depend on the loop count ----
count = lambda m: sum(p.numel() for p in m.parameters())
args = dict(channels=3, patch_size=16, emb_dim=64, num_blocks=5, n_heads=4, head_size=16, keep_ratio=0.25, shallow_blocks=1, final_blocks=1)
n1, n6 = count(ns["LoopedImageTransformer"](**args, n_loops=1)), count(ns["LoopedImageTransformer"](**args, n_loops=6))
assert n1 == n6 == count(model)
print(f"1 ok: {n1:,} parameters with 1 loop and with 6 loops (middle weights are shared)")

# ---- 2. XSA only in the middle blocks, and it removes the part along v ----
kinds = [type(b.MHA).__name__ for b in core.blocks]
assert kinds == ["MHA", "XSAMHA", "XSAMHA", "XSAMHA", "MHA"], kinds
torch.manual_seed(0)
plain, xsa = ns["MHA"](4, 16, 64).to(device), ns["XSAMHA"](4, 16, 64).to(device)
xsa.load_state_dict(plain.state_dict())
plain.out_proj = xsa.out_proj = torch.nn.Identity()
x = torch.randn(3, T, 64, device=device)
core.eval()
rope = core.rope_tables(IMG // 16, IMG // 16)
with torch.no_grad():
    o_plain = plain(x, rope).view(3, T, 4, 16)
    o_xsa = xsa(x, rope).view(3, T, 4, 16)
    v = xsa.QKV(x).chunk(3, dim=-1)[2].view(3, T, 4, 16)
v_hat = F.normalize(v, dim=-1)
want = o_plain - (o_plain * v_hat).sum(-1, keepdim=True) * v_hat
cos_before = F.cosine_similarity(o_plain, v, dim=-1).abs().mean().item()
cos_after = F.cosine_similarity(o_xsa, v, dim=-1).abs().max().item()
assert torch.allclose(o_xsa, want, atol=1e-5) and cos_after < 1e-4 and o_xsa.norm() > 0.1 * o_plain.norm()
print(f"2 ok: XSA in blocks 1-3 only. |cos(head output, own v)|: {cos_before:.3f} without XSA, at most {cos_after:.1e} with XSA")

# ---- 3. exits are read-only, the token set is fixed, loops chain: exit k of a 4-loop pass = final output of a k-loop pass ----
model.train()
images = torch.rand(4, 3, IMG, IMG, device=device)
t = torch.rand(4, device=device)


def seeded(**kw):
    torch.manual_seed(7)
    with torch.no_grad():
        return model(images, t, **kw)


for sparse in (True, False):
    exits = seeded(all_exits=True, sparse=sparse)
    assert len(exits) == 4
    for k in range(1, 5):
        assert torch.equal(seeded(n_loops=k, sparse=sparse), exits[k - 1]), (sparse, k)
    assert not torch.allclose(exits[0], exits[3])
    six = seeded(n_loops=6, all_exits=True, sparse=sparse)
    assert len(six) == 6 and all(torch.equal(a, b) for a, b in zip(six[:4], exits))
calls = []
original = core.sample_indices
core.sample_indices = lambda *a: (calls.append(a[1:3]), original(*a))[1]
seeded(all_exits=True, sparse=True)
core.sample_indices = original
assert calls == [(0, 0), (T, kept)], calls
print(f"3 ok: with or without the other exits, loop k gives the same prediction (no exit feeds the loop). "
      f"Tokens are chosen once per pass: {kept} of {T}")

# ---- 4. the loss rule, and every weight gets a gradient in a sparse and in a dense step ----
step_fn = ns["make_looped_train_step"]()
batch = images.cpu()
for sparse in (True, False):
    before = [p.detach().clone() for p in model.parameters()]
    m = step_fn(batch, ns["optimizer"], sparse)
    e = m["exit_losses"].float()
    assert e.shape == (4,)
    assert torch.allclose(m["loss"].float(), 0.5 * e[:3].mean() + 0.5 * e[3], rtol=1e-5), (m["loss"], e)
    assert torch.equal(m["loss_final"], m["exit_losses"][3])
    no_grad = [n for n, p in model.named_parameters() if p.grad is None or not p.grad.abs().sum() > 0]
    moved = sum(not torch.equal(a, p.detach()) for a, p in zip(before, model.parameters()))
    assert not no_grad, no_grad
    print(f"4 ok ({'sparse' if sparse else 'dense '} step): loss {m['loss']:.4f} = 0.5 * mean{[round(v, 4) for v in e[:3].tolist()]} + 0.5 * {e[3]:.4f}; "
          f"all {len(before)} weight tensors have a gradient, {moved} moved")
with ns["use_loops"](1):
    m = step_fn(batch, ns["optimizer"], True)
assert m["exit_losses"].shape == (1,) and torch.equal(m["loss"], m["loss_final"])

# ---- 5. evaluation and inference: dense, any loop count ----
losses = ns["get_loop_losses"]("test", n_batches=2)
assert list(losses) == [1, 2, 3, 4, 5, 6] and len({round(v, 7) for v in losses.values()}) == 6 and all(np.isfinite(v) for v in losses.values())
assert core.n_loops == 4
seen = []
hook = core.blocks[2].register_forward_hook(lambda m, a, o: seen.append(o.shape[1]))
with ns["use_loops"](6), torch.no_grad():
    sample = ns["denoise_seeds"]([0, 1], steps=2)
hook.remove()
assert set(seen) == {T} and len(seen) == 2 * 6 and sample.shape == (2, 3, IMG, IMG), (set(seen), len(seen))
print(f"5 ok: eval with 1..6 loops gives six different losses {[round(v, 4) for v in losses.values()]}; "
      f"sampling is dense ({T} tokens in the middle) and runs with 6 loops")

# ---- 6. the training loop: cadence, logs, loop-count evals, chart ----
assert [ns["is_sparse_step"](s) for s in range(10)] == [True] * 4 + [False] + [True] * 4 + [False]
ns["clear_logs"]()
used = []
real_step = ns["train_step"]
ns["train_step"] = lambda im, opt, sparse: (used.append(bool(sparse)), real_step(im, opt, sparse))[1]
t0 = time.time()
text = run(ns, [loop_cell])[loop_cell]
ns["train_step"] = real_step
n_steps = len(ns["logs"]["loss"])
assert n_steps == N_TRAIN // 16 and used == [s % 5 < 4 for s in range(n_steps)], (n_steps, used)
lines = text.splitlines()
loops_lines = [l for l in lines if l.startswith("LOOPS step")]
assert len(loops_lines) == n_steps // 10 and len(ns["eval_logs"]["test_loss_L6"]) == n_steps // 10
assert len(ns["logs"]["loss_final"]) == n_steps
first5, last5 = np.mean(ns["logs"]["loss"][:5]), np.mean(ns["logs"]["loss"][-5:])
print(f"6 ok: {n_steps} loop steps in {time.time() - t0:.0f}s, cadence {''.join('S' if s else 'D' for s in used[:10])}..., "
      f"train loss {first5:.4f} -> {last5:.4f}")
print("     ", loops_lines[-1])
assert last5 < first5
run(ns, [len(cells) - 1])       # plot_loop_losses()
if GPU:
    from torch._dynamo.utils import counters
    frames = dict(counters["frames"])
    for s in range(10):                 # this batch has another size than the loop's: both step kinds compile once more
        real_step(batch, ns["optimizer"], s % 5 < 4)
    before = dict(counters["frames"]).get("total", 0)
    for s in range(40):
        real_step(batch, ns["optimizer"], s % 5 < 4)
    after = dict(counters["frames"]).get("total", 0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s in range(40):
        real_step(batch, ns["optimizer"], s % 5 < 4)
    torch.cuda.synchronize()
    assert frames.get("ok", 0) >= 0.8 * frames.get("total", 0) > 0 and after == before, (frames, before, after)
    print(f"7 ok (GPU): train step compiled, frames after the loop {frames}; 40 more steps of both kinds added no compile ({before} -> {after}); {(time.perf_counter() - t0) * 25:.0f} ms/step (tiny model)")
print("ALL OK")
