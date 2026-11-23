"""CPU test of flow_resume_cells.py against the real notebook cells.

    python test_flow_resume.py /path/to/ImageNet1-Latent-Flow-Matching-XM.ipynb

Runs the blocks in notebook order (1 settings, 2 validation loader,
3 helpers, 4A/4B cache cells, 6B/6C bottom cells) and the notebook's
own training-loop cell (with the 5A-5C edits applied by the same
parser that builds the page) on a tiny fake encoder and data. Every
check is on behavior: encoder call counts, tensor equality, files on
disk, loader settings, and what each cell defines.
"""
import copy
import json
import os
import re
import sys
import types
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_flow_resume_page import apply_edit, parse_blocks  # noqa: E402

BLOCKS = parse_blocks()
NB = json.loads(Path(sys.argv[1]).read_text())
CELLS = ["".join(c["source"]) for c in NB["cells"]]


def nb_cell(prefix):
    hits = [c for c in CELLS if c.lstrip().startswith(prefix)]
    assert len(hits) == 1, (prefix, len(hits))
    return hits[0]


checks = []


def ok(cond, what):
    checks.append(what)
    assert cond, "FAILED: " + what
    print("  ok:", what)


tmp = Path(tempfile.mkdtemp())
os.environ["ML_CHECKPOINTS"] = str(tmp / "ckpt")
os.environ["ML_MODELS"] = str(tmp / "models")
N_TRAIN_IMAGES, N_VAL_IMAGES, BATCH = 50, 20, 8


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.w = nn.Linear(3 * 16 * 16, 4 * 8)
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        return self.w(images.flatten(1)).view(-1, 4, 8)


class BufferedImageNet(IterableDataset):
    """Stand-in with the notebook's constructor; yields fixed images."""
    def __init__(self, paths, transform, shards_in_ram, shuffle, num_images):
        self.n = N_VAL_IMAGES
        g = torch.Generator().manual_seed(7)
        self.x = torch.rand(self.n, 3, 16, 16, generator=g)

    def __len__(self):
        return self.n

    def __iter__(self):
        info = get_worker_info()
        ids = range(self.n) if info is None else range(info.id, self.n, info.num_workers)
        for i in ids:
            yield self.x[i], i % 10


def batches(n, seed):
    g = torch.Generator().manual_seed(seed)
    x, y = torch.rand(n, 3, 16, 16, generator=g), torch.arange(n) % 10
    return [(x[i:i + BATCH], y[i:i + BATCH]) for i in range(0, n, BATCH)]


class TrainDs:
    def __len__(self):
        return N_TRAIN_IMAGES


def fresh_namespace(seed, ae_name="fake_ae"):
    """What the notebook has defined when block A runs."""
    enc_dir = tmp / "models" / ae_name
    enc_dir.mkdir(parents=True, exist_ok=True)
    if not (enc_dir / "encoder.pt2").exists():
        (enc_dir / "encoder.pt2").write_bytes(b"encoder-v1")
    ns = dict(
        torch=torch, nn=nn, F=F, os=os, Path=Path, time=time,
        ae_name=ae_name, IMAGE_SIZE=16, n_latent_tokens=4, emb_dim=8,
        CACHE_BATCH_SIZE=BATCH, device="cpu", num_blocks=1, n_heads=1,
        head_size=8, flow_emb_dim=8, flow_noise_scale=1.0, XM_K=2, XM_N=1,
        encoder=FakeEncoder(), train_ds=TrainDs(),
        train_dataloader=batches(N_TRAIN_IMAGES, seed=3),
        BufferedImageNet=BufferedImageNet, val_paths=[], SHARDS_IN_RAM=1,
        ImageNetFullImage=lambda size: None, DATA_SEED=42,
        LOADER_PREFETCH_BATCHES=2, seed_imagenet_worker=lambda i: None,
        DataLoader=DataLoader,
    )
    exec(nb_cell("#log metrics"), ns)          # the notebook's own logs
    torch.manual_seed(seed)
    ns["flow_model"] = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 8))
    ns["optimizer"] = torch.optim.AdamW(ns["flow_model"].parameters(), lr=1e-3)
    ns["steps_run"] = 0

    def train_step_compiled(z):
        ns["optimizer"].zero_grad(set_to_none=True)
        z = z.float()
        loss = (ns["flow_model"](z) - z).square().mean()
        loss.backward()
        ns["optimizer"].step()
        ns["steps_run"] += 1
        return {"flow_loss": loss.detach()}

    ns["train_step_compiled"] = train_step_compiled
    ns["get_split_loss"] = lambda split: torch.tensor(0.5)
    exec(BLOCKS["1"]["code"], ns)             # first cell: run settings
    exec(BLOCKS["3"]["code"], ns)             # helpers, below "#log metrics"
    return ns


def code_lines(bid):
    return [l for l in BLOCKS[bid]["code"].splitlines() if l.strip()]


def func_source(code, name):
    """Text of one top-level function in a block."""
    start = code.index(f"def {name}(")
    end = code.find("\n\n\ndef ", start)
    return code[start:].rstrip() if end < 0 else code[start:end].rstrip()


def nb_loader_kwargs(prefix, var):
    """shuffle / drop_last / num_workers / pin_memory of `var = DataLoader(...)` in a notebook cell."""
    cell = nb_cell(prefix)
    call = cell[cell.index(f"{var} = DataLoader("):]
    call = call[:call.index("\n)")]
    out = {}
    for key in ("shuffle", "drop_last", "num_workers", "pin_memory"):
        out[key] = eval(re.search(rf"{key}=(\w+)", call).group(1))
    assert "batch_size=CACHE_BATCH_SIZE" in call
    return out


def loader_kwargs(loader):
    from torch.utils.data import RandomSampler
    return dict(shuffle=isinstance(loader.sampler, RandomSampler), drop_last=loader.drop_last,
                num_workers=loader.num_workers, pin_memory=loader.pin_memory)


TRAIN_CACHE = "# ---------- Cache encoder outputs for latent-flow training"
VAL_CACHE = "# ---------- Cache encoder VALIDATION outputs for testing"


def loop_cell(num_epochs):
    cell = nb_cell("# MAIN TRAINING LOOP")
    for bid in ("5A", "5B", "5C"):
        cell = apply_edit(cell, BLOCKS[bid])
    assert "num_epochs = 200" in cell
    return cell.replace("num_epochs = 200", f"num_epochs = {num_epochs}")


def same_state(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def opt_tensors(opt):
    return [v for s in opt.state_dict()["state"].values() for v in s.values() if torch.is_tensor(v)]


print("== z cache")
ns = fresh_namespace(seed=1)
n_batches = len(ns["train_dataloader"])
train_file = tmp / "ckpt" / "z_cache" / "fake_ae_16px" / "train.pt"
z1, y1 = ns["z_cache"]("train", ns["train_dataloader"], N_TRAIN_IMAGES)
ok(ns["encoder"].calls == n_batches and train_file.exists(), "first call encodes every batch and writes the file")
ok(z1.dtype == torch.bfloat16 and z1.shape == (N_TRAIN_IMAGES, 4, 8), "cache is bfloat16 with one row per image")
want = torch.cat([ns["encoder"](x) for x, _ in ns["train_dataloader"]]).to(torch.bfloat16)
ok(torch.equal(z1.float(), want.float()) and torch.equal(y1, torch.arange(N_TRAIN_IMAGES) % 10),
   "cached latents and labels equal a direct encode, in order")
calls = ns["encoder"].calls
z2, y2 = ns["z_cache"]("train", ns["train_dataloader"], N_TRAIN_IMAGES)
ok(ns["encoder"].calls == calls, "second call does not call the encoder")
ok(torch.equal(z1.float(), z2.float()) and torch.equal(y1, y2), "loaded tensors are identical to the encoded ones")
ok(not list(train_file.parent.glob("*.tmp")), "no temp file is left behind")

enc_file = tmp / "models" / "fake_ae" / "encoder.pt2"
st = enc_file.stat()
os.utime(enc_file, (st.st_atime, st.st_mtime + 100))
ns["z_cache"]("train", ns["train_dataloader"], N_TRAIN_IMAGES)
ok(ns["encoder"].calls == calls + n_batches, "a changed encoder.pt2 (modified time) rejects the cache")
calls = ns["encoder"].calls
enc_file.write_bytes(b"encoder-v2-longer")
os.utime(enc_file, (st.st_atime, st.st_mtime + 100))
ns["z_cache"]("train", ns["train_dataloader"], N_TRAIN_IMAGES)
ok(ns["encoder"].calls == calls + n_batches, "a changed encoder.pt2 (size) rejects the cache")
calls = ns["encoder"].calls
ns["z_cache"]("train", ns["train_dataloader"][:3], 3 * BATCH)
ok(ns["encoder"].calls == calls + 3, "a different image count rejects the cache")
calls = ns["encoder"].calls
ns["z_cache"]("train", ns["train_dataloader"][:3], 3 * BATCH, force=True)
ok(ns["encoder"].calls == calls + 3, "force=True encodes again")

other = fresh_namespace(seed=1, ae_name="other_ae")
other["z_cache"]("train", other["train_dataloader"], N_TRAIN_IMAGES)
ok(other["encoder"].calls == n_batches and (tmp / "ckpt" / "z_cache" / "other_ae_16px" / "train.pt").exists(),
   "another ae_name encodes into its own folder")

short = fresh_namespace(seed=1, ae_name="short_ae")
zs, _ = short["z_cache"]("train", short["train_dataloader"][:2], N_TRAIN_IMAGES)
ok(len(zs) == 2 * BATCH, "a loader that ends early gives a shorter cache")
calls = short["encoder"].calls
zs2, _ = short["z_cache"]("train", short["train_dataloader"][:2], N_TRAIN_IMAGES)
ok(short["encoder"].calls == calls and len(zs2) == 2 * BATCH, "and that shorter cache loads again")

print("== layout: settings, definition cell, short action cells")
bare = {}
exec(BLOCKS["1"]["code"], bare)
ok({k for k in bare if k != "__builtins__"} == {"RUN_NAME", "KEEP_EVERY_EPOCHS", "AUTOSAVE_MINUTES"},
   "block 1 sets the three run settings and nothing else")
bare = {}
on_disk = sorted(str(p) for p in tmp.rglob("*"))
exec(BLOCKS["3"]["code"], bare)                # nothing defined before it: it reads no global when it runs
kinds = {k: v for k, v in bare.items() if k != "__builtins__"}
ok(all(isinstance(v, (types.FunctionType, types.ModuleType, type)) for v in kinds.values()),
   "block 3 defines only functions and imports (no tensors, timers or settings)")
ok({"z_cache", "load_z_cache", "latent_loader", "save_run", "load_run", "resume_epoch", "autosave"} <= set(kinds),
   "block 3 defines every helper the other blocks call")
ok(sorted(str(p) for p in tmp.rglob("*")) == on_disk, "running block 3 writes nothing to disk")
for name in ("load_z_cache", "latent_loader"):
    ok(func_source(BLOCKS["3"]["code"], name) == func_source(BLOCKS["E1"]["code"], name),
       f"{name} is the same text in block 3 and block E1")
for bid in ("4A", "4B", "6B", "6C", "E2"):
    ok(len(code_lines(bid)) <= 12, f"action block {bid} is at most 12 lines ({len(code_lines(bid))})")
ok(all(l.startswith("#") for l in code_lines("6D")), "block 6D (re-encode) is fully commented out")
val_cell = nb_cell(VAL_CACHE)
moved = "\n".join(BLOCKS["2"]["code"].splitlines()[1:]).strip()
ok(moved in val_cell, "block 2 is the notebook's own validation loader code, unchanged")
later = "\n".join(c for c in CELLS if c not in (val_cell, nb_cell(TRAIN_CACHE)))
made = set(re.findall(r"^(\w+)(?:, (\w+))? = ", nb_cell(TRAIN_CACHE) + "\n" + val_cell, re.M))
made = {n for pair in made for n in pair if n}
read_later = {n for n in made if re.search(rf"\b{n}\b", later)}
new_cells = "\n".join(BLOCKS[b]["code"] for b in ("2", "4A", "4B"))
missing = {n for n in read_later if not re.search(rf"^(?:\w+, )?{n}(?:, \w+)? = ", new_cells, re.M)}
ok(read_later >= {"latent_train_dataloader", "latent_test_dataloader", "test_dataloader"} and not missing,
   f"every name later notebook cells read is still made: {sorted(read_later)}")

print("== blocks 2, 4A and 4B in the notebook namespace")
ns = fresh_namespace(seed=1)
exec(BLOCKS["2"]["code"], ns)
ok(ns["encoder"].calls == 0, "block 2 builds the validation image loader without encoding")
exec(BLOCKS["4A"]["code"], ns)
ok(ns["encoder"].calls == n_batches, "first run of 4A encodes the train split")
exec(BLOCKS["4B"]["code"], ns)
ok(ns["encoder"].calls > n_batches, "first run of 4B encodes the validation split")
ok(len(ns["latent_train_dataloader"]) == N_TRAIN_IMAGES // BATCH and ns["cached_test_z"].shape[0] == N_VAL_IMAGES,
   "latent loaders have the notebook's sizes (train drops the last batch)")
ok(loader_kwargs(ns["latent_train_dataloader"]) == nb_loader_kwargs(TRAIN_CACHE, "latent_train_dataloader")
   and ns["latent_train_dataloader"].batch_size == BATCH,
   "train latent loader: same batch size, shuffle, drop_last, workers, pin_memory as the notebook's cell")
ok(loader_kwargs(ns["latent_test_dataloader"]) == nb_loader_kwargs(VAL_CACHE, "latent_test_dataloader")
   and ns["latent_test_dataloader"].batch_size == BATCH,
   "validation latent loader: same settings as the notebook's cell")
zb, yb = next(iter(ns["latent_test_dataloader"]))
ok(torch.equal(zb.float(), ns["cached_test_z"][:BATCH].float()), "validation loader yields the cached latents in order")
val_first = ns["cached_test_z"].clone()
calls = ns["encoder"].calls
exec(BLOCKS["4A"]["code"], ns)
exec(BLOCKS["4B"]["code"], ns)
ok(ns["encoder"].calls == calls, "second run of 4A and 4B never calls the encoder")
ok(torch.equal(val_first.float(), ns["cached_test_z"].float()), "validation latents are the same after the reload")

print("== training loop cell with the D edits: train, restart, resume")
ns["KEEP_EVERY_EPOCHS"] = 2
per_epoch = len(ns["latent_train_dataloader"])
init_weights = copy.deepcopy(ns["flow_model"].state_dict())
exec(loop_cell(3), ns)
run_dir = tmp / "ckpt" / "flow" / ns["RUN_NAME"]
ok(ns["steps_run"] == 3 * per_epoch, "a new model trains from epoch 0 (3 epochs of steps)")
ok(not same_state(init_weights, ns["flow_model"].state_dict()), "training changed the weights")
ck = torch.load(run_dir / "latest.pt", weights_only=True)
ok(ck["epochs_done"] == 3 and len(ck["logs"]["flow_loss"]) == 3 * per_epoch, "latest.pt records 3 epochs and every logged step")
ok((run_dir / "epoch_002.pt").exists() and not (run_dir / "epoch_003.pt").exists(), "kept copy only every KEEP_EVERY_EPOCHS")
ok(torch.load(run_dir / "epoch_002.pt", weights_only=True)["epochs_done"] == 2, "the kept copy is the epoch-2 state")

print("== running the helpers cell again in the middle of a run")
clock, done = ns["flow_model"].last_save_time, ns["flow_model"].epochs_done
latest = (run_dir / "latest.pt").stat().st_mtime_ns
exec(BLOCKS["3"]["code"], ns)
ok(ns["flow_model"].last_save_time == clock and ns["flow_model"].epochs_done == done == 3 and ns["resume_epoch"]() == 3,
   "the autosave clock and epochs_done are unchanged")
ns["autosave"](3)
ok((run_dir / "latest.pt").stat().st_mtime_ns == latest, "and autosave still waits for AUTOSAVE_MINUTES")
exec(loop_cell(3), ns)
ok(ns["steps_run"] == 3 * per_epoch, "and the loop cell still sees 3 epochs done (trains nothing)")

new = fresh_namespace(seed=99)                      # "kernel restart": new weights, empty logs
exec(BLOCKS["2"]["code"], new)
exec(BLOCKS["4A"]["code"], new)
exec(BLOCKS["4B"]["code"], new)
ok(new["encoder"].calls == 0, "after a restart both caches load without the encoder")
ok(not same_state(new["flow_model"].state_dict(), ns["flow_model"].state_dict()), "the restarted model starts different")
ok(new["resume_epoch"]() == 0, "before the resume cell the loop would start at epoch 0")
exec(BLOCKS["6B"]["code"], new)                     # the bottom "RESUME A RUN" cell
got = new["start_epoch"]
ok(got == 3 and new["resume_epoch"]() == 3, "the resume cell (6B) sets start_epoch to 3")
ok(same_state(new["flow_model"].state_dict(), ns["flow_model"].state_dict()), "weights are identical after load_run")
a, b = opt_tensors(new["optimizer"]), opt_tensors(ns["optimizer"])
ok(len(a) == len(b) > 0 and all(torch.equal(x, y) for x, y in zip(a, b)), "optimizer moments and step counts are identical")
ok(torch.allclose(torch.tensor(new["logs"]["flow_loss"]), torch.tensor(ns["logs"]["flow_loss"]), rtol=1e-6, atol=0),
   "loss history is restored (float32)")
new["KEEP_EVERY_EPOCHS"] = 2
exec(loop_cell(5), new)
ok(new["steps_run"] == 2 * per_epoch, "the resumed loop runs epochs 4 and 5 only")
ck = torch.load(run_dir / "latest.pt", weights_only=True)
ok(ck["epochs_done"] == 5 and len(ck["logs"]["flow_loss"]) == 5 * per_epoch, "latest.pt then records 5 epochs")
ok((run_dir / "epoch_004.pt").exists(), "epoch_004.pt was kept")
exec(loop_cell(5), new)
ok(new["steps_run"] == 2 * per_epoch, "running the loop cell again at num_epochs trains nothing")

print("== save-now cell (6C)")
before = (run_dir / "latest.pt").stat().st_mtime_ns
exec(BLOCKS["6C"]["code"], new)
ck = torch.load(run_dir / "latest.pt", weights_only=True)
ok((run_dir / "latest.pt").stat().st_mtime_ns > before and ck["epochs_done"] == 5, "6C rewrites latest.pt at the current epoch count")

print("== autosave inside an epoch")
young = fresh_namespace(seed=4)
young["RUN_NAME"] = "young_run"
young["AUTOSAVE_MINUTES"] = 0
young["autosave"](0)
ok(not (tmp / "ckpt" / "flow" / "young_run").exists() and young["flow_model"].last_save_time > 0,
   "the first autosave call of a new model only starts the clock")
young["autosave"](0)
ok((tmp / "ckpt" / "flow" / "young_run" / "latest.pt").exists(), "the next call saves once the time has passed")
before = (run_dir / "latest.pt").stat().st_mtime_ns
new["autosave"](7)
ok((run_dir / "latest.pt").stat().st_mtime_ns == before, "no save before AUTOSAVE_MINUTES have passed")
new["AUTOSAVE_MINUTES"] = 0
new["autosave"](6)
ck = torch.load(run_dir / "latest.pt", weights_only=True)
ok(ck["epochs_done"] == 6 and not (run_dir / "epoch_006.pt").exists(), "autosave records the running epoch as not done, no kept copy")
new["AUTOSAVE_MINUTES"] = 30

print("== interrupted write")
real_save = torch.save


def broken_save(obj, path, *a, **kw):
    Path(path).write_bytes(b"half a file")
    raise KeyboardInterrupt


torch.save = broken_save
try:
    new["save_run"](9)
    ok(False, "save_run should have been interrupted")
except KeyboardInterrupt:
    pass
finally:
    torch.save = real_save
ck = torch.load(run_dir / "latest.pt", weights_only=True)
ok(ck["epochs_done"] == 6, "latest.pt is still the previous complete save")

print("== model with extra parameters, and a compiled-style wrapper")
grown = fresh_namespace(seed=5)
grown["flow_model"].add_module("extra", nn.Linear(8, 8))
grown["optimizer"] = torch.optim.AdamW(grown["flow_model"].parameters(), lr=1e-3)
extra_before = grown["flow_model"].extra.weight.clone()
grown["load_run"]()
ok(torch.equal(grown["flow_model"][0].weight, new["flow_model"][0].weight), "shared weights load into the larger model")
ok(torch.equal(grown["flow_model"].extra.weight, extra_before), "new parameters keep their initial values")
ok(not opt_tensors(grown["optimizer"]), "the optimizer state that no longer fits is not loaded")


class Wrapped(nn.Module):
    def __init__(self, mod):
        super().__init__()
        self._orig_mod = mod


wrap = fresh_namespace(seed=6)
wrap["RUN_NAME"] = "wrapped_run"
inner = wrap["flow_model"]
wrap["flow_model"] = Wrapped(inner)
wrap["save_run"](1)
ck = torch.load(tmp / "ckpt" / "flow" / "wrapped_run" / "latest.pt", weights_only=True)
ok(not any(k.startswith("_orig_mod.") for k in ck["flow_model"]) and same_state(ck["flow_model"], inner.state_dict()),
   "the _orig_mod. prefix is stripped on save")
plain = fresh_namespace(seed=8)
plain["load_run"]("wrapped_run")
ok(same_state(plain["flow_model"].state_dict(), inner.state_dict()), "and that file loads into a plain model by run name")

print(f"\nALL {len(checks)} CHECKS PASSED")
