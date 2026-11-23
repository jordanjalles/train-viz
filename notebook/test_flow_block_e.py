"""CPU test of block E (standalone z-cache loader) of flow_resume_cells.py.

    python test_flow_block_e.py

Writes a cache with block A's z_cache from a fake encoder, then runs block E in an empty
namespace and checks it returns the same tensors without an encoder or image loader.
"""
import os
import re
import tempfile
import time
from pathlib import Path

import torch

text = (Path(__file__).parent / "flow_resume_cells.py").read_text()
parts = re.split(r"^# === BLOCK (\w+):.*\n", text, flags=re.M)[1:]
blocks = {parts[i]: "\n".join(l for l in parts[i + 1].splitlines() if not l.startswith(("# MODE:", "# FIND:"))) for i in range(0, len(parts), 2)}
assert all(len(l) <= 72 for b in ("E1", "E2") for l in blocks[b].splitlines()), "block E has a line over 72 characters"

tmp = Path(tempfile.mkdtemp())
os.environ["ML_CHECKPOINTS"], os.environ["ML_MODELS"] = str(tmp / "ckpt"), str(tmp / "models")
AE = "ae_25M_selfteach_imagenet_MOcrop_180k"
(tmp / "models" / AE).mkdir(parents=True)
(tmp / "models" / AE / "encoder.pt2").write_bytes(b"x")

a = dict(torch=torch, os=os, Path=Path, time=time, ae_name=AE, IMAGE_SIZE=128, n_latent_tokens=48, emb_dim=48, device="cpu")
a["encoder"] = lambda x: x.mean((2, 3))[:, :1].repeat(1, 48 * 48).reshape(-1, 48, 48) + torch.arange(48)
exec(blocks["3"], a)   # the notebook's helpers cell, which writes the cache
torch.manual_seed(0)
made = {}
for split, n in (("train", 200), ("val", 72)):
    loader = [(torch.rand(8, 3, 4, 4), torch.randint(0, 1000, (8,))) for _ in range(n // 8)]
    made[split] = a["z_cache"](split, loader, n)

e = {}
exec(blocks["E1"], e)  # an empty namespace: block E1 must bring everything it needs
defined = {k for k in e if not k.startswith("__")}
assert defined == {"os", "Path", "torch", "TensorDataset", "DataLoader", "load_z_cache", "latent_loader"}, defined
assert not any(torch.is_tensor(v) for v in e.values()), "E1 must only define, not load"
exec(blocks["E1"], e)  # running the definitions again changes nothing
exec(blocks["E2"], e)
assert len(blocks["E2"].strip().splitlines()) <= 12, "the action cell must stay short"
assert torch.equal(e["cached_z"].float(), made["train"][0].float()) and torch.equal(e["cached_y"], made["train"][1])
assert torch.equal(e["cached_test_z"].float(), made["val"][0].float()) and torch.equal(e["cached_test_y"], made["val"][1])
assert e["cached_z"].dtype == torch.bfloat16 and e["cached_z"].shape == (200, 48, 48)
zb, yb = next(iter(e["latent_train_dataloader"]))
assert zb.shape == (64, 48, 48) and yb.shape == (64,)
assert len(e["latent_train_dataloader"]) == 3 and len(e["latent_test_dataloader"]) == 2
assert "encoder" not in e and "train_dataloader" not in e
print("block E ok: same tensors as the cache written by z_cache; no encoder, no image loader; loaders give (64, 48, 48)")
