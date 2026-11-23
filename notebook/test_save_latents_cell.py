"""CPU test of save_latents_cell.py with the helper cell from the user's pushed notebook.

    python test_save_latents_cell.py NOTEBOOK.ipynb

Fake latents in memory -> run the cell -> the notebook's own load_z_cache and z_cache must both
accept the files (z_cache must load, not encode).
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import torch

cells = ["".join(c["source"]) for c in json.load(open(sys.argv[1]))["cells"]]
helper = next(c for c in cells if c.lstrip().startswith("# ---------- save / resume / z cache helpers: definitions only"))
cell = (Path(__file__).parent / "save_latents_cell.py").read_text()
assert all(len(l) <= 72 for l in cell.splitlines())

tmp = Path(tempfile.mkdtemp())
os.environ["ML_CHECKPOINTS"], os.environ["ML_MODELS"] = str(tmp / "ckpt"), str(tmp / "models")
AE = "ae_25M_selfteach_imagenet_MOcrop_180k"
(tmp / "models" / AE).mkdir(parents=True)
(tmp / "models" / AE / "encoder.pt2").write_bytes(b"x")
g = dict(torch=torch, os=os, Path=Path, time=time, ae_name=AE, IMAGE_SIZE=128, n_latent_tokens=48, emb_dim=48, device="cpu")
exec(helper, g)
torch.manual_seed(0)
g["cached_z"], g["cached_y"] = torch.randn(40, 48, 48).to(torch.bfloat16), torch.randint(0, 1000, (40,))
g["cached_test_z"], g["cached_test_y"] = torch.randn(16, 48, 48).to(torch.bfloat16), torch.randint(0, 1000, (16,))
small = cell.replace("1_281_167", "40").replace("50_000", "16")   # the test uses small counts
assert small != cell
exec(small, g)

z, y = g["load_z_cache"](AE, "train")
assert torch.equal(z.float(), g["cached_z"].float()) and torch.equal(y, g["cached_y"]) and z.dtype == torch.bfloat16
zt, yt = g["load_z_cache"](AE, "val")
assert torch.equal(zt.float(), g["cached_test_z"].float()) and torch.equal(yt, g["cached_test_y"])


def no_encoder(*a, **k):
    raise AssertionError("z_cache encoded again instead of loading the saved file")


g["encoder"] = no_encoder
z2, _ = g["z_cache"]("train", [("never", "used")], 40)
assert torch.equal(z2.float(), g["cached_z"].float())
print("ok: the cell writes both files; load_z_cache reads them back equal; z_cache loads them without encoding")
