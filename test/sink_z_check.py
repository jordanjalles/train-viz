"""The sink cell saves z of the first test images at every eval, for the Latent view.
Needs torch (run in the NGC container): python test/sink_z_check.py <out_dir>
Writes the snapshot under <out_dir> so test/sink_z_read.mjs can parse it with lib/torchpt.js."""
import sys
from pathlib import Path

import torch

out = Path(sys.argv[1])
src = (Path(__file__).resolve().parent.parent / "notebook/trainviz_sink_cell.py").read_text()
src = src.rstrip()[: -len("trainviz_attach()")]
ns = {"__name__": "nb"}
exec(src, ns)
ns["TRAINVIZ_ROOT"] = out


class Enc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3 * 16 * 16, 48)

    def forward(self, x):
        p = x.unfold(2, 16, 16).unfold(3, 16, 16).permute(0, 2, 3, 1, 4, 5)
        return self.lin(p.reshape(x.shape[0], 48, -1)[..., :768])


enc = Enc().train()
loader = [(torch.rand(8, 3, 128, 128), None) for _ in range(3)]
evals = []
ns.update(
    _active_run=None, encoder=enc, test_dataloader=loader,
    start_run=lambda name, config=None: (ns.update(_active_run={"run_id": name}), name)[1],
    finish_run=lambda: ns.update(_active_run=None),
    evaluate_multiview=lambda max_batches=20: (evals.append(1), {"test clean": 0.05})[1],
)
ns["trainviz_attach"](ns)
ns["start_run"]("zrun", {})
ns["_tv"]["step"] = 2000
ns["evaluate_multiview"]()
f = out / "zrun" / "z_002000.pt"
ok = f.exists()
z = torch.load(f) if ok else None
checks = [
    (ok, "z snapshot written at the eval step"),
    (ok and tuple(z.shape) == (16, 48, 48), f"shape (2 batches x 8, 48, 48): {None if z is None else tuple(z.shape)}"),
    (ok and z.dtype == torch.float16, "stored as float16"),
    (enc.training, "encoder back in training mode"),
    ((out / "zrun" / "z_images.pt").exists() and tuple(torch.load(out / "zrun" / "z_images.pt").shape) == (16, 3, 64, 64)
     and torch.load(out / "zrun" / "z_images.pt").dtype == torch.uint8, "z_images.pt: uint8 (16, 3, 64, 64) thumbnails"),
    (ns["_tv"]["file"] is not None, "sink still on"),
]
ns["encoder"] = None
ns["evaluate_multiview"]()
checks.append((ns["_tv"]["file"] is not None, "no encoder: snapshot skipped, sink still on"))
bad = [m for c, m in checks if not c]
for c, m in checks:
    print(("ok   " if c else "FAIL ") + m)
sys.exit(1 if bad else 0)
