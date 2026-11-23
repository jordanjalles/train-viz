"""Exercise notebook/trainviz_sink.py against a mock of the self-teach notebook's functions.
Run: python3 test/sink_check.py (no torch needed). Exits non-zero on failure."""
import json
import os
import sys
import tempfile
from pathlib import Path

tmp = Path(tempfile.mkdtemp())
os.environ["TRAINVIZ_ROOT"] = str(tmp / "sink")
ns = {"__name__": "nb"}
# SINK_FILE=notebook/trainviz_sink_cell.py checks the paste-safe cell (it hard-codes its root and
# attaches on its last line, so that line is dropped here and the root overridden).
sink = Path(__file__).resolve().parent.parent / os.environ.get("SINK_FILE", "notebook/trainviz_sink.py")
src = sink.read_text()
if src.rstrip().endswith("\ntrainviz_attach()"):
    src = src.rstrip()[: -len("trainviz_attach()")]
exec(src, ns)
ns["TRAINVIZ_ROOT"] = tmp / "sink"

preview_dir = tmp / "preview"
preview_dir.mkdir()
calls = {"log": 0}
ns.update(
    _active_run=None, preview_dir=preview_dir,
    start_run=lambda name, config=None: (ns.update(_active_run={"run_id": f"2026_{name}", "config": config}), f"2026_{name}")[1],
    resume_run=lambda: ns.update(_active_run={"run_id": "2026_r1", "config": {"lr": 1e-4}}),
    finish_run=lambda: ns.update(_active_run=None),
    log_metrics=lambda **m: calls.__setitem__("log", calls["log"] + 1),
    evaluate_multiview=lambda max_batches=20: {"test clean": 0.01, "train clean": 0.02, "test missing": 0.03, "train missing": 0.04},
    save_reconstruction_snapshot=lambda masked=False: ((preview_dir / "latest_recon_clean.png").write_bytes(b"png"), True)[1],
)
ns["TRAINVIZ_DESCRIPTION"] = "mock run"
ns["trainviz_attach"](ns)
ns["trainviz_attach"](ns)   # attaching twice must not double-wrap

ns["start_run"]("r1", {"lr": 1e-4})
ns["evaluate_multiview"]()                       # readout.start: eval at step 0
for epoch in range(2):
    for _ in range(100):                          # epoch length == TRAINVIZ_EVERY: eval lands on a flush step
        ns["log_metrics"](clean_term=0.5, view_term=0.2, laplacian_term=0.1,
                          swd_term=0.01, bwd_term=0.02, c_swd_term=0.03, t_swd_term=0.04, ae_loss=1.0)
    ns["evaluate_multiview"]()
    ns["save_reconstruction_snapshot"](masked=False)
    ns["save_reconstruction_snapshot"](masked=True)
for _ in range(50):
    ns["log_metrics"](clean_term=0.4, view_term=0.2, laplacian_term=0.1,
                      swd_term=0.0, bwd_term=0.0, c_swd_term=0.0, t_swd_term=0.0)
ns["finish_run"]()
finished_after_first = (tmp / "sink" / "2026_r1" / "FINISHED").exists()
ns["resume_run"]()                               # resumed run continues the step count
finished_during_resume = (tmp / "sink" / "2026_r1" / "FINISHED").exists()
for _ in range(100):
    ns["log_metrics"](clean_term=0.3, view_term=0.2, laplacian_term=0.1,
                      swd_term=0.0, bwd_term=0.0, c_swd_term=0.0, t_swd_term=0.0)
ns["finish_run"]()

d = tmp / "sink" / "2026_r1"
recs = [json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines()]
steps = [r["step"] for r in recs]
ok = True
def check(cond, msg):
    global ok
    print(("ok   " if cond else "FAIL ") + msg)
    ok &= bool(cond)

check(calls["log"] == 350, "original log_metrics still called every step (single wrap)")
check(all(b > a for a, b in zip(steps, steps[1:])), f"steps strictly increasing: {steps}")
check(steps == [0, 100, 200, 250, 300, 350], "records at 0 (eval), 100, 200, partial 250, resumed 300 and 350")
r100 = recs[1]
check("test_clean" in r100 and "clean" in r100, "eval and train average at the same step merged into one record")
check(abs(r100["reg"] - 0.10) < 1e-9 and abs(r100["clean"] - 0.5) < 1e-9, "reg = swd+bwd+c_swd+t_swd, averages right")
check("test_clean" in recs[0] and "clean" not in recs[0], "step-0 record is the initial eval only")
check(sorted(p.name for p in d.glob("recon_*.png")) == ["recon_000100.png", "recon_000200.png"], "clean recon snapshots copied per epoch")
check(json.loads((d / "config.json").read_text())["run"] == "2026_r1", "config.json written")
check((d / "DESCRIPTION.md").read_text().strip() == "mock run", "DESCRIPTION.md written")
check(finished_after_first and not finished_during_resume and (d / "FINISHED").exists(),
      "FINISHED marker: written on finish, removed while resumed, written again at the end")
sys.exit(0 if ok else 1)
