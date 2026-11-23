"""The paste-safe sink cell must never break a run: (1) on a machine with no feed folder it attaches,
does nothing and lets every notebook function work; (2) a write failure mid-run switches the sink off
and training carries on. Run: python3 test/sink_cell_safety.py"""
import sys
import tempfile
from pathlib import Path

src = (Path(__file__).resolve().parent.parent / "notebook/trainviz_sink_cell.py").read_text()
src = src.rstrip()[: -len("trainviz_attach()")]
ok = True


def check(cond, msg):
    global ok
    print(("ok   " if cond else "FAIL ") + msg)
    ok &= bool(cond)


def notebook(ns, calls):
    ns.update(
        _active_run=None, preview_dir=tempfile.mkdtemp(),
        start_run=lambda name, config=None: f"2026_{name}",
        finish_run=lambda: calls.append("finish"),
        log_metrics=lambda **m: calls.append("log"),
        evaluate_multiview=lambda: {"test clean": 0.01},
        save_reconstruction_snapshot=lambda masked=False: True,
    )


# (1) no feed folder: neither default root exists on this machine
ns, calls = {"__name__": "nb"}, []
exec(src, ns)
check(ns["TRAINVIZ_ROOT"] is None, "no feed folder found here -> sink has no root")
notebook(ns, calls)
ns["trainviz_attach"](ns)
rid = ns["start_run"]("r", {"lr": 1e-4})
for _ in range(250):
    ns["log_metrics"](clean_term=0.5, view_term=0.2)
res = ns["evaluate_multiview"]()
ns["save_reconstruction_snapshot"](masked=False)
ns["finish_run"]()
check(rid == "2026_r" and res == {"test clean": 0.01}, "start_run / evaluate_multiview return their own values")
check(calls.count("log") == 250 and calls[-1] == "finish", "notebook functions all ran, no exception")

# (2) a write that fails mid-run
ns, calls = {"__name__": "nb"}, []
exec(src, ns)
root = Path(tempfile.mkdtemp())
ns["TRAINVIZ_ROOT"] = root
notebook(ns, calls)
ns["trainviz_attach"](ns)
ns["start_run"]("r", {})


class Broken:
    def write(self, s):
        raise OSError("No space left on device")

    def close(self):
        pass


for _ in range(100):
    ns["log_metrics"](clean_term=0.5, view_term=0.2)
ns["_tv"]["file"] = Broken()
for _ in range(300):   # two flushes would hit the broken file
    ns["log_metrics"](clean_term=0.5, view_term=0.2)
ns["evaluate_multiview"]()
ns["finish_run"]()
check(calls.count("log") == 400 and calls[-1] == "finish", "write failure mid-run: training calls all completed")
check(ns["_tv"]["file"] is None, "sink switched itself off after the failure")

# (3) the cell is run while a run is already active: it joins it and back-fills from memory
import json
ns, calls = {"__name__": "nb"}, []
exec(src, ns)
root = Path(tempfile.mkdtemp())
ns["TRAINVIZ_ROOT"] = root
notebook(ns, calls)
ns["_active_run"] = {"run_id": "2026_live", "config": {"lr": 1e-4}}
ns["logs"] = {"clean_term": [0.5] * 100 + [0.3] * 150, "view_term": [0.2] * 250,
              "swd_term": [0.01] * 250, "bwd_term": [0.02] * 250}
ns["trainviz_attach"](ns)
ns["trainviz_attach"](ns)                      # running the cell again must not join twice
for _ in range(50):
    ns["log_metrics"](clean_term=0.1, view_term=0.2, swd_term=0.0, bwd_term=0.0)
ns["finish_run"]()
recs = [json.loads(l) for l in (root / "2026_live" / "metrics.jsonl").read_text().splitlines()]
check([r["step"] for r in recs] == [100, 200, 300], "joined an active run: back-filled 100, 200, then live 300")
check(abs(recs[0]["clean"] - 0.5) < 1e-9 and abs(recs[1]["clean"] - 0.3) < 1e-9 and abs(recs[0]["reg"] - 0.03) < 1e-9,
      "back-filled averages match the notebook's in-memory history")
check(abs(recs[2]["clean"] - 0.1) < 1e-9, "live record after joining averages only the new steps")
sys.exit(0 if ok else 1)
