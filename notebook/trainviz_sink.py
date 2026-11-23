"""train-viz sink for the self-teach notebook (STL10-AE-working-self-teach.ipynb).

Paste this into a cell after the run-manager / eval / snapshot cells, run it, then call
`trainviz_attach()` once, before `start_run(...)`. The training loop stays as it is: the
sink wraps the notebook's own functions, adds no losses, measures no gradients and is
torch.compile-neutral.

  start_run / resume_run      -> <TRAINVIZ_ROOT>/<run_id>/config.json (+ DESCRIPTION.md)
  log_metrics (every step)    -> one metrics.jsonl record per 100 steps: averaged clean,
                                 view, lap, reg (= swd + bwd + c_swd + t_swd), as the harness
  evaluate_multiview          -> eval record at the current step: test_clean, test_missing,
                                 train_clean_eval, train_missing
  save_reconstruction_snapshot(masked=False)
                              -> recon_<step>.png (the reconstruction timeline)
  finish_run                  -> flushes the partial average, closes the file

TRAINVIZ_ROOT must be readable by the runner account (train-viz runs as runner) and
writable by the notebook's account; see the train-viz README for the one-time setup.
"""
import json
import os
import shutil
from pathlib import Path

TRAINVIZ_ROOT = Path(os.environ.get("TRAINVIZ_ROOT", "<sink-dir>"))
TRAINVIZ_EVERY = 100
TRAINVIZ_DESCRIPTION = ""   # optional paragraph shown on the run's page and card

_TRAIN_KEYS = {"clean_term": "clean", "view_term": "view", "laplacian_term": "lap"}
_REG_KEYS = ("swd_term", "bwd_term", "c_swd_term", "t_swd_term")
_EVAL_KEYS = {"test clean": "test_clean", "test missing": "test_missing",
              "train clean": "train_clean_eval", "train missing": "train_missing"}
_tv = {"file": None, "dir": None, "step": 0, "acc": {}, "n": 0, "pending": None}


def _tv_write(rec):
    """Hold the newest record back one write so a train average and an eval at the same step
    become one line (train-viz reads a repeated step as a rewritten log)."""
    if not _tv["file"]:
        return
    p = _tv["pending"]
    if p is not None and p["step"] == rec["step"]:
        p.update(rec)
        return
    if p is not None:
        _tv["file"].write(json.dumps(p) + "\n")
    _tv["pending"] = dict(rec)


def _tv_flush():
    if _tv["n"]:
        _tv_write({"step": _tv["step"], **{k: v / _tv["n"] for k, v in _tv["acc"].items()}})
    _tv["acc"], _tv["n"] = {}, 0


def _tv_open(run_id, config):
    d = TRAINVIZ_ROOT / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"run": run_id, **(config or {})}, indent=1, default=str))
    if TRAINVIZ_DESCRIPTION:
        (d / "DESCRIPTION.md").write_text(TRAINVIZ_DESCRIPTION.strip() + "\n")
    step = 0
    if (d / "metrics.jsonl").exists():   # resumed run: continue the step count
        for line in (d / "metrics.jsonl").read_text().splitlines():
            try:
                step = max(step, json.loads(line)["step"])
            except (ValueError, KeyError):
                pass
    (d / "FINISHED").unlink(missing_ok=True)   # a resumed run is running again
    _tv.update(file=open(d / "metrics.jsonl", "a", buffering=1), dir=d, step=step, acc={}, n=0, pending=None)


def _tv_close():
    _tv_flush()
    if _tv["file"]:
        if _tv["pending"] is not None:
            _tv["file"].write(json.dumps(_tv["pending"]) + "\n")
        _tv["file"].close()
        (_tv["dir"] / "FINISHED").write_text("")   # marker for a collector: no more writes until a resume
    _tv.update(file=None, dir=None, pending=None)


def trainviz_attach(namespace=None):
    """Wrap the notebook's run-manager, logging, eval and snapshot functions (default: globals())."""
    g = namespace if namespace is not None else globals()

    def num(v):
        v = v.item() if hasattr(v, "item") else v
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    def wrap(name, make):
        if name in g and not getattr(g[name], "_trainviz", False):
            f = make(g[name])
            f._trainviz = True
            g[name] = f

    def start_run(orig):
        def f(name, config=None, *a, **kw):
            run_id = orig(name, config, *a, **kw)
            _tv_open(run_id or (g.get("_active_run") or {}).get("run_id", name), config)
            return run_id
        return f

    def resume_run(orig):
        def f(*a, **kw):
            out = orig(*a, **kw)
            run = g.get("_active_run") or {}
            _tv_open(run.get("run_id", "resumed"), run.get("config"))
            return out
        return f

    def finish_run(orig):
        def f(*a, **kw):
            try:
                return orig(*a, **kw)
            finally:
                _tv_close()
        return f

    def log_metrics(orig):
        def f(**metrics):
            orig(**metrics)
            if not _tv["file"]:
                return
            _tv["step"] += 1
            acc = _tv["acc"]
            for k, out in _TRAIN_KEYS.items():
                v = num(metrics.get(k))
                if v is not None:
                    acc[out] = acc.get(out, 0.0) + v
            reg = [num(metrics.get(k)) for k in _REG_KEYS]
            if any(v is not None for v in reg):
                acc["reg"] = acc.get("reg", 0.0) + sum(v for v in reg if v is not None)
            _tv["n"] += 1
            if _tv["step"] % TRAINVIZ_EVERY == 0:
                _tv_flush()
        return f

    def evaluate_multiview(orig):
        def f(*a, **kw):
            res = orig(*a, **kw)
            rec = {out: num(res[k]) for k, out in _EVAL_KEYS.items() if k in res and num(res[k]) is not None}
            if rec:
                _tv_write({"step": _tv["step"], **rec})
            return res
        return f

    def save_reconstruction_snapshot(orig):
        def f(masked=False, *a, **kw):
            out = orig(masked, *a, **kw)
            src = Path(str(g.get("preview_dir", ""))) / "latest_recon_clean.png"
            if out and not masked and _tv["dir"] and src.exists():
                shutil.copyfile(src, _tv["dir"] / f"recon_{_tv['step']:06d}.png")
            return out
        return f

    for name, make in (("start_run", start_run), ("resume_run", resume_run), ("finish_run", finish_run),
                       ("log_metrics", log_metrics), ("evaluate_multiview", evaluate_multiview),
                       ("save_reconstruction_snapshot", save_reconstruction_snapshot)):
        wrap(name, make)
    TRAINVIZ_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"train-viz sink attached -> {TRAINVIZ_ROOT}")
