# train-viz sink cell for the 3090 notebooks (short lines only).
# Put it in a cell above start_run(...). Same logic as trainviz_sink.py.
import json
import os
import shutil
from pathlib import Path

import torch

# Feed folder per machine: first one that exists and is writable.
_TV_ROOTS = [
    "/var/lib/exp3090/notebook-feed",   # 3090 kernel
    "<sink-dir>",               # Spark kernel
]
TRAINVIZ_ROOT = None
for _r in _TV_ROOTS:
    if os.path.isdir(_r) and os.access(_r, os.W_OK):
        TRAINVIZ_ROOT = Path(_r)
        break
TRAINVIZ_EVERY = 100
TRAINVIZ_DESCRIPTION = ""   # optional: a sentence about this run

_TRAIN_KEYS = {
    "clean_term": "clean",
    "view_term": "view",
    "laplacian_term": "lap",
}
_REG_KEYS = ("swd_term", "bwd_term", "c_swd_term", "t_swd_term")
_EVAL_KEYS = {
    "test clean": "test_clean",
    "test missing": "test_missing",
    "train clean": "train_clean_eval",
    "train missing": "train_missing",
}
_tv = {
    "file": None,
    "dir": None,
    "step": 0,
    "acc": {},
    "n": 0,
    "pending": None,
}


def _tv_write(rec):
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
        rec = {"step": _tv["step"]}
        for k, v in _tv["acc"].items():
            rec[k] = v / _tv["n"]
        _tv_write(rec)
    _tv["acc"] = {}
    _tv["n"] = 0


def _tv_open(run_id, config):
    # Never let the sink break a run: any problem just turns it off.
    try:
        _tv_open_files(run_id, config)
    except Exception as e:
        _tv["file"] = None
        _tv["dir"] = None
        print("train-viz sink off for this run:", e)


def _tv_open_files(run_id, config):
    if TRAINVIZ_ROOT is None:
        raise OSError("no feed folder on this machine")
    d = TRAINVIZ_ROOT / run_id
    d.mkdir(parents=True, exist_ok=True)
    cfg = {"run": run_id}
    cfg.update(config or {})
    text = json.dumps(cfg, indent=1, default=str)
    (d / "config.json").write_text(text)
    if TRAINVIZ_DESCRIPTION:
        desc = TRAINVIZ_DESCRIPTION.strip() + "\n"
        (d / "DESCRIPTION.md").write_text(desc)
    step = 0
    mfile = d / "metrics.jsonl"
    if mfile.exists():
        for line in mfile.read_text().splitlines():
            try:
                step = max(step, json.loads(line)["step"])
            except (ValueError, KeyError):
                pass
    (d / "FINISHED").unlink(missing_ok=True)
    _tv["file"] = open(mfile, "a", buffering=1)
    _tv["dir"] = d
    _tv["step"] = step
    _tv["acc"] = {}
    _tv["n"] = 0
    _tv["pending"] = None


def _tv_close():
    _tv_flush()
    if _tv["file"]:
        if _tv["pending"] is not None:
            line = json.dumps(_tv["pending"]) + "\n"
            _tv["file"].write(line)
        _tv["file"].close()
        (_tv["dir"] / "FINISHED").write_text("")
    _tv["file"] = None
    _tv["dir"] = None
    _tv["pending"] = None


def _tv_num(v):
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _tv_try(fn, *args):
    # Run a sink step; on any error switch the sink off, keep training.
    try:
        fn(*args)
    except Exception as e:
        _tv["file"] = None
        _tv["dir"] = None
        print("train-viz sink stopped:", e)


def _tv_log(metrics):
    if not _tv["file"]:
        return
    _tv["step"] += 1
    acc = _tv["acc"]
    for k, out in _TRAIN_KEYS.items():
        v = _tv_num(metrics.get(k))
        if v is not None:
            acc[out] = acc.get(out, 0.0) + v
    reg = [_tv_num(metrics.get(k)) for k in _REG_KEYS]
    reg = [v for v in reg if v is not None]
    if reg:
        acc["reg"] = acc.get("reg", 0.0) + sum(reg)
    _tv["n"] += 1
    if _tv["step"] % TRAINVIZ_EVERY == 0:
        _tv_flush()


def _tv_eval(res):
    rec = {}
    for k, out in _EVAL_KEYS.items():
        v = _tv_num(res.get(k))
        if v is not None:
            rec[out] = v
    if rec:
        rec["step"] = _tv["step"]
        _tv_write(rec)


def _tv_z(g, n_batches=2):
    # z of the first test images, for the Latent view.
    enc = g.get("encoder")
    loader = g.get("test_dataloader")
    if not _tv["dir"] or enc is None or loader is None:
        return
    dev = next(enc.parameters()).device
    was_training = enc.training
    enc.eval()
    # once per run: 64x64 thumbnails of the same images, same order
    img_file = _tv["dir"] / "z_images.pt"
    want_imgs = not img_file.exists()
    zs, ims = [], []
    try:
        with torch.no_grad():
            for i, (x, _) in enumerate(loader):
                if i >= n_batches:
                    break
                zs.append(enc(x.to(dev)).float().cpu())
                if want_imgs:
                    t = torch.nn.functional.interpolate(
                        x.float(), size=(64, 64), mode="area")
                    t = (t * 255).round().clamp(0, 255)
                    ims.append(t.to(torch.uint8).cpu())
    finally:
        enc.train(was_training)
    name = "z_%06d.pt" % _tv["step"]
    torch.save(torch.cat(zs).half(), _tv["dir"] / name)
    if want_imgs:
        torch.save(torch.cat(ims).contiguous(), img_file)


def _tv_snap(preview_dir):
    src = Path(str(preview_dir)) / "latest_recon_clean.png"
    if _tv["dir"] and src.exists():
        name = "recon_%06d.png" % _tv["step"]
        shutil.copyfile(src, _tv["dir"] / name)


def _tv_adopt(g):
    # A run is already active when this cell runs: attach to it now
    # and back-fill the loss history the notebook holds in memory.
    run = g.get("_active_run")
    if not run or _tv["file"]:
        return
    _tv_open(run.get("run_id"), run.get("config"))
    if not _tv["file"]:
        return
    logs = g.get("logs") or {}
    n = max([len(v) for v in logs.values()] or [0])
    first = _tv["step"] - _tv["step"] % 100 + 100
    for s in range(first, n + 1, 100):
        rec = {"step": s, "reg": 0.0}
        for k, out in _TRAIN_KEYS.items():
            v = logs.get(k)
            if v and len(v) >= s:
                rec[out] = sum(v[s - 100:s]) / 100
        for k in _REG_KEYS:
            v = logs.get(k)
            if v and len(v) >= s:
                rec["reg"] += sum(v[s - 100:s]) / 100
        _tv_write(rec)
    _tv["step"] = max(n, _tv["step"])
    print("train-viz sink: joined the active run at step", n)


def trainviz_attach(namespace=None):
    g = namespace if namespace is not None else globals()

    def wrap(name, make):
        if name not in g:
            return
        if getattr(g[name], "_trainviz", False):
            return
        f = make(g[name])
        f._trainviz = True
        g[name] = f

    def start_run(orig):
        def f(name, config=None, *a, **kw):
            run_id = orig(name, config, *a, **kw)
            if not run_id:
                run = g.get("_active_run") or {}
                run_id = run.get("run_id", name)
            _tv_open(run_id, config)
            return run_id
        return f

    def resume_run(orig):
        def f(*a, **kw):
            out = orig(*a, **kw)
            run = g.get("_active_run") or {}
            run_id = run.get("run_id", "resumed")
            _tv_open(run_id, run.get("config"))
            return out
        return f

    def finish_run(orig):
        def f(*a, **kw):
            try:
                return orig(*a, **kw)
            finally:
                _tv_try(_tv_close)
        return f

    def log_metrics(orig):
        def f(**metrics):
            orig(**metrics)
            _tv_try(_tv_log, metrics)
        return f

    def evaluate_multiview(orig):
        def f(*a, **kw):
            res = orig(*a, **kw)
            _tv_try(_tv_eval, res)
            try:
                _tv_z(g)
            except Exception as e:
                # A failed snapshot must not switch the sink off.
                print("train-viz sink: z snapshot skipped:", e)
            return res
        return f

    def save_reconstruction_snapshot(orig):
        def f(masked=False, *a, **kw):
            out = orig(masked, *a, **kw)
            if out and not masked:
                _tv_try(_tv_snap, g.get("preview_dir", ""))
            return out
        return f

    wrap("start_run", start_run)
    wrap("resume_run", resume_run)
    wrap("finish_run", finish_run)
    wrap("log_metrics", log_metrics)
    wrap("evaluate_multiview", evaluate_multiview)
    snap = "save_reconstruction_snapshot"
    wrap(snap, save_reconstruction_snapshot)
    if TRAINVIZ_ROOT is None:
        print("train-viz sink: no feed folder here, runs not mirrored")
    else:
        print("train-viz sink attached ->", TRAINVIZ_ROOT)
        _tv_try(_tv_adopt, g)


trainviz_attach()
