"""Build public/contrast350.html from public/contrast.html: the same page for step 350,000 of the 100M XM run.

    python3 scripts/build_contrast350.py

Grids: fixgrid_step350000_*.png of the run folder -> public/contrast350/. Scores: gen_fix.jsonl of the run folder (rows
of step 350000). A setting without a score row yet shows "pending" and the statistics of the 250 grid samples stay out.
"""
import json, re, shutil
from pathlib import Path

PUB = Path(__file__).resolve().parent.parent / "public"
RUN = Path("<data-root>/outputs/training-runs/unify-wclamp01-selfflow3to6-selfteach-xmot3-100m-400k-spark")
STEP = 350000
KEYS = {"plain": ("{}", "plain", "Plain sampler"), "cap": ('{"mean_cap": 0.137}', "cap0137", "Spread capped at 0.137, from step 2"),
        "dc05": ('{"noise_dc": 0.5}', "dc05", "Noise mean × 0.5"), "dc0": ('{"noise_dc": 0}', "dc0", "Noise mean × 0")}

(PUB / "contrast350").mkdir(exist_ok=True)
shutil.copy(PUB / "contrast/real.png", PUB / "contrast350/real.png")
rows = [json.loads(line) for line in (RUN / "gen_fix.jsonl").read_text().splitlines() if line.strip()]
stat = {}
for key, (fix, file, title) in KEYS.items():
    shutil.copy(RUN / f"fixgrid_step{STEP:06d}_{file}.png", PUB / f"contrast350/{key}.png")
    hit = [r for r in rows if r["step"] == STEP and json.loads(r["gen_fix"]) == json.loads(fix)]
    if hit:
        r = hit[-1]
        stat[key] = (f"{r['fid_ema']:.2f}", f"{r['kid_ema']:.1f}", f"{r['fd_ae_z_ema']:.0f}", f"{r['gen_between']:.3f}", f"{100 * r['gen_dark']:.1f}%", f"{100 * r['gen_light']:.1f}%")
    else:
        stat[key] = ("pending",) * 6

out, figure_key = [], None
for line in (PUB / "contrast.html").read_text().splitlines():
    shown = re.search(r'<img src="contrast/(\w+)\.png"', line)
    figure_key = shown.group(1) if shown else figure_key          # the caption line that follows belongs to this grid
    line = line.replace("contrast/", "contrast350/").replace("step 200,000", "step 350,000")
    for key, (_, _, title) in KEYS.items():
        fid, kid, fd, spread, black, white = stat[key]
        text = f"FID {fid}, KID {kid}, fd_ae_z {fd}. Spread {spread}, black {black}, white {white}."
        if line.startswith(f"  <tr><td>{title}</td>"):
            line = f"  <tr><td>{title}</td><td>{fid}</td><td>{kid}</td><td>{fd}</td><td>{spread}</td><td>{black}</td><td>{white}</td></tr>"
        elif re.match(rf"\s*{key}: '", line):
            line = f"  {key}: '{title}. {text}',"
        elif figure_key == key and "<figcaption><b>" in line:
            line = re.sub(r"</b>.*", f"</b> {text}" + ("</figcaption>" if "</figcaption>" in line else ""), line)
    out.append(line)
html = "\n".join(out) + "\n"
html = html.replace("<h1>Black and white pixels:", "<h1>Step 350,000. Black and white pixels:").replace('<a href="/">Back to train-viz</a>', '<a href="contrast.html">Step 200,000</a> · <a href="/">Back to train-viz</a>')
(PUB / "contrast350.html").write_text(html)
print("contrast350.html written:", {k: v[0] for k, v in stat.items()})
