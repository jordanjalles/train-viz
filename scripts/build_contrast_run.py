"""Build public/contrast_<tag>.html from public/contrast.html: the sampler comparison page for any run folder and step.

    python3 scripts/build_contrast_run.py RUN_FOLDER STEP

Like build_contrast350.py (step 350,000 of the 100M XM run), for any run: grids fixgrid_step<STEP>_*.png of the run folder
-> public/contrast_<tag>/, scores from the rows of that step in gen_fix.jsonl. A setting without a score row shows "pending".
tag: the last word of the run folder name and the step in thousands, e.g. contrast_uniform685.
"""
import json, re, shutil, sys
from pathlib import Path

PUB = Path(__file__).resolve().parent.parent / "public"
RUN, STEP = Path(sys.argv[1]), int(sys.argv[2])
TAG = f"{RUN.name.split('-')[-1]}{STEP // 1000}"
KEYS = {"plain": ('{}', "plain", "Plain sampler"), "cap": ('{"mean_cap": 0.137}', "cap0137", "Spread capped at 0.137, from step 2"),
        "dc05": ('{"noise_dc": 0.5}', "dc05", "Noise mean × 0.5"), "dc0": ('{"noise_dc": 0}', "dc0", "Noise mean × 0")}

(PUB / f"contrast_{TAG}").mkdir(exist_ok=True)
shutil.copy(PUB / "contrast/real.png", PUB / f"contrast_{TAG}/real.png")
rows = [json.loads(line) for line in (RUN / "gen_fix.jsonl").read_text().splitlines() if line.strip()]
stat = {}
for key, (fix, file, title) in KEYS.items():
    shutil.copy(RUN / f"fixgrid_step{STEP:06d}_{file}.png", PUB / f"contrast_{TAG}/{key}.png")
    hit = [r for r in rows if r["step"] == STEP and json.loads(r["gen_fix"]) == json.loads(fix) and "fid_ema" in r]
    if hit:
        r = hit[-1]
        stat[key] = (f"{r['fid_ema']:.2f}", f"{r['kid_ema']:.1f}", f"{r['fd_ae_z_ema']:.0f}", f"{r['gen_between']:.3f}", f"{100 * r['gen_dark']:.1f}%", f"{100 * r['gen_light']:.1f}%")
    else:
        stat[key] = ("pending",) * 6

out, figure_key = [], None
for line in (PUB / "contrast.html").read_text().splitlines():
    shown = re.search(r'<img src="contrast/(\w+)\.png"', line)
    figure_key = shown.group(1) if shown else figure_key          # the caption line that follows belongs to this grid
    line = line.replace("contrast/", f"contrast_{TAG}/").replace("step 200,000", f"step {STEP:,}")
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
html = html.replace("<h1>Black and white pixels:", f"<h1>{RUN.name.split('-')[-1]} run, step {STEP:,}. Black and white pixels:")
html = html.replace("<p>Model: the 100M run with XM transport,", f"<p>Model: run <code>{RUN.name}</code>,")
html = html.replace('<a href="/">Back to train-viz</a>', '<a href="contrast.html">Long run, step 200,000</a> · <a href="contrast350.html">step 350,000</a> · <a href="/">Back to train-viz</a>')
(PUB / f"contrast_{TAG}.html").write_text(html)
print(f"contrast_{TAG}.html written:", {k: v[0] for k, v in stat.items()})
