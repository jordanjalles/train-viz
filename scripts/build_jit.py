"""Build public/jit.html: the JiT noise recipe runs of the 3090 against the baseline and the noise-mean spread run.

    python3 scripts/build_jit.py

Reads gen_metrics.jsonl and recon_100000.png (in-run samples) of each run folder, and traj_probe.json of each
trajectory probe (<data-root>/outputs/contrast-probe/traj-*, averaged weights).
"""
import json, shutil
from pathlib import Path

PUB = Path(__file__).resolve().parent.parent / "public"
RUNS = Path("<data-root>/outputs/3090-experiments/runs")
PROBES = Path("<data-root>/outputs/contrast-probe")
STEP = 100000
# key, run id, name, what changes against the baseline, color light, color dark
SERIES = [
    ("control", "3b58b666", "Baseline", "logit-normal noise levels, mean 0, std 1.5", "#2a78d6", "#3987e5"),
    ("jit", "c88aa584", "JiT recipe", "noise levels of JiT: logit-normal mean 0.586, std 0.8 (more low-noise inputs)", "#eb6834", "#d95926"),
    ("pure5", "8c11fbb2", "JiT + 5% pure noise", "JiT recipe, and 5% of the inputs at noise levels 0.9 to 1", "#1baf7a", "#199e70"),
    ("mean14", "d5278df3", "Noise-mean spread", "baseline noise levels, and each noise sample gets one constant offset: 0.0136 × randn on all pixels", "#eda100", "#c98500"),
]


def run_dir(i):
    return next(RUNS.glob(f"*--{i}*"))


def probe(i):
    d = next(PROBES.glob(f"traj-*{i}*"))
    data = json.loads((d / "traj_probe.json").read_text())
    return data[f"checkpoint_final_step_{STEP:06d}.pt averaged"]


def fids(i):
    f = run_dir(i) / "gen_metrics.jsonl"
    rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()] if f.exists() else []
    ema = {int(r["step"]): r["fid_ema"] for r in rows if "fid_ema" in r and int(r.get("gen_sampler_steps", 16)) == 16}
    raw = {int(r["step"]): r["fid"] for r in rows if "fid" in r and int(r.get("gen_sampler_steps", 16)) == 16}
    return ema, raw


(PUB / "jit").mkdir(exist_ok=True)
data, real = {}, None
for key, i, *_ in SERIES:
    p = probe(i)
    real = real or p["real"]
    data[key] = (p, *fids(i))
    shutil.copy(run_dir(i) / f"recon_{STEP:06d}.png", PUB / f"jit/{key}.png")
data["control"][1].setdefault(STEP, 136.8)                  # the baseline scores of the loop (memory: EMA 136.8, raw 135.1)
data["control"][2].setdefault(STEP, 135.1)

# chart: brightness over the 20 sampler steps
W, H, L, R, T, B = 720, 320, 52, 150, 20, 44
lo, hi = 0.1, 0.5
x = lambda s: L + (W - L - R) * (s - 1) / 19
y = lambda v: T + (H - T - B) * (hi - v) / (hi - lo)
svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Mean brightness of the samples at each sampler step">']
for v in (0.1, 0.2, 0.3, 0.4, 0.5):
    svg.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}"/><text class="tick" x="{L - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{v:.1f}</text>')
for s in (1, 5, 10, 15, 20):
    svg.append(f'<text class="tick" x="{x(s):.1f}" y="{H - B + 18}" text-anchor="middle">{s}</text>')
svg.append(f'<text class="tick" x="{(L + W - R) / 2:.0f}" y="{H - 6}" text-anchor="middle">sampler step (20 Euler steps, noise 1 to 0)</text>')
svg.append(f'<line class="ref" x1="{L}" x2="{W - R}" y1="{y(real["bright"]):.1f}" y2="{y(real["bright"]):.1f}"/>'
           f'<text class="reflabel" x="{W - R + 8}" y="{y(real["bright"]) + 4:.1f}">real images {real["bright"]:.2f}</text>')
ends = []
for key, _, name, _, *_ in SERIES:
    p = data[key][0]
    pts = [(s, p[f"sampler_{s}"]["bright"]) for s in range(1, 21)]
    path = " ".join(f"{x(s):.1f},{y(v):.1f}" for s, v in pts)
    svg.append(f'<polyline class="line s-{key}" points="{path}"/>')
    for s, v in pts:
        svg.append(f'<circle class="hit" cx="{x(s):.1f}" cy="{y(v):.1f}" r="7"><title>{name}, step {s}: brightness {v:.3f}</title></circle>')
    ends.append([pts[-1][1], name, key])
ends.sort()
for j in range(1, len(ends)):                                   # keep the end labels 14 px apart
    if y(ends[j - 1][0]) - y(ends[j][0]) < 14:
        ends[j][0] = hi - (y(ends[j - 1][0]) - 14 - T) * (hi - lo) / (H - T - B)
for v, name, key in ends:
    svg.append(f'<circle class="dot s-{key}" cx="{x(20):.1f}" cy="{y(data[key][0]["final"]["bright"]):.1f}" r="4"/>'
               f'<text class="val" x="{W - R + 8}" y="{y(v) + 4:.1f}">{name}</text>')
svg.append("</svg>")
legend = "".join(f'<span><i class="sw s-{key}"></i>{name}</span>' for key, _, name, *_ in SERIES)

rows = []
for key, i, name, change, *_ in SERIES:
    p, ema, raw = data[key]
    f, first = p["final"], p["sampler_1"]
    c = p.get("coupling", {}).get("noise_to_final", [None])[0]
    fid_e = f"{ema[STEP]:.1f}" if STEP in ema else "pending"
    fid_r = f"{raw[STEP]:.1f}" if STEP in raw else "–"
    rows.append(f"<tr><td class='note'><i class='sw s-{key}'></i>{name}</td><td>{fid_e}</td><td>{fid_r}</td><td>{first['bright']:.2f}</td><td>{f['bright']:.2f}</td>"
                f"<td>{f['between']:.3f}</td><td>{f['within']:.3f}</td><td>{100 * f['dark']:.1f}%</td><td>{100 * f['light']:.1f}%</td><td>{'' if c is None else f'{c:.1f}'}</td></tr>")
rows.append(f"<tr><td class='note'>Real images</td><td></td><td></td><td></td><td>{real['bright']:.2f}</td><td>{real['between']:.3f}</td><td>{real['within']:.3f}</td>"
            f"<td>{100 * real['dark']:.1f}%</td><td>{100 * real['light']:.1f}%</td><td></td></tr>")
fid_over = []
for key, _, name, *_ in SERIES:
    ema = data[key][1]
    fid_over.append(f"<tr><td class='note'><i class='sw s-{key}'></i>{name}</td>" + "".join(f"<td>{ema[s]:.1f}</td>" if s in ema else "<td>–</td>" for s in (25000, 50000, 75000, 100000)) + "</tr>")
settings = "".join(f"<li><b>{name}:</b> {change}.</li>" for _, _, name, change, *_ in SERIES)
grids = "".join(f'<figure><img src="jit/{key}.png" alt="{name}: 12 samples at step 100,000" loading="lazy"><figcaption><i class="sw s-{key}"></i>{name}</figcaption></figure>'
                for key, _, name, *_ in SERIES)
colors_l = " ".join(f".s-{k} {{ --c: {cl}; }}" for k, _, _, _, cl, _ in SERIES)
colors_d = " ".join(f".s-{k} {{ --c: {cd}; }}" for k, _, _, _, _, cd in SERIES)

html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>JiT noise recipe</title>
<style>
  :root {{ --bg: #090c12; --panel: #0f141d; --line: rgba(255,255,255,0.13); --ink: #e8ecf3; --ink2: #a7b0c0; --accent: hsl(192 86% 64%); }}
  {colors_d}
  @media (prefers-color-scheme: light) {{
    :root {{ --bg: #f5f6f8; --panel: #ffffff; --line: rgba(0,0,0,0.14); --ink: #141821; --ink2: #4c5566; --accent: hsl(192 80% 32%); }}
    {colors_l}
  }}
  body {{ margin: 0; padding: 20px 16px 48px; background: var(--bg); color: var(--ink); font: 15px/1.5 system-ui, sans-serif; }}
  main {{ max-width: 1080px; margin: 0 auto; }}
  h1 {{ font-size: 20px; margin: 0 0 6px; }}
  h2 {{ font-size: 16px; margin: 30px 0 4px; }}
  p, li {{ color: var(--ink2); max-width: 82ch; }}
  p {{ margin: 0 0 12px; }}
  a {{ color: var(--accent); }}
  svg {{ display: block; width: 100%; max-width: 720px; height: auto; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }}
  .grid {{ stroke: var(--line); stroke-width: 1; }}
  .tick, .reflabel {{ fill: var(--ink2); font-size: 12px; }}
  .val {{ fill: var(--ink); font-size: 12px; }}
  .line {{ fill: none; stroke: var(--c); stroke-width: 2; }}
  .dot {{ fill: var(--c); stroke: var(--panel); stroke-width: 2; }}
  .hit {{ fill: transparent; }}
  .hit:hover {{ fill: var(--line); }}
  .ref {{ stroke: var(--ink2); stroke-width: 1.5; stroke-dasharray: 5 5; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 4px 16px; margin: 0 0 8px; color: var(--ink2); font-size: 13px; }}
  .sw {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; background: var(--c); margin-right: 6px; vertical-align: -1px; }}
  table {{ border-collapse: collapse; margin: 6px 0 4px; font-size: 14px; font-variant-numeric: tabular-nums; }}
  th, td {{ text-align: right; padding: 5px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; }}
  th {{ color: var(--ink2); font-weight: 600; }}
  td.note, th.note {{ text-align: left; }}
  .wrap {{ overflow-x: auto; }}
  .grids {{ display: grid; grid-template-columns: 1fr; gap: 14px; }}
  @media (min-width: 900px) {{ .grids {{ grid-template-columns: 1fr 1fr; }} }}
  figure {{ margin: 0; }}
  figure img {{ display: block; width: 100%; height: auto; border-radius: 8px; border: 1px solid var(--line); }}
  figcaption {{ color: var(--ink2); font-size: 13px; margin-top: 4px; }}
</style>
</head>
<body>
<main>
<h1>JiT noise recipe: FID, brightness and spread</h1>
<p>Base model (8 blocks, width 360), seed 2, 100,000 steps on the 3090. Scores: 10,000 samples, 16 Euler steps.
Probe: 96 samples, 20 Euler steps, averaged (EMA) weights. <a href="/">Back to train-viz</a></p>
<ul>{settings}</ul>

<h2>Scores and sample statistics at step 100,000</h2>
<div class="wrap">
<table>
  <tr><th class="note">Run</th><th>FID EMA</th><th>FID raw</th><th>First prediction brightness</th><th>Final brightness</th><th>Spread between images</th><th>Contrast within images</th><th>Black pixels</th><th>White pixels</th><th>Noise mean → final brightness</th></tr>
{chr(10).join(rows)}
</table>
</div>
<p>Spread between images: std of the image means. Contrast within images: mean std of the pixels in one image.
The last column is the slope of the final brightness against the mean of the starting noise: how much a tiny offset of the noise moves the image.</p>

<h2>Brightness at each sampler step</h2>
<div class="legend">{legend}</div>
{chr(10).join(svg)}
<p>The JiT recipe trains almost never at noise levels above 0.9 (0.025% of the inputs), so its first prediction is dark and the samples stay dark.
5% of inputs at nearly pure noise fixes the brightness but not the spread between images.</p>

<h2>FID over training (EMA)</h2>
<div class="wrap">
<table>
  <tr><th class="note">Run</th><th>25k</th><th>50k</th><th>75k</th><th>100k</th></tr>
{chr(10).join(fid_over)}
</table>
</div>

<h2>Samples at step 100,000</h2>
<p>The in-run sample grids, 12 samples per run.</p>
<div class="grids">{grids}</div>
</main>
</body>
</html>
"""
(PUB / "jit.html").write_text(html)
print("jit.html written:", {k: (round(v[1].get(STEP, 0), 1), round(v[0]["final"]["bright"], 3)) for k, v in data.items()})
