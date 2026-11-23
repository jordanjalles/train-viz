"""Build public/corrector.html: samples per number of corrections K and the FID per K of the corrector run of the 3090.

    python3 scripts/build_corrector.py

Reads gen_corr.jsonl (scores per K, gen_eval_spark.py --corr-k), gen_metrics.jsonl (the standard score, K = 1) and
corrgrid_step100000.png (gen_eval_spark.py --corr-grid) of the run folder. A missing score shows as "pending".
"""
import json, shutil
from pathlib import Path

PUB = Path(__file__).resolve().parent.parent / "public"
RUN = Path("<data-root>/outputs/3090-experiments/runs/unify-wclamp01-corrector-seed2-inet128-100k--a0ecb425a1184541a35964e7ac433886")
STEP, BASE_FID, SPLIT, BLOCKS = 100000, 136.8, 4, 8
KS = [0, 1, 2, 3, 4, 5, 6, 8]


def rows_of(name):
    f = RUN / name
    return [json.loads(line) for line in f.read_text().splitlines() if line.strip()] if f.exists() else []


corr = [r for r in rows_of("gen_corr.jsonl") if r.get("step") == STEP and "fid_ema" in r]
fid = {}                                                     # (k, sampler steps) -> row
ranged = [r for r in corr if r.get("gen_corr_range")]         # scores with more corrections in a range of noise levels only
for r in corr:
    if not r.get("gen_corr_range"):
        fid[(int(r["gen_corr_k"]), int(r["gen_sampler_steps"]))] = r
for r in rows_of("gen_metrics.jsonl"):                       # the standard score: k = 1, 16 steps
    if r.get("step") == STEP and "fid_ema" in r and int(r.get("gen_sampler_steps", 16)) == 16:
        fid.setdefault((1, 16), r)
grid = RUN / f"corrgrid_step{STEP:06d}.png"
grid_ks, grid_steps = list(range(7)), 16
if not grid.exists() and (RUN / f"corrk_{STEP:06d}.png").exists():
    grid, grid_ks, grid_steps = RUN / f"corrk_{STEP:06d}.png", [0, 1, 2, 4, 8], 20       # the grid of the run itself, until the K = 0..6 grid is rendered
if grid.exists():
    (PUB / "corrector").mkdir(exist_ok=True)
    shutil.copy(grid, PUB / "corrector/grid.png")

# chart: FID against K at 16 sampler steps
points = [(k, fid[(k, 16)]["fid_ema"]) for k in KS if (k, 16) in fid]
W, H, L, R, T, B = 720, 320, 56, 24, 20, 44
values = [v for _, v in points] + [BASE_FID]
lo, hi = min(values) - 4, max(values) + 4
x = lambda k: L + (W - L - R) * k / 8
y = lambda v: T + (H - T - B) * (hi - v) / (hi - lo)
ticks = [lo + (hi - lo) * i / 4 for i in range(5)]
svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="FID per number of corrections K">']
for v in ticks:
    svg.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}"/><text class="tick" x="{L - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{v:.0f}</text>')
for k in range(9):
    svg.append(f'<text class="tick" x="{x(k):.1f}" y="{H - B + 18}" text-anchor="middle">{k}</text>')
svg.append(f'<text class="tick" x="{(L + W - R) / 2:.0f}" y="{H - 6}" text-anchor="middle">corrections per sampler step (K)</text>')
svg.append(f'<line class="ref" x1="{L}" x2="{W - R}" y1="{y(BASE_FID):.1f}" y2="{y(BASE_FID):.1f}"/><text class="reflabel" x="{W - R}" y="{y(BASE_FID) - 6:.1f}" text-anchor="end">baseline, 8 blocks: {BASE_FID}</text>')
if len(points) > 1:
    svg.append('<polyline class="line" points="' + " ".join(f"{x(k):.1f},{y(v):.1f}" for k, v in points) + '"/>')
for k, v in points:
    svg.append(f'<g><circle class="dot" cx="{x(k):.1f}" cy="{y(v):.1f}" r="5"><title>K = {k}: FID {v:.1f}, {16 * (SPLIT + k * (BLOCKS - SPLIT))} block evaluations</title></circle>'
               f'<text class="val" x="{x(k):.1f}" y="{y(v) - 10:.1f}" text-anchor="middle">{v:.1f}</text></g>')
svg.append("</svg>")
chart = "\n".join(svg) if points else "<p>No score yet.</p>"


def table_row(k, steps, note=""):
    r = fid.get((k, steps))
    evals = steps * (SPLIT + k * (BLOCKS - SPLIT))
    cells = (f"{r['fid_ema']:.1f}", f"{r['kid_ema']:.1f}", f"{r['fd_ae_z_ema']:.0f}") if r else ("pending",) * 3
    return f"<tr><td>{k}</td><td>{steps}</td><td>{evals}</td><td>{cells[0]}</td><td>{cells[1]}</td><td>{cells[2]}</td><td class='note'>{note}</td></tr>"


table = "\n".join([table_row(k, 16, "base denoiser alone" if k == 0 else "the trained setting" if k == 1 else "") for k in KS]
                  + [table_row(0, 32, "same cost as K = 1 at 16 steps"), table_row(1, 24, "same cost as K = 2 at 16 steps"), table_row(4, 8, "few sampler steps, many corrections"),
                     table_row(0, 48, "many sampler steps"), table_row(1, 48, "many sampler steps"), table_row(3, 48, "many sampler steps")])
RANGES = [([0.7, 1.0], "high noise: the first sampler steps"), ([0.3, 0.7], "medium noise"), ([0.0, 0.3], "low noise: the last sampler steps"),
          ([0.0, 0.2], "the range of the paper's JiT setting (t 0.8 to 1, t = 1 clean)")]
range_rows = []
for k_out in (1, 0):
    for limits, label in RANGES:
        hit = [r for r in ranged if json.loads(r["gen_corr_range"]) == limits and int(r["gen_corr_k_out"]) == k_out and int(r["gen_sampler_steps"]) == 16]
        r = hit[-1] if hit else None
        cells = (f"{r['gen_block_evals']:.0f}", f"{r['fid_ema']:.1f}", f"{r['kid_ema']:.1f}", f"{r['fd_ae_z_ema']:.0f}") if r else ("", "pending", "pending", "pending")
        range_rows.append(f"<tr><td>{limits[0]:g} to {limits[1]:g}</td><td>3</td><td>{k_out}</td><td>{cells[0]}</td><td>{cells[1]}</td><td>{cells[2]}</td><td>{cells[3]}</td><td class='note'>{label}</td></tr>")
range_table = "\n".join(range_rows)
same = lambda k: f"{fid[(k, 16)]['fid_ema']:.1f}" if (k, 16) in fid else "pending"
image = ('<img src="corrector/grid.png" alt="Samples: one row per K, the same 8 seeds in every row" loading="lazy">'
         if grid.exists() else "<p>The sample grid is not rendered yet.</p>")

html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Corrections per step</title>
<style>
  :root {{ --bg: #090c12; --panel: #0f141d; --line: rgba(255,255,255,0.13); --ink: #e8ecf3; --ink2: #a7b0c0; --accent: hsl(192 86% 64%); }}
  @media (prefers-color-scheme: light) {{
    :root {{ --bg: #f5f6f8; --panel: #ffffff; --line: rgba(0,0,0,0.14); --ink: #141821; --ink2: #4c5566; --accent: hsl(192 80% 32%); }}
  }}
  body {{ margin: 0; padding: 20px 16px 48px; background: var(--bg); color: var(--ink); font: 15px/1.5 system-ui, sans-serif; }}
  main {{ max-width: 1080px; margin: 0 auto; }}
  h1 {{ font-size: 20px; margin: 0 0 6px; }}
  h2 {{ font-size: 16px; margin: 30px 0 4px; }}
  p {{ color: var(--ink2); margin: 0 0 12px; max-width: 78ch; }}
  a {{ color: var(--accent); }}
  img {{ display: block; width: 100%; height: auto; border-radius: 8px; border: 1px solid var(--line); background: #fff; }}
  svg {{ display: block; width: 100%; max-width: 720px; height: auto; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }}
  .grid {{ stroke: var(--line); stroke-width: 1; }}
  .tick, .reflabel {{ fill: var(--ink2); font-size: 12px; }}
  .val {{ fill: var(--ink); font-size: 12px; font-variant-numeric: tabular-nums; }}
  .line {{ fill: none; stroke: var(--accent); stroke-width: 2; }}
  .dot {{ fill: var(--accent); stroke: var(--panel); stroke-width: 2; }}
  .ref {{ stroke: var(--ink2); stroke-width: 1.5; stroke-dasharray: 5 5; }}
  .rows {{ display: grid; grid-template-columns: auto 1fr; gap: 0 10px; align-items: stretch; }}
  .klabels {{ display: grid; color: var(--ink2); font-size: 13px; font-variant-numeric: tabular-nums; }}
  .klabels span {{ display: flex; align-items: center; white-space: nowrap; }}
  table {{ border-collapse: collapse; margin: 6px 0 4px; font-size: 14px; font-variant-numeric: tabular-nums; }}
  th, td {{ text-align: right; padding: 5px 12px; border-bottom: 1px solid var(--line); }}
  th {{ color: var(--ink2); font-weight: 600; }}
  td.note, th.note {{ text-align: left; color: var(--ink2); }}
  .wrap {{ overflow-x: auto; }}
</style>
</head>
<body>
<main>
<h1>Correction training: samples and FID per number of corrections K</h1>
<p>Run: <code>unify-wclamp01-corrector-seed2-inet128-100k</code>, EMA weights, step 100,000.
Blocks 1 to 4 predict the clean image. Blocks 5 to 8 correct that image. The sampler applies the correction K times in each step.
K = 1 is the trained setting. <a href="/">Back to train-viz</a></p>

<h2>FID per K</h2>
<p>16 Euler steps, 10,000 samples. Lower is better. Move the pointer over a point for the block evaluations.</p>
{chart}

<h2>Samples per K</h2>
<p>Each column is one seed. Each row is one K, {grid_steps} Euler steps.{"" if grid_ks == list(range(7)) else " This is the grid of the run itself. The grid for K = 0 to 6 replaces it when it is rendered."}</p>
<div class="rows">
  <div class="klabels">{"".join(f"<span>K = {k}</span>" for k in grid_ks)}</div>
  {image}
</div>

<h2>All scores</h2>
<div class="wrap">
<table>
  <tr><th>K</th><th>Sampler steps</th><th>Block evaluations</th><th>FID</th><th>KID × 1000</th><th>fd_ae_z</th><th class="note">Note</th></tr>
{table}
</table>
</div>
<p>One sampler step costs 4 + 4 × K block evaluations. The baseline with 8 blocks costs 128 at 16 steps and has FID {BASE_FID}.</p>

<h2>More corrections in a range of noise levels only</h2>
<p>16 Euler steps. The sampler applies 3 corrections at the steps whose noise level is inside the range, and K outside at the other steps.
For comparison at 16 steps: K = 0 everywhere {same(0)}, K = 1 everywhere {same(1)}, K = 3 everywhere {same(3)}.</p>
<div class="wrap">
<table>
  <tr><th>Noise levels</th><th>K inside</th><th>K outside</th><th>Block evaluations</th><th>FID</th><th>KID × 1000</th><th>fd_ae_z</th><th class="note">Note</th></tr>
{range_table}
</table>
</div>
</main>
</body>
</html>
"""
(PUB / "corrector.html").write_text(html)
print("corrector.html written:", {f"k{k}@{s}": round(r["fid_ema"], 1) for (k, s), r in sorted(fid.items())}, "| grid:", grid.exists())
