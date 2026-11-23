"""Build public/unify_shape.html from a shape_probe.py result: where each block of a Unify model works.

    python3 build_unify_shape_page.py [<data-root>/outputs/imagenet-selfteach/unify_eval/shape.json]

One column per model. Three small charts per model share the x axis (the blocks, in order):
update size of the attention and MLP branch, the loss rise when the block is skipped, and the
learned gain (post-norm weight x time gate). A looped middle block has one thin bar per loop.
"""
import html
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
src = Path(sys.argv[1] if len(sys.argv) > 1 else "<data-root>/outputs/imagenet-selfteach/unify_eval/shape.json")
data = json.loads(src.read_text())
css = re.search(r"<style>(.*?)</style>", (HERE / "build_unify_page.py").read_text(), re.S).group(1)
ATTN, MLP, SKIP, INK, GRID = "hsl(192 86% 64%)", "hsl(36 86% 65%)", "hsl(340 75% 68%)", "#9aa4b5", "rgba(255,255,255,0.10)"
ROLE_BG = {"shallow": "rgba(255,255,255,0.00)", "middle": "rgba(255,255,255,0.055)", "final": "rgba(255,255,255,0.00)", "block": "rgba(255,255,255,0.00)"}
W, PAD_L, PAD_R = 430, 44, 6


def chart(model, title, lo, hi, ticks, bars, height=118, fmt=lambda v: f"{v:g}"):
    """bars: per block a list of (value, color, tooltip). Log y axis from lo to hi."""
    n = len(bars)
    top, bottom = 16, 18
    plot_h = height - top - bottom
    col_w = (W - PAD_L - PAD_R) / n
    y = lambda v: top + plot_h * (1 - (math.log10(min(max(v, lo), hi)) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)))
    out = [f'<svg viewBox="0 0 {W} {height}" role="img" aria-label="{html.escape(title)}">',
           f'<text x="0" y="10" fill="{INK}" font-size="10.5">{html.escape(title)}</text>']
    roles = [l["role"] for l in model["block_rows"]]
    for i, role in enumerate(roles):
        out.append(f'<rect x="{PAD_L + i * col_w:.1f}" y="{top}" width="{col_w:.1f}" height="{plot_h}" fill="{ROLE_BG[role]}"/>')
    for t in ticks:
        out.append(f'<line x1="{PAD_L}" x2="{W - PAD_R}" y1="{y(t):.1f}" y2="{y(t):.1f}" stroke="{GRID}"/>')
        out.append(f'<text x="{PAD_L - 5}" y="{y(t) + 3:.1f}" fill="{INK}" font-size="9" text-anchor="end">{fmt(t)}</text>')
    for i, group in enumerate(bars):
        inner = col_w - 6
        bw = inner / max(len(group), 1)
        for j, (value, color, tip) in enumerate(group):
            x = PAD_L + i * col_w + 3 + j * bw
            yy = y(value)
            out.append(f'<rect x="{x:.1f}" y="{yy:.1f}" width="{max(bw - 1, 1):.1f}" height="{max(top + plot_h - yy, 1):.1f}" fill="{color}" rx="1.5">'
                       f'<title>{html.escape(tip)}</title></rect>')
        out.append(f'<text x="{PAD_L + (i + 0.5) * col_w:.1f}" y="{height - 5}" fill="{INK}" font-size="9" text-anchor="middle">{i}</text>')
    out.append("</svg>")
    return "".join(out)


def model_panel(m):
    blocks = sorted({l["block"] for l in m["layers"]})
    rows = {b: [l for l in m["layers"] if l["block"] == b] for b in blocks}
    m["block_rows"] = [rows[b][0] for b in blocks]
    name = lambda l: f"block {l['block']}" + (f", loop {l['loop']}" if l["loop"] else "")
    upd = [[(l[k], c, f"{name(l)}: {lab} update = {l[k]:.2f} x the stream") for l in rows[b] for k, c, lab in (("attn", ATTN, "attention"), ("mlp", MLP, "MLP"))]
           for b in blocks]
    skips = {s["blocks"][0]: s for s in m["skips"] if s["what"].startswith("block ")}
    rise = lambda s: 100 * (s["loss"] / m["loss"] - 1)
    skip = [[(max(rise(skips[b]), 0.5), SKIP, f"block {b} skipped: test loss {skips[b]['loss']:.4f}, {rise(skips[b]):+.1f}%")] for b in blocks]
    gain = [[(rows[b][0][f"gain_{k}"] * rows[b][0][f"gate_{k}"], c, f"block {b} {lab}: post-norm gain {rows[b][0][f'gain_{k}']:.4f} x time gate {rows[b][0][f'gate_{k}']:.1f}")
             for k, c, lab in (("attn", ATTN, "attention"), ("mlp", MLP, "MLP"))] for b in blocks]
    extra = ""
    if m["looped"]:
        group = {s["what"]: rise(s) for s in m["skips"] if s["role"] == "group"}
        loops = {s["what"].split()[0]: rise(s) for s in m["skips"] if s["role"] == "loops"}
        extra = ('<p class="small"><b>Groups skipped:</b> ' + " · ".join(f"{k} {v:+.1f}%" for k, v in group.items())
                 + '<br><b>Other loop counts</b> (trained with 4): ' + " · ".join(f"{k} loops {v:+.1f}%" for k, v in loops.items())
                 + f'<br><b>Fusion layer:</b> weight norm of the shallow half {m["fusion"]["shallow_half_norm"]:.2f}, of the looped half '
                   f'{m["fusion"]["deep_half_norm"]:.2f} (both {m["fusion"]["init_half_norm"]:.0f} at init).</p>')
    return (f'<div class="panel"><h3>{html.escape(m["label"])}</h3>'
            f'<p class="small">{"looped middle (blocks 2 to 5, shaded, one bar pair per loop)" if m["looped"] else "plain 8 blocks"} · '
            f'test loss {m["loss"]:.4f} on these images</p>'
            + chart(m, "update size: RMS(update) / RMS(stream), log scale", 0.1, 10, (0.1, 0.3, 1, 3, 10), upd)
            + chart(m, "loss rise when the block is skipped (%), log scale", 0.5, 500, (1, 10, 100), skip, fmt=lambda v: f"{v:g}%")
            + chart(m, "learned gain: post-norm weight x time gate, log scale", 0.03, 10, (0.1, 1, 10), gain)
            + extra + "</div>")


def table(m):
    head = "<tr><th>block</th><th>role</th><th>loop</th><th>stream RMS</th><th>attention update</th><th>MLP update</th><th>change</th><th>gain attn</th><th>gain MLP</th><th>gate attn</th><th>gate MLP</th></tr>"
    body = "".join(f"<tr><td>{l['block']}</td><td>{l['role']}</td><td>{l['loop'] or ''}</td><td>{l['stream']:.3f}</td><td>{l['attn']:.3f}</td><td>{l['mlp']:.3f}</td>"
                   f"<td>{l['change']:.3f}</td><td>{l['gain_attn']:.4f}</td><td>{l['gain_mlp']:.4f}</td><td>{l['gate_attn']:.1f}</td><td>{l['gate_mlp']:.1f}</td></tr>"
                   for l in m["layers"])
    skips = "".join(f"<tr><td colspan='4'>skip {html.escape(s['what'])}</td><td colspan='7'>loss {s['loss']:.5f} ({100 * (s['loss'] / m['loss'] - 1):+.1f}%)</td></tr>" for s in m["skips"])
    return f"<details><summary>All numbers: {html.escape(m['label'])}</summary><div style='overflow-x:auto'><table class='nums'>{head}{body}{skips}</table></div></details>"


panels = "".join(model_panel(m) for m in data["models"])
tables = "".join(table(m) for m in data["models"])
PAGE = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Where the blocks work</title>
<style>{css}
.models {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 14px; }}
.models .panel {{ padding: 12px 14px; border: 1px solid rgba(255,255,255,0.10); border-radius: 12px; }}
.models svg {{ width: 100%; height: auto; display: block; margin-top: 8px; }}
.key i {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin: 0 5px 0 12px; }}
table.nums {{ border-collapse: collapse; font-size: 12px; margin: 8px 0; }}
table.nums td, table.nums th {{ padding: 2px 8px; text-align: right; border-bottom: 1px solid rgba(255,255,255,0.08); }}
</style>
</head>
<body>
<main style="max-width: 1500px">
<h1>Where the blocks work</h1>
<p class="lead">Unify pixel models, {data['n_images']} validation images, the same t and noise for every model, dense, eval mode.
The x axis of every chart is the block number, first block on the left.</p>
<p class="small key"><i style="background:{ATTN}"></i>attention branch<i style="background:{MLP}"></i>MLP branch<i style="background:{SKIP}"></i>loss rise when skipped
<span style="margin-left:12px">Hover a bar for its number.</span></p>
<div class="models">{panels}</div>

<h2>How to read it</h2>
<ul class="small">
  <li><b>Update size:</b> how much a branch adds to the token stream, as a multiple of the stream it enters. 1 = the update is as large as the stream.</li>
  <li><b>Loss rise when skipped:</b> the block is replaced by "do nothing" and the test loss is measured again. This is what the block contributes.
      A skipped early block also hands the later blocks an input they never saw, so early blocks read high on this test.</li>
  <li><b>Learned gain:</b> each branch ends in a post-norm (weight 1e-4 at init) times a time gate (1 at init). Their product is the size the model
      gave the branch. The gates grew to between 10 and 100, so the product is far from small although the post-norm weights still are.</li>
</ul>
{tables}
<p class="small">Source: <code>{html.escape(str(src))}</code>, made by <code>unify3090/shape_probe.py</code>. <a href="/">Back to train-viz</a></p>
</main>
</body>
</html>
"""
out = HERE.parent / "public" / "unify_shape.html"
out.write_text(PAGE)
print("wrote", out, len(PAGE), "bytes")
