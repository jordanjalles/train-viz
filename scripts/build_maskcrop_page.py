"""Regenerate public/maskcrop.html (copy-button page) from notebook/maskcrop_cell.py and
notebook/maskcrop_scale.py. Run after changing either: python3 scripts/build_maskcrop_page.py"""
import html
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
cell = (ROOT / "notebook" / "maskcrop_cell.py").read_text()
scale = (ROOT / "notebook" / "maskcrop_scale.py").read_text()
evals = (ROOT / "notebook" / "eval_every_block.py").read_text()
assert max(len(l) for l in (cell + scale + evals).splitlines()) <= 72, "keep lines <= 72 chars"

page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Masked crop cells</title>
<style>
  body {{ margin: 0; padding: 20px 16px 40px; background: #090c12; color: #e8ecf3; font: 15px/1.5 system-ui, sans-serif; }}
  main {{ max-width: 860px; margin: 0 auto; }}
  h1 {{ font-size: 20px; margin: 0 0 6px; }}
  h2 {{ font-size: 16px; margin: 28px 0 4px; }}
  p, li {{ color: #a7b0c0; }}
  code {{ color: #e8ecf3; }}
  button {{ font: inherit; font-weight: 600; padding: 10px 20px; border-radius: 999px; border: 0; background: hsl(192 86% 64%); color: #06222a; cursor: pointer; min-height: 44px; }}
  .msg {{ margin-left: 12px; color: hsl(150 60% 55%); }}
  pre {{ margin-top: 12px; padding: 14px; background: #0f141d; border: 1px solid rgba(255,255,255,0.13); border-radius: 12px; overflow-x: auto; font: 12.5px/1.45 ui-monospace, Menlo, Consolas, monospace; }}
  a {{ color: hsl(192 86% 64%); }}
</style>
</head>
<body>
<main>
<h1>Masked, overhanging crops for ImageNet-AE-selfteach</h1>
<p>The same crop code as the Spark run <code>…maskcrop_ww03</code>, checked
identical. Do these steps between runs.</p>
<ol>
  <li>Replace the whole crop-definitions cell (it starts with
  <code>#ImageNet view / crop definitions</code>) with block 1.</li>
  <li>In the train-step cell, replace <code>def crop_scale(...)</code> with
  block 2. Leave the rest of that cell as it is.</li>
  <li>Restart the kernel and run the cells in order up to the training loop.
  The data loaders must be rebuilt, so re-running only the two edited cells
  is not enough.</li>
</ol>

<section>
<h2>Block 1: crop-definitions cell (replace the whole cell)</h2>
<button>Copy block 1</button><span class="msg" role="status"></span>
<pre>{html.escape(cell)}</pre>
</section>

<section>
<h2>Block 2: <code>crop_scale</code> in the train-step cell</h2>
<button>Copy block 2</button><span class="msg" role="status"></span>
<pre>{html.escape(scale)}</pre>
</section>
<section>
<h2>Block 3 (optional): test every 2k steps up to step 50k</h2>
<p>In the training-loop cell, paste this directly below the
<code>readout.update(...)</code> call, at the same indentation (8 spaces).
Each test also saves a reconstruction image. With the
<a href="/sink.html">current sink cell</a>, each test also saves a Latent
view snapshot. It takes effect the next time you run the training-loop cell.</p>
<button>Copy block 3</button><span class="msg" role="status"></span>
<pre>{html.escape(evals)}</pre>
</section>
<p><a href="/">Back to train-viz</a></p>
</main>
<script>
  for (const section of document.querySelectorAll('section')) {{
    const pre = section.querySelector('pre');
    const msg = section.querySelector('.msg');
    section.querySelector('button').addEventListener('click', async () => {{
      const code = pre.textContent;
      try {{ await navigator.clipboard.writeText(code); msg.textContent = 'Copied ' + (code.split('\\n').length - 1) + ' lines'; }}
      catch (e) {{
        const r = document.createRange(); r.selectNodeContents(pre);
        const s = getSelection(); s.removeAllRanges(); s.addRange(r);
        msg.textContent = 'Selected: press Ctrl+C (or Cmd+C) to copy';
      }}
    }});
  }}
</script>
</body>
</html>
"""
(ROOT / "public" / "maskcrop.html").write_text(page)
print("maskcrop.html updated")

# crop_check.html: one-block page, same look
check = (ROOT / "notebook" / "crop_check_cell.py").read_text()
assert max(len(l) for l in check.splitlines()) <= 72, "keep lines <= 72 chars"
start, end = page.index("<main>"), page.index("</main>")
body = f"""<main>
<h1>Check which crops training uses</h1>
<ol>
  <li>Stop the training loop (interrupt the kernel).</li>
  <li>Copy, paste into a new cell below the training-loop cell, and run it.</li>
  <li>Every line should end in <code>True</code>. Send me the output if not.</li>
</ol>
<table style="border-collapse:collapse;margin:8px 0 4px">
<tr><td style="padding:4px 10px 4px 0">1–3 False</td><td style="color:#a7b0c0">the crop changes are not live: re-run the crop cell, the data cell and the train-step cell, or restart the kernel</td></tr>
<tr><td style="padding:4px 10px 4px 0">4 False</td><td style="color:#a7b0c0">the judge copies an old encoder: re-run the train-step and compile cells after the model cell</td></tr>
</table>
<section>
<button>Copy check cell</button><span class="msg" role="status"></span>
<pre>{html.escape(check)}</pre>
</section>
<p><a href="/maskcrop.html">Crop cells</a> · <a href="/">Back to train-viz</a></p>
"""
check_page = page[:start] + body + page[end:]
check_page = check_page.replace("<title>Masked crop cells</title>", "<title>Crop check</title>")
(ROOT / "public" / "crop_check.html").write_text(check_page)
print("crop_check.html updated")
