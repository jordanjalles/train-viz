"""Build public/unify_loader.html: the loader edits for ImageNet-Unify.ipynb.

    python3 build_unify_loader_page.py

The edit texts are read from the experiment job that ran on the 3090
(imagenet-selfteach/unify3090/job/main.py), so the page shows exactly what was timed.
"""
import ast
import html
import re
from pathlib import Path

HERE = Path(__file__).parent
JOB = Path("<data-root>/workspaces/imagenet-selfteach/unify3090/job/main.py")
css = re.search(r"<style>(.*?)</style>", (HERE / "build_unify_page.py").read_text(), re.S).group(1)
consts = {}
for node in ast.parse(JOB.read_text()).body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("LOCK_OLD", "LOCK_NEW", "LOCK_DEF", "DRAFT_LINE"):
        consts[node.targets[0].id] = ast.literal_eval(node.value)
assert len(consts) == 4, list(consts)


def copy(code, label):
    return f"""<div class="copy">
<button>{label}</button><span class="msg" role="status"></span>
<pre>{html.escape(code)}</pre>
</div>"""


def show(code):
    return f"<pre>{html.escape(code)}</pre>"


PAGE = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Faster image loading</title>
<style>{css}
  .swap {{ display: grid; gap: 4px; margin-top: 8px; font: 12.5px ui-monospace, Menlo, Consolas, monospace; }}
  .swap span {{ padding: 5px 9px; border-radius: 8px; background: #0b1018; border: 1px solid rgba(255,255,255,0.1); overflow-x: auto; white-space: pre; }}
  .swap .old {{ color: #e66767; }}
  .swap .new {{ color: hsl(150 60% 55%); }}
</style>
</head>
<body>
<main>
<h1>Faster image loading</h1>
<p class="lead">Notebook: <code>ImageNet-Unify.ipynb</code>. Five small edits, then four cells to run again. The model and its weights are not
touched. These are the same edits the 3090 timing runs used.</p>

<h2>Measured on the 3090 (baseline train step, batch 64)</h2>
<table>
  <tr><th>Loader</th><th>Per step, shards in memory</th><th>Per step, shards read from the disk</th></tr>
  <tr><td>As in the notebook (2 workers)</td><td>about 99 ms</td><td>about 99 ms, with stalls</td></tr>
  <tr><td>These edits (4 workers, one disk reader, reduced-size decode)</td><td>31 ms, the GPU's own speed</td><td>about 150 ms</td></tr>
</table>
<p class="small">Over a 4,000-step run the edits averaged 79 ms per step against about 99 ms: 1.25 times faster. The image decoding is no longer
the limit. Reading 112 KiB per image from the dataset disk is.</p>

<h2>Edits</h2>
<ol class="steps">
  <li><span class="where">First cell (hyperparameters)</span>
    <span class="do">Change one value. With more workers, one shard per worker keeps memory use the same.</span>
    <div class="swap"><span class="old">SHARDS_IN_RAM = 2</span><span class="new">SHARDS_IN_RAM = 1</span></div>
  </li>
  <li><span class="where">Cell <code>#imagenet full image</code></span>
    <span class="do">Add this line as the first line inside <code>def __call__(self, image):</code>. It lets the JPEG decoder shrink the image while it
    decodes.</span>
    {copy(consts["DRAFT_LINE"], "Copy the line")}
  </li>
  <li><span class="where">Cell <code>class BufferedImageNet(IterableDataset):</code></span>
    <span class="do">Paste this above the line <code>class BufferedImageNet(IterableDataset):</code>, at the top of the cell.</span>
    {copy(consts["LOCK_DEF"], "Copy the lock")}
  </li>
  <li><span class="where">Same cell, inside <code>def load_pool(pool_ids):</code></span>
    <span class="do">Replace these six lines …</span>
    {show(consts["LOCK_OLD"])}
    <span class="do">… with these eight. Only one worker then reads the disk at a time.</span>
    {copy(consts["LOCK_NEW"], "Copy the new lines")}
  </li>
  <li><span class="where">Cells <code>train_ds = BufferedImageNet(</code> and <code># Validation image loader</code></span>
    <span class="do">In both cells, change the number of workers.</span>
    <div class="swap"><span class="old">    num_workers=2,</span><span class="new">    num_workers=4,</span></div>
  </li>
</ol>

<h2>Then run these cells again, in this order</h2>
<ol class="steps">
  <li><span class="where">First cell</span><span class="do">Run it. It only sets values.</span></li>
  <li><span class="where">Cell <code>#imagenet full image</code></span><span class="do">Run it.</span></li>
  <li><span class="where">Cell <code>class BufferedImageNet(IterableDataset):</code></span><span class="do">Run it.</span></li>
  <li><span class="where">Cells <code>train_ds = BufferedImageNet(</code> and <code># Validation image loader</code></span>
    <span class="do">Run both. They build the loaders again.</span>
    <span class="result-label">Then check the speed in the next <code>step</code> lines of the training loop:</span>
    <div class="result">step 12,500 | epoch 1 | 31 ms/step | 2066.0 img/s | …</div>
  </li>
</ol>

<h2>Good to know</h2>
<ul class="small">
  <li><b>The speed comes in bursts.</b> While the shards are in the PC's memory cache, steps run at the GPU's speed. When the loader has to read new
      shards from the dataset disk, steps slow to about 150 ms until it catches up. A long run reads far more than fits in memory.</li>
  <li><b>The reduced-size decode changes the pixels a little.</b> On 256 real images: mean difference 0.007 on the 0 to 1 scale, 99% of values
      within 0.066, largest 0.37 at sharp edges. The image mean does not shift. Leave out step 2 to keep the pixels exactly as before. Full decode
      with 6 workers measured about 1,100 images per second on the 3090, which is 58 ms per step.</li>
  <li><b>Why one disk reader at a time.</b> In a direct test the dataset disk read 154 MiB/s as one stream, and 9 to 32 MiB/s in total with 2 to 6
      streams at once.</li>
  <li><b>Evaluations before and after are not on the same images,</b> if you use the reduced-size decode: <code>get_split_loss</code> keeps its
      first 20 batches. To compare cleanly, run <code>get_eval_batches.cache.clear()</code> and <code>eval_logs.clear()</code> after the edits.</li>
  <li><b>What would remove the disk limit:</b> a pre-resized copy of the dataset, or the shards on a faster disk. Not part of these edits.</li>
</ul>

<p class="small"><a href="/unify_next.html">The setup checklist</a> · <a href="/">Back to train-viz</a></p>
</main>
<script>
  for (const box of document.querySelectorAll('.copy')) {{
    const pre = box.querySelector('pre');
    const msg = box.querySelector('.msg');
    box.querySelector('button').addEventListener('click', async () => {{
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
out = HERE.parent / "public" / "unify_loader.html"
out.write_text(PAGE)
print("wrote", out, len(PAGE), "bytes")
