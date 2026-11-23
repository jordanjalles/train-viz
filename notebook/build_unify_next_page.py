"""Build public/unify_next.html: the one ordered checklist for ImageNet-Unify.ipynb.

    python3 build_unify_next_page.py

It pulls each code block from the tested cell files, so the checklist cannot drift from them.
State of the notebook it starts from: commit 726e062 (2026-10-05 15:10).
"""
import html
import re
from pathlib import Path

HERE = Path(__file__).parent
css = re.search(r"<style>(.*?)</style>", (HERE / "build_unify_page.py").read_text(), re.S).group(1)


def block(file, bid):
    text = (HERE / file).read_text()
    m = re.search(rf"^# === BLOCK {bid}[: ].*?===\n(.*?)(?=^# === BLOCK |\Z)", text, re.M | re.S)
    assert m, (file, bid)
    return m.group(1).rstrip("\n") + "\n"


def copy(code, label):
    return f"""<div class="copy">
<button>{label}</button><span class="msg" role="status"></span>
<details><summary>Show the code ({len(code.splitlines())} lines)</summary><pre>{html.escape(code)}</pre></details>
</div>"""


PATCH = copy(block("unify_patch_cells.py", 1), "Copy the patch cell")
CHECK = copy(block("unify_patch_cells.py", 2), "Copy the color check")
VAL = copy(block("unify_eval_cells.py", 1), "Copy the validation loader")
READOUT = copy(block("unify_train_cells.py", 3), "Copy the readouts cell")
LOOP = copy(block("unify_train_cells.py", 4), "Copy the training loop")

PAGE = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Unify notebook: next steps</title>
<style>{css}
  .part {{ margin: 30px 0 8px; font-size: 13px; letter-spacing: .06em; text-transform: uppercase; color: #6b7482; }}
  ol.steps {{ counter-reset: none; }}
  ol.steps > li {{ counter-increment: step; }}
  main {{ counter-reset: step; }}
</style>
</head>
<body>
<main>
<h1>Unify notebook: what to do, in order</h1>
<p class="lead">Notebook: <code>ImageNet-Unify.ipynb</code>, as you pushed it at 15:10. Nine steps. Do them from top to bottom.
Steps 1 to 6 edit cells. Steps 7 to 9 run the notebook.</p>

<div class="part">Part A · edit the notebook (run nothing yet)</div>
<ol class="steps">
  <li><span class="where">Cell <code>class ChannelGroupResize(nn.Module):</code></span>
    <span class="do">Replace the whole cell with this. It fixes the grey samples.</span>
    {PATCH}
  </li>
  <li><span class="where">The next two cells: <code>class LinSkipPatchEmbed</code> and <code>class LinSkipUnEmbed</code></span>
    <span class="do">Delete both cells. The cell from step 1 already holds these two classes.</span>
  </li>
  <li><span class="where">New cell, directly below the cell from step 1</span>
    <span class="do">Paste this. It is a check that prints three lines when it runs.</span>
    {CHECK}
  </li>
  <li><span class="where">New cell, directly below the cell that starts with <code>train_ds = BufferedImageNet(</code></span>
    <span class="do">Paste this. It gives the loop a test loss.</span>
    {VAL}
  </li>
  <li><span class="where">Cell <code># training readouts (definitions only)</code></span>
    <span class="do">Replace the whole cell with this. It adds the fix for the resume error.</span>
    {READOUT}
  </li>
  <li><span class="where">Cell <code># MAIN TRAINING LOOP</code></span>
    <span class="do">Replace the whole cell with this.</span>
    {LOOP}
  </li>
</ol>

<div class="part">Part B · run</div>
<ol class="steps">
  <li><span class="where">Kernel menu</span>
    <span class="do">Restart the kernel. This clears the old grey model, the old logs and the broken data loader in one go.</span>
  </li>
  <li><span class="where">From the first cell down to <code># training readouts (definitions only)</code></span>
    <span class="do">Run the cells in order, down to and including the readouts cell.</span>
    <span class="result-label">The check cell from step 3 should print:</span>
    <div class="result">red   in -> out RGB [1.0, 0.0, 0.0]
green in -> out RGB [0.0, 1.0, 0.0]
blue  in -> out RGB [0.0, 0.0, 1.0]</div>
    <div class="caution">If the three lines show equal numbers such as <code>[0.33, 0.33, 0.33]</code>, the old patch cells are still in the
    notebook. Go back to steps 1 and 2.</div>
    <span class="result-label">The init cell should print the parameter count:</span>
    <div class="result">… ae parameters</div>
  </li>
  <li><span class="where">Cell <code># MAIN TRAINING LOOP</code></span>
    <span class="do">Run it. Skip the three micro-batch cells above it. Your overfit cell is commented out, so they do nothing.</span>
    <span class="result-label">The first line should look like this (numbers will differ):</span>
    <div class="result">START step 0 | train-loss: … | test-loss: … | best test … at step 0</div>
  </li>
</ol>

<h2>Later: if you stop the loop and want to go on</h2>
<p class="small">Run the <code># MAIN TRAINING LOOP</code> cell again. Nothing else is needed. It continues from the logged step. If the data
loader workers had stopped, it prints this first:</p>
<div class="result">data loader: workers had stopped, starting new ones
START step 12,500 | train-loss: … | test-loss: … | best test … at step …</div>

<h2>What each edit is for</h2>
<table>
  <tr><th>Step</th><th>Why</th></tr>
  <tr><td>1, 2</td><td>The old patch skip grouped the R, G and B of each pixel into one token channel, so the model's default path was grey.</td></tr>
  <tr><td>3</td><td>A check that the new patch cell passes color.</td></tr>
  <tr><td>4</td><td>The notebook had no <code>test_dataloader</code>, so the loop printed <code>test-loss: n/a</code>.</td></tr>
  <tr><td>5, 6</td><td>Stopping the kernel ends the data loader's workers. The loop now restarts them, so "DataLoader worker exited unexpectedly" does not come back.</td></tr>
  <tr><td>7</td><td>A model trained with the grey skip cannot be continued. A restart gives a fresh model and empty logs.</td></tr>
</table>

<p class="small">Every code block here is pulled from the files I tested on the CPU with your notebook's cells. None has run on a GPU.
The longer explanations are on <a href="/unify_patch.html">grey samples</a>, <a href="/unify_train.html">training loop</a> and
<a href="/unify_eval.html">split loss</a>. You do not need them to follow this page.</p>
<p class="small"><a href="/">Back to train-viz</a></p>
</main>
<script>
  for (const box of document.querySelectorAll('.copy')) {{
    const pre = box.querySelector('pre');
    const msg = box.querySelector('.msg');
    box.querySelector('button').addEventListener('click', async () => {{
      const code = pre.textContent;
      try {{ await navigator.clipboard.writeText(code); msg.textContent = 'Copied ' + (code.split('\\n').length - 1) + ' lines'; }}
      catch (e) {{
        box.querySelector('details').open = true;
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

out = HERE.parent / "public" / "unify_next.html"
out.write_text(PAGE)
print("wrote", out, len(PAGE), "bytes")
