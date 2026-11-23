"""Build public/sprint.html from sprint_cells.py, so the page and the tested code cannot drift.

    python3 build_sprint_page.py
"""
import html
import re
from pathlib import Path

HERE = Path(__file__).parent
text = (HERE / "sprint_cells.py").read_text()
parts = re.split(r"^# === (BLOCK \d): (.*?) ===\n", text, flags=re.M)[1:]
blocks = {parts[i]: parts[i + 2].rstrip("\n") + "\n" for i in range(0, len(parts), 3)}
assert len(blocks) == 6, list(blocks)


def copy(key, label):
    code = blocks[key]
    return f"""<div class="copy">
<button>{label}</button><span class="msg" role="status"></span>
<details><summary>Show the code ({len(code.splitlines())} lines)</summary><pre>{html.escape(code)}</pre></details>
</div>"""


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Switch to sparse middle</title>
<style>
  body { margin: 0; padding: 20px 16px 48px; background: #090c12; color: #e8ecf3; font: 16px/1.55 system-ui, sans-serif; }
  main { max-width: 760px; margin: 0 auto; }
  h1 { font-size: 22px; margin: 0 0 4px; }
  h2 { font-size: 16px; margin: 34px 0 6px; }
  p, td, th { color: #a7b0c0; }
  .lead { margin: 0 0 22px; }
  code { color: #e8ecf3; font: 13px ui-monospace, Menlo, Consolas, monospace; background: rgba(255,255,255,0.06); padding: 1px 5px; border-radius: 5px; }
  b { color: #e8ecf3; }
  ol.steps { list-style: none; padding: 0; margin: 0; counter-reset: step; }
  ol.steps > li { counter-increment: step; position: relative; padding: 14px 16px 14px 58px; margin: 0 0 10px; background: #121822; border: 1px solid rgba(255,255,255,0.09); border-radius: 14px; }
  ol.steps > li::before { content: counter(step); position: absolute; left: 16px; top: 14px; width: 28px; height: 28px; border-radius: 50%; background: hsl(192 86% 64%); color: #06222a; font-weight: 700; display: grid; place-items: center; }
  .where { display: block; font-size: 12.5px; color: #6b7482; text-transform: uppercase; letter-spacing: .04em; margin-bottom: 2px; }
  .do { color: #e8ecf3; }
  button { font: inherit; font-size: 14px; font-weight: 600; padding: 8px 18px; border-radius: 999px; border: 0; background: hsl(192 86% 64%); color: #06222a; cursor: pointer; min-height: 40px; }
  .copy { margin-top: 10px; }
  .msg { margin-left: 10px; color: hsl(150 60% 55%); font-size: 14px; }
  details { margin-top: 8px; }
  summary { cursor: pointer; color: #6b7482; font-size: 13px; }
  pre { margin: 8px 0 0; padding: 12px; background: #0b1018; border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; overflow-x: auto; font: 12.5px/1.45 ui-monospace, Menlo, Consolas, monospace; color: #e8ecf3; }
  .swap { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 10px; font: 13px ui-monospace, Menlo, Consolas, monospace; }
  .swap span { padding: 6px 10px; border-radius: 8px; background: #0b1018; border: 1px solid rgba(255,255,255,0.1); }
  .swap .old { color: #e66767; text-decoration: line-through; }
  .swap .new { color: hsl(150 60% 55%); }
  .result { margin-top: 10px; padding: 10px 12px; background: #0b1018; border-left: 3px solid hsl(150 60% 55%); border-radius: 6px; overflow-x: auto; white-space: pre; font: 12.5px/1.5 ui-monospace, Menlo, Consolas, monospace; color: #e8ecf3; }
  .result-label { display: block; margin-top: 10px; font-size: 13px; color: #6b7482; }
  table { border-collapse: collapse; width: 100%; margin: 4px 0; font-size: 15px; }
  th, td { text-align: left; padding: 8px 12px 8px 0; border-bottom: 1px solid rgba(255,255,255,0.09); vertical-align: top; }
  th { color: #e8ecf3; font-weight: 600; }
  a { color: hsl(192 86% 64%); }
  .small { font-size: 14px; }
</style>
</head>
<body>
<main>
<h1>Switch the flow model to a sparse middle</h1>
<p class="lead">Notebook: <code>ImageNet1-Latent-Flow-Matching-XM.ipynb</code>. Three small edits and one check.
The model is your <code>SparseMiddleTransformer</code> from the AE notebook, ported to the flow model.</p>

<ol class="steps">
  <li><span class="where">First cell (hyperparameters)</span>
    <span class="do">Paste this below the line that starts with <code>XM_N = 1</code>.</span>
    COPY1
  </li>
  <li><span class="where">New cell, directly below <code>#define latent flow model</code></span>
    <span class="do">Paste this into the new cell. It only defines things.</span>
    COPY2
  </li>
  <li><span class="where">Cell <code># -------------------- Init Models + Optimizer</code></span>
    <span class="do">Change one word in the line that builds the model.</span>
    <div class="swap"><span class="old">flow_model = FlowModel(</span><span>&rarr;</span><span class="new">flow_model = SparseMiddleFlowModel(</span></div>
  </li>
  <li><span class="where">Any cell, after the init cell has run</span>
    <span class="do">Restart the kernel and run the cells from the top through the init cell. Then run <code>check_sparse_middle()</code>.</span>
    <span class="result-label">You should see:</span>
    <div class="result">training step, tokens per block: [48, 48, 12, 12, 12, 12, 12, 12, 48, 48]
sampling / eval, tokens per block: [48, 48, 48, 48, 48, 48, 48, 48, 48, 48]</div>
  </li>
</ol>
<p class="small">That is all. Run the rest of the notebook as usual. Sampling and the split losses always use all 48 tokens.</p>

<h2>Turn it on and off</h2>
<table>
  <tr><th>You want</th><th>Do this</th></tr>
  <tr><td>Dense for a whole run</td><td>Set <code>use_sparse_middle = False</code> in the first cell.</td></tr>
  <tr><td>To switch during a run</td><td>Run <code>set_sparse_middle(False)</code> or <code>set_sparse_middle(True)</code> in any cell.</td></tr>
</table>
<p class="small">A switch needs no other cell. The next training step compiles again on its own, so expect one slow step.</p>

<h2>Optional: switch to dense on its own at an epoch</h2>
<ol class="steps">
  <li><span class="where">First cell (hyperparameters)</span>
    <span class="do">Paste this below the lines from step 1. Set the epoch you want.</span>
    COPY3
  </li>
  <li><span class="where">Cell <code># MAIN TRAINING LOOP</code></span>
    <span class="do">Paste this as the first lines inside the <code>for epoch in range(...):</code> loop.</span>
    COPY4
  </li>
</ol>

<h2>Test path-drop guidance</h2>
<p class="small">From the SPRINT paper: the same weights with the middle blocks skipped act as a weak model. Sampling then pushes
away from the weak model: <code>velocity = weak + guidance * (full - weak)</code>. The weak pass runs only the shallow and
final blocks, so it is cheap.</p>
<ol class="steps">
  <li><span class="where">New cell, directly below the cell that starts with <code># # Sweep single-block path-drop autoguidance</code></span>
    <span class="do">Paste this into the new cell. It only defines things.</span>
    COPY5
  </li>
  <li><span class="where">New cell, directly below the cell from the step above</span>
    <span class="do">Paste this and run it after training. Edit the seeds and guidance values here.</span>
    COPY6
    <span class="result-label">You should see:</span>
    <div class="result">a grid of images: one row per seed, one column per guidance value
column 1 "weak model only", column 2 "no guidance", then guidance 1.5, 2 and 3</div>
  </li>
</ol>
<p class="small"><b>How to read it:</b> column 2 is your normal sampler, and it equals <code>denoise_seed</code> for the same seed.
Guidance helps if the columns to its right look sharper or more coherent. Column 1 shows what the weak model makes by
itself. It should be a blurrier or less coherent version of column 2.<br>
<b>One difference from the paper:</b> the paper also trains with the middle path dropped for 10% of samples, so the weak
model is trained for this. Your model was not. It still sees a zero deep path at three of four tokens in every sparse step, so the
weak pass is close to what it trained on. If guidance looks promising, I can add that 10% as one setting.<br>
<b>The paper combines this with class conditioning.</b> Your flow model has no class input, so this is the path-drop part alone.</p>

<h2>Good to know</h2>
<p class="small"><b>Loss lines inside an epoch</b> are from sparse steps and read higher. <b>The epoch-end <code>train-loss</code> and
<code>test-loss</code></b> are dense, so they stay comparable with earlier runs.<br>
<b>Old flow checkpoints</b> load with <code>strict=False</code>, but their outputs change: the fusion layer starts at half shallow,
half deep, as in your AE class.<br>
<b>Tested</b> on the CPU with your notebook's own cells and a tiny fake dataset. It has not run on a GPU.</p>

<p class="small"><a href="/flow_resume.html">Save, load and resume cells</a> · <a href="/">Back to train-viz</a></p>
</main>
<script>
  for (const box of document.querySelectorAll('.copy')) {
    const pre = box.querySelector('pre');
    const msg = box.querySelector('.msg');
    box.querySelector('button').addEventListener('click', async () => {
      const code = pre.textContent;
      try { await navigator.clipboard.writeText(code); msg.textContent = 'Copied ' + (code.split('\\n').length - 1) + ' lines'; }
      catch (e) {
        box.querySelector('details').open = true;
        const r = document.createRange(); r.selectNodeContents(pre);
        const s = getSelection(); s.removeAllRanges(); s.addRange(r);
        msg.textContent = 'Selected: press Ctrl+C (or Cmd+C) to copy';
      }
    });
  }
</script>
</body>
</html>
"""

page = PAGE
for token, key, label in (("COPY1", "BLOCK 1", "Copy the 4 settings"), ("COPY2", "BLOCK 2", "Copy the model cell"),
                          ("COPY3", "BLOCK 3", "Copy the setting"), ("COPY4", "BLOCK 4", "Copy the loop lines"),
                          ("COPY5", "BLOCK 5", "Copy the guidance helpers"), ("COPY6", "BLOCK 6", "Copy the call cell")):
    assert page.count(token) == 1, token
    page = page.replace(token, copy(key, label))
out = HERE.parent / "public" / "sprint.html"
out.write_text(page)
print("wrote", out, len(page), "bytes")
