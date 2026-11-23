"""Build public/unify_eval.html from unify_eval_cells.py, so the page and the tested code cannot drift.

    python3 build_unify_eval_page.py
"""
import html
import re
from pathlib import Path

HERE = Path(__file__).parent
text = (HERE / "unify_eval_cells.py").read_text()
parts = re.split(r"^# === (BLOCK \d): (.*?) ===\n", text, flags=re.M)[1:]
blocks = {parts[i]: parts[i + 2].rstrip("\n") + "\n" for i in range(0, len(parts), 3)}
assert len(blocks) == 5, list(blocks)
css = re.search(r"<style>(.*?)</style>", (HERE / "build_unify_page.py").read_text(), re.S).group(1)


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
<title>Split loss and micro-batch cells</title>
<style>CSS
  .swap { display: grid; gap: 4px; margin-top: 8px; font: 12.5px ui-monospace, Menlo, Consolas, monospace; }
  .swap span { padding: 5px 9px; border-radius: 8px; background: #0b1018; border: 1px solid rgba(255,255,255,0.1); overflow-x: auto; white-space: pre; }
  .swap .old { color: #e66767; }
  .swap .new { color: hsl(150 60% 55%); }
  .kind { font-size: 11px; font-weight: 600; padding: 1px 8px; border-radius: 999px; border: 1px solid rgba(255,255,255,0.25); color: #a7b0c0; margin-left: 6px; text-transform: none; letter-spacing: 0; }
</style>
</head>
<body>
<main>
<h1>Split loss and micro-batch overfit cells</h1>
<p class="lead">Notebook: <code>ImageNet-Unify.ipynb</code>. Three one-line fixes to the model first, then five cells in notebook order.</p>

<h2>First: three fixes, or no step can run</h2>
<p class="small">The compile cell does not run the model, so these have not shown up yet. The first real call fails on each of them.</p>
<ol class="steps">
  <li><span class="where">Cell <code>#pre-post norm adaLN zero transformer+MLP block</code></span>
    <span class="do"><code>Block.forward</code> needs the RoPE tables: the core calls <code>block(x, rope, time_emb)</code>.</span>
    <div class="swap"><span class="old">    def forward(self, x, time_emb):</span><span class="new">    def forward(self, x, rope, time_emb):</span></div>
  </li>
  <li><span class="where">Same cell, the attention line</span>
    <span class="do">Pass <code>rope</code> on to <code>self.MHA</code>, which takes <code>(x, rope)</code>.</span>
    <div class="swap"><span class="old">self.MHA( self.pre_norm1(x) * (1 + scale_MHA) + shift_MHA )</span><span class="new">self.MHA( self.pre_norm1(x) * (1 + scale_MHA) + shift_MHA, rope )</span></div>
  </li>
  <li><span class="where">Cell <code>#define encoder model</code></span>
    <span class="do"><code>ImageTransformer.forward</code> uses <code>time</code> but does not take it. The train step calls <code>model(noised_images, t)</code>.</span>
    <div class="swap"><span class="old">    def forward(self, images):</span><span class="new">    def forward(self, images, time):</span></div>
  </li>
</ol>
<p class="small">Also: the cell <code>#convenient globals and checks</code> reads <code>train_dataloader</code> before the cell that makes it, and uses
<code>ae_emb_dim</code>, which no longer exists. <code>init_models()</code> needs <code>c</code> from that cell. Move it below the loader cell and
delete its two <code>ae_emb_dim</code> lines.</p>

<h2>Then: the five cells</h2>
<ol class="steps">
  <li><span class="where">Data <span class="kind">action</span> · new cell below <code>train_ds = BufferedImageNet(</code></span>
    <span class="do">The validation loader, same pattern as your train loader. <code>get_split_loss("test")</code> needs it.</span>
    COPY1
  </li>
  <li><span class="where">Helpers <span class="kind">definition</span> · new cell below <code>def laplacian(x):</code></span>
    <span class="do">Paste this. It only defines <code>x_pred_v_loss</code>, <code>get_eval_batches</code> and <code>get_split_loss</code>.</span>
    COPY2
  </li>
  <li><span class="where">Training <span class="kind">action</span> · replace the cell <code># fixed_micro_batch_size = 64</code></span>
    <span class="do">Paste this and run it after the compile cell.</span>
    COPY3
    <span class="result-label">You should see:</span>
    <div class="result">fixed_micro_batch: (6, 3, 128, 128)</div>
  </li>
  <li><span class="where">Training <span class="kind">definition</span> · replace the cell <code># ############## overfit micro batch</code></span>
    <span class="do">Paste this. It only defines the two plots and <code>overfit_micro_batch</code>.</span>
    COPY4
  </li>
  <li><span class="where">Training <span class="kind">action</span> · new cell below the helpers from the step above</span>
    <div class="caution">This trains the model on six images. Run the <code>init models</code> cell again before real training.</div>
    <span class="do">Paste this and run it. Edit <code>steps</code> here.</span>
    COPY5
    <span class="result-label">You should see the six images, then lines like these, then a grid of predictions:</span>
    <div class="result">START MICRO-BATCH | micro-loss: 0.73549
step 200 | 41 ms/step | train-step loss=2.90752
…
END MICRO-BATCH | micro-loss: 0.73549 -> 0.25168</div>
  </li>
</ol>

<h2>How to read the results</h2>
<ul class="small">
  <li><b>The prediction grid is the real check.</b> Top row: the six images. Then, for each <code>t</code>, the noised input and the model's
      clean-image prediction. When the overfit works, the prediction rows look like the top row, also at <code>t=0</code>, where the input
      is pure noise.</li>
  <li><b><code>micro-loss</code> is comparable before and after.</b> <code>get_split_loss</code> uses fixed images, fixed <code>t</code> and fixed
      noise (<code>seed=0</code>), so two calls differ only by the weights. It also runs in eval mode, so the sparse middle is off.</li>
  <li><b>The train-step loss is noisy, and that is the loss, not a fault.</b> The v-loss divides by <code>(1 - t)</code>, clamped at 0.05. A sample
      with <code>t</code> near 1 counts up to 400 times more than one near 0, so a few samples per batch decide the number.</li>
  <li><b><code>get_split_loss("train")</code> and <code>("test")</code></b> load their first 20 batches once and keep them, so later calls are
      fast and do not restart the loader workers.</li>
</ul>

<p class="small"><b>Tested</b> on the CPU with your model cells (with the three fixes), a tiny configuration and fake loaders: the split loss
repeats exactly, equals the train step's loss for the same draws, and 400 micro-batch steps cut the micro-loss from 0.74 to 0.25. Not run
on a GPU or with the compiled step.</p>

<p class="small"><a href="/unify_core.html">Unified core transformer</a> · <a href="/">Back to train-viz</a></p>
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

page = PAGE.replace("CSS", css, 1)
for token, key, label in (("COPY1", "BLOCK 1", "Copy the validation loader"), ("COPY2", "BLOCK 2", "Copy the split-loss helpers"),
                          ("COPY3", "BLOCK 3", "Copy the fix cell"), ("COPY4", "BLOCK 4", "Copy the micro-batch helpers"),
                          ("COPY5", "BLOCK 5", "Copy the overfit cell")):
    assert page.count(token) == 1, token
    page = page.replace(token, copy(key, label))
out = HERE.parent / "public" / "unify_eval.html"
out.write_text(page)
print("wrote", out, len(page), "bytes")
