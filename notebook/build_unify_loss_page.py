"""Build public/unify_loss.html: put the split losses on the scale of the train-step loss.

    python3 build_unify_loss_page.py
"""
import html
import re
from pathlib import Path

HERE = Path(__file__).parent
css = re.search(r"<style>(.*?)</style>", (HERE / "build_unify_page.py").read_text(), re.S).group(1)
text = (HERE / "unify_eval_cells.py").read_text()
code = re.search(r"^# === BLOCK 2[: ].*?===\n(.*?)(?=^# === BLOCK )", text, re.M | re.S).group(1).rstrip("\n") + "\n"
SETTINGS = "# time sampling of the train step (logit-normal)\nt_logit_mean = -0.8\nt_logit_std = 0.8\n"


def copy(code, label):
    return f"""<div class="copy">
<button>{label}</button><span class="msg" role="status"></span>
<details><summary>Show the code ({len(code.splitlines())} lines)</summary><pre>{html.escape(code)}</pre></details>
</div>"""


PAGE = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Split loss on the train-step scale</title>
<style>{css}
  .swap {{ display: grid; gap: 4px; margin-top: 8px; font: 12.5px ui-monospace, Menlo, Consolas, monospace; }}
  .swap span {{ padding: 5px 9px; border-radius: 8px; background: #0b1018; border: 1px solid rgba(255,255,255,0.1); overflow-x: auto; white-space: pre; }}
  .swap .old {{ color: #e66767; }}
  .swap .new {{ color: hsl(150 60% 55%); }}
</style>
</head>
<body>
<main>
<h1>Put train-loss and test-loss on the train-step scale</h1>
<p class="lead">Notebook: <code>ImageNet-Unify.ipynb</code>. Six steps. You do not need to restart the kernel or the model.</p>

<h2>Why they were far apart</h2>
<p class="small">The split loss drew <code>t</code> uniformly. Your train step draws it from a logit-normal that almost never gets near
<code>t = 1</code>. The loss divides by <code>(1 - t)</code>, so the few uniform samples near 1 count up to 400 times more and decide the
average. In my test, the 5% of images with the largest loss carried 55% of the split loss, and the split loss was 12 times the train-step
loss on the same model. With <code>t</code> drawn the same way, it is 0.95 times.</p>

<ol class="steps">
  <li><span class="where">First cell (hyperparameters)</span>
    <span class="do">Paste these two settings below the line <code>beta2 = 0.99</code>. Run the cell.</span>
    {copy(SETTINGS, "Copy the two settings")}
  </li>
  <li><span class="where">Cell <code>################## Training Step Methods</code></span>
    <span class="do">Change one line, so the train step uses the two settings. Run the cell.</span>
    <div class="swap"><span class="old">        ) * 0.8 - 0.8</span><span class="new">        ) * t_logit_std + t_logit_mean</span></div>
  </li>
  <li><span class="where">Cell <code>#make and compile the training steps</code></span>
    <span class="do">Run it again. The model and its weights are not touched.</span>
  </li>
  <li><span class="where">Cell <code># get split losses (definitions only)</code></span>
    <span class="do">Replace the whole cell with this. Run the cell.</span>
    {copy(code, "Copy the split-loss cell")}
  </li>
  <li><span class="where">Any cell</span>
    <span class="do">Run <code>eval_logs.clear()</code>. The old evaluations are on the old scale and would bend the chart.</span>
  </li>
  <li><span class="where">Cell <code># MAIN TRAINING LOOP</code></span>
    <span class="do">Run it. It continues from the logged step.</span>
    <span class="result-label">The START line should now show a train-loss close to the recent <code>loss</code> in the step lines:</span>
    <div class="result">START step 12,500 | train-loss: 0.36… | test-loss: 0.37… | best test 0.37… at step 12,500</div>
  </li>
</ol>

<h2>What changes</h2>
<ul class="small">
  <li><b>One source for the time distribution.</b> <code>t_logit_mean</code> and <code>t_logit_std</code> in the first cell now drive both the
      train step and the split loss, through <code>sample_train_t</code>. Change them in one place and both follow.</li>
  <li><b>The split loss still repeats exactly.</b> Fixed images, fixed seed for <code>t</code> and noise.</li>
  <li><b>It is still not the same number as a step loss.</b> It runs in eval mode with the sparse middle off, on 1,280 fixed images. A step line
      is the mean over <code>report_every</code> sparse training steps. Expect them close, not equal.</li>
  <li><b>The numbers in the result box are from my tiny CPU test,</b> only to show the scale. Yours will differ.</li>
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
out = HERE.parent / "public" / "unify_loss.html"
out.write_text(PAGE)
print("wrote", out, len(PAGE), "bytes")
