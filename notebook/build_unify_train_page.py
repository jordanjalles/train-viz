"""Build public/unify_train.html from unify_train_cells.py, so the page and the tested code cannot drift.

    python3 build_unify_train_page.py
"""
import html
import re
from pathlib import Path

HERE = Path(__file__).parent
text = (HERE / "unify_train_cells.py").read_text()
parts = re.split(r"^# === BLOCK (\w+): (.*?) \| (.*?) \| (.*?) ===\n", text, flags=re.M)[1:]
blocks = [dict(id=parts[i], title=parts[i + 1], kind=parts[i + 2], where=parts[i + 3], code=parts[i + 4].rstrip("\n") + "\n")
          for i in range(0, len(parts), 5)]
assert [b["id"] for b in blocks] == ["1", "2", "3", "4", "5", "6A", "6B", "6C", "6D", "6E", "6F", "6G"]
css = re.search(r"<style>(.*?)</style>", (HERE / "build_unify_page.py").read_text(), re.S).group(1)
ZONE = {"1": "Helpers", "2": "Helpers", "3": "Training", "4": "Training", "5": "After training"}


def linkify(where):
    return re.sub(r'"([^"]+)"', lambda m: f"<code>{html.escape(m.group(1))}</code>", html.escape(where, quote=False))


def card(b, extra=""):
    n = len(b["code"].splitlines())
    zone = ZONE.get(b["id"], "After training")
    return f"""  <li><span class="where">{zone} <span class="kind">{b['kind']}</span> · {linkify(b['where'])}</span>
    <span class="do">{html.escape(b['title'][0].upper() + b['title'][1:])}.</span>{extra}
    <div class="copy">
<button>Copy block {b['id']}</button><span class="msg" role="status"></span>
<details><summary>Show the code ({n} lines)</summary><pre>{html.escape(b['code'])}</pre></details>
</div>
  </li>"""


LOOP_RESULT = """
    <span class="result-label">You should see lines like these (numbers will differ):</span>
    <div class="result">START step 0 | train-loss: 8.64749 | test-loss: 6.00390 | best test 6.00390 at step 0
step 500 | epoch 1 | 142 ms/step | 451.0 img/s | 47.4 min/epoch | loss 1.23456 | lr 1.0e-04 | 0:01:11
EVAL step 2,000 | train-loss: 1.10321 | test-loss: 1.11876 | best test 1.11876 at step 2,000</div>"""

main_cards = "\n".join(card(b, LOOP_RESULT if b["id"] == "4" else "") for b in blocks[:5])
plot_cards = "\n".join(card(b) for b in blocks[5:])

PAGE = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Training loop and seed plots</title>
<style>{css}
  .kind {{ font-size: 11px; font-weight: 600; padding: 1px 8px; border-radius: 999px; border: 1px solid rgba(255,255,255,0.25); color: #a7b0c0; margin-left: 6px; text-transform: none; letter-spacing: 0; }}
</style>
</head>
<body>
<main>
<h1>Training loop, readouts and seed plots</h1>
<p class="lead">Notebook: <code>ImageNet-Unify.ipynb</code>. Five cells for training, then seven one-line plot cells, in notebook order.</p>

<h2>Before you start</h2>
<p class="small">Your notebook has no <code>test_dataloader</code> yet. The loop runs without it and prints <code>test-loss: n/a</code>. To get a test
loss, paste the validation loader from <a href="/unify_eval.html">the split-loss page</a> (its first block) below your train loader cell.</p>

<h2>Training cells</h2>
<ol class="steps">
{main_cards}
</ol>

<h2>Plot cells</h2>
<p class="small">Each block is one cell. They need a model that has trained for a while to show anything but noise.</p>
<ol class="steps">
{plot_cards}
</ol>

<h2>What the readouts mean</h2>
<table>
  <tr><th>Line</th><th>When</th><th>What it shows</th></tr>
  <tr><td><code>step …</code></td><td>every <code>report_every</code> steps</td><td>Speed over those steps, the mean train-step loss over them, the learning rate and the time since the loop started.</td></tr>
  <tr><td><code>EVAL …</code></td><td>every <code>eval_every</code> steps</td><td><code>get_split_loss</code> on fixed images with fixed <code>t</code> and noise. The best test loss so far and its step.</td></tr>
  <tr><td>image grid</td><td>every <code>preview_every</code> steps</td><td>Samples from seeds 0 to 5. The seeds are fixed, so one grid compares with the next.</td></tr>
  <tr><td><code>EPOCH …</code></td><td>end of each epoch</td><td>The same evaluation as <code>EVAL</code>.</td></tr>
</table>

<h2>Good to know</h2>
<ul class="small">
  <li><b>Sampling direction.</b> In this notebook <code>t=0</code> is noise and <code>t=1</code> is the image, the reverse of the latent flow
      notebook. The samplers go from 0 to 1. The model predicts the clean image, so the last step lands exactly on its prediction.</li>
  <li><b>Sampling is always dense.</b> The samplers put the model in eval mode, so the sparse middle is off.</li>
  <li><b>Other resolutions.</b> <code>plot_seeds(image_size=256)</code> samples at another size, since the model has no fixed grid.</li>
  <li><b>The step count continues.</b> Stopping the loop and running the cell again goes on from the logged steps. For a new run after
      <code>init_models()</code>, run <code>clear_logs()</code> first, or the new model's steps add to the old ones.</li>
  <li><b>The split loss uses uniform <code>t</code>.</b> Your train step now draws <code>t</code> from a logit-normal. So <code>train-loss</code> and
      the step loss are not the same number, also on the same images. The split loss is a fixed benchmark over all noise levels.</li>
  <li><b>The log cell is safe to run again.</b> It keeps the logs if they exist.</li>
</ul>

<p class="small"><b>Tested</b> on the CPU with your model, train step and split-loss cells, a tiny configuration and fake loaders. A model that
always predicts one image is sampled to exactly that image by both samplers. The same seed gives the same image, alone or in a batch. The loop
logs every step and prints each kind of line. It has not run on a GPU or with the compiled step.</p>

<p class="small"><a href="/unify_eval.html">Split loss and micro-batch cells</a> · <a href="/unify_core.html">Unified core transformer</a> · <a href="/">Back to train-viz</a></p>
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

out = HERE.parent / "public" / "unify_train.html"
out.write_text(PAGE)
print("wrote", out, len(PAGE), "bytes,", len(blocks), "blocks")
