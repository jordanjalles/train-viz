"""Build public/unify_patch.html from unify_patch_cells.py, so the page and the tested code cannot drift.

    python3 build_unify_patch_page.py
"""
import html
import re
from pathlib import Path

HERE = Path(__file__).parent
text = (HERE / "unify_patch_cells.py").read_text()
parts = re.split(r"^# === (BLOCK \d): (.*?) ===\n", text, flags=re.M)[1:]
blocks = {parts[i]: parts[i + 2].rstrip("\n") + "\n" for i in range(0, len(parts), 3)}
assert len(blocks) == 2, list(blocks)
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
<title>Grey samples: cause and fix</title>
<style>CSS</style>
</head>
<body>
<main>
<h1>Grey samples: the patch skip drops color</h1>
<p class="lead">Notebook: <code>ImageNet-Unify.ipynb</code>. The loss is fine. The fixed skip path in the patch embed and un-embed
carries brightness only, and the part that can carry color starts 10,000 times weaker.</p>

<h2>Why</h2>
<ul class="small">
  <li><b>The numbers line up by accident.</b> A 16 px patch has 768 values, ordered pixel by pixel as R, G, B. Your
      <code>emb_dim</code> is 256. <code>ChannelGroupResize</code> groups 768 / 256 = 3 consecutive values, which is exactly the
      R, G and B of one pixel.</li>
  <li><b>Embed skip:</b> it averages each group, so each token channel is the grey value of one pixel. A color image and its grey
      version give the same tokens.</li>
  <li><b>Un-embed skip:</b> it repeats each token channel over its group, so R = G = B in every output pixel.</li>
  <li><b>Color can only come from the learned linear branches.</b> They sit behind an RMSNorm with gain <code>1e-4</code>. At init the color
      part of the un-embed output is about 0.00007, against 0.8 for the grey part. With <code>lr = 1e-4</code> that gain grows slowly.</li>
  <li><b>The AE notebook did not show this</b> because its width was 336. 768 / 336 is not a whole number, so the groups did not line up
      with the three channels.</li>
</ul>

<h2>Fix</h2>
<ol class="steps">
  <li><span class="where">Definitions: the three cells <code>class ChannelGroupResize</code>, <code>class LinSkipPatchEmbed</code>, <code>class LinSkipUnEmbed</code></span>
    <span class="do">Replace the first of the three cells with this. Delete the other two.</span>
    COPY1
  </li>
  <li><span class="where">New cell, directly below the patch cell</span>
    <span class="do">Paste this and run it once.</span>
    COPY2
    <span class="result-label">You should see:</span>
    <div class="result">red   in -> out RGB [1.0, 0.0, 0.0]
green in -> out RGB [0.0, 1.0, 0.0]
blue  in -> out RGB [0.0, 0.0, 1.0]</div>
  </li>
  <li><span class="where">Then</span>
    <div class="caution">A model trained with the old patch cells cannot be continued. Its weights were learned around a grey skip.</div>
    <span class="do">Run the cells from the patch cell down to <code>init models</code> again, run <code>clear_logs()</code>, and start training.</span>
  </li>
</ol>

<h2>What the new skip does</h2>
<p class="small">Each color channel of a patch is average-pooled from 16 × 16 to 8 × 8, in 2 × 2 blocks. That gives 3 × 64 = 192 values, which fill
the first 192 token channels. The last 64 are zero in the skip. The un-embed does the reverse: each value is repeated over its 2 × 2 block. So the
default path through the model is a slightly blurred color image, not a sharp grey one. The learned linear branches are unchanged and add the
detail.</p>
<ul class="small">
  <li><b>The pooled size always divides the patch size.</b> It is the largest such size that fits: 8 × 8 for <code>emb_dim</code> 192 to 767, and the
      full 16 × 16 for 768 or more. Every pixel is then in exactly one block of the same size, so the skip treats all positions alike.
      (My first version used 9 × 9 for width 256. Its blocks overlapped unevenly, so I changed it.)</li>
  <li><b>Same class names and calls.</b> <code>LinSkipPatchEmbed</code> and <code>LinSkipUnEmbed</code> keep their arguments, so the model cell
      does not change. <code>ChannelGroupResize</code> is gone.</li>
</ul>

<p class="small"><b>Tested</b> on the CPU against your notebook's cells. Old cells: an image and its grey version give identical tokens, and the
un-embed skip output has R = G = B exactly. New cells: constant colors come back exact, a smooth color ramp comes back within 0.03, your model
builds with them and trains. Not run on a GPU.</p>

<p class="small"><a href="/unify_train.html">Training loop and seed plots</a> · <a href="/">Back to train-viz</a></p>
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
for token, key, label in (("COPY1", "BLOCK 1", "Copy the patch cell"), ("COPY2", "BLOCK 2", "Copy the color check")):
    assert page.count(token) == 1, token
    page = page.replace(token, copy(key, label))
out = HERE.parent / "public" / "unify_patch.html"
out.write_text(page)
print("wrote", out, len(page), "bytes")
