"""Build public/unify_loop.html from unify_loop_cells.py, so the page and the tested code cannot drift.

    python3 build_unify_loop_page.py
"""
import html
import re
from pathlib import Path

HERE = Path(__file__).parent
text = (HERE / "unify_loop_cells.py").read_text()
parts = re.split(r"^# === (BLOCK \d): (.*?) ===\n", text, flags=re.M)[1:]
blocks = {parts[i]: parts[i + 2].rstrip("\n") + "\n" for i in range(0, len(parts), 3)}
assert len(blocks) == 9, list(blocks)
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
<title>Looped sparse middle</title>
<style>CSS</style>
</head>
<body>
<main>
<h1>Looped sparse middle with XSA and deep supervision</h1>
<p class="lead">Notebook: <code>ImageNet-Unify.ipynb</code> (commit 5eb53a0). Ten steps, top of the notebook to the bottom.
Cells marked <b>defines</b> are safe to run again at any time. Cells marked <b>acts</b> do one thing.</p>

<h2>Steps</h2>
<ol class="steps">
  <li><span class="where">Zone 1: first cell (hyperparameters)</span>
    <span class="do">Change <code>keep_ratio = 1.0</code> to <code>keep_ratio = 0.25</code>. Give <code>RUN_NAME</code> a new name.</span>
    <span class="small">With <code>keep_ratio = 1.0</code> the sparse steps keep all tokens, so sparse and dense steps are the same.</span>
  </li>
  <li><span class="where">Zone 1: first cell, below the line <code>final_blocks = 2</code></span>
    <span class="do">Paste these settings. <code>loop_fusion = False</code> with <code>keep_ratio = 1.0</code> and <code>sparse_per_dense = 0</code> gives the variant without the skip path.</span>
    COPY1
  </li>
  <li><span class="where">Zone 3: new cell, directly below the cell <code>#define unified model</code></span>
    <span class="do">Paste this cell. It defines the looped model.</span>
    COPY2
  </li>
  <li><span class="where">Zone 3: the cell <code># --------------------Def Init Models (and optimizers)</code></span>
    <span class="do">Replace the cell with this. The only change: it builds <code>LoopedImageTransformer</code>.</span>
    COPY3
  </li>
  <li><span class="where">Zone 3: new cell, directly below the cell <code>#log metrics (definitions only)</code></span>
    <span class="do">Paste this cell. It defines the check, the loop-count evals and the chart.</span>
    COPY4
  </li>
  <li><span class="where">Zone 4: new cell, directly below the <code>init models</code> banner cell</span>
    <span class="do">Paste this one-line cell.</span>
    COPY5
  </li>
  <li><span class="where">Zone 5: new cell, directly below the cell <code>################## Training Step Methods</code></span>
    <span class="do">Paste this cell. It defines the train step with deep supervision.</span>
    COPY6
  </li>
  <li><span class="where">Zone 5: the cell <code>#make and compile the training steps</code></span>
    <span class="do">Replace the cell with this.</span>
    COPY7
  </li>
  <li><span class="where">Zone 5: the cell <code># MAIN TRAINING LOOP</code></span>
    <span class="do">Replace the cell with this.</span>
    COPY8
  </li>
  <li><span class="where">Zone 6: new cell, below the history cells</span>
    <span class="do">Paste this one-line cell. Run it after the first <code>LOOPS</code> line appears.</span>
    COPY9
  </li>
</ol>

<h2>Run it</h2>
<ol class="steps">
  <li><div class="caution">A model trained with the old cells cannot be continued. The looped model has other weights (fusion layer, XSA blocks).</div>
    <span class="do">Restart the kernel. Run every cell from the top down to the check cell of step 6.</span>
    <span class="result-label">The check cell should print (128 px, keep_ratio 0.25, 8 blocks):</span>
    <div class="result">sparse step: 4 exits | shallow blocks see 64 tokens | middle blocks see 16 tokens, 4 times each | final blocks see 64 tokens, 4 times each
dense  step: 4 exits | shallow blocks see 64 tokens | middle blocks see 64 tokens, 4 times each | final blocks see 64 tokens, 4 times each
XSA attention in blocks [2, 3, 4, 5] (middle blocks are [2, 3, 4, 5])
train: 4 loops, keep_ratio 0.25, 4 sparse steps then 1 dense step</div>
  </li>
  <li><span class="do">Run the cells down to the main training loop, then run the loop.</span>
    <span class="result-label">Every <code>eval_every</code> steps you should see one more line:</span>
    <div class="result">LOOPS step 2,000 | test-loss by loop count: 1: ... | 2: ... | 3: ... | 4: ... | 5: ... | 6: ...</div>
  </li>
  <li><span class="do">Run <code>plot_loop_losses()</code> (step 10) to chart the test loss of all six loop counts.</span></li>
</ol>

<h2>How each line of the spec is built</h2>
<ul class="small">
  <li><b>Loop:</b> <code>LoopedCoreTransformer.forward</code> runs the shallow blocks once, then the same middle blocks <code>n_loops</code> times.
      No new weights per loop, and no loop embedding.</li>
  <li><b>Sparse routing:</b> your <code>sample_indices</code> runs once, before the first loop. All loops use that token set.</li>
  <li><b>Exit path:</b> <code>exit_path</code> restores the tokens, fuses with the shallow stream, and runs the final blocks. Its result goes
      only to the output head. The loop variable <code>deep</code> never reads it.</li>
  <li><b>XSA:</b> <code>XSAMHA</code> is your <code>MHA</code> with the two XSA lines after the attention call and before the heads merge. Only the
      middle blocks use it (<code>XSABlock</code>). Pre-norm, time conditioning, QK norm, RoPE, post-norm and gate are your <code>Block</code>.</li>
  <li><b>Deep supervision:</b> the train step gets one image per loop and uses your v-loss on each:
      <code>loss = 0.5 * mean(L1, L2, L3) + 0.5 * L4</code>. The same rule in sparse and dense steps.</li>
  <li><b>S S S S D:</b> <code>is_sparse_step(step)</code> in the loop cell. One model, one optimizer, one compiled step function with a
      <code>sparse</code> argument.</li>
  <li><b>Evaluation and sampling:</b> <code>model(images, t)</code> in eval mode is dense with <code>n_loops</code> loops, so your samplers and
      <code>get_split_loss</code> work unchanged. For another loop count: <code>with use_loops(6): plot_seeds()</code>.</li>
  <li><b>Logs:</b> <code>logs["loss"]</code> is the deep-supervision loss. <code>logs["loss_final"]</code> is the loss of the last loop, the
      number to compare with a model without loops.</li>
</ul>

<h2>Three choices I made where the spec was open</h2>
<ul class="small">
  <li><b>Fusion layer in dense steps:</b> dense steps also fuse the loop state with the shallow stream (the spec says "the exact same fusion").
      So the model always has a fusion layer, also with <code>keep_ratio = 1</code>.</li>
  <li><b>Random RoPE scale:</b> your <code>rope_tables</code> draws one scale per call in training. One draw now serves all loops and all exits
      of a step.</li>
  <li><b>Cost of a step:</b> the final blocks and the output head run once per loop. A dense step runs the middle 4 times on all tokens.
      I did not measure the step time at full size.</li>
</ul>

<p class="small"><b>Tested</b> against your notebook's cells with tiny settings and a fake ImageNet, on the CPU and with the compiled step on the
Spark GPU: TEST_SUMMARY</p>

<p class="small"><a href="/unify_next.html">Unify checklist</a> · <a href="/">Back to train-viz</a></p>
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

SUMMARY = ("the parameter count is the same for 1 and 6 loops. XSA leaves no part of a head's output along its own value vector "
           "(cosine 0.29 before, below 1e-6 after) and is only in the middle blocks. Loop k gives the same prediction with or without "
           "the other exits, so no exit feeds the loop. Tokens are chosen once per pass. The loss equals the rule above. Every weight "
           "gets a gradient in a sparse and in a dense step. Evaluation with 1 to 6 loops and dense sampling work. The loop gives the "
           "S S S S D order. GPU_NOTE Not trained at full size.")
LABELS = {1: "Copy the settings", 2: "Copy the looped model (defines)", 3: "Copy the init cell (defines)", 4: "Copy the loop helpers (defines)",
          5: "Copy the check (acts)", 6: "Copy the train step (defines)", 7: "Copy the compile cell (acts)", 8: "Copy the training loop (acts)",
          9: "Copy the chart call (acts)"}
import sys
gpu_note = sys.argv[1] if len(sys.argv) > 1 else ""
page = PAGE.replace("CSS", css, 1).replace("TEST_SUMMARY", SUMMARY.replace("GPU_NOTE", gpu_note))
for n, label in LABELS.items():
    token = f"COPY{n}"
    assert page.count(token) == 1, token
    page = page.replace(token, copy(f"BLOCK {n}", label))
out = HERE.parent / "public" / "unify_loop.html"
out.write_text(page)
print("wrote", out, len(page), "bytes")
