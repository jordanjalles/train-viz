"""Build public/unify_core.html from unify_core_cells.py, so the page and the tested code cannot drift.

    python3 build_unify_page.py
"""
import html
import re
from pathlib import Path

HERE = Path(__file__).parent
text = (HERE / "unify_core_cells.py").read_text()
parts = re.split(r"^# === (BLOCK \d): (.*?) ===\n", text, flags=re.M)[1:]
blocks = {parts[i]: parts[i + 2].rstrip("\n") + "\n" for i in range(0, len(parts), 3)}
assert len(blocks) == 2, list(blocks)


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
<title>Unified core transformer</title>
<style>
  body { margin: 0; padding: 20px 16px 48px; background: #090c12; color: #e8ecf3; font: 16px/1.55 system-ui, sans-serif; }
  main { max-width: 760px; margin: 0 auto; }
  h1 { font-size: 22px; margin: 0 0 4px; }
  h2 { font-size: 16px; margin: 34px 0 6px; }
  p, td, th, li { color: #a7b0c0; }
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
  .result { margin-top: 10px; padding: 10px 12px; background: #0b1018; border-left: 3px solid hsl(150 60% 55%); border-radius: 6px; overflow-x: auto; white-space: pre; font: 12.5px/1.5 ui-monospace, Menlo, Consolas, monospace; color: #e8ecf3; }
  .caution { margin-top: 10px; padding: 10px 12px; border-left: 3px solid #e3a13a; background: rgba(227,161,58,0.08); border-radius: 6px; color: #e8ecf3; font-size: 14.5px; }
  .result-label { display: block; margin-top: 10px; font-size: 13px; color: #6b7482; }
  table { border-collapse: collapse; width: 100%; margin: 4px 0; font-size: 15px; }
  th, td { text-align: left; padding: 8px 12px 8px 0; border-bottom: 1px solid rgba(255,255,255,0.09); vertical-align: top; }
  th { color: #e8ecf3; font-weight: 600; }
  a { color: hsl(192 86% 64%); }
  .small { font-size: 14.5px; }
</style>
</head>
<body>
<main>
<h1>Unified core transformer</h1>
<p class="lead">Notebook: <code>ImageNet-Unify.ipynb</code>. One <code>CoreTransformer</code> for dense and sparse middle. It reads no
notebook globals and keeps no RoPE cache. It uses your <code>Block(x, rope)</code> as it is.</p>

<ol class="steps">
  <li><span class="where">Definitions: the cell <code>#define core transformer model</code></span>
    <span class="do">Replace that whole cell with this. Then delete the cell below it, <code>class SparseMiddleTransformer(CoreTransformer)</code>.</span>
    COPY1
  </li>
  <li><span class="where">New cell, directly below the core transformer cell</span>
    <div class="caution">Do the fix in "Before the check can run" first. Running the check before it stops with a <code>TypeError</code> in <code>Block</code>.</div>
    <span class="do">Paste this and run it once.</span>
    COPY2
    <span class="result-label">With your settings (8 blocks) you should see:</span>
    <div class="result">training step, tokens per block: [24, 6, 6, 6, 6, 6, 6, 24]
eval, tokens per block:          [24, 24, 24, 24, 24, 24, 24, 24]</div>
  </li>
</ol>

<h2>Before the check can run</h2>
<p class="small">Two things in your new cells stop any model from being built. They are in your code, not in the core class.</p>
<ul class="small">
  <li><b><code>Block</code> passes four arguments to <code>MHA</code>, and <code>MHA</code> takes three.</b> <code>Block.__init__</code> calls
      <code>MHA(n_heads, head_size, emb_dim, n_tokens)</code>. Remove <code>n_tokens</code> from that call, or add it to <code>MHA.__init__</code>.</li>
  <li><b><code>MHA.forward</code> does not use <code>out_proj</code>.</b> It returns the heads flattened. That only has the right width when
      <code>n_heads * head_size == emb_dim</code>, and <code>out_proj</code> then trains nothing. Either return
      <code>self.out_proj(out)</code> or delete the layer.</li>
</ul>

<h2>How to use it</h2>
<table>
  <tr><th>You want</th><th>Build it with</th></tr>
  <tr><td>Plain dense transformer</td><td><code>CoreTransformer(emb_dim, num_blocks, n_heads, head_size, n_tokens)</code></td></tr>
  <tr><td>Sparse middle</td><td>the same, plus <code>keep_ratio=0.25</code></td></tr>
  <tr><td>More dense blocks at the ends</td><td><code>shallow_blocks=2, final_blocks=2</code></td></tr>
  <tr><td>Full-token training on a sparse model</td><td>set <code>model.core_transformer.sparse_middle = False</code></td></tr>
</table>
<p class="small">The call is unchanged: <code>core(x, patch_rows, patch_cols, keep_indices=None)</code>. Tokens are latent tokens first, patch
tokens last. The patch count is <code>patch_rows * patch_cols</code>, and everything before it counts as latent. A model with no patch
tokens passes <code>0, 0</code>.</p>

<h2>What is the same as your two classes</h2>
<ul class="small">
  <li><b>Dense</b> (<code>keep_ratio=1</code>): same output as your <code>CoreTransformer</code> for the same weights, and the same weight names.
      There is no fusion layer in this case.</li>
  <li><b>Sparse</b>: same output as your <code>SparseMiddleTransformer</code> for the same weights and the same random draws, including
      input masking with <code>keep_indices</code>. Same names: <code>keep_ratio</code>, <code>shallow_blocks</code>, <code>final_blocks</code>,
      <code>sparse_middle</code>, <code>fusion</code>.</li>
</ul>

<h2>What is different</h2>
<ul class="small">
  <li><b>No RoPE cache.</b> <code>rope_tables(patch_rows, patch_cols)</code> builds the tables and returns them. The four cache buffers and
      <code>rope_cache_grid</code> are gone. A compiled step at changing resolutions then needs one compiled version per grid. With the
      cache, your old class needed more: this is what slowed the multi-resolution run.</li>
  <li><b>No globals.</b> The class uses only its arguments, <code>Block</code>, <code>torch</code>, <code>nn</code> and <code>math</code>.</li>
  <li><b>One class.</b> The constructor argument <code>n_latent_tokens</code> is now <code>n_tokens</code>, as in your new <code>Block</code>.</li>
</ul>

<h2>Still to do in the models that use it</h2>
<p class="small">The core is free of globals, but two callers in the file are not yet: <code>Decoder.forward</code> reads <code>patch_rows</code> and
<code>patch_cols</code> from the notebook, and <code>make_mask</code> reads <code>pr</code> and <code>pc</code>. Pass them in as arguments when
you rewrite those.</p>

<p class="small"><b>Tested</b> on the CPU against the classes in <code>ImageNet-AE-selfteach.ipynb</code>, which have the same
<code>Block(x, rope)</code> interface. Your new <code>Block</code> in the Unify file could not be built yet, for the reason above.</p>

<p class="small"><a href="/sprint.html">Sparse middle for the flow notebook</a> · <a href="/">Back to train-viz</a></p>
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
for token, key, label in (("COPY1", "BLOCK 1", "Copy the core transformer cell"), ("COPY2", "BLOCK 2", "Copy the check cell")):
    assert page.count(token) == 1, token
    page = page.replace(token, copy(key, label))
out = HERE.parent / "public" / "unify_core.html"
out.write_text(page)
print("wrote", out, len(page), "bytes")
