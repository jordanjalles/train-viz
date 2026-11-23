"""Build public/flow_resume.html from flow_resume_cells.py.

    python3 make_flow_resume_page.py

The code blocks on the page are read from flow_resume_cells.py, so
the page and the tested code cannot drift. The tests import
parse_blocks, apply_edit and PLACE from here.
"""
import html
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELLS = HERE / "flow_resume_cells.py"
PAGE = HERE.parent / "public" / "flow_resume.html"
MAX_LINE = 72


def parse_blocks(text=None):
    """-> {id: {title, mode, find, code}} in file order."""
    text = CELLS.read_text() if text is None else text
    blocks, cur = {}, None
    for line in text.splitlines():
        m = re.match(r"# === BLOCK (\w+): (.*) ===$", line)
        if m:
            cur = {"title": m.group(2), "mode": "", "find": "",
                   "lines": []}
            blocks[m.group(1)] = cur
        elif cur is None:
            continue
        elif line.startswith("# MODE: ") and not cur["lines"]:
            cur["mode"] = line[8:].strip()
        elif line.startswith("# FIND: ") and not cur["lines"]:
            cur["find"] = line[8:].strip()
        else:
            cur["lines"].append(line)
    for b in blocks.values():
        lines = b.pop("lines")
        while lines and not lines[-1].strip():
            lines.pop()
        b["code"] = "\n".join(lines) + "\n"
    return blocks


def apply_edit(cell, block):
    """Apply one edit block to the text of a notebook cell."""
    mode, find, code = block["mode"], block["find"], block["code"]
    if mode == "append":
        return cell.rstrip("\n") + "\n\n" + code
    lines = cell.split("\n")
    hits = [i for i, l in enumerate(lines) if l.strip() == find]
    if len(hits) != 1:
        raise ValueError(f"{find!r} matches {len(hits)} lines")
    i = hits[0]
    new = code.rstrip("\n").split("\n")
    if mode == "replace-line":
        lines[i:i + 1] = new
    elif mode == "insert-above":
        lines[i:i] = new + [""]
    else:
        raise ValueError(f"mode {mode!r} is not an edit")
    return "\n".join(lines)


# block id -> (kind, zone of the notebook, where it goes). HTML.
PLACE = {
    "1": ("settings", "Top: first cell",
          "Add at the end of the first cell (the hyperparameters)."),
    "2": ("action", "Data cells",
          "New cell directly below the cell that starts with "
          "<code>train_ds = BufferedImageNet(</code>."),
    "3": ("definition", "Helpers",
          "New cell directly below the cell that starts with "
          "<code>#log metrics</code>."),
    "4A": ("action", "After init models",
           "Replaces the whole cell that starts with <code># "
           "---------- Cache encoder outputs for latent-flow "
           "training</code>."),
    "4B": ("action", "After init models",
           "Replaces the whole cell that starts with <code># "
           "---------- Cache encoder VALIDATION outputs</code>."),
    "5A": ("edit", "Training loop cell",
           "Replaces the line "
           "<code>for epoch in range(num_epochs):</code>."),
    "5B": ("edit", "Training loop cell",
           "New line above <code>report_steps += 1</code>. Keep the "
           "eight spaces."),
    "5C": ("edit", "Training loop cell",
           "Last line of the cell. Keep the four spaces."),
    "6A": ("marker", "Bottom",
           "New cell directly above the cell that starts with "
           "<code># load flow model</code>."),
    "6B": ("action, overwrites", "Bottom, below the STOP line",
           "New cell below the STOP line."),
    "6C": ("action, overwrites", "Bottom, below the STOP line",
           "New cell below the resume cell."),
    "6D": ("action, overwrites", "Bottom, below the STOP line",
           "New cell below the save cell. It is commented out."),
    "E1": ("definition", "Another notebook",
           "New cell in the other notebook."),
    "E2": ("action", "Another notebook",
           "New cell below E1."),
}

STYLE = """
  body { margin: 0; padding: 20px 16px 40px; background: #090c12;
    color: #e8ecf3; font: 15px/1.5 system-ui, sans-serif; }
  main { max-width: 860px; margin: 0 auto; }
  h1 { font-size: 20px; margin: 0 0 6px; }
  h2 { font-size: 16px; margin: 28px 0 4px; }
  h3 { font-size: 14px; margin: 18px 0 2px; color: #e8ecf3; }
  p, li, td { color: #a7b0c0; }
  li { margin: 4px 0; }
  b { color: #e8ecf3; }
  code { color: #e8ecf3; }
  button { font: inherit; font-weight: 600; padding: 10px 20px;
    border-radius: 999px; border: 0; background: hsl(192 86% 64%);
    color: #06222a; cursor: pointer; min-height: 44px; }
  .msg { margin-left: 12px; color: hsl(150 60% 55%); }
  pre { margin-top: 12px; padding: 14px; background: #0f141d;
    border: 1px solid rgba(255,255,255,0.13); border-radius: 12px;
    overflow-x: auto;
    font: 12.5px/1.45 ui-monospace, Menlo, Consolas, monospace; }
  .result { color: #7f8a9c; }
  .caution { margin: 6px 0; padding: 8px 10px; border-radius: 6px;
    border-left: 3px solid #e3a13a;
    background: rgba(227,161,58,0.08); color: #e8ecf3; }
  a { color: hsl(192 86% 64%); }
  .wrap { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; font-size: 14px;
    margin: 8px 0; }
  th, td { text-align: left; padding: 6px 12px 6px 0;
    border-bottom: 1px solid rgba(255,255,255,0.09);
    vertical-align: top; }
  th { color: #e8ecf3; font-weight: 600; }
  .kind { display: inline-block; padding: 1px 8px; font-size: 12px;
    border-radius: 999px; border: 1px solid rgba(255,255,255,0.2);
    color: #e8ecf3; white-space: nowrap; }
  .kind.danger { border-color: #e3a13a; }
"""

SCRIPT = """
for (const section of document.querySelectorAll('section')) {
  const button = section.querySelector('button');
  const msg = section.querySelector('.msg');
  const pre = section.querySelector('pre');
  button.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(pre.textContent);
      msg.textContent = 'Copied';
    } catch {
      const range = document.createRange();
      range.selectNodeContents(pre);
      const sel = getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
      msg.textContent = 'Selected: press Ctrl+C (or Cmd+C) to copy';
    }
    setTimeout(() => { msg.textContent = ''; }, 4000);
  });
}
"""

HEAD = """
<h1>Save, load and resume for ImageNet1-Latent-Flow-Matching-XM</h1>
<p>After these edits the notebook encodes the latents one time and
loads them from disk on every later start. It saves the flow model,
the optimizer and the loss history at the end of every epoch, and
every 30 minutes inside an epoch.</p>
<p>The blocks follow the notebook from top to bottom. A
<b>definition</b> cell only defines functions: it is safe to run
again at any time. An <b>action</b> cell is short and does one
thing. Every cell that overwrites the model or a file is at the
bottom of the notebook, below a STOP line.</p>

<h2>Blocks and where they go</h2>
<div class="wrap"><table>
<tr><th>Block</th><th>Kind</th><th>Part of the notebook</th>
<th>Where it goes</th></tr>
TABLE
</table></div>
"""

STEPS = """
<h2>Set up (one time)</h2>
<p>Open
<code>MaskedAutoencoders/ImageNet1-Latent-Flow-Matching-XM.ipynb</code>.
Work from the top of the notebook to the bottom.</p>

<h3>Top: first cell</h3>
<ol>
  <li>Paste block 1 at the end of the first cell.</li>
  <li>Set <code>RUN_NAME</code> to a name for this training run.</li>
</ol>

<h3>Data cells</h3>
<ol start="3">
  <li>Add a new cell directly below the cell that starts with
  <code>train_ds = BufferedImageNet(</code>.</li>
  <li>Paste block 2 into the new cell.</li>
</ol>

<h3>Helpers</h3>
<ol start="5">
  <li>Add a new cell directly below the cell that starts with
  <code>#log metrics</code>.</li>
  <li>Paste block 3 into the new cell. You can collapse this cell.</li>
</ol>

<h3>Cache cells (below the init-models cell)</h3>
<ol start="7">
  <li>Replace the whole cell that starts with
  <code># ---------- Cache encoder outputs for latent-flow
  training</code> with block 4A.</li>
  <li>Replace the whole cell that starts with
  <code># ---------- Cache encoder VALIDATION outputs</code> with
  block 4B.</li>
</ol>

<h3>Training loop cell</h3>
<ol start="9">
  <li>Go to the cell that starts with
  <code># MAIN TRAINING LOOP</code>.</li>
  <li>Replace the line <code>for epoch in range(num_epochs):</code>
  with block 5A.</li>
  <li>Paste block 5B on a new line above
  <code>report_steps += 1</code>. Keep the eight spaces.</li>
  <li>Paste block 5C as the last line of the cell. Keep the four
  spaces.</li>
</ol>

<h3>Bottom of the notebook</h3>
<ol start="13">
  <li>Go to the cell that starts with
  <code># load flow model</code>.</li>
  <li>Add four new cells directly above it.</li>
  <li>Paste blocks 6A, 6B, 6C and 6D into them, in that order.</li>
</ol>

<h3>First run</h3>
<ol start="16">
  <li>Restart the kernel.</li>
  <li>Run the cells in order, down to and including the cell that
  starts with <code># MAIN TRAINING LOOP</code>.
  <div class="result">Result, this first time only: each cache cell
  encodes as before and ends with
  <code>z cache [train]: saved 1,281,167 to …</code>. Each epoch
  ends with <code>saved &lt;RUN_NAME&gt; | epochs done N</code>.</div></li>
</ol>
<div class="caution">Do not use "Run All". It also runs the cells
below the STOP line.</div>

<h2>Resume after a kernel restart</h2>
<ol>
  <li>Run the cells in order, from the top down to and including the
  cell that starts with <code>#compile and speed</code>.
  <div class="result">Result: each cache cell prints
  <code>z cache [train]: loaded 1,281,167 latents from …</code>
  in seconds. If it prints <code>rejected, encoding</code>, the
  lines below it name what changed.</div></li>
  <li>Scroll to the bottom of the notebook, below the STOP line.</li>
  <li>Run the cell that starts with <code># RESUME A RUN</code>.
  <div class="result">Result:
  <code>resumed &lt;RUN_NAME&gt; … epochs done: N | steps logged: …
  | lr: … | optimizer restored</code>.</div></li>
  <li>Scroll back up to the cell that starts with
  <code># MAIN TRAINING LOOP</code> and run it.
  <div class="result">Result: the first <code>step</code> line shows
  a step count above N × batches per epoch, and the next epoch line
  is <code>EPOCH N+1</code>.</div></li>
</ol>
<p>A new run needs no step at the bottom. With no resume, the loop
starts at epoch 0.</p>

<h2>Only the latents, in a new notebook</h2>
<ol>
  <li>Run the set-up above one time, so the two cache files exist.</li>
  <li>In the new notebook, paste block E1 into a cell and run it.
  <div class="result">Result: no output. The cell only defines
  <code>load_z_cache</code> and <code>latent_loader</code>. You can
  collapse it and leave it.</div></li>
  <li>Paste block E2 into the next cell.</li>
  <li>If the new notebook uses another autoencoder, change
  <code>AE_NAME</code> in block E2.</li>
  <li>Run the E2 cell.
  <div class="result">Result:
  <code>z cache [train]: (1281167, 48, 48) torch.bfloat16 from …</code>
  and the same line for <code>val</code>.</div></li>
</ol>
<p><b>E1 defines, E2 acts.</b> The two cells need no encoder, no
ImageNet loader and no flow model. They only read
<code>$ML_CHECKPOINTS</code>. They do not check that the file matches
an encoder, because they do not load one.</p>

<h2>Notes</h2>
<ul>
  <li>Latents: <code>$ML_CHECKPOINTS/z_cache/&lt;ae_name&gt;_128px/train.pt</code>
  (about 5.5 GiB) and <code>val.pt</code>. A cache is used only when
  <code>ae_name</code>, the image size, the image count, the token
  dimensions and the file <code>encoder.pt2</code> (size and
  modified time) are the same as when it was written.</li>
  <li>Runs: <code>$ML_CHECKPOINTS/flow/&lt;RUN_NAME&gt;/latest.pt</code>,
  plus <code>epoch_010.pt</code>, <code>epoch_020.pt</code>, … as
  kept copies.</li>
  <li>The resume cell takes arguments: <code>load_run("other_run")</code>
  loads another run, and <code>load_run(tag="epoch_010")</code> loads
  a kept copy.</li>
  <li><code>load_run()</code> restores the learning rate that was
  saved. The <code>#drop lr</code> cell changes it again if you run
  that cell after the resume cell.</li>
  <li>A save inside an epoch records the epoch as not done. A resume
  from it repeats that epoch from its first batch, with the newer
  weights. The loss history keeps the steps already logged.</li>
  <li>If you interrupt the loop and run the loop cell again without
  a restart, it continues at the first epoch that is not done.</li>
  <li>The helpers cell (block 3) keeps no state. The epoch count and
  the autosave clock live on <code>flow_model</code>, so running the
  helpers cell again changes nothing. A new
  <code>init_models()</code> starts both from zero.</li>
  <li>The validation cache cell no longer builds
  <code>test_dataloader</code>. Block 2 builds it in the data cells,
  so the later probe cells that read it still work.</li>
  <li>The loss history is stored as float32.</li>
  <li>The three old cells at the bottom (<code>name = …</code>,
  <code># load flow model</code>, <code># save flow model</code>)
  can stay. The new helpers do not use them. Their
  <code>flow_model</code> key is the same, so the old load cell can
  also read <code>latest.pt</code>.</li>
  <li>A model with new parameters still loads: the new parameters
  keep their initial values, and the printout names them. The
  optimizer state is then not restored.</li>
  <li>The STOP line in block 6A is shorter than the one in the AE
  notebook, to fit 72 characters.</li>
</ul>
"""


def kind_tag(kind):
    cls = "kind danger" if "overwrites" in kind else "kind"
    return f'<span class="{cls}">{kind}</span>'


def build():
    blocks = parse_blocks()
    assert list(blocks) == list(PLACE), (list(blocks), list(PLACE))
    for line in CELLS.read_text().splitlines():
        if len(line) > MAX_LINE and not line.startswith("# FIND"):
            raise SystemExit(f"over {MAX_LINE} chars: {line}")
    rows, parts = [], []
    for bid, b in blocks.items():
        kind, zone, where = PLACE[bid]
        title = html.escape(b["title"])
        rows.append(
            f'<tr><td><a href="#b{bid}">{bid}</a>: {title}</td>'
            f"<td>{kind_tag(kind)}</td><td>{zone}</td>"
            f"<td>{where}</td></tr>"
        )
        parts.append(
            f'<section id="b{bid}">\n<h2>Block {bid}: {title}</h2>\n'
            f"<p>{kind_tag(kind)} {where}</p>\n"
            f"<button>Copy block {bid}</button>"
            f'<span class="msg" role="status"></span>\n'
            f"<pre>{html.escape(b['code'])}</pre>\n</section>\n"
        )
    PAGE.write_text(
        '<!doctype html>\n<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n<meta name="viewport" '
        'content="width=device-width, initial-scale=1">\n'
        "<title>Flow resume cells</title>\n"
        f"<style>{STYLE}</style>\n</head>\n<body>\n<main>\n"
        f"{HEAD.replace('TABLE', chr(10).join(rows))}\n{STEPS}\n"
        f"{''.join(parts)}"
        '<p style="margin-top:26px"><a href="/sprint.html">Sparse '
        'middle (SPRINT) cells</a> · <a href="/">Back to train-viz'
        f"</a></p>\n</main>\n<script>{SCRIPT}</script>\n"
        "</body>\n</html>\n"
    )
    print("wrote", PAGE, "with blocks", ", ".join(blocks))


if __name__ == "__main__":
    build()
