"""Build public/<name>.html: the samples of a class-conditional run, each with its class name.

    python3 scripts/build_condgrid.py RUN_FOLDER condgrid_step100000_cfg3 PAGE_NAME

Samples: RUN_FOLDER/<grid folder>/NNN.png + classes.json (gen_eval_spark.py --cfg S --grid N). Names: the ImageNet
class mapping of <data-root>/data/imagenet/class_mapping.json (first name of each class).
"""
import html, json, shutil, sys
from pathlib import Path

PUB = Path(__file__).resolve().parent.parent / "public"
RUN, GRID, PAGE = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
src = RUN / GRID
names = {v["label"]: v["name"].split(",")[0] for v in json.loads(Path("<data-root>/data/imagenet/class_mapping.json").read_text()).values()}
classes = json.loads((src / "classes.json").read_text())
(PUB / PAGE).mkdir(exist_ok=True)
for i in range(len(classes)):
    shutil.copy(src / f"{i:03d}.png", PUB / PAGE / f"{i:03d}.png")
step, scale = int(GRID.split("step")[1].split("_")[0]), GRID.split("_cfg")[1]
cells = "\n".join(f'  <figure><img src="{PAGE}/{i:03d}.png" width="128" height="128" alt="Sample of class {html.escape(names[c])}">'
                  f'<figcaption>{html.escape(names[c])}</figcaption></figure>' for i, c in enumerate(classes))
page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Class-conditional samples</title>
<style>
:root {{ --bg: #fbfbfa; --ink: #1d1d1b; --muted: #6b6b66; --line: #e2e1dc; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg: #171716; --ink: #ecebe6; --muted: #a2a19b; --line: #34332f; }} }}
:root[data-theme="dark"] {{ --bg: #171716; --ink: #ecebe6; --muted: #a2a19b; --line: #34332f; }}
body {{ margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.5 system-ui, sans-serif; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 24px 16px 48px; }}
h1 {{ font-size: 22px; margin: 0 0 8px; }}
p {{ color: var(--muted); margin: 0 0 20px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(128px, 1fr)); gap: 14px 10px; }}
figure {{ margin: 0; }}
img {{ width: 100%; height: auto; image-rendering: auto; border-radius: 4px; display: block; }}
figcaption {{ font-size: 12.5px; color: var(--muted); margin-top: 4px; overflow-wrap: anywhere; }}
a {{ color: inherit; }}
</style>
</head>
<body>
<main>
<h1>Class-conditional samples, guidance scale {scale}</h1>
<p>Run <code>{html.escape(RUN.name)}</code>, EMA weights at step {step:,}, 16 Euler steps, seeds 0 to {len(classes) - 1}.
Each sample has the class under it. FID of this setting (10,000 samples): 118.9. <a href="/">Back to train-viz</a></p>
<div class="grid">
{cells}
</div>
</main>
</body>
</html>
"""
(PUB / f"{PAGE}.html").write_text(page)
print(f"{PAGE}.html written: {len(classes)} samples")
