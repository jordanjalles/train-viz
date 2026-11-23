"""How much color the user's original ChannelGroupResize skip keeps, by token width (CPU).

    python skip_color_table.py UNIFY_NOTEBOOK.ipynb

For a 16 px RGB patch (768 values, pixel-major R, G, B order): tokens of a color image against
tokens of its grey version, and the color left after embed skip -> un-embed skip.
"""
import json
import sys

import torch
import torch.nn as nn

cells = ["".join(c["source"]) for c in json.load(open(sys.argv[1]))["cells"]]
g = dict(torch=torch, nn=nn)
exec(next(c for c in cells if "class ChannelGroupResize" in c), g)
Resize = g["ChannelGroupResize"]

P, C = 16, 3
n = C * P * P
torch.manual_seed(0)
patch = torch.rand(64, P, P, C)                       # pixel-major: (i, j, c)
grey = patch.mean(-1, keepdim=True).expand_as(patch)
chroma = lambda x: (x - x.mean(-1, keepdim=True)).abs().mean().item()
print(f"patch {P} px, {n} values per patch, color in the input: {chroma(patch):.3f}")
print(f"{'emb_dim':>8} {'values per token':>17} {'group sizes':>14} {'tokens differ from grey':>24} {'color kept after down+up':>25}")
for emb in (192, 256, 288, 320, 336, 384, 512):
    down, up = Resize(n, emb), Resize(emb, n)
    sizes = sorted(set(down.sizes.int().tolist()))
    tok, tok_grey = down(patch.reshape(64, n)), down(grey.reshape(64, n))
    back = up(tok).reshape(64, P, P, C)
    print(f"{emb:>8} {n / emb:>17.2f} {str(sizes):>14} {(tok - tok_grey).abs().mean().item():>24.4f} {chroma(back) / chroma(patch):>24.0%}")
