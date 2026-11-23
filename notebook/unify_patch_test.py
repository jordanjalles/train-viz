"""CPU test: why the Unify model generates grey images, and that the new patch cells fix it.

    python unify_patch_test.py UNIFY_NOTEBOOK.ipynb unify_patch_cells.py
"""
import json
import math
import re
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

cells = ["".join(c["source"]) for c in json.load(open(sys.argv[1]))["cells"]]
text = open(sys.argv[2]).read()
blocks = dict(zip(re.findall(r"^# === (BLOCK \d)", text, re.M), re.split(r"^# === BLOCK \d.*\n", text, flags=re.M)[1:]))
body = text[text.index("# === BLOCK 1"):]
assert not [l for l in body.splitlines() if len(l) > 72 and not l.startswith("# === ")]


def cell(*needles):
    hits = [c for c in cells if all(n in c for n in needles)]
    assert len(hits) == 1, (needles, len(hits))
    return hits[0]


def grey_of(x):
    return x.mean(1, keepdim=True).expand_as(x)


def chroma(x):          # how far the channels are from their per-pixel mean
    return (x - x.mean(1, keepdim=True)).abs().mean().item()


P, E = 16, 256           # the notebook's patch size and emb_dim
torch.manual_seed(0)
image = torch.rand(2, 3, 32, 32)

# ---- the notebook as pushed ----
old = dict(torch=torch, nn=nn, F=F, math=math)
for n in ("class ChannelGroupResize", "class LinSkipPatchEmbed", "class LinSkipUnEmbed"):
    exec(cell(n), old)
emb, un = old["LinSkipPatchEmbed"](P, 3, E), old["LinSkipUnEmbed"](P, 3, E)
print(f"old skip: {3 * P * P} patch values -> {E} token channels = groups of {3 * P * P // E} consecutive values (R, G, B of one pixel)")
tok, tok_grey = emb.skip(emb_in := image.reshape(2, 3, 2, P, 2, P).permute(0, 2, 4, 3, 5, 1).reshape(2, 4, P * P * 3)), None
tok_grey = emb.skip(grey_of(image).reshape(2, 3, 2, P, 2, P).permute(0, 2, 4, 3, 5, 1).reshape(2, 4, P * P * 3))
assert torch.allclose(tok, tok_grey, atol=1e-6), "old embed skip does see colour"
out = un.skip(torch.randn(2, 4, E)).reshape(2, 2, 2, P, P, 3).permute(0, 5, 1, 3, 2, 4).reshape(2, 3, 32, 32)
assert chroma(out) < 1e-7, chroma(out)
full = un(torch.randn(2, 4, E), 2, 2)
print(f"1 ok: OLD embed skip gives the same tokens for an image and its grey version; OLD un-embed skip output has R = G = B exactly")
print(f"      whole old un-embed at init: colour part {chroma(full):.1e} against {full.abs().mean():.2f} overall (the learned branch starts at gain 1e-4)")

# ---- the new cells ----
new = dict(torch=torch, nn=nn, F=F, math=math, patch_size=P, emb_dim=E)
before = set(new)
exec(blocks["BLOCK 1"], new)
assert set(new) - before - {"__builtins__"} == {"PatchPoolSkip", "LinSkipPatchEmbed", "LinSkipUnEmbed"}
emb2, un2 = new["LinSkipPatchEmbed"](P, 3, E), new["LinSkipUnEmbed"](P, 3, E)
assert (emb2.skip.k, emb2.skip.n_used) == (8, 192)
assert [new["PatchPoolSkip"](16, 3, e).k for e in (48, 192, 256, 767, 768)] == [4, 8, 8, 8, 16]
tokens = emb2(image)
assert tokens.shape == (2, 4, E)
with torch.no_grad():
    emb2.norm.weight.zero_(); un2.norm.weight.zero_()      # skip path only
    back = un2(emb2(image), 2, 2)
    back_grey = un2(emb2(grey_of(image)), 2, 2)
assert back.shape == image.shape
assert chroma(back) > 0.3 * chroma(image) and not torch.allclose(back, back_grey, atol=1e-3)
# pure colours survive exactly; a constant image comes back unchanged
for rgb in ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.), (0.2, 0.5, 0.9)):
    const = torch.tensor(rgb)[None, :, None, None].expand(1, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(un2(emb2(const), 2, 2), const, atol=1e-6), rgb
# it is a blur per channel, placed correctly: a smooth image comes back close, patch by patch
yy, xx = torch.meshgrid(torch.linspace(0, 1, 32), torch.linspace(0, 1, 32), indexing="ij")
smooth = torch.stack((xx, yy, 1 - xx))[None]
with torch.no_grad():
    err = (un2(emb2(smooth), 2, 2) - smooth).abs().max().item()
assert err < 0.04, err
# symmetry: all positions alike. Mirror or quarter-turn every patch, and the skip output is mirrored / turned too;
# the patch mean is kept exactly; applying the skip twice changes nothing more.
with torch.no_grad():
    def skip_only(img):
        return un2(emb2(img), 2, 2)
    def per_patch(img, fn):
        p = img.reshape(2, 3, 2, P, 2, P).permute(0, 1, 2, 4, 3, 5)
        return fn(p).permute(0, 1, 2, 4, 3, 5).reshape(2, 3, 32, 32)
    for fn in (lambda p: p.flip(-1), lambda p: p.flip(-2), lambda p: p.rot90(1, (-2, -1))):
        assert torch.allclose(skip_only(per_patch(image, fn)), per_patch(back, fn), atol=1e-6)
    assert torch.allclose(back.reshape(2, 3, 2, P, 2, P).mean((3, 5)), image.reshape(2, 3, 2, P, 2, P).mean((3, 5)), atol=1e-6)
    assert torch.allclose(skip_only(back), back, atol=1e-6)
print(f"2 ok: NEW skip keeps colour: constant colours come back exact, a smooth colour ramp within {err:.3f}, random-image colour kept at"
      f" {chroma(back) / chroma(image):.0%}")

# layout: the embed / un-embed pair puts every patch back where it came from (identity with a full-size token)
embF, unF = new["LinSkipPatchEmbed"](4, 3, 48), new["LinSkipUnEmbed"](4, 3, 48)      # 3 * 4 * 4 = 48: nothing is pooled
with torch.no_grad():
    embF.norm.weight.zero_(); unF.norm.weight.zero_()
    img = torch.rand(2, 3, 8, 12)
    assert torch.allclose(unF(embF(img), 2, 3), img, atol=1e-6)
print("3 ok: with a token as large as the patch (4 px, 48 channels) embed -> un-embed is the identity, also on a non-square grid")

# the whole model still builds and trains with the new cells
g = dict(torch=torch, nn=nn, F=F, math=math, device="cpu")
exec("num_blocks=4; n_heads=4; head_size=16; emb_dim=64; patch_size=16; keep_ratio=0.25; shallow_blocks=1; final_blocks=1; c=3", g)
exec(cell("def apply_2d_rope"), g)
exec(blocks["BLOCK 1"], g)
for n in ("class MHA(nn.Module)", "class SwiGLU(nn.Module)", "class Block(nn.Module)", "class TimestepEmbedder", "class CoreTransformer(nn.Module)",
          "class ImageTransformer(nn.Module)", "def make_train_step"):
    exec(cell(n), g)
model = g["ImageTransformer"](3, 16, 64, 4, 4, 16, 0.25, 1, 1)
g["model"] = model
opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
step = g["make_train_step"]()
batch = torch.rand(6, 3, 32, 32)
losses = [step(batch, opt)["loss"].item() for _ in range(60)]
assert losses[-1] == losses[-1] and sum(losses[-10:]) < sum(losses[:10])
model.eval()
with torch.no_grad():
    out = model(torch.randn(2, 3, 32, 32), torch.rand(2))
assert out.shape == (2, 3, 32, 32) and chroma(out) > 1e-3
print(f"4 ok: the notebook's model builds with the new cells, 60 train steps lower the loss, and its output has colour ({chroma(out):.3f})")

exec(blocks["BLOCK 2"], dict(new))
print("ALL OK")
