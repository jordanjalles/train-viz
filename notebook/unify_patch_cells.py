# Colour-keeping patch embed / un-embed for
# MaskedAutoencoders/ImageNet-Unify.ipynb.
#
# Source of truth for public/unify_patch.html (build_unify_patch_page.py)
# and for unify_patch_test.py.

# === BLOCK 1: patch embed and un-embed (replaces the three cells "class ChannelGroupResize", "class LinSkipPatchEmbed", "class LinSkipUnEmbed") ===
# patch embed / un-embed with a colour-keeping skip (definitions only)


class PatchPoolSkip(nn.Module):
    """Fixed skip between the pixels of a patch and its token.

    Down: every colour channel of the p x p patch is average-pooled
    to k x k. The c * k * k values fill the first token channels;
    the rest are zero. Up: the reverse, with nearest upsampling.

    k is the largest size that divides the patch size and fits:
    c * k * k <= emb_dim. Because k divides p, every pixel is in
    exactly one block of the same size. The skip then treats all
    positions alike, and mirrors and quarter turns of a patch come
    out as mirrors and quarter turns.
    """
    def __init__(self, patch_size, c, emb_dim):
        super().__init__()
        self.p, self.c, self.emb_dim = patch_size, c, emb_dim
        fits = [
            k for k in range(1, patch_size + 1)
            if patch_size % k == 0 and c * k * k <= emb_dim
        ]
        assert fits, "emb_dim is smaller than the channels"
        self.k = max(fits)
        self.n_used = c * self.k * self.k

    def down(self, patches):
        # (B, T, c, p, p) -> (B, T, emb_dim)
        B, T = patches.shape[:2]
        x = F.adaptive_avg_pool2d(
            patches.reshape(B * T, self.c, self.p, self.p), self.k,
        ).reshape(B, T, self.n_used)
        return F.pad(x, (0, self.emb_dim - self.n_used))

    def up(self, tokens):
        # (B, T, emb_dim) -> (B, T, c, p, p)
        B, T = tokens.shape[:2]
        x = tokens[..., :self.n_used].reshape(
            B * T, self.c, self.k, self.k,
        )
        x = F.interpolate(x, size=(self.p, self.p), mode="nearest")
        return x.reshape(B, T, self.c, self.p, self.p)


class LinSkipPatchEmbed(nn.Module):
    def __init__(self, patch_size, c, emb_dim):
        super().__init__()
        self.p, self.c = patch_size, c
        n = c * patch_size * patch_size
        self.skip = PatchPoolSkip(patch_size, c, emb_dim)
        self.linear = nn.Linear(n, emb_dim, bias=False)
        self.norm = nn.RMSNorm(emb_dim)
        nn.init.constant_(self.norm.weight, 1e-4)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.p == 0 and W % self.p == 0
        p, rows, cols = self.p, H // self.p, W // self.p

        # (B, rows * cols, C, p, p): one small image per patch
        patches = (
            x.reshape(B, C, rows, p, cols, p)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(B, rows * cols, C, p, p)
        )
        flat = patches.reshape(B, rows * cols, C * p * p)

        return self.skip.down(patches) + self.norm(self.linear(flat))


class LinSkipUnEmbed(nn.Module):
    def __init__(self, patch_size, c, emb_dim):
        super().__init__()
        self.p, self.c = patch_size, c
        n = c * patch_size * patch_size
        self.skip = PatchPoolSkip(patch_size, c, emb_dim)
        self.linear = nn.Linear(emb_dim, n, bias=False)
        self.norm = nn.RMSNorm(n)
        nn.init.constant_(self.norm.weight, 1e-4)

    def forward(self, x, patch_rows, patch_cols):
        B, T = x.shape[:2]
        assert T == patch_rows * patch_cols
        p, c = self.p, self.c

        patches = (
            self.skip.up(x)
            + self.norm(self.linear(x)).reshape(B, T, c, p, p)
        )

        # (B, rows, cols, c, p, p) -> (B, c, rows * p, cols * p)
        return (
            patches.reshape(B, patch_rows, patch_cols, c, p, p)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(B, c, patch_rows * p, patch_cols * p)
        )

# === BLOCK 2: colour check (new cell below the patch cell; run it once) ===
# Does the fixed skip path keep colour? Pure red, green and blue in,
# through embed skip and un-embed skip, and out again.
def check_patch_colour():
    embed = LinSkipPatchEmbed(patch_size, 3, emb_dim)
    un_embed = LinSkipUnEmbed(patch_size, 3, emb_dim)
    size = 2 * patch_size
    for name, rgb in (
        ("red", (1., 0., 0.)), ("green", (0., 1., 0.)),
        ("blue", (0., 0., 1.)),
    ):
        image = torch.tensor(rgb)[None, :, None, None].expand(
            1, 3, size, size,
        )
        out = un_embed.skip.up(embed.skip.down(
            image.reshape(1, 3, 2, patch_size, 2, patch_size)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(1, 4, 3, patch_size, patch_size)
        ))
        mean = out.mean((0, 1, 3, 4))
        print(f"{name:<5} in -> out RGB",
              [round(v, 2) for v in mean.tolist()])


check_patch_colour()
