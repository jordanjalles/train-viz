# Unified core transformer for MaskedAutoencoders/ImageNet-Unify.ipynb
#
# Source of truth for public/unify_core.html (build_unify_page.py) and
# for unify_core_test.py. One class for dense and sparse middle. It
# reads no notebook globals and keeps no RoPE cache on the module.

# === BLOCK 1: core transformer (replaces the two cells "#define core transformer model" and "class SparseMiddleTransformer") ===
#define core transformer model (dense or sparse middle)
import math


class CoreTransformer(nn.Module):
    """
    Shallow blocks → middle blocks → fusion → final blocks.

    keep_ratio = 1: plain dense transformer, no fusion layer.
    keep_ratio < 1: in a training step the middle blocks see a
    random keep_ratio share of the tokens (SPRINT sparse middle).

    Tokens are [latent tokens | patch tokens]. The patch count is
    patch_rows * patch_cols; everything before it is latent.
    Reads no globals. Keeps no RoPE cache.
    """

    def __init__(
        self,
        emb_dim,
        num_blocks,
        n_heads,
        head_size,
        n_tokens,
        keep_ratio=1.0,
        shallow_blocks=1,
        final_blocks=1,
    ):
        super().__init__()

        if not 0 < keep_ratio <= 1:
            raise ValueError("keep_ratio must be in (0, 1].")
        if shallow_blocks < 1 or final_blocks < 1:
            raise ValueError(
                "Need at least one shallow and final block."
            )
        if (
            keep_ratio < 1
            and shallow_blocks + final_blocks >= num_blocks
        ):
            raise ValueError("Need at least one middle block.")

        self.blocks = nn.ModuleList([
            Block(n_heads, head_size, emb_dim, n_tokens)
                for _ in range(num_blocks)
        ])

        self.norm_in = nn.RMSNorm(emb_dim)
        self.norm_out = nn.RMSNorm(emb_dim)

        self.keep_ratio = keep_ratio
        self.shallow_blocks = shallow_blocks
        self.middle_end = num_blocks - final_blocks

        # Set False for full-token training (needs keep_ratio < 1).
        self.sparse_middle = keep_ratio < 1

        if keep_ratio < 1:
            # Fuse shallow and deep features: 2C → C.
            self.fusion = nn.Linear(2 * emb_dim, emb_dim)

            # Start with equal contributions from both paths.
            with torch.no_grad():
                eye = torch.eye(emb_dim)
                self.fusion.weight.copy_(
                    torch.cat((0.5 * eye, 0.5 * eye), dim=1)
                )
                self.fusion.bias.zero_()

        #ROPE frequencies (a constant, not a cache)
        pair_idx = torch.arange(head_size // 4, dtype=torch.float32)
        inv_freq = 100 ** (-pair_idx / (head_size // 4))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def rope_tables(self, patch_rows, patch_cols):
        # Built on every call and returned. Nothing is stored, so a
        # compiled step never guards on a table from the last call.
        device = self.inv_freq.device

        row_pos = (
            2 * torch.arange(
                patch_rows, device=device, dtype=torch.float32
            ) / max(patch_rows - 1, 1) - float(patch_rows > 1)
        ).repeat_interleave(patch_cols)

        col_pos = (
            2 * torch.arange(
                patch_cols, device=device, dtype=torch.float32
            ) / max(patch_cols - 1, 1) - float(patch_cols > 1)
        ).repeat(patch_rows)

        scale = 1.0

        if self.training:
            # Uniform in log-space: shrinking and enlarging balance.
            log_scale = torch.empty(
                (), device=device
            ).uniform_(-math.log(2), math.log(2))

            scale = log_scale.exp()

        row_angles = (
            2 * math.pi * scale
            * row_pos[:, None] * self.inv_freq[None, :]
        )
        col_angles = (
            2 * math.pi * scale
            * col_pos[:, None] * self.inv_freq[None, :]
        )

        return (
            row_angles.cos(),
            row_angles.sin(),
            col_angles.cos(),
            col_angles.sin(),
        )

    @staticmethod
    def gather_tokens(x, indices):
        return x.gather(
            dim=1,
            index=indices.unsqueeze(-1).expand(
                -1, -1, x.shape[-1],
            ),
        )

    @staticmethod
    def gather_rope(rope, indices):
        selected = []
        for table in rope:
            if table.ndim == 2:
                # Shared full-grid table.
                selected.append(table[indices])
            else:
                # Already selected per image by input masking.
                selected.append(
                    table.gather(
                        dim=1,
                        index=indices.unsqueeze(-1).expand(
                            -1, -1, table.shape[-1],
                        ),
                    )
                )
        return tuple(selected)

    @staticmethod
    def sample_indices(batch, count, keep, device):
        # Sort selected indices to preserve original token order.
        return (
            torch.rand(batch, count, device=device)
            .topk(keep, dim=1, largest=False)
            .indices.sort(dim=1)
            .values
        )

    def forward(self, x, patch_rows, patch_cols, keep_indices=None):

        x = self.norm_in(x)

        rope = self.rope_tables(patch_rows, patch_cols)

        # Input masking: x already contains only visible patches.
        if keep_indices is not None:
            rope = self.gather_rope(rope, keep_indices)

        B, T, C = x.shape

        # RoPE tells us how many patch tokens are present.
        P = rope[0].shape[-2]

        # Everything before those final patch tokens is latent.
        L = T - P

        assert L >= 0, (
            f"Sequence has {T} total tokens but RoPE contains "
            f"{P} patch positions."
        )

        use_sparse = (
            self.training
            and self.sparse_middle
            and self.keep_ratio < 1
        )

        if not use_sparse and self.keep_ratio == 1:
            # Plain dense transformer.
            for block in self.blocks:
                x = block(x, rope)

            return self.norm_out(x)

        # Shallow path sees all available tokens.
        for i in range(self.shallow_blocks):
            x = self.blocks[i](x, rope)

        shallow = x

        if use_sparse:
            # Fixed counts. At least one survives in each nonempty
            # group.
            keep_latents = min(L, max(1, int(L * self.keep_ratio)))
            keep_patches = min(P, max(1, int(P * self.keep_ratio)))

            latent_ids = self.sample_indices(
                B, L, keep_latents, x.device,
            )
            patch_ids = self.sample_indices(
                B, P, keep_patches, x.device,
            )

            # Latents remain the prefix; patches follow.
            token_ids = torch.cat(
                (latent_ids, L + patch_ids), dim=1,
            )

            deep = self.gather_tokens(shallow, token_ids)
            middle_rope = self.gather_rope(rope, patch_ids)
        else:
            deep = shallow
            middle_rope = rope

        for i in range(self.shallow_blocks, self.middle_end):
            deep = self.blocks[i](deep, middle_rope)

        if use_sparse:
            # Restore deep features to their original positions.
            deep = torch.zeros_like(shallow).scatter(
                dim=1,
                index=token_ids.unsqueeze(-1).expand(-1, -1, C),
                src=deep.to(shallow.dtype),
            )

        x = self.fusion(torch.cat((shallow, deep), dim=-1))

        for i in range(self.middle_end, len(self.blocks)):
            x = self.blocks[i](x, rope)

        return self.norm_out(x)

# === BLOCK 2: check (new cell below the core transformer cell; run it once) ===
# Core transformer check: token counts per block, no globals.
def check_core_transformer(keep_ratio=0.25, n_latent=8, grid=4):
    core = CoreTransformer(
        emb_dim=n_heads * head_size,
        num_blocks=num_blocks,
        n_heads=n_heads,
        head_size=head_size,
        n_tokens=n_latent,
        keep_ratio=keep_ratio,
    )
    seen = []
    for block in core.blocks:
        block.register_forward_hook(
            lambda m, args, out: seen.append(args[0].shape[1])
        )
    x = torch.randn(2, n_latent + grid * grid, n_heads * head_size)

    core.train()
    core(x, grid, grid)
    print("training step, tokens per block:", seen)
    seen.clear()

    core.eval()
    with torch.no_grad():
        core(x, grid, grid)
    print("eval, tokens per block:         ", seen)


check_core_transformer()
