# Cells for ImageNet-Unify.ipynb: Pyramid-JiT style readouts.
# Low-resolution x-prediction heads at earlier blocks, each with its
# own loss. Source of unify_pyramid_test.py and of the 3090 / Spark
# job (job/pyramid_cells.py is a copy).
# Every line is at most 72 characters wide.

# === BLOCK 1: settings for the first cell ===
# pyramid readouts: (after block, pixels per token of the readout)
# With 16 px patches at 128 px there are 8 x 8 tokens, so
# 4 pixels per token = a 32 x 32 image, 8 = a 64 x 64 image.
pyramid_readouts = ((3, 4), (6, 8))
pyramid_weight = 1.0  # loss weight of every readout (final head: 1)

# === BLOCK 2: pyramid model (definitions only) ===
# pyramid readouts (definitions only)
# Running this cell builds nothing and changes nothing.


class ReadoutHead(nn.Module):
    """
    A low-resolution clean-image prediction from the token stream
    after a block. Same shape as the model's final path:
    RMSNorm → fixed pooled skip + learned linear → sigmoid.

    The skip reads the stream the way PatchPoolSkip wrote it
    (c x k x k per token) and average-pools it to q x q, so the
    default output is a small colour image, as in the final head.
    """

    def __init__(self, patch_size, out_patch, c, emb_dim):
        super().__init__()
        self.q, self.c = out_patch, c
        self.layout = PatchPoolSkip(patch_size, c, emb_dim)
        k = self.layout.k
        assert out_patch <= k, (
            f"readout of {out_patch} px per token is finer than "
            f"the stream's {k} x {k} skip layout"
        )

        n = c * out_patch * out_patch
        self.norm_in = nn.RMSNorm(emb_dim)
        self.linear = nn.Linear(emb_dim, n, bias=False)
        self.norm = nn.RMSNorm(n)
        nn.init.constant_(self.norm.weight, 1e-4)

    def forward(self, x, patch_rows, patch_cols):
        B, T = x.shape[:2]
        q, c, k = self.q, self.c, self.layout.k

        x = self.norm_in(x)

        skip = F.adaptive_avg_pool2d(
            x[..., :self.layout.n_used].reshape(B * T, c, k, k), q,
        ).reshape(B, T, c, q, q)

        patches = skip + self.norm(self.linear(x)).reshape(
            B, T, c, q, q,
        )

        # (B, rows, cols, c, q, q) -> (B, c, rows * q, cols * q)
        image = (
            patches.reshape(B, patch_rows, patch_cols, c, q, q)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(B, c, patch_rows * q, patch_cols * q)
        )

        return torch.sigmoid(image)


class PyramidImageTransformer(ImageTransformer):
    """
    ImageTransformer with readout heads along the trunk.

    model(images, t) is unchanged: the final image only. So the
    samplers and the split losses work as before, and the readout
    heads cost nothing at sampling time.
    model(images, t, all_readouts=True) returns a list:
    [readout images, low to high resolution..., final image].
    """

    def __init__(
        self,
        channels,
        patch_size,
        emb_dim,
        num_blocks,
        n_heads,
        head_size,
        keep_ratio,
        shallow_blocks,
        final_blocks,
        readouts=((3, 4), (6, 8)),
    ):
        super().__init__(
            channels,
            patch_size,
            emb_dim,
            num_blocks,
            n_heads,
            head_size,
            keep_ratio,
            shallow_blocks,
            final_blocks,
        )

        assert keep_ratio == 1, "pyramid readouts need keep_ratio = 1"
        blocks = [block for block, _ in readouts]
        assert blocks == sorted(set(blocks)), "one readout per block"
        assert all(1 <= b <= num_blocks for b in blocks), (
            "a readout sits after block 1 .. num_blocks (num_blocks: a pooled loss on the final tokens, 2026-10-11)"
        )

        # key = number of blocks that ran before the readout
        self.readout_heads = nn.ModuleDict({
            str(block): ReadoutHead(
                patch_size, out_patch, channels, emb_dim,
            )
            for block, out_patch in readouts
        })

    def forward(self, images, time, all_readouts=False):

        B, C, H, W = images.shape

        assert H % self.patch_size == 0
        assert W % self.patch_size == 0

        patch_rows = H // self.patch_size
        patch_cols = W // self.patch_size

        core = self.core_transformer

        x = core.norm_in(self.embed(images))
        time_emb = self.timestep_embedder(time)
        rope = core.rope_tables(patch_rows, patch_cols)

        outputs = []

        for i, block in enumerate(core.blocks):
            x = block(x, rope, time_emb)

            if all_readouts and str(i + 1) in self.readout_heads:
                # grad_mix (user, 2026-10-10): the gradient into this block = mix x its readout + (1 - mix) x downstream.
                # Values do not change: only the gradient is scaled. Train mode only.
                mix = getattr(self, "grad_mix", None) if self.training else None
                outputs.append(
                    self.readout_heads[str(i + 1)](
                        x if mix is None else mix * x + (1 - mix) * x.detach(), patch_rows, patch_cols,
                    )
                )
                if mix is not None:
                    x = (1 - mix) * x + mix * x.detach()

        final = torch.sigmoid(
            self.un_embed(core.norm_out(x), patch_rows, patch_cols)
        )

        if all_readouts:
            return outputs + [final]

        return final

# === BLOCK 3: init models (replaces the Def Init Models cell) ===
# --------------------Def Init Models (and optimizers) --------------------


def init_models():

    model = PyramidImageTransformer(
        channels = c,
        patch_size = patch_size,
        emb_dim = emb_dim,
        num_blocks = num_blocks,
        n_heads = n_heads,
        head_size = head_size,
        keep_ratio = keep_ratio,
        shallow_blocks = shallow_blocks,
        final_blocks = final_blocks,
        readouts = pyramid_readouts,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    # -------------------- Optimizers --------------------

    parameters = (
        list(model.parameters())
    )
    n_parameters = sum(p.numel() for p in parameters)
    print(f"{n_parameters:,} ae parameters")

    optimizer = torch.optim.AdamW(
        parameters,
        lr=lr,
        betas=(beta1, beta2),
        fused=True,
    )

    return (
        model,
        optimizer,
        device,
    )

# === BLOCK 4: train step (definitions only) ===
################## Pyramid Training Step ########################
# definitions only


def readout_losses(outputs, images, noised_images, t_img):
    # One loss per output of model(..., all_readouts=True).
    # The final image uses the notebook's v-loss. A readout uses
    # the same loss against the clean image resized to its size:
    # (x_pred - x0_small) / (1 - t), the v-loss with the noised
    # image cancelled out.
    denom = (1.0 - t_img).clamp_min(0.05)
    losses = []

    for x_pred in outputs:
        if x_pred.shape[-2:] == images.shape[-2:]:
            v_pred = (x_pred - noised_images) / denom
            v_target = (images - noised_images) / denom
            losses.append(
                F.mse_loss(v_pred.float(), v_target.float())
            )
        else:
            target = F.interpolate(
                images,
                size=x_pred.shape[-2:],
                mode="bilinear",
                antialias=True,
            )
            losses.append(
                ((x_pred.float() - target) / denom).square().mean()
            )

    return losses


def make_pyramid_train_step():

    def train_step(images, optimizer):

        optimizer.zero_grad(set_to_none=True)

        images = images.to(device, non_blocking=True)

        B = images.shape[0]

        # t=0 -> pure noise
        # t=1 -> clean image
        u = torch.randn(
            B,
            device=images.device,
            dtype=torch.float32,
        ) * t_logit_std + t_logit_mean

        t = torch.sigmoid(u)

        noise = torch.randn_like(images)

        t_img = t[:, None, None, None]

        noised_images = (
            t_img * images
            + (1.0 - t_img) * noise
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):

            outputs = model(noised_images, t, all_readouts=True)

            losses = readout_losses(
                outputs, images, noised_images, t_img,
            )

            # final head at weight 1, every readout at
            # pyramid_weight
            loss_final = losses[-1]
            loss = loss_final

            for readout_loss in losses[:-1]:
                loss = loss + pyramid_weight * readout_loss

        loss.backward()
        optimizer.step()

        return {
            "loss": loss.detach(),
            "loss_final": loss_final.detach(),
            "readout_losses": torch.stack(losses[:-1]).detach(),
        }

    return train_step

# === BLOCK 5: compile (replaces the compile cell) ===
#make and compile the training steps
torch.set_float32_matmul_precision('high')
train_step = torch.compile(make_pyramid_train_step())
