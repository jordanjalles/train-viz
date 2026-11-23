# Cells for ImageNet-Unify.ipynb: looped SPRINT middle with XSA and
# deep supervision. Source of public/unify_loop.html (built by
# build_unify_loop_page.py) and of unify_loop_test.py.
# Every line is at most 72 characters wide.

# === BLOCK 1: settings for the first cell ===
# looped sparse middle
n_loops = 4           # passes through the middle in a train step
sparse_per_dense = 4  # sparse steps before each dense step: S S S S D
use_xsa = True        # XSA in the attention of the looped middle
EVAL_LOOPS = (1, 2, 3, 4, 5, 6)  # loop counts that evaluate_loops tests
# True: an exit fuses the loop state with the shallow stream.
# False: no skip path. The loop state goes straight to the final
# blocks. Needs keep_ratio = 1.0; set sparse_per_dense = 0 with it.
loop_fusion = True

# === BLOCK 2: looped model (definitions only) ===
# looped sparse middle with XSA (definitions only)
# Running this cell builds nothing and changes nothing.


class XSAMHA(MHA):
    """
    MHA with XSA. After attention, each head's output loses its
    part along that token's own value vector. Per token and per
    head, before the heads merge and before out_proj.
    """

    def forward(self, x, rope):

        B, T, C = x.shape
        nh = self.n_heads
        hs = self.head_size

        # same slicing as MHA: q, then k, then v
        q, k, v = (
            part.view(B, T, nh, hs).permute(0, 2, 1, 3)
            for part in self.QKV(x).chunk(3, dim=-1)
        )

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = apply_2d_rope(
            q,
            k,
            rope=rope,
        )

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
        )

        # XSA
        v_hat = F.normalize(v, dim=-1)
        out = out - (out * v_hat).sum(dim=-1, keepdim=True) * v_hat

        out = out.permute(0, 2, 1, 3).flatten(-2)  # (B, T, H*hs)

        return self.out_proj(out)


class XSABlock(Block):
    # Block with XSA attention. Norms, gates and MLP are Block's.

    def __init__(self, n_heads, head_size, emb_dim):
        super().__init__(n_heads, head_size, emb_dim)
        self.MHA = XSAMHA(n_heads, head_size, emb_dim)


class LoopedCoreTransformer(CoreTransformer):
    """
    Shallow blocks → route tokens → [middle blocks] × n_loops
    → exit path (restore, fuse with shallow, final blocks).

    The same middle blocks run in every loop (shared weights).
    The token set is chosen once, before the first loop.
    The shallow stream is read only by exit paths. An exit path
    returns a prediction and feeds nothing back into the loop.
    """

    def __init__(
        self,
        emb_dim,
        num_blocks,
        n_heads,
        head_size,
        keep_ratio=1.0,
        shallow_blocks=1,
        final_blocks=1,
        n_loops=4,
        xsa=True,
        skip_fusion=True,
    ):
        super().__init__(
            emb_dim,
            num_blocks,
            n_heads,
            head_size,
            keep_ratio=keep_ratio,
            shallow_blocks=shallow_blocks,
            final_blocks=final_blocks,
        )

        if shallow_blocks + final_blocks >= num_blocks:
            raise ValueError("Need at least one middle block.")
        if not skip_fusion and keep_ratio < 1:
            # Without fusion the final blocks get the loop state
            # only, so it must hold every token.
            raise ValueError("skip_fusion=False needs keep_ratio = 1.")

        # False: no skip path around the loop (see exit_path).
        self.use_fusion = bool(skip_fusion)
        if n_loops < 1:
            raise ValueError("n_loops must be at least 1.")

        # Loop count of a call that does not pass n_loops.
        self.n_loops = n_loops

        if xsa:
            for i in range(self.shallow_blocks, self.middle_end):
                self.blocks[i] = XSABlock(
                    n_heads, head_size, emb_dim,
                )

        if self.use_fusion and not hasattr(self, "fusion"):
            # keep_ratio = 1: the parent made no fusion layer.
            # Exit paths always fuse, so make it here.
            self.fusion = nn.Linear(2 * emb_dim, emb_dim)

            with torch.no_grad():
                eye = torch.eye(emb_dim)
                self.fusion.weight.copy_(
                    torch.cat((0.5 * eye, 0.5 * eye), dim=1)
                )
                self.fusion.bias.zero_()

    def exit_path(self, deep, shallow, token_ids, rope, time_emb):
        # Read-only branch: loop state → prediction features.
        if token_ids is not None:
            # Restore deep features to their original positions.
            deep = torch.zeros_like(shallow).scatter(
                dim=1,
                index=token_ids.unsqueeze(-1).expand(
                    -1, -1, shallow.shape[-1],
                ),
                src=deep.to(shallow.dtype),
            )

        if self.use_fusion:
            x = self.fusion(torch.cat((shallow, deep), dim=-1))
        else:
            # No skip path: the final blocks see the loop state only.
            x = deep

        for i in range(self.middle_end, len(self.blocks)):
            x = self.blocks[i](x, rope, time_emb)

        return self.norm_out(x)

    def forward(
        self,
        x,
        patch_rows,
        patch_cols,
        time_emb,
        keep_indices=None,
        n_loops=None,
        all_exits=False,
        sparse=None,
    ):
        # n_loops=None: use self.n_loops.
        # all_exits=True: return a list, one exit per loop.
        # sparse=None: sparse in training mode when sparse_middle
        # is on, dense in eval mode. True / False: as given.

        if n_loops is None:
            n_loops = self.n_loops

        x = self.norm_in(x)

        rope = self.rope_tables(patch_rows, patch_cols)

        # Input masking: x already contains only visible patches.
        if keep_indices is not None:
            rope = self.gather_rope(rope, keep_indices)

        B, T, C = x.shape
        P = rope[0].shape[-2]
        L = T - P

        if sparse is None:
            sparse = self.training and self.sparse_middle

        use_sparse = bool(sparse) and self.keep_ratio < 1

        # Shallow path sees all tokens.
        for i in range(self.shallow_blocks):
            x = self.blocks[i](x, rope, time_emb)

        shallow = x
        token_ids = None

        if use_sparse:
            # One token set for the whole recurrent pass.
            keep_latents = min(L, max(1, int(L * self.keep_ratio)))
            keep_patches = min(P, max(1, int(P * self.keep_ratio)))

            latent_ids = self.sample_indices(
                B, L, keep_latents, x.device,
            )
            patch_ids = self.sample_indices(
                B, P, keep_patches, x.device,
            )

            token_ids = torch.cat(
                (latent_ids, L + patch_ids), dim=1,
            )

            deep = self.gather_tokens(shallow, token_ids)
            middle_rope = self.gather_rope(rope, patch_ids)
        else:
            deep = shallow
            middle_rope = rope

        exits = []

        for loop in range(n_loops):

            for i in range(self.shallow_blocks, self.middle_end):
                deep = self.blocks[i](deep, middle_rope, time_emb)

            if all_exits or loop == n_loops - 1:
                exits.append(
                    self.exit_path(
                        deep, shallow, token_ids, rope, time_emb,
                    )
                )

        return exits if all_exits else exits[-1]


class LoopedImageTransformer(ImageTransformer):
    # ImageTransformer with the looped core.
    # model(images, t) works as before: self.n_loops loops, dense
    # in eval mode. So samplers and split losses need no change.

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
        n_loops=4,
        xsa=True,
        skip_fusion=True,
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

        self.core_transformer = LoopedCoreTransformer(
            emb_dim=emb_dim,
            num_blocks=num_blocks,
            n_heads=n_heads,
            head_size=head_size,
            keep_ratio=keep_ratio,
            shallow_blocks=shallow_blocks,
            final_blocks=final_blocks,
            n_loops=n_loops,
            xsa=xsa,
            skip_fusion=skip_fusion,
        )

    def forward(
        self, images, time,
        n_loops=None, all_exits=False, sparse=None,
    ):

        B, C, H, W = images.shape

        assert H % self.patch_size == 0
        assert W % self.patch_size == 0

        patch_rows = H // self.patch_size
        patch_cols = W // self.patch_size

        patches = self.embed(images)

        time_emb = self.timestep_embedder(time)

        out = self.core_transformer(
            patches,
            patch_rows,
            patch_cols,
            time_emb,
            keep_indices=None,
            n_loops=n_loops,
            all_exits=all_exits,
            sparse=sparse,
        )

        def to_image(tokens):
            # every exit uses the same output head
            return torch.sigmoid(
                self.un_embed(tokens, patch_rows, patch_cols)
            )

        if all_exits:
            return [to_image(tokens) for tokens in out]

        return to_image(out)

# === BLOCK 3: init models (replaces the Def Init Models cell) ===
# --------------------Def Init Models (and optimizers) --------------------


def init_models():

    model = LoopedImageTransformer(
        channels = c,
        patch_size = patch_size,
        emb_dim = emb_dim,
        num_blocks = num_blocks,
        n_heads = n_heads,
        head_size = head_size,
        keep_ratio = keep_ratio,
        shallow_blocks = shallow_blocks,
        final_blocks = final_blocks,
        n_loops = n_loops,
        xsa = use_xsa,
        skip_fusion = loop_fusion,
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

# === BLOCK 4: loop helpers (definitions only) ===
# looped middle: checks, loop-count evals, charts (definitions only)
import contextlib


@contextlib.contextmanager
def use_loops(n):
    # Inside the "with", model(images, t) runs n middle loops.
    # Example: with use_loops(6): plot_seeds()
    core = model.core_transformer
    saved = core.n_loops
    core.n_loops = int(n)
    try:
        yield
    finally:
        core.n_loops = saved


def is_sparse_step(step):
    # S S S S D, S S S S D, ... for sparse_per_dense = 4.
    return step % (sparse_per_dense + 1) < sparse_per_dense


def check_looped_model():
    # Smoke test. Counts the tokens every block sees in one sparse
    # and one dense train-mode call, and checks the invariants.
    core = model.core_transformer
    was_training = model.training
    model.train()

    seen = []
    hooks = [
        block.register_forward_hook(
            lambda m, args, out, i=i: seen.append((i, out.shape[1]))
        )
        for i, block in enumerate(core.blocks)
    ]

    images = torch.rand(
        2, 3, IMAGE_SIZE, IMAGE_SIZE, device=device,
    )
    t = torch.rand(2, device=device)
    middle = range(core.shallow_blocks, core.middle_end)

    try:
        for name, sparse in (("sparse", True), ("dense", False)):
            seen.clear()
            with torch.no_grad():
                exits = model(images, t, all_exits=True, sparse=sparse)

            tokens = {
                i: sorted({n for j, n in seen if j == i})
                for i in range(len(core.blocks))
            }
            calls = {
                i: sum(j == i for j, _ in seen)
                for i in range(len(core.blocks))
            }
            mid = [tokens[i] for i in middle]
            print(
                f"{name:<6} step: {len(exits)} exits | "
                f"shallow blocks see {tokens[0][0]} tokens | "
                f"middle blocks see {mid[0][0]} tokens, "
                f"{calls[middle[0]]} times each | "
                f"final blocks see {tokens[len(core.blocks) - 1][0]}"
                f" tokens, {calls[len(core.blocks) - 1]} times each"
            )
            assert len(exits) == core.n_loops
            assert all(len(m) == 1 for m in mid), "token set changed"
            assert all(calls[i] == core.n_loops for i in middle)
            shallow = range(core.shallow_blocks)
            assert all(calls[i] == 1 for i in shallow)
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)

    xsa = [type(b.MHA).__name__ == "XSAMHA" for b in core.blocks]
    print(
        "XSA attention in blocks "
        f"{[i for i, on in enumerate(xsa) if on]} "
        f"(middle blocks are {list(middle)})"
    )
    print(
        f"train: {core.n_loops} loops, keep_ratio {core.keep_ratio}, "
        f"{sparse_per_dense} sparse steps then 1 dense step"
    )
    print(
        "exit path: "
        + ("fusion of the shallow stream and the loop state"
           if core.use_fusion else
           "NO fusion, the final blocks see the loop state only")
    )


def get_loop_losses(split="test", loops=None, n_batches=20):
    # The split loss for each loop count: same weights, same
    # images, same t and noise. Dense (eval mode).
    out = {}
    for n in (loops or EVAL_LOOPS):
        with use_loops(n):
            out[n] = get_split_loss(split, n_batches).item()
    return out


def evaluate_loops(step, split="test"):
    # Logs and prints the split loss for every loop count.
    losses = get_loop_losses(split)

    eval_logs["loops_step"].append(step)
    for n, loss in losses.items():
        eval_logs[f"{split}_loss_L{n}"].append(loss)

    print(
        f"LOOPS step {step:,} | {split}-loss by loop count: "
        + " | ".join(f"{n}: {loss:.5f}" for n, loss in losses.items())
    )


def plot_loop_losses(split="test", start=0):
    # Left: the split loss of each loop count over training.
    # Right: the latest split loss against the loop count.
    import matplotlib.pyplot as plt

    steps = eval_logs["loops_step"]
    loops = [
        n for n in EVAL_LOOPS
        if eval_logs.get(f"{split}_loss_L{n}")
    ]
    if not steps or not loops:
        print("no loop evals yet: run evaluate_loops(step) first")
        return

    keep = [i for i, s in enumerate(steps) if s >= start]
    trained = model.core_transformer.n_loops
    blues = plt.cm.Blues
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4))

    for rank, n in enumerate(loops):
        values = eval_logs[f"{split}_loss_L{n}"]
        left.plot(
            [steps[i] for i in keep],
            [values[i] for i in keep],
            color=blues(0.35 + 0.65 * rank / max(len(loops) - 1, 1)),
            linewidth=2.5 if n == trained else 1.2,
            label=f"{n} loops" + (" (trained)" if n == trained else ""),
        )

    left.set_xlabel("train step")
    left.set_ylabel(f"{split} loss")
    left.set_title(f"{split} loss by loop count")
    left.legend()
    left.grid(alpha=0.3)

    latest = [eval_logs[f"{split}_loss_L{n}"][-1] for n in loops]
    right.plot(loops, latest, marker="o", color=blues(0.8))
    right.axvline(trained, color="gray", linestyle=":", linewidth=1)
    right.set_xticks(loops)
    right.set_xlabel("loops at evaluation (dotted: trained)")
    right.set_ylabel(f"{split} loss")
    right.set_title(f"step {steps[-1]:,}")
    right.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

# === BLOCK 5: check the looped model (action) ===
check_looped_model()

# === BLOCK 6: train step (definitions only) ===
################## Looped Training Step ########################
# definitions only


def make_looped_train_step():

    def train_step(images, optimizer, sparse):
        # sparse=True: the middle sees keep_ratio of the tokens.
        # sparse=False: all tokens. Same weights, same loss rule.

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

            # One clean-image prediction after every loop.
            exits = model(
                noised_images, t, all_exits=True, sparse=sparse,
            )

            # JiT x-prediction parameterized as a v-loss.
            denom = (1.0 - t_img).clamp_min(0.05)
            v_target = (images - noised_images) / denom

            exit_losses = [
                F.mse_loss(
                    ((x_pred - noised_images) / denom).float(),
                    v_target.float(),
                )
                for x_pred in exits
            ]

            # Deep supervision: the last loop and the mean of the
            # loops before it weigh the same.
            loss_final = exit_losses[-1]

            if len(exit_losses) > 1:
                loss_intermediate = torch.stack(
                    exit_losses[:-1]
                ).mean()
                loss = 0.5 * loss_intermediate + 0.5 * loss_final
            else:
                loss = loss_final

        loss.backward()
        optimizer.step()

        return {
            "loss": loss.detach(),
            "loss_final": loss_final.detach(),
            "exit_losses": torch.stack(exit_losses).detach(),
        }

    return train_step

# === BLOCK 7: compile (replaces the compile cell) ===
#make and compile the training steps
torch.set_float32_matmul_precision('high')
train_step = torch.compile(make_looped_train_step())

# === BLOCK 8: main training loop (replaces the loop cell) ===
# MAIN TRAINING LOOP
######################################################################
num_epochs = 20
report_every = 500      # steps between speed / loss lines
eval_every = 2_000      # steps between split losses
preview_every = 10_000  # steps between sample previews

model.train()
ensure_loader_alive(train_dataloader)
batches_per_epoch = len(train_dataloader)
readout = TrainReadout(batches_per_epoch)

evaluate(len(logs["loss"]), label="START")

for epoch in range(num_epochs):

    for batch_step, (images, _) in enumerate(train_dataloader):

        # S S S S D: four sparse steps, then one dense step
        sparse = is_sparse_step(len(logs["loss"]))

        metrics = train_step(images, optimizer, sparse)

        log_metrics(
            loss=metrics["loss"],
            loss_final=metrics["loss_final"],
        )
        readout.tick(images.shape[0])
        step = len(logs["loss"])

        if step % report_every == 0:
            readout.report(step, epoch)

        if step % eval_every == 0:
            evaluate(step)
            evaluate_loops(step)
            readout.reset()

        if step % preview_every == 0:
            preview(step)
            readout.reset()

    print("-" * 62)
    evaluate(len(logs["loss"]), label=f"EPOCH {epoch + 1}")
    print("-" * 62)
    readout.reset()

# === BLOCK 9: chart the loop counts (action) ===
plot_loop_losses()
