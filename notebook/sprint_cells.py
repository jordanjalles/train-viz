# Sparse-middle (SPRINT) cells for
# MaskedAutoencoders/ImageNet1-Latent-Flow-Matching-XM.ipynb
#
# Source of truth for public/sprint.html (build_sprint_page.py) and
# for sprint_test.py. Ported from the user's SparseMiddleTransformer
# in ImageNet-AE-selfteach.ipynb: same names (keep_ratio,
# shallow_blocks, final_blocks, sparse_middle, fusion), same fusion
# init, zeros at the tokens the middle skipped.
# Blocks follow the notebook's layout, top to bottom.
# The switch itself is a one-word edit in init_models (no block):
#   flow_model = FlowModel(  ->  flow_model = SparseMiddleFlowModel(

# === BLOCK 1: hyperparameters (add to the first cell) ===
#sparse middle (SPRINT)
use_sparse_middle = True    # False = every block sees all tokens
sparse_keep_ratio = 0.25    # share of tokens the middle sees
sparse_shallow_blocks = 2   # first blocks: all tokens
sparse_final_blocks = 2     # last blocks: all tokens

# === BLOCK 2: architecture (new cell below the flow model cell) ===
#define sparse-middle flow model (definitions only)

class SparseMiddleFlowModel(FlowModel):
    """
    Shallow blocks → sparse middle blocks → fusion → final blocks.

    Same parts as FlowModel, plus the fusion layer.
    """

    def __init__(
        self,
        emb_dim,
        flow_emb_dim,
        num_blocks,
        n_heads,
        head_size,
        keep_ratio=None,
        shallow_blocks=None,
        final_blocks=None,
    ):
        # None = take the value from the first cell.
        if keep_ratio is None:
            keep_ratio = sparse_keep_ratio
        if shallow_blocks is None:
            shallow_blocks = sparse_shallow_blocks
        if final_blocks is None:
            final_blocks = sparse_final_blocks

        super().__init__(
            emb_dim,
            flow_emb_dim,
            num_blocks,
            n_heads,
            head_size,
        )

        if not 0 < keep_ratio <= 1:
            raise ValueError("keep_ratio must be in (0, 1].")
        if shallow_blocks < 1 or final_blocks < 1:
            raise ValueError(
                "Need at least one shallow and final block."
            )
        if shallow_blocks + final_blocks >= num_blocks:
            raise ValueError("Need at least one middle block.")

        self.keep_ratio = keep_ratio
        self.shallow_blocks = shallow_blocks
        self.middle_end = num_blocks - final_blocks

        # False = full-token training. Set in the first cell,
        # or with set_sparse_middle(...) during a run.
        self.sparse_middle = bool(use_sparse_middle)

        # Fuse shallow and deep features: 2C → C.
        self.fusion = nn.Linear(2 * flow_emb_dim, flow_emb_dim)

        # Start with equal contributions from both paths.
        with torch.no_grad():
            eye = torch.eye(flow_emb_dim)
            self.fusion.weight.copy_(
                torch.cat((0.5 * eye, 0.5 * eye), dim=1)
            )
            self.fusion.bias.zero_()

    @staticmethod
    def gather_tokens(x, indices):
        return x.gather(
            dim=1,
            index=indices.unsqueeze(-1).expand(
                -1, -1, x.shape[-1],
            ),
        )

    @staticmethod
    def sample_indices(batch, count, keep, device):
        # Sort selected indices to preserve original token order.
        return (
            torch.rand(batch, count, device=device)
            .topk(keep, dim=1, largest=False)
            .indices.sort(dim=1)
            .values
        )

    def forward(self, z, time):

        z = self.in_proj(z) #emb_dim -> flow_emb_dim

        z = z + self.slot_embed

        time_emb = self.timestep_embedder(time)

        B, T, C = z.shape

        # Shallow path sees all tokens.
        for i in range(self.shallow_blocks):
            z = self.blocks[i](z, time_emb)

        shallow = z

        # Sparse only in a training step. A sampler that forgets
        # .eval() still runs without gradients, so it stays dense.
        use_sparse = (
            self.training
            and self.sparse_middle
            and self.keep_ratio < 1
            and torch.is_grad_enabled()
        )

        if use_sparse:
            keep = min(T, max(1, int(T * self.keep_ratio)))

            token_ids = self.sample_indices(
                B, T, keep, z.device,
            )

            deep = self.gather_tokens(shallow, token_ids)
        else:
            deep = shallow

        for i in range(self.shallow_blocks, self.middle_end):
            deep = self.blocks[i](deep, time_emb)

        if use_sparse:
            # Restore deep features to their original positions.
            deep = torch.zeros_like(shallow).scatter(
                dim=1,
                index=token_ids.unsqueeze(-1).expand(-1, -1, C),
                src=deep.to(shallow.dtype),
            )

        z = self.fusion(torch.cat((shallow, deep), dim=-1))

        for i in range(self.middle_end, len(self.blocks)):
            z = self.blocks[i](z, time_emb)

        ss = self.scale_shift(time_emb)[:, None, :] # (B, 1, C*2)

        (
            scale,
            shift,
        ) = ss.chunk(2, dim=-1) # each is (B, 1, C)

        out = self.out_proj(self.out_norm(z) * (1 + scale) + shift )

        return out


def set_sparse_middle(on, quiet=False):
    # Turn the sparse middle on or off. Works at any time.
    model = getattr(flow_model, "_orig_mod", flow_model)
    changed = model.sparse_middle != bool(on)
    model.sparse_middle = bool(on)

    if quiet and not changed:
        return

    T = model.n_latent_tokens
    n_middle = model.middle_end - model.shallow_blocks
    if model.sparse_middle and model.keep_ratio < 1:
        keep = min(T, max(1, int(T * model.keep_ratio)))
        print(
            f"sparse middle ON: {n_middle} middle blocks "
            f"train on {keep}/{T} tokens"
        )
    else:
        print(f"sparse middle OFF: all blocks see {T} tokens")


def check_sparse_middle():
    # Smoke test: how many tokens does each block see?
    model = getattr(flow_model, "_orig_mod", flow_model)
    seen = []
    hooks = [
        block.register_forward_hook(
            lambda m, args, out: seen.append(args[0].shape[1])
        )
        for block in model.blocks
    ]
    was_training = model.training
    z = torch.randn(4, n_latent_tokens, emb_dim, device=device)
    t = torch.rand(4, device=device)
    kind = "cuda" if str(device).startswith("cuda") else "cpu"
    try:
        model.train()
        with torch.autocast(kind, dtype=torch.bfloat16):
            model(z, t)
        print("training step, tokens per block:", seen)
        seen.clear()
        with torch.no_grad():
            model(z, t)
        print("sampling / eval, tokens per block:", seen)
    finally:
        model.train(was_training)
        for hook in hooks:
            hook.remove()

# === BLOCK 3: optional, automatic dense finish (add to the first cell) ===
sparse_until_epoch = 180    # dense from this epoch on

# === BLOCK 4: optional, automatic dense finish (first lines inside the epoch loop) ===
    #sparse middle follows sparse_until_epoch
    if sparse_until_epoch is not None:
        set_sparse_middle(
            epoch < sparse_until_epoch,
            quiet=True,
        )

# === BLOCK 5: path-drop guidance probe (definition cell) ===
# path-drop guidance (SPRINT): definitions only
#
# weak model = the same weights with the middle blocks skipped:
#   shallow blocks -> fusion (deep path = zeros) -> final blocks
# guided velocity = weak + scale * (full - weak)
#   scale 0 = weak model only, scale 1 = no guidance.

@torch.inference_mode()
def flow_velocity_weak(z, t):
    # The sparse-middle flow model with its middle blocks skipped.
    model = getattr(flow_model, "_orig_mod", flow_model)
    assert hasattr(model, "fusion"), (
        "flow_model is not a SparseMiddleFlowModel"
    )

    x = model.in_proj(z) + model.slot_embed
    time_emb = model.timestep_embedder(t)

    for i in range(model.shallow_blocks):
        x = model.blocks[i](x, time_emb)

    # No token went through the middle: the deep path is zeros,
    # as it is for every skipped token in a training step.
    x = model.fusion(
        torch.cat((x, torch.zeros_like(x)), dim=-1)
    )

    for i in range(model.middle_end, len(model.blocks)):
        x = model.blocks[i](x, time_emb)

    scale, shift = model.scale_shift(time_emb)[:, None, :].chunk(
        2, dim=-1,
    )
    return model.out_proj(model.out_norm(x) * (1 + scale) + shift)


@torch.inference_mode()
def denoise_seed_pdg(
    seed, guidance=2.0, steps=20, sampler="euler",
    scheduler="linear", scheduler_kwargs=None,
):
    # denoise_seed with path-drop guidance. Same seed -> same
    # starting noise as denoise_seed. Samplers: euler, heun,
    # midpoint.
    if sampler not in ("euler", "heun", "midpoint"):
        raise ValueError(f"Unknown sampler: {sampler}")

    def guided(z, t):
        weak = flow_velocity_weak(z, t)
        if guidance == 0:
            return weak
        full = flow_model(z, t)
        return weak + guidance * (full - weak)

    ts = make_timesteps(
        steps, scheduler, device=device,
        **(scheduler_kwargs or {}),
    )
    rng = torch.Generator(device=device).manual_seed(seed)

    was_training = flow_model.training
    flow_model.eval()
    try:
        z = torch.randn(
            (1, n_latent_tokens, emb_dim),
            generator=rng, device=device,
        ) * flow_noise_scale
        for t, t_next in zip(ts[:-1], ts[1:]):
            # own stepping: does not depend on _ode_step
            n = z.shape[0]
            dt = t_next - t
            v1 = guided(z, t.expand(n))
            if sampler == "euler":
                z = z + dt * v1
            elif sampler == "midpoint":
                t_mid = (t + t_next) / 2
                z = z + dt * guided(
                    z + 0.5 * dt * v1, t_mid.expand(n),
                )
            else:  # heun
                v2 = guided(z + dt * v1, t_next.expand(n))
                z = z + 0.5 * dt * (v1 + v2)
        return decoder(z)
    finally:
        flow_model.train(was_training)


@torch.inference_mode()
def path_drop_gap(n=16, times=(0.2, 0.5, 0.8)):
    # How far is the weak model from the full model?
    # Prints |full - weak| / |full| on noised z at a few times.
    # 0 means guidance cannot change anything.
    was_training = flow_model.training
    flow_model.eval()
    gaps = []
    try:
        rng = torch.Generator(device=device).manual_seed(0)
        z = torch.randn(
            (n, n_latent_tokens, emb_dim),
            generator=rng, device=device,
        ) * flow_noise_scale
        for time in times:
            t = torch.full((n,), float(time), device=device)
            full = flow_model(z, t).float()
            weak = flow_velocity_weak(z, t).float()
            size = full.norm().item()
            gap = (full - weak).norm().item()
            gaps.append(gap / size if size > 0 else 0.0)
            print(
                f"t={time:.1f} | full velocity size {size:9.3f}"
                f" | weak differs by {100 * gaps[-1]:6.1f}%"
            )
            if size == 0:
                print(
                    "  full velocity is exactly zero: the model "
                    "is untrained (out_proj starts at zero)"
                )
    finally:
        flow_model.train(was_training)
    return gaps


def plot_path_drop_guidance(
    seeds=range(4),
    guidances=(0.0, 1.0, 1.5, 2.0, 3.0),
    steps=20,
    sampler="euler",
):
    # One row per seed, one column per guidance scale.
    gaps = path_drop_gap()
    if max(gaps) < 1e-4:
        print(
            "Weak and full model are the same here, so every "
            "column will look alike."
        )
    seeds = [int(seed) for seed in seeds]
    fig, axes = plt.subplots(
        len(seeds), len(guidances),
        figsize=(3 * len(guidances), 3 * len(seeds)),
        squeeze=False,
    )

    for row, seed in zip(axes, seeds):
        for ax, guidance in zip(row, guidances):
            image = denoise_seed_pdg(
                seed, guidance, steps, sampler,
            )[0]
            ax.imshow(
                image.detach().float().cpu()
                .permute(1, 2, 0).clamp(0, 1)
            )
            ax.axis("off")
            if guidance == 0:
                name = "weak model only"
            elif guidance == 1:
                name = "no guidance"
            else:
                name = f"guidance {guidance:g}"
            ax.set_title(f"{name} · seed {seed}")

    fig.suptitle(
        f"Path-drop guidance · {steps} {sampler} steps"
    )
    plt.tight_layout()
    plt.show()

# === BLOCK 6: path-drop guidance probe (call cell) ===
plot_path_drop_guidance(
    seeds=range(4),
    guidances=(0.0, 1.0, 1.5, 2.0, 3.0),
    steps=20,
    sampler="euler",
)
