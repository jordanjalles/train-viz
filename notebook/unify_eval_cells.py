# Split-loss and micro-batch overfit cells for
# MaskedAutoencoders/ImageNet-Unify.ipynb (pixel-space x-prediction
# model: model(noised_images, t) -> clean images; t=0 noise, t=1 clean).
#
# Source of truth for public/unify_eval.html (build_unify_eval_page.py)
# and for unify_eval_test.py. Blocks follow the notebook top to bottom.

# === BLOCK 1: validation loader (new cell below the "train_ds = BufferedImageNet(" cell) ===
# Validation image loader
test_ds = BufferedImageNet(
    paths=val_paths,
    transform=ImageNetFullImage(IMAGE_SIZE),
    shards_in_ram=SHARDS_IN_RAM,
    shuffle=False,
    num_images=50_000,
)

test_generator = torch.Generator().manual_seed(
    DATA_SEED + 1
)

test_dataloader = DataLoader(
    test_ds,
    batch_size=CACHE_BATCH_SIZE,
    drop_last=False,

    num_workers=2,
    pin_memory=True,
    prefetch_factor=LOADER_PREFETCH_BATCHES,

    worker_init_fn=seed_imagenet_worker,
    generator=test_generator,

    persistent_workers=True,
    multiprocessing_context="fork",
)

# === BLOCK 2: split losses (new cell below the "def laplacian(x):" cell; definitions only) ===
# get split losses (definitions only)

def sample_train_t(B, generator=None):
    # The time distribution of train_step: logit-normal.
    # t_logit_mean and t_logit_std are set in the first cell and
    # are the same numbers train_step uses.
    u = torch.randn(
        B, device=device, dtype=torch.float32, generator=generator,
    ) * t_logit_std + t_logit_mean
    return torch.sigmoid(u)


def x_pred_v_loss(images, t, noise):
    # The train step's loss for given t and noise, one value per
    # image. Keep this in step with train_step.
    t_img = t[:, None, None, None]

    noised_images = (
        t_img * images
        + (1.0 - t_img) * noise
    )

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
    ):
        x_pred = model(noised_images, t)

    # JiT x-prediction parameterized as a v-loss.
    denom = (1.0 - t_img).clamp_min(0.05)

    v_pred = (x_pred.float() - noised_images) / denom
    v_target = (images - noised_images) / denom

    return (v_pred - v_target).square().mean((1, 2, 3))


def get_eval_batches(split, n_batches=20):
    # The first n_batches of a split, loaded once and kept on the
    # CPU. Every later call reuses the same images.
    cache = get_eval_batches.__dict__.setdefault("cache", {})

    if split == "micro":
        return [fixed_micro_batch]

    if split not in ("train", "test"):
        raise ValueError("split must be 'train', 'test' or 'micro'")

    if len(cache.get(split, [])) < n_batches:
        loader = (
            train_dataloader if split == "train"
            else test_dataloader
        )
        batches = []
        for i, (images, _) in enumerate(loader):
            if i >= n_batches:
                break
            batches.append(images.clone())
        cache[split] = batches

    return cache[split][:n_batches]


@torch.no_grad()
def get_split_loss(split, n_batches=20, seed=0):
    # Mean train-step loss on fixed images with fixed t and noise,
    # so two calls differ only by the model's weights.
    # t is drawn as in train_step (sample_train_t), so this number
    # is on the same scale as the train-step loss.
    # The model is in eval mode here: the sparse middle is off.
    was_training = model.training
    model.eval()

    rng = torch.Generator(device=device).manual_seed(seed)
    losses = []

    try:
        for images in get_eval_batches(split, n_batches):
            images = images.to(device, non_blocking=True)
            B = images.shape[0]

            t = sample_train_t(B, generator=rng)
            noise = torch.randn(
                images.shape, device=device, generator=rng,
            )

            losses.append(x_pred_v_loss(images, t, noise))

    finally:
        model.train(was_training)

    return torch.cat(losses).mean()

# === BLOCK 3: fix micro-batch (replaces the cell "# fixed_micro_batch_size = 64") ===
# FIX MICRO-BATCH
fixed_micro_batch_size = 6
fixed_micro_batch = next(
    iter(train_dataloader)
)[0][:fixed_micro_batch_size].to(device)

print("fixed_micro_batch:", tuple(fixed_micro_batch.shape))

# === BLOCK 4: micro-batch helpers (replaces the cell "# ############## overfit micro batch"; definitions only) ===
# micro-batch helpers (definitions only)

def plot_fixed_micro_batch():
    n = len(fixed_micro_batch)
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2), squeeze=False)

    for image, ax in zip(fixed_micro_batch, axes.flat):
        ax.imshow(image.permute(1, 2, 0).float().cpu().clamp(0, 1))
        ax.axis("off")

    plt.tight_layout()
    plt.show()


@torch.no_grad()
def plot_micro_batch_predictions(ts=(0.0, 0.3, 0.6, 0.9), seed=0):
    # Top row: the fixed images. Then, per t, the noised input
    # and the model's clean-image prediction from it.
    was_training = model.training
    model.eval()

    images = fixed_micro_batch
    n = len(images)
    rng = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(images.shape, device=device, generator=rng)

    rows = [("target", images)]
    try:
        for value in ts:
            t = torch.full((n,), float(value), device=device)
            t_img = t[:, None, None, None]
            noised = t_img * images + (1.0 - t_img) * noise

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                x_pred = model(noised, t)

            rows.append((f"input t={value:g}", noised))
            rows.append((f"prediction t={value:g}", x_pred))
    finally:
        model.train(was_training)

    fig, axes = plt.subplots(
        len(rows), n,
        figsize=(2 * n, 2 * len(rows)),
        squeeze=False,
    )
    for (name, batch), row in zip(rows, axes):
        for image, ax in zip(batch, row):
            ax.imshow(
                image.permute(1, 2, 0).float().cpu().clamp(0, 1)
            )
            ax.axis("off")
        row[0].set_title(name, loc="left", fontsize=9)

    plt.tight_layout()
    plt.show()


def overfit_micro_batch(steps=2000, report_every=200):
    # Train on the fixed micro-batch only. This changes the model:
    # run the init-models cell again before real training.
    model.train()
    start_loss = get_split_loss("micro")
    print(f"START MICRO-BATCH | micro-loss: {start_loss:.5f}")

    losses = []
    report_start = time.perf_counter()

    for step in range(1, steps + 1):
        metrics = train_step(fixed_micro_batch, optimizer)
        losses.append(metrics["loss"])

        if step % report_every == 0:
            elapsed = time.perf_counter() - report_start
            recent = torch.stack(losses[-report_every:]).mean()
            print(
                f"step {step:,} | "
                f"{elapsed / report_every * 1000:.0f} ms/step | "
                f"train-step loss={recent:.5f}"
            )
            report_start = time.perf_counter()

    end_loss = get_split_loss("micro")
    print(
        f"END MICRO-BATCH | micro-loss: {start_loss:.5f} "
        f"-> {end_loss:.5f}"
    )
    return torch.stack(losses).float().cpu()

# === BLOCK 5: overfit micro-batch (new cell below the micro-batch helpers) ===
# OVERFIT MICRO-BATCH
# Changes the model. Run the init-models cell again afterwards.
plot_fixed_micro_batch()

micro_losses = overfit_micro_batch(steps=2000, report_every=200)

plot_micro_batch_predictions(ts=(0.0, 0.3, 0.6, 0.9))
