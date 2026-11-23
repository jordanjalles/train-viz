# Training loop, readouts and seed plots for
# MaskedAutoencoders/ImageNet-Unify.ipynb (pixel-space x-prediction
# model: model(noised_images, t) -> clean images; t=0 noise, t=1 clean).
#
# Source of truth for public/unify_train.html (build_unify_train_page.py)
# and for unify_train_test.py. Blocks follow the notebook top to bottom.

# === BLOCK 1: log metrics | definition | new cell below "# get split losses (definitions only)" ===
#log metrics (definitions only)
from collections import defaultdict

# Keep the logs if this cell is run again.
if "logs" not in globals():
    logs = defaultdict(list)       # one value per training step
if "eval_logs" not in globals():
    eval_logs = defaultdict(list)  # one value per evaluation


def clear_logs():
    logs.clear()
    eval_logs.clear()


def log_metrics(**metrics):
    for name, value in metrics.items():
        if value is None:
            continue

        if torch.is_tensor(value):
            value = value.detach().item()

        logs[name].append(value)


def get_metrics_lists():
    return {
        name: history.copy()
        for name, history in logs.items()
        if history
    }

# === BLOCK 2: samplers and seed plots | definition | new cell below "#log metrics" ===
# samplers and seed plots (definitions only)
# t=0 is pure noise, t=1 is the clean image. The model predicts the
# clean image, so the velocity is (x_pred - x) / (1 - t).
import math


def make_timesteps(steps=20, scheduler="linear", shift=1.0):
    # steps + 1 rising times from 0 (noise) to 1 (image).
    # shift > 1 spends more steps at high noise.
    u = torch.linspace(0, 1, steps + 1, dtype=torch.float64)

    if scheduler == "linear":
        ts = u
    elif scheduler == "cosine":
        ts = 0.5 * (1 - torch.cos(math.pi * u))
    else:
        raise ValueError(f"Unknown scheduler: {scheduler}")

    noise_level = 1 - ts
    noise_level = shift * noise_level / (1 + (shift - 1) * noise_level)
    ts = (1 - noise_level).float()
    ts[0], ts[-1] = 0.0, 1.0
    return ts.to(device)


@torch.inference_mode()
def denoise_seeds(
    seeds, steps=20, sampler="euler", scheduler="linear",
    shift=1.0, image_size=None, return_steps=False,
):
    # One image per seed. An image depends only on its own seed,
    # not on the other seeds in the call. Samplers: euler, heun.
    if sampler not in ("euler", "heun"):
        raise ValueError(f"Unknown sampler: {sampler}")

    size = image_size or IMAGE_SIZE
    channels = model.un_embed.c
    seeds = [int(seed) for seed in seeds]

    x = torch.stack([
        torch.randn(
            (channels, size, size),
            generator=torch.Generator(device=device).manual_seed(s),
            device=device,
        )
        for s in seeds
    ])
    n = len(seeds)
    ts = make_timesteps(steps, scheduler, shift)

    def predict(x, t):
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            return model(x, t.expand(n)).float()

    was_training = model.training
    model.eval()
    trace = []
    try:
        for t, t_next in zip(ts[:-1], ts[1:]):
            dt = t_next - t
            x_pred = predict(x, t)
            v = (x_pred - x) / (1 - t)

            if return_steps:
                trace.append((float(t), x, x_pred))

            if sampler == "heun" and t_next < 1:
                x_euler = x + dt * v
                v_next = (
                    (predict(x_euler, t_next) - x_euler)
                    / (1 - t_next)
                )
                x = x + dt * 0.5 * (v + v_next)
            else:
                # The last step lands exactly on the prediction.
                x = x + dt * v
    finally:
        model.train(was_training)

    x = x.clamp(0, 1)
    return (x, trace) if return_steps else x


def denoise_seed(seed, steps=20, sampler="euler", **kwargs):
    return denoise_seeds([seed], steps, sampler, **kwargs)


def _show(ax, image, title=None):
    ax.imshow(
        image.detach().float().cpu().permute(1, 2, 0).clamp(0, 1)
    )
    ax.axis("off")
    if title is not None:
        ax.set_title(title, fontsize=9)


def plot_seeds(
    seeds=range(6), steps=20, cols=3, sampler="euler",
    title=None, **kwargs,
):
    seeds = [int(seed) for seed in seeds]
    if not seeds:
        return
    if cols < 1:
        raise ValueError("cols must be at least 1")

    images = denoise_seeds(seeds, steps, sampler, **kwargs)

    cols = min(cols, len(seeds))
    rows = math.ceil(len(seeds) / cols)
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(3 * cols, 3 * rows),
        squeeze=False,
    )
    for ax in axes.flat:
        ax.axis("off")
    for seed, image, ax in zip(seeds, images, axes.flat):
        _show(ax, image, f"seed {seed}")

    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_seed_trajectory(seed=1, steps=20, show=6, sampler="euler"):
    # Top row: the noisy image x at some steps.
    # Bottom row: the clean image the model predicts from it.
    final, trace = denoise_seeds(
        [seed], steps, sampler, return_steps=True,
    )
    picks = sorted({
        round(i * (len(trace) - 1) / max(show - 1, 1))
        for i in range(show)
    })

    fig, axes = plt.subplots(
        2, len(picks) + 1,
        figsize=(2.4 * (len(picks) + 1), 5),
        squeeze=False,
    )
    for col, i in enumerate(picks):
        t, x, x_pred = trace[i]
        _show(axes[0, col], x[0], f"x at t={t:.2f}")
        _show(axes[1, col], x_pred[0], "prediction")
    _show(axes[0, -1], final[0], "final")
    axes[1, -1].axis("off")

    fig.suptitle(f"seed {seed} · {steps} {sampler} steps")
    plt.tight_layout()
    plt.show()


def compare_samplers(
    seeds=range(3),
    samplers=("euler", "heun"),
    step_counts=(10, 20, 50),
):
    # One row per seed, one column per sampler and step count.
    seeds = [int(seed) for seed in seeds]
    settings = [
        (sampler, steps)
        for sampler in samplers
        for steps in step_counts
    ]

    fig, axes = plt.subplots(
        len(seeds), len(settings),
        figsize=(2.4 * len(settings), 2.4 * len(seeds)),
        squeeze=False,
    )
    for col, (sampler, steps) in enumerate(settings):
        images = denoise_seeds(seeds, steps, sampler)
        for row, image in enumerate(images):
            title = f"{sampler} {steps}" if row == 0 else None
            _show(axes[row, col], image, title)

    plt.tight_layout()
    plt.show()

# === BLOCK 3: training readouts | definition | new cell below "#make and compile the training steps" ===
# training readouts (definitions only)

class TrainReadout:
    # Speed and loss lines for the training loop. Holds only the
    # clock and the counts since its last line.
    def __init__(self, batches_per_epoch):
        self.batches_per_epoch = batches_per_epoch
        self.run_start = time.perf_counter()
        self.reset()

    def reset(self):
        self.steps = 0
        self.samples = 0
        self.start = time.perf_counter()

    def tick(self, batch_size):
        self.steps += 1
        self.samples += batch_size

    def report(self, step, epoch):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - self.start
        sec_per_step = elapsed / max(self.steps, 1)
        recent = logs["loss"][-self.steps:]
        total = int(time.perf_counter() - self.run_start)

        print(
            f"step {step:,} | epoch {epoch + 1} | "
            f"{sec_per_step * 1000:.0f} ms/step | "
            f"{self.samples / elapsed:.1f} img/s | "
            f"{self.batches_per_epoch * sec_per_step / 60:.1f}"
            f" min/epoch | "
            f"loss {sum(recent) / len(recent):.5f} | "
            f"lr {optimizer.param_groups[0]['lr']:.1e} | "
            f"{total // 3600}:{total % 3600 // 60:02d}"
            f":{total % 60:02d}"
        )
        self.reset()


def ensure_loader_alive(loader):
    # Stopping the kernel also stops the loader's worker processes.
    # A loader with persistent workers then fails on its next use
    # with "DataLoader worker exited unexpectedly". Drop the dead
    # workers so the next loop starts new ones.
    iterator = getattr(loader, "_iterator", None)
    if iterator is None:
        return

    workers = getattr(iterator, "_workers", [])
    if all(worker.is_alive() for worker in workers):
        return

    try:
        iterator._shutdown_workers()
    except Exception:
        pass
    loader._iterator = None
    print("data loader: workers had stopped, starting new ones")


def evaluate(step, label="EVAL"):
    # Split losses on fixed images, t and noise. Logged and printed.
    train_loss = get_split_loss("train").item()
    has_test = "test_dataloader" in globals()
    test_loss = get_split_loss("test").item() if has_test else None

    eval_logs["step"].append(step)
    eval_logs["train_loss"].append(train_loss)
    eval_logs["test_loss"].append(test_loss)

    if has_test:
        # earlier evaluations may have run without a test loader
        best, best_step = min(
            (loss, at)
            for loss, at in zip(
                eval_logs["test_loss"], eval_logs["step"],
            )
            if loss is not None
        )
        tail = (
            f"test-loss: {test_loss:.5f} | "
            f"best test {best:.5f} at step {best_step:,}"
        )
    else:
        tail = "test-loss: n/a (no test_dataloader)"

    print(
        f"{label} step {step:,} | "
        f"train-loss: {train_loss:.5f} | {tail}"
    )


def preview(step, seeds=range(6), steps=20):
    # Samples from fixed seeds, so previews are comparable.
    plot_seeds(seeds, steps=steps, title=f"step {step:,}")

# === BLOCK 4: main training loop | action | replaces the cell "## AE EXPERIMENT TRAINING LOOP ##" ===
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

        metrics = train_step(images, optimizer)

        log_metrics(loss=metrics["loss"])
        readout.tick(images.shape[0])
        step = len(logs["loss"])

        if step % report_every == 0:
            readout.report(step, epoch)

        if step % eval_every == 0:
            evaluate(step)
            readout.reset()

        if step % preview_every == 0:
            preview(step)
            readout.reset()

    print("-" * 62)
    evaluate(len(logs["loss"]), label=f"EPOCH {epoch + 1}")
    print("-" * 62)
    readout.reset()

# === BLOCK 5: history plots | definition | replaces the cell "history = get_metrics_lists()" ===
# history (definitions only)

def print_history():
    steps = len(logs["loss"])
    steps_per_epoch = len(train_dataloader)

    print(f"steps:           {steps:,}")
    print(f"steps / epoch:   {steps_per_epoch:,}")
    print(f"epochs trained:  {steps / steps_per_epoch:.2f}")
    if eval_logs["step"]:
        print(f"evaluations:     {len(eval_logs['step']):,}")


def plot_loss_history(window=200, start=0):
    # Train-step loss as a running mean, with the split losses.
    loss = torch.tensor(logs["loss"][start:], dtype=torch.float32)
    if len(loss) < 2:
        print("not enough steps to plot")
        return

    window = max(1, min(window, len(loss)))
    smooth = loss.unfold(0, window, 1).mean(-1)
    xs = torch.arange(len(smooth)) + start + window

    plt.figure(figsize=(10, 4))
    plt.plot(xs, smooth, label=f"train step (mean of {window})")

    for name in ("train_loss", "test_loss"):
        points = [
            (at, value)
            for at, value in zip(eval_logs["step"], eval_logs[name])
            if at >= start and value is not None
        ]
        if points:
            plt.plot(
                *zip(*points), "o-", ms=4,
                label=name.replace("_", " ") + " (fixed t, noise)",
            )

    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

# === BLOCK 6A: history | action | new cell below the history definitions ===
print_history()
plot_loss_history(window=200)

# === BLOCK 6B: seed plots | action | replaces the cell "render_reconstruction(masked=False, split='train');" ===
plot_seeds(steps=20, sampler="euler")

# === BLOCK 6C: seed plots | action | replaces the cell "render_reconstruction(masked=True);" ===
plot_seeds(sampler="heun")

# === BLOCK 6D: seed plots | action | new cell ===
plot_seeds(torch.randint(6, 90000, (6,)))

# === BLOCK 6E: seed plots | action | new cell ===
plot_seeds(
    torch.randint(6, 90000, (30,)).tolist(), cols=6, sampler="heun",
)

# === BLOCK 6F: seed plots | action | new cell ===
plot_seed_trajectory(seed=1, steps=20)

# === BLOCK 6G: seed plots | action | new cell ===
compare_samplers(
    seeds=range(3),
    samplers=("euler", "heun"),
    step_counts=(10, 20, 50),
)
