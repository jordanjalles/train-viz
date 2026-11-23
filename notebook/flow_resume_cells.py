# Save / load / resume cells for
# MaskedAutoencoders/ImageNet1-Latent-Flow-Matching-XM.ipynb
#
# Source of truth for public/flow_resume.html (built by
# make_flow_resume_page.py) and for the tests.
# A block starts at a "# === BLOCK" line. The "# MODE:" and
# "# FIND:" lines say how the block goes into the notebook and
# are not part of the pasted code. Lines: 72 characters at most.
#
# Blocks follow the notebook from top to bottom:
#   1        hyperparameters (first cell)
#   2        validation image loader (data cells)
#   3        helpers: definitions only
#   4A, 4B   cache cells: short actions
#   5A-5C    edits of the training loop cell
#   6A-6D    bottom of the notebook: cells that overwrite things
#   E1, E2   another notebook that only needs the latents

# === BLOCK 1: run settings ===
# MODE: append
# FIND: from pathlib import Path
# ---- save / resume ----
RUN_NAME = "lf_imagenet_xm12"  # one folder per training run
KEEP_EVERY_EPOCHS = 10         # also keep epoch_010.pt, ...
AUTOSAVE_MINUTES = 30          # save inside an epoch this often

# === BLOCK 2: validation image loader ===
# MODE: new-cell
# FIND: train_ds = BufferedImageNet(
# Validation image loader (moved here from the val cache cell)
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

# === BLOCK 3: save / resume / z cache helpers ===
# MODE: new-cell
# FIND: #log metrics
# ---------- save / resume / z cache helpers: definitions only ---- #
# Running this cell loads nothing, saves nothing and changes nothing.
import shutil

from torch.utils.data import TensorDataset, DataLoader


def _atomic_save(obj, path):
    # Write a temp file, then swap it in. A crash in the middle
    # leaves the previous file as it was.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _z_cache_meta(n_images):
    # What a cache file must match to be used again.
    encoder_file = (
        Path(os.environ["ML_MODELS"]) / ae_name / "encoder.pt2"
    )
    stat = encoder_file.stat()

    return {
        "ae_name": ae_name,
        "image_size": int(IMAGE_SIZE),
        "n_images": int(n_images),
        "n_latent_tokens": int(n_latent_tokens),
        "emb_dim": int(emb_dim),
        # size and modified time of encoder.pt2
        "encoder_file": [int(stat.st_size), int(stat.st_mtime)],
    }


def z_cache(split, loader, n_images, force=False):
    # Latents of one split. Loads the file when it matches this
    # encoder and image size. Otherwise encodes and saves.
    path = (
        Path(os.environ["ML_CHECKPOINTS"])
        / "z_cache"
        / f"{ae_name}_{IMAGE_SIZE}px"
        / f"{split}.pt"
    )
    meta = _z_cache_meta(n_images)

    if force:
        print(f"z cache [{split}]: force=True, encoding again")

    elif not path.exists():
        print(f"z cache [{split}]: no file at {path}, encoding")

    else:
        saved = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
        changed = [
            key for key in meta
            if saved["meta"].get(key) != meta[key]
        ]

        if not changed:
            z, y = saved["z"], saved["y"]
            print(
                f"z cache [{split}]: loaded {len(z):,} latents "
                f"from {path} "
                f"({z.numel() * z.element_size() / 1024**3:.2f}"
                f" GiB)"
            )
            return z, y

        print(f"z cache [{split}]: {path} rejected, encoding")
        for key in changed:
            print(
                f"  {key}: file has {saved['meta'].get(key)!r}"
                f", notebook has {meta[key]!r}"
            )
        del saved

    z_all = torch.empty(
        n_images,
        n_latent_tokens,
        emb_dim,
        dtype=torch.bfloat16,
    )
    y_all = torch.empty(
        n_images,
        dtype=torch.long,
    )

    write_idx = 0

    with torch.inference_mode():

        for i, (images, labels) in enumerate(loader):

            images = images.to(
                device,
                non_blocking=True,
            )

            z = encoder(images)

            end_idx = min(
                write_idx + z.shape[0],
                n_images,
            )
            n = end_idx - write_idx

            z_all[write_idx:end_idx].copy_(
                z[:n].to(
                    device="cpu",
                    dtype=torch.bfloat16,
                )
            )
            y_all[write_idx:end_idx].copy_(
                labels[:n].to(
                    device="cpu",
                    dtype=torch.long,
                )
            )

            write_idx = end_idx

            if i % 100 == 0:
                print(
                    f"{split} batch {i:,} | "
                    f"{write_idx:,}/{n_images:,} "
                    f"({100 * write_idx / n_images:.1f}%)"
                )

    # Safety in case the loader produced fewer images than expected
    if write_idx < n_images:
        z_all = z_all[:write_idx].clone()
        y_all = y_all[:write_idx].clone()

    _atomic_save({"z": z_all, "y": y_all, "meta": meta}, path)
    print(f"z cache [{split}]: saved {len(z_all):,} to {path}")

    return z_all, y_all


def load_z_cache(ae_name, split="train", image_size=128):
    # Needs no encoder, no image loader and no flow model.
    path = (
        Path(os.environ["ML_CHECKPOINTS"])
        / "z_cache"
        / f"{ae_name}_{image_size}px"
        / f"{split}.pt"
    )
    saved = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )
    z, y = saved["z"], saved["y"]
    print(
        f"z cache [{split}]: {tuple(z.shape)} {z.dtype} "
        f"from {path}"
    )
    return z, y


def latent_loader(z, y, train, batch_size=64):
    # train=True: shuffled, full batches only.
    # train=False: fixed order, every sample.
    return DataLoader(
        TensorDataset(z, y),
        batch_size=batch_size,
        shuffle=train,
        num_workers=0,
        pin_memory=True,
        drop_last=train,
    )


def _run_dir(run_name=None):
    return (
        Path(os.environ["ML_CHECKPOINTS"])
        / "flow"
        / (run_name or RUN_NAME)
    )


def save_run(epochs_done, tag="latest", keep=True):
    # Everything needed to continue: model, optimizer, logs.
    state = {
        name.replace("_orig_mod.", ""): value
        for name, value in flow_model.state_dict().items()
    }

    checkpoint = {
        "flow_model": state,
        "optimizer": optimizer.state_dict(),
        "logs": {
            name: torch.tensor(values, dtype=torch.float32)
            for name, values in logs.items()
        },
        "epochs_done": int(epochs_done),
        "config": {
            "ae_name": ae_name,
            "image_size": int(IMAGE_SIZE),
            "n_latent_tokens": n_latent_tokens,
            "emb_dim": emb_dim,
            "num_blocks": num_blocks,
            "n_heads": n_heads,
            "head_size": head_size,
            "flow_emb_dim": flow_emb_dim,
            "flow_noise_scale": flow_noise_scale,
            "XM_K": XM_K,
            "XM_N": XM_N,
        },
    }

    path = _run_dir() / f"{tag}.pt"
    _atomic_save(checkpoint, path)

    # The run state lives on the model, not in this cell, so
    # running this cell again never resets it.
    flow_model.epochs_done = int(epochs_done)
    flow_model.last_save_time = time.perf_counter()

    note = ""
    if (
        keep
        and epochs_done > 0
        and epochs_done % KEEP_EVERY_EPOCHS == 0
    ):
        kept = path.with_name(f"epoch_{epochs_done:03d}.pt")
        # copy to a temp name, then swap it in: stopping the
        # kernel in the middle never leaves a half-written copy
        tmp = kept.with_name(kept.name + ".tmp")
        shutil.copy2(path, tmp)
        os.replace(tmp, kept)
        note = f" (+ {kept.name})"

    print(f"saved {RUN_NAME} | epochs done {epochs_done}{note}")


def load_run(run_name=None, tag="latest"):
    # Restore model, optimizer and logs in place. The training
    # loop then starts at the saved epoch.
    global start_epoch

    path = _run_dir(run_name) / f"{tag}.pt"
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    saved_ae = checkpoint["config"]["ae_name"]
    if saved_ae != ae_name:
        print(
            f"WARNING: run was trained on {saved_ae!r}, "
            f"notebook uses {ae_name!r}"
        )

    model = getattr(flow_model, "_orig_mod", flow_model)
    result = model.load_state_dict(
        checkpoint["flow_model"],
        strict=False,
    )
    if result.missing_keys:
        print("not in checkpoint (kept as initialized):")
        print("  " + ", ".join(result.missing_keys))
    if result.unexpected_keys:
        print("in checkpoint but not in model (ignored):")
        print("  " + ", ".join(result.unexpected_keys))

    try:
        optimizer.load_state_dict(checkpoint["optimizer"])
        optimizer_note = "optimizer restored"
    except (ValueError, KeyError, RuntimeError) as error:
        optimizer_note = f"optimizer NOT restored ({error})"

    logs.clear()
    for name, values in checkpoint["logs"].items():
        logs[name] = values.tolist()

    start_epoch = checkpoint["epochs_done"]
    flow_model.epochs_done = start_epoch

    print(
        f"resumed {run_name or RUN_NAME} from {path}\n"
        f"  epochs done: {start_epoch} | "
        f"steps logged: {len(logs.get('flow_loss', [])):,} | "
        f"lr: {optimizer.param_groups[0]['lr']:.2e} | "
        f"{optimizer_note}"
    )

    return start_epoch


def resume_epoch():
    # Epoch the training loop starts at. 0 for a new model.
    return getattr(flow_model, "epochs_done", 0)


def autosave(epoch):
    # Called every step. Saves when AUTOSAVE_MINUTES have passed
    # since the last save. A resume from this save repeats the
    # interrupted epoch.
    now = time.perf_counter()
    last = getattr(flow_model, "last_save_time", None)

    if last is None:
        # first step of a new model: start the clock
        flow_model.last_save_time = now

    elif now - last >= AUTOSAVE_MINUTES * 60:
        save_run(epoch, keep=False)

# === BLOCK 4A: train latent cache ===
# MODE: replace-cell
# FIND: # ---------- Cache encoder outputs for latent-flow training
# ---------- Cache encoder outputs for latent-flow training -------- #
# Loads from disk when the file matches this encoder.
cached_z, cached_y = z_cache(
    "train", train_dataloader, len(train_ds),
)
latent_train_dataloader = latent_loader(
    cached_z, cached_y, train=True,
    batch_size=CACHE_BATCH_SIZE,
)
print(f"Latent batches: {len(latent_train_dataloader):,}")

# === BLOCK 4B: validation latent cache ===
# MODE: replace-cell
# FIND: # ---------- Cache encoder VALIDATION outputs for testing
# ---------- Cache encoder VALIDATION outputs for testing ---------- #
# Loads from disk when the file matches this encoder.
cached_test_z, cached_test_y = z_cache(
    "val", test_dataloader, len(test_ds),
)
latent_test_dataloader = latent_loader(
    cached_test_z, cached_test_y, train=False,
    batch_size=CACHE_BATCH_SIZE,
)
print(f"Validation batches: {len(latent_test_dataloader):,}")

# === BLOCK 5A: training loop starts at the saved epoch ===
# MODE: replace-line
# FIND: for epoch in range(num_epochs):
start_epoch = resume_epoch()

for epoch in range(start_epoch, num_epochs):

# === BLOCK 5B: save every AUTOSAVE_MINUTES inside an epoch ===
# MODE: insert-above
# FIND:         report_steps += 1
        autosave(epoch)

# === BLOCK 5C: save at the end of every epoch ===
# MODE: append
# FIND: # MAIN TRAINING LOOP
    save_run(epoch + 1)

# === BLOCK 6A: STOP line ===
# MODE: new-cell
# FIND: # load flow model
####################### STOP - SAVING/OVERWRITING BELOW ################

# === BLOCK 6B: resume a run ===
# MODE: new-cell
# FIND: # load flow model
# RESUME A RUN
# Overwrites flow_model, optimizer and logs with the last save
# of RUN_NAME. Then run the "# MAIN TRAINING LOOP" cell.
load_run()

# === BLOCK 6C: save now ===
# MODE: new-cell
# FIND: # load flow model
# SAVE NOW
# Overwrites latest.pt of RUN_NAME on disk.
save_run(resume_epoch())

# === BLOCK 6D: re-encode the z cache ===
# MODE: new-cell
# FIND: # load flow model
# RE-ENCODE THE Z CACHE
# Overwrites both cache files on disk. Takes as long as the
# first encode. Remove the "# " of the lines below to run it.
# Then run the two cache cells again to rebuild the loaders.
# z_cache(
#     "train", train_dataloader, len(train_ds), force=True,
# )
# z_cache(
#     "val", test_dataloader, len(test_ds), force=True,
# )

# === BLOCK E1: z cache helpers for another notebook ===
# MODE: new-cell
# ---------- z cache helpers: definitions only ---------- #
# Running this cell loads nothing and changes nothing.
import os
from pathlib import Path

import torch
from torch.utils.data import TensorDataset, DataLoader


def load_z_cache(ae_name, split="train", image_size=128):
    # Needs no encoder, no image loader and no flow model.
    path = (
        Path(os.environ["ML_CHECKPOINTS"])
        / "z_cache"
        / f"{ae_name}_{image_size}px"
        / f"{split}.pt"
    )
    saved = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )
    z, y = saved["z"], saved["y"]
    print(
        f"z cache [{split}]: {tuple(z.shape)} {z.dtype} "
        f"from {path}"
    )
    return z, y


def latent_loader(z, y, train, batch_size=64):
    # train=True: shuffled, full batches only.
    # train=False: fixed order, every sample.
    return DataLoader(
        TensorDataset(z, y),
        batch_size=batch_size,
        shuffle=train,
        num_workers=0,
        pin_memory=True,
        drop_last=train,
    )

# === BLOCK E2: load the latents in another notebook ===
# MODE: new-cell
# ---------- Load the ImageNet latents ---------- #
AE_NAME = "ae_25M_selfteach_imagenet_MOcrop_180k"

cached_z, cached_y = load_z_cache(AE_NAME, "train")
cached_test_z, cached_test_y = load_z_cache(AE_NAME, "val")

latent_train_dataloader = latent_loader(
    cached_z, cached_y, train=True,
)
latent_test_dataloader = latent_loader(
    cached_test_z, cached_test_y, train=False,
)
