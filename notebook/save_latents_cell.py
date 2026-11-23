# SAVE THE LATENTS THAT ARE IN MEMORY
# Writes the two z cache files from cached_z / cached_test_z.
# Run the helper cell ("save / resume / z cache helpers") first.
for split, z, y, n_images in (
    ("train", cached_z, cached_y, 1_281_167),
    ("val", cached_test_z, cached_test_y, 50_000),
):
    assert z.shape[1:] == (n_latent_tokens, emb_dim), z.shape
    assert len(z) == len(y) == n_images, (len(z), len(y))
    path = (
        Path(os.environ["ML_CHECKPOINTS"]) / "z_cache"
        / f"{ae_name}_{IMAGE_SIZE}px" / f"{split}.pt"
    )
    _atomic_save(
        {"z": z, "y": y, "meta": _z_cache_meta(n_images)}, path,
    )
    print(f"saved {split}: {tuple(z.shape)} {z.dtype} -> {path}")
