def crop_scale(images, grid):
    # FP32 coordinates avoid BF16 rounding of crop positions.
    with torch.autocast(device_type="cuda", enabled=False):
        crops = F.grid_sample(
            images.float(), grid, mode="bilinear",
            padding_mode="border", align_corners=True,
        )
    # Same black mask as the dataset crop.
    mask = crop_inside_mask(grid, images.shape[-1])
    return crops * mask.to(crops.dtype)
