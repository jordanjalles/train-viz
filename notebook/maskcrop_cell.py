#ImageNet view / crop definitions (masked, overhanging crops)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Largest share of the crop width that may lie off the image, per
# axis. 0.5: the crop centre is uniform over the image.
CROP_OVERHANG = 0.5
# Border pixels extend this far past their centre (in image pixels)
# before the crop turns black. 0.5 = the outer pixel's own edge.
CROP_EDGE_PIXELS = 0.5


def crop_inside_mask(grid, image_size):
    # True where a sample lies on the image.
    # grid (B, H, W, 2) -> mask (B, 1, H, W)
    limit = 1 + 2 * CROP_EDGE_PIXELS / (image_size - 1)
    return (grid.abs() <= limit).all(-1)[:, None]


def make_crop_grid_for_size(
    batch_size,
    height,
    width,
    device,
    crop_min=0.2,
    crop_max=0.9,
    output_size=None,
):
    if not 0 < crop_min <= crop_max <= 1:
        raise ValueError("Require 0 < crop_min <= crop_max <= 1.")

    size = height if output_size is None else int(output_size)

    bounds = torch.ones(2, device=device, dtype=torch.float32)

    fraction = torch.empty(
        batch_size, 1, device=device, dtype=torch.float32,
    ).uniform_(crop_min, crop_max)

    half_span = fraction * bounds

    # Up to CROP_OVERHANG of the crop width may lie off the image.
    center_limit = bounds - half_span * (1 - 2 * CROP_OVERHANG)
    center = (
        torch.rand(batch_size, 2, device=device) * 2 - 1
    ) * center_limit

    axis = torch.linspace(-1, 1, size, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    base = torch.stack((xx, yy), dim=-1)

    return (
        base[None] * half_span[:, None, None, :]
        + center[:, None, None, :]
    )


def antialiased_source_crop(source, grid):
    H, W = source.shape[-2:]
    out_h, out_w = grid.shape[1:3]

    step_x = (
        (grid[0, 0, -1, 0] - grid[0, 0, 0, 0]).abs().item()
        * (W - 1) / (2 * (out_w - 1))
    )
    step_y = (
        (grid[0, -1, 0, 1] - grid[0, 0, 0, 1]).abs().item()
        * (H - 1) / (2 * (out_h - 1))
    )

    new_h = min(H, max(2, int(np.ceil(H / max(step_y, 1)))))
    new_w = min(W, max(2, int(np.ceil(W / max(step_x, 1)))))

    filtered = source
    if (new_h, new_w) != (H, W):
        filtered = F.interpolate(
            source,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

    adjusted_grid = grid * grid.new_tensor([
        (W - 1) / W,
        (H - 1) / H,
    ])

    crop = F.grid_sample(
        filtered,
        adjusted_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )

    # Border clamp extends the outer pixels; beyond them, black.
    mask = crop_inside_mask(grid, out_h).to(crop.dtype)
    return crop * mask

class ImageNetViews:
    def __init__(
        self,
        image_size,
        crop_min=0.2,
        crop_max=0.9,
    ):
        self.image_size = int(image_size)
        self.crop_min = float(crop_min)
        self.crop_max = float(crop_max)

    def __call__(self, image):
        source = torch.from_numpy(
            np.array(image.convert("RGB"), copy=True)
        ).permute(2, 0, 1).float().div_(255).unsqueeze(0)

        full = F.interpolate(
            source,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=True,
            antialias=True,
        )[0]

        grid = make_crop_grid_for_size(
            batch_size=1,
            height=self.image_size,
            width=self.image_size,
            device=source.device,
            crop_min=self.crop_min,
            crop_max=self.crop_max,
            output_size=self.image_size,
        )

        crop_real = antialiased_source_crop(source, grid)[0]

        return full, {
            "crop_real": crop_real,
            "crop_grid": grid[0],
        }
