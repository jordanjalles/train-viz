# Which crops does training use? Stop the training loop first.
import inspect

import torch

N = IMAGE_SIZE
print("1. crop cell applied (crop_inside_mask defined):",
      "crop_inside_mask" in globals())

# 2. a fresh dataset batch: do crops overhang, and is the
#    off-image part black? (uses the data cell as it is now)
imgs, ex = next(iter(train_dataloader))
grid, real = ex["crop_grid"], ex["crop_real"]
lim = 1 + 1 / (N - 1)
off = (grid.abs() > lim).any(-1)[:, None].expand_as(real)
print(f"2. dataset crops overhang: {off.any().item()}"
      f" ({off.float().mean():.1%} of pixels off image)")
if off.any():
    print("   off-image pixels black:",
          real[off].abs().max().item() == 0)

# 3. reconstruction crops through the train step's crop_scale
#    (train_step looks crop_scale up by name when it runs)
dev = next(encoder.parameters()).device
ones = torch.ones(4, 3, N, N, device=dev)
rc = crop_scale(ones, grid[:4].to(dev))
o4 = off[:4].to(dev)
if o4.any():
    print("3. recon crops black off image:",
          rc[o4].abs().max().item() == 0)
else:
    print("3. no overhang in this batch: recon mask not tested")

# 4. does the judge copy the encoder you are training?
fn = getattr(train_step, "_torchdynamo_orig_callable", train_step)
pairs = inspect.getclosurevars(fn).nonlocals["state_pairs"]
src = getattr(encoder, "_orig_mod", encoder)
print("4. judge copies the current encoder:",
      pairs[0][1] is next(src.parameters()))
