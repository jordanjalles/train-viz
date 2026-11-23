"""What the pool skip keeps and loses for a 16 px patch, for pooled sizes k = 9 and k = 8 (CPU).

    python pool_skip_symmetry.py
"""
import torch
import torch.nn.functional as F

P = 16
torch.manual_seed(0)
x = torch.rand(256, 1, P, P)                                  # one channel is enough: channels never mix
smooth = F.avg_pool2d(F.pad(torch.rand(256, 1, P, P), (2, 2, 2, 2), mode="replicate"), 5, stride=1)


def down_up(x, k):
    return F.interpolate(F.adaptive_avg_pool2d(x, k), size=(P, P), mode="nearest")


for k in (9, 8):
    starts = [(i * P) // k for i in range(k)]
    ends = [-((-(i + 1) * P) // k) for i in range(k)]
    cover = torch.zeros(P)
    for s, e in zip(starts, ends):
        cover[s:e] += 1.0 / (e - s)
    up_idx = [(j * k) // P for j in range(P)]
    y = down_up(x, k)
    flip_err = (down_up(x.flip(-1), k) - y.flip(-1)).abs().mean().item()
    rot_err = (down_up(x.rot90(1, (-2, -1)), k) - y.rot90(1, (-2, -1))).abs().mean().item()
    mean_err = (y.mean((-2, -1)) - x.mean((-2, -1))).abs().mean().item()
    twice = (down_up(y, k) - y).abs().mean().item()
    err_smooth = (down_up(smooth, k) - smooth).abs().mean().item()
    err_noise = (y - x).abs().mean().item()
    checker = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]).repeat(P // 2, P // 2)[None, None]
    print(f"k = {k}: {k * k} values per channel")
    print(f"  pooling windows (start, end): {list(zip(starts, ends))}")
    print(f"  weight each input pixel gets, along one axis: {[round(v, 2) for v in cover.tolist()]}")
    print(f"  output pixel -> pooled cell, along one axis:  {up_idx}")
    print(f"  mirror the patch, then skip, against skip, then mirror: mean difference {flip_err:.4f}")
    print(f"  rotate 90 degrees, same comparison:                     mean difference {rot_err:.4f}")
    print(f"  patch mean changed by {mean_err:.4f} | skip applied twice differs from once by {twice:.4f}")
    print(f"  error on a smooth patch {err_smooth:.4f} | on pure noise {err_noise:.4f} | 1 px checkerboard comes out as {down_up(checker, k).abs().mean().item():.3f} (input 1.0)")
