"""DDIM decoding from an injected mid_block feature.

`decode_from_h` overrides the U-Net mid_block output with a target h at
t = target_t, then DDIM-denoises to t = 0 with a fixed encoder input.

`decode_plane_grid` is the batch helper: given a plane (origin, u, v)
and arrays of (α, β) values, decode every grid point.

Anchor 0's x_500 is the conventional fixed encoder input — see
FINDINGS.md §6 for why this matters (encoder skips carry most of the
image, so h-injection alone gives subtle variation).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def decode_from_h(
    unet: nn.Module,
    alphas_cumprod: torch.Tensor,
    x_input: torch.Tensor,
    h_target: torch.Tensor,
    *,
    target_t: int = 500,
    step_size: int = 50,
) -> torch.Tensor:
    """Inject `h_target` at the mid_block during the t=target_t step and
    DDIM-denoise from t=target_t to t=0.

    Returns x_0 in [-1, 1] (not clipped — caller may clamp).
    """
    device = x_input.device

    def replace_with_h(_module, _inputs, output):
        if isinstance(output, tuple):
            return (h_target,) + output[1:]
        return h_target

    x = x_input.clone()
    t_curr = target_t
    is_first = True
    while t_curr >= 0:
        t_next = max(0, t_curr - step_size)
        handle = unet.mid_block.register_forward_hook(replace_with_h) if is_first else None
        try:
            with torch.no_grad():
                eps = unet(x, torch.tensor([t_curr], device=device)).sample
        finally:
            if handle is not None:
                handle.remove()
        is_first = False
        alpha_t = alphas_cumprod[t_curr]
        alpha_n = alphas_cumprod[t_next] if t_next > 0 else torch.tensor(1.0, device=device)
        pred_x0 = (x - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
        x = torch.sqrt(alpha_n) * pred_x0 + torch.sqrt(1 - alpha_n) * eps
        if t_next == 0:
            return x
        t_curr = t_next
    return x  # unreachable


def to_uint8_image(x: torch.Tensor) -> np.ndarray:
    """(1, 3, H, W) tensor in [-1, 1] → (H, W, 3) uint8 array."""
    x = x.detach().cpu().clamp(-1, 1)
    x = (x + 1.0) / 2.0
    return (x[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def decode_plane_grid(
    unet: nn.Module,
    alphas_cumprod: torch.Tensor,
    x_input: torch.Tensor,
    origin: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    alphas: list[float],
    betas: list[float],
    channels: int,
    spatial: int,
    target_t: int = 500,
    step_size: int = 50,
    verbose: bool = False,
) -> list[tuple[float, float, np.ndarray]]:
    """Decode every (α, β) on the cartesian product of the two lists.

    Returns a list of (α, β, image_uint8) tuples.
    """
    out = []
    total = len(alphas) * len(betas)
    k = 0
    for b in betas:
        for a in alphas:
            h = (origin + a * u + b * v).reshape(1, channels, spatial, spatial)
            x0 = decode_from_h(
                unet, alphas_cumprod, x_input, h,
                target_t=target_t, step_size=step_size,
            )
            out.append((float(a), float(b), to_uint8_image(x0)))
            k += 1
            if verbose:
                print(f"  decoded {k}/{total}  (α={a:+.2f}, β={b:+.2f})")
    return out
