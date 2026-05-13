"""Forward diffusion (q-sample) for anchor preparation.

For Toy Experiment 1 we don't need full DDIM inversion: we just need a
deterministic mapping from a clean image x_0 to a noised x_t at a fixed
timestep t. The standard q-sample

    x_t = sqrt(αbar_t) · x_0 + sqrt(1 - αbar_t) · ε,   ε ~ N(0, I)

does exactly that. With a fixed seed it is fully reproducible. True DDIM
inversion (ε-prediction-based, self-consistent) is a future improvement that
can drop in here without changing downstream Phase 4/5 code.
"""
from __future__ import annotations

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

SchedulerLike = DDIMScheduler | DDPMScheduler


def noise_to_t(
    x0: torch.Tensor,
    scheduler: SchedulerLike,
    target_t: int,
    seed: int,
) -> torch.Tensor:
    """Return x_t = q-sample(x_0, ε, target_t) with deterministic ε.

    Args:
        x0: clean sample, shape (B, C, H, W). Must be on the device we want
            the result on.
        scheduler: a diffusers scheduler exposing `alphas_cumprod`.
        target_t: integer timestep in [0, num_train_timesteps).
        seed: RNG seed for ε.

    Returns:
        x_t with the same shape and device as x_0.
    """
    if not (0 <= target_t < scheduler.config.num_train_timesteps):
        raise ValueError(
            f"target_t={target_t} out of range [0, {scheduler.config.num_train_timesteps})"
        )
    device = x0.device
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    alpha_bar = alphas_cumprod[target_t]

    # Use a CPU generator + manual_seed for cross-device reproducibility,
    # then move ε to the target device. This avoids device-dependent RNG
    # state surprises.
    generator = torch.Generator().manual_seed(seed)
    eps = torch.randn(x0.shape, generator=generator).to(device=device, dtype=x0.dtype)

    return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * eps
