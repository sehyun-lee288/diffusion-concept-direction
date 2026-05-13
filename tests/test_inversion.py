"""Phase 3 tests for forward diffusion (q-sample).

We don't implement full DDIM inversion in this phase — the simpler q-sample
(x_t = sqrt(αbar)·x_0 + sqrt(1-αbar)·ε) is enough to give each anchor a
deterministic, well-defined x_t and h_t for boundary analysis on the 2D plane.
True DDIM inversion can replace this later as a future ablation.
"""
from __future__ import annotations

import torch
from diffusers import DDIMScheduler

from diffusion_boundary.inversion import noise_to_t


def _make_scheduler() -> DDIMScheduler:
    # Matches the config of `google/ddpm-celebahq-256`. Hard-coding the relevant
    # bits keeps this test independent of network access.
    return DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear")


def test_noise_to_t_deterministic():
    """Same seed must produce bit-identical x_t."""
    sched = _make_scheduler()
    x0 = torch.randn(1, 3, 8, 8)
    x_a = noise_to_t(x0, sched, target_t=500, seed=42)
    x_b = noise_to_t(x0, sched, target_t=500, seed=42)
    assert torch.equal(x_a, x_b)


def test_noise_to_t_different_seeds_differ():
    """Different seeds must produce different x_t (otherwise the noise
    isn't actually being sampled)."""
    sched = _make_scheduler()
    x0 = torch.randn(1, 3, 8, 8)
    x_a = noise_to_t(x0, sched, target_t=500, seed=0)
    x_b = noise_to_t(x0, sched, target_t=500, seed=1)
    assert not torch.equal(x_a, x_b)


def test_noise_to_t_low_t_close_to_x0():
    """At t=1 the added noise weight is tiny, so x_t must stay close to x_0."""
    sched = _make_scheduler()
    x0 = torch.randn(1, 3, 8, 8)
    x_t = noise_to_t(x0, sched, target_t=1, seed=0)
    assert torch.allclose(x_t, x0, atol=0.2)


def test_noise_to_t_high_t_far_from_x0():
    """At t=999 the signal weight ≈ 0, so x_t must look like pure noise (not x_0)."""
    sched = _make_scheduler()
    x0 = torch.ones(1, 3, 8, 8) * 5.0  # large-magnitude, distinctive signal
    x_t = noise_to_t(x0, sched, target_t=999, seed=0)
    # at t=999, sqrt(αbar) ≈ 0, so the signal of x_0 should be heavily attenuated
    assert x_t.mean().abs() < 0.5


def test_noise_to_t_shape_preserved():
    """Output shape matches input shape."""
    sched = _make_scheduler()
    x0 = torch.randn(2, 3, 16, 16)
    x_t = noise_to_t(x0, sched, target_t=300, seed=0)
    assert x_t.shape == x0.shape
