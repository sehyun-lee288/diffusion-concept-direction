"""Unit tests for `diffusion_boundary.multistep` helpers.

We test the math (orthogonalize_against) without requiring any
network access. Hook-based extraction and DDIM denoising are
exercised by the actual scripts on GPU.
"""
from __future__ import annotations

import torch

from diffusion_boundary.multistep import orthogonalize_against


def test_orthogonalize_against_removes_basis_component():
    """After orthogonalization, the inner product with the basis must be zero."""
    torch.manual_seed(0)
    basis = {100: torch.randn(1, 4, 2, 2), 500: torch.randn(1, 4, 2, 2)}
    target = {t: basis[t] * 0.7 + torch.randn_like(basis[t]) * 0.3 for t in basis}
    orth = orthogonalize_against(target, basis)
    for t in basis:
        ip = torch.dot(orth[t].reshape(-1), basis[t].reshape(-1)).item()
        assert abs(ip) < 1e-5, f"t={t}: inner product = {ip}"


def test_orthogonalize_against_keeps_shape():
    """Output tensors must have the same shape as targets."""
    basis = {100: torch.randn(1, 4, 2, 2)}
    target = {100: torch.randn(1, 4, 2, 2)}
    orth = orthogonalize_against(target, basis)
    assert orth[100].shape == target[100].shape


def test_orthogonalize_against_handles_zero_basis():
    """When basis is zero we just keep target as-is (don't divide by zero)."""
    basis = {100: torch.zeros(1, 4, 2, 2)}
    target = {100: torch.randn(1, 4, 2, 2)}
    orth = orthogonalize_against(target, basis)
    # With basis = 0, coefficient is clamped: result ≈ target.
    assert torch.allclose(orth[100], target[100], atol=1e-6)


def test_orthogonalize_against_requires_matching_keys():
    """Different timestep sets must fail loudly."""
    import pytest
    basis = {100: torch.randn(1, 4, 2, 2)}
    target = {200: torch.randn(1, 4, 2, 2)}
    with pytest.raises(ValueError, match="same timesteps"):
        orthogonalize_against(target, basis)
