"""2D plane construction and channel-wise spatial-mean sign vector.

A point in h-space is represented as a *flat* 1D tensor of dimension
D = C·H·W (e.g., 512·8·8 = 32768 for DDPM CelebA-HQ at t=500). The 2D
plane is spanned by three anchors via Gram–Schmidt; grid points on the
plane are evaluated by sign(spatial_mean(reshape(point))).

Why mean instead of the originally-proposed max: on the actual DDPM
CelebA-HQ bottleneck, every channel has at least one positive spatial
location, so spatial-max collapses to all-+1 (Hamming distance between
anchors = 0/512). spatial-mean preserves 42–45% Hamming distance and is
the empirically-grounded choice. See IMPLEMENTATION_PLAN.md Phase 4.
"""
from __future__ import annotations

import torch


def build_plane(
    h1: torch.Tensor, h2: torch.Tensor, h3: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (origin, u, v) where (u, v) is an *orthogonal* (not unit-norm)
    basis of the plane through h1, h2, h3, calibrated so that the anchors map
    to round (α, β) coordinates:

        h1 → (0, 0),  h2 → (1, 0),  h3 → (a, 1) for some a ∈ ℝ.

    This makes a grid like α, β ∈ [-0.5, 1.5] cover the triangle naturally,
    which is what the visualization wants. The basis is still orthogonal
    (so β has the geometric meaning of "distance along the v-axis"), but is
    not normalized — that's a deliberate trade for usable coordinates.
    """
    if not (h1.shape == h2.shape == h3.shape and h1.dim() == 1):
        raise ValueError(f"expected three 1D tensors of equal shape, got {h1.shape}, {h2.shape}, {h3.shape}")
    origin = h1
    u = h2 - h1
    if u.norm().item() == 0.0:
        raise ValueError("h1 and h2 are identical; cannot form basis")
    # v is the component of (h3 - h1) orthogonal to u, scaled so that
    # projecting h3 yields β = 1 exactly (see project_to_plane below).
    d3 = h3 - h1
    v = d3 - (torch.dot(d3, u) / torch.dot(u, u)) * u
    if v.norm().item() == 0.0:
        raise ValueError("h3 is collinear with h1-h2; cannot form 2D plane")
    return origin, u, v


def project_to_plane(
    point: torch.Tensor, origin: torch.Tensor, u: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (α, β) such that `origin + α·u + β·v` is the orthogonal
    projection of `point` onto the plane. Coefficients use ||u||² / ||v||²
    in the denominator because (u, v) is orthogonal but not unit-norm."""
    d = point - origin
    return torch.dot(d, u) / torch.dot(u, u), torch.dot(d, v) / torch.dot(v, v)


def grid_points(
    origin: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    alphas: torch.Tensor,
    betas: torch.Tensor,
) -> torch.Tensor:
    """Return all (α, β) combinations as a stack of plane points.

    Output shape is (len(alphas) * len(betas), D), ordered so that the
    first axis varies α slowest (i.e., contiguous β within each α).
    """
    A, B = torch.meshgrid(alphas, betas, indexing="ij")
    A = A.reshape(-1, 1)
    B = B.reshape(-1, 1)
    return origin.unsqueeze(0) + A * u.unsqueeze(0) + B * v.unsqueeze(0)


def channel_sign_mean(
    h_flat: torch.Tensor, *, channels: int, spatial: int
) -> torch.Tensor:
    """sign(spatial_mean(h)) per channel.

    Args:
        h_flat: (N, C·H·W) flat batch.
        channels: number of channels C.
        spatial: spatial side length (assumes H == W == spatial).

    Returns:
        int8 tensor of shape (N, C) with values in {-1, 0, +1}.
    """
    if h_flat.dim() != 2:
        raise ValueError(f"expected 2D (N, D) tensor, got shape {tuple(h_flat.shape)}")
    expected_d = channels * spatial * spatial
    if h_flat.shape[1] != expected_d:
        raise ValueError(
            f"flat dim {h_flat.shape[1]} != channels·spatial² = {expected_d}"
        )
    h = h_flat.reshape(-1, channels, spatial * spatial)
    return torch.sign(h.mean(dim=-1)).to(torch.int8)
