"""Phase 4 tests for the 2D plane + channel-wise spatial-max sign vector."""
from __future__ import annotations

import torch

from diffusion_boundary.plane import (
    build_plane,
    channel_sign_mean,
    grid_points,
    project_to_plane,
)


def test_build_plane_orthogonal():
    torch.manual_seed(0)
    h1, h2, h3 = (torch.randn(32) for _ in range(3))
    origin, u, v = build_plane(h1, h2, h3)
    assert abs(torch.dot(u, v).item()) < 1e-4


def test_anchors_project_to_round_coordinates():
    """h1 → (0, 0), h2 → (1, 0), h3 → (a, 1) for some a."""
    torch.manual_seed(0)
    h1, h2, h3 = (torch.randn(32) for _ in range(3))
    origin, u, v = build_plane(h1, h2, h3)

    a1, b1 = project_to_plane(h1, origin, u, v)
    a2, b2 = project_to_plane(h2, origin, u, v)
    _, b3 = project_to_plane(h3, origin, u, v)

    assert abs(a1.item()) < 1e-5 and abs(b1.item()) < 1e-5
    assert abs(a2.item() - 1.0) < 1e-5
    assert abs(b2.item()) < 1e-5
    assert abs(b3.item() - 1.0) < 1e-5


def test_grid_points_shape_and_corners():
    """grid_points must produce an N*M × D tensor and recover the corners exactly."""
    origin = torch.zeros(4)
    u = torch.tensor([1.0, 0.0, 0.0, 0.0])
    v = torch.tensor([0.0, 1.0, 0.0, 0.0])
    alphas = torch.linspace(0.0, 1.0, 3)  # 3 values
    betas = torch.linspace(-1.0, 1.0, 5)  # 5 values
    pts = grid_points(origin, u, v, alphas, betas)
    assert pts.shape == (3 * 5, 4)
    # (alpha=0, beta=-1) → origin + 0*u + (-1)*v = (0, -1, 0, 0)
    assert torch.allclose(pts[0], torch.tensor([0.0, -1.0, 0.0, 0.0]))


def test_channel_sign_at_pixel_picks_one_location():
    """sign at (i, j) reads exactly that pixel — no spatial averaging."""
    from diffusion_boundary.plane import channel_sign_at_pixel
    # (1, 2, 2, 2): channel 0 has +3 at (0,0), -1 elsewhere; channel 1 mirror.
    h = torch.tensor([[
        [[3.0, -1.0], [-1.0, -1.0]],   # at (0,0): +3 → +1; at (1,1): -1 → -1
        [[-1.0, -1.0], [-1.0, 3.0]],   # at (0,0): -1 → -1; at (1,1): +3 → +1
    ]])
    h_flat = h.reshape(1, -1)
    s_00 = channel_sign_at_pixel(h_flat, channels=2, spatial=2, pixel_i=0, pixel_j=0)
    s_11 = channel_sign_at_pixel(h_flat, channels=2, spatial=2, pixel_i=1, pixel_j=1)
    assert s_00.tolist() == [[1, -1]]
    assert s_11.tolist() == [[-1, 1]]


def test_channel_sign_mean_shape_and_values():
    """channel_sign_mean reshapes flat → (C,H,W), spatial-mean, then sign."""
    # Hand-crafted (1, 2, 2, 2): channel 0 has mean +1, channel 1 mean -1.5.
    h = torch.tensor([[
        [[2.0, 0.0], [2.0, 0.0]],       # mean = 1.0 → +1
        [[-1.0, -2.0], [-2.0, -1.0]],   # mean = -1.5 → -1
    ]])
    h_flat = h.reshape(1, -1)  # (1, 8)
    s = channel_sign_mean(h_flat, channels=2, spatial=2)
    assert s.shape == (1, 2)
    assert s[0, 0].item() == 1
    assert s[0, 1].item() == -1


def test_real_anchor_sign_vectors_differ():
    """If three anchors collapse to the same sign vector the plane is useless."""
    from pathlib import Path

    anchor_dir = Path(__file__).resolve().parents[1] / "data" / "anchors"
    if not anchor_dir.exists():
        import pytest
        pytest.skip("anchors not prepared; run scripts/03_invert_anchors.py")

    hs_flat = [
        torch.load(anchor_dir / f"h500_{i}.pt", weights_only=True).reshape(1, -1).float()
        for i in range(3)
    ]
    signs = [channel_sign_mean(h, channels=512, spatial=8) for h in hs_flat]
    for i in range(3):
        for j in range(i + 1, 3):
            hamming = (signs[i] != signs[j]).sum().item()
            assert hamming > 0, f"anchors {i} and {j} share an identical sign pattern"
