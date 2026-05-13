"""Phase 5.5: diagnose the radial boundary pattern in exp1_boundary.png.

For each channel c, the cell-wise sign function on the 2D plane is

    sign( A_c · α + B_c · β + C_c )

where
    A_c = mean_spatial( h2_c - h1_c )
    B_c = mean_spatial(            v_c )
    C_c = mean_spatial(           h1_c ).

The zero set `A_c·α + B_c·β + C_c = 0` is a *line* in (α, β). We see all
top-K boundary lines visually converging near a common point. This script
quantifies why: it computes (A, B, C) for every channel, runs SVD on the
3-column matrix M = [A | B | C] ∈ ℝ^(C × 3), and finds the pairwise
intersection of the top-K lines.

If `rank(M) ≈ 1`, then every row of M is (approximately) a scalar multiple
of a single 3-vector (a, b, c), so every line is the same `a·α + b·β + c = 0`
— a perfect pencil through a single common point. If `rank(M) = 2`, lines
generally do *not* share a point but they live in a 1-parameter family,
which can still look like a near-pencil to the eye.

Output:
    figures/exp1_radial_analysis.png  (lines + intersection cloud)
    stdout: SVD singular values, rank diagnostics, intersection centroid
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from diffusion_boundary.plane import build_plane  # noqa: E402
from diffusion_boundary.viz import top_k_balanced_channels  # noqa: E402

ANCHOR_DIR = REPO_ROOT / "data" / "anchors"
SIGN_GRID = REPO_ROOT / "data" / "sign_grid.npy"
OUT_FIG = REPO_ROOT / "figures" / "exp1_radial_analysis.png"

CHANNELS = 512
SPATIAL = 8
TOP_K = 20


def _load_h(i: int) -> torch.Tensor:
    return torch.load(ANCHOR_DIR / f"h500_{i}.pt", weights_only=True).float().reshape(-1)


def _line_intersection(line1, line2):
    """(A1, B1, C1), (A2, B2, C2) → (α, β) intersection or None if parallel."""
    A1, B1, C1 = line1
    A2, B2, C2 = line2
    det = A1 * B2 - A2 * B1
    if abs(det) < 1e-12:
        return None
    # Solve [A1 B1][α]   [-C1]
    #       [A2 B2][β] = [-C2]
    alpha = (-C1 * B2 + C2 * B1) / det
    beta = (-A1 * (-C2) + A2 * (-C1)) / det  # = (-A1*C2 + A2*C1)/det  ... wait
    # Cleaner: invert directly.
    alpha = (-C1 * B2 + B1 * C2) / det
    beta = (-A1 * C2 + A2 * C1) / det
    return alpha, beta


def main() -> None:
    h1, h2, h3 = (_load_h(i) for i in range(3))
    origin, u, v = build_plane(h1, h2, h3)

    # Reshape back to (C, H, W) and spatial-mean along H, W.
    def channel_mean(flat: torch.Tensor) -> np.ndarray:
        return flat.reshape(CHANNELS, SPATIAL * SPATIAL).mean(dim=-1).numpy()

    A = channel_mean(u)        # m(h2 - h1)
    B = channel_mean(v)        # m(v)
    C = channel_mean(origin)   # m(h1)
    M = np.stack([A, B, C], axis=1)  # (C, 3)
    print(f"M shape: {M.shape}  (one row per channel: [A_c, B_c, C_c])")

    # SVD diagnostic — rank of M tells us the dimensionality of the line family.
    _, S, Vt = np.linalg.svd(M, full_matrices=False)
    print(f"SVD singular values: {S}")
    print(f"  s1/s2 ratio: {S[0] / S[1]:.3f}")
    print(f"  s1/s3 ratio: {S[0] / S[2]:.3f}")
    print(f"  energy share: s1={S[0]**2 / (S**2).sum() * 100:.2f}%, "
          f"s2={S[1]**2 / (S**2).sum() * 100:.2f}%, "
          f"s3={S[2]**2 / (S**2).sum() * 100:.2f}%")

    # If rank ~ 1, the dominant right-singular vector tells us the shared (a,b,c).
    # The "common point" of an exact rank-2 M is the (unique) right null vector,
    # normalized so its third coordinate is 1. With nonzero s3 it isn't unique,
    # but Vt[2] gives the best (least-squares) approximation.
    null_vec = Vt[2]
    if abs(null_vec[2]) > 1e-12:
        common_alpha = null_vec[0] / null_vec[2]
        common_beta = null_vec[1] / null_vec[2]
        # Sign-flip: A·α + B·β + C = 0 means we want -[A, B, C] @ [α, β, 1] = 0.
        # null_vec @ (α, β, 1) = 0 ⇒ null_vec[0]*α + null_vec[1]*β + null_vec[2] = 0
        # ⇒ (α, β) = (-null_vec[0]/null_vec[2], -null_vec[1]/null_vec[2]).
        common_alpha = -null_vec[0] / null_vec[2]
        common_beta = -null_vec[1] / null_vec[2]
        print(f"  rank-2 best common point: (α, β) = "
              f"({common_alpha:.4f}, {common_beta:.4f})")
    else:
        print("  Vt[2] has zero third coord → ideal common point is at infinity")

    # --- Sweep K to separate "real low-rank structure" from "selection bias".
    # Top-K balanced channel selection prefers lines passing through the grid
    # *center*; if convergence is purely a selection artifact, the cluster
    # location should drift toward the model's true structure as K → all.
    sign_grid = np.load(SIGN_GRID)
    print("\n--- K sweep ---")

    def intersections_for_channels(idx_list):
        lines = [(A[c], B[c], C[c]) for c in idx_list]
        pts = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                ip = _line_intersection(lines[i], lines[j])
                if ip is not None:
                    pts.append(ip)
        return np.array(pts)

    pts_topk_main = None
    for k in (10, 20, 50, 100, 250, 512):
        idx = top_k_balanced_channels(sign_grid, k=k) if k < 512 else np.arange(CHANNELS)
        pts = intersections_for_channels(idx)
        # Filter out far-away outliers (parallel-near lines yield huge values).
        m = np.median(pts, axis=0)
        keep = np.linalg.norm(pts - m, axis=1) < 5.0
        pts_in = pts[keep]
        med = np.median(pts_in, axis=0)
        mad = np.median(np.abs(pts_in - med), axis=0)
        print(f"  K={k:>3}  n_pairs={len(pts):>5}  inliers={keep.sum():>5}  "
              f"median=({med[0]:+.4f}, {med[1]:+.4f})  MAD=({mad[0]:.4f}, {mad[1]:.4f})")
        if k == TOP_K:
            pts_topk_main = pts
    pts = pts_topk_main

    # ---- Figure: top-K boundary lines on the plane + intersection cloud ----
    topk_idx = top_k_balanced_channels(sign_grid, k=TOP_K)
    lines = [(A[c], B[c], C[c]) for c in topk_idx]
    fig, ax = plt.subplots(figsize=(7, 6.5))
    a_grid = np.linspace(-0.5, 1.5, 200)
    for (Ac, Bc, Cc) in lines:
        if abs(Bc) > 1e-8:
            b_line = -(Ac * a_grid + Cc) / Bc
            mask = (b_line > -0.7) & (b_line < 1.7)
            ax.plot(a_grid[mask], b_line[mask], color="steelblue", alpha=0.55, linewidth=1.0)
        else:
            x = -Cc / Ac
            ax.axvline(x, color="steelblue", alpha=0.55, linewidth=1.0)

    if pts is not None and len(pts) > 0:
        mask = (pts[:, 0] > -1) & (pts[:, 0] < 2) & (pts[:, 1] > -1) & (pts[:, 1] < 2)
        ax.scatter(pts[mask, 0], pts[mask, 1], s=12, c="crimson", alpha=0.65, label="line ∩ line")

    # Anchor markers — h3 coordinates computed via projection.
    a3 = float(((h3 - origin) @ u) / (u @ u))
    b3 = float(((h3 - origin) @ v) / (v @ v))
    anchor_xy = [(0.0, 0.0), (1.0, 0.0), (a3, b3)]
    for k, (a, b) in enumerate(anchor_xy):
        ax.scatter([a], [b], marker="*", s=220, c="white", edgecolors="black", linewidths=1.4)
        ax.annotate(f"h{k + 1}", (a, b), textcoords="offset points", xytext=(7, 5), fontsize=10)
    print(f"anchor coords on plane: h1=(0, 0)  h2=(1, 0)  h3=({a3:.3f}, {b3:.3f})")
    print(f"triangle centroid: ({(1 + a3) / 3:.3f}, {b3 / 3:.3f})")

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta$")
    ax.set_title(f"Top-{TOP_K} boundary lines + pairwise intersections")
    ax.legend(loc="upper right")
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150)
    print(f"saved {OUT_FIG}")


if __name__ == "__main__":
    main()
