"""Phase 9: per-channel statistics of the 64 boundary-line angles.

For each channel c, the 64 spatial pixels give 64 lines on (α, β):
each line has normal vector `(A_c[i,j], B_c[i,j])` (with
`C_c[i,j]` only setting offset). The *direction* of a line is defined
mod π, so we work with the doubled-angle representation:

    θ_c[i,j]  = atan2(B_c[i,j], A_c[i,j])  mod π
    z_c[i,j]  = exp(i · 2θ)               on the unit circle

Channel-level summary statistics:
    mean direction:  μ_c   = arg(mean(z_c)) / 2
    concentration:   R_c   = |mean(z_c)|     ∈ [0, 1]
        R_c → 1: all 64 lines parallel
        R_c → 0: directions uniformly distributed

We then visualize:
  - histogram of R_c across all 512 channels
  - rose plots for the 4 most- and 4 least-concentrated channels
  - actual boundary-line plot for 2 channels at each extreme (sanity)

Output: figures/exp5_channel_angle_stats.png
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

ANCHOR_DIR = REPO_ROOT / "data" / "anchors"
OUT_FIG = REPO_ROOT / "figures" / "exp5_channel_angle_stats.png"

CHANNELS = 512
SPATIAL = 8


def _load_h(i: int) -> torch.Tensor:
    return torch.load(ANCHOR_DIR / f"h500_{i}.pt", weights_only=True).float().reshape(-1)


def main() -> None:
    h1, h2, h3 = (_load_h(i) for i in range(3))
    origin, u, v = build_plane(h1, h2, h3)

    # (C, H, W) per-pixel coefficients.
    A = u.reshape(CHANNELS, SPATIAL * SPATIAL).numpy()   # (C, 64)
    B = v.reshape(CHANNELS, SPATIAL * SPATIAL).numpy()   # (C, 64)
    norms = np.sqrt(A ** 2 + B ** 2)
    valid = norms > 1e-8         # exclude degenerate lines

    theta = np.arctan2(B, A)     # (C, 64) ∈ (-π, π]
    theta = np.mod(theta, np.pi) # line direction is mod π → (-0, π)

    # Circular stats on the doubled-angle representation.
    z = np.exp(1j * 2 * theta)
    z_mean = np.where(valid, z, 0).sum(axis=1) / valid.sum(axis=1).clip(min=1)
    R = np.abs(z_mean)                                         # (C,)
    mu = (np.angle(z_mean) / 2) % np.pi                        # (C,) ∈ [0, π)

    # Sort by concentration.
    order = np.argsort(-R)
    top4 = order[:4]
    bot4 = order[-4:]
    print(f"R distribution: min={R.min():.3f} median={np.median(R):.3f} max={R.max():.3f}")
    print("top-4 most concentrated (R, μ in deg):")
    for c in top4:
        print(f"  ch {int(c):3d}  R={R[c]:.3f}  μ={np.degrees(mu[c]):6.1f}°")
    print("bot-4 least concentrated:")
    for c in bot4:
        print(f"  ch {int(c):3d}  R={R[c]:.3f}  μ={np.degrees(mu[c]):6.1f}°")

    # ---- Figure -----------------------------------------------------------
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.2, 1.2])

    # Row 0 (full width): R histogram.
    ax0 = fig.add_subplot(gs[0, :])
    ax0.hist(R, bins=40, color="steelblue", edgecolor="white")
    for c, color, label in [
        (top4[0], "crimson", f"max (ch {int(top4[0])})"),
        (bot4[-1], "darkgreen", f"min (ch {int(bot4[-1])})"),
    ]:
        ax0.axvline(R[c], color=color, linestyle="--", linewidth=1.3, label=label)
    ax0.set_xlabel("circular concentration R per channel")
    ax0.set_ylabel("# channels")
    ax0.set_title("Per-channel directional bias  R = |⟨exp(2iθ)⟩|  (1 = all parallel, 0 = uniform)")
    ax0.legend(loc="upper right")

    # Row 1: rose plots for top-4 most-concentrated (high R).
    bins = 18  # 10° doubled-angle bins
    for col, c in enumerate(top4):
        ax = fig.add_subplot(gs[1, col], projection="polar")
        t = theta[c, valid[c]]
        # Plot in doubled-angle for symmetry, but display in θ for readability.
        hist, edges = np.histogram(t, bins=bins, range=(0, np.pi))
        widths = np.diff(edges)
        # Mirror so the rose covers full 2π with mod-π symmetry.
        ax.bar(edges[:-1], hist, width=widths, bottom=0, color="crimson", alpha=0.8)
        ax.bar(edges[:-1] + np.pi, hist, width=widths, bottom=0, color="crimson", alpha=0.4)
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_title(f"ch {int(c)}  R={R[c]:.2f}\nμ={np.degrees(mu[c]):.0f}°",
                     fontsize=9, pad=8)
        ax.set_yticklabels([])

    # Row 2: rose plots for bottom-4 least-concentrated.
    for col, c in enumerate(bot4):
        ax = fig.add_subplot(gs[2, col], projection="polar")
        t = theta[c, valid[c]]
        hist, edges = np.histogram(t, bins=bins, range=(0, np.pi))
        widths = np.diff(edges)
        ax.bar(edges[:-1], hist, width=widths, bottom=0, color="darkgreen", alpha=0.8)
        ax.bar(edges[:-1] + np.pi, hist, width=widths, bottom=0, color="darkgreen", alpha=0.4)
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_title(f"ch {int(c)}  R={R[c]:.2f}\nμ={np.degrees(mu[c]):.0f}°",
                     fontsize=9, pad=8)
        ax.set_yticklabels([])

    fig.suptitle("Boundary-line orientation by channel  (rose plots, doubled-angle symmetrized)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150)
    print(f"saved {OUT_FIG}")


if __name__ == "__main__":
    main()
