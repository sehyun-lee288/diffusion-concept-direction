"""Phase 7 (ablation): boundaries for a fixed channel, varying spatial (i, j).

For each channel c, the bottleneck has 64 spatial neurons. We fix c and
draw one line per pixel (i, j):

    h1_c[i,j] + α·(h2-h1)_c[i,j] + β·v_c[i,j] = 0      → line on (α, β)

Same conv filter, different spatial reading. Useful for asking: how
spatially coherent is a single feature across the plane?

We auto-select the top-4 "most diverse" channels: those producing the
largest number of distinct per-pixel sign vectors across the grid. These
are the channels whose spatial activity actually varies as we move
across (α, β).

Output: figures/exp3_single_channel.png  (4 panels, one channel each).
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
from matplotlib.colors import ListedColormap  # noqa: E402

from diffusion_boundary.plane import (  # noqa: E402
    build_plane,
    grid_points,
    pixel_signs_for_channel,
)
from diffusion_boundary.viz import (  # noqa: E402
    active_channel_mask,
    find_boundaries,
    region_ids,
)

ANCHOR_DIR = REPO_ROOT / "data" / "anchors"
OUT_FIG = REPO_ROOT / "figures" / "exp3_single_channel.png"

CHANNELS = 512
SPATIAL = 8
N_GRID = 50
ALPHA_RANGE = (-0.5, 1.5)
BETA_RANGE = (-0.5, 1.5)
N_PANELS = 4


def _load_h(i: int) -> torch.Tensor:
    return torch.load(ANCHOR_DIR / f"h500_{i}.pt", weights_only=True).float().reshape(-1)


def main() -> None:
    h1, h2, h3 = (_load_h(i) for i in range(3))
    origin, u, v = build_plane(h1, h2, h3)
    alphas = torch.linspace(*ALPHA_RANGE, N_GRID)
    betas = torch.linspace(*BETA_RANGE, N_GRID)
    pts = grid_points(origin, u, v, alphas, betas)  # (N², D)
    n_cells = pts.shape[0]

    # Score channels by # unique per-pixel sign patterns across the grid.
    print("scoring channels by spatial-pattern diversity ...")
    scores = np.zeros(CHANNELS, dtype=np.int64)
    chunk = 256
    for c in range(CHANNELS):
        sign = torch.empty(n_cells, SPATIAL * SPATIAL, dtype=torch.int8)
        for s in range(0, n_cells, chunk):
            sign[s:s + chunk] = pixel_signs_for_channel(
                pts[s:s + chunk], channel=c, channels=CHANNELS, spatial=SPATIAL,
            )
        scores[c] = len(np.unique(sign.numpy(), axis=0))
    top_channels = np.argsort(-scores)[:N_PANELS]
    print(f"top-{N_PANELS} channels by diversity: {top_channels.tolist()}")
    print(f"  unique pattern counts: {scores[top_channels].tolist()}")

    # Pre-compute spatial coordinates (i, j) → (A_c[i,j], B_c[i,j], C_c[i,j]).
    def per_pixel_coeffs(flat: torch.Tensor, c: int) -> np.ndarray:
        return flat.reshape(CHANNELS, SPATIAL, SPATIAL)[c].numpy().reshape(-1)
    fig, axes = plt.subplots(2, N_PANELS, figsize=(4.0 * N_PANELS, 8))
    a_line = np.linspace(*ALPHA_RANGE, 200)
    extent = (float(alphas[0]), float(alphas[-1]), float(betas[0]), float(betas[-1]))
    anchor_xy = [(0.0, 0.0), (1.0, 0.0),
                 (float(((h3 - origin) @ u) / (u @ u)),
                  float(((h3 - origin) @ v) / (v @ v)))]

    for col, c in enumerate(top_channels):
        # Build sign_grid (N, N, 64) and region map.
        sign_2d = torch.empty(n_cells, SPATIAL * SPATIAL, dtype=torch.int8)
        for s in range(0, n_cells, chunk):
            sign_2d[s:s + chunk] = pixel_signs_for_channel(
                pts[s:s + chunk], channel=int(c), channels=CHANNELS, spatial=SPATIAL,
            )
        sign_grid = sign_2d.reshape(N_GRID, N_GRID, SPATIAL * SPATIAL).numpy()
        active = active_channel_mask(sign_grid)
        sign_active = sign_grid[:, :, active]
        bmask = find_boundaries(sign_active)
        ids = region_ids(sign_active)
        n_regions = int(ids.max()) + 1

        # Top row: region map.
        ax = axes[0, col]
        base = plt.get_cmap("tab20").colors
        palette = [base[k % len(base)] for k in range(n_regions)]
        ax.imshow(ids.T, origin="lower", extent=extent,
                  cmap=ListedColormap(palette), interpolation="nearest")
        by, bx = np.where(bmask)
        ax.scatter(alphas.numpy()[by], betas.numpy()[bx],
                   s=2, c="black", alpha=0.55, linewidths=0)
        for a, b in anchor_xy:
            ax.scatter([a], [b], marker="*", s=140, c="white",
                       edgecolors="black", linewidths=1.2)
        ax.set_title(f"channel {int(c)}  ({n_regions} regions)", fontsize=10)
        ax.set_xlabel(r"$\alpha$")
        if col == 0:
            ax.set_ylabel(r"$\beta$")

        # Bottom row: 64 spatial boundary lines for this channel.
        A_pp = per_pixel_coeffs(u, int(c))
        B_pp = per_pixel_coeffs(v, int(c))
        C_pp = per_pixel_coeffs(origin, int(c))
        ax_lin = axes[1, col]
        n_drawn = 0
        for p in range(SPATIAL * SPATIAL):
            if not active[p]:
                continue
            Ac, Bc, Cc = A_pp[p], B_pp[p], C_pp[p]
            if abs(Bc) > 1e-8:
                b_line = -(Ac * a_line + Cc) / Bc
                mask = (b_line > BETA_RANGE[0] - 0.2) & (b_line < BETA_RANGE[1] + 0.2)
                ax_lin.plot(a_line[mask], b_line[mask],
                            color="steelblue", alpha=0.65, linewidth=1.0)
            else:
                ax_lin.axvline(-Cc / Ac, color="steelblue", alpha=0.65, linewidth=1.0)
            n_drawn += 1
        for a, b in anchor_xy:
            ax_lin.scatter([a], [b], marker="*", s=140, c="white",
                           edgecolors="black", linewidths=1.2)
        ax_lin.set_xlim(*ALPHA_RANGE)
        ax_lin.set_ylim(*BETA_RANGE)
        ax_lin.set_title(f"{n_drawn} / 64 active pixels", fontsize=10)
        ax_lin.set_xlabel(r"$\alpha$")
        if col == 0:
            ax_lin.set_ylabel(r"$\beta$")

    fig.suptitle("Single-channel boundary lines across spatial positions", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150)
    print(f"saved {OUT_FIG}")


if __name__ == "__main__":
    main()
