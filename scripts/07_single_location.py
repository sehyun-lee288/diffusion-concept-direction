"""Phase 6 (ablation): boundaries from sign at a single spatial location.

Instead of `sign(spatial_mean(h_c))` (which mixes 64 spatial neurons into
one line per channel), use `sign(h_c[i*, j*])` for a single fixed pixel.
Now each of the 512 channels contributes the *real* zero-crossing line of
the corresponding mid_block neuron at that spatial location.

We pick the center pixel (4, 4) of the 8×8 mid_block grid by default.

Outputs:
    data/sign_grid_pix{i}_{j}.npy
    figures/exp2_single_location_pix{i}_{j}.png
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

from diffusion_boundary.plane import (  # noqa: E402
    build_plane,
    channel_sign_at_pixel,
    grid_points,
)
from diffusion_boundary.viz import (  # noqa: E402
    active_channel_mask,
    find_boundaries,
    region_ids,
)

ANCHOR_DIR = REPO_ROOT / "data" / "anchors"
FIG_DIR = REPO_ROOT / "figures"
DATA_DIR = REPO_ROOT / "data"

CHANNELS = 512
SPATIAL = 8
PIXEL_I, PIXEL_J = 4, 4   # center of the 8×8 bottleneck grid
N_GRID = 50
ALPHA_RANGE = (-0.5, 1.5)
BETA_RANGE = (-0.5, 1.5)


def _load_h(i: int) -> torch.Tensor:
    return torch.load(ANCHOR_DIR / f"h500_{i}.pt", weights_only=True).float().reshape(-1)


def main() -> None:
    h1, h2, h3 = (_load_h(i) for i in range(3))
    origin, u, v = build_plane(h1, h2, h3)

    alphas = torch.linspace(*ALPHA_RANGE, N_GRID)
    betas = torch.linspace(*BETA_RANGE, N_GRID)
    pts = grid_points(origin, u, v, alphas, betas)  # (N², D)

    # Sign at the chosen pixel for every grid point.
    rows = []
    chunk = 256
    for s in range(0, pts.shape[0], chunk):
        rows.append(channel_sign_at_pixel(
            pts[s:s + chunk], channels=CHANNELS, spatial=SPATIAL,
            pixel_i=PIXEL_I, pixel_j=PIXEL_J,
        ))
    sign_flat = torch.cat(rows, dim=0)  # (N², C)
    sign_grid = sign_flat.reshape(N_GRID, N_GRID, CHANNELS).numpy().astype(np.int8)

    out_npy = DATA_DIR / f"sign_grid_pix{PIXEL_I}_{PIXEL_J}.npy"
    np.save(out_npy, sign_grid)
    print(f"saved {out_npy} {sign_grid.shape}")

    # Use ALL channels active in the window — single-location signs are not
    # collapsed across spatial neurons, so we don't need the balance heuristic.
    active = active_channel_mask(sign_grid)
    sign_active = sign_grid[:, :, active]
    print(f"active channels: {int(active.sum())} / {CHANNELS}")
    bmask = find_boundaries(sign_active)
    ids = region_ids(sign_active)
    print(f"regions: {int(ids.max()) + 1}  boundary cells: {bmask.sum()} / {bmask.size}")

    # Per-channel line equations: A_c·α + B_c·β + C_c = 0 where
    #   A_c = (h2−h1)_c[i*, j*]   B_c = v_c[i*, j*]   C_c = h1_c[i*, j*]
    def pixel_vec(flat):
        return flat.reshape(CHANNELS, SPATIAL, SPATIAL)[:, PIXEL_I, PIXEL_J].numpy()
    A_arr = pixel_vec(u)
    B_arr = pixel_vec(v)
    C_arr = pixel_vec(origin)

    # ---- Figure -----------------------------------------------------------
    n_regions = int(ids.max()) + 1
    base = plt.get_cmap("tab20").colors
    palette = [base[k % len(base)] for k in range(n_regions)]

    fig, (ax_reg, ax_lin) = plt.subplots(1, 2, figsize=(13, 6))

    # Left: region map.
    from matplotlib.colors import ListedColormap
    extent = (float(alphas[0]), float(alphas[-1]), float(betas[0]), float(betas[-1]))
    ax_reg.imshow(ids.T, origin="lower", extent=extent,
                  cmap=ListedColormap(palette), interpolation="nearest")
    by, bx = np.where(bmask)
    a_arr = alphas.numpy()
    b_arr = betas.numpy()
    ax_reg.scatter(a_arr[by], b_arr[bx], s=2, c="black", alpha=0.55, linewidths=0)
    anchor_xy = [(0.0, 0.0), (1.0, 0.0),
                 (float(((h3 - origin) @ u) / (u @ u)),
                  float(((h3 - origin) @ v) / (v @ v)))]
    for k, (a, b) in enumerate(anchor_xy):
        ax_reg.scatter([a], [b], marker="*", s=240, c="white",
                       edgecolors="black", linewidths=1.4)
        ax_reg.annotate(f"h{k + 1}", (a, b), textcoords="offset points",
                        xytext=(7, 5), fontsize=10)
    ax_reg.set_xlabel(r"$\alpha$")
    ax_reg.set_ylabel(r"$\beta$")
    ax_reg.set_title(f"regions  ({n_regions})  —  sign(h_c[{PIXEL_I},{PIXEL_J}])")

    # Right: actual boundary lines for active channels.
    a_line = np.linspace(*ALPHA_RANGE, 200)
    n_drawn = 0
    for c in np.where(active)[0]:
        Ac, Bc, Cc = A_arr[c], B_arr[c], C_arr[c]
        if abs(Bc) > 1e-8:
            b_line = -(Ac * a_line + Cc) / Bc
            mask = (b_line > BETA_RANGE[0] - 0.2) & (b_line < BETA_RANGE[1] + 0.2)
            ax_lin.plot(a_line[mask], b_line[mask],
                        color="steelblue", alpha=0.25, linewidth=0.7)
        else:
            x = -Cc / Ac
            ax_lin.axvline(x, color="steelblue", alpha=0.25, linewidth=0.7)
        n_drawn += 1
    for k, (a, b) in enumerate(anchor_xy):
        ax_lin.scatter([a], [b], marker="*", s=240, c="white",
                       edgecolors="black", linewidths=1.4)
        ax_lin.annotate(f"h{k + 1}", (a, b), textcoords="offset points",
                        xytext=(7, 5), fontsize=10)
    ax_lin.set_xlim(*ALPHA_RANGE)
    ax_lin.set_ylim(*BETA_RANGE)
    ax_lin.set_xlabel(r"$\alpha$")
    ax_lin.set_ylabel(r"$\beta$")
    ax_lin.set_title(f"all {n_drawn} active neuron boundary lines  "
                     f"(pixel ({PIXEL_I},{PIXEL_J}))")

    out_fig = FIG_DIR / f"exp2_single_location_pix{PIXEL_I}_{PIXEL_J}.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=150)
    print(f"saved {out_fig}")


if __name__ == "__main__":
    main()
