"""Phase 14: sign-pattern boundary on the attribute-paired plane.

Plane basis is now attribute-meaningful (Phase 13):
  h(α, β) = h_anchor + α · Δh_smile_orth + β · Δh_gender   (all at t=500)

Per-channel sign(spatial_mean(h_c(α, β))) flips on a line in (α, β).
If attribute axes truly carry concept meaning, sign-pattern regions
should now align with attribute combinations, not be random shards
the way they were on the random-anchor plane (Phase 4-7).

Output:
  figures/exp10_attribute_boundary.png — three-panel
    L) full region map (all active channels)
    M) top-K=20 most balanced channels region map
    R) overlay of top-K boundaries on the same axes as exp9
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

from diffusion_boundary.viz import (  # noqa: E402
    active_channel_mask,
    find_boundaries,
    region_ids,
    top_k_balanced_channels,
)

T = 500
N_GRID = 50
ALPHA_RANGE = (-2.5, 2.5)
BETA_RANGE = (-2.5, 2.5)
TOP_K = 20
CHANNELS = 512

ANCHOR_DIR = REPO_ROOT / "data" / "anchors"
MULTIATTR = REPO_ROOT / "data" / "delta_h_multiattr.pt"
ORTH = REPO_ROOT / "data" / "delta_h_smile_orth.pt"
FIG_OUT = REPO_ROOT / "figures" / "exp10_attribute_boundary.png"


def _spatial_mean(t: torch.Tensor) -> np.ndarray:
    """(1, C, H, W) → (C,) numpy."""
    return t.reshape(CHANNELS, -1).mean(-1).numpy()


def main() -> None:
    h_anchor = torch.load(ANCHOR_DIR / "h500_0.pt", weights_only=True).float()
    attr = torch.load(MULTIATTR, weights_only=False)
    smile_orth_dict = torch.load(ORTH, weights_only=False)
    # multistep extraction uses t in {50, 150, ..., 950} — pick the closest
    # available key to the analysis timestep T.
    key = min(smile_orth_dict.keys(), key=lambda k: abs(k - T))
    if key != T:
        print(f"using nearest cached t={key} for analysis (requested T={T})")
    delta_smile = smile_orth_dict[key].float()
    delta_gender = attr["gender"][key].float()

    A = _spatial_mean(delta_smile)   # (512,) — smile_orth loading
    B = _spatial_mean(delta_gender)  # (512,) — gender loading
    C = _spatial_mean(h_anchor)      # (512,) — baseline offset

    alphas = np.linspace(*ALPHA_RANGE, N_GRID)
    betas = np.linspace(*BETA_RANGE, N_GRID)
    AA, BB = np.meshgrid(alphas, betas, indexing="xy")  # (N, N)
    # broadcast → (N, N, C)
    sign_grid = np.sign(
        A[None, None, :] * AA[..., None]
        + B[None, None, :] * BB[..., None]
        + C[None, None, :]
    ).astype(np.int8)
    print(f"sign_grid shape: {sign_grid.shape}")

    active = active_channel_mask(sign_grid)
    sign_active = sign_grid[:, :, active]
    ids_full = region_ids(sign_active)
    bmask_full = find_boundaries(sign_active)
    n_active = int(active.sum())
    n_regions_full = int(ids_full.max()) + 1
    print(f"active channels: {n_active}  full regions: {n_regions_full}")

    top_idx = top_k_balanced_channels(sign_grid, k=TOP_K)
    sign_top = sign_grid[:, :, top_idx]
    ids_top = region_ids(sign_top)
    bmask_top = find_boundaries(sign_top)
    n_regions_top = int(ids_top.max()) + 1
    print(f"top-K={TOP_K} regions: {n_regions_top}")

    # ---- Figure --------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    extent = (ALPHA_RANGE[0], ALPHA_RANGE[1], BETA_RANGE[0], BETA_RANGE[1])
    base = plt.get_cmap("tab20").colors

    # L) Full region map (n_active channels)
    pal_full = [base[k % len(base)] for k in range(n_regions_full)]
    axes[0].imshow(ids_full.T, origin="lower", extent=extent,
                   cmap=ListedColormap(pal_full), interpolation="nearest")
    by, bx = np.where(bmask_full)
    axes[0].scatter(alphas[by], betas[bx], s=1.5, c="black", alpha=0.5, linewidths=0)
    axes[0].axhline(0, color="white", linewidth=0.7, alpha=0.6)
    axes[0].axvline(0, color="white", linewidth=0.7, alpha=0.6)
    axes[0].set_title(f"all {n_active} active channels\n({n_regions_full} regions)",
                      fontsize=11)
    axes[0].set_xlabel(r"$\alpha$  (smile ⊥ gender)")
    axes[0].set_ylabel(r"$\beta$  (gender)")

    # M) Top-K region map
    pal_top = [base[k % len(base)] for k in range(n_regions_top)]
    axes[1].imshow(ids_top.T, origin="lower", extent=extent,
                   cmap=ListedColormap(pal_top), interpolation="nearest")
    by, bx = np.where(bmask_top)
    axes[1].scatter(alphas[by], betas[bx], s=2, c="black", alpha=0.6, linewidths=0)
    axes[1].axhline(0, color="white", linewidth=0.7, alpha=0.6)
    axes[1].axvline(0, color="white", linewidth=0.7, alpha=0.6)
    axes[1].set_title(f"top-{TOP_K} balanced channels\n({n_regions_top} regions)",
                      fontsize=11)
    axes[1].set_xlabel(r"$\alpha$")

    # R) The 20 boundary lines themselves
    a_line = np.linspace(*ALPHA_RANGE, 200)
    for c in top_idx:
        Ac, Bc, Cc = A[c], B[c], C[c]
        if abs(Bc) > 1e-8:
            b_line = -(Ac * a_line + Cc) / Bc
            mask = (b_line >= BETA_RANGE[0] - 0.2) & (b_line <= BETA_RANGE[1] + 0.2)
            axes[2].plot(a_line[mask], b_line[mask],
                         color="steelblue", alpha=0.55, linewidth=1.0)
        else:
            axes[2].axvline(-Cc / Ac, color="steelblue", alpha=0.55, linewidth=1.0)
    axes[2].axhline(0, color="black", linewidth=0.6, alpha=0.6)
    axes[2].axvline(0, color="black", linewidth=0.6, alpha=0.6)
    axes[2].set_xlim(*ALPHA_RANGE)
    axes[2].set_ylim(*BETA_RANGE)
    axes[2].set_title(f"top-{TOP_K} boundary lines", fontsize=11)
    axes[2].set_xlabel(r"$\alpha$")
    axes[2].set_aspect("equal")

    fig.suptitle("Phase 14 — sign-pattern boundary on the attribute-paired plane (t=500)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150)
    print(f"saved {FIG_OUT}")


if __name__ == "__main__":
    main()
