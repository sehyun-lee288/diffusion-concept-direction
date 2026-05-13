"""Phase 8: dense (5×5) thumbnail grid overlaid on the boundary plane.

Renders a two-panel figure:
  L) region map from top-K=20 balanced channels (same as exp1)
     + the 5×5 thumbnail grid laid out at the actual (α, β) positions
  R) the same thumbnail grid alone (no region underlay) for a clean
     visual comparison

The point of the dense thumbnail grid is to spot *visible* attribute
boundaries — places where neighboring decoded images change identity,
expression, lighting, etc. — and compare them with the sign-pattern
boundaries from the region map.

Output: figures/exp4_plane_thumbnails.png
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
import yaml  # noqa: E402
from diffusers import DDIMScheduler, DDPMPipeline  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from diffusion_boundary.decoding import decode_plane_grid  # noqa: E402
from diffusion_boundary.plane import build_plane  # noqa: E402
from diffusion_boundary.viz import (  # noqa: E402
    find_boundaries,
    overlay_thumbnails,
    region_ids,
    top_k_balanced_channels,
)

ANCHOR_DIR = REPO_ROOT / "data" / "anchors"
SIGN_GRID = REPO_ROOT / "data" / "sign_grid.npy"
GRID_META = REPO_ROOT / "data" / "grid_meta.yaml"
OUT_FIG = REPO_ROOT / "figures" / "exp4_plane_thumbnails.png"

MODEL_ID = "google/ddpm-celebahq-256"
CHANNELS = 512
SPATIAL = 8
N_THUMB = 5      # 5x5 = 25 decodings (~25s on H100)
TOP_K = 20


def _load_h_flat(i: int) -> torch.Tensor:
    return torch.load(ANCHOR_DIR / f"h500_{i}.pt", weights_only=True).float().reshape(-1)


def main() -> None:
    sign_grid = np.load(SIGN_GRID)
    meta = yaml.safe_load(GRID_META.read_text())
    n_grid = meta["n_grid"]
    alphas_full = np.linspace(*meta["alpha_range"], n_grid)
    betas_full = np.linspace(*meta["beta_range"], n_grid)
    anchors_ab = [(a["alpha"], a["beta"]) for a in meta["anchors_alpha_beta"]]

    # Region map from top-K balanced channels (same as exp1).
    top_idx = top_k_balanced_channels(sign_grid, k=TOP_K)
    sign_top = sign_grid[:, :, top_idx]
    bmask = find_boundaries(sign_top)
    ids = region_ids(sign_top)
    n_regions = int(ids.max()) + 1
    print(f"top-{TOP_K} region map: {n_regions} regions, {bmask.sum()}/{bmask.size} boundary cells")

    # Thumbnail decoding grid (5×5).
    thumb_alphas = np.linspace(*meta["alpha_range"], N_THUMB).tolist()
    thumb_betas = np.linspace(*meta["beta_range"], N_THUMB).tolist()
    print(f"decoding {N_THUMB}×{N_THUMB} = {N_THUMB * N_THUMB} thumbnails ...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = DDPMPipeline.from_pretrained(MODEL_ID)
    pipeline.to(device)
    unet = pipeline.unet
    scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    h1, h2, h3 = (_load_h_flat(i).to(device) for i in range(3))
    origin, u, v = build_plane(h1, h2, h3)
    x_500 = torch.load(ANCHOR_DIR / "x500_0.pt", weights_only=True).to(device)

    thumbnails = decode_plane_grid(
        unet, alphas_cumprod, x_500,
        origin, u, v,
        alphas=thumb_alphas, betas=thumb_betas,
        channels=CHANNELS, spatial=SPATIAL,
        verbose=True,
    )

    # Resize images to ~64×64 for plotting speed.
    thumbnails_small = []
    for a, b, img in thumbnails:
        from PIL import Image
        small = Image.fromarray(img).resize((64, 64), Image.BICUBIC)
        thumbnails_small.append((a, b, np.array(small)))

    # ---- Figure -----------------------------------------------------------
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 6.2))
    extent = (float(alphas_full[0]), float(alphas_full[-1]),
              float(betas_full[0]), float(betas_full[-1]))

    # Left: region map underlay + thumbnails overlaid.
    base = plt.get_cmap("tab20").colors
    palette = [base[k % len(base)] for k in range(n_regions)]
    ax_l.imshow(ids.T, origin="lower", extent=extent,
                cmap=ListedColormap(palette), interpolation="nearest", alpha=0.85)
    by, bx = np.where(bmask)
    ax_l.scatter(alphas_full[by], betas_full[bx], s=2, c="black", alpha=0.55, linewidths=0)
    overlay_thumbnails(ax_l, thumbnails_small, zoom=0.45)
    for a, b in anchors_ab:
        ax_l.scatter([a], [b], marker="*", s=240, c="yellow",
                     edgecolors="black", linewidths=1.4, zorder=5)
    ax_l.set_xlim(extent[0], extent[1])
    ax_l.set_ylim(extent[2], extent[3])
    ax_l.set_xlabel(r"$\alpha$")
    ax_l.set_ylabel(r"$\beta$")
    ax_l.set_title(f"regions ({n_regions}) + decoded thumbnails")

    # Right: thumbnails only.
    ax_r.set_xlim(extent[0], extent[1])
    ax_r.set_ylim(extent[2], extent[3])
    overlay_thumbnails(ax_r, thumbnails_small, zoom=0.7)
    for a, b in anchors_ab:
        ax_r.scatter([a], [b], marker="*", s=200, c="yellow",
                     edgecolors="black", linewidths=1.4, zorder=5)
    ax_r.set_xlabel(r"$\alpha$")
    ax_r.set_ylabel(r"$\beta$")
    ax_r.set_title(f"{N_THUMB}×{N_THUMB} decoded thumbnails (no region overlay)")
    ax_r.set_aspect("equal")

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150)
    print(f"saved {OUT_FIG}")


if __name__ == "__main__":
    main()
