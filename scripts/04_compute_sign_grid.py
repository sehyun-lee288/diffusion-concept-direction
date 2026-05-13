"""Phase 4: compute the channel-wise spatial-mean sign vector on a 2D grid.

Loads the 3 anchor h-vectors, builds the orthonormal plane (h1, h2, h3),
samples a 50×50 grid in (α, β), and stores the sign matrix.

Outputs:
    data/sign_grid.npy   int8 array of shape (n_alpha, n_beta, channels)
    data/grid_meta.yaml  origin/u/v paths, α/β ranges, anchor (α, β), channels/spatial
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from diffusion_boundary.plane import (  # noqa: E402
    build_plane,
    channel_sign_mean,
    grid_points,
    project_to_plane,
)

ANCHOR_DIR = REPO_ROOT / "data" / "anchors"
OUT_GRID = REPO_ROOT / "data" / "sign_grid.npy"
OUT_META = REPO_ROOT / "data" / "grid_meta.yaml"

N_GRID = 50
ALPHA_RANGE = (-0.5, 1.5)
BETA_RANGE = (-0.5, 1.5)
CHANNELS = 512
SPATIAL = 8


def _load_h_flat(i: int) -> torch.Tensor:
    h = torch.load(ANCHOR_DIR / f"h500_{i}.pt", weights_only=True).float()
    return h.reshape(-1)  # (C·H·W,)


def main() -> None:
    h1, h2, h3 = (_load_h_flat(i) for i in range(3))
    origin, u, v = build_plane(h1, h2, h3)

    # Sanity: record each anchor's (α, β) for downstream visualization.
    anchor_coords = []
    for i, h in enumerate([h1, h2, h3]):
        a, b = project_to_plane(h, origin, u, v)
        anchor_coords.append({"index": i, "alpha": float(a), "beta": float(b)})

    alphas = torch.linspace(*ALPHA_RANGE, N_GRID)
    betas = torch.linspace(*BETA_RANGE, N_GRID)
    pts = grid_points(origin, u, v, alphas, betas)  # (N*N, D)
    print(f"grid shape: {tuple(pts.shape)}  α∈{ALPHA_RANGE}, β∈{BETA_RANGE}, n={N_GRID}")

    # Compute in CPU-side chunks to keep memory predictable.
    sign_rows = []
    chunk = 256
    for start in range(0, pts.shape[0], chunk):
        s = channel_sign_mean(pts[start:start + chunk], channels=CHANNELS, spatial=SPATIAL)
        sign_rows.append(s)
    sign_flat = torch.cat(sign_rows, dim=0)  # (N*N, C)
    sign_grid = sign_flat.reshape(N_GRID, N_GRID, CHANNELS).numpy().astype(np.int8)
    OUT_GRID.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_GRID, sign_grid)

    # Diagnostics: per-axis Hamming activity.
    horiz = (sign_grid[:, 1:, :] != sign_grid[:, :-1, :]).sum(axis=-1)
    vert = (sign_grid[1:, :, :] != sign_grid[:-1, :, :]).sum(axis=-1)
    print(f"horizontal Hamming: mean={horiz.mean():.2f} max={horiz.max()}")
    print(f"vertical   Hamming: mean={vert.mean():.2f} max={vert.max()}")
    print(f"unique sign patterns: {len({tuple(row) for row in sign_grid.reshape(-1, CHANNELS)})}")

    meta = {
        "n_grid": N_GRID,
        "alpha_range": list(ALPHA_RANGE),
        "beta_range": list(BETA_RANGE),
        "channels": CHANNELS,
        "spatial": SPATIAL,
        "anchors_alpha_beta": anchor_coords,
        "sign_grid_path": str(OUT_GRID.relative_to(REPO_ROOT)),
    }
    with OUT_META.open("w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    print(f"saved {OUT_GRID} {sign_grid.shape} dtype={sign_grid.dtype}")
    print(f"saved {OUT_META}")


if __name__ == "__main__":
    main()
