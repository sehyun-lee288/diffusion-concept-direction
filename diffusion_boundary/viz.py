"""Boundary detection and figure assembly for the sign grid.

Three utilities:

- `active_channel_mask(sign_grid)` returns a boolean (C,) mask of channels
  whose sign actually varies across the grid window. Channels that are
  constantly +1 (or constantly -1) over the whole grid contribute no
  boundary; filtering them out reduces the region count from O(n²) to
  something interpretable.
- `find_boundaries(sign_grid)` marks cells whose sign pattern differs
  from any 4-connected neighbor.
- `region_ids(sign_grid)` labels each cell with a 4-connected-component
  id where two cells belong to the same component iff their sign vectors
  are identical.

Plotting helpers (`plot_boundary_panel`) are matplotlib-only; they take
already-computed arrays so they're easy to unit-smoke-test.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np


def active_channel_mask(sign_grid: np.ndarray) -> np.ndarray:
    """Return (C,) bool: True for channels whose sign isn't constant on the grid."""
    if sign_grid.ndim != 3:
        raise ValueError(f"expected (n, m, C) sign grid, got {sign_grid.shape}")
    flat = sign_grid.reshape(-1, sign_grid.shape[-1])
    return (flat.min(axis=0) != flat.max(axis=0))


def top_k_balanced_channels(sign_grid: np.ndarray, k: int) -> np.ndarray:
    """Return integer indices of the K channels with the most balanced ±1 split.

    Balance is measured as min(p, 1 - p) where p is the fraction of +1 cells:
    a perfectly balanced channel scores 0.5, a constant channel scores 0.
    Selecting balanced channels gives boundary lines that actually pass
    through the grid window, instead of channels whose sign flips in a
    near-degenerate corner.
    """
    if sign_grid.ndim != 3:
        raise ValueError(f"expected (n, m, C) sign grid, got {sign_grid.shape}")
    flat = sign_grid.reshape(-1, sign_grid.shape[-1])
    p = (flat > 0).mean(axis=0)
    score = np.minimum(p, 1.0 - p)
    return np.argsort(-score)[:k]


def find_boundaries(sign_grid: np.ndarray) -> np.ndarray:
    """Return (n, m) bool mask: True where any 4-neighbor disagrees."""
    if sign_grid.ndim != 3:
        raise ValueError(f"expected (n, m, C) sign grid, got {sign_grid.shape}")
    n, m, _ = sign_grid.shape
    mask = np.zeros((n, m), dtype=bool)
    # vertical neighbors (rows i, i+1)
    diff_v = (sign_grid[1:, :, :] != sign_grid[:-1, :, :]).any(axis=-1)
    mask[1:, :] |= diff_v
    mask[:-1, :] |= diff_v
    # horizontal neighbors (cols j, j+1)
    diff_h = (sign_grid[:, 1:, :] != sign_grid[:, :-1, :]).any(axis=-1)
    mask[:, 1:] |= diff_h
    mask[:, :-1] |= diff_h
    return mask


def region_ids(sign_grid: np.ndarray) -> np.ndarray:
    """Label cells by 4-connected components of identical sign vectors.

    Returns an (n, m) int array of region ids starting from 0.
    """
    if sign_grid.ndim != 3:
        raise ValueError(f"expected (n, m, C) sign grid, got {sign_grid.shape}")
    n, m, _ = sign_grid.shape
    ids = np.full((n, m), -1, dtype=np.int32)
    next_id = 0
    for i in range(n):
        for j in range(m):
            if ids[i, j] != -1:
                continue
            # BFS flood-fill on matching sign vectors
            ids[i, j] = next_id
            stack = [(i, j)]
            ref = sign_grid[i, j]
            while stack:
                ci, cj = stack.pop()
                for ni, nj in ((ci - 1, cj), (ci + 1, cj), (ci, cj - 1), (ci, cj + 1)):
                    if 0 <= ni < n and 0 <= nj < m and ids[ni, nj] == -1:
                        if np.array_equal(sign_grid[ni, nj], ref):
                            ids[ni, nj] = next_id
                            stack.append((ni, nj))
            next_id += 1
    return ids


def plot_boundary_panel(
    ids: np.ndarray,
    bmask: np.ndarray,
    alphas: Sequence[float],
    betas: Sequence[float],
    anchor_coords: Sequence[tuple[float, float]],
    thumbnails: Sequence[tuple[float, float, np.ndarray]] | None,
    out_path: str | Path,
) -> None:
    """Compose a single figure showing region coloring, boundary edges,
    anchor markers, and optional decoded thumbnails along the bottom strip.
    """
    # matplotlib import is local so importing this module doesn't pull it in.
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    n_regions = int(ids.max()) + 1
    # tab20 wraps; for many regions we tile.
    base = plt.get_cmap("tab20").colors
    palette = [base[k % len(base)] for k in range(n_regions)]
    cmap = ListedColormap(palette)

    has_thumbs = thumbnails is not None and len(thumbnails) > 0
    fig_h = 6 if not has_thumbs else 7.8
    fig, ax = plt.subplots(figsize=(7, fig_h))
    extent = (float(alphas[0]), float(alphas[-1]), float(betas[0]), float(betas[-1]))
    ax.imshow(ids.T, origin="lower", extent=extent, cmap=cmap, interpolation="nearest")

    # Overlay boundary as black dots on cells that touch a region boundary.
    by, bx = np.where(bmask)
    a_arr = np.asarray(alphas)
    b_arr = np.asarray(betas)
    ax.scatter(a_arr[by], b_arr[bx], s=2, c="black", alpha=0.6, linewidths=0)

    for k, (a, b) in enumerate(anchor_coords):
        ax.scatter([a], [b], marker="*", s=240, c="white", edgecolors="black", linewidths=1.5)
        ax.annotate(f"h{k + 1}", (a, b), textcoords="offset points", xytext=(7, 5),
                    fontsize=10, color="black",
                    bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1})

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta$")
    ax.set_title(f"H-space sign-pattern regions  (n={n_regions} regions)")

    if has_thumbs:
        # Inset a row of thumbnails below the main plot showing decoded images.
        # We use the figure-coordinate axes instead of additional subplots.
        k = len(thumbnails)
        for idx, (a, b, img) in enumerate(thumbnails):
            ax.scatter([a], [b], marker="o", s=70, facecolors="none", edgecolors="red", linewidths=1.5)
            ax.annotate(str(idx + 1), (a, b), textcoords="offset points", xytext=(4, 4),
                        fontsize=8, color="red")
            ax_thumb = fig.add_axes(((idx + 0.5) / k * 0.92 + 0.04, 0.02, 0.85 / k, 0.18))
            ax_thumb.imshow(img)
            ax_thumb.set_xticks([])
            ax_thumb.set_yticks([])
            ax_thumb.set_title(str(idx + 1), fontsize=8)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.22 if has_thumbs else 0.0, 1.0, 1.0))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
