"""Phase 5 tests for boundary detection and visualization."""
from __future__ import annotations

import numpy as np

from diffusion_boundary.viz import (
    active_channel_mask,
    find_boundaries,
    region_ids,
    top_k_balanced_channels,
)


def test_active_channel_mask():
    """A channel is active iff its sign isn't constant across the grid."""
    grid = np.ones((4, 4, 3), dtype=np.int8)
    grid[:, :, 1] = -1  # channel 1 is constant -1 → not active
    grid[2:, :, 2] = -1  # channel 2 varies → active
    # channel 0 is constant +1 → not active
    m = active_channel_mask(grid)
    assert m.tolist() == [False, False, True]


def test_top_k_balanced_channels_prefers_balanced_split():
    """A 50/50 channel must outrank a 90/10 channel when K=1."""
    grid = np.ones((10, 10, 2), dtype=np.int8)
    grid[5:, :, 0] = -1   # channel 0: exactly 50% +1
    grid[1:, :, 1] = -1   # channel 1: 10% +1
    idx = top_k_balanced_channels(grid, k=1)
    assert idx.tolist() == [0]


def test_find_boundaries_synthetic_split():
    """Half +1 / half -1 grid: boundary mask is True only on cells touching the seam."""
    grid = np.ones((4, 4, 1), dtype=np.int8)
    grid[2:, :, :] = -1
    bmask = find_boundaries(grid)
    assert bmask.shape == (4, 4)
    # Row 1 (last +1 row) and row 2 (first -1 row) each touch the seam.
    assert bmask[1, :].all() and bmask[2, :].all()
    # Rows 0 and 3 are interior and should be clean.
    assert not bmask[0, :].any() and not bmask[3, :].any()


def test_find_boundaries_uniform_grid_is_empty():
    grid = np.ones((5, 5, 3), dtype=np.int8)
    bmask = find_boundaries(grid)
    assert not bmask.any()


def test_region_ids_assigns_two_components_to_split():
    """A grid split by a single seam must produce exactly two distinct labels."""
    grid = np.ones((4, 4, 1), dtype=np.int8)
    grid[2:, :, :] = -1
    ids = region_ids(grid)
    assert ids.shape == (4, 4)
    assert ids[0, 0] != ids[3, 3]
    assert (ids[:2, :] == ids[0, 0]).all()
    assert (ids[2:, :] == ids[3, 3]).all()


def test_region_ids_uniform_grid_is_one_component():
    grid = np.full((3, 3, 5), 1, dtype=np.int8)
    ids = region_ids(grid)
    assert (ids == ids[0, 0]).all()
