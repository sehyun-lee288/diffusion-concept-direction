"""Phase 3 post-condition tests for prepared anchors.

These tests don't re-run the (network-bound) anchor download script; they
just verify that the artifacts it produced (under data/anchors/) have the
expected shape, distinct values, and consistent metadata. Skipped if
scripts/03_invert_anchors.py hasn't been run yet.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

ANCHOR_DIR = Path(__file__).resolve().parents[1] / "data" / "anchors"
META_PATH = ANCHOR_DIR / "meta.yaml"

requires_anchors = pytest.mark.skipif(
    not META_PATH.exists(),
    reason="anchor artifacts missing — run scripts/03_invert_anchors.py first",
)


@requires_anchors
def test_meta_describes_three_anchors():
    meta = yaml.safe_load(META_PATH.read_text())
    assert len(meta["anchors"]) == 3
    assert meta["target_t"] == 500


@requires_anchors
def test_anchor_artifacts_present():
    for i in range(3):
        assert (ANCHOR_DIR / f"anchor_{i}.png").is_file()
        assert (ANCHOR_DIR / f"x500_{i}.pt").is_file()
        assert (ANCHOR_DIR / f"h500_{i}.pt").is_file()


@requires_anchors
def test_h_vectors_have_expected_shape():
    for i in range(3):
        h = torch.load(ANCHOR_DIR / f"h500_{i}.pt", weights_only=True)
        assert h.shape == (1, 512, 8, 8), f"anchor {i} has h.shape={h.shape}"


@requires_anchors
def test_h_vectors_are_distinct():
    """If two anchors collapse to the same h, Phase 4's 2D plane is degenerate."""
    hs = [torch.load(ANCHOR_DIR / f"h500_{i}.pt", weights_only=True).flatten() for i in range(3)]
    for i in range(3):
        for j in range(i + 1, 3):
            d = (hs[i] - hs[j]).norm().item()
            assert d > 1e-3, f"anchors {i} and {j} have nearly identical h (L2={d:.2e})"
