# Implementation Plan — Toy Experiment 1

Phase-by-phase execution of [research_plan.md](research_plan.md) §5.
Authoritative design rationale lives in the research plan; this file tracks
**status and per-phase deliverables/results only**.

## Phase Status

- [x] Phase 1 — Environment & Smoke Test
- [x] Phase 2 — H-Space Capture Hook
- [x] Phase 3 — DDIM Inversion & Anchors
- [x] Phase 4 — 2D Plane & Sign Grid
- [x] Phase 5 — Boundary Visualization

## Confirmed Design Decisions

| 항목 | 값 |
|---|---|
| Model | `google/ddpm-celebahq-256` |
| Conda env | `diffloc` (torch 2.11, diffusers 0.37.1) |
| Timestep | t = 500 |
| Bottleneck | `unet.mid_block` 출력 |
| Sign 단위 | channel-wise spatial **mean** → sign (evidence-based; see Phase 4) |
| Anchor | HF `datasets`에서 CelebA-HQ 3장 |
| Code 구조 | flat `scripts/0N_*.py` + `diffusion_boundary/` utility |

## Phase 1 — Environment & Smoke Test

**Deliverables**
- `requirements.txt`
- `scripts/01_smoke_test.py` → `data/smoke_sample.png`
- `tests/test_smoke.py` (import + `unet.mid_block` 존재 확인)

**Status**: completed.

**Results**: see git commit message and `data/smoke_sample.png`.

## Phase 2 — H-Space Capture Hook

**Deliverables**
- `diffusion_boundary/hooks.py::MidBlockCapture` — context manager hook
- `scripts/02_capture_h.py` — real-model demo
- `tests/test_hooks.py` (5 tests)

**Status**: completed.

**Result**: `h.shape = (1, 512, 8, 8)` for DDPM CelebA-HQ at t=500.
Phase 4 sign vector dim = 512 (channel-wise spatial-max).

## Phase 3 — DDIM Inversion & Anchors

**Deviation from original plan**: substituted q-sample (forward diffusion) for
DDIM inversion. Justification: for boundary analysis on a 2D plane we only
need deterministic `(x_0, ε, t) ↦ x_t` and a corresponding h-vector;
self-consistent ε prediction adds complexity without changing Phase 4 inputs.
True DDIM inversion is queued as a future ablation.

**Deliverables**
- `diffusion_boundary/inversion.py::noise_to_t` (q-sample with fixed seed)
- `scripts/03_invert_anchors.py` (CelebA-HQ from `mattymchen/celeba-hq` ×3)
- `tests/test_inversion.py` (5 tests) + `tests/test_anchors.py` (4 post-condition tests)

**Data products** (under `data/anchors/`):
- `anchor_{0,1,2}.png` — 256×256 source images (ds idx 0, 1000, 5000)
- `x500_{0,1,2}.pt` — (1, 3, 256, 256) latent at t=500
- `h500_{0,1,2}.pt` — (1, 512, 8, 8) bottleneck feature
- `meta.yaml` — model_id / dataset_id / target_t / per-anchor seeds & paths

**Status**: completed. All 16 tests pass; h-vectors pairwise-distinct.

## Phase 4 — 2D Plane & Sign Grid

**Two evidence-based deviations from the original plan**:

1. **spatial-mean instead of spatial-max** (user-confirmed).
   On real DDPM CelebA-HQ bottleneck features, every channel has at least
   one positive spatial location, so spatial-max collapses to all-+1 and
   Hamming distance between anchors = 0/512. spatial-mean preserves
   42–45% Hamming distance and is the meaningful choice.

2. **orthogonal (not unit-norm) basis** chosen so that anchors map to
   round coordinates `h1→(0,0), h2→(1,0), h3→(a,1)`. The original
   unit-normalized basis put anchor 1 at α≈278 (the L2 norm of `h2-h1`),
   far outside any sensible grid window. With the new basis, the grid
   α, β ∈ [-0.5, 1.5] naturally contains the triangle of anchors.

**Deliverables**
- `diffusion_boundary/plane.py`: `build_plane`, `project_to_plane`,
  `grid_points`, `channel_sign_mean`
- `scripts/04_compute_sign_grid.py` (50×50 grid)
- `tests/test_plane.py` (5 tests)

**Data products**
- `data/sign_grid.npy` — int8 (50, 50, 512)
- `data/grid_meta.yaml`

**Result**: 2432 unique sign patterns / 2500 grid cells; horizontal Hamming
mean = 5.79 (max 19), vertical mean = 6.28 (max 23). Structure is rich
enough to make Phase 5 region clustering meaningful.

**Status**: completed. All 21 tests pass.

## Phase 5 — Boundary Visualization

**Deliverables**
- `diffusion_boundary/viz.py`: `find_boundaries`, `region_ids`,
  `active_channel_mask`, `top_k_balanced_channels`, `plot_boundary_panel`
- `scripts/05_visualize.py` — loads sign grid, picks top-K balanced
  channels, decodes 9 sparse grid points via `decode_from_h` (mid_block
  hook-injection + DDIM denoise from t=500 to t=0), produces the figure
- `tests/test_viz.py` (6 tests)

**Key parameters**
- `TOP_K_CHANNELS = 20` — full 512-channel sign pattern is near-unique per
  cell (2446 / 2500 regions); selecting the 20 most-balanced channels
  brings the count to 134 regions and makes structure visible
- `DECODE_STEP_SIZE = 50` — DDIM coarse stride 500 → 450 → … → 0
- Anchor 0's x_500 used as fixed encoder input across all 9 decodings

**Result** (`figures/exp1_boundary.png`)
- 134 distinct regions in the (α, β) ∈ [-0.5, 1.5]² window
- Anchors `h1`, `h2`, `h3` are visible at (0, 0), (1, 0), (a, 1)
- Boundary lines appear to converge in the central region — empirically
  observed radial pattern worth investigating in future ablations
- 9 decoded thumbnails show the bottleneck-only variation: with fixed
  encoder skip connections (from anchor 0), variations are subtle
  (similar identity, varying expression/highlights) — suggests h-space
  in DDPM CelebA-HQ carries less independent semantic signal than in
  larger latent-diffusion models. A claim worth quantifying.

**Status**: completed. All 27 tests pass; figure renders.

## Findings / Follow-ups (for Phase 6+)

1. **Radial boundary pattern**: lines from many channels appear to pass
   through a common point near the triangle centroid. Math says they
   *shouldn't* in general — investigate whether mid_block features are
   dominated by a low-rank structure that forces this.
2. **Subtle thumbnail variation**: decoding from h-only (with fixed
   skips) yields small attribute change. Test (a) injection at multiple
   t values, (b) sweeping x_500 along with h.
3. **DDIM inversion**: queued ablation to replace q-sample.
4. **K sensitivity**: how does the region count and the perceived
   "structure" change as K varies in {5, 10, 20, 50}?
5. **Per-channel line plot**: draw the K boundary lines explicitly
   (not just region coloring) — closer to RDR Figure 1 style.
