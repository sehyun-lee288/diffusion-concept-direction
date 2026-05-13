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

## Phase 5.5 — Radial Pattern Analysis (resolved)

**Question**: Why do the top-20 boundary lines visually converge near
(α, β) ≈ (0.5, 0.5)?

**Theory**. Each channel c's boundary on the plane is the line
`A_c·α + B_c·β + C_c = 0` where `(A_c, B_c, C_c) = (mean_c(h2−h1),
mean_c(v), mean_c(h1))`. Stacking these into `M ∈ ℝ^{512×3}`:

- `rank(M) = 1` ⇒ all lines coincide (degenerate).
- `rank(M) = 2` ⇒ all lines pass through the unique projective null vector
  of M (a true *pencil* of lines).
- `rank(M) = 3` ⇒ no common point.

**Measurement** (`scripts/06_analyze_radial.py`).
SVD of `M`: singular values **[11.17, 8.24, 3.85]**, energy share
**60.2 / 32.7 / 7.1 %**. So `M` has effective rank ≈ 2 (s3 carries only 7%
of energy) — but is *not* exactly rank-2; the third direction is small but
nonzero.

Naively this would predict a "near-pencil" with a single common point.
But the SVD-derived best-fit common point is (−0.785, −0.323), while the
empirical pairwise-intersection cluster sits at (0.495, 0.500) — they
disagree, so the SVD/null-space story alone is not the whole explanation.

**K-sweep** reveals the dominant cause:

| K   | n_pairs | median (α, β)        | MAD              |
|-----|---------|----------------------|------------------|
| 10  | 45      | (+0.500, +0.502)     | (0.007, 0.005)   |
| 20  | 190     | (+0.495, +0.500)     | (0.012, 0.013)   |
| 50  | 1 225   | (+0.504, +0.498)     | (0.036, 0.035)   |
| 100 | 4 950   | (+0.488, +0.478)     | (0.073, 0.061)   |
| 250 | 31 125  | (+0.534, +0.433)     | (0.184, 0.203)   |
| 512 | 130 816 | (+0.647, +0.274)     | (0.456, 0.474)   |

The cluster tightness drops monotonically as K grows. The full-channel
median (0.647, 0.274) doesn't match the triangle centroid (0.498, 0.333)
either, so the model's intrinsic structure isn't "lines through the
centroid."

**Conclusion**. The visual radial pattern at K=20 is *mostly a selection
artifact*: top-K-balanced picks channels whose ±1 split across the grid
window is closest to 50/50, which is equivalent to picking channels
whose boundary line passes near the center of the grid window. With the
grid α, β ∈ [−0.5, 1.5] the center is (0.5, 0.5), and that is where the
chosen lines converge. The underlying model has effective rank-2 features
(60.2 + 32.7 = 92.9% energy in first two directions), which makes the
artifact especially clean — but it is still an artifact, not a discovery
about DDPM CelebA-HQ.

**Implication for follow-ups**. If we want to draw boundaries that
reflect intrinsic model structure rather than the grid window, two
candidate fixes:
1. Replace top-K-balanced with **top-K-variance** of `(A_c, B_c)`
   (largest contribution to the boundary direction, not to the in-window
   split).
2. Center the grid on the SVD null point or the anchor triangle centroid,
   then use **all active** channels.

Diagnostics file: `figures/exp1_radial_analysis.png`.
Script: `scripts/06_analyze_radial.py`.

## Phase 6 — Single-location (per-neuron) boundary

**Motivation**. The Phase 4/5 reduction `sign(spatial_mean(h_c))` collapses
64 spatial neurons per channel into one "averaged" line, which is *not*
the boundary of any actual neuron in the U-Net. To draw RDR-style real
neuron boundaries we use `sign(h_c[i*, j*])` at a fixed pixel — each line
on the plane is then the genuine zero-crossing of one mid_block neuron.

**Deliverable**
- `diffusion_boundary.plane.channel_sign_at_pixel` + unit test
- `scripts/07_single_location.py` — runs at center pixel (4, 4),
  produces `figures/exp2_single_location_pix4_4.png` (two panels:
  region map + all active neuron lines)

**Result** (pixel `(4, 4)`).
- 475 / 512 channels active in the window
- 2438 regions / 2500 cells (essentially each cell is its own region —
  475-dim sign vector saturates the 50×50 grid resolution)
- **Right panel — all 475 boundary lines drawn directly — show NO radial
  pattern**. Lines are distributed approximately uniformly in direction
  and offset. This decisively confirms the Phase 5.5 conclusion: the
  radial appearance in `exp1_boundary.png` is a *selection artifact* of
  top-K-balanced, not a property of the bottleneck.

**Implication**. Real per-neuron boundaries are *not* concentrated near
any common point. A useful next step is to find a region-merging or
channel-selection scheme that respects this true geometry but still
yields a small number of *meaningful* regions for visualization.

## Phase 7 — Single channel, varying spatial position

**Setup**. Fix one channel `c`, draw 64 boundary lines — one per
spatial location `(i, j)`. Same conv filter, different spatial reading.

**Deliverable**
- `diffusion_boundary.plane.pixel_signs_for_channel` + unit test
- `scripts/08_single_channel.py` — auto-selects the top-4 channels by
  "pattern diversity" (largest number of distinct per-pixel sign vectors
  across the 50×50 grid), produces a 2×4 panel
- `figures/exp3_single_channel.png`

**Result**. Selected channels: 386, 107, 131, 338 with 751, 723, 708,
705 unique 64-bit patterns respectively. Per channel ~62–64 / 64 pixels
are active in the window. Region counts: 868, 826, 810, 814 (still
saturating but ~3× sparser than the all-channel 2438).

**Geometric observation**. Per-channel line bundles are *not* radial,
matching Phase 6. They *are* somewhat correlated in direction within a
channel (visible in channel 107 / 131 in particular) — neighboring
spatial pixels share the same conv filter so their coefficients
(A, B, C) at adjacent (i, j) are smooth and the lines tilt similarly.
This is a sign that a single conv channel encodes a low-rank
*spatial-direction* preference, even though across channels the
directions are uniform.

## Findings / Follow-ups (for Phase 6+)

1. ~~**Radial boundary pattern**~~ — explained above (Phase 5.5).
2. **Subtle thumbnail variation**: decoding from h-only (with fixed
   skips) yields small attribute change. Test (a) injection at multiple
   t values, (b) sweeping x_500 along with h.
3. **DDIM inversion**: queued ablation to replace q-sample.
4. **K sensitivity**: how does the region count and the perceived
   "structure" change as K varies in {5, 10, 20, 50}?
5. **Per-channel line plot**: draw the K boundary lines explicitly
   (not just region coloring) — closer to RDR Figure 1 style.
