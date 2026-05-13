# Implementation Plan — Toy Experiment 1

Phase-by-phase execution of [research_plan.md](research_plan.md) §5.
Authoritative design rationale lives in the research plan; this file tracks
**status and per-phase deliverables/results only**.

## Phase Status

- [x] Phase 1 — Environment & Smoke Test
- [x] Phase 2 — H-Space Capture Hook
- [x] Phase 3 — DDIM Inversion & Anchors
- [x] Phase 4 — 2D Plane & Sign Grid
- [ ] Phase 5 — Boundary Visualization

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
(미실행)
