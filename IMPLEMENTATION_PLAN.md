# Implementation Plan — Toy Experiment 1

Phase-by-phase execution of [research_plan.md](research_plan.md) §5.
이 파일은 **진행 상태 + 파일 산출물**만 추적한다. 분석/해석 결과는
[FINDINGS.md](FINDINGS.md)로 분리. 연구 동기/RQ/관련 연구는
[research_plan.md](research_plan.md) 참조.

## Phase Status

- [x] Phase 1 — Environment & Smoke Test
- [x] Phase 2 — H-Space Capture Hook
- [x] Phase 3 — Forward Diffusion & Anchors
- [x] Phase 4 — 2D Plane & Sign Grid
- [x] Phase 5 — Boundary Visualization
- [x] Phase 5.5 — Radial pattern diagnostic
- [x] Phase 6 — Single spatial location boundary
- [x] Phase 7 — Single channel, varying spatial

## Confirmed Design Decisions

| 항목 | 값 |
|---|---|
| Model | `google/ddpm-celebahq-256` |
| Conda env | `diffloc` (torch 2.11, diffusers 0.37.1) |
| Timestep | t = 500 |
| Bottleneck | `unet.mid_block` 출력, shape (1, 512, 8, 8) |
| Sign 단위 | channel-wise spatial **mean** (Phase 4 evidence; FINDINGS.md §1) |
| Plane basis | orthogonal but **not** unit-norm; anchors at round coords |
| Anchor | `mattymchen/celeba-hq` indices 0 / 1000 / 5000 |
| Inversion | q-sample (forward diffusion) with fixed seed per anchor |
| Code 구조 | flat `scripts/0N_*.py` + `diffusion_boundary/` utility |
| Test | pytest, 29 tests, GPU-dependent ones marked `@pytest.mark.gpu` |

## Phase 1 — Environment & Smoke Test

**Deliverables**
- `requirements.txt`, `pyproject.toml`, `README.md`, `.gitignore`
- `scripts/01_smoke_test.py` → `data/smoke_sample.png`
- `tests/test_smoke.py` (2 tests)

## Phase 2 — H-Space Capture Hook

**Deliverables**
- `diffusion_boundary/hooks.py::MidBlockCapture` (context manager)
- `scripts/02_capture_h.py` (real-model demo, prints actual shape)
- `tests/test_hooks.py` (5 tests, including a synthetic `_FakeUNet`)

## Phase 3 — Forward Diffusion & Anchors

q-sample as inversion (FINDINGS.md §0). Dataset auto-selected to
`mattymchen/celeba-hq` after Hub probing.

**Deliverables**
- `diffusion_boundary/inversion.py::noise_to_t`
- `scripts/03_invert_anchors.py` → `data/anchors/`
- `tests/test_inversion.py` (5) + `tests/test_anchors.py` (4)

**Data products**: anchor_{0,1,2}.png, x500_{0,1,2}.pt, h500_{0,1,2}.pt,
meta.yaml.

## Phase 4 — 2D Plane & Sign Grid

Two evidence-based deviations: spatial-max → mean, orthonormal → non-unit
orthogonal basis. Detailed reasoning in FINDINGS.md §§1–2.

**Deliverables**
- `diffusion_boundary/plane.py` (`build_plane`, `project_to_plane`,
  `grid_points`, `channel_sign_mean`, `channel_sign_at_pixel`,
  `pixel_signs_for_channel`)
- `scripts/04_compute_sign_grid.py` → `data/sign_grid.npy` (int8 50×50×512)
- `tests/test_plane.py` (7 tests)

## Phase 5 — Boundary Visualization

**Deliverables**
- `diffusion_boundary/viz.py` (`find_boundaries`, `region_ids`,
  `active_channel_mask`, `top_k_balanced_channels`, `plot_boundary_panel`)
- `scripts/05_visualize.py` — DDIM-inject decode at 9 grid points
- `tests/test_viz.py` (6 tests)

**Figure**: `figures/exp1_boundary.png` (134 regions, top-K=20 channels,
9 decoded thumbnails).

## Phase 5.5 — Radial pattern diagnostic

**Deliverables**
- `scripts/06_analyze_radial.py` — SVD on M = [A | B | C], K-sweep
  over {10, 20, 50, 100, 250, 512}

**Figure**: `figures/exp1_radial_analysis.png`.

Conclusion in FINDINGS.md §3.

## Phase 6 — Single spatial location boundary

**Deliverables**
- `diffusion_boundary.plane.channel_sign_at_pixel`
- `scripts/07_single_location.py` (center pixel (4, 4))

**Figure**: `figures/exp2_single_location_pix4_4.png` — left: region map;
right: all 475 active neuron boundary lines.

Conclusion in FINDINGS.md §4.

## Phase 7 — Single channel, varying spatial

**Deliverables**
- `diffusion_boundary.plane.pixel_signs_for_channel`
- `scripts/08_single_channel.py` (auto-picks top-4 most-diverse channels)

**Figure**: `figures/exp3_single_channel.png` — 2 × 4 panel for channels
386 / 107 / 131 / 338.

Conclusion in FINDINGS.md §5.

## Next phases

See [FINDINGS.md §8](FINDINGS.md) for prioritized follow-ups. Each will
introduce a new `scripts/NN_*.py` plus a figure under `figures/`.

## Quality gates (each commit)

- [x] pytest 통과 (29 / 29)
- [x] ruff clean
- [x] phase commit 메시지에 "왜" 명시
- [x] IMPLEMENTATION_PLAN.md + FINDINGS.md 상응 업데이트
