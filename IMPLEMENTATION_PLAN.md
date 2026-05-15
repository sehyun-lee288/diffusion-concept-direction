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
- [x] Phase 8 — Plane-grid sampled-image overlay
- [x] Phase 9 — Per-channel angle statistics
- [x] Phase 10 — Supervised Δh feasibility probe → **FAILED (pivot)**
- [x] Phase 11 — Multi-step h trajectory editing → **WORKS**
- [x] Phase 12 — Orthogonalize smile against gender
- [x] Phase 13 — Attribute-paired plane disentangled

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

## Phase 8 — Plane-grid sampled-image overlay (utility + demo)

Going forward, every plane plot should be able to attach a decoded
thumbnail grid so visible attribute changes can be cross-checked
against sign-pattern boundaries.

**Deliverables**
- `diffusion_boundary/decoding.py` (`decode_from_h`, `decode_plane_grid`,
  `to_uint8_image`) — refactored from script 05
- `diffusion_boundary/viz.py::overlay_thumbnails`
  (matplotlib `AnnotationBbox` placement at plane coordinates)
- `scripts/09_plane_thumbnails.py` (5×5 demo)
- `tests/test_viz.py::test_overlay_thumbnails_adds_artists`

**Figure**: `figures/exp4_plane_thumbnails.png` — 2 panels: region map +
5×5 thumbnails overlaid (left); thumbnails alone (right).

## Phase 9 — Per-channel angle statistics

Quantify the Phase 7 qualitative observation ("some channels show
clustered line slopes") with circular statistics.

**Deliverables**
- `scripts/10_channel_angle_stats.py` — for each channel computes the
  doubled-angle mean direction μ_c and concentration R_c over its 64
  per-pixel line normals

**Figure**: `figures/exp5_channel_angle_stats.png` — R histogram +
rose plots for top-4 and bottom-4 channels.

Conclusion in FINDINGS.md §5 (now quantified): all channels have
R ≤ 0.43, median R ≈ 0.14 — directional bias exists but is weak.

## Phase 10 — Supervised Δh feasibility probe (FAILED — pivot)

Gate experiment: does the mid_block at t=500 carry an *editable* smile
direction? Probe with mean-shift Δh from 20 vs 20 attribute-labeled
images and sweep injection magnitude s ∈ [−3, +3].

**Deliverables**
- `scripts/11_smile_probe.py` (uses `eurecom-ds/celeba-hq-256`)
- `data/delta_h_smile.pt` — saved direction for reuse
- Sanity: in-h-space class separation d′ = **4.23** (perfect)

**Figure**: `figures/exp6_smile_probe.png` + `exp6_smile_probe_frames/`.

**Result**: failure. Δh strongly separates classes in h-space but the
decoded image changes by only 1.25 mean abs pixel diff at s=+3 vs
s=0 — well below the 11.10 reconstruction noise floor. Encoder skip
dominance confirmed at the supervised limit.

Conclusion in FINDINGS.md §6b. Pivot plan: multi-step trajectory edit.

## Phase 11 — Multi-step h trajectory editing (Asyrp-style)

Inject `+ s·Δh_{t_nearest}` into mid_block **at every DDIM step**, not
just one. Bent x_t at step k becomes encoder input at step k+1, so the
skip cannot remain pinned to a fixed input — the edit accumulates.

**Deliverables**
- `scripts/12_multistep_smile.py` — extracts Δh_t for
  t ∈ {50, 150, …, 950}, 50-step DDIM denoise with per-step hook inject,
  s ∈ [-3, +3] sweep
- `data/delta_h_smile_multistep.pt` — dict {t: Δh_t}

**Figure**: `figures/exp7_multistep_smile.png` + frame dump.

**Result**: works. Decoded faces show clear smile transition across s.
Skip-dominance bottleneck is bypassed by trajectory-level accumulation.
However, smile direction is entangled with age/gender (small N=20
covariate skew). Discussed in FINDINGS.md §6c.

## Phase 12 — Orthogonalize smile against gender

**Refactor**
- `diffusion_boundary/multistep.py` — pulled `extract_delta_h_multistep`,
  `denoise_with_injection`, `orthogonalize_against`,
  `collect_images_by_attribute` out of `scripts/12_*.py` for reuse.
- `tests/test_multistep.py` — 4 tests for `orthogonalize_against`.

**Deliverable**
- `scripts/13_orthogonalization.py` — extracts Δh_smile and Δh_gender at
  10 timesteps, applies per-t Gram–Schmidt, sweeps both directions on the
  same fixed x_T.
- `data/delta_h_multiattr.pt`, `data/delta_h_smile_orth.pt`

**Figure**: `figures/exp8_orthogonalization.png` (2-row sweep).

**Result**: gender component is **~44%** of the smile-direction norm at
low t — substantial confound, not a sample artifact. Orth helps but
doesn't fully decouple at s=+3.

## Phase 13 — Attribute-paired plane

**Deliverable**
- `scripts/14_attribute_plane.py` — multi-step inject `α·Δh_smile_orth +
  β·Δh_gender` per (α, β) grid point, 5×5 thumbnail grid on the plane.

**Figure**: `figures/exp9_attribute_plane.png`.

**Result**: clean disentangled 2D semantic plane. α axis controls smile
in every row, β axis controls gender in every column. All four corners
realize the correct attribute combinations. Discussed in FINDINGS.md
§6d.2 — this is the foundation for re-running boundary analysis with
*meaningful* axes (queued as Phase 14).

## Next phases

See [FINDINGS.md §8](FINDINGS.md) for prioritized follow-ups. Each will
introduce a new `scripts/NN_*.py` plus a figure under `figures/`.

## Quality gates (each commit)

- [x] pytest 통과 (29 / 29)
- [x] ruff clean
- [x] phase commit 메시지에 "왜" 명시
- [x] IMPLEMENTATION_PLAN.md + FINDINGS.md 상응 업데이트
