# Implementation Plan — Toy Experiment 1

Phase-by-phase execution of [research_plan.md](research_plan.md) §5.
Authoritative design rationale lives in the research plan; this file tracks
**status and per-phase deliverables/results only**.

## Phase Status

- [x] Phase 1 — Environment & Smoke Test
- [x] Phase 2 — H-Space Capture Hook
- [ ] Phase 3 — DDIM Inversion & Anchors
- [ ] Phase 4 — 2D Plane & Sign Grid
- [ ] Phase 5 — Boundary Visualization

## Confirmed Design Decisions

| 항목 | 값 |
|---|---|
| Model | `google/ddpm-celebahq-256` |
| Conda env | `diffloc` (torch 2.11, diffusers 0.37.1) |
| Timestep | t = 500 |
| Bottleneck | `unet.mid_block` 출력 |
| Sign 단위 | channel-wise spatial **max** → sign |
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
(미실행)

## Phase 4 — 2D Plane & Sign Grid
(미실행)

## Phase 5 — Boundary Visualization
(미실행)
