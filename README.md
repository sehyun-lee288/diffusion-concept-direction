# diffusion-concept-direction

Decision boundary visualization in the **h-space** of a DDPM, applied to
CelebA-HQ. See [`research_plan.md`](research_plan.md) for the research goal
and [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) for the phase-by-phase
execution plan.

## Setup

```bash
conda activate diffloc       # torch 2.11, diffusers 0.37.1
pip install -r requirements.txt
```

## Run

Scripts are numbered to be run in order. Each phase corresponds to one script
and a matching test module.

```bash
# Phase 1 — smoke test
pytest tests/test_smoke.py -v
python scripts/01_smoke_test.py            # -> data/smoke_sample.png

# Phase 2-5 will be added incrementally.
```

## Layout

```
diffusion_boundary/  reusable utilities (hooks, inversion, plane, viz)
scripts/             phase-numbered runnable entry points
tests/               pytest suite (tests/test_*.py)
data/                generated artifacts (anchor images, h-vectors, sign grid)
figures/             output figures
```
