"""Phase 1 smoke test.

Loads `google/ddpm-celebahq-256`, runs a few denoising steps, saves a sample.
Goal: verify that the environment + pretrained model are usable end-to-end
before we invest in Phase 2+ utilities.

Output:
    data/smoke_sample.png  — a 256x256 RGB image
"""
from __future__ import annotations

from pathlib import Path

import torch
from diffusers import DDIMScheduler, DDPMPipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "data" / "smoke_sample.png"
MODEL_ID = "google/ddpm-celebahq-256"
SEED = 0
NUM_INFERENCE_STEPS = 25  # DDIM with few steps — Phase 1 only checks plumbing


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline = DDPMPipeline.from_pretrained(MODEL_ID)
    # Swap to DDIM for fast deterministic sampling. Same UNet, different scheduler.
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.to(device)

    generator = torch.Generator(device=device).manual_seed(SEED)
    result = pipeline(
        batch_size=1,
        num_inference_steps=NUM_INFERENCE_STEPS,
        generator=generator,
    )
    image = result.images[0]
    image.save(OUT_PATH)
    print(f"Saved smoke sample to {OUT_PATH} ({image.size})")


if __name__ == "__main__":
    main()
