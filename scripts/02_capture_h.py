"""Phase 2 demonstration: capture the h-vector of DDPM CelebA-HQ at t=500.

Prints the bottleneck output shape so Phase 4 (sign-vector dim) can be
pinned down. Also serves as a runtime smoke check that MidBlockCapture
works on the real pretrained UNet.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from diffusers import DDPMPipeline  # noqa: E402

from diffusion_boundary.hooks import MidBlockCapture  # noqa: E402

MODEL_ID = "google/ddpm-celebahq-256"
TARGET_T = 500
SEED = 0


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = DDPMPipeline.from_pretrained(MODEL_ID)
    pipeline.to(device)
    unet = pipeline.unet

    sample_size = unet.config.sample_size
    in_channels = unet.config.in_channels
    generator = torch.Generator(device=device).manual_seed(SEED)
    x = torch.randn(1, in_channels, sample_size, sample_size,
                    generator=generator, device=device)
    t = torch.tensor([TARGET_T], device=device)

    with MidBlockCapture(unet) as cap, torch.no_grad():
        _ = unet(x, t).sample
    h = cap.feature

    print(f"input shape : {tuple(x.shape)}  (sample_size={sample_size}, in_channels={in_channels})")
    print(f"timestep    : {TARGET_T}")
    print(f"h shape     : {tuple(h.shape)}")
    print(f"h dtype     : {h.dtype}")
    print(f"h device    : {h.device}")


if __name__ == "__main__":
    main()
