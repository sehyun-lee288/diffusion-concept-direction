"""Phase 13: attribute-paired plane analysis.

Replace anchor-spanned plane with an axis-spanned plane:

  injection_t(α, β) = α · Δh_smile_orth_t  +  β · Δh_gender_t

Decode a 5×5 grid in (α, β) and lay the thumbnails out on the plane.
Smile should change along α, gender along β. If yes, the multi-step
attribute axes are *the* concept directions for downstream boundary
analysis — and the sign-pattern boundary on this plane should encode
attribute regions directly (planned as Phase 14).

Outputs:
  figures/exp9_attribute_plane.png   — 5×5 thumbnail grid on (α, β)
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from diffusers import DDIMScheduler, DDPMPipeline  # noqa: E402

from diffusion_boundary.decoding import to_uint8_image  # noqa: E402
from diffusion_boundary.multistep import (  # noqa: E402
    collect_images_by_attribute,
    denoise_with_injection,
    extract_delta_h_multistep,
    orthogonalize_against,
)
from diffusion_boundary.viz import overlay_thumbnails  # noqa: E402

MODEL_ID = "google/ddpm-celebahq-256"
DATASET_ID = "eurecom-ds/celeba-hq-256"
ATTR_SMILING = 31
ATTR_MALE = 20
N_PER_CLASS = 20
TIMESTEPS = [50, 150, 250, 350, 450, 550, 650, 750, 850, 950]
NUM_DDIM_STEPS = 50
N_THUMB = 5
ALPHA_RANGE = (-2.5, 2.5)   # smile axis
BETA_RANGE = (-2.5, 2.5)    # gender axis

DATA_DIR = REPO_ROOT / "data"
MULTIATTR_PATH = DATA_DIR / "delta_h_multiattr.pt"
ORTH_PATH = DATA_DIR / "delta_h_smile_orth.pt"
FIG_OUT = REPO_ROOT / "figures" / "exp9_attribute_plane.png"


def _ensure_directions(unet, scheduler, device: str) -> tuple[dict, dict]:
    """Load cached attribute Δh and orth-smile; extract+save if missing."""
    if MULTIATTR_PATH.exists():
        attr = torch.load(MULTIATTR_PATH, weights_only=False)
    else:
        print(f"cache miss — streaming {DATASET_ID} ...")
        smile_pos = collect_images_by_attribute(DATASET_ID, ATTR_SMILING, 1, N_PER_CLASS)
        smile_neg = collect_images_by_attribute(DATASET_ID, ATTR_SMILING, 0, N_PER_CLASS)
        male_pos = collect_images_by_attribute(DATASET_ID, ATTR_MALE, 1, N_PER_CLASS)
        male_neg = collect_images_by_attribute(DATASET_ID, ATTR_MALE, 0, N_PER_CLASS)
        smile = extract_delta_h_multistep(unet, scheduler, smile_pos, smile_neg,
                                          TIMESTEPS, device, progress=print)
        gender = extract_delta_h_multistep(unet, scheduler, male_pos, male_neg,
                                           TIMESTEPS, device, progress=print)
        attr = {"smile": smile, "gender": gender}
        torch.save(attr, MULTIATTR_PATH)
    if ORTH_PATH.exists():
        smile_orth = torch.load(ORTH_PATH, weights_only=False)
    else:
        smile_orth = orthogonalize_against(attr["smile"], attr["gender"])
        torch.save(smile_orth, ORTH_PATH)
    return smile_orth, attr["gender"]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = DDPMPipeline.from_pretrained(MODEL_ID).to(device)
    unet = pipeline.unet
    scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    smile_orth, gender = _ensure_directions(unet, scheduler, device)

    torch.manual_seed(0)
    x_T = torch.randn(1, 3, 256, 256, device=device)

    alphas = np.linspace(*ALPHA_RANGE, N_THUMB)
    betas = np.linspace(*BETA_RANGE, N_THUMB)
    print(f"decoding {N_THUMB}×{N_THUMB} grid over α ∈ {ALPHA_RANGE}, β ∈ {BETA_RANGE} ...")

    thumbnails: list[tuple[float, float, np.ndarray]] = []
    total = N_THUMB * N_THUMB
    k = 0
    for b in betas:
        for a in alphas:
            x0 = denoise_with_injection(unet, scheduler, x_T,
                                        [smile_orth, gender], [a, b],
                                        NUM_DDIM_STEPS, device)
            img = to_uint8_image(x0)
            thumbnails.append((float(a), float(b), img))
            k += 1
            print(f"  {k:>2}/{total}  (α={a:+.2f}, β={b:+.2f})")

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.set_xlim(ALPHA_RANGE[0] - 0.7, ALPHA_RANGE[1] + 0.7)
    ax.set_ylim(BETA_RANGE[0] - 0.7, BETA_RANGE[1] + 0.7)
    overlay_thumbnails(ax, thumbnails, zoom=0.55)
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.set_xlabel(r"$\alpha$  (smile $\perp$ gender)", fontsize=12)
    ax.set_ylabel(r"$\beta$  (gender)", fontsize=12)
    ax.set_title("Attribute-paired plane — multi-step injection per (α, β)", fontsize=12)
    ax.set_aspect("equal")
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_OUT, dpi=150)
    print(f"saved {FIG_OUT}")

    import os
    os._exit(0)


if __name__ == "__main__":
    main()
