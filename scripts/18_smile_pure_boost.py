"""Phase 16b: amplify s on the smile-pure channel mask.

Phase 16 showed selective injection eliminates gender drift but
weakens the smile signal. Quick test: if we double the magnitude
(s up to ±6) on the smile-pure mask, can we recover Phase-11-strength
smile editing without recovering the gender entanglement?

Output: figures/exp12b_smile_pure_boost.png
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
from diffusion_boundary.multistep import denoise_with_injection  # noqa: E402

MODEL_ID = "google/ddpm-celebahq-256"
CHANNELS = 512
NUM_DDIM_STEPS = 50
S_VALUES = [-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0]
T_CATEGORIZE = 450

MULTIATTR = REPO_ROOT / "data" / "delta_h_multiattr.pt"
ORTH = REPO_ROOT / "data" / "delta_h_smile_orth.pt"
FIG_OUT = REPO_ROOT / "figures" / "exp12b_smile_pure_boost.png"


def _spatial_mean(t: torch.Tensor) -> np.ndarray:
    return t.reshape(CHANNELS, -1).mean(-1).numpy()


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = DDPMPipeline.from_pretrained(MODEL_ID).to(device)
    unet = pipeline.unet
    scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    attr = torch.load(MULTIATTR, weights_only=False)
    smile_orth = torch.load(ORTH, weights_only=False)

    A = _spatial_mean(smile_orth[T_CATEGORIZE].float())
    B = _spatial_mean(attr["gender"][T_CATEGORIZE].float())
    norm = np.hypot(A, B)
    smile_pur = np.abs(A) / np.maximum(norm, 1e-12)
    gender_pur = np.abs(B) / np.maximum(norm, 1e-12)
    cat = np.full(CHANNELS, "joint", dtype=object)
    cat[smile_pur > 0.85] = "smile"
    cat[gender_pur > 0.85] = "gender"
    cat[norm < np.quantile(norm, 0.25)] = "weak"
    smile_mask = (cat == "smile")
    m = torch.from_numpy(smile_mask.astype(np.float32))[None, :, None, None]
    masked = {t: smile_orth[t] * m for t in smile_orth}

    torch.manual_seed(0)
    x_T = torch.randn(1, 3, 256, 256, device=device)

    decoded = []
    for s in S_VALUES:
        x0 = denoise_with_injection(unet, scheduler, x_T, [masked], [s],
                                    NUM_DDIM_STEPS, device)
        decoded.append((s, to_uint8_image(x0)))
        print(f"  s={s:+.1f} ok")

    n = len(S_VALUES)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.4))
    for i, (s, img) in enumerate(decoded):
        axes[i].imshow(img)
        axes[i].set_title(f"s = {s:+.1f}", fontsize=11,
                          color=("crimson" if s > 0 else "darkblue" if s < 0 else "black"))
        axes[i].axis("off")
    fig.suptitle("Phase 16b — smile-pure mask, amplified s range (same x_T)",
                 fontsize=12)
    fig.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150)
    print(f"saved {FIG_OUT}")

    import os
    os._exit(0)


if __name__ == "__main__":
    main()
