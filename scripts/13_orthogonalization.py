"""Phase 12: orthogonalize the smile direction against gender.

Phase 11's smile sweep was entangled with gender/age — moving along
Δh_smile changed identity too. Standard fix (InterFaceGAN / StyleGAN
concept axis): project the gender component out of the smile direction
*per timestep*.

  Δh_smile^⊥_t = Δh_smile_t − ⟨Δh_smile_t, Δh_gender_t⟩ / ‖Δh_gender_t‖² · Δh_gender_t

Then compare two sweeps from the SAME starting noise x_T:

  row 1: inject + s·Δh_smile        (original; expect entangled drift)
  row 2: inject + s·Δh_smile_orth   (gender-orthogonal; expect smile-only)

Outputs:
  data/delta_h_multiattr.pt           — {"smile": {t: ...}, "gender": {t: ...}}
  data/delta_h_smile_orth.pt          — {t: orth direction}
  figures/exp8_orthogonalization.png  — 2-row sweep
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from diffusers import DDIMScheduler, DDPMPipeline  # noqa: E402
from PIL import Image  # noqa: E402

from diffusion_boundary.decoding import to_uint8_image  # noqa: E402
from diffusion_boundary.multistep import (  # noqa: E402
    collect_images_by_attribute,
    denoise_with_injection,
    extract_delta_h_multistep,
    orthogonalize_against,
)

MODEL_ID = "google/ddpm-celebahq-256"
DATASET_ID = "eurecom-ds/celeba-hq-256"
ATTR_SMILING = 31
ATTR_MALE = 20
N_PER_CLASS = 20
TIMESTEPS = [50, 150, 250, 350, 450, 550, 650, 750, 850, 950]
NUM_DDIM_STEPS = 50
S_VALUES = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]

DATA_DIR = REPO_ROOT / "data"
MULTIATTR_OUT = DATA_DIR / "delta_h_multiattr.pt"
ORTH_OUT = DATA_DIR / "delta_h_smile_orth.pt"
FIG_OUT = REPO_ROOT / "figures" / "exp8_orthogonalization.png"
FRAMES_DIR = REPO_ROOT / "figures" / "exp8_orthogonalization_frames"


def _maybe_load_multiattr(unet, scheduler, device: str) -> dict[str, dict[int, torch.Tensor]]:
    if MULTIATTR_OUT.exists():
        print(f"loading cached {MULTIATTR_OUT}")
        return torch.load(MULTIATTR_OUT, weights_only=False)
    print(f"streaming {DATASET_ID} for smile + gender labels ...")
    smile_pos = collect_images_by_attribute(DATASET_ID, ATTR_SMILING, 1, N_PER_CLASS)
    smile_neg = collect_images_by_attribute(DATASET_ID, ATTR_SMILING, 0, N_PER_CLASS)
    male_pos = collect_images_by_attribute(DATASET_ID, ATTR_MALE, 1, N_PER_CLASS)
    male_neg = collect_images_by_attribute(DATASET_ID, ATTR_MALE, 0, N_PER_CLASS)
    print("extracting Δh_smile ...")
    smile = extract_delta_h_multistep(unet, scheduler, smile_pos, smile_neg,
                                      TIMESTEPS, device, progress=print)
    print("extracting Δh_gender ...")
    gender = extract_delta_h_multistep(unet, scheduler, male_pos, male_neg,
                                       TIMESTEPS, device, progress=print)
    out = {"smile": smile, "gender": gender}
    torch.save(out, MULTIATTR_OUT)
    print(f"saved {MULTIATTR_OUT}")
    return out


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = DDPMPipeline.from_pretrained(MODEL_ID).to(device)
    unet = pipeline.unet
    scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    attr = _maybe_load_multiattr(unet, scheduler, device)
    delta_smile = attr["smile"]
    delta_gender = attr["gender"]

    print("orthogonalizing smile against gender (per-t) ...")
    delta_smile_orth = orthogonalize_against(delta_smile, delta_gender)
    torch.save(delta_smile_orth, ORTH_OUT)
    print(f"saved {ORTH_OUT}")

    # Report magnitudes.
    print(f"{'t':>6} {'||smile||':>12} {'||smile_orth||':>16} {'gender comp':>14}")
    for t in TIMESTEPS:
        n_orig = delta_smile[t].norm().item()
        n_orth = delta_smile_orth[t].norm().item()
        gender_comp = ((n_orig ** 2 - n_orth ** 2) ** 0.5) if n_orig > n_orth else 0.0
        print(f"{t:>6} {n_orig:>12.2f} {n_orth:>16.2f} {gender_comp:>14.2f}")

    # Fixed-identity sampling.
    torch.manual_seed(0)
    x_T = torch.randn(1, 3, 256, 256, device=device)

    print(f"sweeping s ∈ {S_VALUES} for original vs orthogonalized ...")
    decoded_orig = []
    decoded_orth = []
    for s in S_VALUES:
        x_o = denoise_with_injection(unet, scheduler, x_T, [delta_smile], [s],
                                     NUM_DDIM_STEPS, device)
        decoded_orig.append((s, to_uint8_image(x_o)))
        x_p = denoise_with_injection(unet, scheduler, x_T, [delta_smile_orth], [s],
                                     NUM_DDIM_STEPS, device)
        decoded_orth.append((s, to_uint8_image(x_p)))
        print(f"  s={s:+.1f} ok (orig + orth)")

    n_cols = len(S_VALUES)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.0 * n_cols, 6.6))
    for i, (s, img) in enumerate(decoded_orig):
        ax = axes[0, i]
        ax.imshow(img)
        ax.set_title(f"s = {s:+.1f}", fontsize=11,
                     color=("crimson" if s > 0 else "darkblue" if s < 0 else "black"))
        ax.axis("off")
    for i, (_s, img) in enumerate(decoded_orth):
        ax = axes[1, i]
        ax.imshow(img)
        ax.axis("off")
    fig.text(0.01, 0.74, "smile (original)", rotation=90, fontsize=12, va="center")
    fig.text(0.01, 0.27, "smile ⊥ gender", rotation=90, fontsize=12,
             va="center", color="crimson")
    fig.suptitle("Orthogonalization comparison — same x_T, multi-step injection",
                 fontsize=12)
    fig.tight_layout(rect=(0.03, 0, 1, 0.97))
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150)

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    for s, img in decoded_orig:
        Image.fromarray(img).save(FRAMES_DIR / f"orig_s{s:+.1f}.png")
    for s, img in decoded_orth:
        Image.fromarray(img).save(FRAMES_DIR / f"orth_s{s:+.1f}.png")
    print(f"saved {FIG_OUT}")

    import os
    os._exit(0)


if __name__ == "__main__":
    main()
