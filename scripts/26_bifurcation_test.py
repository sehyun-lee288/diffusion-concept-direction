"""Phase 20: bifurcation test at the speciation point (t≈700).

Phase 18 located the smile speciation zone at t≈650-750. The
symmetry-breaking view of diffusion (Raya & Ambrogioni) predicts that
near the commit point a small perturbation can flip the trajectory to
the other attribute *basin*. We test it directly:

  1. Start a clean DDIM trajectory from x_T (seed 0); denoise to the
     step nearest t=700; capture x_700.
  2. Δx_700 = mean(q-sample smile, 700) − mean(q-sample no-smile, 700)
     — the smile direction in x_700-space.
  3. Sweep s; set x_700 ← x_700 + s·Δx_700; continue DDIM 700→0.
  4. Score each decoded x_0 by projecting onto the *pixel-space* smile
     direction Δpix = mean(smile imgs) − mean(no-smile imgs).
  5. Plot smile-score(s):
       sharp step  → genuine bifurcation / decision boundary
       smooth ramp → continuous control, no sharp boundary

A random-direction control sweep is included: if only the Δx direction
produces a transition, the boundary is direction-specific.

Output:
  figures/exp21_bifurcation.png — score(s) curve + decoded strip
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
from PIL import Image  # noqa: E402

from diffusion_boundary.decoding import to_uint8_image  # noqa: E402
from diffusion_boundary.inversion import noise_to_t  # noqa: E402
from diffusion_boundary.multistep import collect_images_by_attribute  # noqa: E402

MODEL_ID = "google/ddpm-celebahq-256"
DATASET_ID = "eurecom-ds/celeba-hq-256"
ATTR_SMILING = 31
N_PER_CLASS = 20
TARGET_T = 700
NUM_DDIM_STEPS = 50
S_VALUES = np.linspace(-6.0, 6.0, 25)
STRIP_S = [-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0]

FIG_OUT = REPO_ROOT / "figures" / "exp21_bifurcation.png"


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB").resize((256, 256), Image.BICUBIC),
                     dtype=np.float32) / 255.0
    return torch.from_numpy(arr * 2.0 - 1.0).permute(2, 0, 1).unsqueeze(0)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = DDPMPipeline.from_pretrained(MODEL_ID).to(device)
    unet = pipeline.unet
    scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    print(f"streaming {DATASET_ID} ...")
    smile_imgs = collect_images_by_attribute(DATASET_ID, ATTR_SMILING, 1, N_PER_CLASS)
    nosmile_imgs = collect_images_by_attribute(DATASET_ID, ATTR_SMILING, 0, N_PER_CLASS)
    smile_x0 = torch.cat([_pil_to_tensor(i) for i in smile_imgs]).to(device)
    nosmile_x0 = torch.cat([_pil_to_tensor(i) for i in nosmile_imgs]).to(device)

    # Pixel-space smile direction (for scoring outputs).
    delta_pix = (smile_x0.mean(0) - nosmile_x0.mean(0)).reshape(-1)
    delta_pix = delta_pix / delta_pix.norm()

    # x_700-space smile direction (for perturbing).
    def qsample_mean(x0_batch, t):
        accum = []
        for i in range(x0_batch.shape[0]):
            accum.append(noise_to_t(x0_batch[i:i + 1], scheduler, target_t=t, seed=i))
        return torch.cat(accum).mean(0, keepdim=True)
    dx_smile = qsample_mean(smile_x0, TARGET_T)
    dx_nosmile = qsample_mean(nosmile_x0, TARGET_T)
    delta_x = dx_smile - dx_nosmile
    print(f"||Δx_{TARGET_T}|| = {delta_x.norm().item():.3f}")

    # Random control direction, matched magnitude.
    torch.manual_seed(123)
    rand_dir = torch.randn_like(delta_x)
    rand_dir = rand_dir / rand_dir.norm() * delta_x.norm()

    # Clean trajectory from x_T → capture x at the step nearest TARGET_T.
    scheduler.set_timesteps(NUM_DDIM_STEPS)
    timesteps = scheduler.timesteps
    split_idx = int(torch.argmin(torch.abs(timesteps - TARGET_T)).item())
    t_split = int(timesteps[split_idx].item())
    print(f"perturbation injected at DDIM step {split_idx}, t={t_split}")

    torch.manual_seed(0)
    x = torch.randn(1, 3, 256, 256, device=device)
    for t in timesteps[:split_idx]:
        with torch.no_grad():
            eps = unet(x, t.to(device)).sample
        x = scheduler.step(eps, t, x).prev_sample
    x_split = x.clone()  # state at t_split

    def continue_denoise(x_start):
        xx = x_start.clone()
        for t in timesteps[split_idx:]:
            with torch.no_grad():
                eps = unet(xx, t.to(device)).sample
            xx = scheduler.step(eps, t, xx).prev_sample
        return xx

    def smile_score(x0):
        return float(x0.reshape(-1) @ delta_pix)

    print("sweeping Δx direction ...")
    scores_dx, scores_rand = [], []
    strip_imgs = []
    for s in S_VALUES:
        x0 = continue_denoise(x_split + s * delta_x)
        scores_dx.append(smile_score(x0))
        if any(abs(s - ss) < 1e-6 for ss in STRIP_S):
            strip_imgs.append((float(s), to_uint8_image(x0)))
        x0r = continue_denoise(x_split + s * rand_dir)
        scores_rand.append(smile_score(x0r))
        print(f"  s={s:+.2f}  smile_score(Δx)={scores_dx[-1]:+.2f}  "
              f"(rand)={scores_rand[-1]:+.2f}")

    # Figure: score curve (top) + decoded strip (bottom)
    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(2, len(STRIP_S), height_ratios=[2.0, 1.2])
    ax = fig.add_subplot(gs[0, :])
    ax.plot(S_VALUES, scores_dx, "o-", color="crimson", label="perturb along Δx (smile dir)")
    ax.plot(S_VALUES, scores_rand, "s--", color="gray", label="perturb along random dir (control)")
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("perturbation magnitude s   (x₇₀₀ ← x₇₀₀ + s·Δx)")
    ax.set_ylabel("smile score of decoded x₀")
    ax.set_title(f"Phase 20 — bifurcation test at t={t_split}: "
                 f"sharp step ⇒ decision boundary, smooth ramp ⇒ continuous control")
    ax.legend()
    ax.grid(alpha=0.3)

    for i, (s, img) in enumerate(strip_imgs):
        axi = fig.add_subplot(gs[1, i])
        axi.imshow(img)
        axi.set_title(f"s={s:+.0f}", fontsize=9)
        axi.axis("off")

    fig.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150)
    print(f"\nsaved {FIG_OUT}")

    import os
    os._exit(0)


if __name__ == "__main__":
    main()
