"""Phase 25: subspace + supervision hybrid.

Phase 24 gave a 12-D unsupervised semantic subspace S = span(v_1..v_12)
of h-Jacobian directions at t≈700, but the singular basis is entangled.
Hybrid idea: find S unsupervised, then use a *small* amount of
supervision (the labeled smile mean-shift Δx_smile) only to pick the
smile axis *within* S.

  smile_proj = Σ_i ⟨Δx_smile, v_i⟩ v_i      (projection of Δx onto S)
  capture ratio r = ‖smile_proj‖ / ‖Δx_smile‖

If r is high, the unsupervised subspace already contains the smile
concept and the supervision only rotates inside it. We decode along:
  (a) raw supervised Δx_smile
  (b) smile_proj  (the in-subspace smile axis)
  (c) projection of Δx_smile onto a RANDOM 12-D subspace (control)

Output: figures/exp26_subspace_supervision.png
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
S_VALUES = [-3.0, -1.5, 0.0, 1.5, 3.0]

K12_PATH = REPO_ROOT / "data" / "h_jacobian_k12.pt"
FIG_OUT = REPO_ROOT / "figures" / "exp26_subspace_supervision.png"


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB").resize((256, 256), Image.BICUBIC),
                     dtype=np.float32) / 255.0
    return torch.from_numpy(arr * 2.0 - 1.0).permute(2, 0, 1).unsqueeze(0)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = DDPMPipeline.from_pretrained(MODEL_ID).to(device)
    unet = pipeline.unet
    scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    scheduler.set_timesteps(NUM_DDIM_STEPS)
    timesteps = scheduler.timesteps
    split_idx = int(torch.argmin(torch.abs(timesteps - TARGET_T)).item())
    t_split = int(timesteps[split_idx].item())

    # 12-D subspace from Phase 24.
    k12 = torch.load(K12_PATH, weights_only=False)
    svecs = [v.to(device) for v in k12["svecs"]]
    V = torch.stack([v.reshape(-1) for v in svecs])  # (12, D), orthonormal rows
    D = V.shape[1]
    print(f"loaded 12-D subspace, D={D}")

    # Reproduce x_split (deterministic).
    torch.manual_seed(0)
    x = torch.randn(1, 3, 256, 256, device=device)
    for t in timesteps[:split_idx]:
        with torch.no_grad():
            eps = unet(x, t.to(device)).sample
        x = scheduler.step(eps, t, x).prev_sample
    x_split = x.clone()

    # Supervised smile direction Δx at t_split.
    print(f"streaming {DATASET_ID} ...")
    sm = collect_images_by_attribute(DATASET_ID, ATTR_SMILING, 1, N_PER_CLASS)
    ns = collect_images_by_attribute(DATASET_ID, ATTR_SMILING, 0, N_PER_CLASS)

    def qmean(imgs):
        acc = []
        for i, im in enumerate(imgs):
            acc.append(noise_to_t(_pil_to_tensor(im).to(device), scheduler,
                                  target_t=t_split, seed=i))
        return torch.cat(acc).mean(0, keepdim=True)
    d_smile = (qmean(sm) - qmean(ns)).reshape(-1)
    d_smile = d_smile / d_smile.norm()

    # Projection onto the 12-D Jacobian subspace.
    coeffs = V @ d_smile                       # (12,)
    smile_proj = coeffs @ V                    # (D,)
    capture = smile_proj.norm().item()         # since d_smile is unit
    print(f"capture ratio (Jacobian subspace) = {capture:.3f}")
    print(f"per-direction |coeff|: {np.array2string(coeffs.abs().cpu().numpy(), precision=3)}")

    # Random 12-D subspace control.
    torch.manual_seed(7)
    R = torch.randn(12, D, device=device)
    # Gram-Schmidt
    rows = []
    for i in range(12):
        w = R[i].clone()
        for u in rows:
            w = w - (w @ u) * u
        rows.append(w / w.norm())
    Vr = torch.stack(rows)
    smile_proj_rand = (Vr @ d_smile) @ Vr
    capture_rand = smile_proj_rand.norm().item()
    print(f"capture ratio (random subspace)   = {capture_rand:.3f}")

    smile_proj_u = smile_proj / smile_proj.norm()
    smile_proj_rand_u = smile_proj_rand / smile_proj_rand.norm()
    variants = [
        ("(a) raw supervised Δx_smile", d_smile.reshape(x_split.shape)),
        (f"(b) Δx projected onto Jac subspace (r={capture:.2f})",
         smile_proj_u.reshape(x_split.shape)),
        (f"(c) Δx projected onto RANDOM subspace (r={capture_rand:.2f})",
         smile_proj_rand_u.reshape(x_split.shape)),
    ]

    # ‖Δx_smile‖ in raw units to scale s consistently with Phase 20.
    raw_norm = 6.94  # ≈ ‖Δx_700‖ measured in Phase 20
    s_scale = raw_norm

    def continue_denoise(x_start):
        xx = x_start.clone()
        for t in timesteps[split_idx:]:
            with torch.no_grad():
                eps = unet(xx, t.to(device)).sample
            xx = scheduler.step(eps, t, xx).prev_sample
        return xx

    print("decoding sweeps ...")
    rows_imgs = []
    for label, direction in variants:
        imgs = []
        for s in S_VALUES:
            x0 = continue_denoise(x_split + s * s_scale * direction)
            imgs.append(to_uint8_image(x0))
        rows_imgs.append((label, imgs))
        print(f"  {label} done")

    fig, axes = plt.subplots(3, len(S_VALUES), figsize=(2.7 * len(S_VALUES), 8.4))
    for r, (label, imgs) in enumerate(rows_imgs):
        for c, (s, img) in enumerate(zip(S_VALUES, imgs, strict=True)):
            ax = axes[r, c]
            ax.imshow(img)
            ax.axis("off")
            if r == 0:
                ax.set_title(f"s={s:+.1f}", fontsize=11)
        fig.text(0.005, 1 - (r + 0.5) / 3, label, rotation=90,
                 fontsize=9, va="center", ha="center")
    fig.suptitle("Phase 25 — supervised smile axis: raw vs projected into the "
                 "unsupervised 12-D Jacobian subspace", fontsize=11)
    fig.tight_layout(rect=(0.04, 0, 1, 0.96))
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150)
    print(f"\nsaved {FIG_OUT}")

    import os
    os._exit(0)


if __name__ == "__main__":
    main()
