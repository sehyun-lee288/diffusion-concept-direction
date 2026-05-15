"""Phase 24: many concept directions from a single point.

Phase 23 showed the top-5 h-space Jacobian singular vectors are
meaningful (but entangled). Here we extract a larger set (k=12) from
the *same single point* x_split at t≈700, decode each, and label each
direction by its cosine similarity with the supervised smile and
gender directions.

This directly answers "can we discover multiple concept directions
from one point": the top-k right singular vectors of ∂h/∂x span the
local semantic subspace — k distinct directions, all from one x.

Output:
  figures/exp25_h_jacobian_many.png  — 12 directions × (−, 0, +) decode
  data/h_jacobian_k12.pt             — the 12 directions + σ + cosines
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
from diffusion_boundary.hooks import MidBlockCapture  # noqa: E402
from diffusion_boundary.inversion import noise_to_t  # noqa: E402
from diffusion_boundary.multistep import collect_images_by_attribute  # noqa: E402

MODEL_ID = "google/ddpm-celebahq-256"
DATASET_ID = "eurecom-ds/celeba-hq-256"
ATTR_SMILING = 31
ATTR_MALE = 20
N_PER_CLASS = 20
TARGET_T = 700
NUM_DDIM_STEPS = 50
TOP_K = 12
N_ITER = 14
FD_EPS = 0.02

FIG_OUT = REPO_ROOT / "figures" / "exp25_h_jacobian_many.png"
DATA_OUT = REPO_ROOT / "data" / "h_jacobian_k12.pt"


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB").resize((256, 256), Image.BICUBIC),
                     dtype=np.float32) / 255.0
    return torch.from_numpy(arr * 2.0 - 1.0).permute(2, 0, 1).unsqueeze(0)


def _orthonormalize(vecs):
    out = []
    for v in vecs:
        w = v.clone()
        for u in out:
            w = w - (w.flatten() @ u.flatten()) * u
        n = w.norm()
        out.append(w / n if n > 1e-12 else torch.randn_like(v) / v.numel() ** 0.5)
    return out


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = DDPMPipeline.from_pretrained(MODEL_ID).to(device)
    unet = pipeline.unet
    scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    scheduler.set_timesteps(NUM_DDIM_STEPS)
    timesteps = scheduler.timesteps
    split_idx = int(torch.argmin(torch.abs(timesteps - TARGET_T)).item())
    t_split = int(timesteps[split_idx].item())
    t_dev = timesteps[split_idx].to(device)
    print(f"h-Jacobian (k={TOP_K}) at DDIM step {split_idx}, t={t_split}")

    torch.manual_seed(0)
    x = torch.randn(1, 3, 256, 256, device=device)
    for t in timesteps[:split_idx]:
        with torch.no_grad():
            eps = unet(x, t.to(device)).sample
        x = scheduler.step(eps, t, x).prev_sample
    x_split = x.clone()

    def h_fn(xx):
        with MidBlockCapture(unet) as cap, torch.no_grad():
            _ = unet(xx, t_dev).sample
        return cap.feature

    def jvp_h(xx, v, h=FD_EPS):
        return (h_fn(xx + h * v) - h_fn(xx - h * v)) / (2 * h)

    def vjp_h(xx, u):
        xr = xx.detach().requires_grad_(True)
        with MidBlockCapture(unet, detach=False) as cap:
            _ = unet(xr, t_dev).sample
        (jtu,) = torch.autograd.grad(cap.feature, xr, grad_outputs=u)
        return jtu.detach()

    print(f"block subspace iteration (k={TOP_K}, n_iter={N_ITER}) ...")
    V = _orthonormalize([torch.randn_like(x_split) for _ in range(TOP_K)])
    for it in range(N_ITER):
        JV = [jvp_h(x_split, v) for v in V]
        JtJV = [vjp_h(x_split, jv) for jv in JV]
        V = _orthonormalize(JtJV)
        if (it + 1) % 5 == 0:
            print(f"  iter {it + 1}/{N_ITER}")
    JV = [jvp_h(x_split, v) for v in V]
    flat = torch.stack([jv.flatten() for jv in JV])
    gram = flat @ flat.T
    eigvals, eigvecs = torch.linalg.eigh(gram)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order].clamp(min=0)
    eigvecs = eigvecs[:, order]
    sigmas = torch.sqrt(eigvals).cpu().numpy()
    Vmat = torch.stack([v.flatten() for v in V])
    sv = (eigvecs.T @ Vmat)
    svecs = [(sv[i].reshape(x_split.shape)) for i in range(TOP_K)]
    svecs = [v / v.norm() for v in svecs]
    print(f"  σ = {np.array2string(sigmas, precision=2)}")

    # Supervised reference directions (smile + gender) in x_700 space.
    print(f"streaming {DATASET_ID} for supervised references ...")

    def qmean(imgs, t):
        acc = []
        for i, im in enumerate(imgs):
            acc.append(noise_to_t(_pil_to_tensor(im).to(device), scheduler,
                                  target_t=t, seed=i))
        return torch.cat(acc).mean(0, keepdim=True)

    sm = collect_images_by_attribute(DATASET_ID, ATTR_SMILING, 1, N_PER_CLASS)
    ns = collect_images_by_attribute(DATASET_ID, ATTR_SMILING, 0, N_PER_CLASS)
    male = collect_images_by_attribute(DATASET_ID, ATTR_MALE, 1, N_PER_CLASS)
    fem = collect_images_by_attribute(DATASET_ID, ATTR_MALE, 0, N_PER_CLASS)
    d_smile = (qmean(sm, t_split) - qmean(ns, t_split)).reshape(-1)
    d_smile = d_smile / d_smile.norm()
    d_gender = (qmean(male, t_split) - qmean(fem, t_split)).reshape(-1)
    d_gender = d_gender / d_gender.norm()

    cos_smile, cos_gender = [], []
    for v in svecs:
        vf = v.reshape(-1)
        cos_smile.append(float(vf @ d_smile))
        cos_gender.append(float(vf @ d_gender))

    s_mag = 0.10 * x_split.norm().item()

    def continue_denoise(x_start):
        xx = x_start.clone()
        for t in timesteps[split_idx:]:
            with torch.no_grad():
                eps = unet(xx, t.to(device)).sample
            xx = scheduler.step(eps, t, xx).prev_sample
        return xx

    print(f"decoding {TOP_K} directions × 3 (−s, 0, +s) ...")
    base_img = to_uint8_image(continue_denoise(x_split))
    rows = []
    for i, v in enumerate(svecs):
        neg = to_uint8_image(continue_denoise(x_split - s_mag * v))
        pos = to_uint8_image(continue_denoise(x_split + s_mag * v))
        rows.append((neg, base_img, pos))
        print(f"  v{i+1:>2}  σ={sigmas[i]:6.2f}  "
              f"cos(smile)={cos_smile[i]:+.2f}  cos(gender)={cos_gender[i]:+.2f}")

    torch.save({"svecs": [v.cpu() for v in svecs], "sigmas": sigmas,
                "cos_smile": cos_smile, "cos_gender": cos_gender,
                "t_split": t_split}, DATA_OUT)

    # Figure: 12 rows × 3 cols.
    fig, axes = plt.subplots(TOP_K, 3, figsize=(8.2, 2.5 * TOP_K))
    col_titles = ["−s", "base", "+s"]
    for i in range(TOP_K):
        for j in range(3):
            ax = axes[i, j]
            ax.imshow(rows[i][j])
            ax.axis("off")
            if i == 0:
                ax.set_title(col_titles[j], fontsize=11)
        # label which concept this direction is closest to
        cs, cg = cos_smile[i], cos_gender[i]
        if abs(cs) > abs(cg) and abs(cs) > 0.12:
            tag = f"~smile ({cs:+.2f})"
        elif abs(cg) > 0.12:
            tag = f"~gender ({cg:+.2f})"
        else:
            tag = "other"
        axes[i, 0].set_ylabel(f"v{i+1}\nσ={sigmas[i]:.1f}\n{tag}",
                              fontsize=8, rotation=0, ha="right", va="center",
                              labelpad=34)
    fig.suptitle(f"Phase 24 — {TOP_K} h-Jacobian concept directions from ONE point (t={t_split})",
                 fontsize=12)
    fig.tight_layout(rect=(0.05, 0, 1, 0.98))
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150)
    print(f"\nsaved {FIG_OUT} and {DATA_OUT}")

    import os
    os._exit(0)


if __name__ == "__main__":
    main()
