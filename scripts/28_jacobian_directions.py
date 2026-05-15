"""Phase 22: unsupervised concept directions from the score-Jacobian.

The user's question: can we discover concept directions by sampling
around a single point — without attribute labels? Polytope sampling
(Phase 17) cannot. But the *local Jacobian* can: the top right-singular
vectors of ∂ε_θ(x_t,t)/∂x_t are the directions that most change the
denoiser output, i.e., candidate concept axes (GANSpace / Park et al.
arXiv:2302.12469 do exactly this).

Procedure:
  1. Clean DDIM trajectory from x_T (seed 0) → state x_split at the step
     nearest t=700 (the speciation zone from Phase 18).
  2. Block subspace iteration for the top-k right singular vectors of
     J = ∂ε/∂x_split (FD-jvp + autograd-vjp), rotated by the Gram-matrix
     eigenvectors so they are properly ordered.
  3. For each singular vector v_i, sweep s and decode x_split + s·v_i
     to x_0.
  4. Quantitative check: cosine similarity of each v_i with the
     *supervised* smile direction Δx_700 (from labeled images). A high
     cos-sim means we rediscovered "smile" unsupervised.

Output:
  figures/exp23_jacobian_directions.png — k×(s-sweep) decoded grid
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
TOP_K = 5
N_ITER = 12
FD_EPS = 0.02
N_S = 5  # s-sweep points per direction

FIG_OUT = REPO_ROOT / "figures" / "exp23_jacobian_directions.png"


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


def jvp_fd(eps_fn, x, v, h=FD_EPS):
    with torch.no_grad():
        return (eps_fn(x + h * v) - eps_fn(x - h * v)) / (2 * h)


def vjp_ad(eps_fn, x, u):
    xr = x.detach().requires_grad_(True)
    y = eps_fn(xr)
    (jtu,) = torch.autograd.grad(y, xr, grad_outputs=u)
    return jtu.detach()


def top_k_singular_vectors(eps_fn, x, k, n_iter):
    """Return (sigmas, [v_1..v_k]) — ordered right singular vectors of ∂eps/∂x."""
    V = _orthonormalize([torch.randn_like(x) for _ in range(k)])
    for _ in range(n_iter):
        JV = [jvp_fd(eps_fn, x, v) for v in V]
        JtJV = [vjp_ad(eps_fn, x, jv) for jv in JV]
        V = _orthonormalize(JtJV)
    JV = [jvp_fd(eps_fn, x, v) for v in V]
    flat = torch.stack([jv.flatten() for jv in JV])      # (k, D)
    gram = flat @ flat.T                                 # (k, k)
    eigvals, eigvecs = torch.linalg.eigh(gram)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order].clamp(min=0)
    eigvecs = eigvecs[:, order]
    sigmas = torch.sqrt(eigvals)
    Vmat = torch.stack([v.flatten() for v in V])         # (k, D)
    sv = (eigvecs.T @ Vmat)                              # (k, D) ordered
    sv = [sv[i].reshape(x.shape) for i in range(k)]
    sv = [v / v.norm() for v in sv]
    return sigmas.cpu().numpy(), sv


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = DDPMPipeline.from_pretrained(MODEL_ID).to(device)
    unet = pipeline.unet
    scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    scheduler.set_timesteps(NUM_DDIM_STEPS)
    timesteps = scheduler.timesteps
    split_idx = int(torch.argmin(torch.abs(timesteps - TARGET_T)).item())
    t_split = int(timesteps[split_idx].item())
    print(f"speciation analysis at DDIM step {split_idx}, t={t_split}")

    # Clean trajectory → x_split.
    torch.manual_seed(0)
    x = torch.randn(1, 3, 256, 256, device=device)
    for t in timesteps[:split_idx]:
        with torch.no_grad():
            eps = unet(x, t.to(device)).sample
        x = scheduler.step(eps, t, x).prev_sample
    x_split = x.clone()
    print(f"||x_split|| = {x_split.norm().item():.2f}")

    # Top-k singular vectors of ∂eps/∂x at t_split.
    t_dev = timesteps[split_idx].to(device)

    def eps_fn(xx):
        return unet(xx, t_dev).sample

    print("computing top-k Jacobian singular vectors ...")
    sigmas, svecs = top_k_singular_vectors(eps_fn, x_split, TOP_K, N_ITER)
    print(f"  σ = {np.array2string(sigmas, precision=3)}")

    # Supervised smile direction Δx_700 for the cosine check.
    print(f"streaming {DATASET_ID} for the supervised smile reference ...")
    smile_imgs = collect_images_by_attribute(DATASET_ID, ATTR_SMILING, 1, N_PER_CLASS)
    nosmile_imgs = collect_images_by_attribute(DATASET_ID, ATTR_SMILING, 0, N_PER_CLASS)

    def qsample_mean(imgs, t):
        accum = []
        for i, im in enumerate(imgs):
            x0 = _pil_to_tensor(im).to(device)
            accum.append(noise_to_t(x0, scheduler, target_t=t, seed=i))
        return torch.cat(accum).mean(0, keepdim=True)
    delta_smile = qsample_mean(smile_imgs, t_split) - qsample_mean(nosmile_imgs, t_split)
    delta_smile_u = (delta_smile / delta_smile.norm()).reshape(-1)
    print("cosine similarity of each Jacobian direction with supervised smile Δx:")
    cos_sims = []
    for i, v in enumerate(svecs):
        c = float(v.reshape(-1) @ delta_smile_u)
        cos_sims.append(c)
        print(f"  v{i + 1}: cos = {c:+.3f}")

    # s-sweep magnitude: a fraction of ||x_split||.
    s_max = 0.10 * x_split.norm().item()
    s_values = np.linspace(-s_max, s_max, N_S)
    print(f"s-sweep: {np.array2string(s_values, precision=1)}")

    def continue_denoise(x_start):
        xx = x_start.clone()
        for t in timesteps[split_idx:]:
            with torch.no_grad():
                eps = unet(xx, t.to(device)).sample
            xx = scheduler.step(eps, t, xx).prev_sample
        return xx

    print("decoding perturbations along each singular vector ...")
    grid = []  # rows: directions, cols: s
    for i, v in enumerate(svecs):
        row = []
        for s in s_values:
            x0 = continue_denoise(x_split + s * v)
            row.append(to_uint8_image(x0))
        grid.append(row)
        print(f"  v{i + 1} done")

    fig, axes = plt.subplots(TOP_K, N_S, figsize=(2.6 * N_S, 2.7 * TOP_K))
    for i in range(TOP_K):
        for j in range(N_S):
            ax = axes[i, j]
            ax.imshow(grid[i][j])
            ax.axis("off")
            if i == 0:
                ax.set_title(f"s={s_values[j]:+.0f}", fontsize=10)
        axes[i, 0].set_ylabel(f"v{i+1}\nσ={sigmas[i]:.2f}\ncos(smile)={cos_sims[i]:+.2f}",
                              fontsize=9, rotation=0, ha="right", va="center", labelpad=38)
    fig.suptitle(f"Phase 22 — unsupervised Jacobian concept directions at t={t_split}",
                 fontsize=12)
    fig.tight_layout(rect=(0.04, 0, 1, 0.97))
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150)
    print(f"\nsaved {FIG_OUT}")

    import os
    os._exit(0)


if __name__ == "__main__":
    main()
