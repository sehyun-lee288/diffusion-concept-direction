"""Phase 23: unsupervised concept directions from the h-space Jacobian.

Phase 22 used the ε-output Jacobian ∂ε/∂x and found it near-isotropic
→ no clean concept directions. But Park et al. (arXiv:2302.12469)
power-iterate the *h-space* Jacobian ∂h/∂x — the Jacobian of the
bottleneck feature, not the network output. ε-Jacobian isotropy does
not imply h-Jacobian isotropy. This phase tests the h-space version.

J = ∂h/∂x_t maps x-perturbations (3·256·256) → h-perturbations
(512·8·8). Its top right-singular vectors are the x-directions that
most change the bottleneck representation — Park et al.'s definition
of an unsupervised semantic direction.

  - J·v   via central finite differences on h_fn (2 forward passes)
  - Jᵀ·u  via autograd grad(h, x, grad_outputs=u)   (u in h-space)

Same speciation timestep t≈700, same decode-and-inspect protocol as
Phase 22, including the cosine check against the supervised smile Δx.

Output:
  figures/exp24_h_jacobian_directions.png
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
N_PER_CLASS = 20
TARGET_T = 700
NUM_DDIM_STEPS = 50
TOP_K = 5
N_ITER = 12
FD_EPS = 0.02
N_S = 5

FIG_OUT = REPO_ROOT / "figures" / "exp24_h_jacobian_directions.png"


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
    print(f"h-space Jacobian at DDIM step {split_idx}, t={t_split}")

    # Clean trajectory → x_split.
    torch.manual_seed(0)
    x = torch.randn(1, 3, 256, 256, device=device)
    for t in timesteps[:split_idx]:
        with torch.no_grad():
            eps = unet(x, t.to(device)).sample
        x = scheduler.step(eps, t, x).prev_sample
    x_split = x.clone()
    print(f"||x_split|| = {x_split.norm().item():.2f}")

    # h_fn: bottleneck feature as a function of x (no grad).
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
        hh = cap.feature
        (jtu,) = torch.autograd.grad(hh, xr, grad_outputs=u)
        return jtu.detach()

    # Block subspace iteration for top-k right singular vectors of ∂h/∂x.
    print("computing top-k h-space Jacobian singular vectors ...")
    V = _orthonormalize([torch.randn_like(x_split) for _ in range(TOP_K)])
    for _ in range(N_ITER):
        JV = [jvp_h(x_split, v) for v in V]
        JtJV = [vjp_h(x_split, jv) for jv in JV]
        V = _orthonormalize(JtJV)
    JV = [jvp_h(x_split, v) for v in V]
    flat = torch.stack([jv.flatten() for jv in JV])      # (k, Dh)
    gram = flat @ flat.T
    eigvals, eigvecs = torch.linalg.eigh(gram)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order].clamp(min=0)
    eigvecs = eigvecs[:, order]
    sigmas = torch.sqrt(eigvals).cpu().numpy()
    Vmat = torch.stack([v.flatten() for v in V])
    sv = (eigvecs.T @ Vmat)
    svecs = [(-(sv[i].reshape(x_split.shape))).clone() for i in range(TOP_K)]  # sign arb.
    svecs = [v / v.norm() for v in svecs]
    print(f"  σ = {np.array2string(sigmas, precision=3)}")
    print(f"  σ1/σ5 ratio = {sigmas[0] / sigmas[-1]:.3f}")

    # Supervised smile direction Δx_700 for the cosine check.
    print(f"streaming {DATASET_ID} for supervised smile reference ...")
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
    cos_sims = []
    print("cosine similarity of each h-Jacobian direction with supervised smile Δx:")
    for i, v in enumerate(svecs):
        c = float(v.reshape(-1) @ delta_smile_u)
        cos_sims.append(c)
        print(f"  v{i + 1}: cos = {c:+.3f}")

    s_max = 0.10 * x_split.norm().item()
    s_values = np.linspace(-s_max, s_max, N_S)

    def continue_denoise(x_start):
        xx = x_start.clone()
        for t in timesteps[split_idx:]:
            with torch.no_grad():
                eps = unet(xx, t.to(device)).sample
            xx = scheduler.step(eps, t, xx).prev_sample
        return xx

    print("decoding perturbations along each h-Jacobian direction ...")
    grid = []
    for i, v in enumerate(svecs):
        row = [to_uint8_image(continue_denoise(x_split + s * v)) for s in s_values]
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
    fig.suptitle(f"Phase 23 — unsupervised h-space Jacobian (∂h/∂x) directions at t={t_split}",
                 fontsize=12)
    fig.tight_layout(rect=(0.04, 0, 1, 0.97))
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150)
    print(f"\nsaved {FIG_OUT}")

    import os
    os._exit(0)


if __name__ == "__main__":
    main()
