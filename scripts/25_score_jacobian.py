"""Phase 19: track the score-Jacobian eigenspectrum along the trajectory.

Following the geometry-of-diffusion literature (Stanczuk et al. 2024
intrinsic dimension; Achilli et al. 2024 geometric memorization), the
spectrum of the denoiser Jacobian J_t = ∂ε_θ(x_t, t) / ∂x_t is a
trajectory-level diagnostic: where a few eigendirections become
dominant, the trajectory is "committing" to structure.

We run a clean DDIM trajectory and, at selected timesteps, estimate the
top-k singular values of J_t by block subspace iteration:
  - J·v   via central finite differences (2 forward passes, no AD graph)
  - Jᵀ·u  via autograd vjp (1 backward pass)
  - converged block V spans the top-k right-singular subspace; the
    k×k Gram matrix of {J v_i} gives σ_i².

Output:
  data/score_jacobian_spectrum.npy  — (n_t, k) singular values
  figures/exp20_score_jacobian.png  — spectrum + concentration vs t
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

MODEL_ID = "google/ddpm-celebahq-256"
NUM_DDIM_STEPS = 50
CHECK_STEP_INDICES = [3, 11, 19, 27, 35, 43]   # spread across the 50-step schedule
TOP_K = 4
N_ITER = 10
FD_EPS = 0.02

DATA_OUT = REPO_ROOT / "data" / "score_jacobian_spectrum.npy"
FIG_OUT = REPO_ROOT / "figures" / "exp20_score_jacobian.png"


def _orthonormalize(vecs: list[torch.Tensor]) -> list[torch.Tensor]:
    """Gram-Schmidt on a list of same-shape tensors."""
    out: list[torch.Tensor] = []
    for v in vecs:
        w = v.clone()
        for u in out:
            w = w - (w.flatten() @ u.flatten()) * u
        n = w.norm()
        if n > 1e-12:
            out.append(w / n)
        else:
            out.append(torch.randn_like(v) / v.numel() ** 0.5)
    return out


def jvp_fd(eps_fn, x, v, h=FD_EPS):
    """Central finite-difference Jacobian-vector product (no autograd graph)."""
    with torch.no_grad():
        return (eps_fn(x + h * v) - eps_fn(x - h * v)) / (2 * h)


def vjp_ad(eps_fn, x, u):
    """Autograd vector-Jacobian product Jᵀu."""
    xr = x.detach().requires_grad_(True)
    y = eps_fn(xr)
    (jtu,) = torch.autograd.grad(y, xr, grad_outputs=u)
    return jtu.detach()


def top_k_singular(eps_fn, x, k: int, n_iter: int) -> np.ndarray:
    """Block subspace iteration for the top-k singular values of ∂eps/∂x."""
    V = _orthonormalize([torch.randn_like(x) for _ in range(k)])
    for _ in range(n_iter):
        JV = [jvp_fd(eps_fn, x, v) for v in V]
        JtJV = [vjp_ad(eps_fn, x, jv) for jv in JV]
        V = _orthonormalize(JtJV)
    JV = [jvp_fd(eps_fn, x, v) for v in V]
    flat = torch.stack([jv.flatten() for jv in JV])   # (k, D)
    gram = flat @ flat.T                              # (k, k) ≈ Vᵀ JᵀJ V
    eigvals = torch.linalg.eigvalsh(gram).clamp(min=0)
    sigmas = torch.sqrt(eigvals).sort(descending=True).values
    return sigmas.cpu().numpy()


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = DDPMPipeline.from_pretrained(MODEL_ID).to(device)
    unet = pipeline.unet
    scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    scheduler.set_timesteps(NUM_DDIM_STEPS)

    torch.manual_seed(0)
    x = torch.randn(1, 3, 256, 256, device=device)

    rows = []  # (t_int, sigmas)
    check = set(CHECK_STEP_INDICES)
    for idx, t in enumerate(scheduler.timesteps):
        t_dev = t.to(device)
        if idx in check:
            def eps_fn(xx, _t=t_dev):
                return unet(xx, _t).sample
            sigmas = top_k_singular(eps_fn, x, TOP_K, N_ITER)
            rows.append((int(t.item()), sigmas))
            conc = sigmas[0] ** 2 / (sigmas ** 2).sum()
            print(f"  t={int(t.item()):>3}  σ = {np.array2string(sigmas, precision=3)}  "
                  f"σ1²-share = {conc:.3f}")
        with torch.no_grad():
            eps = unet(x, t_dev).sample
        x = scheduler.step(eps, t, x).prev_sample

    ts = [r[0] for r in rows]
    spectra = np.array([r[1] for r in rows])  # (n_t, k)
    DATA_OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(DATA_OUT, spectra)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for j in range(TOP_K):
        ax1.plot(ts, spectra[:, j], "o-", label=f"σ{j + 1}")
    ax1.set_xlabel("timestep t  (higher = noisier)")
    ax1.set_ylabel("singular value of ∂ε/∂x")
    ax1.set_title("Score-Jacobian top-k spectrum along trajectory")
    ax1.legend()
    ax1.grid(alpha=0.3)

    conc = spectra[:, 0] ** 2 / (spectra ** 2).sum(axis=1)
    ax2.plot(ts, conc, "o-", color="purple")
    ax2.set_xlabel("timestep t")
    ax2.set_ylabel("σ₁² / Σσ²  (top-direction concentration)")
    ax2.set_title("Spectral concentration — higher = trajectory committing")
    ax2.grid(alpha=0.3)

    fig.suptitle("Phase 19 — score-Jacobian eigenspectrum as a commit diagnostic",
                 fontsize=12)
    fig.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150)
    print(f"\nsaved {FIG_OUT} and {DATA_OUT}")

    import os
    os._exit(0)


if __name__ == "__main__":
    main()
