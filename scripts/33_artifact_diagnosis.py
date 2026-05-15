"""Phase 27: diagnose artifacts in the h-Jacobian direction sweeps.

In exp25 (Phase 24) some perturbed decodes are clean faces, others are
broken artifacts — often asymmetric (one of ±s is a face, the other is
garbage). Three candidate causes:

  H1  σ imbalance   — high-σ directions push h harder → off-distribution
  H2  |s| onset     — artifacts appear once the perturbation exceeds a
                      magnitude threshold, regardless of direction
  H3  manifold curvature — one sign moves off the data manifold, the
                      other along/toward it ⇒ sign-asymmetric artifacts

Artifact score = relative PCA residual against the real CelebA-HQ face
manifold. We build a PCA basis from 60 real faces (64×64); a decoded
image that lies on the face manifold projects with low residual, an
artifact has high residual. Then:

  - artifact vs |s| (per direction)  → tests H2
  - artifact level vs σ              → tests H1
  - artifact(+s) vs artifact(-s)     → tests H3

Output: figures/exp28_artifact_diagnosis.png
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
from sklearn.decomposition import PCA  # noqa: E402

from diffusion_boundary.decoding import to_uint8_image  # noqa: E402
from diffusion_boundary.multistep import collect_images_by_attribute  # noqa: E402

MODEL_ID = "google/ddpm-celebahq-256"
DATASET_ID = "eurecom-ds/celeba-hq-256"
TARGET_T = 700
NUM_DDIM_STEPS = 50
S_VALUES = [-66.0, -44.0, -22.0, 0.0, 22.0, 44.0, 66.0]
PCA_DIM = 40
FACE_SIZE = 64

K12_PATH = REPO_ROOT / "data" / "h_jacobian_k12.pt"
FIG_OUT = REPO_ROOT / "figures" / "exp28_artifact_diagnosis.png"


def _img_to_vec(uint8_img: np.ndarray) -> np.ndarray:
    small = Image.fromarray(uint8_img).resize((FACE_SIZE, FACE_SIZE), Image.BICUBIC)
    return np.asarray(small, dtype=np.float32).reshape(-1)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = DDPMPipeline.from_pretrained(MODEL_ID).to(device)
    unet = pipeline.unet
    scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    scheduler.set_timesteps(NUM_DDIM_STEPS)
    timesteps = scheduler.timesteps
    split_idx = int(torch.argmin(torch.abs(timesteps - TARGET_T)).item())

    k12 = torch.load(K12_PATH, weights_only=False)
    svecs = [v.to(device) for v in k12["svecs"]]
    sigmas = np.asarray(k12["sigmas"])
    print(f"loaded 12 directions, σ = {np.array2string(sigmas, precision=1)}")

    # Build a real-face PCA basis (artifact = high residual against it).
    print(f"streaming {DATASET_ID} for the real-face PCA basis ...")
    faces = (collect_images_by_attribute(DATASET_ID, 31, 1, 30)
             + collect_images_by_attribute(DATASET_ID, 31, 0, 30))
    face_vecs = np.stack([_img_to_vec(np.asarray(im.convert("RGB"))) for im in faces])
    face_mean = face_vecs.mean(0)
    pca = PCA(n_components=PCA_DIM).fit(face_vecs - face_mean)

    def artifact_score(uint8_img: np.ndarray) -> float:
        v = _img_to_vec(uint8_img) - face_mean
        recon = pca.inverse_transform(pca.transform(v[None]))[0]
        return float(np.linalg.norm(v - recon) / (np.linalg.norm(v) + 1e-9))

    # Calibration: residual of held-out real faces vs random noise.
    real_res = np.mean([artifact_score(np.asarray(im.convert("RGB").resize((256, 256))))
                        for im in faces[:10]])
    noise_res = artifact_score((np.random.rand(256, 256, 3) * 255).astype(np.uint8))
    print(f"calibration — real-face residual ≈ {real_res:.3f}, noise residual ≈ {noise_res:.3f}")

    torch.manual_seed(0)
    x = torch.randn(1, 3, 256, 256, device=device)
    for t in timesteps[:split_idx]:
        with torch.no_grad():
            eps = unet(x, t.to(device)).sample
        x = scheduler.step(eps, t, x).prev_sample
    x_split = x.clone()

    def continue_denoise(x_start):
        xx = x_start.clone()
        for t in timesteps[split_idx:]:
            with torch.no_grad():
                eps = unet(xx, t.to(device)).sample
            xx = scheduler.step(eps, t, xx).prev_sample
        return xx

    # artifact[i, j] = PCA residual for direction i at S_VALUES[j]
    artifact = np.zeros((12, len(S_VALUES)))
    for i, v in enumerate(svecs):
        for j, s in enumerate(S_VALUES):
            x0 = continue_denoise(x_split + s * v)
            artifact[i, j] = artifact_score(to_uint8_image(x0))
        print(f"  v{i+1:>2} (σ={sigmas[i]:.1f}): "
              f"residual = {np.array2string(artifact[i], precision=3)}")

    zero_col = S_VALUES.index(0.0)
    # H3: sign asymmetry — compare matched ±s magnitudes
    asym = []
    for i in range(12):
        a_plus = artifact[i, zero_col + 1:]
        a_minus = artifact[i, :zero_col][::-1]
        asym.append(float(np.mean(np.abs(a_plus - a_minus))))
    # H1: artifact at max |s| vs σ
    max_artifact = artifact[:, [0, -1]].max(axis=1)
    corr_sigma = float(np.corrcoef(sigmas, max_artifact)[0, 1])
    print(f"\nH1  corr(σ, max-artifact)         = {corr_sigma:+.3f}")
    print(f"H3  mean sign-asymmetry           = {np.mean(asym):.4f}")
    print(f"    baseline overshoot at s=0     = {artifact[:, zero_col].mean():.4f}")

    # ---- Figure ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (1) H2: artifact vs s, one curve per direction
    cmap = plt.get_cmap("viridis")
    for i in range(12):
        axes[0].plot(S_VALUES, artifact[i], "o-",
                     color=cmap(i / 11), label=f"v{i+1} σ={sigmas[i]:.0f}",
                     linewidth=1)
    axes[0].set_xlabel("perturbation s")
    axes[0].set_ylabel("PCA residual vs real faces  (artifact severity)")
    axes[0].set_title("H2: artifact vs s  (per direction)")
    axes[0].legend(fontsize=6, ncol=2)
    axes[0].grid(alpha=0.3)

    # (2) H1: max artifact vs σ
    axes[1].scatter(sigmas, max_artifact, c=range(12), cmap="viridis", s=80)
    for i in range(12):
        axes[1].annotate(f"v{i+1}", (sigmas[i], max_artifact[i]),
                         fontsize=8, xytext=(3, 3), textcoords="offset points")
    axes[1].set_xlabel("σ (h-Jacobian singular value)")
    axes[1].set_ylabel("max overshoot over the s-sweep")
    axes[1].set_title(f"H1: artifact vs σ   (corr = {corr_sigma:+.2f})")
    axes[1].grid(alpha=0.3)

    # (3) H3: artifact(+s) vs artifact(−s) at matched |s|
    a_plus_all, a_minus_all = [], []
    for i in range(12):
        a_plus_all.extend(artifact[i, zero_col + 1:])
        a_minus_all.extend(artifact[i, :zero_col][::-1])
    axes[2].scatter(a_minus_all, a_plus_all, c="crimson", alpha=0.6, s=40)
    lim = max(max(a_plus_all), max(a_minus_all)) * 1.1 + 1e-3
    axes[2].plot([0, lim], [0, lim], "k--", linewidth=0.8, label="symmetric")
    axes[2].set_xlabel("overshoot at −s")
    axes[2].set_ylabel("overshoot at +s")
    axes[2].set_title(f"H3: sign asymmetry   (mean |Δ| = {np.mean(asym):.3f})")
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    axes[2].set_xlim(0, lim)
    axes[2].set_ylim(0, lim)

    fig.suptitle("Phase 27 — what causes the artifacts in h-Jacobian direction sweeps?",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150)
    print(f"\nsaved {FIG_OUT}")

    import os
    os._exit(0)


if __name__ == "__main__":
    main()
