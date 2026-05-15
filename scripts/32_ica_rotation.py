"""Phase 26: ICA rotation of the h-Jacobian subspace.

The 12 h-Jacobian singular vectors (Phase 24) span a semantic subspace
but are entangled — each mixes attributes. ICA finds a rotation of the
basis whose components have statistically independent effects, an
unsupervised route to a more disentangled basis.

Procedure:
  1. Load the 12 directions; reproduce x_split at t≈700.
  2. Decode the effect of each direction: e_i = decode(x+s·v_i) −
     decode(x−s·v_i), at 64×64 resolution.
  3. Stack E (12 × pixels). Run FastICA(n_components=12) treating
     pixels as samples and the 12 directions as features → 12×12
     unmixing matrix W.
  4. Rotated (ICA) directions: ṽ_k = Σ_j W[k,j] v_j.
  5. Decode along each ICA direction; check cosine with supervised
     smile / gender — does ICA concentrate an attribute into one axis?

Output: figures/exp27_ica_rotation.png
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
from sklearn.decomposition import FastICA  # noqa: E402

from diffusion_boundary.decoding import to_uint8_image  # noqa: E402
from diffusion_boundary.inversion import noise_to_t  # noqa: E402
from diffusion_boundary.multistep import collect_images_by_attribute  # noqa: E402

MODEL_ID = "google/ddpm-celebahq-256"
DATASET_ID = "eurecom-ds/celeba-hq-256"
ATTR_SMILING = 31
ATTR_MALE = 20
N_PER_CLASS = 20
TARGET_T = 700
NUM_DDIM_STEPS = 50
K = 12
EFFECT_S = 30.0          # perturbation magnitude for measuring effects

K12_PATH = REPO_ROOT / "data" / "h_jacobian_k12.pt"
FIG_OUT = REPO_ROOT / "figures" / "exp27_ica_rotation.png"


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB").resize((256, 256), Image.BICUBIC),
                     dtype=np.float32) / 255.0
    return torch.from_numpy(arr * 2.0 - 1.0).permute(2, 0, 1).unsqueeze(0)


def _downsample_flat(uint8_img, size=64):
    return np.asarray(Image.fromarray(uint8_img).resize((size, size),
                      Image.BICUBIC), dtype=np.float32).reshape(-1)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = DDPMPipeline.from_pretrained(MODEL_ID).to(device)
    unet = pipeline.unet
    scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    scheduler.set_timesteps(NUM_DDIM_STEPS)
    timesteps = scheduler.timesteps
    split_idx = int(torch.argmin(torch.abs(timesteps - TARGET_T)).item())
    t_split = int(timesteps[split_idx].item())

    k12 = torch.load(K12_PATH, weights_only=False)
    svecs = [v.to(device) for v in k12["svecs"]]

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

    # Measure each direction's image effect e_i = decode(+) − decode(−).
    print(f"decoding ± effects for {K} directions ...")
    effects = []
    for i, v in enumerate(svecs):
        pos = _downsample_flat(to_uint8_image(continue_denoise(x_split + EFFECT_S * v)))
        neg = _downsample_flat(to_uint8_image(continue_denoise(x_split - EFFECT_S * v)))
        effects.append(pos - neg)
        print(f"  v{i+1} effect computed")
    E = np.stack(effects)          # (12, pixels)

    # ICA: pixels are samples, the 12 directions are the mixed signals.
    print("running FastICA (n_components=12) ...")
    ica = FastICA(n_components=K, random_state=0, max_iter=2000, whiten="unit-variance")
    ica.fit(E.T)                   # fit on (pixels, 12)
    W = ica.components_            # (12, 12) unmixing in direction space
    # Normalize rows so each ICA direction is a unit rotation of the v's.
    W = W / np.linalg.norm(W, axis=1, keepdims=True)

    # Rotated directions ṽ_k = Σ_j W[k,j] v_j.
    Vmat = torch.stack([v.reshape(-1) for v in svecs])     # (12, D)
    Wt = torch.tensor(W, dtype=Vmat.dtype, device=device)
    ica_dirs = Wt @ Vmat                                    # (12, D)
    ica_dirs = ica_dirs / ica_dirs.norm(dim=1, keepdim=True)
    ica_dirs = [ica_dirs[i].reshape(x_split.shape) for i in range(K)]

    # Supervised references for the cosine check.
    print(f"streaming {DATASET_ID} for supervised references ...")

    def qmean(imgs):
        acc = []
        for i, im in enumerate(imgs):
            acc.append(noise_to_t(_pil_to_tensor(im).to(device), scheduler,
                                  target_t=t_split, seed=i))
        return torch.cat(acc).mean(0, keepdim=True)
    sm = collect_images_by_attribute(DATASET_ID, ATTR_SMILING, 1, N_PER_CLASS)
    ns = collect_images_by_attribute(DATASET_ID, ATTR_SMILING, 0, N_PER_CLASS)
    male = collect_images_by_attribute(DATASET_ID, ATTR_MALE, 1, N_PER_CLASS)
    fem = collect_images_by_attribute(DATASET_ID, ATTR_MALE, 0, N_PER_CLASS)
    d_smile = (qmean(sm) - qmean(ns)).reshape(-1)
    d_smile = d_smile / d_smile.norm()
    d_gender = (qmean(male) - qmean(fem)).reshape(-1)
    d_gender = d_gender / d_gender.norm()

    # Cosine of original vs ICA-rotated directions with smile/gender.
    def cos_table(dirs):
        return [(float(d.reshape(-1) @ d_smile), float(d.reshape(-1) @ d_gender))
                for d in dirs]
    cos_orig = cos_table(svecs)
    cos_ica = cos_table(ica_dirs)
    best_orig_smile = max(abs(c[0]) for c in cos_orig)
    best_ica_smile = max(abs(c[0]) for c in cos_ica)
    best_orig_gender = max(abs(c[1]) for c in cos_orig)
    best_ica_gender = max(abs(c[1]) for c in cos_ica)
    print(f"best |cos(smile)|:  original {best_orig_smile:.3f}  →  ICA {best_ica_smile:.3f}")
    print(f"best |cos(gender)|: original {best_orig_gender:.3f}  →  ICA {best_ica_gender:.3f}")

    # Decode each ICA direction (−, base, +).
    print("decoding ICA-rotated directions ...")
    base = to_uint8_image(continue_denoise(x_split))
    rows = []
    for i, v in enumerate(ica_dirs):
        neg = to_uint8_image(continue_denoise(x_split - EFFECT_S * v))
        pos = to_uint8_image(continue_denoise(x_split + EFFECT_S * v))
        rows.append((neg, base, pos))
        cs, cg = cos_ica[i]
        print(f"  ICA dir {i+1}: cos(smile)={cs:+.2f}  cos(gender)={cg:+.2f}")

    fig, axes = plt.subplots(K, 3, figsize=(8.2, 2.5 * K))
    for i in range(K):
        for j in range(3):
            ax = axes[i, j]
            ax.imshow(rows[i][j])
            ax.axis("off")
            if i == 0:
                ax.set_title(["−s", "base", "+s"][j], fontsize=11)
        cs, cg = cos_ica[i]
        axes[i, 0].set_ylabel(f"IC{i+1}\ncos_sm={cs:+.2f}\ncos_gd={cg:+.2f}",
                              fontsize=8, rotation=0, ha="right", va="center",
                              labelpad=34)
    fig.suptitle(f"Phase 26 — ICA-rotated h-Jacobian basis  "
                 f"(best |cos(smile)|: {best_orig_smile:.2f}→{best_ica_smile:.2f})",
                 fontsize=11)
    fig.tight_layout(rect=(0.05, 0, 1, 0.98))
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150)
    print(f"\nsaved {FIG_OUT}")

    import os
    os._exit(0)


if __name__ == "__main__":
    main()
