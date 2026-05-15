"""Phase 18: measure attribute speciation time.

Biroli et al. (Nat Comms 2024) define "speciation time" — when a
denoising trajectory commits to a class. Handke et al. (2506.10433)
find different attributes commit at different t. We test the CelebA-HQ
*attribute* analogue: how separable is "smile" at each noise level t?

Procedure (forward / q-sample proxy for the noise level):
  1. Collect 40 smile + 40 no-smile images from eurecom-ds/celeba-hq-256.
     Split: 20+20 FIT, 20+20 TEST (held-out → no circularity).
  2. For each t in {50, 150, …, 950}:
       - q-sample every image to t
       - h-space: capture mid_block h_t; Δh_t = mean(fit smile) −
         mean(fit no-smile); project TEST h onto Δh_t; d′ separation
       - x-space: same with the raw x_t tensor and Δx_t
  3. Plot d′(t) for h-space and x-space. A sharp rise = speciation.

The d′ here is measured on a held-out split so it reflects genuine
separability, not the fit. We use q-sample (forward diffusion) as a
cheap proxy for "the model's state at noise level t"; the true reverse
trajectory is left for a follow-up.

Output:
  data/speciation_dprime.npy   — (n_t, 2) array [d'_h, d'_x] per timestep
  figures/exp19_speciation_time.png
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

from diffusion_boundary.hooks import MidBlockCapture  # noqa: E402
from diffusion_boundary.inversion import noise_to_t  # noqa: E402
from diffusion_boundary.multistep import collect_images_by_attribute  # noqa: E402

MODEL_ID = "google/ddpm-celebahq-256"
DATASET_ID = "eurecom-ds/celeba-hq-256"
ATTR_SMILING = 31
N_PER_CLASS = 40           # 20 fit + 20 test
TIMESTEPS = [50, 150, 250, 350, 450, 550, 650, 750, 850, 950]

DATA_OUT = REPO_ROOT / "data" / "speciation_dprime.npy"
FIG_OUT = REPO_ROOT / "figures" / "exp19_speciation_time.png"


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB").resize((256, 256), Image.BICUBIC),
                     dtype=np.float32) / 255.0
    return torch.from_numpy(arr * 2.0 - 1.0).permute(2, 0, 1).unsqueeze(0)


def _d_prime(proj_pos: np.ndarray, proj_neg: np.ndarray) -> float:
    """Separation: |mean_pos − mean_neg| / avg-std."""
    s = (proj_pos.std() + proj_neg.std()) / 2.0
    return float(abs(proj_pos.mean() - proj_neg.mean()) / max(s, 1e-9))


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = DDPMPipeline.from_pretrained(MODEL_ID).to(device)
    unet = pipeline.unet
    scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    print(f"streaming {DATASET_ID} ...")
    smile_imgs = collect_images_by_attribute(DATASET_ID, ATTR_SMILING, 1, N_PER_CLASS)
    nosmile_imgs = collect_images_by_attribute(DATASET_ID, ATTR_SMILING, 0, N_PER_CLASS)
    print(f"  collected smile={len(smile_imgs)}, no-smile={len(nosmile_imgs)}")

    # Pre-convert to tensors.
    smile_x0 = [_pil_to_tensor(im).to(device) for im in smile_imgs]
    nosmile_x0 = [_pil_to_tensor(im).to(device) for im in nosmile_imgs]
    # fit / test split
    fit_s, test_s = smile_x0[:20], smile_x0[20:]
    fit_n, test_n = nosmile_x0[:20], nosmile_x0[20:]

    def capture_h_and_x(x0_list, t, seed_offset):
        """Return (h_flat (N, Dh), x_flat (N, Dx)) at noise level t."""
        hs, xs = [], []
        t_tensor = torch.tensor([t], device=device)
        for i, x0 in enumerate(x0_list):
            x_t = noise_to_t(x0, scheduler, target_t=t, seed=seed_offset + i)
            with MidBlockCapture(unet) as cap, torch.no_grad():
                _ = unet(x_t, t_tensor).sample
            hs.append(cap.feature.reshape(-1).cpu())
            xs.append(x_t.reshape(-1).cpu())
        return torch.stack(hs), torch.stack(xs)

    results = []  # (t, d'_h, d'_x)
    for t in TIMESTEPS:
        # seeds disjoint across fit/test so noise is independent
        h_fs, x_fs = capture_h_and_x(fit_s, t, 0)
        h_fn, x_fn = capture_h_and_x(fit_n, t, 100)
        h_ts, x_ts = capture_h_and_x(test_s, t, 200)
        h_tn, x_tn = capture_h_and_x(test_n, t, 300)

        # direction from FIT, projection measured on TEST
        dh = (h_fs.mean(0) - h_fn.mean(0))
        dh = dh / dh.norm().clamp(min=1e-9)
        dx = (x_fs.mean(0) - x_fn.mean(0))
        dx = dx / dx.norm().clamp(min=1e-9)

        dp_h = _d_prime((h_ts @ dh).numpy(), (h_tn @ dh).numpy())
        dp_x = _d_prime((x_ts @ dx).numpy(), (x_tn @ dx).numpy())
        results.append((t, dp_h, dp_x))
        print(f"  t={t:>3}  d'_h = {dp_h:6.3f}   d'_x = {dp_x:6.3f}")

    arr = np.array([[r[1], r[2]] for r in results])
    DATA_OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(DATA_OUT, arr)

    ts = [r[0] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ts, [r[1] for r in results], "o-", color="crimson",
            label="h-space (mid_block)")
    ax.plot(ts, [r[2] for r in results], "s-", color="steelblue",
            label="x-space (noisy image)")
    ax.set_xlabel("timestep t  (higher = noisier)")
    ax.set_ylabel("d′  (smile vs no-smile separation, held-out)")
    ax.set_title("Phase 18 — attribute speciation: smile separability vs noise level")
    ax.legend()
    ax.grid(alpha=0.3)
    # mark steepest-rise interval for h-space
    dp_h = np.array([r[1] for r in results])
    slopes = np.diff(dp_h) / np.diff(ts)
    steep = int(np.argmin(slopes))  # most negative slope = sharpest rise as t↓
    ax.axvspan(ts[steep + 1], ts[steep], color="orange", alpha=0.15,
               label="steepest h-space rise")
    ax.annotate(f"steepest rise\nt∈[{ts[steep+1]}, {ts[steep]}]",
                xy=((ts[steep] + ts[steep + 1]) / 2, dp_h[steep:steep + 2].mean()),
                fontsize=9, ha="center")
    fig.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150)
    print(f"\nsaved {FIG_OUT} and {DATA_OUT}")
    print(f"steepest h-space d' rise in t ∈ [{ts[steep+1]}, {ts[steep]}]")

    import os
    os._exit(0)


if __name__ == "__main__":
    main()
