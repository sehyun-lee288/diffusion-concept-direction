"""Phase 10: feasibility probe for h-only attribute editing.

Does the U-Net mid_block at t=500 carry an editable smile direction?
If yes, supervised mean-shift Δh injection should visibly add/remove
smile from a decoded image. If no, h-space alone is too weak and we
need skip-aware editing (FINDINGS §8 item 4).

Procedure:
  1. Stream `eurecom-ds/celeba-hq-256` (40-attribute CelebA-HQ at 256).
     Collect N=20 images per class for Smiling (attr index 31).
  2. For each image, q-sample to t=500 (seed = image_idx so noise is
     varied), forward through U-Net, capture h via MidBlockCapture.
  3. Δh_smile = mean(h | Smiling=1) − mean(h | Smiling=0).
  4. Use anchor 0's (x_500, h_test) as the injection target. Sweep
     s ∈ {-3, -2, -1, 0, 1, 2, 3}, inject `h_test + s·Δh_smile`,
     DDIM-denoise to t=0. Save the 7 decoded images.
  5. Side-by-side figure: baseline anchor 0 + 7 sweeps.

If the decoded faces show a visible monotonic smile change across s,
Phase 11 (full attribute-paired analysis) is justified. If not, pivot.

Outputs:
  data/delta_h_smile.pt           — saved direction (1, 512, 8, 8) for reuse
  figures/exp6_smile_probe.png    — sweep figure
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from diffusers import DDIMScheduler, DDPMPipeline  # noqa: E402
from PIL import Image  # noqa: E402

from diffusion_boundary.decoding import decode_from_h, to_uint8_image  # noqa: E402
from diffusion_boundary.hooks import MidBlockCapture  # noqa: E402
from diffusion_boundary.inversion import noise_to_t  # noqa: E402

MODEL_ID = "google/ddpm-celebahq-256"
DATASET_ID = "eurecom-ds/celeba-hq-256"
ATTR_IDX_SMILING = 31
N_PER_CLASS = 20
TARGET_T = 500
S_VALUES = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
CHANNELS = 512
SPATIAL = 8

ANCHOR_DIR = REPO_ROOT / "data" / "anchors"
DELTA_OUT = REPO_ROOT / "data" / "delta_h_smile.pt"
FIG_OUT = REPO_ROOT / "figures" / "exp6_smile_probe.png"


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL 256x256 → (1, 3, 256, 256) in [-1, 1]."""
    arr = np.asarray(img.convert("RGB").resize((256, 256), Image.BICUBIC),
                     dtype=np.float32) / 255.0
    return torch.from_numpy(arr * 2.0 - 1.0).permute(2, 0, 1).unsqueeze(0)


def _collect_images_by_attribute(target_val: int, n_needed: int) -> list:
    """Stream eurecom-ds/celeba-hq-256; return first n_needed images whose
    attributes[ATTR_IDX_SMILING] == target_val. PIL copies — dataset
    iterator is released inside this function."""
    ds = load_dataset(DATASET_ID, split="train", streaming=True)
    out = []
    try:
        for sample in ds:
            if int(sample["attributes"][ATTR_IDX_SMILING]) == target_val:
                out.append(sample["image"].copy())
                if len(out) >= n_needed:
                    break
    finally:
        del ds
        gc.collect()
    return out


def _capture_h_batch(unet, scheduler, images: list, device: str) -> torch.Tensor:
    """For each image, q-sample to TARGET_T with seed=index and capture
    the mid_block output. Returns stacked tensor (N, 512, 8, 8) on CPU."""
    hs = []
    t_tensor = torch.tensor([TARGET_T], device=device)
    for i, img in enumerate(images):
        x0 = _pil_to_tensor(img).to(device)
        x_t = noise_to_t(x0, scheduler, target_t=TARGET_T, seed=i)
        with MidBlockCapture(unet) as cap, torch.no_grad():
            _ = unet(x_t, t_tensor).sample
        hs.append(cap.feature.cpu())
        if (i + 1) % 5 == 0:
            print(f"  captured {i + 1}/{len(images)}")
    return torch.cat(hs, dim=0)  # (N, 512, 8, 8)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {MODEL_ID} ...")
    pipeline = DDPMPipeline.from_pretrained(MODEL_ID)
    pipeline.to(device)
    unet = pipeline.unet
    scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    print(f"Streaming {DATASET_ID} for {N_PER_CLASS}× Smiling=1 and =0 ...")
    smile_imgs = _collect_images_by_attribute(1, N_PER_CLASS)
    nosmile_imgs = _collect_images_by_attribute(0, N_PER_CLASS)
    print(f"  collected: smile={len(smile_imgs)}, no-smile={len(nosmile_imgs)}")

    print("Capturing h-vectors for Smiling=1 ...")
    h_smile = _capture_h_batch(unet, scheduler, smile_imgs, device)
    print("Capturing h-vectors for Smiling=0 ...")
    h_nosmile = _capture_h_batch(unet, scheduler, nosmile_imgs, device)

    delta_h = h_smile.mean(0, keepdim=True) - h_nosmile.mean(0, keepdim=True)
    delta_norm = delta_h.norm().item()
    print(f"||Δh|| = {delta_norm:.3f}  (over {N_PER_CLASS} per class)")
    torch.save(delta_h, DELTA_OUT)
    print(f"saved {DELTA_OUT}")

    # Test image: pick the first no-smile face we collected (known label,
    # known to be frontal-ish since CelebA-HQ is screened for it).
    test_img = nosmile_imgs[0]
    test_pil_256 = test_img.convert("RGB").resize((256, 256), Image.BICUBIC)
    test_pil_256.save(REPO_ROOT / "data" / "probe_test_image.png")
    x0_test = _pil_to_tensor(test_img).to(device)
    x_test_500 = noise_to_t(x0_test, scheduler, target_t=TARGET_T, seed=0)
    with MidBlockCapture(unet) as cap_test, torch.no_grad():
        _ = unet(x_test_500, torch.tensor([TARGET_T], device=device)).sample
    h_test = cap_test.feature
    print(f"||h_test|| = {h_test.norm().item():.3f}  "
          f"||Δh|| / ||h_test|| = {delta_norm / h_test.norm().item():.3f}")

    print(f"decoding sweep over s ∈ {S_VALUES} ...")
    delta_d = delta_h.to(device)
    decoded = []
    for s in S_VALUES:
        h_inj = h_test + s * delta_d
        x0 = decode_from_h(unet, alphas_cumprod, x_test_500, h_inj,
                           target_t=TARGET_T, step_size=50)
        decoded.append((s, to_uint8_image(x0)))
        print(f"  s={s:+.1f} ok")

    # Figure: source test image + 7 sweep images, large per-thumbnail.
    test_img_arr = np.asarray(test_pil_256)
    n_cols = len(S_VALUES) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(3.2 * n_cols, 3.6))
    axes[0].imshow(test_img_arr)
    axes[0].set_title("source (Smiling=0)", fontsize=11)
    axes[0].axis("off")
    for i, (s, img) in enumerate(decoded):
        ax = axes[i + 1]
        ax.imshow(img)
        ax.set_title(f"s = {s:+.1f}", fontsize=12,
                     color=("crimson" if s > 0 else "darkblue" if s < 0 else "black"))
        ax.axis("off")
    fig.suptitle(
        f"Smile Δh injection sweep at t={TARGET_T}    "
        f"||Δh|| = {delta_norm:.1f},  ||Δh||/||h_test|| = {delta_norm / h_test.norm().item():.2f}",
        fontsize=12,
    )
    fig.tight_layout()
    # Also dump individual frames for forensic inspection.
    indiv_dir = REPO_ROOT / "figures" / "exp6_smile_probe_frames"
    indiv_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(test_img_arr).save(indiv_dir / "source.png")
    for s, img in decoded:
        Image.fromarray(img).save(indiv_dir / f"s_{s:+.1f}.png")
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150)
    print(f"saved {FIG_OUT}")

    import os
    os._exit(0)  # avoid post-main fsspec SIGABRT (script 03 pattern)


if __name__ == "__main__":
    main()
