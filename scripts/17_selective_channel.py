"""Phase 16: selective-channel injection.

Phase 15 showed mid_block channels split cleanly by attribute loading.
Hypothesis: if smile editing leaks into gender (Phase 11), masking
*non-smile* channels out of the injection should give cleaner
attribute control.

Three variants (same fixed x_T, same s sweep):
  A) baseline      — all 512 channels (Phase 11 reproduction)
  B) smile-pure    — 110 channels with smile_purity > 0.85
  C) smile leak-trim — same as B minus the 5 leakiest channels by
                      |A_c · B_c|  (i.e., smile-pure but with the
                      largest residual gender component removed)

For each variant we mask `Δh_smile_orth` per timestep (zero out the
disallowed channels) and multi-step inject as in Phase 11.

Output:
  figures/exp12_selective_channel.png — 3-row, 7-column sweep comparison
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
from diffusion_boundary.multistep import denoise_with_injection  # noqa: E402

MODEL_ID = "google/ddpm-celebahq-256"
CHANNELS = 512
NUM_DDIM_STEPS = 50
S_VALUES = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
T_CATEGORIZE = 450
LEAKY_TOP_K = 5

MULTIATTR = REPO_ROOT / "data" / "delta_h_multiattr.pt"
ORTH = REPO_ROOT / "data" / "delta_h_smile_orth.pt"
FIG_OUT = REPO_ROOT / "figures" / "exp12_selective_channel.png"
FRAMES_DIR = REPO_ROOT / "figures" / "exp12_selective_channel_frames"


def _spatial_mean(t: torch.Tensor) -> np.ndarray:
    return t.reshape(CHANNELS, -1).mean(-1).numpy()


def _mask_delta_dict(delta_dict: dict[int, torch.Tensor], channel_mask: np.ndarray) -> dict[int, torch.Tensor]:
    """Multiply Δh at every timestep by a per-channel 0/1 mask."""
    m = torch.from_numpy(channel_mask.astype(np.float32))[None, :, None, None]
    return {t: delta_dict[t] * m for t in delta_dict}


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = DDPMPipeline.from_pretrained(MODEL_ID).to(device)
    unet = pipeline.unet
    scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    attr = torch.load(MULTIATTR, weights_only=False)
    smile_orth = torch.load(ORTH, weights_only=False)

    # Channel categorization at t=450 (same as Phase 15).
    A = _spatial_mean(smile_orth[T_CATEGORIZE].float())
    B = _spatial_mean(attr["gender"][T_CATEGORIZE].float())
    norm = np.hypot(A, B)
    smile_pur = np.abs(A) / np.maximum(norm, 1e-12)
    gender_pur = np.abs(B) / np.maximum(norm, 1e-12)

    cat = np.full(CHANNELS, "joint", dtype=object)
    cat[smile_pur > 0.85] = "smile"
    cat[gender_pur > 0.85] = "gender"
    cat[norm < np.quantile(norm, 0.25)] = "weak"

    smile_mask = (cat == "smile")
    leak_score = np.abs(A * B) * smile_mask
    leaky_idx = np.argsort(-leak_score)[:LEAKY_TOP_K]
    smile_trim_mask = smile_mask.copy()
    smile_trim_mask[leaky_idx] = False

    print(f"counts: smile={smile_mask.sum()}, gender={(cat=='gender').sum()}, "
          f"joint={(cat=='joint').sum()}, weak={(cat=='weak').sum()}")
    print(f"leakiest smile channels (excluded from variant C): {leaky_idx.tolist()}")
    print(f"  remaining smile-pure channels after trim: {smile_trim_mask.sum()}")

    all_mask = np.ones(CHANNELS, dtype=bool)
    variants = [
        ("A) all 512", _mask_delta_dict(smile_orth, all_mask)),
        ("B) smile-pure (110)", _mask_delta_dict(smile_orth, smile_mask)),
        (f"C) smile-pure − top-{LEAKY_TOP_K} leaky", _mask_delta_dict(smile_orth, smile_trim_mask)),
    ]

    torch.manual_seed(0)
    x_T = torch.randn(1, 3, 256, 256, device=device)

    rows = []
    for label, masked_delta in variants:
        print(f"running variant: {label}")
        row = []
        for s in S_VALUES:
            x0 = denoise_with_injection(unet, scheduler, x_T, [masked_delta], [s],
                                        NUM_DDIM_STEPS, device)
            row.append((s, to_uint8_image(x0)))
            print(f"  s={s:+.1f} ok")
        rows.append((label, row))

    # ---- Figure --------------------------------------------------------
    n_cols = len(S_VALUES)
    fig, axes = plt.subplots(len(variants), n_cols,
                             figsize=(2.6 * n_cols, 2.9 * len(variants)))
    for i, (label, row) in enumerate(rows):
        for j, (s, img) in enumerate(row):
            ax = axes[i, j]
            ax.imshow(img)
            ax.axis("off")
            if i == 0:
                ax.set_title(f"s = {s:+.1f}", fontsize=11,
                             color=("crimson" if s > 0 else "darkblue" if s < 0 else "black"))
        fig.text(0.005, 1 - (i + 0.55) / len(variants), label,
                 rotation=90, fontsize=11, va="center", ha="center")
    fig.suptitle("Phase 16 — selective-channel smile injection (multi-step, same x_T)",
                 fontsize=12)
    fig.tight_layout(rect=(0.025, 0, 1, 0.96))
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150)

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    for label, row in rows:
        tag = label.split(")")[0].lower()
        for s, img in row:
            Image.fromarray(img).save(FRAMES_DIR / f"v{tag}_s{s:+.1f}.png")
    print(f"saved {FIG_OUT}")

    import os
    os._exit(0)


if __name__ == "__main__":
    main()
