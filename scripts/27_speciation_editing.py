"""Phase 21: speciation-window editing vs all-step editing.

Phase 18 found the smile attribute commits around t≈700. Until now we
injected Δh at *every* DDIM step (Phase 11). Hypothesis: concentrating
the injection in the speciation window should be more efficient — more
smile change per unit of perturbation, possibly less identity drift.

Three timestep windows for the Δh injection:
  A) all          — every step (Phase 11 baseline)
  B) speciation   — t ∈ [600, 800]  (the commit zone)
  C) early/off    — t ∈ [350, 550]  (our old t=500 assumption)

For each window, sweep s and decode with the per-timestep Δh_t (from
Phase 11, data/delta_h_smile_multistep.pt). Same fixed x_T.

Output: figures/exp22_speciation_editing.png — 3-row sweep
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

from diffusion_boundary.decoding import to_uint8_image  # noqa: E402

MODEL_ID = "google/ddpm-celebahq-256"
NUM_DDIM_STEPS = 50
S_VALUES = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
WINDOWS = {
    "A) all steps": (0, 1000),
    "B) speciation [600,800]": (600, 800),
    "C) early [350,550]": (350, 550),
}

DELTA_PATH = REPO_ROOT / "data" / "delta_h_smile_multistep.pt"
FIG_OUT = REPO_ROOT / "figures" / "exp22_speciation_editing.png"


def decode_windowed(unet, scheduler, x_T, delta_dict, s, window, num_steps, device):
    """Multi-step inject `s·Δh_{t_nearest}` but only when t is inside `window`."""
    scheduler.set_timesteps(num_steps)
    x = x_T.clone()
    keys = np.array(sorted(delta_dict.keys()))
    staged = {k: delta_dict[k].to(device) for k in delta_dict}
    lo, hi = window
    n_injected = 0
    for t in scheduler.timesteps:
        t_int = int(t.item())
        if lo <= t_int <= hi:
            nearest = int(keys[np.argmin(np.abs(keys - t_int))])
            inj = s * staged[nearest]
            n_injected += 1

            def hook(_m, _i, output, _d=inj):
                if isinstance(output, tuple):
                    return (output[0] + _d,) + output[1:]
                return output + _d

            handle = unet.mid_block.register_forward_hook(hook)
        else:
            handle = None
        try:
            with torch.no_grad():
                eps = unet(x, t.to(device)).sample
        finally:
            if handle is not None:
                handle.remove()
        x = scheduler.step(eps, t, x).prev_sample
    return x, n_injected


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = DDPMPipeline.from_pretrained(MODEL_ID).to(device)
    unet = pipeline.unet
    scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    delta_dict = torch.load(DELTA_PATH, weights_only=False)

    torch.manual_seed(0)
    x_T = torch.randn(1, 3, 256, 256, device=device)

    rows = []
    for label, window in WINDOWS.items():
        print(f"window {label}  t∈{window}")
        imgs = []
        n_inj = None
        for s in S_VALUES:
            x0, n_inj = decode_windowed(unet, scheduler, x_T, delta_dict, s,
                                        window, NUM_DDIM_STEPS, device)
            imgs.append((s, to_uint8_image(x0)))
            print(f"  s={s:+.1f}  ({n_inj} steps injected)")
        rows.append((label, imgs, n_inj))

    n_cols = len(S_VALUES)
    fig, axes = plt.subplots(len(WINDOWS), n_cols,
                             figsize=(2.6 * n_cols, 2.9 * len(WINDOWS)))
    for r, (label, imgs, n_inj) in enumerate(rows):
        for c, (s, img) in enumerate(imgs):
            ax = axes[r, c]
            ax.imshow(img)
            ax.axis("off")
            if r == 0:
                ax.set_title(f"s={s:+.1f}", fontsize=11,
                             color=("crimson" if s > 0 else "darkblue" if s < 0 else "black"))
        fig.text(0.005, 1 - (r + 0.5) / len(WINDOWS),
                 f"{label}\n({n_inj}/{NUM_DDIM_STEPS} steps)",
                 rotation=90, fontsize=9, va="center", ha="center")
    fig.suptitle("Phase 21 — speciation-window editing vs all-step  (same x_T, per-step Δh_t)",
                 fontsize=12)
    fig.tight_layout(rect=(0.03, 0, 1, 0.96))
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150)
    print(f"\nsaved {FIG_OUT}")

    import os
    os._exit(0)


if __name__ == "__main__":
    main()
