"""Phase 11: multi-step (Asyrp-style) h trajectory editing for smile.

Hypothesis. Phase 10 showed single-step h-injection at t=500 is killed
by encoder-skip dominance. But if we inject at *every* DDIM denoising
step, the modified mid_block output bends epsilon_theta, which bends
x_{t-1}, so the next step's encoder skip is already modified. The
correction accumulates along the trajectory, so the skip cannot pin the
final output.

Procedure:
  1. Stream `eurecom-ds/celeba-hq-256`, collect N=20 smile + 20 no-smile.
  2. For each t in T_DELTAS = {50, 150, …, 950}:
       capture h(x_t, t) per image via q-sample(seed=image_idx);
       Δh_t = mean(h | smile=1) − mean(h | smile=0).
  3. Fix x_T = randn(seed=0).
  4. DDIM 20-step denoise from x_T to x_0; at every step register a
     mid_block forward hook that *adds* s·Δh_(t_nearest).
  5. Sweep s ∈ {−3, −2, −1, 0, +1, +2, +3}.

Outputs:
  data/delta_h_smile_multistep.pt  — dict {t_int: tensor (1,512,8,8)}
  figures/exp7_multistep_smile.png  — 7-image sweep
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

from diffusion_boundary.decoding import to_uint8_image  # noqa: E402
from diffusion_boundary.hooks import MidBlockCapture  # noqa: E402
from diffusion_boundary.inversion import noise_to_t  # noqa: E402

MODEL_ID = "google/ddpm-celebahq-256"
DATASET_ID = "eurecom-ds/celeba-hq-256"
ATTR_IDX_SMILING = 31
N_PER_CLASS = 20
T_DELTAS = [50, 150, 250, 350, 450, 550, 650, 750, 850, 950]
NUM_DDIM_STEPS = 50
S_VALUES = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]

DELTA_OUT = REPO_ROOT / "data" / "delta_h_smile_multistep.pt"
FIG_OUT = REPO_ROOT / "figures" / "exp7_multistep_smile.png"
FRAMES_DIR = REPO_ROOT / "figures" / "exp7_multistep_smile_frames"


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB").resize((256, 256), Image.BICUBIC),
                     dtype=np.float32) / 255.0
    return torch.from_numpy(arr * 2.0 - 1.0).permute(2, 0, 1).unsqueeze(0)


def _collect_images(target_val: int, n_needed: int) -> list:
    ds = load_dataset(DATASET_ID, split="train", streaming=True)
    out = []
    try:
        for s in ds:
            if int(s["attributes"][ATTR_IDX_SMILING]) == target_val:
                out.append(s["image"].copy())
                if len(out) >= n_needed:
                    break
    finally:
        del ds
        gc.collect()
    return out


def _compute_delta_at_t(unet, scheduler, smile_imgs, nosmile_imgs, t: int, device: str) -> torch.Tensor:
    t_tensor = torch.tensor([t], device=device)

    def capture_all(imgs):
        hs = []
        for i, img in enumerate(imgs):
            x0 = _pil_to_tensor(img).to(device)
            x_t = noise_to_t(x0, scheduler, target_t=t, seed=i)
            with MidBlockCapture(unet) as cap, torch.no_grad():
                _ = unet(x_t, t_tensor).sample
            hs.append(cap.feature)
        return torch.cat(hs, dim=0)
    h_s = capture_all(smile_imgs).mean(0, keepdim=True)
    h_n = capture_all(nosmile_imgs).mean(0, keepdim=True)
    return (h_s - h_n).detach()


def _denoise_with_injection(
    unet, scheduler, x_T, delta_h_dict, s, num_steps, device,
):
    scheduler.set_timesteps(num_steps)
    x = x_T.clone()
    keys = np.array(sorted(delta_h_dict.keys()))
    deltas_by_key = {k: delta_h_dict[k].to(device) for k in delta_h_dict}

    for t in scheduler.timesteps:
        t_int = int(t.item())
        nearest = int(keys[np.argmin(np.abs(keys - t_int))])
        delta_t = deltas_by_key[nearest]

        def inject(_m, _i, output, _d=delta_t):
            if isinstance(output, tuple):
                return (output[0] + s * _d,) + output[1:]
            return output + s * _d

        handle = unet.mid_block.register_forward_hook(inject)
        try:
            with torch.no_grad():
                eps = unet(x, t.to(device)).sample
        finally:
            handle.remove()
        x = scheduler.step(eps, t, x).prev_sample
    return x


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {MODEL_ID} ...")
    pipeline = DDPMPipeline.from_pretrained(MODEL_ID).to(device)
    unet = pipeline.unet
    scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    print(f"Streaming {DATASET_ID} for {N_PER_CLASS} per class ...")
    smile_imgs = _collect_images(1, N_PER_CLASS)
    nosmile_imgs = _collect_images(0, N_PER_CLASS)
    print(f"  collected smile={len(smile_imgs)}, no-smile={len(nosmile_imgs)}")

    delta_dict: dict[int, torch.Tensor] = {}
    for t in T_DELTAS:
        print(f"Computing Δh at t={t} ...")
        delta_dict[t] = _compute_delta_at_t(unet, scheduler,
                                            smile_imgs, nosmile_imgs, t, device).cpu()
        print(f"  ||Δh_{t}|| = {delta_dict[t].norm().item():.2f}")
    torch.save(delta_dict, DELTA_OUT)
    print(f"saved {DELTA_OUT}")

    # Fixed-identity sampling: same x_T across the s sweep.
    torch.manual_seed(0)
    x_T = torch.randn(1, 3, 256, 256, device=device)

    print(f"Running multi-step injection sweep over s ∈ {S_VALUES} ...")
    decoded: list[tuple[float, np.ndarray]] = []
    for s in S_VALUES:
        x0 = _denoise_with_injection(unet, scheduler, x_T, delta_dict, s,
                                     NUM_DDIM_STEPS, device)
        decoded.append((s, to_uint8_image(x0)))
        print(f"  s={s:+.1f} ok")

    n_cols = len(S_VALUES)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.2 * n_cols, 3.6))
    for i, (s, img) in enumerate(decoded):
        ax = axes[i]
        ax.imshow(img)
        ax.set_title(f"s = {s:+.1f}", fontsize=12,
                     color=("crimson" if s > 0 else "darkblue" if s < 0 else "black"))
        ax.axis("off")
    norms_str = ", ".join(f"||Δh_{t}||={delta_dict[t].norm():.0f}" for t in T_DELTAS[::3])
    fig.suptitle(
        f"Multi-step smile-direction sweep — DDIM {NUM_DDIM_STEPS}-step, inject every step\n"
        f"({norms_str})",
        fontsize=11,
    )
    fig.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150)

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    for s, img in decoded:
        Image.fromarray(img).save(FRAMES_DIR / f"s_{s:+.1f}.png")
    print(f"saved {FIG_OUT} and frames in {FRAMES_DIR}/")

    import os
    os._exit(0)


if __name__ == "__main__":
    main()
