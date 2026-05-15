"""Phase 17c: GB-RRT on a selective polytope (smile-pure channels only).

Phase 17b: 512-channel polytope didn't visibly constrain image semantics.
Hypothesis here: maybe enforcing the *full* sign pattern is too coarse
— the polytope is so large it doesn't isolate attributes. Restricting
the constraint to the 110 smile-pure channels (Phase 15) should keep
*smile* constant while letting other attributes drift.

Procedure:
  1. Identify smile-pure channels (purity > 0.85 at t=450 categorization).
  2. GB-RRT random walk; accept iff smile-pure-channel signs match
     query. Other 402 channels are free to flip.
  3. Also: variant GB-RRT_gender (gender-pure 173 channels only).
  4. Same multi-step inject decoding as Phase 17b.

Compare three rows: eps-ball (no constraint), GB-RRT_smile, GB-RRT_gender.
Output: figures/exp17_selective_polytope.png
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

CHANNELS = 512
SPATIAL = 8
N_SAMPLES = 5
NUM_DDIM_STEPS = 50
T_CATEGORIZE = 450

EPS_BALL_STD = 0.08
GBRRT_STEP = 2.0
GBRRT_N_STEPS = 80
GBRRT_MAX_TRIES = 80
INJECT_SCALE = 1.0

ANCHOR_DIR = REPO_ROOT / "data" / "anchors"
MULTIATTR = REPO_ROOT / "data" / "delta_h_multiattr.pt"
ORTH = REPO_ROOT / "data" / "delta_h_smile_orth.pt"
MODEL_ID = "google/ddpm-celebahq-256"
FIG_OUT = REPO_ROOT / "figures" / "exp17_selective_polytope.png"


def channel_signs(h: torch.Tensor) -> torch.Tensor:
    return torch.sign(h.reshape(CHANNELS, -1).mean(-1)).to(torch.int8)


def compute_category_masks() -> dict[str, np.ndarray]:
    """Reproduce Phase 15 categorization: smile / gender / joint / weak."""
    attr = torch.load(MULTIATTR, weights_only=False)
    smile_orth = torch.load(ORTH, weights_only=False)
    A = smile_orth[T_CATEGORIZE].reshape(CHANNELS, -1).mean(-1).numpy()
    B = attr["gender"][T_CATEGORIZE].reshape(CHANNELS, -1).mean(-1).numpy()
    norm = np.hypot(A, B)
    smile_pur = np.abs(A) / np.maximum(norm, 1e-12)
    gender_pur = np.abs(B) / np.maximum(norm, 1e-12)
    cat = np.full(CHANNELS, "joint", dtype=object)
    cat[smile_pur > 0.85] = "smile"
    cat[gender_pur > 0.85] = "gender"
    cat[norm < np.quantile(norm, 0.25)] = "weak"
    return {
        "smile": cat == "smile",
        "gender": cat == "gender",
    }


def eps_ball(query, n, std, seed):
    g = torch.Generator(device=query.device).manual_seed(seed)
    out = []
    for _ in range(n):
        noise = torch.randn(query.shape, generator=g, device=query.device) * std * query.std()
        out.append(query + noise)
    return out


def gbrrt_masked(query, n, step, n_walk, max_tries, mask, seed):
    """GB-RRT random walk where only `mask` channels must keep their sign."""
    g = torch.Generator(device=query.device).manual_seed(seed)
    target = channel_signs(query)
    mask_t = torch.from_numpy(mask).to(query.device)
    accepted = rejected = 0
    current = query.clone()
    out = []
    for _ in range(n):
        for _ in range(n_walk):
            for _ in range(max_tries):
                d = torch.randn(current.shape, generator=g, device=current.device)
                d = d / d.norm() * step
                proposal = current + d
                p_sig = channel_signs(proposal)
                if torch.equal(p_sig[mask_t], target[mask_t]):
                    current = proposal
                    accepted += 1
                    break
                rejected += 1
        out.append(current.clone())
    return out, {"accepted": accepted, "rejected": rejected,
                 "acc_rate": accepted / max(accepted + rejected, 1)}


def decode_with_const_delta(unet, scheduler, x_T, delta, num_steps, scale):
    scheduler.set_timesteps(num_steps)
    x = x_T.clone()
    inj = (scale * delta).to(x.device)

    def hook(_m, _i, output):
        if isinstance(output, tuple):
            return (output[0] + inj,) + output[1:]
        return output + inj

    for t in scheduler.timesteps:
        h = unet.mid_block.register_forward_hook(hook)
        try:
            with torch.no_grad():
                eps = unet(x, t.to(x.device)).sample
        finally:
            h.remove()
        x = scheduler.step(eps, t, x).prev_sample
    return x


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = DDPMPipeline.from_pretrained(MODEL_ID).to(device)
    unet = pipeline.unet
    scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    masks = compute_category_masks()
    print(f"smile-pure: {masks['smile'].sum()}, gender-pure: {masks['gender'].sum()}")

    h_query = torch.load(ANCHOR_DIR / "h500_0.pt", weights_only=True).to(device)
    target_signs = channel_signs(h_query)

    eps_samples = eps_ball(h_query, N_SAMPLES, EPS_BALL_STD, seed=0)
    print("\nε-ball samples:")
    for i, s in enumerate(eps_samples):
        flips = (channel_signs(s) != target_signs).sum().item()
        print(f"  [{i}] ||Δ||={(s - h_query).norm():.1f}  total flips={flips}")

    smile_samples, smile_stats = gbrrt_masked(
        h_query, N_SAMPLES, GBRRT_STEP, GBRRT_N_STEPS, GBRRT_MAX_TRIES,
        masks["smile"], seed=0,
    )
    print(f"\nGB-RRT_smile (smile-pure {masks['smile'].sum()} ch constrained): "
          f"accept={smile_stats['acc_rate']*100:.1f}%")
    for i, s in enumerate(smile_samples):
        sig = channel_signs(s)
        smile_flips = (sig[torch.from_numpy(masks['smile'])] !=
                       target_signs[torch.from_numpy(masks['smile'])]).sum().item()
        other_flips = (sig[torch.from_numpy(~masks['smile'])] !=
                       target_signs[torch.from_numpy(~masks['smile'])]).sum().item()
        print(f"  [{i}] ||Δ||={(s - h_query).norm():.1f}  "
              f"smile_flips={smile_flips}/110 (should be 0)  other_flips={other_flips}/402")

    gender_samples, gender_stats = gbrrt_masked(
        h_query, N_SAMPLES, GBRRT_STEP, GBRRT_N_STEPS, GBRRT_MAX_TRIES,
        masks["gender"], seed=0,
    )
    print(f"\nGB-RRT_gender (gender-pure {masks['gender'].sum()} ch constrained): "
          f"accept={gender_stats['acc_rate']*100:.1f}%")
    for i, s in enumerate(gender_samples):
        sig = channel_signs(s)
        gender_flips = (sig[torch.from_numpy(masks['gender'])] !=
                        target_signs[torch.from_numpy(masks['gender'])]).sum().item()
        other_flips = (sig[torch.from_numpy(~masks['gender'])] !=
                       target_signs[torch.from_numpy(~masks['gender'])]).sum().item()
        print(f"  [{i}] ||Δ||={(s - h_query).norm():.1f}  "
              f"gender_flips={gender_flips}/173 (should be 0)  other_flips={other_flips}/339")

    torch.manual_seed(0)
    x_T = torch.randn(1, 3, 256, 256, device=device)
    print("\ndecoding baseline ...")
    baseline = to_uint8_image(decode_with_const_delta(
        unet, scheduler, x_T, torch.zeros_like(h_query), NUM_DDIM_STEPS, 0.0))

    def decode_all(samples):
        return [to_uint8_image(decode_with_const_delta(
            unet, scheduler, x_T, s - h_query, NUM_DDIM_STEPS, INJECT_SCALE))
                for s in samples]

    print("decoding ε-ball ...")
    eps_imgs = decode_all(eps_samples)
    print("decoding GB-RRT_smile ...")
    smile_imgs = decode_all(smile_samples)
    print("decoding GB-RRT_gender ...")
    gender_imgs = decode_all(gender_samples)

    # Figure: 3 rows × 6 cols
    n_cols = N_SAMPLES + 1
    fig, axes = plt.subplots(3, n_cols, figsize=(2.6 * n_cols, 8.0))
    rows = [
        ("ε-ball (no constraint)", eps_imgs, eps_samples),
        ("GB-RRT_smile (110 ch)", smile_imgs, smile_samples),
        ("GB-RRT_gender (173 ch)", gender_imgs, gender_samples),
    ]
    for r, (label, imgs, samps) in enumerate(rows):
        axes[r, 0].imshow(baseline)
        axes[r, 0].set_title("baseline", fontsize=10)
        axes[r, 0].axis("off")
        for i, (img, s) in enumerate(zip(imgs, samps, strict=True)):
            ax = axes[r, i + 1]
            ax.imshow(img)
            ax.set_title(f"||Δ||={(s - h_query).norm():.0f}", fontsize=9)
            ax.axis("off")
        fig.text(0.005, 1 - (r + 0.55) / 3, label, rotation=90,
                 fontsize=11, va="center")
    fig.suptitle("Phase 17c — Selective polytope GB-RRT (multi-step inject)",
                 fontsize=11)
    fig.tight_layout(rect=(0.03, 0, 1, 0.96))
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150)
    print(f"\nsaved {FIG_OUT}")

    import os
    os._exit(0)


if __name__ == "__main__":
    main()
