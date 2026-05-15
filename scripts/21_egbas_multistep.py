"""Phase 17b: E-GBAS analog with multi-step injection (visible variation).

scripts/20_egbas_analog.py used single-step injection to be faithful to
the original E-GBAS GAN setup, but skip dominance squashes the visible
difference. Here we keep the same sampling (ε-ball vs GB-RRT) but
**inject every DDIM step** with Δh = (h_sample − h_query) as a constant
offset. Phase 11 showed this breaks skip lock.

To make the comparison clean, GB-RRT walk parameters are tuned so the
final ‖Δh‖ matches the ε-ball case. We then ask: at matched perturbation
magnitude, does GB-RRT (in-polytope) yield more attribute-coherent
images than ε-ball (cross-polytope)?

Output:
  figures/exp16_egbas_multistep.png — 2 × 6 sweep
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from diffusers import DDIMScheduler, DDPMPipeline  # noqa: E402
from PIL import Image  # noqa: E402

from diffusion_boundary.decoding import to_uint8_image  # noqa: E402

CHANNELS = 512
SPATIAL = 8
N_SAMPLES = 5
NUM_DDIM_STEPS = 50

# magnitude-matched perturbations
TARGET_DELTA_NORM = 17.0          # ‖Δh‖ target for fair comparison
EPS_BALL_STD = 0.08               # produces ‖Δ‖ ≈ 17 for our h scale
GBRRT_STEP = 2.0                  # larger walk step
GBRRT_N_STEPS = 80                # ⇒ ‖Δ‖ ≈ 2·√80 ≈ 17.9 (matched)
GBRRT_MAX_TRIES_PER_STEP = 80

# scale Δh's per-step injection so the multi-step effect is comparable to
# Phase 11 (where natural ‖Δh_t‖ ranged from 12 to 84).
INJECT_SCALE = 1.0  # at every step, h ← h + INJECT_SCALE · Δh

ANCHOR_DIR = REPO_ROOT / "data" / "anchors"
MODEL_ID = "google/ddpm-celebahq-256"
FIG_OUT = REPO_ROOT / "figures" / "exp16_egbas_multistep.png"


def channel_signs(h: torch.Tensor) -> torch.Tensor:
    return torch.sign(h.reshape(CHANNELS, -1).mean(-1)).to(torch.int8)


def eps_ball(query: torch.Tensor, n: int, std: float, seed: int) -> list[torch.Tensor]:
    g = torch.Generator(device=query.device).manual_seed(seed)
    out = []
    for _ in range(n):
        noise = torch.randn(query.shape, generator=g, device=query.device) * std * query.std()
        out.append(query + noise)
    return out


def gbrrt(query: torch.Tensor, n: int, step_norm: float, n_walk: int,
          max_tries: int, seed: int):
    g = torch.Generator(device=query.device).manual_seed(seed)
    target = channel_signs(query)
    accepted = rejected = 0
    current = query.clone()
    out = []
    for _ in range(n):
        for _ in range(n_walk):
            for _ in range(max_tries):
                d = torch.randn(current.shape, generator=g, device=current.device)
                d = d / d.norm() * step_norm
                proposal = current + d
                if torch.equal(channel_signs(proposal), target):
                    current = proposal
                    accepted += 1
                    break
                rejected += 1
        out.append(current.clone())
    return out, {"accepted": accepted, "rejected": rejected,
                 "acc_rate": accepted / max(accepted + rejected, 1)}


def decode_with_constant_delta(
    unet, scheduler, x_T: torch.Tensor, delta_h: torch.Tensor,
    *, num_steps: int, scale: float = 1.0,
) -> torch.Tensor:
    """DDIM denoise from x_T while adding `scale·delta_h` to mid_block output
    at every step (constant offset across timesteps)."""
    scheduler.set_timesteps(num_steps)
    x = x_T.clone()
    device = x.device
    inject_tensor = (scale * delta_h).to(device)

    def hook(_m, _i, output):
        if isinstance(output, tuple):
            return (output[0] + inject_tensor,) + output[1:]
        return output + inject_tensor

    for t in scheduler.timesteps:
        handle = unet.mid_block.register_forward_hook(hook)
        try:
            with torch.no_grad():
                eps = unet(x, t.to(device)).sample
        finally:
            handle.remove()
        x = scheduler.step(eps, t, x).prev_sample
    return x


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = DDPMPipeline.from_pretrained(MODEL_ID).to(device)
    unet = pipeline.unet
    scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    h_query = torch.load(ANCHOR_DIR / "h500_0.pt", weights_only=True).to(device)
    q_norm = h_query.norm().item()
    print(f"||h_query|| = {q_norm:.2f}")

    eps_samples = eps_ball(h_query, N_SAMPLES, EPS_BALL_STD, seed=0)
    target_signs = channel_signs(h_query)
    print("\nε-ball deltas:")
    for i, s in enumerate(eps_samples):
        flips = (channel_signs(s) != target_signs).sum().item()
        print(f"  [{i}] ||Δ|| = {(s - h_query).norm().item():.2f}  flips = {flips}")

    gb_samples, gb_stats = gbrrt(h_query, N_SAMPLES, GBRRT_STEP,
                                 GBRRT_N_STEPS, GBRRT_MAX_TRIES_PER_STEP, seed=0)
    print("\nGB-RRT deltas:")
    for i, s in enumerate(gb_samples):
        flips = (channel_signs(s) != target_signs).sum().item()
        print(f"  [{i}] ||Δ|| = {(s - h_query).norm().item():.2f}  flips = {flips} (should be 0)")
    print(f"GB-RRT acceptance: {gb_stats['acc_rate']*100:.1f}%")

    # x_T: fresh noise (seed=0). Multi-step inject with Δh.
    torch.manual_seed(0)
    x_T = torch.randn(1, 3, 256, 256, device=device)

    # Baseline: query inject = 0 perturbation = natural generation from x_T
    print("\ndecoding baseline (Δh = 0) ...")
    baseline_x0 = decode_with_constant_delta(
        unet, scheduler, x_T, torch.zeros_like(h_query),
        num_steps=NUM_DDIM_STEPS, scale=0.0,
    )
    baseline_img = to_uint8_image(baseline_x0)

    def decode_set(samples):
        imgs = []
        for s in samples:
            delta = s - h_query
            x0 = decode_with_constant_delta(unet, scheduler, x_T, delta,
                                             num_steps=NUM_DDIM_STEPS,
                                             scale=INJECT_SCALE)
            imgs.append(to_uint8_image(x0))
        return imgs

    print("decoding ε-ball multi-step ...")
    eps_imgs = decode_set(eps_samples)
    print("decoding GB-RRT multi-step ...")
    gb_imgs = decode_set(gb_samples)

    # Figure
    n_cols = N_SAMPLES + 1
    fig, axes = plt.subplots(2, n_cols, figsize=(2.6 * n_cols, 5.6))
    rows = [("ε-ball (ignore signs)", eps_imgs, eps_samples),
            ("GB-RRT (in-polytope)", gb_imgs, gb_samples)]
    for r, (label, imgs, samps) in enumerate(rows):
        axes[r, 0].imshow(baseline_img)
        axes[r, 0].set_title("baseline (Δh=0)", fontsize=10)
        axes[r, 0].axis("off")
        for i, (img, s) in enumerate(zip(imgs, samps, strict=True)):
            ax = axes[r, i + 1]
            ax.imshow(img)
            flips = (channel_signs(s) != target_signs).sum().item()
            ax.set_title(f"||Δ||={(s - h_query).norm().item():.0f}\n"
                         f"{flips} flips", fontsize=8)
            ax.axis("off")
        fig.text(0.005, 1 - (r + 0.55) / 2, label, rotation=90,
                 fontsize=11, va="center")
    fig.suptitle(
        f"Phase 17b — ε-ball vs GB-RRT, multi-step inject  "
        f"(scale={INJECT_SCALE}, ||Δh||≈17 matched)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.025, 0, 1, 0.96))
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150)
    print(f"\nsaved {FIG_OUT}")

    fdir = REPO_ROOT / "figures" / "exp16_egbas_multistep_frames"
    fdir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(baseline_img).save(fdir / "baseline.png")
    for i, img in enumerate(eps_imgs):
        Image.fromarray(img).save(fdir / f"eps_{i}.png")
    for i, img in enumerate(gb_imgs):
        Image.fromarray(img).save(fdir / f"gbrrt_{i}.png")
    print(f"saved frames in {fdir}/")

    import os
    os._exit(0)


if __name__ == "__main__":
    main()
