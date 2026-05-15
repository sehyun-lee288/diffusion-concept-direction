"""Phase 17: E-GBAS analog — ε-ball vs GB-RRT sampling in h-space.

E-GBAS (Jeon et al. 2019) compared:
  (a) ε-ball sampling around a query latent z₀
  (b) GB-RRT: random walk that *rejects* moves crossing any active
      neuron boundary, so all samples stay in the same activation
      polytope as z₀

For DCGAN-MNIST (2D latent) they could draw the polytope; for PGGAN-
CelebA (512D) they sampled in the full latent and only showed images.

Our diffusion analog: query = anchor 0's h_500 (shape (1, 512, 8, 8)).
"Polytope" = the orthant defined by sign(spatial_mean(h_c)) per channel
(512 bits). We work in h-space at t=500 directly.

Decoding (faithful to E-GBAS single-pass GAN setting):
  Use anchor 0's x_500 as the encoder input. At the t=500 DDIM step,
  REPLACE mid_block output with the sampled h. Continue 500 → 0
  normally. This is single-step injection per Phase 10 — we expect
  the change to be subtle, but the *pattern* of variation (random
  drift for ε-ball vs polytope-bounded for GB-RRT) is what we test.

Output:
  figures/exp15_egbas_analog.png — 2 rows × 6 cols: query + 5 samples each
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

from diffusion_boundary.decoding import decode_from_h, to_uint8_image  # noqa: E402

CHANNELS = 512
SPATIAL = 8
TARGET_T = 500
STEP_SIZE = 50
N_SAMPLES = 5

# perturbation scales — tuned so both methods produce comparable ||Δh||
EPS_BALL_STD = 0.08          # ε-ball: h + EPS_BALL_STD · randn  (norm ≈ EPS · √D)
GBRRT_STEP = 0.5             # GB-RRT step magnitude (in raw h units)
GBRRT_N_STEPS = 50           # walk length
GBRRT_MAX_TRIES_PER_STEP = 60

ANCHOR_DIR = REPO_ROOT / "data" / "anchors"
MODEL_ID = "google/ddpm-celebahq-256"
FIG_OUT = REPO_ROOT / "figures" / "exp15_egbas_analog.png"


def channel_signs(h: torch.Tensor) -> torch.Tensor:
    """(1, C, H, W) → (C,) ±1 sign of spatial-mean per channel."""
    return torch.sign(h.reshape(CHANNELS, -1).mean(-1)).to(torch.int8)


def eps_ball_samples(query: torch.Tensor, n: int, std: float, seed: int) -> list[torch.Tensor]:
    g = torch.Generator(device=query.device).manual_seed(seed)
    out = []
    for _ in range(n):
        noise = torch.randn(query.shape, generator=g, device=query.device) * std * query.std()
        out.append(query + noise)
    return out


def gbrrt_samples(
    query: torch.Tensor, n: int, step_norm: float,
    n_walk_steps: int, max_tries: int, seed: int,
) -> tuple[list[torch.Tensor], dict]:
    """Random walk that *rejects* sign-flipping proposals. Returns N samples
    spaced along a single chain. Statistics about acceptance are also returned."""
    g = torch.Generator(device=query.device).manual_seed(seed)
    target_signs = channel_signs(query)
    chain_len = n_walk_steps * n  # space N samples across chain
    accepted_total = 0
    rejected_total = 0
    current = query.clone()
    samples: list[torch.Tensor] = []
    for _ in range(n):
        for _ in range(n_walk_steps):
            for _ in range(max_tries):
                direction = torch.randn(current.shape, generator=g, device=current.device)
                direction = direction / direction.norm() * step_norm
                proposal = current + direction
                if torch.equal(channel_signs(proposal), target_signs):
                    current = proposal
                    accepted_total += 1
                    break
                rejected_total += 1
            else:
                # exhausted max_tries; stay
                pass
        samples.append(current.clone())
    stats = {
        "accepted": accepted_total,
        "rejected": rejected_total,
        "chain_len_target": chain_len,
        "acceptance_rate": accepted_total / max(accepted_total + rejected_total, 1),
    }
    return samples, stats


def decode_sample(unet, alphas_cumprod, x_encoder: torch.Tensor, h_inject: torch.Tensor) -> np.ndarray:
    """Single-step inject at t=500 (E-GBAS faithful)."""
    x0 = decode_from_h(unet, alphas_cumprod, x_encoder, h_inject,
                       target_t=TARGET_T, step_size=STEP_SIZE)
    return to_uint8_image(x0)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = DDPMPipeline.from_pretrained(MODEL_ID).to(device)
    unet = pipeline.unet
    scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    h_query = torch.load(ANCHOR_DIR / "h500_0.pt", weights_only=True).to(device)
    x_encoder = torch.load(ANCHOR_DIR / "x500_0.pt", weights_only=True).to(device)
    q_norm = h_query.norm().item()
    print(f"||h_query|| = {q_norm:.2f}")
    print(f"query channel-mean std = {h_query.std().item():.4f}")

    # ε-ball samples
    eps_samples = eps_ball_samples(h_query, N_SAMPLES, EPS_BALL_STD, seed=0)
    eps_signs = [channel_signs(s) for s in eps_samples]
    target_signs = channel_signs(h_query)
    print("\nε-ball samples:")
    for i, (s, sig) in enumerate(zip(eps_samples, eps_signs, strict=True)):
        flipped = (sig != target_signs).sum().item()
        dist = (s - h_query).norm().item()
        print(f"  [{i}] ||Δ|| = {dist:.2f} ({dist/q_norm*100:.1f}% of ||q||), "
              f"{flipped}/{CHANNELS} signs flipped")

    # GB-RRT samples
    gb_samples, stats = gbrrt_samples(h_query, N_SAMPLES, GBRRT_STEP,
                                      GBRRT_N_STEPS, GBRRT_MAX_TRIES_PER_STEP,
                                      seed=0)
    print("\nGB-RRT samples:")
    for i, s in enumerate(gb_samples):
        sig = channel_signs(s)
        flipped = (sig != target_signs).sum().item()
        dist = (s - h_query).norm().item()
        print(f"  [{i}] ||Δ|| = {dist:.2f} ({dist/q_norm*100:.1f}% of ||q||), "
              f"{flipped}/{CHANNELS} signs flipped (should be 0)")
    print(f"GB-RRT acceptance: {stats['accepted']}/{stats['accepted']+stats['rejected']} "
          f"= {stats['acceptance_rate']*100:.1f}%")

    # Decode query baseline + all samples.
    print("\ndecoding query baseline ...")
    query_img = decode_sample(unet, alphas_cumprod, x_encoder, h_query)
    print("decoding ε-ball samples ...")
    eps_imgs = [decode_sample(unet, alphas_cumprod, x_encoder, s) for s in eps_samples]
    print("decoding GB-RRT samples ...")
    gb_imgs = [decode_sample(unet, alphas_cumprod, x_encoder, s) for s in gb_samples]

    # Figure: 2 rows × 6 cols (query + 5 samples per row)
    rows = [("ε-ball (ignore signs)", eps_imgs, eps_samples),
            ("GB-RRT (in-polytope)", gb_imgs, gb_samples)]
    n_cols = N_SAMPLES + 1
    fig, axes = plt.subplots(2, n_cols, figsize=(2.6 * n_cols, 5.6))
    for r, (label, imgs, samps) in enumerate(rows):
        axes[r, 0].imshow(query_img)
        axes[r, 0].set_title("query (anchor 0)", fontsize=10)
        axes[r, 0].axis("off")
        for i, (img, s) in enumerate(zip(imgs, samps, strict=True)):
            ax = axes[r, i + 1]
            ax.imshow(img)
            sig = channel_signs(s)
            flips = (sig != target_signs).sum().item()
            ax.set_title(f"||Δ||={(s - h_query).norm().item():.0f}\n"
                         f"{flips} flips",
                         fontsize=8)
            ax.axis("off")
        fig.text(0.005, 1 - (r + 0.55) / 2, label, rotation=90,
                 fontsize=11, va="center")
    fig.suptitle("Phase 17 — ε-ball vs GB-RRT sampling around query h₀ at t=500 "
                 "(single-step inject)", fontsize=11)
    fig.tight_layout(rect=(0.025, 0, 1, 0.96))
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150)
    print(f"\nsaved {FIG_OUT}")

    # Also dump frames for inspection.
    fdir = REPO_ROOT / "figures" / "exp15_egbas_analog_frames"
    fdir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(query_img).save(fdir / "query.png")
    for i, img in enumerate(eps_imgs):
        Image.fromarray(img).save(fdir / f"eps_{i}.png")
    for i, img in enumerate(gb_imgs):
        Image.fromarray(img).save(fdir / f"gbrrt_{i}.png")
    print(f"saved frames in {fdir}/")

    import os
    os._exit(0)


if __name__ == "__main__":
    main()
