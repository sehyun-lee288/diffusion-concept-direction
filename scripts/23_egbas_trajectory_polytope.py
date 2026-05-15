"""Phase 17d: GB-RRT on a trajectory-level polytope.

Phase 17b constrained the sign pattern at a single timestep (t=500).
Here we constrain it at *multiple* timesteps simultaneously: a candidate
perturbation Δh is accepted only if (h_t_natural + Δh) keeps the same
512-channel sign pattern as h_t_natural at every checked timestep.

Rationale: the decoded image is produced by the whole DDIM trajectory,
not by t=500 alone. A perturbation that stays in-polytope across the
trajectory should — if E-GBAS's premise holds anywhere — be the one
that best preserves attributes.

Procedure:
  1. Run a clean DDIM trajectory from x_T (seed 0). Capture h_t at every
     step → the query trajectory {t: h_t}.
  2. Pick CHECK_TIMESTEPS subset for the constraint.
  3. GB-RRT random walk; accept Δ only if it keeps all 512 signs at ALL
     checked timesteps. (Much stricter than Phase 17b.)
  4. Multi-step inject decoding, compare with ε-ball and 17b single-t.

Output: figures/exp18_trajectory_polytope.png
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

from diffusion_boundary.decoding import to_uint8_image  # noqa: E402
from diffusion_boundary.hooks import MidBlockCapture  # noqa: E402

CHANNELS = 512
N_SAMPLES = 5
NUM_DDIM_STEPS = 50
# subset of DDIM timesteps to enforce the polytope at (indices into the
# 50-step schedule, spread across the trajectory)
CHECK_STEP_INDICES = [5, 15, 25, 35, 45]

EPS_BALL_STD = 0.08
GBRRT_STEP = 2.0
GBRRT_N_STEPS = 80
GBRRT_MAX_TRIES = 400  # stricter constraint → allow more tries
INJECT_SCALE = 1.0

ANCHOR_DIR = REPO_ROOT / "data" / "anchors"
MODEL_ID = "google/ddpm-celebahq-256"
FIG_OUT = REPO_ROOT / "figures" / "exp18_trajectory_polytope.png"


def channel_signs(h: torch.Tensor) -> torch.Tensor:
    return torch.sign(h.reshape(CHANNELS, -1).mean(-1)).to(torch.int8)


def capture_trajectory(unet, scheduler, x_T, num_steps, check_indices):
    """Run a clean DDIM trajectory; return {t_int: h_t} for checked steps."""
    scheduler.set_timesteps(num_steps)
    x = x_T.clone()
    device = x.device
    captured: dict[int, torch.Tensor] = {}
    for idx, t in enumerate(scheduler.timesteps):
        with MidBlockCapture(unet) as cap, torch.no_grad():
            eps = unet(x, t.to(device)).sample
        if idx in check_indices:
            captured[int(t.item())] = cap.feature.clone()
        x = scheduler.step(eps, t, x).prev_sample
    return captured


def eps_ball(query, n, std, seed):
    g = torch.Generator(device=query.device).manual_seed(seed)
    return [query + torch.randn(query.shape, generator=g, device=query.device)
            * std * query.std() for _ in range(n)]


def gbrrt_trajectory(query, traj_h, n, step, n_walk, max_tries, seed):
    """GB-RRT accepting Δ only if (h_t + Δ) keeps sign at ALL checked t."""
    g = torch.Generator(device=query.device).manual_seed(seed)
    target_per_t = {t: channel_signs(h) for t, h in traj_h.items()}
    accepted = rejected = 0
    current = query.clone()
    delta = torch.zeros_like(query)
    out = []
    for _ in range(n):
        for _ in range(n_walk):
            for _ in range(max_tries):
                d = torch.randn(current.shape, generator=g, device=current.device)
                d = d / d.norm() * step
                new_delta = delta + d
                ok = all(
                    torch.equal(channel_signs(h + new_delta), target_per_t[t])
                    for t, h in traj_h.items()
                )
                if ok:
                    delta = new_delta
                    current = query + delta
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

    h_query = torch.load(ANCHOR_DIR / "h500_0.pt", weights_only=True).to(device)
    target_signs = channel_signs(h_query)

    torch.manual_seed(0)
    x_T = torch.randn(1, 3, 256, 256, device=device)

    print("capturing clean DDIM trajectory ...")
    traj_h = capture_trajectory(unet, scheduler, x_T, NUM_DDIM_STEPS,
                                set(CHECK_STEP_INDICES))
    print(f"  checked timesteps: {sorted(traj_h.keys())}")
    for t, h in sorted(traj_h.items()):
        print(f"    t={t}: ||h||={h.norm():.1f}")

    eps_samples = eps_ball(h_query, N_SAMPLES, EPS_BALL_STD, seed=0)
    print("\nε-ball samples:")
    for i, s in enumerate(eps_samples):
        flips = (channel_signs(s) != target_signs).sum().item()
        print(f"  [{i}] ||Δ||={(s - h_query).norm():.1f}  t=500 flips={flips}")

    traj_samples, stats = gbrrt_trajectory(
        h_query, traj_h, N_SAMPLES, GBRRT_STEP, GBRRT_N_STEPS,
        GBRRT_MAX_TRIES, seed=0,
    )
    print(f"\nGB-RRT_trajectory (constrain {len(traj_h)} timesteps): "
          f"accept={stats['acc_rate']*100:.1f}%")
    for i, s in enumerate(traj_samples):
        delta = s - h_query
        worst = max(
            (channel_signs(h + delta) != channel_signs(h)).sum().item()
            for h in traj_h.values()
        )
        print(f"  [{i}] ||Δ||={delta.norm():.1f}  worst-t flips={worst} (should be 0)")

    print("\ndecoding baseline ...")
    baseline = to_uint8_image(decode_with_const_delta(
        unet, scheduler, x_T, torch.zeros_like(h_query), NUM_DDIM_STEPS, 0.0))

    def decode_all(samples):
        return [to_uint8_image(decode_with_const_delta(
            unet, scheduler, x_T, s - h_query, NUM_DDIM_STEPS, INJECT_SCALE))
                for s in samples]

    print("decoding ε-ball ...")
    eps_imgs = decode_all(eps_samples)
    print("decoding GB-RRT_trajectory ...")
    traj_imgs = decode_all(traj_samples)

    n_cols = N_SAMPLES + 1
    fig, axes = plt.subplots(2, n_cols, figsize=(2.6 * n_cols, 5.6))
    rows = [("ε-ball (no constraint)", eps_imgs, eps_samples),
            (f"GB-RRT_trajectory ({len(traj_h)} t)", traj_imgs, traj_samples)]
    for r, (label, imgs, samps) in enumerate(rows):
        axes[r, 0].imshow(baseline)
        axes[r, 0].set_title("baseline", fontsize=10)
        axes[r, 0].axis("off")
        for i, (img, s) in enumerate(zip(imgs, samps, strict=True)):
            ax = axes[r, i + 1]
            ax.imshow(img)
            ax.set_title(f"||Δ||={(s - h_query).norm():.0f}", fontsize=9)
            ax.axis("off")
        fig.text(0.005, 1 - (r + 0.55) / 2, label, rotation=90,
                 fontsize=11, va="center")
    fig.suptitle(
        f"Phase 17d — Trajectory polytope GB-RRT  "
        f"(sign held at t={sorted(traj_h.keys())})",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.03, 0, 1, 0.96))
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150)
    print(f"\nsaved {FIG_OUT}")

    import os
    os._exit(0)


if __name__ == "__main__":
    main()
