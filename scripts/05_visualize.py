"""Phase 5: assemble the boundary figure with decoded thumbnails.

Pipeline:
  1. Load data/sign_grid.npy + data/grid_meta.yaml.
  2. Compute region ids (connected components on sign vector) and the
     boundary mask (4-neighbor sign-pattern disagreement).
  3. Pick 9 sparse representative grid points (3×3 sub-grid).
  4. For each representative point, override the U-Net mid_block output
     with the corresponding h-vector at t≈500 and DDIM-denoise from there
     to t=0, yielding a synthetic image. Anchor 0's x_500 is used as the
     fixed encoder input across all decodings.
  5. Save figures/exp1_boundary.png.

Output: figures/exp1_boundary.png
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from diffusers import DDIMScheduler, DDPMPipeline  # noqa: E402

from diffusion_boundary.plane import build_plane  # noqa: E402
from diffusion_boundary.viz import (  # noqa: E402
    find_boundaries,
    plot_boundary_panel,
    region_ids,
    top_k_balanced_channels,
)

ANCHOR_DIR = REPO_ROOT / "data" / "anchors"
SIGN_GRID = REPO_ROOT / "data" / "sign_grid.npy"
GRID_META = REPO_ROOT / "data" / "grid_meta.yaml"
OUT_FIG = REPO_ROOT / "figures" / "exp1_boundary.png"

MODEL_ID = "google/ddpm-celebahq-256"
DECODE_STEP_SIZE = 50      # DDIM coarse stride: 500 -> 450 -> ... -> 0 (~10 steps)
CHANNELS = 512
SPATIAL = 8
TOP_K_CHANNELS = 20        # plot only the K most balanced channels' boundaries


def _load_h_flat(i: int) -> torch.Tensor:
    return torch.load(ANCHOR_DIR / f"h500_{i}.pt", weights_only=True).float().reshape(-1)


def _to_pil_array(x: torch.Tensor) -> np.ndarray:
    """(1, 3, H, W) tensor in [-1, 1] → (H, W, 3) uint8 array."""
    x = x.detach().cpu().clamp(-1, 1)
    x = (x + 1.0) / 2.0
    arr = (x[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return arr


def decode_from_h(
    unet,
    alphas_cumprod: torch.Tensor,
    x_500: torch.Tensor,
    h_target: torch.Tensor,
    *,
    target_t: int = 500,
    step_size: int = DECODE_STEP_SIZE,
) -> torch.Tensor:
    """Inject `h_target` at the mid_block during the t=target_t step and
    DDIM-denoise to t=0. Returns x_0 in [-1, 1] (clipped by caller)."""
    device = x_500.device

    def replace_with_h(_module, _inputs, output):
        if isinstance(output, tuple):
            return (h_target,) + output[1:]
        return h_target

    x = x_500.clone()
    t_curr = target_t
    is_first = True
    while t_curr >= 0:
        t_next = max(0, t_curr - step_size)
        handle = unet.mid_block.register_forward_hook(replace_with_h) if is_first else None
        try:
            with torch.no_grad():
                eps = unet(x, torch.tensor([t_curr], device=device)).sample
        finally:
            if handle is not None:
                handle.remove()
        is_first = False
        alpha_t = alphas_cumprod[t_curr]
        alpha_n = alphas_cumprod[t_next] if t_next > 0 else torch.tensor(1.0, device=device)
        pred_x0 = (x - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
        x = torch.sqrt(alpha_n) * pred_x0 + torch.sqrt(1 - alpha_n) * eps
        if t_next == 0:
            return x
        t_curr = t_next
    return x  # unreachable; loop returns


def main() -> None:
    sign_grid = np.load(SIGN_GRID)
    meta = yaml.safe_load(GRID_META.read_text())
    n_grid = meta["n_grid"]
    alphas = np.linspace(*meta["alpha_range"], n_grid)
    betas = np.linspace(*meta["beta_range"], n_grid)
    anchors_ab = [(a["alpha"], a["beta"]) for a in meta["anchors_alpha_beta"]]
    print(f"sign_grid: {sign_grid.shape} dtype={sign_grid.dtype}")

    print(f"selecting top-{TOP_K_CHANNELS} most balanced channels ...")
    top_idx = top_k_balanced_channels(sign_grid, k=TOP_K_CHANNELS)
    sign_top = sign_grid[:, :, top_idx]
    print(f"  selected channel indices: {top_idx.tolist()}")
    bmask = find_boundaries(sign_top)
    ids = region_ids(sign_top)
    print(f"  boundary cells = {bmask.sum()} / {bmask.size}")
    print(f"  regions        = {int(ids.max()) + 1}")

    print("loading model for sparse decoding ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = DDPMPipeline.from_pretrained(MODEL_ID)
    pipeline.to(device)
    unet = pipeline.unet
    scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    # Rebuild the plane in the SAME flat space, to compute grid h-vectors.
    h1, h2, h3 = (_load_h_flat(i).to(device) for i in range(3))
    origin, u, v = build_plane(h1, h2, h3)

    # Anchor 0's x_500 is the fixed encoder input across all decodings.
    x_500 = torch.load(ANCHOR_DIR / "x500_0.pt", weights_only=True).to(device)

    # 3x3 sparse sub-grid: pick alpha/beta indices in {16%, 50%, 84%} → indexes 8, 25, 41 for n=50.
    sample_idx = [int(n_grid * f) for f in (0.16, 0.5, 0.84)]
    decode_points = []
    print("decoding 9 representative grid points ...")
    for ai in sample_idx:
        for bi in sample_idx:
            a, b = float(alphas[ai]), float(betas[bi])
            h_target = (origin + a * u + b * v).reshape(1, CHANNELS, SPATIAL, SPATIAL)
            x0 = decode_from_h(unet, alphas_cumprod, x_500, h_target)
            img = _to_pil_array(x0)
            decode_points.append((a, b, img))
            print(f"  (α={a:.2f}, β={b:.2f}) ok  (region={int(ids[ai, bi])})")

    print(f"writing {OUT_FIG} ...")
    plot_boundary_panel(
        ids=ids,
        bmask=bmask,
        alphas=alphas,
        betas=betas,
        anchor_coords=anchors_ab,
        thumbnails=decode_points,
        out_path=OUT_FIG,
    )
    print("done.")


if __name__ == "__main__":
    main()
