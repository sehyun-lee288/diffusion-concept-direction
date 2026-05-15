"""Add per-neuron boundary visualizations for the two main planes.

Figure A (`exp13_random_anchor_all_lines.png`)
  All-active-channel spatial-mean boundary lines on the random-anchor
  plane. Complements `exp1_boundary.png` (region only) and
  `exp1_radial_analysis.png` (top-K only) — now you can see every
  active line at once and confirm no model-level radial concentration.

Figure B (`exp14_attribute_plane_class_lines.png`)
  Attribute-paired plane (smile_orth × gender) with boundary lines
  colored by Phase-15 channel category (smile-pure red, gender-pure
  blue, joint purple, weak gray) plus the 5×5 decoded thumbnails from
  Phase 13 overlaid. Tests the prediction that smile-pure boundaries
  are roughly vertical (α-determining) and gender-pure boundaries are
  roughly horizontal (β-determining), because B_c ≈ 0 and A_c ≈ 0
  respectively for those classes.
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
from diffusion_boundary.plane import build_plane  # noqa: E402
from diffusion_boundary.viz import (  # noqa: E402
    active_channel_mask,
    overlay_thumbnails,
)

CHANNELS = 512
SPATIAL = 8
T_ATTR = 450

ANCHOR_DIR = REPO_ROOT / "data" / "anchors"
SIGN_GRID = REPO_ROOT / "data" / "sign_grid.npy"
MULTIATTR = REPO_ROOT / "data" / "delta_h_multiattr.pt"
ORTH = REPO_ROOT / "data" / "delta_h_smile_orth.pt"
FIG_A = REPO_ROOT / "figures" / "exp13_random_anchor_all_lines.png"
FIG_B = REPO_ROOT / "figures" / "exp14_attribute_plane_class_lines.png"

ATTR_RANGE = (-2.5, 2.5)
RANDOM_RANGE = (-0.5, 1.5)
N_THUMB = 5
NUM_DDIM_STEPS = 50
MODEL_ID = "google/ddpm-celebahq-256"


def _spatial_mean(t: torch.Tensor) -> np.ndarray:
    return t.reshape(CHANNELS, -1).mean(-1).numpy()


def _load_h(i: int) -> torch.Tensor:
    return torch.load(ANCHOR_DIR / f"h500_{i}.pt", weights_only=True).float().reshape(-1)


def _plot_lines(ax, A, B, C, mask, *, color, alpha, lw=0.8, alpha_range=RANDOM_RANGE):
    """Plot `A_c α + B_c β + C_c = 0` for channels in `mask`."""
    a_grid = np.linspace(*alpha_range, 240)
    drawn = 0
    for c in np.where(mask)[0]:
        Ac, Bc, Cc = A[c], B[c], C[c]
        if abs(Bc) > 1e-8:
            b_line = -(Ac * a_grid + Cc) / Bc
            keep = (b_line >= alpha_range[0] - 0.5) & (b_line <= alpha_range[1] + 0.5)
            if keep.any():
                ax.plot(a_grid[keep], b_line[keep], color=color, alpha=alpha, linewidth=lw)
                drawn += 1
        else:
            x = -Cc / Ac if abs(Ac) > 1e-8 else None
            if x is not None and alpha_range[0] <= x <= alpha_range[1]:
                ax.axvline(x, color=color, alpha=alpha, linewidth=lw)
                drawn += 1
    return drawn


def make_figure_a() -> None:
    """All-active boundary lines on the random-anchor plane."""
    h1, h2, h3 = (_load_h(i) for i in range(3))
    origin, u, v = build_plane(h1, h2, h3)
    A = _spatial_mean(u)        # mean(h2 - h1)
    B = _spatial_mean(v)        # mean(v)
    C = _spatial_mean(origin)   # mean(h1)

    # Active channel mask from the existing sign_grid (where each channel's
    # sign actually changes inside the grid window).
    sign_grid = np.load(SIGN_GRID)
    active = active_channel_mask(sign_grid)
    n_active = int(active.sum())
    print(f"random-anchor plane: {n_active} / {CHANNELS} active channels")

    a3 = float(((h3 - origin) @ u) / (u @ u))
    b3 = float(((h3 - origin) @ v) / (v @ v))
    anchors = [(0.0, 0.0), (1.0, 0.0), (a3, b3)]

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    drawn = _plot_lines(ax, A, B, C, active,
                        color="steelblue", alpha=0.18, lw=0.6,
                        alpha_range=RANDOM_RANGE)
    for k, (a, b) in enumerate(anchors):
        ax.scatter([a], [b], marker="*", s=240, c="white",
                   edgecolors="black", linewidths=1.5)
        ax.annotate(f"h{k + 1}", (a, b), textcoords="offset points",
                    xytext=(8, 6), fontsize=10)
    ax.set_xlim(*RANDOM_RANGE)
    ax.set_ylim(*RANDOM_RANGE)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta$")
    ax.set_title(
        f"Random-anchor plane — all {drawn} active neuron boundary lines\n"
        f"(spatial-mean sign per channel, t=500)",
        fontsize=11,
    )
    FIG_A.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_A, dpi=150)
    plt.close(fig)
    print(f"saved {FIG_A}")


def _decode_attribute_grid(
    device: str,
    smile_orth_dict: dict[int, torch.Tensor],
    gender_dict: dict[int, torch.Tensor],
) -> list[tuple[float, float, np.ndarray]]:
    """Multi-step decode of a 5×5 grid; same procedure as scripts/14_*.py."""
    print("decoding 5x5 attribute grid (re-running Phase 13 for thumbnails) ...")
    pipeline = DDPMPipeline.from_pretrained(MODEL_ID).to(device)
    unet = pipeline.unet
    scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    torch.manual_seed(0)
    x_T = torch.randn(1, 3, 256, 256, device=device)

    alphas = np.linspace(*ATTR_RANGE, N_THUMB)
    betas = np.linspace(*ATTR_RANGE, N_THUMB)
    thumbs: list[tuple[float, float, np.ndarray]] = []
    k = 0
    for b in betas:
        for a in alphas:
            x0 = denoise_with_injection(unet, scheduler, x_T,
                                        [smile_orth_dict, gender_dict],
                                        [float(a), float(b)],
                                        NUM_DDIM_STEPS, device)
            img = np.array(Image.fromarray(to_uint8_image(x0)).resize((96, 96), Image.BICUBIC))
            thumbs.append((float(a), float(b), img))
            k += 1
            print(f"  {k}/{N_THUMB * N_THUMB}  (α={a:+.2f}, β={b:+.2f})")
    return thumbs


def make_figure_b() -> None:
    """Channel-category-colored lines on attribute plane + decoded thumbnails."""
    h_anchor = torch.load(ANCHOR_DIR / "h500_0.pt", weights_only=True).float()
    attr = torch.load(MULTIATTR, weights_only=False)
    smile_orth_dict = torch.load(ORTH, weights_only=False)
    key = min(smile_orth_dict.keys(), key=lambda k: abs(k - T_ATTR))

    A = _spatial_mean(smile_orth_dict[key].float())
    B = _spatial_mean(attr["gender"][key].float())
    C = _spatial_mean(h_anchor)

    norm = np.hypot(A, B)
    smile_pur = np.abs(A) / np.maximum(norm, 1e-12)
    gender_pur = np.abs(B) / np.maximum(norm, 1e-12)
    cat = np.full(CHANNELS, "joint", dtype=object)
    cat[smile_pur > 0.85] = "smile"
    cat[gender_pur > 0.85] = "gender"
    cat[norm < np.quantile(norm, 0.25)] = "weak"
    counts = {c: int((cat == c).sum()) for c in ["smile", "gender", "joint", "weak"]}
    print(f"channel categories: {counts}")

    # Decode 5x5 thumbnails for overlay (re-runs the Phase 13 decoding).
    # denoise_with_injection expects per-attribute dicts {t: tensor}, not single
    # tensors — so pass the whole dicts through.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    thumbs = _decode_attribute_grid(device, smile_orth_dict, attr["gender"]) \
             if device == "cuda" else []

    # Plot: lines first (background), then thumbnails on top.
    fig, ax = plt.subplots(figsize=(9.5, 9.0))
    palette = {"smile": "crimson", "gender": "steelblue",
               "joint": "purple", "weak": "lightgray"}
    alphas = {"smile": 0.42, "gender": 0.42, "joint": 0.22, "weak": 0.08}
    order = ["weak", "joint", "gender", "smile"]  # draw weak first (background)
    for c in order:
        mask = (cat == c)
        n = _plot_lines(ax, A, B, C, mask, color=palette[c], alpha=alphas[c],
                        lw=0.8, alpha_range=ATTR_RANGE)
        print(f"  drew {n} {c} lines")

    # Legend.
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=palette[c], lw=2,
                      label=f"{c}-pure ({counts[c]})") for c in ["smile", "gender", "joint", "weak"]]
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9)

    if thumbs:
        overlay_thumbnails(ax, thumbs, zoom=0.32, frame=True)

    ax.axhline(0, color="black", linewidth=0.4, alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.4, alpha=0.5)
    ax.set_xlim(ATTR_RANGE[0] - 0.5, ATTR_RANGE[1] + 0.5)
    ax.set_ylim(ATTR_RANGE[0] - 0.5, ATTR_RANGE[1] + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\alpha$  (smile ⊥ gender)", fontsize=11)
    ax.set_ylabel(r"$\beta$  (gender)", fontsize=11)
    ax.set_title(
        f"Attribute plane — neuron boundary lines by channel category (t={key})\n"
        f"prediction: smile-pure ≈ vertical (α-determining), gender-pure ≈ horizontal",
        fontsize=11,
    )

    FIG_B.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_B, dpi=150)
    plt.close(fig)
    print(f"saved {FIG_B}")


def main() -> None:
    make_figure_a()
    make_figure_b()
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
