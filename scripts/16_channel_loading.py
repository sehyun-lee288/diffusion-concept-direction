"""Phase 15: per-channel loading on (smile, gender) attribute axes.

For each of 512 mid_block channels, measure how strongly its
spatial-mean responds to the two attribute directions at t=500:

    A_c = ⟨Δh_smile_orth_500, e_c⟩  (per-channel spatial mean)
    B_c = ⟨Δh_gender_500,    e_c⟩

Both `Δh` are mean-shift directions, so A_c is the signed magnitude of
that channel's contribution to "smile ⊥ gender" and B_c to "gender".

We then categorize channels by their quadrant in (A_c, B_c) — pure
smile, pure gender, joint, or near-zero — and list the most extreme
channels per category.

We also compare orthogonalized smile vs original smile loadings to
quantify how much of the original smile loading came from gender.

Output:
  figures/exp11_channel_loading.png — scatter, marginals, top-K table
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

T = 500
CHANNELS = 512
TOP_N_TABLE = 8

MULTIATTR = REPO_ROOT / "data" / "delta_h_multiattr.pt"
ORTH = REPO_ROOT / "data" / "delta_h_smile_orth.pt"
FIG_OUT = REPO_ROOT / "figures" / "exp11_channel_loading.png"


def _spatial_mean(t: torch.Tensor) -> np.ndarray:
    return t.reshape(CHANNELS, -1).mean(-1).numpy()


def main() -> None:
    attr = torch.load(MULTIATTR, weights_only=False)
    smile_orth_dict = torch.load(ORTH, weights_only=False)
    key = min(smile_orth_dict.keys(), key=lambda k: abs(k - T))
    if key != T:
        print(f"using nearest cached t={key} for analysis (requested T={T})")

    A_orth = _spatial_mean(smile_orth_dict[key].float())      # smile ⊥ gender
    A_orig = _spatial_mean(attr["smile"][key].float())        # original smile
    B = _spatial_mean(attr["gender"][key].float())            # gender

    # Stats
    print(f"Magnitudes at t={T}:")
    print(f"  |A_orth|: min={np.abs(A_orth).min():.3f}  median={np.median(np.abs(A_orth)):.3f}  max={np.abs(A_orth).max():.3f}")
    print(f"  |A_orig|: min={np.abs(A_orig).min():.3f}  median={np.median(np.abs(A_orig)):.3f}  max={np.abs(A_orig).max():.3f}")
    print(f"  |B|:      min={np.abs(B).min():.3f}  median={np.median(np.abs(B)):.3f}  max={np.abs(B).max():.3f}")
    print(f"  Corr(A_orig, B) = {np.corrcoef(A_orig, B)[0, 1]:+.3f}  (should be high if smile-gender entangled)")
    print(f"  Corr(A_orth, B) = {np.corrcoef(A_orth, B)[0, 1]:+.3f}  (should be ~0 after orthogonalization)")

    # Categorize channels by quadrant magnitude.
    # Use ratio criterion: a channel "loads on X" if |X| > 0.6 of ||(A, B)||.
    norm = np.hypot(A_orth, B)
    smile_pur = np.abs(A_orth) / np.maximum(norm, 1e-12)   # [0, 1] — 1 = pure smile
    gender_pur = np.abs(B) / np.maximum(norm, 1e-12)

    cat = np.full(CHANNELS, "joint", dtype=object)
    cat[smile_pur > 0.85] = "smile"
    cat[gender_pur > 0.85] = "gender"
    cat[norm < np.quantile(norm, 0.25)] = "weak"
    print("\nCategory counts:")
    for c in ("smile", "gender", "joint", "weak"):
        print(f"  {c:>7}: {(cat == c).sum()}")

    # ---- Figure: scatter + marginals + top-K table -----------------------
    fig = plt.figure(figsize=(13.5, 8.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 4, 1.6],
                          width_ratios=[4, 1, 5], hspace=0.35, wspace=0.25)

    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_x = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
    ax_y = fig.add_subplot(gs[1, 1], sharey=ax_scatter)
    ax_table = fig.add_subplot(gs[:2, 2])
    ax_corr = fig.add_subplot(gs[2, :])

    cmap_pts = {"smile": "crimson", "gender": "steelblue",
                "joint": "purple", "weak": "lightgray"}
    for c in ("weak", "joint", "smile", "gender"):
        mask = cat == c
        ax_scatter.scatter(A_orth[mask], B[mask], s=10,
                           c=cmap_pts[c], alpha=0.55, label=f"{c} ({mask.sum()})")
    ax_scatter.axhline(0, color="black", linewidth=0.4, alpha=0.5)
    ax_scatter.axvline(0, color="black", linewidth=0.4, alpha=0.5)
    ax_scatter.set_xlabel(r"$A_c = \langle \Delta h_{smile\;\perp\;gender}, e_c\rangle$")
    ax_scatter.set_ylabel(r"$B_c = \langle \Delta h_{gender}, e_c\rangle$")
    ax_scatter.legend(loc="lower right", fontsize=8)
    ax_scatter.set_title("per-channel attribute loadings (t=500)", fontsize=11)

    ax_x.hist(A_orth, bins=40, color="crimson", alpha=0.7)
    ax_x.set_ylabel("# ch")
    ax_x.tick_params(labelbottom=False)
    ax_y.hist(B, bins=40, orientation="horizontal", color="steelblue", alpha=0.7)
    ax_y.set_xlabel("# ch")
    ax_y.tick_params(labelleft=False)

    # Tables of top channels by category.
    def fmt_row(idx):
        return f"  ch {idx:3d}  A={A_orth[idx]:+5.2f}  B={B[idx]:+5.2f}  |·|={norm[idx]:.2f}"

    smile_top = np.argsort(-np.abs(A_orth) * (cat == "smile"))[:TOP_N_TABLE]
    gender_top = np.argsort(-np.abs(B) * (cat == "gender"))[:TOP_N_TABLE]
    joint_top = np.argsort(-norm * (cat == "joint"))[:TOP_N_TABLE]
    lines = ["Top smile-only channels:"]
    lines += [fmt_row(c) for c in smile_top]
    lines += ["", "Top gender-only channels:"]
    lines += [fmt_row(c) for c in gender_top]
    lines += ["", "Top joint channels:"]
    lines += [fmt_row(c) for c in joint_top]
    ax_table.text(0.0, 1.0, "\n".join(lines),
                  family="monospace", fontsize=9, va="top", ha="left")
    ax_table.axis("off")

    # Correlation row: A_orig vs B should show entanglement that A_orth vs B doesn't.
    ax_corr.scatter(A_orig, B, s=8, color="gray", alpha=0.5, label="A_orig vs B")
    ax_corr.scatter(A_orth, B, s=8, color="crimson", alpha=0.5, label="A_orth vs B")
    ax_corr.axhline(0, color="black", linewidth=0.3)
    ax_corr.axvline(0, color="black", linewidth=0.3)
    ax_corr.set_xlabel(r"per-channel smile loading  ($A_c$)")
    ax_corr.set_ylabel(r"$B_c$")
    ax_corr.legend(loc="upper left", fontsize=8)
    ax_corr.set_title(f"Entanglement check — corr(A_orig, B)={np.corrcoef(A_orig, B)[0, 1]:+.3f},  "
                      f"corr(A_orth, B)={np.corrcoef(A_orth, B)[0, 1]:+.3f}",
                      fontsize=10)

    fig.suptitle("Phase 15 — per-channel attribute loadings", fontsize=12)
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150, bbox_inches="tight")
    print(f"saved {FIG_OUT}")


if __name__ == "__main__":
    main()
