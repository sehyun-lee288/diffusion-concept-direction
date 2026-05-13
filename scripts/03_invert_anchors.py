"""Phase 3: download 3 CelebA-HQ anchors, compute x_500 and h_500 for each.

Pipeline per anchor:
    image (PIL, RGB, 256x256)
    → to tensor in [-1, 1]
    → q-sample to t=500 with a fixed seed       (diffusion_boundary.inversion)
    → forward through unet, hook mid_block       (diffusion_boundary.hooks)
    → save x_500 latent and h_500 feature

Outputs (under data/anchors/):
    anchor_{0,1,2}.png       source images (256x256 RGB)
    x500_{0,1,2}.pt          latent at t=500           (1, 3, 256, 256)
    h500_{0,1,2}.pt          mid_block output at t=500 (1, 512, 8, 8)
    meta.yaml                source dataset + indices  (for reproducibility)
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402
from datasets import load_dataset  # noqa: E402
from diffusers import DDIMScheduler, DDPMPipeline  # noqa: E402
from PIL import Image  # noqa: E402

from diffusion_boundary.hooks import MidBlockCapture  # noqa: E402
from diffusion_boundary.inversion import noise_to_t  # noqa: E402

MODEL_ID = "google/ddpm-celebahq-256"
DATASET_ID = "mattymchen/celeba-hq"
ANCHOR_DIR = REPO_ROOT / "data" / "anchors"
TARGET_T = 500
ANCHOR_INDICES = [0, 1000, 5000]  # spread out indices for visual diversity
NOISE_SEEDS = [10, 20, 30]        # one per anchor


def _to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB → (1, 3, 256, 256) tensor in [-1, 1]."""
    img = img.convert("RGB").resize((256, 256), Image.BICUBIC)
    arr = torch.from_numpy(_pil_to_numpy(img)).float() / 255.0
    return (arr * 2.0 - 1.0).permute(2, 0, 1).unsqueeze(0)


def _pil_to_numpy(img: Image.Image):
    import numpy as np
    return np.asarray(img)


def _save_tensor(t: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(t.detach().cpu(), path)


def _fetch_anchor_images(dataset_id: str, indices: set[int]) -> dict[int, Image.Image]:
    """Stream the dataset, deep-copy the requested PIL images, then release.

    Scoping the stream to its own function ensures the IterableDataset is
    garbage-collected before model work begins. Otherwise its background
    download threads can race with interpreter shutdown.
    """
    import gc
    ds = load_dataset(dataset_id, split="train", streaming=True)
    collected: dict[int, Image.Image] = {}
    try:
        for idx, sample in enumerate(ds):
            if idx in indices:
                # .copy() detaches the PIL image from any dataset-internal handle.
                collected[idx] = sample["image"].copy()
            if len(collected) == len(indices):
                break
    finally:
        del ds
        gc.collect()
    return collected


def main() -> None:
    ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {MODEL_ID} ...")
    pipeline = DDPMPipeline.from_pretrained(MODEL_ID)
    pipeline.to(device)
    unet = pipeline.unet
    scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    print(f"Streaming {DATASET_ID} ...")
    collected = _fetch_anchor_images(DATASET_ID, set(ANCHOR_INDICES))
    assert len(collected) == 3, f"only collected {len(collected)} images"

    meta = {
        "model_id": MODEL_ID,
        "dataset_id": DATASET_ID,
        "target_t": TARGET_T,
        "anchors": [],
    }
    for i, ds_idx in enumerate(ANCHOR_INDICES):
        img = collected[ds_idx]
        img.save(ANCHOR_DIR / f"anchor_{i}.png")
        x0 = _to_tensor(img).to(device)
        x_t = noise_to_t(x0, scheduler, target_t=TARGET_T, seed=NOISE_SEEDS[i])
        t_tensor = torch.tensor([TARGET_T], device=device)
        with MidBlockCapture(unet) as cap, torch.no_grad():
            _ = unet(x_t, t_tensor).sample
        h = cap.feature
        _save_tensor(x_t, ANCHOR_DIR / f"x500_{i}.pt")
        _save_tensor(h, ANCHOR_DIR / f"h500_{i}.pt")
        print(f"anchor[{i}] ds_idx={ds_idx} seed={NOISE_SEEDS[i]} "
              f"x500={tuple(x_t.shape)} h500={tuple(h.shape)}")
        meta["anchors"].append({
            "index": i,
            "dataset_index": ds_idx,
            "noise_seed": NOISE_SEEDS[i],
            "png_path": str((ANCHOR_DIR / f"anchor_{i}.png").relative_to(REPO_ROOT)),
            "x500_path": str((ANCHOR_DIR / f"x500_{i}.pt").relative_to(REPO_ROOT)),
            "h500_path": str((ANCHOR_DIR / f"h500_{i}.pt").relative_to(REPO_ROOT)),
        })

    with (ANCHOR_DIR / "meta.yaml").open("w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    print(f"meta saved to {ANCHOR_DIR / 'meta.yaml'}")


if __name__ == "__main__":
    main()
    # `datasets.IterableDataset` keeps fsspec/aiohttp background threads alive
    # past `main`'s return on this platform, and they SIGABRT on interpreter
    # finalization. All our work is synchronous and persisted to disk by here,
    # so a hard exit is safe and avoids spurious non-zero exit codes.
    import os
    os._exit(0)
