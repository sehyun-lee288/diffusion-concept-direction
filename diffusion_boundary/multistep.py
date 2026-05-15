"""Multi-step Δh extraction and trajectory editing.

Two reusable utilities lifted out of `scripts/12_multistep_smile.py`:

- `extract_delta_h_multistep`: given two image groups (positive vs
  negative for some attribute), compute the mean-shift direction
  `Δh_t = mean(h | pos) − mean(h | neg)` at each requested timestep,
  using q-sample(seed=idx) for noise so the noise variance averages out.

- `denoise_with_injection`: DDIM denoise from x_T with a list of
  `(delta_dict, scale)` pairs added to the mid_block output at every
  step. The pre-computed `delta_dict_per_attr` use nearest-timestep
  lookup; multiple attributes can be combined linearly (this is what
  makes the attribute-paired plane analysis possible).

Both functions take the unet and scheduler explicitly so they're
testable and side-effect free except for the requested hook.
"""
from __future__ import annotations

import gc
from collections.abc import Callable, Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn

from diffusion_boundary.hooks import MidBlockCapture
from diffusion_boundary.inversion import noise_to_t


def _pil_to_tensor(img: Image.Image, size: int = 256) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB").resize((size, size), Image.BICUBIC),
                     dtype=np.float32) / 255.0
    return torch.from_numpy(arr * 2.0 - 1.0).permute(2, 0, 1).unsqueeze(0)


def capture_h_at_t(unet: nn.Module, scheduler, imgs: list[Image.Image], t: int, device: str) -> torch.Tensor:
    """For each image: q-sample to t with seed=index, forward unet, capture h."""
    t_tensor = torch.tensor([t], device=device)
    hs = []
    for i, img in enumerate(imgs):
        x0 = _pil_to_tensor(img).to(device)
        x_t = noise_to_t(x0, scheduler, target_t=t, seed=i)
        with MidBlockCapture(unet) as cap, torch.no_grad():
            _ = unet(x_t, t_tensor).sample
        hs.append(cap.feature)
    return torch.cat(hs, dim=0)


def extract_delta_h_multistep(
    unet: nn.Module,
    scheduler,
    pos_imgs: list[Image.Image],
    neg_imgs: list[Image.Image],
    timesteps: Sequence[int],
    device: str,
    progress: Callable[[str], None] | None = None,
) -> dict[int, torch.Tensor]:
    """Return {t: Δh_t} where Δh_t = mean(h|pos) − mean(h|neg) at timestep t.

    Each Δh_t is shape (1, C, H, W) on CPU.
    """
    out: dict[int, torch.Tensor] = {}
    for t in timesteps:
        h_pos = capture_h_at_t(unet, scheduler, pos_imgs, t, device).mean(0, keepdim=True)
        h_neg = capture_h_at_t(unet, scheduler, neg_imgs, t, device).mean(0, keepdim=True)
        delta = (h_pos - h_neg).detach().cpu()
        out[int(t)] = delta
        if progress is not None:
            progress(f"  t={t}  ||Δh||={delta.norm().item():.2f}")
    return out


def denoise_with_injection(
    unet: nn.Module,
    scheduler,
    x_T: torch.Tensor,
    delta_dicts: Sequence[dict[int, torch.Tensor]],
    scales: Sequence[float],
    num_steps: int,
    device: str,
) -> torch.Tensor:
    """DDIM denoise from x_T with `Σ_i scale_i · Δh_t^(i)` added to the
    mid_block output at every step. Returns x_0."""
    if len(delta_dicts) != len(scales):
        raise ValueError("delta_dicts and scales must have matching length")
    scheduler.set_timesteps(num_steps)
    # Pre-stage every delta on device, indexed by attribute then key.
    keys_per_attr = [np.array(sorted(d.keys())) for d in delta_dicts]
    staged = [{k: d[k].to(device) for k in d} for d in delta_dicts]

    x = x_T.clone()
    for t in scheduler.timesteps:
        t_int = int(t.item())
        # Total inject for this timestep across all attributes (linear combination).
        total = None
        for keys, st, s in zip(keys_per_attr, staged, scales, strict=True):
            nearest = int(keys[np.argmin(np.abs(keys - t_int))])
            term = s * st[nearest]
            total = term if total is None else total + term

        def inject(_m, _i, output, _d=total):
            if isinstance(output, tuple):
                return (output[0] + _d,) + output[1:]
            return output + _d

        handle = unet.mid_block.register_forward_hook(inject)
        try:
            with torch.no_grad():
                eps = unet(x, t.to(device)).sample
        finally:
            handle.remove()
        x = scheduler.step(eps, t, x).prev_sample
    return x


def orthogonalize_against(
    target: dict[int, torch.Tensor], basis: dict[int, torch.Tensor]
) -> dict[int, torch.Tensor]:
    """Per-timestep Gram-Schmidt: remove the basis component from target.

    `target_orth_t = target_t − ⟨target_t, basis_t⟩ / ⟨basis_t, basis_t⟩ · basis_t`

    Each Δh stays at its own timestep — we project per-t, not globally.
    """
    if set(target.keys()) != set(basis.keys()):
        raise ValueError("target and basis must cover the same timesteps")
    out: dict[int, torch.Tensor] = {}
    for t in target:
        tgt = target[t].reshape(-1)
        bas = basis[t].reshape(-1)
        coeff = torch.dot(tgt, bas) / torch.dot(bas, bas).clamp(min=1e-12)
        orth = tgt - coeff * bas
        out[t] = orth.reshape(target[t].shape).clone()
    return out


def collect_images_by_attribute(
    dataset_id: str, attribute_idx: int, target_val: int, n_needed: int
) -> list[Image.Image]:
    """Stream a HuggingFace dataset and return n_needed PIL copies whose
    `attributes[attribute_idx] == target_val`. The dataset iterator is
    released before returning."""
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split="train", streaming=True)
    out: list[Image.Image] = []
    try:
        for sample in ds:
            if int(sample["attributes"][attribute_idx]) == target_val:
                out.append(sample["image"].copy())
                if len(out) >= n_needed:
                    break
    finally:
        del ds
        gc.collect()
    return out


__all__ = [
    "capture_h_at_t",
    "collect_images_by_attribute",
    "denoise_with_injection",
    "extract_delta_h_multistep",
    "orthogonalize_against",
]
