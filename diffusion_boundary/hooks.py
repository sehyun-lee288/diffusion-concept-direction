"""Forward-hook utilities for capturing U-Net bottleneck (h-space) features.

Single responsibility: attach a forward hook to `unet.mid_block`, store its
output, release the hook on exit. Spatial pooling, sign computation, and
downstream analysis live elsewhere.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle


class MidBlockCapture:
    """Context manager that captures the output of `unet.mid_block`.

    Usage::

        with MidBlockCapture(unet) as cap:
            unet(x_t, t)
        h = cap.feature   # torch.Tensor of shape (B, C, H, W)

    The captured tensor is detached from the autograd graph but kept on the
    original device. Callers that don't need GPU memory should `.cpu()`
    themselves — keeping device explicit avoids surprise copies.
    """

    def __init__(self, unet: nn.Module, *, detach: bool = True):
        if not hasattr(unet, "mid_block"):
            raise AttributeError("provided module has no `mid_block` submodule")
        self._target: nn.Module = unet.mid_block
        self._detach = detach
        self._handle: RemovableHandle | None = None
        self.feature: torch.Tensor | None = None

    def _hook(self, _module, _inputs, output):
        # Some block implementations return a tuple/dict; pick the primary tensor.
        if isinstance(output, tuple):
            output = output[0]
        elif isinstance(output, dict) and "sample" in output:
            output = output["sample"]
        self.feature = output.detach() if self._detach else output

    def __enter__(self) -> MidBlockCapture:
        self._handle = self._target.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
