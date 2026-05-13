"""Phase 1 smoke tests.

These tests verify that the diffusion environment is set up correctly and that
the pretrained DDPM CelebA-HQ model can be loaded. They do NOT run sampling
(that's covered by the script). The goal is to fail fast if the env is wrong.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn


def test_diffusers_importable():
    """Verify that diffusers exposes the API surface we plan to rely on."""
    from diffusers import DDIMScheduler, DDPMPipeline  # noqa: F401


@pytest.mark.gpu
def test_pipeline_loads_and_has_mid_block(model_id: str):
    """The pretrained pipeline must expose `unet.mid_block` as an nn.Module.

    Phase 2 will hook this submodule; if it doesn't exist or has changed
    name, the rest of the plan is invalid and we should know now.
    """
    from diffusers import DDPMPipeline

    pipeline = DDPMPipeline.from_pretrained(model_id)
    try:
        assert hasattr(pipeline, "unet"), "pipeline has no .unet"
        assert hasattr(pipeline.unet, "mid_block"), "unet has no .mid_block"
        assert isinstance(pipeline.unet.mid_block, nn.Module)
    finally:
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
