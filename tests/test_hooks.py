"""Phase 2 tests for MidBlockCapture.

These tests use a small synthetic nn.Module mimicking the mid_block interface
so they don't depend on downloading the pretrained pipeline. A separate GPU-
marked test verifies the real DDPM mid_block shape.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from diffusion_boundary.hooks import MidBlockCapture


class _FakeUNet(nn.Module):
    """A toy UNet-like module exposing `.mid_block` for hook testing."""

    def __init__(self, channels: int = 8, spatial: int = 4):
        super().__init__()
        self.encoder = nn.Conv2d(3, channels, kernel_size=1)
        self.mid_block = nn.Conv2d(channels, channels, kernel_size=1)
        self.decoder = nn.Conv2d(channels, 3, kernel_size=1)
        self._spatial = spatial

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.mid_block(x)
        return self.decoder(x)


@pytest.fixture
def fake_unet() -> _FakeUNet:
    torch.manual_seed(0)
    return _FakeUNet()


@pytest.fixture
def fake_input() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(1, 3, 4, 4)


def test_hook_attaches_and_releases(fake_unet: _FakeUNet, fake_input: torch.Tensor):
    """The hook count on mid_block must go up by 1 inside the with block,
    and back down after it exits."""
    before = len(fake_unet.mid_block._forward_hooks)
    with MidBlockCapture(fake_unet) as cap:
        assert len(fake_unet.mid_block._forward_hooks) == before + 1
        fake_unet(fake_input)
        assert cap.feature is not None
    assert len(fake_unet.mid_block._forward_hooks) == before


def test_hook_captures_correct_shape(fake_unet: _FakeUNet, fake_input: torch.Tensor):
    """Captured feature must match the mid_block output shape."""
    with MidBlockCapture(fake_unet) as cap:
        fake_unet(fake_input)
    assert cap.feature.shape == (1, 8, 4, 4)


def test_hook_is_deterministic(fake_unet: _FakeUNet, fake_input: torch.Tensor):
    """Two forward passes with identical input must produce identical features."""
    with MidBlockCapture(fake_unet) as cap1:
        fake_unet(fake_input)
    with MidBlockCapture(fake_unet) as cap2:
        fake_unet(fake_input)
    assert torch.equal(cap1.feature, cap2.feature)


def test_hook_releases_on_exception(fake_unet: _FakeUNet, fake_input: torch.Tensor):
    """If the with-body raises, the hook must still be removed."""
    before = len(fake_unet.mid_block._forward_hooks)
    with pytest.raises(RuntimeError, match="intentional"):
        with MidBlockCapture(fake_unet):
            fake_unet(fake_input)
            raise RuntimeError("intentional")
    assert len(fake_unet.mid_block._forward_hooks) == before


@pytest.mark.gpu
def test_real_mid_block_captures_expected_shape(model_id: str):
    """End-to-end check: capture h-vector from the real DDPM CelebA-HQ unet.

    Documents the actual bottleneck shape so Phase 4 sign-vector dim is known.
    """
    from diffusers import DDPMPipeline

    pipeline = DDPMPipeline.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(device)
    unet = pipeline.unet
    sample_size = unet.config.sample_size
    in_channels = unet.config.in_channels
    x = torch.randn(1, in_channels, sample_size, sample_size, device=device)
    t = torch.tensor([500], device=device)

    with MidBlockCapture(unet) as cap, torch.no_grad():
        _ = unet(x, t).sample
    h = cap.feature
    assert h.dim() == 4
    assert h.shape[0] == 1
    # Document the actual channel + spatial dims; this becomes Phase 4's input.
    print(f"\n[real mid_block] h.shape = {tuple(h.shape)}")
