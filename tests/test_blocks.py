import pytest
import torch
import torch.nn.functional as F
from torch import nn

from eve.config import preset
from eve.model.blocks import GELU, DecoderBlock, EncoderBlock, FeedForward, LayerNorm

CFG = preset("tiny")
D = CFG.d_model


def inputs(*shape, seed=1) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def test_layer_norm_matches_pytorch():
    norm = LayerNorm(D)
    with torch.no_grad():
        norm.scale.copy_(inputs(D, seed=2))
        norm.shift.copy_(inputs(D, seed=3))
    ref = nn.LayerNorm(D)
    with torch.no_grad():
        ref.weight.copy_(norm.scale)
        ref.bias.copy_(norm.shift)
    x = inputs(2, 5, D) * 3 + 1
    torch.testing.assert_close(norm(x), ref(x))


def test_layer_norm_gives_mean_zero_variance_one():
    y = LayerNorm(D)(inputs(2, 5, D) * 3 + 1)
    torch.testing.assert_close(y.mean(-1), torch.zeros(2, 5), atol=1e-5, rtol=0)
    torch.testing.assert_close(y.var(-1, unbiased=False), torch.ones(2, 5), atol=1e-3, rtol=0)


def test_gelu_matches_pytorch():
    x = torch.linspace(-6, 6, 101)
    torch.testing.assert_close(GELU()(x), F.gelu(x, approximate="tanh"))


def test_feed_forward():
    ff = FeedForward(CFG)
    assert ff(inputs(2, 5, D)).shape == (2, 5, D)
    hidden = CFG.ffn_mult * D
    assert sum(p.numel() for p in ff.parameters()) == 2 * D * hidden + hidden + D


def test_encoder_block_shapes_and_padding():
    block = EncoderBlock(CFG).eval()
    x = inputs(1, 5, D)
    assert block(x).shape == (1, 5, D)
    padded = torch.cat([x, inputs(1, 3, D, seed=7)], dim=1)
    mask = torch.tensor([[True] * 5 + [False] * 3])
    torch.testing.assert_close(block(padded, mask)[:, :5], block(x))


def test_encoder_block_is_bidirectional():
    block = EncoderBlock(CFG).eval()
    x = inputs(1, 8, D)
    changed = x.clone()
    changed[:, 5:] = inputs(1, 3, D, seed=9)
    assert not torch.allclose(block(x)[:, :5], block(changed)[:, :5])


@pytest.mark.parametrize("causal", [False, True])
def test_decoder_block_reads_memory(causal):
    block = DecoderBlock(CFG, causal=causal).eval()
    x, memory = inputs(2, 4, D), inputs(2, 6, D, seed=2)
    out = block(x, memory)
    assert out.shape == (2, 4, D)
    assert not torch.allclose(out, block(x, inputs(2, 6, D, seed=3)))

    padded = torch.cat([memory, inputs(2, 2, D, seed=7)], dim=1)
    mask = torch.tensor([[True] * 6 + [False] * 2] * 2)
    torch.testing.assert_close(block(x, padded, memory_mask=mask), out)


@pytest.mark.parametrize("causal", [False, True])
def test_decoder_block_causality(causal):
    block = DecoderBlock(CFG, causal=causal).eval()
    x, memory = inputs(1, 8, D), inputs(1, 6, D, seed=2)
    changed = x.clone()
    changed[:, 5:] = inputs(1, 3, D, seed=9)
    same = torch.allclose(block(x, memory)[:, :5], block(changed, memory)[:, :5])
    assert same == causal


def test_block_parameter_counts_match_config():
    attention = 4 * D * D + 4 * D
    hidden = CFG.ffn_mult * D
    feed_forward = 2 * D * hidden + hidden + D
    norm = 2 * D
    assert sum(p.numel() for p in EncoderBlock(CFG).parameters()) == (
        attention + feed_forward + 2 * norm
    )
    assert sum(p.numel() for p in DecoderBlock(CFG).parameters()) == (
        2 * attention + feed_forward + 3 * norm
    )
