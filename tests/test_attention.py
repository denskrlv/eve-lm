import pytest
import torch
from torch import nn

from eve.model.attention import MultiHeadAttention, apply_rope, rope_angles

D, H = 32, 4


def make(causal=False, seed=0, **kwargs) -> MultiHeadAttention:
    torch.manual_seed(seed)
    return MultiHeadAttention(D, H, causal=causal, **kwargs).eval()


def inputs(*shape, seed=1) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def both_paths(attn, *args, **kwargs):
    attn.use_sdpa = False
    manual = attn(*args, **kwargs)
    attn.use_sdpa = True
    fast = attn(*args, **kwargs)
    attn.use_sdpa = False
    return manual, fast


MASK = torch.tensor([[True] * 7, [True] * 4 + [False] * 3])  # 2nd sequence: 4 tokens + 3 pads


def test_parameters():
    attn = make()
    for name in ("W_query", "W_key", "W_value", "out_proj"):
        layer = getattr(attn, name)
        assert isinstance(layer, nn.Linear) and layer.bias is not None
        assert layer.weight.shape == (D, D)
    assert sum(p.numel() for p in attn.parameters()) == 4 * D * D + 4 * D


def test_heads_must_divide_d_model():
    with pytest.raises(ValueError):
        MultiHeadAttention(30, 4)


def test_shapes():
    assert make()(inputs(2, 7, D)).shape == (2, 7, D)
    assert make()(inputs(2, 5, D), kv=inputs(2, 9, D)).shape == (2, 5, D)  # one output per query


@pytest.mark.parametrize("causal", [False, True])
def test_matches_pytorch_multihead_attention(causal):
    """An independent check of the head splitting and merging: copy our weights into PyTorch's
    own implementation and compare."""
    attn = make(causal)
    ref = nn.MultiheadAttention(D, H, batch_first=True).eval()
    with torch.no_grad():
        ref.in_proj_weight.copy_(
            torch.cat([attn.W_query.weight, attn.W_key.weight, attn.W_value.weight])
        )
        ref.in_proj_bias.copy_(torch.cat([attn.W_query.bias, attn.W_key.bias, attn.W_value.bias]))
        ref.out_proj.weight.copy_(attn.out_proj.weight)
        ref.out_proj.bias.copy_(attn.out_proj.bias)

    x = inputs(2, 7, D)
    future = torch.triu(torch.ones(7, 7, dtype=torch.bool), diagonal=1) if causal else None
    expected, _ = ref(x, x, x, key_padding_mask=~MASK, attn_mask=future)
    torch.testing.assert_close(attn(x, key_mask=MASK), expected)

    if not causal:  # cross-attention
        q = inputs(2, 5, D, seed=2)
        expected, _ = ref(q, x, x, key_padding_mask=~MASK)
        torch.testing.assert_close(attn(q, kv=x, key_mask=MASK), expected)


@pytest.mark.parametrize("causal", [False, True])
def test_manual_path_matches_sdpa(causal):
    x = inputs(2, 7, D)
    torch.testing.assert_close(*both_paths(make(causal), x))
    torch.testing.assert_close(*both_paths(make(causal), x, key_mask=MASK))


def test_manual_path_matches_sdpa_for_cross_attention():
    q, kv = inputs(2, 5, D), inputs(2, 7, D, seed=2)
    torch.testing.assert_close(*both_paths(make(), q, kv=kv, key_mask=MASK))


@pytest.mark.parametrize("use_sdpa", [False, True])
def test_causal_ignores_future_tokens(use_sdpa):
    attn = make(causal=True)
    attn.use_sdpa = use_sdpa
    x = inputs(1, 8, D)
    changed = x.clone()
    changed[:, 5:] = inputs(1, 3, D, seed=9)
    torch.testing.assert_close(attn(x)[:, :5], attn(changed)[:, :5])


def test_bidirectional_sees_future_tokens():
    attn = make(causal=False)
    x = inputs(1, 8, D)
    changed = x.clone()
    changed[:, 5:] = inputs(1, 3, D, seed=9)
    assert not torch.allclose(attn(x)[:, :5], attn(changed)[:, :5])


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("use_sdpa", [False, True])
def test_padding_does_not_change_real_outputs(causal, use_sdpa):
    attn = make(causal)
    attn.use_sdpa = use_sdpa
    x = inputs(1, 5, D)
    padded = torch.cat([x, inputs(1, 3, D, seed=7)], dim=1)  # pad vectors can be anything
    mask = torch.tensor([[True] * 5 + [False] * 3])
    torch.testing.assert_close(attn(padded, key_mask=mask)[:, :5], attn(x))


@pytest.mark.parametrize("use_sdpa", [False, True])
def test_padding_does_not_change_cross_attention(use_sdpa):
    attn = make()
    attn.use_sdpa = use_sdpa
    q, kv = inputs(1, 4, D), inputs(1, 6, D, seed=2)
    padded = torch.cat([kv, inputs(1, 2, D, seed=7)], dim=1)
    mask = torch.tensor([[True] * 6 + [False] * 2])
    torch.testing.assert_close(attn(q, kv=padded, key_mask=mask), attn(q, kv=kv))


def test_attention_weights():
    x = inputs(2, 7, D)
    _, weights = make(causal=True)(x, key_mask=MASK, return_weights=True)
    assert weights.shape == (2, H, 7, 7)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, H, 7))
    future = torch.triu(torch.ones(7, 7, dtype=torch.bool), diagonal=1)
    assert (weights[..., future] == 0).all()
    assert (weights[1, :, :, 4:] == 0).all()  # padding keys

    _, weights = make()(inputs(2, 5, D), kv=x, key_mask=MASK, return_weights=True)
    assert weights.shape == (2, H, 5, 7)


def test_cross_attention_is_never_causal():
    with pytest.raises(ValueError):
        make(causal=True)(inputs(1, 3, D), kv=inputs(1, 4, D))


@pytest.mark.parametrize("use_sdpa", [False, True])
def test_dropout_only_while_training(use_sdpa):
    attn = make(dropout=0.5)
    attn.use_sdpa = use_sdpa
    x = inputs(1, 6, D)
    torch.testing.assert_close(attn(x), attn(x))  # eval(): no dropout
    attn.train()
    assert not torch.allclose(attn(x), attn(x))


def test_gradients_reach_every_parameter():
    attn = make(causal=True)
    attn(inputs(2, 7, D), key_mask=MASK).sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in attn.parameters())


# ---- options: RoPE and QK-norm ---------------------------------------------------------------

OPTIONS = {"rope": True, "qk_norm": True}


def test_rope_scores_depend_only_on_the_offset():
    cos, sin = rope_angles(10, 8, 10000.0, torch.device("cpu"))
    q = apply_rope(inputs(1, 8).expand(10, 8), cos, sin)  # the same query at every position
    k = apply_rope(inputs(1, 8, seed=2).expand(10, 8), cos, sin)
    torch.testing.assert_close(q[3] @ k[1], q[7] @ k[5])  # offset 2 both times
    assert not torch.allclose(q[3] @ k[1], q[3] @ k[2])  # offset 1: a different score
    torch.testing.assert_close(q.norm(dim=-1), q[0].norm().expand(10))  # rotations keep lengths


def test_rope_tells_identical_tokens_apart():
    x = inputs(1, 2, D).repeat(1, 3, 1)  # A B A B A B: without positions, every A gets one output
    assert torch.allclose(make()(x)[0, 0], make()(x)[0, 2])
    out = make(rope=True)(x)
    assert not torch.allclose(out[0, 0], out[0, 2])


def test_rope_leaves_cross_attention_alone():
    q, kv = inputs(2, 5, D), inputs(2, 7, D, seed=2)
    torch.testing.assert_close(make(rope=True)(q, kv=kv), make()(q, kv=kv))


def test_qk_norm_parameters():
    attn = make(qk_norm=True)
    assert attn.q_norm.scale.shape == attn.k_norm.scale.shape == (D // H,)
    assert sum(p.numel() for p in attn.parameters()) == 4 * D * D + 4 * D + 2 * (D // H)


@pytest.mark.parametrize("causal", [False, True])
def test_options_manual_path_matches_sdpa(causal):
    x = inputs(2, 7, D)
    torch.testing.assert_close(*both_paths(make(causal, **OPTIONS), x, key_mask=MASK))
    if not causal:
        q = inputs(2, 5, D, seed=2)
        torch.testing.assert_close(*both_paths(make(**OPTIONS), q, kv=x, key_mask=MASK))


@pytest.mark.parametrize("use_sdpa", [False, True])
def test_options_keep_causality_and_padding(use_sdpa):
    attn = make(causal=True, **OPTIONS)
    attn.use_sdpa = use_sdpa
    x = inputs(1, 8, D)
    changed = x.clone()
    changed[:, 5:] = inputs(1, 3, D, seed=9)
    torch.testing.assert_close(attn(x)[:, :5], attn(changed)[:, :5])
    mask = torch.tensor([[True] * 5 + [False] * 3])
    torch.testing.assert_close(attn(changed, key_mask=mask)[:, :5], attn(x[:, :5]))


def test_options_get_gradients():
    attn = make(causal=True, **OPTIONS)
    attn(inputs(2, 7, D), key_mask=MASK).sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in attn.parameters())
