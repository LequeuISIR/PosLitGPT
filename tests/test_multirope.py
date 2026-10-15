# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from __future__ import annotations

import io
from pathlib import Path

import pytest
import torch

from litgpt.api import PosLLM
from litgpt.config import Config
from litgpt.model import GPT
from litgpt.pe.multirope import MultiRoPE
from litgpt.pe.rope import StandardRoPE


@torch.inference_mode()
def test_multirope_equivalence_to_standard_rope_l1():
    """Test 1: Equivalence to StandardRoPE when L=1."""
    config = Config(
        name="test_model",
        n_layer=2,
        n_head=4,
        n_embd=64,  # head_size = 16
        rotary_percentage=0.5,  # rope_n_elem = 8
        pe_type="standard_rope",
    )
    std_rope = StandardRoPE(config)
    multi_rope = MultiRoPE(config)

    max_seq_len = 32
    std_rope.reset_parameters(max_seq_len)
    multi_rope.reset_parameters(max_seq_len)

    torch.manual_seed(42)
    B, nh, T, hs = 2, 4, 8, config.head_size
    q = torch.randn(B, nh, T, hs)
    k = torch.randn(B, nh, T, hs)

    # 1a. pe_pos is None (sequential slicing)
    std_cos, std_sin = std_rope.get_cos_sin(None, seq_len=T)
    multi_cos, multi_sin = multi_rope.get_cos_sin(None, seq_len=T)
    torch.testing.assert_close(std_cos, multi_cos)
    torch.testing.assert_close(std_sin, multi_sin)

    std_q, std_k = std_rope.apply_rotary(q, k, std_cos, std_sin)
    multi_q, multi_k = multi_rope.apply_rotary(q, k, multi_cos, multi_sin)
    torch.testing.assert_close(std_q, multi_q)
    torch.testing.assert_close(std_k, multi_k)

    # 1b. pe_pos is 1D (T,)
    pe_pos_1d = torch.randint(0, max_seq_len, (T,))
    std_cos, std_sin = std_rope.get_cos_sin(pe_pos_1d, seq_len=T)
    multi_cos, multi_sin = multi_rope.get_cos_sin(pe_pos_1d, seq_len=T)
    torch.testing.assert_close(std_cos, multi_cos)
    torch.testing.assert_close(std_sin, multi_sin)

    std_q, std_k = std_rope.apply_rotary(q, k, std_cos, std_sin)
    multi_q, multi_k = multi_rope.apply_rotary(q, k, multi_cos, multi_sin)
    torch.testing.assert_close(std_q, multi_q)
    torch.testing.assert_close(std_k, multi_k)

    # 1c. pe_pos is 2D (T, 1)
    pe_pos_2d = pe_pos_1d.unsqueeze(-1)
    std_cos, std_sin = std_rope.get_cos_sin(pe_pos_2d, seq_len=T)
    multi_cos, multi_sin = multi_rope.get_cos_sin(pe_pos_2d, seq_len=T)
    torch.testing.assert_close(std_cos, multi_cos)
    torch.testing.assert_close(std_sin, multi_sin)

    std_q, std_k = std_rope.apply_rotary(q, k, std_cos, std_sin)
    multi_q, multi_k = multi_rope.apply_rotary(q, k, multi_cos, multi_sin)
    torch.testing.assert_close(std_q, multi_q)
    torch.testing.assert_close(std_k, multi_k)

    # 1d. pe_pos is 3D (B, T, 1)
    pe_pos_3d = torch.randint(0, max_seq_len, (B, T, 1))
    std_cos, std_sin = std_rope.get_cos_sin(pe_pos_3d, seq_len=T)
    multi_cos, multi_sin = multi_rope.get_cos_sin(pe_pos_3d, seq_len=T)
    torch.testing.assert_close(std_cos, multi_cos)
    torch.testing.assert_close(std_sin, multi_sin)

    std_q, std_k = std_rope.apply_rotary(q, k, std_cos, std_sin)
    multi_q, multi_k = multi_rope.apply_rotary(q, k, multi_cos, multi_sin)
    torch.testing.assert_close(std_q, multi_q)
    torch.testing.assert_close(std_k, multi_k)


@torch.inference_mode()
def test_multirope_end_to_end_gpt_equivalence_l1():
    """End-to-end model forward equivalence between StandardRoPE and MultiRoPE at L=1."""
    config_std = Config(
        name="test_model",
        n_layer=2,
        n_head=4,
        n_embd=64,
        vocab_size=100,
        padded_vocab_size=128,
        rotary_percentage=0.5,
        pe_type="standard_rope",
    )
    config_multi = Config(
        name="test_model",
        n_layer=2,
        n_head=4,
        n_embd=64,
        vocab_size=100,
        padded_vocab_size=128,
        rotary_percentage=0.5,
        pe_type="multirope",
    )

    torch.manual_seed(1234)
    model_std = GPT(config_std)
    model_multi = GPT(config_multi)

    # Copy exact weights
    model_multi.load_state_dict(model_std.state_dict())

    idx = torch.randint(0, 100, (2, 8))
    pe_pos = torch.arange(8)

    out_std = model_std(idx, pe_pos=pe_pos)
    out_multi = model_multi(idx, pe_pos=pe_pos)
    torch.testing.assert_close(out_std, out_multi)


@torch.inference_mode()
def test_multirope_subspace_isolation_and_orthogonality():
    """Test 2: Subspace isolation & orthogonality.

    Perturbing coordinate 0 must strictly modify subspace 0 while leaving subspace 1
    and the unrotated tail bit-for-bit identical.
    """
    config = Config(
        name="test_model",
        n_layer=1,
        n_head=2,
        n_embd=128,  # head_size = 64
        rotary_percentage=0.25,  # rope_n_elem = 16
        pe_type="multirope",
    )
    multi_rope = MultiRoPE(config)
    multi_rope.reset_parameters(max_seq_length=64)

    # Subspaces:
    # Subspace 0: [0:16]
    # Subspace 1: [16:32]
    # Unrotated:  [32:64]
    d_rope = config.rope_n_elem  # 16
    hs = config.head_size  # 64

    torch.manual_seed(999)
    B, nh, T = 1, 2, 4
    q = torch.randn(B, nh, T, hs)
    k = torch.randn(B, nh, T, hs)

    # Base positions: L=2
    # Coord 0: [2, 3, 4, 5], Coord 1: [10, 11, 12, 13]
    pos_base = torch.tensor([[[2, 10], [3, 11], [4, 12], [5, 13]]])  # (1, 4, 2)
    cos_base, sin_base = multi_rope.get_cos_sin(pos_base, seq_len=T)
    q_base, k_base = multi_rope.apply_rotary(q, k, cos_base, sin_base)

    # Perturb ONLY Coordinate 0: [20, 21, 22, 23], Coord 1 remains [10, 11, 12, 13]
    pos_perturb_0 = torch.tensor([[[20, 10], [21, 11], [22, 12], [23, 13]]])
    cos_p0, sin_p0 = multi_rope.get_cos_sin(pos_perturb_0, seq_len=T)
    q_p0, k_p0 = multi_rope.apply_rotary(q, k, cos_p0, sin_p0)

    # Subspace 0 must differ
    assert not torch.equal(q_base[..., :d_rope], q_p0[..., :d_rope])
    assert not torch.equal(k_base[..., :d_rope], k_p0[..., :d_rope])

    # Subspace 1 must be strictly IDENTICAL
    torch.testing.assert_close(q_base[..., d_rope : 2 * d_rope], q_p0[..., d_rope : 2 * d_rope])
    torch.testing.assert_close(k_base[..., d_rope : 2 * d_rope], k_p0[..., d_rope : 2 * d_rope])

    # Unrotated tail must be strictly IDENTICAL
    torch.testing.assert_close(q_base[..., 2 * d_rope :], q_p0[..., 2 * d_rope :])
    torch.testing.assert_close(k_base[..., 2 * d_rope :], k_p0[..., 2 * d_rope :])

    # Perturb ONLY Coordinate 1: Coord 0 remains [2, 3, 4, 5], Coord 1 becomes [40, 41, 42, 43]
    pos_perturb_1 = torch.tensor([[[2, 40], [3, 41], [4, 42], [5, 43]]])
    cos_p1, sin_p1 = multi_rope.get_cos_sin(pos_perturb_1, seq_len=T)
    q_p1, k_p1 = multi_rope.apply_rotary(q, k, cos_p1, sin_p1)

    # Subspace 0 must be strictly IDENTICAL
    torch.testing.assert_close(q_base[..., :d_rope], q_p1[..., :d_rope])
    torch.testing.assert_close(k_base[..., :d_rope], k_p1[..., :d_rope])

    # Subspace 1 must differ
    assert not torch.equal(q_base[..., d_rope : 2 * d_rope], q_p1[..., d_rope : 2 * d_rope])
    assert not torch.equal(k_base[..., d_rope : 2 * d_rope], k_p1[..., d_rope : 2 * d_rope])

    # Unrotated tail must be strictly IDENTICAL
    torch.testing.assert_close(q_base[..., 2 * d_rope :], q_p1[..., 2 * d_rope :])
    torch.testing.assert_close(k_base[..., 2 * d_rope :], k_p1[..., 2 * d_rope :])


@torch.inference_mode()
def test_multirope_identity_at_zero_position():
    """Test 3: Identity rotation when coordinate is zero (p_l = 0)."""
    config = Config(
        name="test_model",
        n_layer=1,
        n_head=2,
        n_embd=64,  # head_size = 32
        rotary_percentage=0.25,  # rope_n_elem = 8
        pe_type="multirope",
    )
    multi_rope = MultiRoPE(config)
    multi_rope.reset_parameters(max_seq_length=32)

    T = 2
    hs = config.head_size
    q = torch.randn(1, 2, T, hs)
    k = torch.randn(1, 2, T, hs)

    # Both coordinates are 0
    pos_zero = torch.zeros((1, T, 2), dtype=torch.long)
    cos, sin = multi_rope.get_cos_sin(pos_zero, seq_len=T)
    q_rot, k_rot = multi_rope.apply_rotary(q, k, cos, sin)

    # When pos=0, R(0) = I, so q_rot and k_rot must equal q and k exactly
    torch.testing.assert_close(q, q_rot)
    torch.testing.assert_close(k, k_rot)


@torch.inference_mode()
def test_multirope_relative_shift_invariance():
    """Test 4: Rotary relative shift invariance ((p_{j,l} - p_{i,l}) preserved)."""
    config = Config(
        name="test_model",
        n_layer=1,
        n_head=1,
        n_embd=32,  # head_size = 32
        rotary_percentage=0.25,  # rope_n_elem = 8
        pe_type="multirope",
    )
    multi_rope = MultiRoPE(config)
    multi_rope.reset_parameters(max_seq_length=64)

    T = 2
    hs = config.head_size
    torch.manual_seed(77)
    q = torch.randn(1, 1, T, hs)
    k = torch.randn(1, 1, T, hs)

    # Token 0 pos: (2, 5), Token 1 pos: (6, 9)
    # Differences: delta_0 = 4, delta_1 = 4
    pos_a = torch.tensor([[[2, 5], [6, 9]]])
    cos_a, sin_a = multi_rope.get_cos_sin(pos_a, seq_len=T)
    q_a, k_a = multi_rope.apply_rotary(q, k, cos_a, sin_a)

    # Compute dot-product attention score between token 1 query and token 0 key
    score_a = torch.matmul(q_a[:, :, 1:2, :], k_a[:, :, 0:1, :].transpose(-1, -2))

    # Shift both tokens by (+10 on coord 0, +15 on coord 1)
    # Token 0 pos: (12, 20), Token 1 pos: (16, 24)
    # Differences remain delta_0 = 4, delta_1 = 4
    pos_b = torch.tensor([[[12, 20], [16, 24]]])
    cos_b, sin_b = multi_rope.get_cos_sin(pos_b, seq_len=T)
    q_b, k_b = multi_rope.apply_rotary(q, k, cos_b, sin_b)

    score_b = torch.matmul(q_b[:, :, 1:2, :], k_b[:, :, 0:1, :].transpose(-1, -2))

    # Attention score must be invariant under uniform coordinate shifts
    torch.testing.assert_close(score_a, score_b)


@torch.inference_mode()
def test_multirope_kv_cache_prefill_and_step_parity():
    """Test 5: KV cache prefill vs autoregressive step decoding parity with L=2."""
    config = Config(
        name="test_model",
        n_layer=2,
        n_head=2,
        n_embd=32,  # head_size = 16
        vocab_size=50,
        padded_vocab_size=64,
        rotary_percentage=0.25,  # rope_n_elem = 4 (4 * 2 = 8 <= 16)
        pe_type="multirope",
    )
    torch.manual_seed(42)
    model = GPT(config)
    model.max_seq_length = 32

    # Sequence of 4 tokens with 2D positions: (doc_id, intra_doc_pos)
    # Token 0: (0, 0), Token 1: (0, 1), Token 2: (1, 0), Token 3: (1, 1)
    prompt = torch.tensor([[10, 20, 30, 40]])  # (1, 4)
    pe_pos = torch.tensor([[[0, 0], [0, 1], [1, 0], [1, 1]]])  # (1, 4, 2)

    # 1. Full sequence prefill
    prefill_logits = model(prompt, pe_pos=pe_pos)

    # 2. Step-by-step autoregressive decoding with KV cache
    model.set_kv_cache(batch_size=1)
    step_logits = []
    for step in range(4):
        token_step = prompt[:, step : step + 1]  # (1, 1)
        pe_step = pe_pos[:, step : step + 1]  # (1, 1, 2)
        cache_step = torch.tensor([step])
        out_step = model(token_step, cache_pos=cache_step, pe_pos=pe_step)
        step_logits.append(out_step)

    all_step_logits = torch.cat(step_logits, dim=1)
    torch.testing.assert_close(prefill_logits, all_step_logits)


def test_multirope_budget_constraint_error():
    """Test 6: Budget constraint validation (L * rope_n_elem > head_size raises ValueError)."""
    # head_size = 32, rotary_percentage = 0.5 -> rope_n_elem = 16
    config = Config(
        name="test_model",
        n_layer=1,
        n_head=2,
        n_embd=64,  # head_size = 32
        rotary_percentage=0.5,  # rope_n_elem = 16
        pe_type="multirope",
    )
    multi_rope = MultiRoPE(config)
    multi_rope.reset_parameters(max_seq_length=32)

    # L=2 is allowed: 2 * 16 = 32 <= 32
    pos_l2 = torch.zeros((1, 4, 2), dtype=torch.long)
    cos, sin = multi_rope.get_cos_sin(pos_l2, seq_len=4)
    assert cos.shape == (1, 2, 4, 16)

    # L=3 exceeds budget: 3 * 16 = 48 > 32 -> MUST raise ValueError
    pos_l3 = torch.zeros((1, 4, 3), dtype=torch.long)
    with pytest.raises(ValueError, match="MultiRoPE budget exceeded"):
        multi_rope.get_cos_sin(pos_l3, seq_len=4)


@torch.inference_mode()
def test_multirope_state_dict_cleanliness_and_checkpoint_loading():
    """Test 7: Verification that MultiRoPE introduces 0 state_dict parameters/buffers."""
    config_std = Config(
        name="test_model",
        n_layer=2,
        n_head=2,
        n_embd=32,
        vocab_size=50,
        padded_vocab_size=64,
        rotary_percentage=0.25,
        pe_type="standard_rope",
    )
    config_multi = Config(
        name="test_model",
        n_layer=2,
        n_head=2,
        n_embd=32,
        vocab_size=50,
        padded_vocab_size=64,
        rotary_percentage=0.25,
        pe_type="multirope",
    )

    model_std = GPT(config_std)
    model_multi = GPT(config_multi)

    # Verify pe module has 0 state_dict keys
    assert len(model_std.pe.state_dict()) == 0
    assert len(model_multi.pe.state_dict()) == 0

    # Save std checkpoint
    buf = io.BytesIO()
    torch.save(model_std.state_dict(), buf)
    buf.seek(0)

    # Load with strict=True into multirope model
    state_dict = torch.load(buf, weights_only=True)
    incompatible_keys = model_multi.load_state_dict(state_dict, strict=True)
    assert len(incompatible_keys.missing_keys) == 0
    assert len(incompatible_keys.unexpected_keys) == 0


def test_posllm_load_with_config_overrides():
    """Test 8: PosLLM.load() accepts config_overrides (e.g. pe_type='multirope')."""
    phi_dir = Path("checkpoints/microsoft/phi-2")
    if not phi_dir.exists():
        pytest.skip("checkpoints/microsoft/phi-2 not found")

    # Load with random init and config_overrides
    llm = PosLLM.load(
        model="phi-2",
        init="random",
        tokenizer_dir=phi_dir,
        distribute="auto",
        config_overrides={"pe_type": "multirope", "rotary_percentage": 0.5},
    )

    assert llm.config.pe_type == "multirope"
    assert llm.config.rotary_percentage == 0.5
    assert llm.config.rope_n_elem == int(0.5 * llm.config.head_size)
    assert isinstance(llm.model.pe, MultiRoPE)


def test_posllm_load_with_invalid_config_override():
    """Test 9: PosLLM.load() raises ValueError on unknown config_overrides keys."""
    phi_dir = Path("checkpoints/microsoft/phi-2")
    tokenizer_dir = phi_dir if phi_dir.exists() else None

    with pytest.raises(ValueError, match="Unknown config attribute: nonexistent_setting_xyz"):
        PosLLM.load(
            model="phi-2",
            init="random",
            tokenizer_dir=tokenizer_dir,
            distribute=None,
            config_overrides={"nonexistent_setting_xyz": True},
        )
