# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from __future__ import annotations

import math
from typing import Any

import torch

from litgpt.pe.base import BasePositionalEncoding


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def build_rope_cache(
    seq_len: int,
    n_elem: int,
    device: torch.device | None = None,
    base: int = 10000,
    condense_ratio: int = 1,
    extra_config: dict | None = None,
    rope_local_base_freq: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Enhanced Transformer with Rotary Position Embedding.

    Args:
        seq_len (int): Sequence length.
        n_elem (int): Number of elements (head dimension).
        device (torch.device, optional): Device for tensor allocations.
        base (int, optional): Base for computing inverse frequencies.
        condense_ratio (int, optional): Ratio to condense the position indices.
        extra_config (dict, optional): Configuration parameters for frequency adjustments (used by Llama 3.1 and 3.2)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Cosine and sine caches for RoPE.
            Shapes are `(seq_len, n_elem)`.
    """

    # Compute the inverse frequencies theta
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))

    # Initialize attention scaling factor (modified for YaRN)
    attention_scaling = 1.0

    if extra_config is not None:
        factor = extra_config["factor"]
        # Check YaRN first (has beta_fast/beta_slow)
        if "beta_fast" in extra_config or "beta_slow" in extra_config:
            # YaRN-style RoPE scaling
            beta_fast = extra_config["beta_fast"]
            beta_slow = extra_config["beta_slow"]
            original_max_seq_len = extra_config["original_max_seq_len"]

            # Calculate attention scaling factor based on mscale and mscale_all_dim
            mscale = extra_config.get("mscale")
            mscale_all_dim = extra_config.get("mscale_all_dim")
            if mscale and mscale_all_dim:
                attention_scaling = yarn_get_mscale(factor, mscale) / yarn_get_mscale(factor, mscale_all_dim)
            elif mscale_all_dim:
                attention_scaling = yarn_get_mscale(factor, mscale_all_dim)
            elif mscale:
                attention_scaling = yarn_get_mscale(factor, mscale)
            # else: attention_scaling remains 1.0

            # Create two frequency sets: extrapolation (unscaled) and interpolation (scaled)
            pos_freqs = base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem)
            theta_extrapolation = 1.0 / pos_freqs
            theta_interpolation = 1.0 / (factor * pos_freqs)

            # Find correction range based on rotation counts
            # Inverse dimension formula to find dimension based on number of rotations
            def find_correction_dim(num_rotations, dim, base_val, max_pos):
                return (dim * math.log(max_pos / (num_rotations * 2 * math.pi))) / (2 * math.log(base_val))

            low_dim = find_correction_dim(beta_fast, n_elem, base, original_max_seq_len)
            high_dim = find_correction_dim(beta_slow, n_elem, base, original_max_seq_len)

            # Apply truncation if specified
            if extra_config.get("truncate", True):
                low_dim = math.floor(low_dim)
                high_dim = math.ceil(high_dim)

            low_dim = max(low_dim, 0)
            high_dim = min(high_dim, n_elem // 2 - 1)

            # Create linear ramp factor for blending
            dim_range = torch.arange(n_elem // 2, device=device, dtype=torch.float32)
            if low_dim == high_dim:
                high_dim += 0.001  # Prevent singularity

            linear_func = (dim_range - low_dim) / (high_dim - low_dim)
            ramp_func = torch.clamp(linear_func, 0.0, 1.0)

            # Blend extrapolation and interpolation frequencies
            # ramp_func = 0 -> use interpolation (scaled), ramp_func = 1 -> use extrapolation (unscaled)
            theta_extrapolation_factor = ramp_func
            theta = (
                theta_interpolation * (1 - theta_extrapolation_factor)
                + theta_extrapolation * theta_extrapolation_factor
            )
        elif "original_max_seq_len" in extra_config:
            # Llama3-style RoPE scaling
            orig_context_len = extra_config["original_max_seq_len"]
            low_freq_factor = extra_config["low_freq_factor"]
            high_freq_factor = extra_config["high_freq_factor"]

            wavelen = 2 * torch.pi / theta
            ratio = orig_context_len / wavelen
            smooth_factor = (ratio - low_freq_factor) / (high_freq_factor - low_freq_factor)
            smooth_factor = torch.clamp(smooth_factor, min=0.0, max=1.0)

            # Compute adjusted_theta without masked indexing
            adjusted_theta = (1 - smooth_factor) * (theta / factor) + smooth_factor * theta
            theta = adjusted_theta
        else:
            # Linear scaling fallback
            theta = theta / factor

    # Create position indices `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device).float() / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)
    # If `n_elem` is odd, the final dimension of `idx_theta` has size
    # `n_elem + 1`, so need to cut something off.
    # Due to a current bug in Hugging Face, in the case `n_elem == 1`, we leave
    # `idx_theta`, `cos`, `sin` as is. Things work out in `apply_rope` due to
    # broadcasting. If we shorten `idx_theta`, unit tests comparing to
    # Hugging Face fail.
    # https://github.com/huggingface/transformers/issues/35233
    if idx_theta.shape[-1] > n_elem > 1:
        idx_theta = idx_theta[..., :n_elem]

    # if rope_local_base_freq is given, have a separate rope value for local embedding
    # For now, we use default RoPE for local embedding
    if rope_local_base_freq is not None:
        local_theta = 1.0 / (rope_local_base_freq ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))
        local_idx_theta = torch.outer(seq_idx, local_theta)
        local_idx_theta = local_idx_theta.repeat(1, 2)
        if local_idx_theta.shape[-1] > n_elem > 1:
            local_idx_theta = local_idx_theta[..., :n_elem]

        idx_theta = torch.stack((idx_theta, local_idx_theta), dim=-1)

    cos = torch.cos(idx_theta) * attention_scaling
    sin = torch.sin(idx_theta) * attention_scaling
    return cos, sin


def batched_index_select(t: torch.Tensor, dim: int, idx: torch.Tensor) -> torch.Tensor:
    """index_select for batched index and unbatched t."""
    if idx.dim() == 1:
        return torch.index_select(t, dim, idx)

    *batch_shape, idx_size = idx.shape
    res = torch.index_select(t, dim, idx.reshape(-1))  # flat index
    res = res.view(*t.shape[:dim], -1, idx_size, *t.shape[dim + 1 :])
    if dim > 0:
        dims = [dim] + list(range(res.dim()))
        del dims[dim + 1]
        res = res.permute(dims)
    res = res.view(*batch_shape, *res.shape[1:])
    return res


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies RoPE transform to `x`."""
    if cos.dim() != 3:
        raise ValueError(f"cos must be three-dimensional, but shape is {cos.shape}")
    if cos.shape != sin.shape:
        raise ValueError(f"cos, sin must have same shape, but cos.shape={cos.shape}, sin.shape={sin.shape}")
    head_size_half = x.size(-1) // 2
    x1 = x[..., :head_size_half]  # (B, ..., T, head_size/2)
    x2 = x[..., head_size_half:]  # (B, ..., T, head_size/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, ..., T, head_size)
    dims_diff = x.dim() - cos.dim()
    if dims_diff > 0:
        new_shape = cos.shape[0:1] + (1,) * dims_diff + cos.shape[1:]
        cos = cos.view(*new_shape)
        sin = sin.view(*new_shape)

    roped = (x * cos) + (rotated * sin)
    return roped.to(dtype=x.dtype)


def apply_rope_interleave(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings with interleaved tensor layout."""
    if cos.dim() != 3:
        raise ValueError(f"cos must be three-dimensional, but shape is {cos.shape}")
    if cos.shape != sin.shape:
        raise ValueError(f"cos, sin must have same shape, but cos.shape={cos.shape}, sin.shape={sin.shape}")

    head_dim = x.size(-1)
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for interleaved RoPE, but got {head_dim}")

    x_even = x[..., 0::2]  # Even indices: 0, 2, 4, ...
    x_odd = x[..., 1::2]  # Odd indices: 1, 3, 5, ...
    rotated = torch.stack((-x_odd, x_even), dim=-1).flatten(-2)

    dims_diff = x.dim() - cos.dim()
    if dims_diff > 0:
        new_shape = cos.shape[0:1] + (1,) * dims_diff + cos.shape[1:]
        cos = cos.view(*new_shape)
        sin = sin.view(*new_shape)

    roped = (x * cos) + (rotated * sin)
    return roped.to(dtype=x.dtype)


class StandardRoPE(BasePositionalEncoding):
    """Standard RoPE (Rotary Position Embedding) implementation for LitGPT."""

    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.register_buffer("cos", None, persistent=False)
        self.register_buffer("sin", None, persistent=False)

    def rope_cache(self, seq_len: int, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculates cos and sin tensors based on model config."""
        if getattr(self.config, "rope_adjustments", None) is None:
            extra_config = None
        else:
            llama3_params = ["low_freq_factor", "high_freq_factor"]
            yarn_params = ["beta_fast", "beta_slow"]

            has_llama3 = any(param in self.config.rope_adjustments for param in llama3_params)
            has_yarn = any(param in self.config.rope_adjustments for param in yarn_params)

            if has_llama3 and has_yarn:
                raise ValueError(
                    "RoPE adjustments cannot contain both Llama3 parameters (low_freq_factor, high_freq_factor) "
                    "and YaRN parameters (beta_fast, beta_slow). These are mutually exclusive."
                )

            if has_llama3:
                adjusted_params_required = ["factor", "low_freq_factor", "high_freq_factor", "original_max_seq_len"]
                params_present = [param in self.config.rope_adjustments for param in adjusted_params_required]
                if all(params_present):
                    extra_config = {name: self.config.rope_adjustments[name] for name in adjusted_params_required}
                else:
                    missing_params = [
                        param for param, present in zip(adjusted_params_required, params_present) if not present
                    ]
                    raise ValueError(
                        f"The following Llama3 RoPE parameters are missing in rope_adjustments: {', '.join(missing_params)}. "
                        "All Llama3 parameters must be specified together."
                    )
            elif has_yarn:
                yarn_required_params = ["factor", "beta_fast", "beta_slow", "original_max_seq_len"]
                params_present = [param in self.config.rope_adjustments for param in yarn_required_params]
                if not all(params_present):
                    missing_params = [
                        param for param, present in zip(yarn_required_params, params_present) if not present
                    ]
                    raise ValueError(
                        f"The following YaRN RoPE parameters are missing in rope_adjustments: {', '.join(missing_params)}. "
                        "All YaRN parameters must be specified together."
                    )
                extra_config = {name: self.config.rope_adjustments[name] for name in yarn_required_params}
                if "mscale" in self.config.rope_adjustments:
                    extra_config["mscale"] = self.config.rope_adjustments["mscale"]
                if "mscale_all_dim" in self.config.rope_adjustments:
                    extra_config["mscale_all_dim"] = self.config.rope_adjustments["mscale_all_dim"]
            else:
                adjusted_params_required = ["factor"]
                extra_config = {name: self.config.rope_adjustments[name] for name in adjusted_params_required}

        return build_rope_cache(
            seq_len=seq_len,
            n_elem=self.config.rope_n_elem,
            device=device,
            condense_ratio=self.config.rope_condense_ratio,
            base=self.config.rope_base,
            extra_config=extra_config,
            rope_local_base_freq=getattr(self.config, "rope_local_base_freq", None),
        )

    def reset_parameters(self, max_seq_length: int, device: torch.device | None = None) -> None:
        """Precomputes and updates cos and sin non-persistent buffers."""
        cos, sin = self.rope_cache(seq_len=max_seq_length, device=device)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def get_cos_sin(
        self,
        pe_pos: torch.Tensor | None,
        seq_len: int,
        device: torch.device | None = None,
        batch_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolves cos and sin for the forward step given pe_pos or default sequential slicing."""
        if pe_pos is not None:
            # Squeeze trailing L=1 dimension if present
            if pe_pos.dim() >= 2 and pe_pos.shape[-1] == 1:
                pe_pos_1d = pe_pos.squeeze(-1)
            else:
                pe_pos_1d = pe_pos

            if pe_pos_1d.dim() > 2:
                raise ValueError(
                    f"pe_pos must have 1 or 2 dimensions (or (..., 1) for L=1), pe_pos.shape = {pe_pos.shape}"
                )
            if pe_pos_1d.shape[-1] != seq_len:
                raise ValueError(f"pe_pos.shape[-1] = {pe_pos_1d.shape[-1]} != {seq_len}, must be the same")

            cos = batched_index_select(self.cos, 0, pe_pos_1d)
            sin = batched_index_select(self.sin, 0, pe_pos_1d)
            if pe_pos_1d.dim() == 1:
                cos = cos.unsqueeze(0)
                sin = sin.unsqueeze(0)
        else:
            cos = self.cos[:seq_len].unsqueeze(0)
            sin = self.sin[:seq_len].unsqueeze(0)

        return cos, sin

    def apply_rotary(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Splits features, applies RoPE to rotary slice, and concatenates back."""
        rope_n_elem = self.config.rope_n_elem

        q_rot = q[..., :rope_n_elem]
        q_pass = q[..., rope_n_elem:]
        k_rot = k[..., :rope_n_elem]
        k_pass = k[..., rope_n_elem:]

        if self.config.rope_interleave:
            q_roped = apply_rope_interleave(q_rot, cos, sin)
            k_roped = apply_rope_interleave(k_rot, cos, sin)
        else:
            q_roped = apply_rope(q_rot, cos, sin)
            k_roped = apply_rope(k_rot, cos, sin)

        q = torch.cat((q_roped, q_pass), dim=-1)
        k = torch.cat((k_roped, k_pass), dim=-1)
        return q, k
