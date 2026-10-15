# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from __future__ import annotations

from typing import Any

import torch

from litgpt.pe.rope import StandardRoPE, apply_rope, apply_rope_interleave, batched_index_select


class MultiRoPE(StandardRoPE):
    """Multi-coordinate Rotary Position Embedding (MultiRoPE) for LitGPT.

    Applies rotary embeddings across L independent position coordinates into L disjoint subspaces
    of head_size. Each coordinate occupies a subspace of size `config.rope_n_elem` and shares the
    exact same frequency spectrum. Any remaining head dimensions pass through unrotated.

    Dynamic L inference: L is dynamically inferred at runtime from the final dimension of pe_pos.
    Budget constraint: If L * rope_n_elem > head_size (or rotary_percentage * L > 1.0), raises ValueError.
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)

    def get_cos_sin(
        self,
        pe_pos: torch.Tensor | None,
        seq_len: int,
        device: torch.device | None = None,
        batch_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolves cos and sin for forward steps given multi-coordinate pe_pos or default prefill slicing.

        Args:
            pe_pos: Position tensor of shape:
                - None: sequential indices 0..seq_len-1 (L=1).
                - (T,): 1D sequential positions (L=1).
                - (T, L): sequence length T with L position coordinates (B=1).
                - (B, T): batch size B with sequence length T (L=1).
                - (B, T, L): batch size B, sequence length T with L position coordinates.
            seq_len: Sequence length T.
            device: Target device.
            batch_size: Optional batch size for disambiguating (T, T) shapes.

        Returns:
            Tuple of (cos, sin) tensors:
                - If L == 1: shape (B, T, rope_n_elem)
                - If L > 1: shape (B, L, T, rope_n_elem)
        """
        rope_n_elem = self.config.rope_n_elem
        head_size = self.config.head_size

        if pe_pos is None:
            # Standard sequential prefill (L=1)
            cos = self.cos[:seq_len].unsqueeze(0)
            sin = self.sin[:seq_len].unsqueeze(0)
            return cos, sin

        # Canonicalize pe_pos to (B, T, L)
        if pe_pos.dim() == 1:
            if pe_pos.shape[0] != seq_len:
                raise ValueError(f"pe_pos.shape[0] = {pe_pos.shape[0]} != {seq_len}, must match seq_len")
            # Shape (1, T, 1)
            pe_pos_canonical = pe_pos.unsqueeze(0).unsqueeze(-1)
        elif pe_pos.dim() == 2:
            # Ambiguity resolution between (B, T) [L=1] and (T, L) [B=1]
            if pe_pos.shape[1] == seq_len and pe_pos.shape[0] != seq_len:
                # Shape (B, T) -> (B, T, 1) with L=1
                pe_pos_canonical = pe_pos.unsqueeze(-1)
            elif pe_pos.shape[0] == seq_len:
                # Shape (T, L) -> (1, T, L)
                # Note: if shape is (T, T), check if batch_size indicates (B, T)
                if (
                    pe_pos.shape[1] == seq_len
                    and batch_size is not None
                    and batch_size == pe_pos.shape[0]
                    and batch_size != 1
                ):
                    pe_pos_canonical = pe_pos.unsqueeze(-1)
                else:
                    pe_pos_canonical = pe_pos.unsqueeze(0)
            elif pe_pos.shape[1] == seq_len:
                pe_pos_canonical = pe_pos.unsqueeze(-1)
            else:
                raise ValueError(
                    f"pe_pos shape {pe_pos.shape} is incompatible with seq_len={seq_len}. "
                    "Expected (T,), (T, L), (B, T), or (B, T, L)."
                )
        elif pe_pos.dim() == 3:
            if pe_pos.shape[1] != seq_len:
                raise ValueError(f"pe_pos.shape[1] = {pe_pos.shape[1]} != {seq_len}, must match seq_len")
            pe_pos_canonical = pe_pos
        else:
            raise ValueError(f"pe_pos must have at most 3 dimensions (B, T, L), got shape {pe_pos.shape}")

        B, T, L = pe_pos_canonical.shape
        if L < 1:
            raise ValueError(f"Number of position coordinates L must be >= 1, got {L}")

        # Budget constraint check
        if L * rope_n_elem > head_size:
            raise ValueError(
                f"MultiRoPE budget exceeded: L={L} coordinates with subspace dimension "
                f"rope_n_elem={rope_n_elem} requires {L * rope_n_elem} channels, "
                f"which exceeds head_size={head_size} (rotary_percentage * L = "
                f"{(rope_n_elem * L) / head_size:.2f} > 1.0)."
            )

        if L == 1:
            # Fast path: squeeze L dimension to produce (B, T, rope_n_elem)
            pe_pos_1d = pe_pos_canonical.squeeze(-1)
            cos = batched_index_select(self.cos, 0, pe_pos_1d)
            sin = batched_index_select(self.sin, 0, pe_pos_1d)
            return cos, sin

        # Multi-coordinate path: index cos/sin for each coordinate l
        cos_slices = []
        sin_slices = []
        for l in range(L):
            pos_l = pe_pos_canonical[:, :, l]  # (B, T)
            cos_l = batched_index_select(self.cos, 0, pos_l)  # (B, T, rope_n_elem)
            sin_l = batched_index_select(self.sin, 0, pos_l)  # (B, T, rope_n_elem)
            cos_slices.append(cos_l)
            sin_slices.append(sin_l)

        cos = torch.stack(cos_slices, dim=1)  # (B, L, T, rope_n_elem)
        sin = torch.stack(sin_slices, dim=1)  # (B, L, T, rope_n_elem)
        return cos, sin

    def apply_rotary(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies rotary embeddings to query and key head slices.

        If cos has 3 dimensions (B, T, d_rope), treats as L=1 standard RoPE.
        If cos has 4 dimensions (B, L, T, d_rope), applies rotation to L consecutive
        disjoint slices of size `config.rope_n_elem`, leaving the remainder unrotated.
        """
        rope_n_elem = self.config.rope_n_elem
        head_size = self.config.head_size

        if cos.dim() == 3:
            # L = 1 fast path
            return super().apply_rotary(q, k, cos, sin)

        if cos.dim() != 4:
            raise ValueError(f"cos must have 3 or 4 dimensions, but got shape {cos.shape}")

        B, L, T, d_rope = cos.shape
        if L * rope_n_elem > head_size:
            raise ValueError(
                f"MultiRoPE budget exceeded in apply_rotary: L={L} * rope_n_elem={rope_n_elem} > head_size={head_size}"
            )

        q_rot_slices = []
        k_rot_slices = []
        for l in range(L):
            start = l * rope_n_elem
            end = (l + 1) * rope_n_elem
            cos_l = cos[:, l]  # (B, T, rope_n_elem)
            sin_l = sin[:, l]  # (B, T, rope_n_elem)
            q_l = q[..., start:end]
            k_l = k[..., start:end]
            if self.config.rope_interleave:
                q_rot_slices.append(apply_rope_interleave(q_l, cos_l, sin_l))
                k_rot_slices.append(apply_rope_interleave(k_l, cos_l, sin_l))
            else:
                q_rot_slices.append(apply_rope(q_l, cos_l, sin_l))
                k_rot_slices.append(apply_rope(k_l, cos_l, sin_l))

        pass_start = L * rope_n_elem
        if pass_start < head_size:
            q_pass = q[..., pass_start:]
            k_pass = k[..., pass_start:]
            q_out = torch.cat((*q_rot_slices, q_pass), dim=-1)
            k_out = torch.cat((*k_rot_slices, k_pass), dim=-1)
        else:
            q_out = torch.cat(q_rot_slices, dim=-1)
            k_out = torch.cat(k_rot_slices, dim=-1)

        return q_out, k_out
