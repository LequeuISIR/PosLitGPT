# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BasePositionalEncoding(nn.Module, ABC):
    """Abstract base class for modular positional encodings in LitGPT."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def reset_parameters(self, max_seq_length: int, device: torch.device | None = None) -> None:
        """Precomputes and allocates non-persistent buffers for max_seq_length on device."""
        pass

    def apply_to_embeddings(self, x: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        """Hook for additive token or segment embeddings before transformer blocks."""
        return x

    @abstractmethod
    def get_cos_sin(
        self,
        rope_pos: torch.Tensor | None,
        seq_len: int,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (cos, sin) tensors for the given rope_pos or sequential prefill slice."""
        pass

    @abstractmethod
    def apply_rotary(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies rotary embeddings to query and key head slices."""
        pass

    def get_attention_bias_or_mask(
        self,
        cache_pos: torch.Tensor | None = None,
        rope_pos: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Hook for relative biases (ALiBi) or document-isolation attention masks."""
        return mask
