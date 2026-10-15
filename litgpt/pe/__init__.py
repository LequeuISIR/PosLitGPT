# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from __future__ import annotations

from typing import Any

from litgpt.pe.base import BasePositionalEncoding
from litgpt.pe.multirope import MultiRoPE
from litgpt.pe.rope import (
    StandardRoPE,
    apply_rope,
    apply_rope_interleave,
    batched_index_select,
    build_rope_cache,
    yarn_get_mscale,
)


def build_positional_encoding(config: Any) -> BasePositionalEncoding:
    """Factory to instantiate the appropriate positional encoding module based on config."""
    pe_type = getattr(config, "pe_type", "standard_rope")
    if pe_type in ("standard_rope", "rope", "default", None):
        return StandardRoPE(config)
    if pe_type in ("multirope", "multi_rope"):
        return MultiRoPE(config)
    raise ValueError(f"Unknown positional encoding type: {pe_type}")


__all__ = [
    "BasePositionalEncoding",
    "StandardRoPE",
    "MultiRoPE",
    "build_positional_encoding",
    "build_rope_cache",
    "apply_rope",
    "apply_rope_interleave",
    "yarn_get_mscale",
    "batched_index_select",
]
