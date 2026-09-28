"""Shared bookkeeping for physical transformer-block removal."""

from __future__ import annotations

import math
from typing import Any, Iterable

from src.models.base import BaseModelAdapter


def ratio_to_block_count(original_block_count: int, sparsity: float) -> int:
    """Map the project ratio API to an explicit block count using ``ceil``."""

    if original_block_count <= 0:
        raise ValueError("Original block count must be positive")
    if not 0 <= sparsity < 1:
        raise ValueError("Block sparsity must be in [0, 1)")
    return math.ceil(original_block_count * sparsity)


def retained_indices(
    original_block_count: int,
    removed_original_indices: Iterable[int],
) -> tuple[int, ...]:
    removed = tuple(int(index) for index in removed_original_indices)
    if len(removed) != len(set(removed)):
        raise ValueError("Removed block indices must be unique")
    if any(index < 0 or index >= original_block_count for index in removed):
        raise IndexError("Removed block index is outside the original model depth")
    removed_set = set(removed)
    return tuple(index for index in range(original_block_count) if index not in removed_set)


def remove_blocks(
    model: Any,
    adapter: BaseModelAdapter,
    removed_original_indices: Iterable[int],
) -> tuple[int, ...]:
    """Physically remove selected original blocks and finalize native metadata."""

    blocks = list(adapter.get_blocks(model))
    keep = retained_indices(len(blocks), removed_original_indices)
    metadata = adapter.capture_block_removal_metadata(model)
    adapter.replace_blocks(model, [blocks[index] for index in keep])
    adapter.finalize_block_removal(model, keep, metadata)
    return keep
