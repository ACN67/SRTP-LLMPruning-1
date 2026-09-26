"""SLEB block removal following the pinned official executable baseline.

Search, loss, barriers, tie handling, and calibration semantics follow the
MIT-licensed official repository at commit
d07129af60520e751087b8abb04a268a3c7ec861. Native ModuleList replacement is
the Qwen3/Granite compatibility adaptation used instead of legacy wrappers.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterator

from src.models.base import BaseModelAdapter

from .base import BasePruner, PruningRequest
from .sleb_calibration import SLEBCalibrationConfig, SLEBCalibrationContext


@dataclass(frozen=True)
class SLEBSearchResult:
    removal_order_original_indices: tuple[int, ...]
    retained_original_indices: tuple[int, ...]
    candidate_evaluation_count: int
    search_trace: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class SLEBPruningSummary:
    pruner: str
    implementation_version: str
    pruning_type: str
    scope: str
    target_sparsity_ratio: float
    ratio_to_count_policy: str
    original_block_count: int
    requested_remove_count: int
    removed_block_count: int
    achieved_block_sparsity: float
    removal_order_original_indices: tuple[int, ...]
    retained_original_indices: tuple[int, ...]
    early_barrier: int
    latter_barrier: int
    selection_metric: str
    selection_semantics: str
    candidate_tie_rule: str
    calibration_source: str
    calibration_source_rows: int
    calibration_seed: int
    loss_sequence_length: int
    calibration_sampling_semantics: str
    separator: str
    calibration_token_stream_length: int
    calibration_tokenizer_class: str | None
    calibration_tokenizer_is_fast: bool | None
    scored_full_chunks: int
    dropped_tail_tokens: int
    candidate_evaluation_count: int
    search_trace: tuple[dict[str, Any], ...]
    retraining: bool
    gradients: bool
    hessian: bool
    weight_masking: bool
    weight_update: bool
    original_parameter_count: int
    final_parameter_count: int
    parameter_reduction_ratio: float
    post_pruning_num_hidden_layers: int
    post_pruning_actual_block_count: int

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["removal_order_original_indices"] = list(
            self.removal_order_original_indices
        )
        result["retained_original_indices"] = list(self.retained_original_indices)
        result["search_trace"] = list(self.search_trace)
        return result


def ratio_to_remove_count(original_block_count: int, sparsity: float) -> int:
    """Map the project's ratio API to the official integer core input."""

    return math.ceil(original_block_count * sparsity)


def sleb_get_loss(
    model: Any,
    input_ids: Any,
    *,
    sequence_length: int,
    device: Any = None,
    batch_size: int = 1,
) -> float:
    """Compute the pinned official chunked, scaled CrossEntropyLoss sum."""

    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError("SLEB scoring requires PyTorch") from error
    if sequence_length <= 0:
        raise ValueError("SLEB loss sequence length must be positive")
    if batch_size <= 0:
        raise ValueError("SLEB loss batch size must be positive")
    nsamples = input_ids.numel() // sequence_length
    if nsamples == 0:
        raise ValueError(
            "SLEB calibration token stream is shorter than one full loss sequence"
        )

    losses = []
    with torch.no_grad():
        for i in range(0, nsamples, batch_size):
            j = min(i + batch_size, nsamples)
            inputs = input_ids[:, i * sequence_length : j * sequence_length].to(device)
            inputs = inputs.reshape(j - i, sequence_length)
            lm_logits = model(inputs, use_cache=False).logits
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:]
            loss = nn.CrossEntropyLoss()(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )
            losses.append(loss.float() * sequence_length * (j - i))
    return torch.stack(losses).sum().item()


@contextmanager
def temporary_block_removal(
    model: Any,
    adapter: BaseModelAdapter,
    position: int,
) -> Iterator[None]:
    """Temporarily remove one current block and restore the exact container."""

    original_container = adapter.get_blocks(model)
    current_blocks = list(original_container)
    if not 0 <= position < len(current_blocks):
        raise IndexError(f"Block position out of range: {position}")
    adapter.replace_blocks(
        model,
        current_blocks[:position] + current_blocks[position + 1 :],
    )
    try:
        yield
    finally:
        adapter.replace_blocks(model, original_container)


def greedy_block_search(
    model: Any,
    adapter: BaseModelAdapter,
    requested_remove_count: int,
    scorer: Callable[[Any], float],
    *,
    early_barrier: int = 1,
    latter_barrier: int = 1,
) -> SLEBSearchResult:
    """Run official-order greedy removal with strict-less tie handling."""

    original_block_count = adapter.get_num_blocks(model)
    if requested_remove_count < 0:
        raise ValueError("SLEB requested remove count must be non-negative")
    if early_barrier < 0 or latter_barrier < 0:
        raise ValueError("SLEB barriers must be non-negative")
    capacity = original_block_count - early_barrier - latter_barrier
    if requested_remove_count > capacity:
        raise ValueError(
            f"SLEB requested_remove_count={requested_remove_count} exceeds "
            f"barrier-constrained capacity={max(capacity, 0)} for "
            f"original_blocks={original_block_count}, early_barrier={early_barrier}, "
            f"latter_barrier={latter_barrier}"
        )

    alive = list(range(original_block_count))
    removal_order: list[int] = []
    trace: list[dict[str, Any]] = []
    candidate_evaluations = 0
    for round_index in range(requested_remove_count):
        alive_before = list(alive)
        positions = range(early_barrier, len(alive) - latter_barrier)
        eligible_original_indices = [alive[position] for position in positions]
        best_score = float("inf")
        best_position = -1
        candidate_scores: list[dict[str, Any]] = []
        for position in range(early_barrier, len(alive) - latter_barrier):
            original_index = alive[position]
            with temporary_block_removal(model, adapter, position):
                score = float(scorer(model))
            candidate_evaluations += 1
            candidate_scores.append(
                {
                    "current_position": position,
                    "original_index": original_index,
                    "score": score,
                }
            )
            if score < best_score:
                best_score = score
                best_position = position

        selected_original_index = alive[best_position]
        current_blocks = list(adapter.get_blocks(model))
        adapter.replace_blocks(
            model,
            current_blocks[:best_position] + current_blocks[best_position + 1 :],
        )
        removal_order.append(selected_original_index)
        del alive[best_position]
        trace.append(
            {
                "round_index": round_index,
                "alive_original_indices_before": alive_before,
                "eligible_original_indices": eligible_original_indices,
                "candidate_scores": candidate_scores,
                "selected_original_index": selected_original_index,
                "selected_score": best_score,
            }
        )

    return SLEBSearchResult(
        removal_order_original_indices=tuple(removal_order),
        retained_original_indices=tuple(alive),
        candidate_evaluation_count=candidate_evaluations,
        search_trace=tuple(trace),
    )


class SLEBPruner(BasePruner):
    method_id = "sleb"
    implementation_version = "1.0"

    def __init__(
        self,
        *,
        early_barrier: int = 1,
        latter_barrier: int = 1,
        calibration_config: SLEBCalibrationConfig | None = None,
    ) -> None:
        if early_barrier < 0 or latter_barrier < 0:
            raise ValueError("SLEB barriers must be non-negative")
        self.early_barrier = early_barrier
        self.latter_barrier = latter_barrier
        self.calibration_config = calibration_config or SLEBCalibrationConfig()

    def prune(
        self,
        model: Any,
        adapter: BaseModelAdapter,
        request: PruningRequest,
        context: SLEBCalibrationContext | None = None,
    ) -> SLEBPruningSummary:
        if request.method != self.method_id:
            raise ValueError(
                f"Request method {request.method!r} does not match {self.method_id!r}"
            )
        original_block_count = adapter.get_num_blocks(model)
        requested_remove_count = ratio_to_remove_count(
            original_block_count, request.sparsity
        )
        capacity = original_block_count - self.early_barrier - self.latter_barrier
        if requested_remove_count > capacity:
            raise ValueError(
                f"SLEB requested_remove_count={requested_remove_count} exceeds "
                f"barrier-constrained capacity={max(capacity, 0)}"
            )
        if requested_remove_count and context is None:
            raise ValueError("SLEB pruning requires WikiText-2 calibration context")

        config = context.config if context is not None else self.calibration_config
        original_parameter_count = sum(parameter.numel() for parameter in model.parameters())
        metadata = adapter.capture_block_removal_metadata(model)
        previous_use_cache = getattr(model.config, "use_cache", None)
        model.eval()

        if requested_remove_count == 0:
            search = SLEBSearchResult(
                removal_order_original_indices=(),
                retained_original_indices=tuple(range(original_block_count)),
                candidate_evaluation_count=0,
                search_trace=(),
            )
            token_stream_length = 0
        else:
            token_stream_length = context.token_stream_length
            embedding = adapter.get_embedding(model)
            device = getattr(getattr(embedding, "weight", None), "device", None)
            try:
                import torch
            except ImportError as error:
                raise RuntimeError("SLEB scoring requires PyTorch") from error

            def scorer(current_model: Any) -> float:
                score = sleb_get_loss(
                    current_model,
                    context.input_ids,
                    sequence_length=config.sequence_length,
                    device=device,
                    batch_size=1,
                )
                torch.cuda.empty_cache()
                return score

            try:
                if previous_use_cache is not None:
                    model.config.use_cache = False
                search = greedy_block_search(
                    model,
                    adapter,
                    requested_remove_count,
                    scorer,
                    early_barrier=self.early_barrier,
                    latter_barrier=self.latter_barrier,
                )
            finally:
                if previous_use_cache is not None:
                    model.config.use_cache = previous_use_cache

            adapter.finalize_block_removal(
                model,
                search.retained_original_indices,
                metadata,
            )

        final_parameter_count = sum(parameter.numel() for parameter in model.parameters())
        final_block_count = adapter.get_num_blocks(model)
        scored_full_chunks = token_stream_length // config.sequence_length
        dropped_tail_tokens = token_stream_length % config.sequence_length
        return SLEBPruningSummary(
            pruner=self.method_id,
            implementation_version=self.implementation_version,
            pruning_type="block_structured",
            scope="transformer_decoder_blocks",
            target_sparsity_ratio=request.sparsity,
            ratio_to_count_policy="project_interface_ceil",
            original_block_count=original_block_count,
            requested_remove_count=requested_remove_count,
            removed_block_count=len(search.removal_order_original_indices),
            achieved_block_sparsity=(
                len(search.removal_order_original_indices) / original_block_count
            ),
            removal_order_original_indices=search.removal_order_original_indices,
            retained_original_indices=search.retained_original_indices,
            early_barrier=self.early_barrier,
            latter_barrier=self.latter_barrier,
            selection_metric="official_repo_get_loss",
            selection_semantics="chunked_scaled_cross_entropy_sum",
            candidate_tie_rule="strict_less_first_candidate",
            calibration_source=config.source,
            calibration_source_rows=config.source_rows,
            calibration_seed=config.seed,
            loss_sequence_length=config.sequence_length,
            calibration_sampling_semantics=config.sampling_semantics,
            separator=config.separator,
            calibration_token_stream_length=token_stream_length,
            calibration_tokenizer_class=(
                context.tokenizer_class if context is not None else None
            ),
            calibration_tokenizer_is_fast=(
                context.tokenizer_is_fast if context is not None else None
            ),
            scored_full_chunks=scored_full_chunks,
            dropped_tail_tokens=dropped_tail_tokens,
            candidate_evaluation_count=search.candidate_evaluation_count,
            search_trace=search.search_trace,
            retraining=False,
            gradients=False,
            hessian=False,
            weight_masking=False,
            weight_update=False,
            original_parameter_count=original_parameter_count,
            final_parameter_count=final_parameter_count,
            parameter_reduction_ratio=(
                1 - final_parameter_count / original_parameter_count
            ),
            post_pruning_num_hidden_layers=model.config.num_hidden_layers,
            post_pruning_actual_block_count=final_block_count,
        )
