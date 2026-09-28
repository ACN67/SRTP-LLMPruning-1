"""Independent TaBP implementation matching pinned executable semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from src.models.base import BaseModelAdapter

from .base import BasePruner, PruningRequest
from .block_removal import ratio_to_block_count, remove_blocks
from .tabp_calibration import TaBPCalibrationContext

SUPPORTED_TABP_MEASURES = {
    "entropy", "confidence_score", "gap", "key", "cross_entropy",
    "kl_divergence", "js_divergence", "wasserstein_dist", "hellinger_dist",
    "bhattacharyya_dist", "cosine_similarity", "tvd", "energy",
}
SUPPORTED_SSN_MEASURES = SUPPORTED_TABP_MEASURES
MEASURE_DIRECTION = {
    "entropy": -1, "confidence_score": 1, "gap": 1, "key": 1,
    "cross_entropy": -1, "kl_divergence": -1, "js_divergence": -1,
    "wasserstein_dist": -1, "hellinger_dist": -1,
    "bhattacharyya_dist": -1, "cosine_similarity": 1, "tvd": -1,
    "energy": -1,
}


@dataclass(frozen=True)
class TaBPRanking:
    eligible_original_indices: tuple[int, ...]
    scores_by_original_index: Mapping[int, float]
    ranking_original_indices: tuple[int, ...]
    sample_count: int


@dataclass(frozen=True)
class TaBPPruningSummary:
    pruner: str
    implementation_version: str
    upstream_repo: str
    upstream_commit: str
    official_semantics: str
    project_adaptation: str
    pruning_type: str
    scope: str
    target_sparsity_ratio: float
    requested_retention_ratio: float
    ratio_to_count_policy: str
    original_block_count: int
    requested_remove_count: int
    removed_block_count: int
    achieved_block_sparsity: float
    achieved_block_retention: float
    ranking_strategy: str
    mode: str
    measure: str
    ranking_semantics: str
    ranking_passes: int
    candidate_tie_rule: str
    eligible_original_indices: tuple[int, ...]
    scores_by_original_index: Mapping[int, float]
    ranked_block_indices: tuple[int, ...]
    removal_order_original_indices: tuple[int, ...]
    retained_original_indices: tuple[int, ...]
    calibration_source: str
    calibration_revision: str | None
    calibration_task_type: str
    calibration_samples: int
    calibration_seed: int
    calibration_sampling_semantics: str
    calibration_tokenizer_class: str
    calibration_tokenizer_is_fast: bool | None
    lm_head_type: str
    trained_lm_head_support: str
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
        for key in ("eligible_original_indices", "ranked_block_indices", "removal_order_original_indices", "retained_original_indices"):
            result[key] = list(result[key])
        result["scores_by_original_index"] = {str(k): v for k, v in self.scores_by_original_index.items()}
        return result


class _AllowedTokensProcessor:
    def __init__(self, token_ids: Sequence[int]) -> None:
        self.token_ids = tuple(int(value) for value in token_ids)

    def __call__(self, _input_ids: Any, scores: Any) -> Any:
        import torch
        mask = torch.full_like(scores, -float("inf"))
        mask[:, list(self.token_ids)] = 0
        return scores + mask


def _metric_value(measure: str, raw_logits: Any, allowed: Sequence[int], target_probs: Any, key_token_id: int) -> float:
    """Mirror pruning/stats.py, including its indexing behavior."""
    import torch

    if measure not in SUPPORTED_TABP_MEASURES:
        raise ValueError(f"Unsupported TaBP measure: {measure!r}")
    raw = raw_logits.float()
    allowed_list = list(allowed)
    masked = torch.full_like(raw, -float("inf"))
    masked[allowed_list] = raw[allowed_list]
    layer_probs = torch.softmax(masked, dim=-1)[allowed_list]
    target = target_probs.float()
    if measure == "entropy":
        return float((-(layer_probs * torch.log(layer_probs + 1e-9))).sum().item())
    if measure == "confidence_score":
        return float(layer_probs.max().item())
    if measure == "gap":
        values = torch.sort(layer_probs / layer_probs.sum()).values
        return float((values[-1] - values[-2]).item())
    if measure == "key":
        return float(layer_probs[allowed_list.index(key_token_id)].item())
    if measure == "cross_entropy":
        return float((-(target[allowed_list] * torch.log_softmax(raw[allowed_list], dim=-1))).sum().item())
    if measure == "kl_divergence":
        p = target[allowed_list] + 1e-10
        q = layer_probs + 1e-10
        return float((p * torch.log(p / q)).sum().item())

    # The remaining upstream functions index both P and Q by allowed IDs.
    # Q is already restricted, so QA may raise IndexError exactly as upstream.
    p = target[allowed_list]
    q = layer_probs[allowed_list]
    p = p / p.sum()
    q = q / q.sum()
    if measure == "js_divergence":
        midpoint = 0.5 * (p + q)
        return float((0.5 * ((p * torch.log((p + 1e-8) / (midpoint + 1e-8))).sum() + (q * torch.log((q + 1e-8) / (midpoint + 1e-8))).sum())).item())
    if measure == "tvd":
        return float((0.5 * torch.abs(p - q).sum()).item())
    if measure == "hellinger_dist":
        return float(torch.sqrt(0.5 * ((torch.sqrt(p) - torch.sqrt(q)) ** 2).sum()).item())
    if measure == "cosine_similarity":
        return float((torch.dot(p, q) / (torch.linalg.vector_norm(p) * torch.linalg.vector_norm(q))).item())
    if measure == "bhattacharyya_dist":
        return float((-torch.log(torch.sqrt(p * q).sum() + 1e-10)).item())
    if measure == "energy":
        return float(torch.abs(p - q).mean().item())
    # scipy.stats.wasserstein_distance(P, Q) treats probability values as samples.
    return float(torch.abs(torch.sort(p).values - torch.sort(q).values).mean().item())


def _generate_capture(model: Any, adapter: BaseModelAdapter, input_ids: Any, attention_mask: Any | None, start_block: int, allowed: Sequence[int] | None) -> tuple[dict[int, Any], Any, Any]:
    import torch

    captured: dict[int, Any] = {}
    hooks = []
    for index, block in enumerate(adapter.get_blocks(model)):
        if index < start_block:
            continue
        def capture(_module: Any, _inputs: Any, output: Any, *, block_index: int = index) -> None:
            captured[block_index] = adapter.normalize_block_output(output).detach().cpu()
        hooks.append(block.register_forward_hook(capture))
    device = next(model.parameters()).device
    inputs = {"input_ids": input_ids.to(device)}
    if attention_mask is not None:
        inputs["attention_mask"] = attention_mask.to(device)
    processors = [_AllowedTokensProcessor(allowed)] if allowed is not None else None
    pad_token_id = getattr(model.config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(model.config, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = 0
    try:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs, max_new_tokens=1, pad_token_id=pad_token_id,
                return_dict_in_generate=True, output_scores=True,
                logits_processor=processors,
            )
    finally:
        for hook in hooks:
            hook.remove()
    expected = len(adapter.get_blocks(model)) - start_block
    if len(captured) != expected:
        raise RuntimeError(f"TaBP captured {len(captured)} blocks; expected {expected}")
    return captured, outputs.scores[-1].detach(), outputs.sequences[:, -1:]


def _score_pass(model: Any, adapter: BaseModelAdapter, input_ids: Any, attention_mask: Any | None, start: int, measure: str, allowed: Sequence[int] | None, key_token_id: int) -> tuple[dict[int, float], Any]:
    import torch

    captured, target_logits, generated = _generate_capture(model, adapter, input_ids, attention_mask, start, allowed)
    target_probs = torch.softmax(target_logits[0].float(), dim=-1).cpu()
    lm_head = adapter.get_lm_head(model)
    lm_device = next(lm_head.parameters()).device
    values: dict[int, float] = {}
    for index in range(start, adapter.get_num_blocks(model)):
        hidden = captured[index]
        with torch.inference_mode():
            logits = lm_head(hidden.to(lm_device))[:, -1, :][0].detach().cpu()
        token_ids = tuple(range(logits.numel())) if allowed is None else tuple(allowed)
        values[index] = _metric_value(measure, logits, token_ids, target_probs, key_token_id)
    return values, generated.cpu()


def aggregate_ssn_scores(measurements: Sequence[Mapping[int, float]], eligible_original_indices: Sequence[int]) -> dict[int, float]:
    if not measurements:
        raise ValueError("SSN requires at least one calibration sample")
    scores: dict[int, float] = {}
    for index in map(int, eligible_original_indices):
        shifts = []
        for sample in measurements:
            if index not in sample or index - 1 not in sample:
                raise ValueError(f"SSN sample is missing adjacent measurements for block {index}")
            shifts.append(abs(float(sample[index]) - float(sample[index - 1])))
        scores[index] = sum(shifts) / len(shifts)
    return scores


def aggregate_ddf_scores(measurements: Sequence[Mapping[int, float]], block_count: int, measure: str, task_type: str) -> dict[int, float]:
    if not measurements:
        raise ValueError("DDF requires at least one ranking pass")
    if measure not in MEASURE_DIRECTION:
        raise ValueError(f"Unsupported TaBP measure: {measure!r}")
    alpha = MEASURE_DIRECTION[measure]
    counts = np.zeros(block_count, dtype=float)
    for values in measurements:
        observed = tuple(values)
        for index in observed[1:]:
            delta = float(values[index]) - float(values[index - 1])
            if (task_type == "qa" and alpha * delta < 0) or (task_type == "text_generation" and alpha * delta > 0):
                counts[index] += 1
    counts /= len(measurements)
    return {index: float(counts[index]) for index in range(block_count)}


def rank_blocks_ssn(model: Any, adapter: BaseModelAdapter, context: TaBPCalibrationContext, *, mode: str, measure: str, lm_head_type: str = "frozen") -> TaBPRanking:
    if context.config.task_type != "qa":
        raise ValueError("Pinned SSN project path requires QA calibration")
    if lm_head_type != "frozen":
        raise NotImplementedError("trained LM heads require explicit compatible upstream checkpoints")
    block_count = adapter.get_num_blocks(model)
    start = block_count // 2 if mode == "latter" else 0
    eligible = tuple(range(start + 1, block_count))
    measurements = [
        _score_pass(model, adapter, sample.input_ids, sample.attention_mask, start, measure, sample.allowed_token_ids, sample.key_token_id)[0]
        for sample in context.samples
    ]
    adjacent_scores = aggregate_ssn_scores(measurements, eligible)
    full_scores = np.zeros(block_count, dtype=float)
    for index, value in adjacent_scores.items():
        full_scores[index] = value
    if mode == "latter":
        midpoint = block_count // 2 + 1
        fixed = list(range(midpoint))
        sorted_latter = [int(index) for index in np.argsort(full_scores) if index not in fixed]
        sorted_blocks = np.concatenate((fixed, sorted_latter))
        ranking = tuple(int(index) for index in sorted_blocks[len(sorted_blocks) // 2 + 1:])
    else:
        ranking = tuple(int(index) for index in np.argsort(full_scores)[1:])
    scores = {index: float(full_scores[index]) for index in range(block_count)}
    return TaBPRanking(eligible, scores, ranking, len(measurements))


def rank_blocks_ddf(model: Any, adapter: BaseModelAdapter, context: TaBPCalibrationContext, *, mode: str, measure: str) -> TaBPRanking:
    block_count = adapter.get_num_blocks(model)
    start = block_count // 2 if mode == "latter" else 0
    measurements: list[dict[int, float]] = []
    if context.config.task_type == "qa":
        for sample in context.samples:
            values, _ = _score_pass(model, adapter, sample.input_ids, sample.attention_mask, start, measure, sample.allowed_token_ids, sample.key_token_id)
            measurements.append(values)
    else:
        if context.text_token_ids is None:
            raise ValueError("DDF text generation requires prepared WikiText token IDs")
        for window_index in range(context.config.n_windows):
            begin = window_index * context.config.window_size
            inputs = context.text_token_ids[:, begin:begin + context.config.window_size].clone()
            for _ in range(context.config.n_steps):
                values, generated = _score_pass(model, adapter, inputs, None, start, measure, None, 0)
                measurements.append(values)
                inputs = np_append_token(inputs, generated)
    scores = aggregate_ddf_scores(measurements, block_count, measure, context.config.task_type)
    values = np.array([scores[index] for index in range(block_count)])
    ranking = tuple(int(index) for index in np.argsort(-values))
    return TaBPRanking(tuple(range(block_count)), scores, ranking, len(measurements))


def np_append_token(inputs: Any, generated: Any) -> Any:
    import torch
    return torch.cat([inputs, generated.to(inputs.device)], dim=1)


class TaBPPruner(BasePruner):
    method_id = "tabp"
    implementation_version = "2.0"
    upstream_repo = "https://github.com/Song-haJo/TaBP"
    upstream_commit = "f56a0d2c38f490bec7b223bb98d1bf1740e5e92a"

    def __init__(self, *, ranking_strategy: str = "ssn", mode: str = "latter", measure: str = "entropy", lm_head_type: str = "frozen") -> None:
        if ranking_strategy not in {"ssn", "ddf"}:
            raise ValueError("TaBP ranking_strategy must be 'ssn' or 'ddf'")
        if mode not in {"latter", "whole"}:
            raise ValueError("TaBP mode must be 'latter' or 'whole'")
        if measure not in SUPPORTED_TABP_MEASURES:
            raise ValueError(f"Unsupported TaBP measure: {measure!r}")
        if ranking_strategy == "ddf" and lm_head_type != "frozen":
            raise ValueError("Pinned DDF uses the frozen model LM head")
        self.ranking_strategy = ranking_strategy
        self.mode = mode
        self.measure = measure
        self.lm_head_type = lm_head_type

    def prune(self, model: Any, adapter: BaseModelAdapter, request: PruningRequest, context: TaBPCalibrationContext | None = None) -> TaBPPruningSummary:
        if request.method != self.method_id:
            raise ValueError(f"Request method {request.method!r} does not match {self.method_id!r}")
        original_count = adapter.get_num_blocks(model)
        requested = ratio_to_block_count(original_count, request.sparsity)
        start = original_count // 2 if self.mode == "latter" else 0
        capacity = original_count - 1 if self.ranking_strategy == "ddf" else original_count - start - 1
        if requested > capacity:
            raise ValueError(f"TaBP requested_remove_count={requested} exceeds strategy capacity={capacity}")
        if requested and context is None:
            raise ValueError("TaBP pruning requires calibration context")
        original_parameters = sum(parameter.numel() for parameter in model.parameters())
        if requested:
            if self.ranking_strategy == "ssn":
                ranking = rank_blocks_ssn(model, adapter, context, mode=self.mode, measure=self.measure, lm_head_type=self.lm_head_type)
            else:
                ranking = rank_blocks_ddf(model, adapter, context, mode=self.mode, measure=self.measure)
            removed = ranking.ranking_original_indices[:requested]
        else:
            eligible = tuple(range(original_count)) if self.ranking_strategy == "ddf" else tuple(range(start + 1, original_count))
            ranking = TaBPRanking(eligible, {}, eligible, 0)
            removed = ()
        retained = remove_blocks(model, adapter, removed)
        final_parameters = sum(parameter.numel() for parameter in model.parameters())
        final_count = adapter.get_num_blocks(model)
        config = context.config if context is not None else None
        semantics = ("one_shot_mean_absolute_adjacent_shift_ascending" if self.ranking_strategy == "ssn" else "qa_undesirable_or_text_desirable_frequency_descending_all_blocks")
        return TaBPPruningSummary(
            pruner=self.method_id, implementation_version=self.implementation_version,
            upstream_repo=self.upstream_repo, upstream_commit=self.upstream_commit,
            official_semantics="pinned_executable_behavior",
            project_adaptation="adapter_based_blocks_and_ceil_ratio_to_count",
            pruning_type="block_structured", scope="transformer_decoder_blocks",
            target_sparsity_ratio=request.sparsity, requested_retention_ratio=1 - request.sparsity,
            ratio_to_count_policy="project_interface_ceil", original_block_count=original_count,
            requested_remove_count=requested, removed_block_count=len(removed),
            achieved_block_sparsity=len(removed) / original_count,
            achieved_block_retention=final_count / original_count,
            ranking_strategy=self.ranking_strategy, mode=self.mode, measure=self.measure,
            ranking_semantics=semantics, ranking_passes=1 if requested else 0,
            candidate_tie_rule="numpy_argsort_default_matching_upstream",
            eligible_original_indices=ranking.eligible_original_indices,
            scores_by_original_index=ranking.scores_by_original_index,
            ranked_block_indices=ranking.ranking_original_indices,
            removal_order_original_indices=tuple(removed), retained_original_indices=retained,
            calibration_source=config.source if config else "none",
            calibration_revision=config.dataset_revision if config else None,
            calibration_task_type=config.task_type if config else "none",
            calibration_samples=ranking.sample_count, calibration_seed=config.seed if config else 0,
            calibration_sampling_semantics=config.sampling_semantics if config else "none",
            calibration_tokenizer_class=context.tokenizer_class if context else "none",
            calibration_tokenizer_is_fast=context.tokenizer_is_fast if context else None,
            lm_head_type=self.lm_head_type,
            trained_lm_head_support="requires_explicit_compatible_upstream_checkpoints_not_enabled",
            retraining=False, gradients=False, hessian=False, weight_masking=False, weight_update=False,
            original_parameter_count=original_parameters, final_parameter_count=final_parameters,
            parameter_reduction_ratio=1 - final_parameters / original_parameters,
            post_pruning_num_hidden_layers=model.config.num_hidden_layers,
            post_pruning_actual_block_count=final_count,
        )
