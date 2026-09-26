"""Basic unstructured Wanda with sequential native block replay.

The score/statistic and layer-wise flow are independently implemented from the
MIT-licensed official Wanda repository at commit
8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.models.base import BaseModelAdapter
from src.models.replay import capture_native_calibration, replay_captured_block

from .base import BasePruner, ModulePruningStats, PruningRequest, PruningSummary
from .calibration import WandaPruningContext


class WandaActivationStats:
    """Accumulate squared L2 statistics for Linear input channels."""

    def __init__(self, in_features: int, device: Any) -> None:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("Wanda activation statistics require PyTorch") from error
        self.sum_squared = torch.zeros(in_features, dtype=torch.float32, device=device)
        self.sample_count = 0
        self.token_count = 0

    def add(self, inputs: Any) -> None:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("Wanda activation statistics require PyTorch") from error
        if inputs.shape[-1] != self.sum_squared.numel():
            raise ValueError(
                f"Activation width {inputs.shape[-1]} does not match Linear "
                f"in_features {self.sum_squared.numel()}"
            )
        flattened = inputs.detach().reshape(-1, inputs.shape[-1]).to(torch.float32)
        self.sum_squared.add_(flattened.square().sum(dim=0))
        self.sample_count += inputs.shape[0] if inputs.ndim > 1 else 1
        self.token_count += flattened.shape[0]

    def input_l2(self) -> Any:
        return self.sum_squared.sqrt()


def wanda_mask(weight: Any, input_l2: Any, sparsity: float) -> Any:
    """Return the stable exact-k basic Wanda mask for one Linear."""

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Wanda masking requires PyTorch") from error
    if weight.ndim != 2:
        raise ValueError(f"Wanda requires matrix weights, received {tuple(weight.shape)}")
    if input_l2.ndim != 1 or input_l2.numel() != weight.shape[1]:
        raise ValueError("Input-channel statistic shape does not match weight columns")
    requested_per_row = int(weight.shape[1] * sparsity)
    mask = torch.zeros_like(weight, dtype=torch.bool)
    if requested_per_row:
        score = weight.detach().abs().to(torch.float32) * input_l2.to(
            device=weight.device, dtype=torch.float32
        ).reshape(1, -1)
        selected = torch.argsort(score, dim=-1, stable=True)[:, :requested_per_row]
        mask.scatter_(1, selected, True)
    return mask


@dataclass(frozen=True)
class WandaPruningSummary(PruningSummary):
    score: str
    calibration_source: str
    calibration_sample_count: int
    calibration_sequence_length: int
    calibration_seed: int
    sequential_layerwise: bool
    weight_update: bool
    retraining: bool


class WandaPruner(BasePruner):
    method_id = "wanda"
    implementation_version = "1.0"
    pruning_type = "unstructured"
    scope = "per_output_row"
    score = "absolute_weight_times_input_activation_l2"
    target_policy = "torch.nn.Linear weights inside decoder/transformer blocks"
    excluded_components = (
        "token_embeddings",
        "final_norm",
        "all_norm_parameters",
        "lm_head",
        "bias",
        "parameters_outside_blocks",
    )

    def prune(
        self,
        model: Any,
        adapter: BaseModelAdapter,
        request: PruningRequest,
        context: WandaPruningContext | None = None,
    ) -> WandaPruningSummary:
        if context is None:
            raise ValueError("Wanda pruning requires calibration context")
        if request.method != self.method_id:
            raise ValueError(
                f"Request method {request.method!r} does not match {self.method_id!r}"
            )
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("Wanda pruning requires PyTorch") from error

        native = capture_native_calibration(model, adapter, context.samples)
        blocks = adapter.get_blocks(model)
        current_inputs = [sample.first_block_input for sample in native.samples]
        per_module: list[ModulePruningStats] = []

        with torch.no_grad():
            for block_index, block in enumerate(blocks):
                linears = adapter.get_linear_modules(block)
                if not linears:
                    raise ValueError(f"Block {block_index} contains no torch.nn.Linear modules")
                accumulators = {
                    name: WandaActivationStats(module.in_features, module.weight.device)
                    for name, module in linears.items()
                }

                def make_hook(name: str):
                    def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
                        if not inputs:
                            raise RuntimeError(f"Linear {name!r} received no positional input")
                        accumulators[name].add(inputs[0])

                    return hook

                handles = [
                    module.register_forward_pre_hook(make_hook(name))
                    for name, module in linears.items()
                ]
                try:
                    for sample_index, hidden_states in enumerate(current_inputs):
                        replay_captured_block(
                            adapter,
                            block,
                            hidden_states,
                            native.samples[sample_index].block_contexts[block_index],
                        )
                finally:
                    for handle in handles:
                        handle.remove()

                for local_name, module in linears.items():
                    weight = module.weight
                    preexisting = int(torch.count_nonzero(weight == 0).item())
                    mask = wanda_mask(
                        weight,
                        accumulators[local_name].input_l2(),
                        request.sparsity,
                    )
                    requested = int(torch.count_nonzero(mask).item())
                    expected = weight.shape[0] * int(weight.shape[1] * request.sparsity)
                    if requested != expected:
                        raise RuntimeError(
                            f"Wanda mask count mismatch for block {block_index} "
                            f"{local_name}: {requested} != {expected}"
                        )
                    newly_zeroed = int(torch.count_nonzero(mask & (weight != 0)).item())
                    weight.masked_fill_(mask, 0)
                    post_zeros = int(torch.count_nonzero(weight == 0).item())
                    if post_zeros != preexisting + newly_zeroed:
                        raise RuntimeError(
                            f"Zero accounting mismatch for block {block_index} {local_name}"
                        )
                    targeted = weight.numel()
                    per_module.append(
                        ModulePruningStats(
                            module=f"blocks.{block_index}.{local_name}",
                            shape=tuple(weight.shape),
                            targeted_weights=targeted,
                            requested_mask_count=requested,
                            preexisting_zeros=preexisting,
                            newly_zeroed_weights=newly_zeroed,
                            post_pruning_zeros=post_zeros,
                            achieved_mask_sparsity=requested / targeted,
                            achieved_zero_sparsity=post_zeros / targeted,
                        )
                    )

                next_inputs = []
                for sample_index, hidden_states in enumerate(current_inputs):
                    output = replay_captured_block(
                        adapter,
                        block,
                        hidden_states,
                        native.samples[sample_index].block_contexts[block_index],
                    )
                    next_inputs.append(output.detach())
                current_inputs = next_inputs

        targeted_weights = sum(item.targeted_weights for item in per_module)
        requested_mask_count = sum(item.requested_mask_count for item in per_module)
        preexisting_zeros = sum(item.preexisting_zeros for item in per_module)
        newly_zeroed_weights = sum(item.newly_zeroed_weights for item in per_module)
        post_pruning_zeros = sum(item.post_pruning_zeros for item in per_module)
        return WandaPruningSummary(
            pruner=self.method_id,
            implementation_version=self.implementation_version,
            pruning_type=self.pruning_type,
            scope=self.scope,
            sparsity_ratio=request.sparsity,
            target_policy=self.target_policy,
            excluded_components=self.excluded_components,
            number_of_target_modules=len(per_module),
            targeted_weights=targeted_weights,
            requested_mask_count=requested_mask_count,
            preexisting_zeros=preexisting_zeros,
            newly_zeroed_weights=newly_zeroed_weights,
            post_pruning_zeros=post_pruning_zeros,
            achieved_mask_sparsity=requested_mask_count / targeted_weights,
            achieved_zero_sparsity=post_pruning_zeros / targeted_weights,
            per_module=tuple(per_module),
            score=self.score,
            calibration_source=context.config.source,
            calibration_sample_count=context.config.samples,
            calibration_sequence_length=context.config.sequence_length,
            calibration_seed=context.config.seed,
            sequential_layerwise=True,
            weight_update=False,
            retraining=False,
        )
