"""Basic unstructured SparseGPT with sequential native block replay.

The mathematical core is adapted from the Apache-2.0 licensed official
SparseGPT repository at commit 147d2159dc4f3e9f73e47b32c04d7b3708f44436.
Only the ordinary ``torch.nn.Linear`` unstructured path is implemented here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from src.models.base import BaseModelAdapter
from src.models.replay import capture_native_calibration, replay_captured_block

from .base import BasePruner, ModulePruningStats, PruningRequest, PruningSummary
from .calibration import CalibrationContext


class SparseGPTHessian:
    """Accumulate the official full FP32 input Hessian/Gram matrix."""

    def __init__(self, in_features: int, device: Any) -> None:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("SparseGPT Hessian accumulation requires PyTorch") from error
        self.H = torch.zeros(
            (in_features, in_features), dtype=torch.float32, device=device
        )
        self.nsamples = 0

    def add(self, inputs: Any) -> None:
        if self.H is None:
            raise RuntimeError("Cannot add inputs after SparseGPT Hessian was cleared")
        if inputs.shape[-1] != self.H.shape[0]:
            raise ValueError(
                f"Activation width {inputs.shape[-1]} does not match Hessian width "
                f"{self.H.shape[0]}"
            )
        inp = inputs.detach()
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def clear(self) -> None:
        self.H = None


@dataclass(frozen=True)
class SparseGPTCoreResult:
    weight: Any
    mask: Any
    damping: float


def sparsegpt_reconstruct(
    weight: Any,
    hessian: Any,
    sparsity: float,
    *,
    percdamp: float = 0.01,
    blocksize: int = 128,
) -> SparseGPTCoreResult:
    """Apply the official basic unstructured ``fasterprune`` matrix core."""

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("SparseGPT reconstruction requires PyTorch") from error
    if weight.ndim != 2:
        raise ValueError(f"SparseGPT requires matrix weights, got {tuple(weight.shape)}")
    if hessian.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("SparseGPT Hessian shape does not match weight input width")
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity must be in the half-open interval [0, 1)")
    if percdamp < 0:
        raise ValueError("SparseGPT percdamp must be non-negative")
    if blocksize <= 0:
        raise ValueError("SparseGPT blocksize must be positive")

    W = weight.detach().clone().float()
    selected = torch.zeros_like(W, dtype=torch.bool)
    if sparsity == 0:
        return SparseGPTCoreResult(W.to(weight.dtype), selected, 0.0)

    H = hessian.detach().clone().to(device=W.device, dtype=torch.float32)
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    damp_tensor = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(W.shape[1], device=W.device)
    H[diag, diag] += damp_tensor
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    Hinv = torch.linalg.cholesky(H, upper=True)

    for i1 in range(0, W.shape[1], blocksize):
        i2 = min(i1 + blocksize, W.shape[1])
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        tmp = W1 ** 2 / torch.diag(Hinv1).reshape((1, -1)) ** 2
        thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
        mask1 = tmp <= thresh

        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]
            q = w.clone()
            q[mask1[:, i]] = 0
            Q1[:, i] = q
            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(
                Hinv1[i, i:].unsqueeze(0)
            )
            Err1[:, i] = err1

        W[:, i1:i2] = Q1
        selected[:, i1:i2] = mask1
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

    return SparseGPTCoreResult(
        W.reshape(weight.shape).to(weight.dtype),
        selected,
        float(damp_tensor.item()),
    )


@dataclass(frozen=True)
class SparseGPTPruningSummary(PruningSummary):
    method: str
    structure: str
    criterion: str
    target_sparsity_ratio: float
    actual_mask_count: int
    actual_mask_sparsity: float
    actual_sparsity: float
    percdamp: float
    blocksize: int
    adaptive_mask_selection: bool
    error_compensation: bool
    reconstruction: bool
    weight_update: bool
    retraining: bool
    quantization: bool
    nm_sparsity: bool
    true_sequential: bool
    calibration_source: str
    calibration_sample_count: int
    calibration_sequence_length: int
    calibration_seed: int
    sequential_layerwise: bool


class SparseGPTPruner(BasePruner):
    method_id = "sparsegpt"
    implementation_version = "1.0"
    pruning_type = "unstructured"
    scope = "per_linear_adaptive_input_column_blocks"
    criterion = "obs_reconstruction_error"
    target_policy = "torch.nn.Linear weights inside decoder/transformer blocks"
    excluded_components = (
        "token_embeddings",
        "final_norm",
        "all_norm_parameters",
        "lm_head",
        "bias",
        "parameters_outside_blocks",
    )

    def __init__(self, *, percdamp: float = 0.01, blocksize: int = 128) -> None:
        if percdamp < 0:
            raise ValueError("SparseGPT percdamp must be non-negative")
        if blocksize <= 0:
            raise ValueError("SparseGPT blocksize must be positive")
        self.percdamp = percdamp
        self.blocksize = blocksize

    def prune(
        self,
        model: Any,
        adapter: BaseModelAdapter,
        request: PruningRequest,
        context: CalibrationContext | None = None,
    ) -> SparseGPTPruningSummary:
        if context is None:
            raise ValueError("SparseGPT pruning requires calibration context")
        if request.method != self.method_id:
            raise ValueError(
                f"Request method {request.method!r} does not match {self.method_id!r}"
            )
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("SparseGPT pruning requires PyTorch") from error

        native = capture_native_calibration(model, adapter, context.samples)
        blocks = adapter.get_blocks(model)
        current_inputs = [sample.first_block_input for sample in native.samples]
        per_module: list[ModulePruningStats] = []
        total_actual_mask_count = 0

        with torch.no_grad():
            for block_index, block in enumerate(blocks):
                linears = adapter.get_linear_modules(block)
                if not linears:
                    raise ValueError(f"Block {block_index} contains no torch.nn.Linear modules")
                accumulators = {
                    name: SparseGPTHessian(module.in_features, module.weight.device)
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
                    before = weight.detach().clone()
                    preexisting = int(torch.count_nonzero(before == 0).item())
                    accumulator = accumulators[local_name]
                    try:
                        result = sparsegpt_reconstruct(
                            weight,
                            accumulator.H,
                            request.sparsity,
                            percdamp=self.percdamp,
                            blocksize=self.blocksize,
                        )
                    except torch.linalg.LinAlgError as error:
                        raise RuntimeError(
                            "SparseGPT Cholesky failed for "
                            f"block={block_index}, module={local_name}, "
                            f"input_width={weight.shape[1]}, percdamp={self.percdamp}"
                        ) from error
                    finally:
                        accumulator.clear()
                    weight.copy_(result.weight)
                    actual_mask_count = int(torch.count_nonzero(result.mask).item())
                    nominal_requested = sum(
                        int(
                            weight.shape[0]
                            * min(self.blocksize, weight.shape[1] - block_start)
                            * request.sparsity
                        )
                        for block_start in range(0, weight.shape[1], self.blocksize)
                    )
                    total_actual_mask_count += actual_mask_count
                    newly_zeroed = int(
                        torch.count_nonzero((before != 0) & (weight == 0)).item()
                    )
                    post_zeros = int(torch.count_nonzero(weight == 0).item())
                    targeted = weight.numel()
                    per_module.append(
                        ModulePruningStats(
                            module=f"blocks.{block_index}.{local_name}",
                            shape=tuple(weight.shape),
                            targeted_weights=targeted,
                            requested_mask_count=nominal_requested,
                            preexisting_zeros=preexisting,
                            newly_zeroed_weights=newly_zeroed,
                            post_pruning_zeros=post_zeros,
                            achieved_mask_sparsity=actual_mask_count / targeted,
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
        return SparseGPTPruningSummary(
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
            achieved_mask_sparsity=total_actual_mask_count / targeted_weights,
            achieved_zero_sparsity=post_pruning_zeros / targeted_weights,
            per_module=tuple(per_module),
            method=self.method_id,
            structure=self.pruning_type,
            criterion=self.criterion,
            target_sparsity_ratio=request.sparsity,
            actual_mask_count=total_actual_mask_count,
            actual_mask_sparsity=total_actual_mask_count / targeted_weights,
            actual_sparsity=post_pruning_zeros / targeted_weights,
            percdamp=self.percdamp,
            blocksize=self.blocksize,
            adaptive_mask_selection=True,
            error_compensation=True,
            reconstruction=True,
            weight_update=True,
            retraining=False,
            quantization=False,
            nm_sparsity=False,
            true_sequential=False,
            calibration_source=context.config.source,
            calibration_sample_count=context.config.samples,
            calibration_sequence_length=context.config.sequence_length,
            calibration_seed=context.config.seed,
            sequential_layerwise=True,
        )
