"""Deterministic per-module unstructured magnitude pruning."""

from __future__ import annotations

from typing import Any

from src.models.base import BaseModelAdapter

from .base import BasePruner, ModulePruningStats, PruningRequest, PruningSummary


class MagnitudePruner(BasePruner):
    method_id = "magnitude"
    implementation_version = "1.0"
    pruning_type = "unstructured"
    scope = "per_module_flattened"
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
        context: Any | None = None,
    ) -> PruningSummary:
        if context is not None:
            raise ValueError("Magnitude pruning does not accept calibration context")
        if request.method != self.method_id:
            raise ValueError(
                f"Request method {request.method!r} does not match {self.method_id!r}"
            )

        try:
            import torch
        except ImportError as error:
            raise RuntimeError("Magnitude pruning requires PyTorch") from error

        per_module: list[ModulePruningStats] = []
        with torch.no_grad():
            for block_index, block in enumerate(adapter.get_blocks(model)):
                for local_name, module in adapter.get_linear_modules(block).items():
                    weight = module.weight
                    if weight.ndim != 2:
                        raise ValueError(
                            f"Target Linear {local_name!r} has non-matrix weight shape "
                            f"{tuple(weight.shape)!r}"
                        )
                    requested = int(weight.numel() * request.sparsity)
                    preexisting = int(torch.count_nonzero(weight == 0).item())

                    if requested:
                        # Stable sorting resolves ties by row-major flattened index.
                        metric = weight.detach().abs().reshape(-1)
                        selected = torch.argsort(metric, stable=True)[:requested]
                        flat_mask = torch.zeros_like(metric, dtype=torch.bool)
                        flat_mask.scatter_(0, selected, True)
                        mask = flat_mask.reshape_as(weight)
                        actual_mask_count = int(torch.count_nonzero(mask).item())
                        if actual_mask_count != requested:
                            raise RuntimeError(
                                f"Mask count mismatch for block {block_index} {local_name}: "
                                f"{actual_mask_count} != {requested}"
                            )
                        newly_zeroed = int(
                            torch.count_nonzero(mask & (weight != 0)).item()
                        )
                        weight.masked_fill_(mask, 0)
                    else:
                        newly_zeroed = 0

                    post_zeros = int(torch.count_nonzero(weight == 0).item())
                    targeted = weight.numel()
                    if post_zeros != preexisting + newly_zeroed:
                        raise RuntimeError(
                            f"Zero accounting mismatch for block {block_index} {local_name}"
                        )
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

        if not per_module:
            raise ValueError("Adapter traversal found no torch.nn.Linear modules in blocks")

        targeted_weights = sum(item.targeted_weights for item in per_module)
        requested_mask_count = sum(item.requested_mask_count for item in per_module)
        preexisting_zeros = sum(item.preexisting_zeros for item in per_module)
        newly_zeroed_weights = sum(item.newly_zeroed_weights for item in per_module)
        post_pruning_zeros = sum(item.post_pruning_zeros for item in per_module)
        return PruningSummary(
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
        )
