"""Capture and replay native Hugging Face decoder block invocations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .base import BaseModelAdapter


@dataclass(frozen=True)
class BlockCallContext:
    positional_args: tuple[Any, ...]
    keyword_args: dict[str, Any]


@dataclass(frozen=True)
class CapturedModelSample:
    first_block_input: Any
    block_contexts: tuple[BlockCallContext, ...]
    native_block_outputs: tuple[Any, ...] | None = None


@dataclass(frozen=True)
class NativeCalibrationCapture:
    samples: tuple[CapturedModelSample, ...]
    number_of_blocks: int


def _detach_tree(value: Any) -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Native block capture requires PyTorch") from error
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_detach_tree(item) for item in value)
    if isinstance(value, list):
        return [_detach_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: _detach_tree(item) for key, item in value.items()}
    return value


def move_tree(value: Any, device: Any) -> Any:
    """Move tensors in nested native kwargs while preserving container types."""

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Native block replay requires PyTorch") from error
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(move_tree(item, device) for item in value)
    if isinstance(value, list):
        return [move_tree(item, device) for item in value]
    if isinstance(value, dict):
        return {key: move_tree(item, device) for key, item in value.items()}
    return value


def block_device(block: Any) -> Any:
    try:
        return next(block.parameters()).device
    except StopIteration as error:
        raise ValueError("Cannot replay a block without parameters") from error


def replay_captured_block(
    adapter: BaseModelAdapter,
    block: Any,
    hidden_states: Any,
    context: BlockCallContext,
) -> Any:
    """Replay a block on its device with its exact native call context."""

    device = block_device(block)
    hidden_states = hidden_states.to(device)
    positional_args = move_tree(context.positional_args, device)
    keyword_args = move_tree(context.keyword_args, device)
    return adapter.replay_block(block, hidden_states, positional_args, keyword_args)


def capture_native_calibration(
    model: Any,
    adapter: BaseModelAdapter,
    samples: Sequence[Any],
    *,
    capture_outputs: bool = False,
) -> NativeCalibrationCapture:
    """Use native backbone forward to capture first input and per-block kwargs.

    Only the first block input and call context are retained for production.
    Native block outputs are optional and intended for replay-equivalence tests.
    """

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Native block capture requires PyTorch") from error

    blocks = adapter.get_blocks(model)
    embedding_device = adapter.get_embedding(model).weight.device
    captured: list[CapturedModelSample] = []
    state: dict[str, Any] = {}
    handles = []

    def pre_hook(block_index: int):
        def hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            if not args and "hidden_states" not in kwargs:
                raise RuntimeError("Native block call did not provide hidden_states")
            hidden = args[0] if args else kwargs["hidden_states"]
            if block_index == 0:
                state["first_input"] = hidden.detach().clone()
            positional = tuple(_detach_tree(item) for item in args[1:])
            keyword = {
                key: _detach_tree(value)
                for key, value in kwargs.items()
                if key != "hidden_states"
            }
            state["contexts"][block_index] = BlockCallContext(positional, keyword)

        return hook

    def post_hook(block_index: int):
        def hook(
            _module: Any,
            _args: tuple[Any, ...],
            _kwargs: dict[str, Any],
            output: Any,
        ) -> None:
            if capture_outputs:
                hidden = adapter.normalize_block_output(output)
                state["outputs"][block_index] = hidden.detach().clone()

        return hook

    for index, block in enumerate(blocks):
        handles.append(block.register_forward_pre_hook(pre_hook(index), with_kwargs=True))
        handles.append(block.register_forward_hook(post_hook(index), with_kwargs=True))

    try:
        backbone = adapter.get_backbone(model)
        for sample in samples:
            state["first_input"] = None
            state["contexts"] = [None] * len(blocks)
            state["outputs"] = [None] * len(blocks)
            input_ids = sample.input_ids.to(embedding_device)
            kwargs: dict[str, Any] = {"input_ids": input_ids, "use_cache": False}
            if sample.attention_mask is not None:
                kwargs["attention_mask"] = sample.attention_mask.to(embedding_device)
            # Captured rotary tensors must remain legal inputs to later no_grad
            # replay; inference tensors cannot be reused by some native ops.
            with torch.no_grad():
                backbone(**kwargs)
            if state["first_input"] is None or any(
                item is None for item in state["contexts"]
            ):
                raise RuntimeError("Native model forward did not invoke every decoder block")
            outputs = None
            if capture_outputs:
                if any(item is None for item in state["outputs"]):
                    raise RuntimeError("Native model forward did not capture every block output")
                outputs = tuple(state["outputs"])
            captured.append(
                CapturedModelSample(
                    first_block_input=state["first_input"],
                    block_contexts=tuple(state["contexts"]),
                    native_block_outputs=outputs,
                )
            )
    finally:
        for handle in handles:
            handle.remove()

    return NativeCalibrationCapture(tuple(captured), len(blocks))
