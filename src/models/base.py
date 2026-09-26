"""Model metadata and architecture-adapter contracts."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


IMMUTABLE_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class ModelSpec:
    """Stable project metadata for one upstream model repository."""

    project_model_id: str
    display_name: str
    huggingface_repo_id: str
    architecture: str
    adapter: str
    model_type: str
    expected_model_class: str
    revision: str
    default_dtype: str
    local_path: Path | None
    trust_remote_code: bool
    expected_num_hidden_layers: int
    expected_hidden_size: int
    expected_intermediate_size: int
    expected_num_attention_heads: int
    expected_num_key_value_heads: int

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ModelSpec":
        required = (
            "project_model_id",
            "display_name",
            "huggingface_repo_id",
            "architecture",
            "adapter",
            "model_type",
            "expected_model_class",
            "revision",
            "default_dtype",
            "expected_num_hidden_layers",
            "expected_hidden_size",
            "expected_intermediate_size",
            "expected_num_attention_heads",
            "expected_num_key_value_heads",
        )
        missing = [key for key in required if data.get(key) in (None, "")]
        if missing:
            raise ValueError(f"Missing model configuration keys: {', '.join(missing)}")

        revision = str(data["revision"])
        if not IMMUTABLE_REVISION_PATTERN.fullmatch(revision):
            raise ValueError(
                "Model revision must be a 40-character lowercase hexadecimal commit: "
                f"{revision!r}"
            )

        local_path_value = data.get("local_path")
        local_path = Path(str(local_path_value)).expanduser() if local_path_value else None
        integer_fields = {
            key: int(data[key])
            for key in (
                "expected_num_hidden_layers",
                "expected_hidden_size",
                "expected_intermediate_size",
                "expected_num_attention_heads",
                "expected_num_key_value_heads",
            )
        }
        if any(value <= 0 for value in integer_fields.values()):
            raise ValueError("Expected model dimensions must be positive integers")

        return cls(
            project_model_id=str(data["project_model_id"]),
            display_name=str(data["display_name"]),
            huggingface_repo_id=str(data["huggingface_repo_id"]),
            architecture=str(data["architecture"]),
            adapter=str(data["adapter"]),
            model_type=str(data["model_type"]),
            expected_model_class=str(data["expected_model_class"]),
            revision=revision,
            default_dtype=str(data["default_dtype"]),
            local_path=local_path,
            trust_remote_code=bool(data.get("trust_remote_code", False)),
            **integer_fields,
        )

    def expected_structure(self) -> dict[str, int]:
        return {
            "num_hidden_layers": self.expected_num_hidden_layers,
            "hidden_size": self.expected_hidden_size,
            "intermediate_size": self.expected_intermediate_size,
            "num_attention_heads": self.expected_num_attention_heads,
            "num_key_value_heads": self.expected_num_key_value_heads,
        }


class BaseModelAdapter(ABC):
    """Stable traversal API over a native Hugging Face causal LM.

    Adapters expose model structure only. They intentionally do not copy or
    replace the architecture's decoder forward implementation.
    """

    adapter_id: str
    architecture: str
    model_type: str

    def validate_spec(self, spec: ModelSpec) -> None:
        if spec.adapter != self.adapter_id:
            raise ValueError(
                f"Model {spec.project_model_id!r} requests adapter {spec.adapter!r}, "
                f"not {self.adapter_id!r}."
            )
        if spec.model_type != self.model_type:
            raise ValueError(
                f"Model {spec.project_model_id!r} has model_type {spec.model_type!r}, "
                f"not {self.model_type!r}."
            )

    @abstractmethod
    def get_backbone(self, model: Any) -> Any:
        """Return the decoder-only backbone that owns blocks and embeddings."""

    def get_blocks(self, model: Any) -> Sequence[Any]:
        blocks = self.get_backbone(model).layers
        if not hasattr(blocks, "__len__") or not hasattr(blocks, "__getitem__"):
            raise TypeError("Model blocks must be an indexable sequence")
        return blocks

    def get_num_blocks(self, model: Any) -> int:
        return len(self.get_blocks(model))

    def replace_blocks(self, model: Any, blocks: Sequence[Any]) -> Any:
        """Replace the native block container without copying block forwards."""

        try:
            from torch import nn
        except ImportError as error:
            raise RuntimeError("PyTorch is required for block replacement") from error
        container = blocks if isinstance(blocks, nn.ModuleList) else nn.ModuleList(blocks)
        self.get_backbone(model).layers = container
        return container

    def capture_block_removal_metadata(self, model: Any) -> Any:
        """Capture architecture metadata needed after physical block removal."""

        return None

    def finalize_block_removal(
        self,
        model: Any,
        retained_original_indices: Sequence[int],
        metadata: Any = None,
    ) -> None:
        """Finalize native config depth and cache-facing attention indices."""

        blocks = self.get_blocks(model)
        if len(blocks) != len(retained_original_indices):
            raise ValueError("Retained index count does not match final block count")
        model.config.num_hidden_layers = len(blocks)
        for new_index, block in enumerate(blocks):
            attention = getattr(block, "self_attn", None)
            if attention is not None and hasattr(attention, "layer_idx"):
                attention.layer_idx = new_index

    def get_linear_modules(
        self,
        block: Any,
        *,
        linear_type: type[Any] | tuple[type[Any], ...] | None = None,
    ) -> dict[str, Any]:
        """Return named linear modules inside one block.

        ``linear_type`` enables dependency-free lightweight tests. Runtime
        callers normally omit it and use ``torch.nn.Linear``.
        """

        if linear_type is None:
            try:
                from torch import nn
            except ImportError as error:
                raise RuntimeError("PyTorch is required for runtime model traversal") from error
            linear_type = nn.Linear
        return {
            name: module
            for name, module in block.named_modules()
            if name and isinstance(module, linear_type)
        }

    def get_embedding(self, model: Any) -> Any:
        return self.get_backbone(model).embed_tokens

    def get_final_norm(self, model: Any) -> Any | None:
        return getattr(self.get_backbone(model), "norm", None)

    def get_lm_head(self, model: Any) -> Any:
        return model.lm_head

    @abstractmethod
    def normalize_block_output(self, output: Any) -> Any:
        """Return hidden states from this architecture's native block output."""

    def replay_block(
        self,
        block: Any,
        hidden_states: Any,
        positional_args: tuple[Any, ...],
        keyword_args: Mapping[str, Any],
    ) -> Any:
        """Invoke a native block with context captured from model forward."""

        output = block(hidden_states, *positional_args, **keyword_args)
        return self.normalize_block_output(output)

    def get_structure(self, model: Any) -> dict[str, Any]:
        config = model.config
        return {
            "model_type": getattr(config, "model_type", None),
            "num_hidden_layers": getattr(config, "num_hidden_layers", None),
            "hidden_size": getattr(config, "hidden_size", None),
            "intermediate_size": getattr(config, "intermediate_size", None),
            "num_attention_heads": getattr(config, "num_attention_heads", None),
            "num_key_value_heads": getattr(config, "num_key_value_heads", None),
            "actual_block_count": self.get_num_blocks(model),
        }

    def validate_config(self, config: Any, spec: ModelSpec) -> dict[str, Any]:
        self.validate_spec(spec)
        observed = {
            "model_type": getattr(config, "model_type", None),
            **{
                field: getattr(config, field, None)
                for field in spec.expected_structure()
            },
        }
        mismatches: list[str] = []
        if observed["model_type"] != spec.model_type:
            mismatches.append(
                f"model_type={observed['model_type']!r}, expected {spec.model_type!r}"
            )
        for field, expected in spec.expected_structure().items():
            if observed[field] != expected:
                mismatches.append(f"{field}={observed[field]!r}, expected {expected!r}")
        if mismatches:
            raise ValueError("Loaded model config mismatch: " + "; ".join(mismatches))
        return observed

    def validate_loaded_model(self, model: Any, spec: ModelSpec) -> dict[str, Any]:
        self.validate_config(model.config, spec)
        structure = self.get_structure(model)
        mismatches: list[str] = []
        if structure["actual_block_count"] != spec.expected_num_hidden_layers:
            mismatches.append(
                "actual_block_count="
                f"{structure['actual_block_count']!r}, expected {spec.expected_num_hidden_layers!r}"
            )
        if mismatches:
            raise ValueError("Loaded model structure mismatch: " + "; ".join(mismatches))
        return structure
