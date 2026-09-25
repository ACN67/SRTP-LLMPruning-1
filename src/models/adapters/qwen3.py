"""Qwen3 adapter placeholder for Klear-AgentForge-8B."""

from pathlib import Path
from typing import Any

from ..base import BaseModelAdapter, ModelSpec


class Qwen3Adapter(BaseModelAdapter):
    adapter_id = "qwen3"
    architecture = "Qwen3"

    def load_model(
        self,
        spec: ModelSpec,
        *,
        model_root: Path,
        cache_dir: Path,
    ) -> Any:
        self.validate(spec)
        raise NotImplementedError(
            "Qwen3 loading is a placeholder pending verification against "
            "Kwai-Klear/Klear-AgentForge-8B and the selected pruner."
        )
