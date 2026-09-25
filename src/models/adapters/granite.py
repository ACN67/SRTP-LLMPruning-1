"""Granite adapter placeholder for Granite-4.2-8B."""

from pathlib import Path
from typing import Any

from ..base import BaseModelAdapter, ModelSpec


class GraniteAdapter(BaseModelAdapter):
    adapter_id = "granite"
    architecture = "Granite"

    def load_model(
        self,
        spec: ModelSpec,
        *,
        model_root: Path,
        cache_dir: Path,
    ) -> Any:
        self.validate(spec)
        raise NotImplementedError(
            "Granite loading is a placeholder pending verification against "
            "ibm-granite/granite-4.2-8b and the selected pruner."
        )
