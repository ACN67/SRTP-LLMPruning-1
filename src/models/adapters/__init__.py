"""Architecture-specific model adapters."""

from .granite import GraniteAdapter
from .qwen3 import Qwen3Adapter

__all__ = ["GraniteAdapter", "Qwen3Adapter"]
