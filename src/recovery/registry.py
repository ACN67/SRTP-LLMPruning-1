"""Recovery method registry."""

RECOVERY_METHODS = ("lora",)


def get_recovery_method(name: str):
    if name != "lora":
        raise KeyError(f"Unknown recovery method {name!r}; available: lora")
    from .lora import LoRARecovery
    return LoRARecovery()
