"""Transformers-native generation driven by a pinned evaluation profile."""

from __future__ import annotations

import hashlib
import random
import time
from typing import Any, Mapping, Sequence

from .profiles import EvaluationProfile


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _token_id(model: Any, tokenizer: Any, name: str) -> Any:
    generation_config = getattr(model, "generation_config", None)
    for owner in (generation_config, getattr(model, "config", None), tokenizer):
        value = getattr(owner, name, None) if owner is not None else None
        if value is not None:
            return value
    return None


def _generation_value(model: Any, name: str, default: Any = None) -> Any:
    generation_config = getattr(model, "generation_config", None)
    value = getattr(generation_config, name, None) if generation_config is not None else None
    return default if value is None else value


def generate_one(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    messages: Sequence[Mapping[str, str]],
    profile: EvaluationProfile,
    seed: int,
) -> dict[str, Any]:
    """Generate one completion without replacing model-native stop-token policy."""

    _seed_everything(seed)
    rendered_prompt = prompt
    if profile.use_chat_template:
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("Tokenizer does not support apply_chat_template")
        rendered_prompt = tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True,
            **dict(profile.chat_template_kwargs),
        )
    encoded = tokenizer(rendered_prompt, return_tensors="pt", truncation=False)
    input_ids = encoded["input_ids"]
    model_limit = getattr(model.config, "max_position_embeddings", None)
    if model_limit is not None and input_ids.shape[-1] + profile.max_new_tokens > model_limit:
        raise ValueError("Prompt plus requested generation exceeds model context; truncation refused")
    device = getattr(model, "device", None)
    if device is not None:
        encoded = {key: value.to(device) for key, value in encoded.items()}
    effective_eos = _token_id(model, tokenizer, "eos_token_id")
    effective_pad = _token_id(model, tokenizer, "pad_token_id")
    effective_sampling = {
        key: (getattr(profile, key) if getattr(profile, key) is not None
              else _generation_value(model, key))
        for key in ("temperature", "top_p", "top_k")
    }
    if not profile.do_sample:
        effective_sampling = {key: None for key in effective_sampling}
    effective_num_beams = _generation_value(model, "num_beams", 1)
    effective_repetition_penalty = _generation_value(model, "repetition_penalty", 1.0)
    kwargs: dict[str, Any] = {
        "max_new_tokens": profile.max_new_tokens,
        "do_sample": profile.do_sample,
        "use_cache": True,
    }
    if effective_pad is None:
        effective_pad = effective_eos
        if effective_pad is not None:
            kwargs["pad_token_id"] = effective_pad
    if profile.do_sample:
        for key, value in effective_sampling.items():
            if value is not None:
                kwargs[key] = value
    cuda_device = None
    try:
        import torch
        candidate = getattr(model, "device", None)
        if candidate is None:
            candidate = next(model.parameters()).device
        candidate = torch.device(candidate) if candidate is not None else None
        if candidate is not None and candidate.type == "cuda" and torch.cuda.is_available():
            cuda_device = candidate
            torch.cuda.synchronize(cuda_device)
            torch.cuda.reset_peak_memory_stats(cuda_device)
    except (AttributeError, ImportError, StopIteration, RuntimeError, TypeError):
        cuda_device = None
    started = time.perf_counter()
    output = model.generate(**encoded, **kwargs)
    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)
    wall_time = time.perf_counter() - started
    peak_cuda_vram = (
        int(torch.cuda.max_memory_allocated(cuda_device))
        if cuda_device is not None else None
    )
    prompt_tokens = input_ids.shape[-1]
    generated = output[0, prompt_tokens:]
    raw = tokenizer.decode(generated, skip_special_tokens=True)
    generated_tokens = int(generated.numel())
    return {
        "prompt": rendered_prompt,
        "prompt_sha256": hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest(),
        "raw_generation": raw,
        "prompt_tokens": int(prompt_tokens),
        "generated_tokens": generated_tokens,
        "generation_wall_time_seconds": wall_time,
        "generated_tokens_per_second": generated_tokens / wall_time,
        "peak_cuda_vram_bytes": peak_cuda_vram,
        "truncated": False,
        "seed": seed,
        "chat_template_used": profile.use_chat_template,
        "chat_template_kwargs": dict(profile.chat_template_kwargs),
        "effective_generation_config": {
            **kwargs,
            **effective_sampling,
            "eos_token_id": effective_eos,
            "pad_token_id": effective_pad,
            "num_beams": effective_num_beams,
            "repetition_penalty": effective_repetition_penalty,
        },
    }
