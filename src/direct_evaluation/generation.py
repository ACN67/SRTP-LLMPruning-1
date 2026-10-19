"""Transformers-native generation driven by a pinned evaluation profile."""

from __future__ import annotations

import hashlib
import random
import time
from typing import Any, Mapping, Sequence

from .profiles import EvaluationProfile


TOKENIZER_INPUT_PROTOCOL = "rendered_chat_token_ids_v1"
DECODE_PROTOCOL = "skip_special_tokens_true_v1"
KLEAR_EMPTY_THINK_PROTOCOL = "klear_qwen3_empty_think_prefill_v1"
KLEAR_TEMPLATE_SHA256 = "f858b0b34b6c89118490c6c087cbbc5df911f56994cc8cb61ee7a619903abcd8"
KLEAR_REVISION = "fa3d41e92e9ce7a5b4a52a3e7439aa00521f40c9"
KLEAR_ASSISTANT_START = "<|im_start|>assistant\n"
KLEAR_EMPTY_THINK_SUFFIX = "<think>\n\n</think>\n\n"
GRANITE_NATIVE_NONTHINK_PROTOCOL = "granite_native_enable_thinking_false_v1"
GRANITE_TEMPLATE_SHA256 = "f0ba43f79b3cabca5e5a7584c77aec92f49feae45cdf7779c87d9fc54cd90258"
GRANITE_THINK_TAIL = "<|im_start|>assistant\n<think>\n"
GRANITE_NONTHINK_TAIL = "<|im_start|>assistant\n<think></think>"


def render_and_encode(tokenizer: Any, prompt: str, messages: Sequence[Mapping[str, str]],
                      profile: EvaluationProfile) -> tuple[str, Any, dict[str, Any]]:
    """Keep legacy rendering exact; apply the pinned Klear hard switch only by opt-in."""
    rendered = prompt
    if profile.use_chat_template:
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("Tokenizer does not support apply_chat_template")
        rendered = tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True,
            **dict(profile.chat_template_kwargs),
        )
    provenance: dict[str, Any] = {}
    if profile.input_protocol == KLEAR_EMPTY_THINK_PROTOCOL:
        if (profile.project_model_id != "klear_agentforge_8b" or
                not profile.use_chat_template or profile.chat_template_kwargs):
            raise ValueError("Unsupported Klear input protocol or profile")
        template = getattr(tokenizer, "chat_template", None)
        if not isinstance(template, str) or hashlib.sha256(template.encode()).hexdigest() != KLEAR_TEMPLATE_SHA256:
            raise ValueError("Pinned Klear chat template hash mismatch")
        if not rendered.endswith(KLEAR_ASSISTANT_START) or rendered.endswith(KLEAR_EMPTY_THINK_SUFFIX):
            raise ValueError("Klear assistant prefix mismatch or duplicate prefill")
        old_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        rendered += KLEAR_EMPTY_THINK_SUFFIX
        suffix_ids = tokenizer(KLEAR_EMPTY_THINK_SUFFIX, add_special_tokens=False)["input_ids"]
        new_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        if new_ids != old_ids + suffix_ids or suffix_ids != [151667, 271, 151668, 271]:
            raise ValueError("Klear prefill token IDs are not the pinned append-only encoding")
        provenance = {
            "input_protocol": profile.input_protocol,
            "chat_template_sha256": KLEAR_TEMPLATE_SHA256,
            "prefill_suffix_sha256": hashlib.sha256(KLEAR_EMPTY_THINK_SUFFIX.encode()).hexdigest(),
            "prefill_control_tokens": suffix_ids,
            "original_prompt_sha256": hashlib.sha256(rendered[:-len(KLEAR_EMPTY_THINK_SUFFIX)].encode()).hexdigest(),
        }
    elif profile.input_protocol == GRANITE_NATIVE_NONTHINK_PROTOCOL:
        if (profile.project_model_id != "granite_4_2_8b" or
                profile.benchmark != "livecodebench" or
                not profile.use_chat_template or
                dict(profile.chat_template_kwargs) != {"enable_thinking": False}):
            raise ValueError("Unsupported Granite input protocol or profile")
        template = getattr(tokenizer, "chat_template", None)
        if not isinstance(template, str) or hashlib.sha256(template.encode()).hexdigest() != GRANITE_TEMPLATE_SHA256:
            raise ValueError("Pinned Granite chat template hash mismatch")
        thinking = tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True, enable_thinking=True)
        if (not rendered.endswith(GRANITE_NONTHINK_TAIL) or
                not thinking.endswith(GRANITE_THINK_TAIL) or
                rendered[:-len(GRANITE_NONTHINK_TAIL)] != thinking[:-len(GRANITE_THINK_TAIL)]):
            raise ValueError("Granite native thinking switch is ineffective or changes messages")
        provenance = {
            "input_protocol": profile.input_protocol,
            "chat_template_sha256": GRANITE_TEMPLATE_SHA256,
            "thinking_prompt_sha256": hashlib.sha256(thinking.encode()).hexdigest(),
            "nonthinking_template_tail": GRANITE_NONTHINK_TAIL,
        }
    elif profile.input_protocol is not None:
        raise ValueError("Unsupported explicit input protocol")
    encoded = tokenizer(rendered, return_tensors="pt", truncation=False,
                        add_special_tokens=not profile.use_chat_template)
    if provenance:
        ids = encoded["input_ids"][0].tolist()
        provenance["input_ids_sha256"] = hashlib.sha256(
            b"".join(int(token).to_bytes(4, "little") for token in ids)
        ).hexdigest()
    return rendered, encoded, provenance


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


def _finish_status(generated: Any, eos_token_id: Any, max_new_tokens: int) -> tuple[str, bool]:
    """Infer a stop reason from the actual returned token IDs, conservatively."""
    ids = generated.tolist()
    if not ids:
        return "unknown", False
    eos_ids = (eos_token_id if isinstance(eos_token_id, (list, tuple, set))
               else [eos_token_id])
    if ids[-1] in {int(value) for value in eos_ids if value is not None}:
        return "eos", False
    if len(ids) >= max_new_tokens:
        return "length", True
    return "unknown", False


def _cuda_devices(model: Any, torch: Any) -> tuple[Any, ...]:
    values: list[Any] = []
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, Mapping):
        values.extend(device_map.values())
    if not values:
        candidate = getattr(model, "device", None)
        if candidate is None:
            candidate = next(model.parameters()).device
        values.append(candidate)
    devices = []
    for value in values:
        try:
            device = torch.device("cuda", value) if isinstance(value, int) else torch.device(value)
        except (RuntimeError, TypeError):
            continue
        if device.type == "cuda" and device not in devices:
            devices.append(device)
    return tuple(devices)


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
    rendered_prompt, encoded, input_provenance = render_and_encode(
        tokenizer, prompt, messages, profile)
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
    cuda_devices: tuple[Any, ...] = ()
    try:
        import torch
        if torch.cuda.is_available():
            cuda_devices = _cuda_devices(model, torch)
            for cuda_device in cuda_devices:
                torch.cuda.synchronize(cuda_device)
                torch.cuda.reset_peak_memory_stats(cuda_device)
    except (AttributeError, ImportError, StopIteration, RuntimeError, TypeError):
        cuda_devices = ()
    started = time.perf_counter()
    output = model.generate(**encoded, **kwargs)
    for cuda_device in cuda_devices:
        torch.cuda.synchronize(cuda_device)
    wall_time = time.perf_counter() - started
    peak_cuda_vram_per_device = {
        str(cuda_device): int(torch.cuda.max_memory_allocated(cuda_device))
        for cuda_device in cuda_devices
    }
    prompt_tokens = input_ids.shape[-1]
    generated = output[0, prompt_tokens:]
    raw = tokenizer.decode(generated, skip_special_tokens=True)
    if input_provenance:
        input_provenance["raw_generation_with_special_tokens"] = tokenizer.decode(
            generated, skip_special_tokens=False)
        input_provenance["generated_token_ids_sha256"] = hashlib.sha256(
            b"".join(int(token).to_bytes(4, "little") for token in generated.tolist())
        ).hexdigest()
    generated_tokens = int(generated.numel())
    finish_reason, truncated = _finish_status(generated, effective_eos, profile.max_new_tokens)
    return {
        "prompt": rendered_prompt,
        "prompt_sha256": hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest(),
        "raw_generation": raw,
        "prompt_tokens": int(prompt_tokens),
        "generated_tokens": generated_tokens,
        "generation_wall_time_seconds": wall_time,
        "generated_tokens_per_second": generated_tokens / wall_time,
        "peak_cuda_vram_per_device_bytes": peak_cuda_vram_per_device or None,
        "peak_cuda_vram_max_device_bytes": (
            max(peak_cuda_vram_per_device.values()) if peak_cuda_vram_per_device else None
        ),
        "truncated": truncated,
        "finish_reason": finish_reason,
        "seed": seed,
        "chat_template_used": profile.use_chat_template,
        "tokenizer_input_protocol": profile.input_protocol or TOKENIZER_INPUT_PROTOCOL,
        "decode_protocol": DECODE_PROTOCOL,
        "chat_template_kwargs": dict(profile.chat_template_kwargs),
        **input_provenance,
        "effective_generation_config": {
            **kwargs,
            **effective_sampling,
            "eos_token_id": effective_eos,
            "pad_token_id": effective_pad,
            "num_beams": effective_num_beams,
            "repetition_penalty": effective_repetition_penalty,
        },
    }
