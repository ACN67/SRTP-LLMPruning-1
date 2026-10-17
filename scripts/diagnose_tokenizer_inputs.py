#!/usr/bin/env python3
"""Audit rendered chat token IDs with and without added special tokens."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.direct_evaluation import get_benchmark, load_evaluation_profile  # noqa: E402


def digest(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", type=Path, default=Path("/data/models"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    rows = []
    for model in ("klear_agentforge_8b", "swe_lego_qwen3_8b", "granite_4_2_8b"):
        tokenizer = AutoTokenizer.from_pretrained(args.models_root / model, local_files_only=True)
        for name in ("humaneval", "mbpp", "livecodebench"):
            benchmark = get_benchmark(name)
            task = benchmark.load_tasks()[0]
            profile = load_evaluation_profile(model, name)
            rendered = tokenizer.apply_chat_template(
                benchmark.build_messages(task), tokenize=False, add_generation_prompt=True,
                **dict(profile.chat_template_kwargs),
            )
            old_ids = tokenizer(rendered, add_special_tokens=True)["input_ids"]
            new_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
            rows.append({
                "model": model, "benchmark": name, "task_id": task.task_id,
                "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                "old_length": len(old_ids), "new_length": len(new_ids),
                "old_ids_sha256": digest(old_ids), "new_ids_sha256": digest(new_ids),
                "same_token_ids": old_ids == new_ids,
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(rows), "identical": sum(row["same_token_ids"] for row in rows)}))


if __name__ == "__main__":
    main()
