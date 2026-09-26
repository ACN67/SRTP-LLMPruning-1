"""Pinned LiveCodeBench v6 asset, prompt, extraction, and test decoding."""

from __future__ import annotations

import base64
import hashlib
import json
import pickle
import zlib
from pathlib import Path
from typing import Any

from .base import BenchmarkTask


LCB_CODE_COMMIT = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
LCB_DATASET_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
LCB_V6_SHA256 = "bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5"
LCB_V6_TASK_COUNT = 175

SYSTEM_MESSAGE_GENERIC = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program that "
    "matches the specification and passes all tests."
)
FORMAT_WITH_STARTER = (
    "You will use the following starter code to write the solution to the "
    "problem and enclose your code within delimiters."
)
FORMAT_WITHOUT_STARTER = (
    "Read the inputs from stdin solve the problem and write the answer to stdout "
    "(do not directly test on the sample inputs). Enclose your code within "
    "delimiters as follows. Ensure that when the python program runs, it reads "
    "the inputs, runs the algorithm and writes output to STDOUT."
)


def decode_test_cases(value: Any, *, verified_asset: bool) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [dict(item) for item in value]
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        if not verified_asset:
            raise ValueError(
                "Compressed LiveCodeBench private tests require a verified pinned asset"
            )
        try:
            unpacked = pickle.loads(zlib.decompress(base64.b64decode(value.encode("utf-8"))))
            decoded = json.loads(unpacked)
        except Exception as error:
            raise ValueError("Invalid compressed LiveCodeBench private tests") from error
    if not isinstance(decoded, list):
        raise ValueError("LiveCodeBench test cases must decode to a list")
    return [dict(item) for item in decoded]


def _is_json_encoded(value: Any) -> bool:
    try:
        json.loads(value)
        return True
    except (TypeError, json.JSONDecodeError):
        return False


def load_verified_v6(path: Path, expected_sha256: str = LCB_V6_SHA256) -> list[BenchmarkTask]:
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"LiveCodeBench v6 SHA256 mismatch: {actual}; expected {expected_sha256}"
        )
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]
    if len(rows) != LCB_V6_TASK_COUNT:
        raise ValueError(f"LiveCodeBench v6 must contain {LCB_V6_TASK_COUNT} rows")
    tasks = []
    seen = set()
    for row in rows:
        question_id = str(row["question_id"])
        if question_id in seen:
            raise ValueError(f"Duplicate LiveCodeBench question_id: {question_id}")
        seen.add(question_id)
        normalized = dict(row)
        normalized["private_test_encoding"] = (
            "json" if _is_json_encoded(row["private_test_cases"]) else "base64_zlib_pickle_json"
        )
        normalized["public_test_cases"] = decode_test_cases(
            row["public_test_cases"], verified_asset=True
        )
        normalized["private_test_cases"] = decode_test_cases(
            row["private_test_cases"], verified_asset=True
        )
        metadata = row.get("metadata", {})
        normalized["metadata"] = json.loads(metadata) if isinstance(metadata, str) else metadata
        normalized["asset_sha256"] = actual
        tasks.append(BenchmarkTask("livecodebench", question_id, row["question_content"], normalized))
    return tasks


def generic_question_prompt(question: str, starter_code: str) -> str:
    prompt = f"### Question:\n{question}\n\n"
    if starter_code:
        prompt += f"### Format: {FORMAT_WITH_STARTER}\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {FORMAT_WITHOUT_STARTER}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    return prompt + "### Answer: (use the provided format with backticks)\n\n"


def chat_messages(question: str, starter_code: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_MESSAGE_GENERIC},
        {"role": "user", "content": generic_question_prompt(question, starter_code)},
    ]


def extract_lcb_code(model_output: str) -> str:
    """Pinned Generic/Chat extraction: return the final fenced block or empty."""

    lines = model_output.split("\n")
    fence_indices = [index for index, line in enumerate(lines) if "```" in line]
    if len(fence_indices) < 2:
        return ""
    return "\n".join(lines[fence_indices[-2] + 1 : fence_indices[-1]])


def evaluation_sample(task: BenchmarkTask) -> dict[str, str]:
    tests = list(task.data["public_test_cases"]) + list(task.data["private_test_cases"])
    metadata = task.data.get("metadata") or {}
    return {
        "input_output": json.dumps({
            "inputs": [test["input"] for test in tests],
            "outputs": [test["output"] for test in tests],
            "fn_name": metadata.get("func_name"),
        })
    }


def audit_v6(path: Path) -> dict[str, Any]:
    tasks = load_verified_v6(path)
    compressed = 0
    functional = 0
    stdin = 0
    for task in tasks:
        if task.data["private_test_encoding"] == "base64_zlib_pickle_json":
            compressed += 1
        if (task.data.get("metadata") or {}).get("func_name"):
            functional += 1
        else:
            stdin += 1
    return {
        "rows": len(tasks), "unique_question_ids": len({task.task_id for task in tasks}),
        "private_compressed_rows_handled": compressed,
        "functional": functional, "stdin": stdin,
        "sha256": LCB_V6_SHA256,
    }
