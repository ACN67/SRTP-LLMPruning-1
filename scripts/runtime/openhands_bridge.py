#!/usr/bin/env python3
"""Run pinned OpenHands CodeActAgent against an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from openhands.core.config import LLMConfig, OpenHandsConfig
from openhands.core.main import run_controller
from openhands.events.action import MessageAction


def _bool(value: str) -> bool:
    if value.lower() in {"true", "1", "yes"}:
        return True
    if value.lower() in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--repo-path", type=Path, required=True)
    value.add_argument("--task-file", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--endpoint", required=True)
    value.add_argument("--model", required=True)
    value.add_argument("--max-iterations", type=int, required=True)
    value.add_argument("--command-timeout", type=int, required=True)
    value.add_argument("--runtime", choices=("local", "docker"), required=True)
    value.add_argument("--native-tool-calling", type=_bool, required=True)
    value.add_argument("--mode", required=True)
    value.add_argument("--instruction-template", required=True)
    value.add_argument("--use-hint-text", type=_bool, required=True)
    value.add_argument("--enable-plan-mode", type=_bool, required=True)
    value.add_argument("--add-icl-example", type=_bool, required=True)
    value.add_argument("--max-input-tokens", type=int, required=True)
    value.add_argument("--max-output-tokens", type=int, required=True)
    return value


def _jsonable(value: object) -> object:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "value"):
        return getattr(value, "value")
    return str(value)


def main() -> int:
    args = parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    request = json.loads(os.environ.get("SRTP_GENERATION_CONFIG", "{}"))
    llm = LLMConfig(
        model=f"openai/{args.model}", api_key=os.environ["OPENAI_API_KEY"],
        base_url=args.endpoint.rstrip("/") + "/v1",
        temperature=float(request.get("temperature", 0.0)),
        top_p=float(request.get("top_p", 1.0)),
        max_input_tokens=args.max_input_tokens,
        max_output_tokens=args.max_output_tokens,
        drop_params=True, native_tool_calling=args.native_tool_calling,
    )
    config = OpenHandsConfig(
        default_agent="CodeActAgent", runtime=args.runtime, file_store="memory",
        workspace_base=str(args.repo_path.resolve()),
        workspace_mount_path_in_sandbox=str(args.repo_path.resolve()),
        max_iterations=args.max_iterations, enable_browser=False,
        save_trajectory_path=str((args.output_dir / "trajectory.json").resolve()),
    )
    config.sandbox.timeout = args.command_timeout
    config.sandbox.trusted_dirs = [str(args.repo_path.resolve())]
    config.set_llm_config(llm)
    state = asyncio.run(run_controller(
        config, MessageAction(content=args.task_file.read_text(encoding="utf-8")),
        headless_mode=True,
    ))
    agent_state = getattr(getattr(state, "agent_state", None), "value", "error")
    result = {
        "agent_state": agent_state,
        "iteration": getattr(state, "iteration", 0) if state else 0,
        "metrics": _jsonable(getattr(state, "metrics", {})) if state else {},
        "token_usage_status": "reported_in_metrics" if state and getattr(state, "metrics", None) else "not_reported_by_framework",
        "task_recipe": {
            "mode": args.mode,
            "instruction_template": args.instruction_template,
            "use_hint_text": args.use_hint_text,
            "enable_plan_mode": args.enable_plan_mode,
            "add_in_context_learning_example": args.add_icl_example,
            "application_status": "recorded_for_benchmark_provisioning; local RepositoryTask text is passed directly",
        },
    }
    (args.output_dir / "bridge_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0 if agent_state == "finished" else 1


if __name__ == "__main__":
    raise SystemExit(main())
