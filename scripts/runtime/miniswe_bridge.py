#!/usr/bin/env python3
"""Run the pinned mini-swe-agent-plus framework against an OpenAI endpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.run.utils.save import save_traj


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--repo-path", type=Path, required=True)
    value.add_argument("--task-file", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--endpoint", required=True)
    value.add_argument("--model", required=True)
    value.add_argument("--config", type=Path, required=True)
    value.add_argument("--step-limit", type=int, required=True)
    value.add_argument("--cost-limit", type=float, required=True)
    value.add_argument("--command-timeout", type=int, required=True)
    value.add_argument("--max-input-tokens", type=int, required=True)
    value.add_argument("--max-output-tokens", type=int, required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    generation = json.loads(os.environ.get("SRTP_GENERATION_CONFIG", "{}"))
    model_kwargs = {
        **raw.get("model", {}).get("model_kwargs", {}),
        "api_base": args.endpoint.rstrip("/") + "/v1",
        "api_key": os.environ["OPENAI_API_KEY"],
        "max_tokens": args.max_output_tokens,
        **generation,
    }
    model = LitellmModel(model_name=f"openai/{args.model}", model_kwargs=model_kwargs)
    environment = LocalEnvironment(
        cwd=str(args.repo_path.resolve()), timeout=args.command_timeout,
        env=raw.get("environment", {}).get("env", {}),
    )
    agent_config = dict(raw["agent"])
    agent_config.update(step_limit=args.step_limit, cost_limit=args.cost_limit)
    agent = DefaultAgent(model, environment, **agent_config)
    exit_status, result = agent.run(
        args.task_file.read_text(encoding="utf-8"),
        working_dir=str(args.repo_path.resolve()),
    )
    save_traj(
        agent, args.output_dir / "trajectory.json", print_path=False,
        exit_status=exit_status, result=result,
        extra_info={"max_input_tokens": args.max_input_tokens},
    )
    bridge_result = {
        "exit_status": exit_status,
        "submission": result,
        "api_calls": model.n_calls,
        "cost": model.cost,
        "token_usage_status": "not_reported_by_framework",
    }
    (args.output_dir / "bridge_result.json").write_text(
        json.dumps(bridge_result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0 if exit_status == "Submitted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
