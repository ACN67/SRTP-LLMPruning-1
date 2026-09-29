#!/usr/bin/env python3
"""Idempotently create isolated pinned vLLM and Agent environments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
PINS = yaml.safe_load((ROOT / "configs/agent_runner/runtime_pins.yaml").read_text(encoding="utf-8"))


def run(*args: str) -> None:
    subprocess.run(args, check=True, cwd=ROOT)


def python_in(venv: Path) -> Path:
    return venv / "bin" / "python"


def ensure_venv(path: Path) -> None:
    if not python_in(path).is_file():
        run(sys.executable, "-m", "venv", str(path))


def ensure_checkout(name: str, destination: Path) -> None:
    pin = PINS[name]
    if not (destination / ".git").is_dir():
        destination.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--filter=blob:none", pin["repository"], str(destination))
    run("git", "-C", str(destination), "fetch", "--depth", "1", "origin", pin["revision"])
    current = subprocess.run(
        ("git", "-C", str(destination), "rev-parse", "HEAD"), check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if current != pin["revision"]:
        run("git", "-C", str(destination), "checkout", "--detach", pin["revision"])


def install(component: str, index_url: str | None) -> None:
    pip_extra = ("--index-url", index_url) if index_url else ()
    if component == "vllm":
        venv = ROOT / ".venv-vllm"
        ensure_venv(venv)
        run(str(python_in(venv)), "-m", "pip", "install", *pip_extra, f"vllm=={PINS['vllm']['version']}")
        return
    checkout_name = "mini-swe-agent-plus" if component == "mini_swe_agent_plus" else "OpenHands-0.53.0"
    checkout = ROOT / ".agent_runner" / checkout_name
    ensure_checkout(component, checkout)
    venv_name = ".venv-miniswe" if component == "mini_swe_agent_plus" else ".venv-openhands"
    venv = ROOT / venv_name
    ensure_venv(venv)
    run(str(python_in(venv)), "-m", "pip", "install", *pip_extra, "-e", str(checkout))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component", action="append",
        choices=("vllm", "mini_swe_agent_plus", "openhands"),
        help="Repeat to install selected components; default installs all",
    )
    parser.add_argument("--index-url", help="Optional domestic or private PyPI mirror")
    args = parser.parse_args()
    for component in args.component or ("vllm", "mini_swe_agent_plus", "openhands"):
        install(component, args.index_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
