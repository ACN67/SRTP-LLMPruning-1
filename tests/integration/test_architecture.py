"""Dependency-direction guard for the frozen repository architecture."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"

FORBIDDEN = {
    "models": {"artifacts", "pruning", "recovery", "direct_evaluation", "agent_runtime", "agent_benchmarks"},
    "artifacts": {"pruning", "recovery", "direct_evaluation", "agent_runtime", "agent_benchmarks"},
    "pruning": {"recovery", "direct_evaluation", "agent_runtime", "agent_benchmarks"},
    "recovery": {"pruning", "direct_evaluation", "agent_runtime", "agent_benchmarks"},
    "direct_evaluation": {"pruning", "recovery", "agent_runtime", "agent_benchmarks"},
    "agent_runtime": {"pruning", "recovery", "direct_evaluation", "agent_benchmarks"},
}


def _module_package(path: Path) -> list[str]:
    relative = path.relative_to(REPOSITORY_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        return parts[:-1]
    return parts[:-1]


def _targets(node: ast.AST, path: Path) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)
    if not isinstance(node, ast.ImportFrom):
        return ()
    if node.level == 0:
        return (node.module,) if node.module else ()
    package = _module_package(path)
    keep = len(package) - (node.level - 1)
    if keep < 0:
        return ()
    base = package[:keep]
    suffix = node.module.split(".") if node.module else []
    return (".".join(base + suffix),)


class ArchitectureDependencyTests(unittest.TestCase):
    def test_forbidden_reverse_dependencies_are_absent(self) -> None:
        violations: list[str] = []
        for path in sorted(SRC_ROOT.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            source_package = path.relative_to(SRC_ROOT).parts[0]
            forbidden = FORBIDDEN.get(source_package, set())
            if not forbidden:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                for target in _targets(node, path):
                    if not target.startswith("src."):
                        continue
                    parts = target.split(".")
                    if len(parts) < 2:
                        continue
                    target_package = parts[1]
                    if target_package in forbidden:
                        violations.append(
                            f"{path.relative_to(REPOSITORY_ROOT)} imports {target}; "
                            f"{source_package} must not depend on {target_package}"
                        )
        self.assertEqual(violations, [], "\n" + "\n".join(violations))


if __name__ == "__main__":
    unittest.main()
