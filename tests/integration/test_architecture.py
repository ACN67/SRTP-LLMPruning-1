"""Dependency-direction guard for the frozen repository architecture."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"

FORBIDDEN = {
    "models": {"artifacts", "pruning", "recovery", "direct_evaluation", "agent_runner", "agent_benchmarks"},
    "artifacts": {"pruning", "recovery", "direct_evaluation", "agent_runner", "agent_benchmarks"},
    "pruning": {"recovery", "direct_evaluation", "agent_runner", "agent_benchmarks"},
    "recovery": {"pruning", "direct_evaluation", "agent_runner", "agent_benchmarks"},
    "direct_evaluation": {"pruning", "recovery", "agent_runner", "agent_benchmarks"},
    "agent_runner": {"pruning", "recovery", "direct_evaluation", "agent_benchmarks"},
    "agent_benchmarks": {"pruning", "recovery"},
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


def forbidden_targets(source_package: str, source: str, path: Path) -> list[str]:
    tree = ast.parse(source, filename=str(path))
    forbidden = FORBIDDEN.get(source_package, set())
    result = []
    for node in ast.walk(tree):
        for target in _targets(node, path):
            parts = target.split(".")
            if target.startswith("src.") and len(parts) >= 2 and parts[1] in forbidden:
                result.append(target)
    return result


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
            for target in forbidden_targets(source_package, path.read_text(encoding="utf-8"), path):
                target_package = target.split(".")[1]
                violations.append(
                    f"{path.relative_to(REPOSITORY_ROOT)} imports {target}; "
                    f"{source_package} must not depend on {target_package}"
                )
        self.assertEqual(violations, [], "\n" + "\n".join(violations))

    def test_agent_runner_to_agent_benchmarks_violation_is_detected(self) -> None:
        path = SRC_ROOT / "agent_runner" / "synthetic.py"
        self.assertEqual(
            forbidden_targets("agent_runner", "from src.agent_benchmarks import get_agent_benchmark\n", path),
            ["src.agent_benchmarks"],
        )

    def test_removed_legacy_agent_package_has_no_imports_or_paths(self) -> None:
        old_name = "agent" + "_runtime"
        offenders = []
        for root in (REPOSITORY_ROOT / name for name in ("src", "scripts", "tests", "configs", "third_party")):
            for path in root.rglob("*"):
                if old_name in path.as_posix():
                    offenders.append(str(path.relative_to(REPOSITORY_ROOT)))
                if path.is_file() and path.suffix in {".py", ".yaml", ".md"}:
                    if old_name in path.read_text(encoding="utf-8", errors="ignore"):
                        offenders.append(str(path.relative_to(REPOSITORY_ROOT)))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
