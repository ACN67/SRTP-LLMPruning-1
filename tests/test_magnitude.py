"""Magnitude algorithm, target-policy, statistics, and CLI tests."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
    from torch import nn
except ImportError:  # Lightweight config-only environments remain usable.
    torch = None
    nn = None

from src.models.adapters.granite import GraniteAdapter
from src.models.adapters.qwen3 import Qwen3Adapter
from src.pruning import MagnitudePruner, PruningRequest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


if nn is not None:
    class ToyBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(4, 3, bias=True)
            self.norm = nn.LayerNorm(4)


    class ToyBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = nn.Embedding(8, 4)
            self.layers = nn.ModuleList([ToyBlock(), ToyBlock()])
            self.norm = nn.LayerNorm(4)


    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = ToyBackbone()
            self.lm_head = nn.Linear(4, 8, bias=False)


@unittest.skipIf(torch is None, "PyTorch is required for pruning tests")
class MagnitudeAlgorithmTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = ToyModel()
        self.adapter = Qwen3Adapter()
        with torch.no_grad():
            for block in self.model.model.layers:
                block.proj.weight.copy_(
                    torch.tensor(
                        [
                            [0.0, 1.0, 2.0, 3.0],
                            [10.0, 11.0, 12.0, 13.0],
                            [20.0, 21.0, 22.0, 23.0],
                        ]
                    )
                )

    def test_zero_sparsity_changes_nothing(self) -> None:
        before = {name: value.clone() for name, value in self.model.state_dict().items()}
        summary = MagnitudePruner().prune(
            self.model,
            self.adapter,
            PruningRequest("toy", "magnitude", 0.0),
        )
        self.assertEqual(summary.requested_mask_count, 0)
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)

    def test_exact_module_count_exclusions_and_statistics(self) -> None:
        embedding = self.model.model.embed_tokens.weight.detach().clone()
        final_norm = self.model.model.norm.weight.detach().clone()
        lm_head = self.model.lm_head.weight.detach().clone()
        biases = [block.proj.bias.detach().clone() for block in self.model.model.layers]
        shapes = [tuple(block.proj.weight.shape) for block in self.model.model.layers]

        summary = MagnitudePruner().prune(
            self.model,
            self.adapter,
            PruningRequest("toy", "magnitude", 0.5),
        )
        self.assertEqual(summary.number_of_target_modules, 2)
        self.assertEqual(summary.targeted_weights, 24)
        self.assertEqual(summary.requested_mask_count, 12)
        self.assertEqual(summary.preexisting_zeros, 2)
        self.assertEqual(summary.newly_zeroed_weights, 10)
        self.assertEqual(summary.post_pruning_zeros, 12)
        for index, block in enumerate(self.model.model.layers):
            # Six smallest matrix-wide values: all of row 0 and two of row 1.
            self.assertEqual((block.proj.weight == 0).sum(dim=1).tolist(), [4, 2, 0])
            self.assertEqual(tuple(block.proj.weight.shape), shapes[index])
            self.assertTrue(torch.equal(block.proj.bias, biases[index]))
        self.assertTrue(torch.equal(self.model.model.embed_tokens.weight, embedding))
        self.assertTrue(torch.equal(self.model.model.norm.weight, final_norm))
        self.assertTrue(torch.equal(self.model.lm_head.weight, lm_head))
        self.assertEqual(
            sum(item.requested_mask_count for item in summary.per_module),
            summary.requested_mask_count,
        )

    def test_common_ratios_use_per_module_floor(self) -> None:
        for ratio, expected_per_module in ((0.25, 3), (0.5, 6)):
            with self.subTest(ratio=ratio):
                model = ToyModel()
                summary = MagnitudePruner().prune(
                    model,
                    self.adapter,
                    PruningRequest("toy", "magnitude", ratio),
                )
                self.assertTrue(
                    all(
                        item.requested_mask_count == expected_per_module
                        for item in summary.per_module
                    )
                )
                self.assertEqual(summary.requested_mask_count, 2 * expected_per_module)

    def test_ties_select_exact_k_by_flattened_index(self) -> None:
        with torch.no_grad():
            for block in self.model.model.layers:
                block.proj.weight.copy_(
                    torch.tensor(
                        [
                            [1.0, -1.0, 1.0, -1.0],
                            [-1.0, 1.0, -1.0, 1.0],
                            [1.0, -1.0, 1.0, -1.0],
                        ]
                    )
                )
        summary = MagnitudePruner().prune(
            self.model,
            self.adapter,
            PruningRequest("toy", "magnitude", 0.25),
        )
        self.assertTrue(all(item.requested_mask_count == 3 for item in summary.per_module))
        for block in self.model.model.layers:
            flat_mask = (block.proj.weight == 0).reshape(-1)
            self.assertEqual(int(flat_mask.sum().item()), 3)
            self.assertTrue(torch.all(flat_mask[:3]))
            self.assertFalse(bool(flat_mask[3:].any()))

    def test_result_is_deterministic(self) -> None:
        second = ToyModel()
        second.load_state_dict(self.model.state_dict())
        request = PruningRequest("toy", "magnitude", 0.25)
        MagnitudePruner().prune(self.model, self.adapter, request)
        MagnitudePruner().prune(second, self.adapter, request)
        for left, right in zip(self.model.parameters(), second.parameters()):
            self.assertTrue(torch.equal(left, right))

    def test_fake_qwen3_and_granite_use_only_block_linears(self) -> None:
        for adapter in (Qwen3Adapter(), GraniteAdapter()):
            with self.subTest(adapter=adapter.adapter_id):
                model = ToyModel()
                outside_before = {
                    "embedding": model.model.embed_tokens.weight.detach().clone(),
                    "final_norm": model.model.norm.weight.detach().clone(),
                    "lm_head": model.lm_head.weight.detach().clone(),
                }
                summary = MagnitudePruner().prune(
                    model,
                    adapter,
                    PruningRequest("toy", "magnitude", 0.25),
                )
                self.assertEqual(summary.number_of_target_modules, 2)
                self.assertTrue(
                    torch.equal(model.model.embed_tokens.weight, outside_before["embedding"])
                )
                self.assertTrue(
                    torch.equal(model.model.norm.weight, outside_before["final_norm"])
                )
                self.assertTrue(torch.equal(model.lm_head.weight, outside_before["lm_head"]))


class CliValidationTests(unittest.TestCase):
    def test_unimplemented_execute_fails_clearly(self) -> None:
        for pruner in ("sparsegpt", "sleb"):
            with self.subTest(pruner=pruner):
                result = subprocess.run(
                    [
                        sys.executable,
                        "scripts/run_experiment.py",
                        "--model", "klear_agentforge_8b",
                        "--pruner", pruner,
                        "--sparsity", "0.2",
                        "--execute",
                        "--output-dir", "/tmp/not-created",
                    ],
                    cwd=REPOSITORY_ROOT,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 3)
                self.assertIn("not implemented", result.stderr)

    def test_invalid_sparsity_fails(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_experiment.py",
                "--model", "klear_agentforge_8b",
                "--pruner", "magnitude",
                "--sparsity", "1.0",
            ],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("half-open interval", result.stderr)


@unittest.skipIf(torch is None, "PyTorch is required for execute-flow test")
class CliExecuteFlowTests(unittest.TestCase):
    def test_magnitude_execute_lightweight_flow(self) -> None:
        script_path = REPOSITORY_ROOT / "scripts" / "run_experiment.py"
        spec = importlib.util.spec_from_file_location("run_experiment_test", script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        model = ToyModel()
        tokenizer = SimpleNamespace(save_pretrained=lambda path: Path(path, "tokenizer.ok").touch())
        model.save_pretrained = lambda path: Path(path, "model.ok").touch()
        loaded = SimpleNamespace(model=model, tokenizer=tokenizer)
        args = argparse.Namespace(
            model="klear_agentforge_8b",
            pruner="magnitude",
            sparsity=0.25,
            benchmark=None,
            output_dir=None,
            local_path=None,
            cache_dir=None,
            dtype=None,
            device=None,
            device_map=None,
            local_files_only=False,
            overwrite_output_dir=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            args.output_dir = Path(directory)
            fake_model_manifest = {
                "schema_version": 2,
                "timestamp": "2026-01-01T00:00:00+00:00",
                "mode": "magnitude_pruning",
                "status": "completed",
                "project_model_id": args.model,
            }
            with patch.object(module, "load_dense_model", return_value=loaded), patch.object(
                module, "build_model_manifest", return_value=fake_model_manifest
            ):
                manifest = module.execute_experiment(args)
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["pruning"]["pruner"], "magnitude")
            self.assertTrue(Path(directory, "model.ok").exists())
            self.assertTrue(Path(directory, "tokenizer.ok").exists())
            persisted = json.loads(Path(directory, "pruning_manifest.json").read_text())
            self.assertEqual(persisted["schema_version"], 3)


class OutputDirectorySafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        script_path = REPOSITORY_ROOT / "scripts" / "run_experiment.py"
        spec = importlib.util.spec_from_file_location("run_experiment_safety", script_path)
        cls.module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(cls.module)

    def test_repo_output_dir_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the Git repository"):
            self.module._validate_output_directory(
                REPOSITORY_ROOT / "checkpoints" / "unsafe",
                local_path=None,
                overwrite=True,
            )

    def test_output_equal_local_path_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaisesRegex(ValueError, "must not be equal, nested, or overlap"):
                self.module._validate_output_directory(
                    path, local_path=path, overwrite=True
                )

    def test_output_nested_under_local_path_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            local_path = Path(directory)
            with self.assertRaisesRegex(ValueError, "must not be equal, nested, or overlap"):
                self.module._validate_output_directory(
                    local_path / "pruned", local_path=local_path, overwrite=True
                )

    def test_local_path_nested_under_output_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            local_path = output_dir / "dense"
            local_path.mkdir()
            with self.assertRaisesRegex(ValueError, "must not be equal, nested, or overlap"):
                self.module._validate_output_directory(
                    output_dir, local_path=local_path, overwrite=True
                )

    def test_nonempty_output_rejected_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / "existing.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "--output-dir is non-empty"):
                self.module._validate_output_directory(
                    output_dir, local_path=None, overwrite=False
                )

    def test_overwrite_allows_nonempty_external_directory_without_deleting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            existing = output_dir / "existing.txt"
            existing.write_text("keep", encoding="utf-8")
            resolved = self.module._validate_output_directory(
                output_dir, local_path=None, overwrite=True
            )
            self.assertEqual(resolved, output_dir.resolve())
            self.assertEqual(existing.read_text(encoding="utf-8"), "keep")

    def test_overwrite_cannot_bypass_repo_or_local_overlap_rules(self) -> None:
        cases = []
        with tempfile.TemporaryDirectory() as directory:
            local_path = Path(directory)
            cases.append((REPOSITORY_ROOT / "unsafe", None))
            cases.append((local_path / "pruned", local_path))
            for output_dir, source in cases:
                with self.subTest(output_dir=output_dir, local_path=source):
                    with self.assertRaises(ValueError):
                        self.module._validate_output_directory(
                            output_dir, local_path=source, overwrite=True
                        )

    def test_empty_external_output_directory_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            resolved = self.module._validate_output_directory(
                output_dir, local_path=None, overwrite=False
            )
            self.assertEqual(resolved, output_dir.resolve())


if __name__ == "__main__":
    unittest.main()
