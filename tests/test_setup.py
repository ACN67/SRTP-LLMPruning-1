"""Offline setup, verified transport, snapshot, calibration, and preflight tests."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.calibration_assets import C4_ASSET, WIKITEXT_ASSET, load_local_c4, load_local_wikitext2
from src.downloads import file_hash, verified_download, verify_file
from src.models import load_snapshot_manifest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def tiny_manifest():
    return {
        "schema_version": 1,
        "project_model_id": "klear_agentforge_8b",
        "canonical_hf_repo": "Kwai-Klear/Klear-AgentForge-8B",
        "canonical_hf_revision": "fa3d41e92e9ce7a5b4a52a3e7439aa00521f40c9",
        "domestic_modelscope_repo": "Kwai-Klear/Klear-AgentForge-8B",
        "required_runtime_files": [
            {"path": "config.json", "size": 2, "hash_type": "git_blob_sha1", "expected_hash": "x"},
            {"path": "model.safetensors.index.json", "size": 2, "hash_type": "git_blob_sha1", "expected_hash": "y"},
        ],
    }


class VerifiedDownloadTests(unittest.TestCase):
    def test_sha256_and_git_blob_sha1(self):
        payload = b"hello\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "file")
            path.write_bytes(payload)
            self.assertEqual(file_hash(path, "sha256"), hashlib.sha256(payload).hexdigest())
            expected = hashlib.sha1(b"blob 6\0" + payload).hexdigest()
            self.assertEqual(file_hash(path, "git_blob_sha1"), expected)

    def test_size_and_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "file")
            path.write_bytes(b"abc")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                verify_file(path, size=4, hash_type="sha256", expected_hash="0" * 64)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_file(path, size=3, hash_type="sha256", expected_hash="0" * 64)

    def test_stream_part_atomic_and_valid_skip(self):
        payload = b"verified bytes"
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory, "nested", "file")
            with patch("src.downloads.urllib.request.urlopen", return_value=io.BytesIO(payload)) as open_url:
                result = verified_download("https://example.test/file", target, size=len(payload),
                                           hash_type="sha256", expected_hash=digest)
            self.assertTrue(result["downloaded"])
            self.assertEqual(target.read_bytes(), payload)
            self.assertFalse(Path(str(target) + ".part").exists())
            with patch("src.downloads.urllib.request.urlopen") as open_url:
                result = verified_download("https://example.test/file", target, size=len(payload),
                                           hash_type="sha256", expected_hash=digest)
                open_url.assert_not_called()
            self.assertFalse(result["downloaded"])

    def test_failure_removes_part_and_preserves_no_target(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "src.downloads.urllib.request.urlopen", return_value=io.BytesIO(b"bad")
        ):
            target = Path(directory, "file")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                verified_download("https://example.test/file", target, size=4,
                                  hash_type="sha256", expected_hash="0" * 64)
            self.assertFalse(target.exists())
            self.assertFalse(Path(str(target) + ".part").exists())


class ModelTransportTests(unittest.TestCase):
    def setUp(self):
        self.module = load_script("download_models_test", "scripts/setup/download_models.py")

    def test_modelscope_and_official_url_mapping(self):
        manifest = tiny_manifest()
        domestic = self.module.runtime_file_url(manifest, "config.json", download_source="domestic")
        official = self.module.runtime_file_url(manifest, "config.json", download_source="official")
        self.assertEqual(domestic, "https://modelscope.cn/models/Kwai-Klear/Klear-AgentForge-8B/resolve/master/config.json")
        self.assertIn("/resolve/fa3d41e92e9ce7a5b4a52a3e7439aa00521f40c9/config.json", official)

    def test_downloader_has_no_verify_only_interface(self):
        destinations = {action.dest for action in self.module._parser()._actions}
        self.assertNotIn("local_files_only", destinations)
        self.assertNotIn("--local-files-only", self.module._parser().format_help())

    def test_default_domestic_writes_v2_sidecar_after_complete_success(self):
        manifest = tiny_manifest()
        with tempfile.TemporaryDirectory() as directory, patch.object(
            self.module, "load_snapshot_manifest", return_value=manifest
        ), patch.object(self.module, "verified_download") as download, patch.object(
            self.module, "verify_runtime_snapshot", return_value=manifest["required_runtime_files"]
        ), patch.object(self.module, "snapshot_manifest_sha256", return_value="m" * 64):
            self.module.download_model("klear_agentforge_8b", Path(directory))
            sidecar = json.loads(Path(directory, "klear_agentforge_8b", ".srtp_model_source.json").read_text())
        self.assertEqual(download.call_count, 2)
        self.assertTrue(all("modelscope.cn" in call.args[0] for call in download.call_args_list))
        self.assertEqual(sidecar["download_transport"], "modelscope_resolve")
        self.assertEqual(sidecar["canonical_hf_revision"], manifest["canonical_hf_revision"])

    def test_partial_failure_never_writes_sidecar_or_falls_back(self):
        manifest = tiny_manifest()
        with tempfile.TemporaryDirectory() as directory, patch.object(
            self.module, "load_snapshot_manifest", return_value=manifest
        ), patch.object(self.module, "verified_download", side_effect=[{}, RuntimeError("stop")]) as download:
            with self.assertRaisesRegex(RuntimeError, "stop"):
                self.module.download_model("klear_agentforge_8b", Path(directory))
            self.assertFalse(Path(directory, "klear_agentforge_8b", ".srtp_model_source.json").exists())
        self.assertTrue(all("huggingface.co" not in call.args[0] for call in download.call_args_list))

    def test_small_smoke_does_not_write_success_sidecar(self):
        manifest = tiny_manifest()
        with tempfile.TemporaryDirectory() as directory, patch.object(
            self.module, "load_snapshot_manifest", return_value=manifest
        ), patch.object(self.module, "verified_download"), patch.object(
            self.module, "verify_runtime_snapshot", return_value=manifest["required_runtime_files"]
        ):
            self.module.download_model("klear_agentforge_8b", Path(directory),
                                       include_files=self.module.SMOKE_FILES)
            self.assertFalse(Path(directory, "klear_agentforge_8b", ".srtp_model_source.json").exists())

    def test_repository_manifests_match_model_registry_and_include_index_shards(self):
        for model_id in ("klear_agentforge_8b", "granite_4_2_8b"):
            manifest = load_snapshot_manifest(model_id)
            paths = {item["path"] for item in manifest["required_runtime_files"]}
            self.assertIn("config.json", paths)
            self.assertIn("generation_config.json", paths)
            self.assertIn("tokenizer_config.json", paths)
            self.assertIn("model.safetensors.index.json", paths)
            self.assertEqual(sum(path.endswith(".safetensors") for path in paths), 4)

    def test_snapshot_verifier_reuses_manifest_and_sidecar_identity(self):
        module = load_script("verify_snapshot_manifest_test", "scripts/setup/verify_model_snapshot.py")
        from src.models import load_model_spec
        spec = load_model_spec("klear_agentforge_8b")
        manifest = {
            **tiny_manifest(),
            "required_runtime_files": [
                {"path": "model.safetensors.index.json", "size": 1,
                 "hash_type": "git_blob_sha1", "expected_hash": "i"},
                {"path": "model-00001-of-00001.safetensors", "size": 1,
                 "hash_type": "sha256", "expected_hash": "s"},
            ],
        }
        verified = manifest["required_runtime_files"]
        config = SimpleNamespace(
            model_type=spec.model_type, architectures=[spec.expected_model_class],
            num_hidden_layers=spec.expected_num_hidden_layers,
            hidden_size=spec.expected_hidden_size,
            intermediate_size=spec.expected_intermediate_size,
            num_attention_heads=spec.expected_num_attention_heads,
            num_key_value_heads=spec.expected_num_key_value_heads,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"x": "model-00001-of-00001.safetensors"}
            }))
            sidecar = {
                "schema_version": 2, "project_model_id": "klear_agentforge_8b",
                "canonical_hf_repo": manifest["canonical_hf_repo"],
                "canonical_hf_revision": manifest["canonical_hf_revision"],
                "runtime_snapshot_manifest_sha256": "m" * 64,
                "verified_runtime_files": verified,
                "download_endpoint": "https://modelscope.cn/models",
                "download_transport": "modelscope_resolve",
            }
            (root / ".srtp_model_source.json").write_text(json.dumps(sidecar))
            with patch.object(module, "load_snapshot_manifest", return_value=manifest), patch.object(
                module, "verify_runtime_snapshot", return_value=verified
            ) as runtime_verify, patch.object(
                module, "snapshot_manifest_sha256", return_value="m" * 64
            ), patch("transformers.AutoConfig.from_pretrained", return_value=config), patch(
                "transformers.AutoTokenizer.from_pretrained", return_value=object()
            ):
                result = module.verify_snapshot("klear_agentforge_8b", root)
        runtime_verify.assert_called_once_with("klear_agentforge_8b", root.resolve())
        self.assertEqual(result["referenced_weight_shards"], 1)


class CalibrationAssetTests(unittest.TestCase):
    def test_fixed_raw_asset_identity_and_paths(self):
        self.assertEqual((C4_ASSET["size"], C4_ASSET["sha256"]),
                         (319308785, "8ef8d75b0e045dec4aa5123a671b4564466b0707086a7ed1ba8721626dfffbc9"))
        self.assertEqual((WIKITEXT_ASSET["size"], WIKITEXT_ASSET["sha256"]),
                         (6357543, "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7"))
        self.assertTrue(C4_ASSET["relative_path"].endswith(".json.gz"))
        self.assertTrue(WIKITEXT_ASSET["relative_path"].endswith(".parquet"))

    def test_prefetch_uses_direct_exact_resolve_urls(self):
        module = load_script("prefetch_calibration_test", "scripts/setup/prefetch_assets.py")
        with tempfile.TemporaryDirectory() as directory, patch.object(module, "verified_download") as download:
            result = module._calibration(Path(directory))
        self.assertEqual(download.call_count, 2)
        urls = [call.args[0] for call in download.call_args_list]
        self.assertTrue(any(C4_ASSET["revision"] in url and C4_ASSET["file"] in url for url in urls))
        self.assertTrue(any(WIKITEXT_ASSET["revision"] in url and WIKITEXT_ASSET["file"] in url for url in urls))
        self.assertNotIn("dataset_fingerprint", json.dumps(result))

    def test_local_adapters_use_json_and_parquet_without_remote_repo(self):
        calls = []
        def loader(kind, **kwargs):
            calls.append((kind, kwargs))
            return [{"text": "first"}, {"text": "second"}]
        fake = SimpleNamespace(load_dataset=loader)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"datasets": fake}), patch(
            "src.calibration_assets.verify_file"
        ):
            root = Path(directory)
            c4 = load_local_c4(root)
            wiki = load_local_wikitext2(root)
        self.assertEqual([item["text"] for item in c4], ["first", "second"])
        self.assertEqual([item["text"] for item in wiki], ["first", "second"])
        self.assertEqual([call[0] for call in calls], ["json", "parquet"])
        self.assertTrue(all(call[1]["split"] == "train" for call in calls))

    def test_verify_assets_covers_calibration_hash_and_structure(self):
        module = load_script("verify_assets_calibration_test", "scripts/setup/verify_assets.py")
        c4 = [{"text": "x"}]
        wiki = [{"text": "x"}] * 36718
        with patch.object(module, "verify_file") as verify, patch.object(
            module, "load_local_c4", return_value=c4
        ), patch.object(module, "load_local_wikitext2", return_value=wiki):
            result = module.verify_calibration_assets(Path("/data/datasets"))
        self.assertEqual(verify.call_count, 2)
        self.assertEqual(result["wikitext2_rows"], 36718)
        self.assertEqual(result["c4"]["sha256"], C4_ASSET["sha256"])


class SetupAndPreflightTests(unittest.TestCase):
    def test_docker_and_dependency_baseline_is_pinned(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        requirements = set((ROOT / "requirements.txt").read_text().splitlines())
        self.assertIn("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime@sha256:", dockerfile)
        self.assertIn("ARG SRTP_PROJECT_GIT_COMMIT=unknown", dockerfile)
        self.assertIn("ARG SRTP_SOURCE_DIRTY=unknown", dockerfile)
        build_script = (ROOT / "scripts" / "setup" / "build_image.sh").read_text()
        self.assertIn("--build-arg SRTP_PROJECT_GIT_COMMIT=", build_script)
        self.assertIn("--build-arg SRTP_SOURCE_DIRTY=", build_script)
        self.assertIn("datasets==5.0.1", requirements)
        self.assertNotIn("vllm", "\n".join(requirements).lower())

    def test_software_preflight_checks_all_registries_without_data(self):
        module = load_script("preflight_test", "scripts/setup/preflight.py")
        with patch.object(module, "version", side_effect=lambda package: module.EXPECTED_VERSIONS[package]), patch.object(
            module, "import_module", return_value=object()
        ):
            result = module.software_check()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["benchmarks"], ["humaneval", "livecodebench", "mbpp"])

    def test_path_check_requires_writable_contract(self):
        module = load_script("preflight_paths_test", "scripts/setup/preflight.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in module.DATA_DIRECTORIES:
                (root / name).mkdir()
            self.assertEqual(module.paths_check(root)["status"], "pass")


if __name__ == "__main__":
    unittest.main()
