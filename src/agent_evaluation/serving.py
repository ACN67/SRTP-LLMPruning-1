"""Artifact validation and managed vLLM server lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from src.models import get_model_adapter, load_model_spec
from src.models.artifacts import (
    REDUCED_DEPTH_PRUNERS, SAME_DEPTH_PRUNERS, _read_manifest,
    _validate_manifest_identity,
)

from .base import AgentSystemSpec, ServingSpec


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
GRANITE_PARSER = REPOSITORY_ROOT / "third_party" / "runtime" / "granite_thinking_parser.py"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ArtifactServingSpec:
    path: str
    kind: str
    project_model_id: str
    model_type: str
    num_hidden_layers: int
    pruning_method: str
    pruning_provenance: Mapping[str, Any]
    config_sha256: str
    inventory_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_artifact(system: AgentSystemSpec, artifact_path: Path) -> ArtifactServingSpec:
    """Fail fast on local HF artifact identity without loading model weights."""

    root = artifact_path.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Artifact directory does not exist: {root}")
    config_path = root / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Artifact is missing config.json: {root}")
    weight_files = [
        item for pattern in ("*.safetensors", "*.bin", "*.safetensors.index.json", "*.bin.index.json")
        for item in root.glob(pattern)
    ]
    if not weight_files:
        raise ValueError(f"Artifact contains no Hugging Face weight files: {root}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_spec = load_model_spec(system.project_model_id)
    expected_config = {
        "model_type": model_spec.model_type,
        "hidden_size": model_spec.expected_hidden_size,
        "intermediate_size": model_spec.expected_intermediate_size,
        "num_attention_heads": model_spec.expected_num_attention_heads,
        "num_key_value_heads": model_spec.expected_num_key_value_heads,
    }
    mismatches = [
        f"{key}={config.get(key)!r}, expected {expected!r}"
        for key, expected in expected_config.items() if config.get(key) != expected
    ]
    if mismatches:
        raise ValueError("Artifact config mismatch: " + "; ".join(mismatches))
    depth = int(config.get("num_hidden_layers", 0))
    if depth <= 0:
        raise ValueError("Artifact config has no positive num_hidden_layers")

    manifest_path = root / "pruning_manifest.json"
    pruning: Mapping[str, Any] = {}
    method = "dense"
    kind = "dense"
    if manifest_path.is_file():
        manifest = _read_manifest(root)
        validated_method, validated_depth = _validate_manifest_identity(
            manifest, model_spec, get_model_adapter(model_spec)
        )
        pruning = manifest.get("pruning", {})
        method = validated_method
        kind = "pruned"
        recorded = validated_depth
        if recorded != depth:
            raise ValueError(
                f"Pruned artifact manifest depth {recorded} does not match config depth {depth}"
            )
        if method in SAME_DEPTH_PRUNERS and depth != model_spec.expected_num_hidden_layers:
            raise ValueError("Same-depth pruning artifact unexpectedly changed model depth")
        if method in REDUCED_DEPTH_PRUNERS and depth >= model_spec.expected_num_hidden_layers:
            raise ValueError("Reduced-depth pruning artifact did not reduce model depth")
    elif depth != model_spec.expected_num_hidden_layers:
        raise ValueError("Reduced-depth artifact requires pruning_manifest.json")

    inventory = []
    for item in sorted(path for path in root.rglob("*") if path.is_file() and not path.is_symlink()):
        relative = item.relative_to(root).as_posix()
        entry: dict[str, Any] = {
            "path": relative, "size": item.stat().st_size, "sha256": _sha256(item),
        }
        inventory.append(entry)
    inventory_hash = hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ArtifactServingSpec(
        path=str(root), kind=kind, project_model_id=system.project_model_id,
        model_type=model_spec.model_type, num_hidden_layers=depth,
        pruning_method=method, pruning_provenance=pruning,
        config_sha256=_sha256(config_path), inventory_sha256=inventory_hash,
    )


def resolve_granite_parser(system: AgentSystemSpec, explicit_path: Path | None = None) -> Path | None:
    if system.project_model_id != "granite_4_2_8b":
        return None
    path = (explicit_path or GRANITE_PARSER).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Granite reasoning parser plugin is missing: {path}")
    text = path.read_text(encoding="utf-8")
    if 'register_module("granite_thinking_parser")' not in text:
        raise ValueError("Granite parser plugin does not register granite_thinking_parser")
    return path


def build_vllm_command(
    serving: ServingSpec,
    artifact: ArtifactServingSpec,
    *,
    parser_plugin: Path | None = None,
) -> tuple[str, ...]:
    executable = Path(serving.executable)
    if not executable.is_absolute():
        executable = REPOSITORY_ROOT / executable
    command = [
        str(executable), "serve", artifact.path,
        "--served-model-name", serving.served_model_name,
        "--host", serving.host, "--port", str(serving.port),
        "--tensor-parallel-size", str(serving.tensor_parallel_size),
        "--gpu-memory-utilization", str(serving.gpu_memory_utilization),
        "--max-model-len", str(serving.max_model_len),
        "--max-num-seqs", str(serving.max_num_seqs),
        "--dtype", serving.dtype, "--api-key", serving.api_key,
    ]
    command.extend(serving.extra_args)
    if parser_plugin is not None:
        command.extend(("--reasoning-parser-plugin", str(parser_plugin)))
    return tuple(command)


def _connect_host(host: str) -> str:
    return "127.0.0.1" if host in {"0.0.0.0", "::"} else host


def port_is_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((_connect_host(host), port))
        except OSError:
            return False
    return True


def query_models(endpoint: str, api_key: str, timeout: float = 2.0) -> tuple[str, ...]:
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return tuple(str(item["id"]) for item in payload.get("data", ()))


class ServingStartupError(RuntimeError):
    pass


class VLLMServer:
    """Own a vLLM subprocess or validate/reuse an external endpoint."""

    def __init__(
        self,
        system: AgentSystemSpec,
        artifact: ArtifactServingSpec,
        output_dir: Path,
        *,
        serving: ServingSpec | None = None,
        endpoint: str | None = None,
        parser_plugin: Path | None = None,
        startup_timeout: float = 600.0,
        terminate_timeout: float = 20.0,
        poll_interval: float = 0.25,
        popen_factory: Callable[..., Any] = subprocess.Popen,
        models_query: Callable[[str, str, float], tuple[str, ...]] = query_models,
    ) -> None:
        self.system = system
        self.artifact = artifact
        self.serving = serving or system.serving
        self.output_dir = output_dir.expanduser().resolve()
        self.external = endpoint is not None
        self.endpoint = endpoint.rstrip("/") if endpoint else f"http://{_connect_host(self.serving.host)}:{self.serving.port}"
        self.parser_plugin = parser_plugin
        self.startup_timeout = startup_timeout
        self.terminate_timeout = terminate_timeout
        self.poll_interval = poll_interval
        self.popen_factory = popen_factory
        self.models_query = models_query
        self.process: Any | None = None
        self.started_at: str | None = None
        self.ready_at: str | None = None
        self.stopped_at: str | None = None
        self.stdout_path = self.output_dir / "vllm.stdout.log"
        self.stderr_path = self.output_dir / "vllm.stderr.log"
        self._stdout: Any | None = None
        self._stderr: Any | None = None

    @property
    def command(self) -> tuple[str, ...]:
        return build_vllm_command(self.serving, self.artifact, parser_plugin=self.parser_plugin)

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        last_error = "not queried"
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise ServingStartupError(
                    f"vLLM exited before readiness with code {self.process.returncode}"
                )
            try:
                models = self.models_query(self.endpoint, self.serving.api_key, 2.0)
                if self.serving.served_model_name not in models:
                    raise ServingStartupError(
                        f"Endpoint model mismatch: expected {self.serving.served_model_name!r}, got {models}"
                    )
                self.ready_at = _utc_now()
                return
            except ServingStartupError:
                raise
            except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError) as error:
                last_error = f"{type(error).__name__}: {error}"
                time.sleep(self.poll_interval)
        raise ServingStartupError(
            f"Timed out after {self.startup_timeout}s waiting for {self.endpoint}/v1/models; last_error={last_error}"
        )

    def start(self) -> "VLLMServer":
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.started_at = _utc_now()
        if not self.external:
            if not port_is_available(self.serving.host, self.serving.port):
                raise ServingStartupError(
                    f"Port {self.serving.port} is already in use on {_connect_host(self.serving.host)}"
                )
            executable = Path(self.command[0])
            if self.popen_factory is subprocess.Popen and not executable.is_file():
                raise FileNotFoundError(f"Pinned vLLM executable is missing: {executable}")
            self._stdout = self.stdout_path.open("wb")
            self._stderr = self.stderr_path.open("wb")
            try:
                self.process = self.popen_factory(
                    self.command, cwd=REPOSITORY_ROOT, stdout=self._stdout,
                    stderr=self._stderr, start_new_session=True,
                )
            except Exception:
                self._close_logs()
                raise
        try:
            self._wait_ready()
        except Exception:
            self.stop()
            raise
        return self

    def _close_logs(self) -> None:
        for handle in (self._stdout, self._stderr):
            if handle is not None and not handle.closed:
                handle.close()

    def stop(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            if isinstance(process, subprocess.Popen) and os.name == "posix":
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=self.terminate_timeout)
            except subprocess.TimeoutExpired:
                if isinstance(process, subprocess.Popen) and os.name == "posix":
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                else:
                    process.kill()
                process.wait(timeout=self.terminate_timeout)
        self._close_logs()
        if self.started_at is not None and self.stopped_at is None:
            self.stopped_at = _utc_now()

    def manifest(self) -> dict[str, Any]:
        result = {
            "mode": "external_endpoint" if self.external else "managed_subprocess",
            "endpoint": self.endpoint,
            "served_model_name": self.serving.served_model_name,
            "vllm_version": self.serving.version,
        }
        for key, value in (
            ("started_at", self.started_at), ("ready_at", self.ready_at),
            ("stopped_at", self.stopped_at),
        ):
            if value is not None:
                result[key] = value
        if self.process is not None:
            result["pid"] = self.process.pid
        if not self.external:
            result["stdout_path"] = str(self.stdout_path)
            result["stderr_path"] = str(self.stderr_path)
        return result

    def __enter__(self) -> "VLLMServer":
        return self.start()

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.stop()
