"""SWE-bench Multilingual adapter with language-aware aggregation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .base import AgentBenchmarkInstance, AgentBenchmarkSpec
from .swebench_verified import SWEbenchVerifiedAdapter


# The v5 dataset schema does not carry a language column.  This mapping is the
# official Appendix A repository classification, pinned by the config source.
REPOSITORY_LANGUAGES = {
    **dict.fromkeys(("redis/redis", "jqlang/jq", "micropython/micropython", "valkey-io/valkey"), "C"),
    **dict.fromkeys(("nlohmann/json", "fmtlib/fmt"), "C++"),
    **dict.fromkeys(("caddyserver/caddy", "hashicorp/terraform", "prometheus/prometheus", "gohugoio/hugo", "gin-gonic/gin"), "Go"),
    **dict.fromkeys(("google/gson", "apache/druid", "projectlombok/lombok", "apache/lucene", "reactivex/rxjava", "javaparser/javaparser"), "Java"),
    **dict.fromkeys(("babel/babel", "vuejs/core", "facebook/docusaurus", "immutable-js/immutable-js", "mrdoob/three.js", "preactjs/preact", "axios/axios"), "JavaScript/TypeScript"),
    **dict.fromkeys(("phpoffice/phpspreadsheet", "laravel/framework", "php-cs-fixer/php-cs-fixer", "briannesbitt/carbon"), "PHP"),
    **dict.fromkeys(("jekyll/jekyll", "fluent/fluentd", "fastlane/fastlane", "jordansissel/fpm", "faker-ruby/faker", "rubocop/rubocop"), "Ruby"),
    **dict.fromkeys(("tokio-rs/tokio", "uutils/coreutils", "nushell/nushell", "tokio-rs/axum", "burntsushi/ripgrep", "sharkdp/bat", "astral-sh/ruff"), "Rust"),
}


class SWEbenchMultilingualAdapter(SWEbenchVerifiedAdapter):
    benchmark_id = "swebench_multilingual"

    def __init__(self, spec: AgentBenchmarkSpec) -> None:
        if spec.benchmark_id != self.benchmark_id:
            raise ValueError("SWE-bench Multilingual adapter/config mismatch")
        if spec.prompt.template != "{problem_statement}":
            raise ValueError("SWE-bench adapters require the non-leaking problem_statement template")
        self.spec = spec

    def load_instance(self, row: Mapping[str, Any]) -> AgentBenchmarkInstance:
        base = super().load_instance({**row})
        language = row.get("language") or row.get("language_name") or REPOSITORY_LANGUAGES.get(base.repo)
        if not language:
            raise ValueError("Multilingual instance is missing language")
        metadata = dict(base.metadata)
        metadata.update({"language": str(language), **{key: row[key] for key in ("platform",) if key in row}})
        return AgentBenchmarkInstance(self.benchmark_id, base.instance_id, base.problem_statement, base.repo, base.base_commit, metadata)

    def evaluator_command(self, *args: Any, **kwargs: Any) -> tuple[str, ...]:
        command = list(super().evaluator_command(*args, **kwargs))
        dataset_index = command.index("eval") + 1
        if not self.spec.dataset.local_path:
            command[dataset_index] = "multilingual"
        return tuple(command)

    def aggregate(self, report: Mapping[str, Any], instances: Mapping[str, AgentBenchmarkInstance]) -> dict[str, Any]:
        result = dict(report)
        buckets: dict[str, dict[str, int | float]] = {}
        for instance_id, verdict in report["per_instance"].items():
            language = str(instances[instance_id].metadata["language"])
            verdict["language"] = language
            bucket = buckets.setdefault(language, {"attempted": 0, "resolved": 0, "rate": 0.0})
            bucket["attempted"] = int(bucket["attempted"]) + 1
            bucket["resolved"] = int(bucket["resolved"]) + int(bool(verdict["resolved"]))
        for bucket in buckets.values():
            bucket["rate"] = int(bucket["resolved"]) / int(bucket["attempted"])
        result["per_language"] = dict(sorted(buckets.items()))
        return result
