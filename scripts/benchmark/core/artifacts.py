# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Strict JSON artifacts and stable workload identities."""

from __future__ import annotations

from datetime import datetime, timezone
from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

__all__ = [
    "write_json",
    "read_json",
    "read_json_object",
    "stable_hash",
    "create_experiment_directory",
    "ArtifactStore",
    "JsonlLedger",
    "append_jsonl",
]


def write_json(path: Path, value: object) -> Path:
    """Atomically replace a JSON file after validating finite JSON values."""
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite JSON number: {value}")
    return parsed


def read_json(path: Path) -> Any:
    """Read finite JSON values, including raw run lists."""
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
        parse_float=_finite_float,
    )


def read_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON object; reject non-object and non-finite worker output."""
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def append_jsonl(path: Path, value: object) -> Path:
    """Append one finite JSON record and flush it for crash recovery."""
    payload = (
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return path


class JsonlLedger:
    """Append-only JSONL ledger with idempotent record identities."""

    def __init__(self, path: Path, *, identity_key: str = "record_id") -> None:
        if not identity_key.strip():
            raise ValueError("identity_key must be a nonempty string")
        self.path = Path(path)
        self.identity_key = identity_key

    def _records(self) -> list[dict[str, Any]]:
        """Read existing records, rejecting malformed lines."""
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                value = json.loads(
                    line, parse_constant=_reject_constant, parse_float=_finite_float
                )
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record at line {line_number} is not an object")
            records.append(value)
        return records

    def append(self, value: Mapping[str, Any] | object) -> bool:
        """Append a record, returning false when the identical record exists.

        Reusing an identity with a different payload raises instead of silently
        increasing a retry denominator or overwriting evidence.
        """
        if hasattr(value, "to_dict"):
            value = value.to_dict()
        if not isinstance(value, Mapping):
            raise TypeError("ledger records must be mappings or expose to_dict()")
        record = dict(value)
        identity = record.get(self.identity_key)
        if identity is None:
            raise ValueError(f"record must contain {self.identity_key!r}")
        for existing in self._records():
            if existing.get(self.identity_key) != identity:
                continue
            if existing == record:
                return False
            raise ValueError(f"different payload for {self.identity_key}={identity!r}")
        append_jsonl(self.path, record)
        return True


class ArtifactStore:
    """Manage the standard raw, metric, quality and artifact-index files."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).absolute()
        self.raw = JsonlLedger(self.root / "raw.jsonl")

    def append_raw(self, record: Mapping[str, Any] | object) -> bool:
        """Append an idempotent raw observation or attempt record."""
        return self.raw.append(record)

    def write_metrics(self, metrics: Mapping[str, Any]) -> Path:
        """Atomically replace the aggregate metrics artifact."""
        return write_json(self.root / "metrics.json", dict(metrics))

    def write_quality(self, quality: Mapping[str, Any]) -> Path:
        """Atomically replace the independent quality artifact."""
        return write_json(self.root / "quality.json", dict(quality))

    def register_artifact(self, path: Path, *, kind: str) -> dict[str, object]:
        """Register an evidence file once with bytes and SHA-256 identity."""
        path = Path(path).absolute()
        try:
            relative = path.relative_to(self.root).as_posix()
        except ValueError as exc:
            raise ValueError(
                "artifact must be inside the experiment directory"
            ) from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        record = {
            "path": relative,
            "kind": kind,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        index_path = self.root / "artifact_index.json"
        index = (
            read_json_object(index_path) if index_path.exists() else {"artifacts": []}
        )
        artifacts = index.setdefault("artifacts", [])
        if not isinstance(artifacts, list):
            raise ValueError("artifact_index.json artifacts must be a list")
        for existing in artifacts:
            if existing.get("path") != relative:
                continue
            if existing == record:
                return record
            raise ValueError(
                f"artifact path already registered with different content: {relative}"
            )
        artifacts.append(record)
        write_json(index_path, index)
        return record


def stable_hash(value: object) -> str:
    """Hash JSON semantics using the original camera-pilot encoding."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def create_experiment_directory(
    output_root: Path,
    *,
    experiment_id: str,
    config: dict[str, Any],
    definition: Mapping[str, Any] | None = None,
    assets_manifest: Mapping[str, Any] | None = None,
) -> Path:
    """Create one experiment directory with the standard v1 artifact protocol.

    ``effective_config.yaml`` deliberately contains JSON syntax, which is a
    valid YAML 1.2 document and keeps report rebuilds dependency-free.
    """
    if not experiment_id.strip():
        raise ValueError("experiment_id must not be empty")
    if definition is not None and hasattr(definition, "to_dict"):
        definition = definition.to_dict()
    definition_payload = definition or {
        "schema_version": 1,
        "experiment_id": experiment_id,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    root = output_root.absolute() / stamp
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "definition.json", definition_payload)
    write_json(root / "config.json", config)
    write_json(root / "effective_config.yaml", config)
    write_json(root / "assets_manifest.json", assets_manifest or {"assets": []})
    write_json(root / "metrics.json", {"schema_version": 1, "metrics": []})
    write_json(root / "quality.json", {"schema_version": 1, "runs": []})
    write_json(root / "artifact_index.json", {"schema_version": 1, "artifacts": []})
    (root / "raw.jsonl").touch()
    write_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "invocation_id": stamp,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "parameter_sha256": stable_hash(config),
            "definition": definition,
            "artifact_protocol": {
                "definition": "definition.json",
                "effective_config": "effective_config.yaml",
                "assets": "assets_manifest.json",
                "raw": "raw.jsonl",
                "metrics": "metrics.json",
                "quality": "quality.json",
                "artifact_index": "artifact_index.json",
            },
        },
    )
    return root
