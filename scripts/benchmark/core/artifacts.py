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


def stable_hash(value: object) -> str:
    """Hash JSON semantics using the original camera-pilot encoding."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def create_experiment_directory(
    output_root: Path, *, experiment_id: str, config: dict[str, Any]
) -> Path:
    """Create one new experiment directory and freeze its effective config."""
    if not experiment_id.strip():
        raise ValueError("experiment_id must not be empty")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    root = output_root.absolute() / stamp
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "config.json", config)
    write_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "invocation_id": stamp,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "parameter_sha256": stable_hash(config),
        },
    )
    return root
