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

"""Stable configuration fingerprints without importing the simulator."""

from __future__ import annotations

import dataclasses
import hashlib
from enum import Enum
from importlib import metadata
import json
from pathlib import Path
import platform
from collections.abc import Mapping
from typing import Any

__all__ = ["stable_config_hash", "build_recording_provenance"]


def _canonical(value: Any, ancestors: set[int] | None = None) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Enum):
        return {
            "enum": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": _canonical(value.value),
        }
    if value is dataclasses.MISSING:
        return {"missing": True}
    if callable(value):
        return {
            "callable": f"{getattr(value, '__module__', type(value).__module__)}."
            f"{getattr(value, '__qualname__', type(value).__qualname__)}"
        }
    ancestors = set() if ancestors is None else ancestors
    if id(value) in ancestors:
        return {"recursive_type": type(value).__qualname__}
    ancestors = ancestors | {id(value)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical(getattr(value, field.name), ancestors)
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _canonical(item, ancestors) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item, ancestors) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonical(item, ancestors) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    # Arrays/config tensors are uncommon, but their values must affect the hash.
    if type(value).__module__.startswith(("numpy", "torch")):
        if hasattr(value, "tolist"):
            return _canonical(value.tolist(), ancestors)
        return str(value)
    if hasattr(value, "__dict__") and type(value).__module__ == "types":
        return _canonical(vars(value), ancestors)
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def stable_config_hash(config: Any) -> str:
    """Fingerprint a config by values and stable callable names.

    Args:
        config: Dataclass, mapping, sequence, or JSON-compatible configuration.

    Returns:
        SHA-256 hexadecimal digest of the canonical configuration. Runtime
        object addresses are deliberately excluded; unknown object types are
        represented by their qualified type name.
    """
    encoded = json.dumps(
        _canonical(config), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_recording_provenance(env: Any) -> dict[str, Any]:
    """Snapshot configuration, runtime versions, and control timing.

    Args:
        env: Environment exposing ``cfg`` and optional simulation timing fields.

    Returns:
        JSON-compatible provenance. This records fingerprints rather than a
        replacement for the full simulation initial state in a replay artifact.
    """
    cfg = getattr(env, "cfg", None)
    program = getattr(cfg, "task_program", None)
    versions: dict[str, str] = {"python": platform.python_version()}
    for package in ("embodichain", "lerobot", "torch"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "uninstalled"
    result: dict[str, Any] = {
        "config_hash": stable_config_hash(cfg),
        "program_hash": stable_config_hash(program) if program is not None else None,
        "provenance": {"versions": versions},
    }
    for name in ("step_dt", "physics_dt", "control_frequency"):
        value = getattr(env, name, None)
        if isinstance(value, (int, float)):
            result["provenance"][name] = value
    sim_cfg = getattr(env, "sim_cfg", None)
    for owner in (cfg, sim_cfg):
        backend = getattr(owner, "physics", None)
        if isinstance(backend, str):
            result["provenance"]["physics"] = backend
            break
    backend = getattr(getattr(env, "sim", None), "physics_backend", None)
    if isinstance(backend, str):
        result["provenance"]["physics"] = backend
    steps = getattr(cfg, "sim_steps_per_control", None)
    if isinstance(steps, int):
        result["provenance"]["sim_steps_per_control"] = steps
    return result
