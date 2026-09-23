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

"""Small value contracts shared by benchmark domains."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping

__all__ = ["SCHEMA_VERSION", "RunStatus", "RunSpec", "validate_run_result"]

SCHEMA_VERSION = 1
RunStatus = Literal["not_run", "completed", "failed", "timeout", "interrupted"]


@dataclass(frozen=True)
class RunSpec:
    """A resolved process invocation, not an authored simulator configuration.

    Domain launchers own configuration and runtime selection. Only a list of
    argv strings crosses this boundary; no shell command or simulator object.
    Environment values are never serialized into the public plan.
    """

    backend: str
    repeat: int
    command: tuple[str, ...]
    output: Path
    case_id: str = "default"
    timeout_s: float = 300.0
    env: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or not self.backend.strip():
            raise ValueError("backend must be a nonempty string")
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("case_id must be a nonempty string")
        if type(self.repeat) is not int or self.repeat < 0:
            raise ValueError("repeat must be a nonnegative integer")
        if not self.command or any(
            not isinstance(a, str) or not a for a in self.command
        ):
            raise ValueError("command must contain nonempty argv strings")
        if (
            isinstance(self.timeout_s, bool)
            or not math.isfinite(self.timeout_s)
            or self.timeout_s <= 0
        ):
            raise ValueError("timeout_s must be positive and finite")
        object.__setattr__(self, "command", tuple(self.command))
        object.__setattr__(self, "output", Path(self.output).absolute())
        if self.env is not None:
            if any(
                not isinstance(k, str) or not isinstance(v, str)
                for k, v in self.env.items()
            ):
                raise ValueError("environment keys and values must be strings")
            object.__setattr__(self, "env", MappingProxyType(dict(self.env)))


def validate_run_result(value: dict[str, Any]) -> None:
    """Validate common fields without discarding domain evidence or legacy v1."""
    version = value.get("schema_version", SCHEMA_VERSION)
    if type(version) is not int or version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported run schema_version: {version!r}")
    status = value.get("status")
    if not isinstance(status, str) or status not in {
        "not_run",
        "completed",
        "failed",
        "timeout",
        "interrupted",
    }:
        raise ValueError("Missing or unknown execution status")
    for key in ("backend", "case_id", "run_id"):
        if key in value and not isinstance(value[key], str):
            raise ValueError(f"{key} must be a string")
    for key in ("config_sha256", "hardware_id", "quality_status"):
        if value.get(key) is not None and not isinstance(value[key], str):
            raise ValueError(f"{key} must be a string or null")
    if "metrics" in value and not isinstance(value["metrics"], dict):
        raise ValueError("metrics must be an object")
