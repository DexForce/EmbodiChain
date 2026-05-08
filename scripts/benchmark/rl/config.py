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

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

BENCHMARK_ROOT = Path(__file__).resolve().parent


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in YAML file {path}, got {type(data)!r}.")
    return data


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into `base` and return a new mapping."""
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_task_spec(name: str) -> dict[str, Any]:
    """Load a benchmark task specification by name."""
    return load_yaml(BENCHMARK_ROOT / "tasks" / f"{name}.yaml")


def load_algorithm_spec(name: str) -> dict[str, Any]:
    """Load a benchmark algorithm specification by name."""
    return load_yaml(BENCHMARK_ROOT / "algorithms" / f"{name}.yaml")


def load_suite_spec(name: str = "default") -> dict[str, Any]:
    """Load a benchmark suite specification by name."""
    return load_yaml(BENCHMARK_ROOT / "suites" / f"{name}.yaml")


__all__ = [
    "BENCHMARK_ROOT",
    "deep_update",
    "load_algorithm_spec",
    "load_suite_spec",
    "load_task_spec",
    "load_yaml",
]
