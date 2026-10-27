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

"""Evaluate motion planners using installed suites and external adapters.

Configuration and evaluation contracts load on access. Importing the package
or loading a suite does not construct a simulator.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "BenchmarkCase",
    "BenchmarkRunResult",
    "PlannerAdapter",
    "PlannerContext",
    "PlannerSpecCfg",
    "SuiteCfg",
    "load_suite",
    "run_all_benchmarks",
    "run_benchmark",
    "prepare_case_manifest",
]

_EXPORT_MODULES = {
    "BenchmarkCase": ".models",
    "BenchmarkRunResult": ".runner",
    "PlannerAdapter": ".planners.base",
    "PlannerContext": ".planners.base",
    "PlannerSpecCfg": ".config",
    "SuiteCfg": ".config",
    "load_suite": ".config",
    "run_all_benchmarks": ".cli",
    "run_benchmark": ".api",
    "prepare_case_manifest": ".api",
}


def __getattr__(name: str) -> Any:
    """Load one public benchmark export on first access."""
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORT_MODULES[name], __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy public exports in interactive discovery."""
    return sorted(set(globals()) | set(__all__))
