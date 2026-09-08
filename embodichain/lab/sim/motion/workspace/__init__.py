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

"""Robot workspace analysis and runtime sampling.

Runtime types are imported eagerly because :class:`~embodichain.lab.sim.objects.Robot`
depends on them. Analyzer types and their heavier visualization dependencies are
loaded lazily to keep the Robot import path free of circular dependencies.
"""

from __future__ import annotations

import importlib

from .cfg import RobotWorkspaceCfg
from .runtime import RobotWorkspace, WorkspaceSample

__all__ = [
    "RobotWorkspaceCfg",
    "RobotWorkspace",
    "WorkspaceSample",
    "AnalysisMode",
    "WorkspaceAnalyzerConfig",
    "WorkspaceAnalyzer",
    "configs",
    "samplers",
    "caches",
    "visualizers",
    "metrics",
    "constraints",
]

_ANALYZER_EXPORTS = {
    "AnalysisMode",
    "WorkspaceAnalyzerConfig",
    "WorkspaceAnalyzer",
}
_SUBMODULE_EXPORTS = {
    "configs",
    "samplers",
    "caches",
    "visualizers",
    "metrics",
    "constraints",
}


def __getattr__(name: str):
    """Lazily resolve analyzer APIs and submodules."""
    if name in _ANALYZER_EXPORTS:
        module = importlib.import_module(f"{__name__}.analyzer")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _SUBMODULE_EXPORTS:
        value = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return eagerly and lazily exported workspace names."""
    return sorted(set(globals()) | set(__all__))
