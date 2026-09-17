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

"""EmbodiChain's robotics laboratory.

Bundles Task Programs, simulation and environment runtime components,
real-device controllers, and browser visualization.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

__all__ = [
    "devices",
    "task_program",
    "gym",
    "sim",
    "visualization",
]


def __getattr__(name: str) -> ModuleType:
    """Import laboratory subsystems on first attribute access.

    This keeps CPU-only consumers of the visualization protocol from loading
    the simulation runtime while preserving ``embodichain.lab.sim`` and the
    other historical package attributes.
    """
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    """Include lazily exported subsystems in interactive discovery."""
    return sorted(set(globals()) | set(__all__))
