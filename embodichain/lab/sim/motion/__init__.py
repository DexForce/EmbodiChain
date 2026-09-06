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

"""Robot motion solving, planning, workspace analysis, and trajectory augmentation.

Motion capabilities load on access so importing solvers during Robot initialization
does not also import planners or workspace analyzers. Importing this package
still follows the normal :mod:`embodichain.lab.sim` initialization lifecycle.
"""

from __future__ import annotations

import importlib
from types import ModuleType

__all__ = [
    "solvers",
    "planners",
    "workspace",
    "trajectory_augmentation",
    "motion_generator",
]


def __getattr__(name: str) -> ModuleType:
    """Load a public motion module or subpackage on first access."""
    if name in __all__:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return eagerly and lazily available motion names."""
    return sorted(set(globals()) | set(__all__))
