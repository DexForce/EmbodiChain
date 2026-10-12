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

"""Generic dual-arm composition; concrete presets live in embodichain_robots."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "DualArmRobotCfg",
    "build_dual_arm_cfg",
]

_EXPORTS = {
    "DualArmRobotCfg": ".dual_arm",
    "build_dual_arm_cfg": ".dual_arm",
}
_MODULES = ("dual_arm",)


def __getattr__(name: str) -> Any:
    if name in _MODULES:
        value = import_module(f".{name}", __name__)
        globals()[name] = value
        return value
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_MODULES))
