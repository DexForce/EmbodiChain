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

"""Dexforce W1 robot configuration, imported on first access."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["DexforceW1Cfg"]
_MODULES = ("cfg", "types", "specs", "hand_specs", "params", "utils")


def __getattr__(name: str) -> Any:
    if name in _MODULES:
        value = import_module(f".{name}", __name__)
        globals()[name] = value
        return value
    if name != "DexforceW1Cfg":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = import_module("embodichain_robots.dexforce_w1", __name__).DexforceW1Cfg
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_MODULES))
