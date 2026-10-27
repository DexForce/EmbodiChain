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

"""Compatible public imports for robot-owned assets."""

from __future__ import annotations

__all__ = [
    "DexforceW1V021",
    "DexforceW1V022",
    "DexforceW1V025",
    "DexforceW1V030",
    "DexforceW1ChassisV021",
    "DexforceW1TorsoV021",
    "DexforceW1EyesV021",
    "DexforceW1HeadV021",
    "DexforceW1LeftArmV021",
    "DexforceW1RightArmV021",
]

from importlib import import_module

_ASSET_EXPORTS = {
    "DexforceW1V021": "embodichain_robots.dexforce.w1.assets",
    "DexforceW1V022": "embodichain_robots.dexforce.w1.assets",
    "DexforceW1V025": "embodichain_robots.dexforce.w1.assets",
    "DexforceW1V030": "embodichain_robots.dexforce.w1.assets",
    "DexforceW1ChassisV021": "embodichain_robots.dexforce.w1.assets",
    "DexforceW1TorsoV021": "embodichain_robots.dexforce.w1.assets",
    "DexforceW1EyesV021": "embodichain_robots.dexforce.w1.assets",
    "DexforceW1HeadV021": "embodichain_robots.dexforce.w1.assets",
    "DexforceW1LeftArmV021": "embodichain_robots.dexforce.w1.assets",
    "DexforceW1RightArmV021": "embodichain_robots.dexforce.w1.assets",
}


def __getattr__(name: str) -> type:
    if name not in _ASSET_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_ASSET_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
