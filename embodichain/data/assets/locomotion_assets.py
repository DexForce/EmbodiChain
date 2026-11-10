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


"""Legacy locomotion asset imports and the core HumanoidRun archive."""

from __future__ import annotations

from embodichain.data._locomotion_asset import _LocomotionAsset

__all__ = [
    "ANYmalCLocomotion",
    "HumanoidRun",
    "MicroDuckLocomotion",
    "UnitreeG1Locomotion",
    "UnitreeGo1Locomotion",
    "UnitreeGo2Locomotion",
    "UnitreeH1_2Locomotion",
]


class HumanoidRun(_LocomotionAsset):
    """Robot descriptions and dependencies for HumanoidRun."""

    archive_md5 = "846cb8db9d069d0eae41c8b799ad1872"


from importlib import import_module

_ASSET_EXPORTS = {
    "UnitreeGo1Locomotion": "embodichain_robots.unitree.go1.assets",
    "UnitreeGo2Locomotion": "embodichain_robots.unitree.go2.assets",
    "UnitreeG1Locomotion": "embodichain_robots.unitree.g1.assets",
    "UnitreeH1_2Locomotion": "embodichain_robots.unitree.h1_2.assets",
    "ANYmalCLocomotion": "embodichain_robots.anybotics.anymal_c.assets",
    "MicroDuckLocomotion": "embodichain_robots.pollen_robotics.microduck.assets",
}


def __getattr__(name: str) -> type:
    if name not in _ASSET_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_ASSET_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
