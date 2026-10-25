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

"""Official robot configurations bundled with the EmbodiChain distribution.

Configuration classes are imported on first access. Importing the package or
its catalog does not initialize the simulation package or load every robot.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AlohaMiniCfg",
    "TianjiMarvinCfg",
    "DexforceW1Cfg",
    "CobotMagicCfg",
    "FrankaPandaCfg",
    "FrankaPandaPgiCfg",
    "FlexivRizon4SCfg",
    "URRobotCfg",
    "SO101FollowerCfg",
    "UnitreeGo1Cfg",
    "UnitreeGo2Cfg",
    "UnitreeG1Cfg",
    "UnitreeH12Cfg",
    "ANYmalCCfg",
    "MicroDuckCfg",
]

_EXPORTS = {
    "AlohaMiniCfg": ".aloha_mini",
    "TianjiMarvinCfg": ".tianji_marvin",
    "DexforceW1Cfg": ".dexforce_w1.cfg",
    "CobotMagicCfg": ".cobotmagic",
    "FrankaPandaCfg": ".franka_panda",
    "FrankaPandaPgiCfg": ".franka_panda",
    "FlexivRizon4SCfg": ".flexiv_rizon4s",
    "URRobotCfg": ".ur_robot",
    "SO101FollowerCfg": ".so101",
    "UnitreeGo1Cfg": ".unitree_go1",
    "UnitreeGo2Cfg": ".unitree_go2",
    "UnitreeG1Cfg": ".unitree_g1",
    "UnitreeH12Cfg": ".unitree_h1_2",
    "ANYmalCCfg": ".anymal_c",
    "MicroDuckCfg": ".microduck",
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
