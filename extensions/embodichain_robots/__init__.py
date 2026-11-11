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
    "AgibotG1Cfg",
    "YamI2rtCfg",
    "UnitreeGo1Cfg",
    "UnitreeGo2Cfg",
    "UnitreeG1Cfg",
    "UnitreeH12Cfg",
    "ANYmalCCfg",
    "MicroDuckCfg",
    "ARX5Cfg",
    "AlohaMiniCfg",
    "TianjiMarvinCfg",
    "DexforceW1Cfg",
    "CobotMagicCfg",
    "AgilexPiperXCfg",
    "FrankaCfg",
    "FrankaPandaCfg",
    "FlexivRizon4SCfg",
    "URRobotCfg",
    "SO101FollowerCfg",
    "BrainCoRevo1Cfg",
    "SharpaWaveCfg",
    "WujiHandCfg",
    "FrankaPandaHandCfg",
    "DHPGI14080Cfg",
    "CobotMagicV100GripperCfg",
    "PikaGripperCfg",
    "TianjiMarvinGripperCfg",
    "RobotiqArg2F85Cfg",
    "RobotiqArg2F140Cfg",
]

_EXPORTS = {
    "AgibotG1Cfg": ".agibot.g1",
    "YamI2rtCfg": ".i2rt.yam",
    "UnitreeGo1Cfg": ".unitree.go1",
    "UnitreeGo2Cfg": ".unitree.go2",
    "UnitreeG1Cfg": ".unitree.g1",
    "UnitreeH12Cfg": ".unitree.h1_2",
    "ANYmalCCfg": ".anybotics.anymal_c",
    "MicroDuckCfg": ".pollen_robotics.microduck",
    "ARX5Cfg": ".arx.arx5",
    "AlohaMiniCfg": ".alohamini.aloha_mini",
    "TianjiMarvinCfg": ".tianji.marvin",
    "DexforceW1Cfg": ".dexforce.w1",
    "CobotMagicCfg": ".agilex.cobotmagic",
    "AgilexPiperXCfg": ".agilex.piper_x",
    "FrankaCfg": ".franka.cfg",
    "FrankaPandaCfg": ".franka.panda",
    "FlexivRizon4SCfg": ".flexiv.rizon4s",
    "URRobotCfg": ".universal_robots.ur_robot",
    "SO101FollowerCfg": ".the_robot_studio.so101",
    "BrainCoRevo1Cfg": ".brainco.revo1",
    "SharpaWaveCfg": ".sharpa.wave",
    "WujiHandCfg": ".wuji.hand",
    "FrankaPandaHandCfg": ".franka.hand.cfg",
    "DHPGI14080Cfg": ".dh_robotics.pgi_140_80.cfg",
    "CobotMagicV100GripperCfg": ".agilex.cobotmagic.gripper.cfg",
    "PikaGripperCfg": ".agilex.pika.cfg",
    "TianjiMarvinGripperCfg": ".tianji.marvin.gripper.cfg",
    "RobotiqArg2F85Cfg": ".robotiq.arg2f85.cfg",
    "RobotiqArg2F140Cfg": ".robotiq.arg2f140.cfg",
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
