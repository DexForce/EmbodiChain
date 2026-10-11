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

"""Official robot IDs and historical CLI/dual-arm aliases; no runtime imports."""

from __future__ import annotations

_CONFIG_CLASSES = {
    "AlohaMini": "embodichain_robots.aloha_mini:AlohaMiniCfg",
    "TianjiMarvin": "embodichain_robots.tianji_marvin:TianjiMarvinCfg",
    "DexforceW1": "embodichain_robots.dexforce_w1.cfg:DexforceW1Cfg",
    "CobotMagic": "embodichain_robots.cobotmagic:CobotMagicCfg",
    "FrankaPanda": "embodichain_robots.franka_panda:FrankaPandaCfg",
    "URRobot": "embodichain_robots.ur_robot:URRobotCfg",
}

ROBOT_CONFIGS = {
    name: {"class_path": class_path} for name, class_path in _CONFIG_CLASSES.items()
}

for _alias, _name in {
    "aloha_mini": "AlohaMini",
    "tianji_marvin": "TianjiMarvin",
    "dexforce_w1": "DexforceW1",
    "cobotmagic": "CobotMagic",
    "franka": "FrankaPanda",
    "franka_panda": "FrankaPanda",
    "ur": "URRobot",
}.items():
    ROBOT_CONFIGS[_alias] = ROBOT_CONFIGS[_name]

for _variant in ("ur3", "ur3e", "ur5", "ur5e", "ur10", "ur10e"):
    ROBOT_CONFIGS[_variant] = {
        "class_path": _CONFIG_CLASSES["URRobot"],
        "defaults": {"robot_type": _variant},
    }
