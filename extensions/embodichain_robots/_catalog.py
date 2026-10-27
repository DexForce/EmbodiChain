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

"""Official robot IDs and UR variant presets; no runtime imports."""

from __future__ import annotations

from itertools import product

_CONFIG_CLASSES = {
    "UnitreeGo1": "embodichain_robots.unitree.go1:UnitreeGo1Cfg",
    "UnitreeGo2": "embodichain_robots.unitree.go2:UnitreeGo2Cfg",
    "UnitreeG1": "embodichain_robots.unitree.g1:UnitreeG1Cfg",
    "UnitreeH1_2": "embodichain_robots.unitree.h1_2:UnitreeH12Cfg",
    "ANYmalC": "embodichain_robots.anybotics.anymal_c:ANYmalCCfg",
    "MicroDuck": "embodichain_robots.pollen_robotics.microduck:MicroDuckCfg",
    "ARX5": "embodichain_robots.arx.arx5:ARX5Cfg",
    "AlohaMini": "embodichain_robots.alohamini.aloha_mini:AlohaMiniCfg",
    "TianjiMarvin": "embodichain_robots.tianji.marvin:TianjiMarvinCfg",
    "DexforceW1": "embodichain_robots.dexforce.w1:DexforceW1Cfg",
    "CobotMagic": "embodichain_robots.agilex.cobotmagic:CobotMagicCfg",
    "FrankaPanda": "embodichain_robots.franka.panda:FrankaPandaCfg",
    "FlexivRizon4S": "embodichain_robots.flexiv.rizon4s:FlexivRizon4SCfg",
    "URRobot": "embodichain_robots.universal_robots.ur_robot:URRobotCfg",
    "SO101Follower": "embodichain_robots.the_robot_studio.so101:SO101FollowerCfg",
}

_DEFAULT_CONTROL_INFO = {
    "UnitreeGo1": {"dof": 12, "control_parts": {"locomotion": 12}},
    "UnitreeGo2": {"dof": 12, "control_parts": {"locomotion": 12}},
    "UnitreeG1": {"dof": 29, "control_parts": {"locomotion": 29}},
    "UnitreeH1_2": {"dof": 27, "control_parts": {"locomotion": 27}},
    "ANYmalC": {"dof": 12, "control_parts": {"locomotion": 12}},
    "MicroDuck": {"dof": 14, "control_parts": {"locomotion": 14}},
    "ARX5": {"dof": 8, "control_parts": {"arm": 6, "eef": 2}},
    "AlohaMini": {
        "dof": 15,
        "control_parts": {
            "left_arm": 6,
            "right_arm": 6,
            "left_eef": 1,
            "right_eef": 1,
            "torso": 1,
        },
    },
    "TianjiMarvin": {
        "dof": 16,
        "control_parts": {
            "left_arm": 7,
            "right_arm": 7,
            "left_eef": 2,
            "right_eef": 2,
        },
    },
    "DexforceW1": {
        "dof": 32,
        "control_parts": {
            "torso": 4,
            "head": 2,
            "left_arm": 7,
            "right_arm": 7,
            "dual_arm": 14,
            "left_eef": 6,
            "right_eef": 6,
            "full_body": 20,
        },
    },
    "CobotMagic": {
        "dof": 14,
        "control_parts": {
            "left_arm": 6,
            "left_eef": 2,
            "right_arm": 6,
            "right_eef": 2,
        },
    },
    "FrankaPanda": {
        "dof": 8,
        "control_parts": {"arm": 7, "eef": 2},
    },
    "URRobot": {"dof": 6, "control_parts": {"arm": 6}},
    "SO101Follower": {"dof": 6, "control_parts": {"arm": 5, "eef": 1}},
    "FlexivRizon4S": {"dof": 7, "control_parts": {"arm": 7}},
}

_MIMIC_JOINTS = {
    "AlohaMini": {},
    "TianjiMarvin": {
        "LEFT_HAND_FINGER_2": "LEFT_HAND_FINGER_1",
        "RIGHT_HAND_FINGER_2": "RIGHT_HAND_FINGER_1",
    },
    "DexforceW1": {
        f"{side}_{finger}_PIP": f"{side}_HAND_{finger}"
        for side in ("LEFT", "RIGHT")
        for finger in ("INDEX", "MIDDLE", "RING", "PINKY")
    },
    "CobotMagic": {
        "left_joint8": "left_joint7",
        "right_joint8": "right_joint7",
    },
    "FrankaPanda": {"fr3_finger_joint2": "fr3_finger_joint1"},
    "URRobot": {},
}
_CONTROL_PART_MIMICS = {
    "TianjiMarvin": {
        "left_eef": ("LEFT_HAND_FINGER_2",),
        "right_eef": ("RIGHT_HAND_FINGER_2",),
    },
    "CobotMagic": {"left_eef": ("left_joint8",), "right_eef": ("right_joint8",)},
    "FrankaPanda": {"eef": ("fr3_finger_joint2",)},
}

_CLASSIFICATION = {
    "UnitreeGo1": ("unitree", "quadruped", 0),
    "UnitreeGo2": ("unitree", "quadruped", 0),
    "UnitreeG1": ("unitree", "bipedal_humanoid", 2),
    "UnitreeH1_2": ("unitree", "bipedal_humanoid", 2),
    "ANYmalC": ("anybotics", "quadruped", 0),
    "MicroDuck": ("pollen_robotics", "biped", 0),
    "ARX5": ("arx", "fixed_manipulator", 1),
    "AlohaMini": (None, "wheeled_manipulator", 2),
    "TianjiMarvin": ("tianji", "fixed_manipulator", 2),
    "DexforceW1": ("dexforce", "wheeled_humanoid", 2),
    "CobotMagic": ("agilex", "fixed_manipulator", 2),
    "FrankaPanda": ("franka", "fixed_manipulator", 1),
    "FlexivRizon4S": ("flexiv", "fixed_manipulator", 1),
    "URRobot": ("universal_robots", "fixed_manipulator", 1),
    "SO101Follower": ("the_robot_studio", "fixed_manipulator", 1),
}

ROBOT_CONFIGS = {
    name: {
        "class_path": class_path,
        **dict(zip(("vendor", "morphology", "arm_count"), _CLASSIFICATION[name])),
        **_DEFAULT_CONTROL_INFO.get(name, {}),
        "mimic_joints": _MIMIC_JOINTS.get(name, {}),
        "control_part_mimics": _CONTROL_PART_MIMICS.get(name, {}),
        "supports_dual_arm": name in ("FrankaPanda", "URRobot"),
    }
    for name, class_path in _CONFIG_CLASSES.items()
}


_W1_VERSIONS = ("v021", "v022", "v025", "v030")
_W1_HANDS = {
    "BRAINCO_HAND": ("BrainCo", 6, 6),
    "DH_PGC_GRIPPER": ("DH_PGC", 2, 1),
    "DH_PGC_GRIPPER_M": ("DH_PGC_M", 2, 1),
}


def _w1_variant_profile(
    version: str, hands: tuple[str, str] | None
) -> dict[str, object]:
    """Describe W1 body/hand defaults without loading configuration classes."""
    parts: dict[str, int] = {
        name: count
        for name, count in _DEFAULT_CONTROL_INFO["DexforceW1"]["control_parts"].items()
        if name not in ("left_eef", "right_eef")
    }
    defaults: dict[str, object] = {"version": version}
    dof = 20
    mimics: dict[str, str] = {}
    part_mimics: dict[str, tuple[str, ...]] = {}
    if hands is None:
        defaults["with_default_eef"] = False
    else:
        defaults["hand_types"] = dict(zip(("left", "right"), hands))
        defaults["hand_versions"] = {"left": "v021", "right": "v021"}
        for side, brand in zip(("left", "right"), hands):
            _label, count, independent = _W1_HANDS[brand]
            parts[f"{side}_eef"] = count
            dof += independent
            prefix = side.upper()
            if brand == "BRAINCO_HAND":
                mimics.update(
                    {
                        f"{prefix}_{finger}_PIP": f"{prefix}_HAND_{finger}"
                        for finger in ("INDEX", "MIDDLE", "RING", "PINKY")
                    }
                )
            else:
                suffix = "_JOINT" if brand == "DH_PGC_GRIPPER" else ""
                follower = f"{prefix}_FINGER2{suffix}"
                mimics[follower] = f"{prefix}_FINGER1{suffix}"
                part_mimics[f"{side}_eef"] = (follower,)
    return {
        "dof": dof,
        "control_parts": parts,
        "mimic_joints": mimics,
        "control_part_mimics": part_mimics,
        "defaults": defaults,
    }


_W1_VARIANTS: dict[str, dict[str, object]] = {}
for _version in _W1_VERSIONS:
    if _version == "v030":
        _W1_VARIANTS[f"{_version}/bare"] = _w1_variant_profile(_version, None)
        continue
    for _left, _right in product(_W1_HANDS, repeat=2):
        if (_left == "BRAINCO_HAND") != (_right == "BRAINCO_HAND"):
            continue
        _left_label = _W1_HANDS[_left][0]
        _right_label = _W1_HANDS[_right][0]
        _label = _left_label if _left == _right else f"L={_left_label},R={_right_label}"
        _W1_VARIANTS[f"{_version}/{_label}"] = _w1_variant_profile(
            _version, (_left, _right)
        )
    _W1_VARIANTS[f"{_version}/bare"] = _w1_variant_profile(_version, None)
ROBOT_CONFIGS["DexforceW1"].update(
    variants=_W1_VARIANTS, default_variant="v021/BrainCo"
)

_UR_VARIANTS = {
    variant: {
        **_DEFAULT_CONTROL_INFO["URRobot"],
        "mimic_joints": {},
        "control_part_mimics": {},
        "supports_dual_arm": True,
        "defaults": {"robot_type": variant},
    }
    for variant in ("ur3", "ur3e", "ur5", "ur5e", "ur10", "ur10e")
}
ROBOT_CONFIGS["URRobot"].update(variants=_UR_VARIANTS, default_variant="ur10")
for _variant, _profile in _UR_VARIANTS.items():
    ROBOT_CONFIGS[_variant] = {
        "class_path": _CONFIG_CLASSES["URRobot"],
        **_profile,
        **dict(zip(("vendor", "morphology", "arm_count"), _CLASSIFICATION["URRobot"])),
        "variant_of": "URRobot",
    }
