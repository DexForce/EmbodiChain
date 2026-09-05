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

"""Regression coverage for qpos-only-qualified robot drive defaults."""

from __future__ import annotations

from embodichain.lab.sim.robots import CobotMagicCfg, FrankaPandaCfg, URRobotCfg
from embodichain.lab.sim.robots.dexforce_w1 import DexforceW1Cfg

_FRANKA_QPOS_DAMPING = {
    "fr3_joint[1-7]": 1e2,
    "fr3_finger_joint[1-2]": 2e1,
}
_COBOT_GRIPPER_STIFFNESS = 1e3
_COBOT_GRIPPER_DAMPING = 2e1
_UR10E_QPOS_DAMPING = 1e2
_UR10_DAMPING = 1e3
_W1_ARM_AND_HEAD_DAMPING = 1e2
_W1_HAND_STIFFNESS = 1e3
_W1_HAND_DAMPING = 2e1


def test_franka_default_damping_is_qpos_only_qualified() -> None:
    """Keep the Panda arm and hand damping at the qualified defaults."""
    cfg = FrankaPandaCfg.from_dict({})

    assert cfg.drive_pros.damping == _FRANKA_QPOS_DAMPING


def test_ur10e_damping_does_not_change_other_ur_variants() -> None:
    """Scope the qpos-only adjustment to the qualified UR10e preset."""
    ur10e_cfg = URRobotCfg.from_dict({"robot_type": "ur10e"})
    ur10_cfg = URRobotCfg.from_dict({"robot_type": "ur10"})

    assert (
        ur10e_cfg.drive_pros.damping["arm"],
        ur10_cfg.drive_pros.damping["arm"],
    ) == (_UR10E_QPOS_DAMPING, _UR10_DAMPING)


def test_cobotmagic_gripper_drive_is_qpos_only_qualified() -> None:
    """Keep both CobotMagic grippers at the qualified drive values."""
    cfg = CobotMagicCfg.from_dict({})

    assert (
        cfg.drive_pros.stiffness["left_joint[7-8]"],
        cfg.drive_pros.stiffness["right_joint[7-8]"],
        cfg.drive_pros.damping["left_joint[7-8]"],
        cfg.drive_pros.damping["right_joint[7-8]"],
    ) == (
        _COBOT_GRIPPER_STIFFNESS,
        _COBOT_GRIPPER_STIFFNESS,
        _COBOT_GRIPPER_DAMPING,
        _COBOT_GRIPPER_DAMPING,
    )


def test_w1_arm_head_and_hand_drives_are_qpos_only_qualified() -> None:
    """Keep W1 qpos-only drive tuning separate by joint family."""
    cfg = DexforceW1Cfg.from_dict({})
    hands = "(LEFT|RIGHT)_HAND_(THUMB[12]|INDEX|MIDDLE|RING|PINKY)"

    assert (
        cfg.drive_pros.damping["(RIGHT|LEFT)_J[0-9]"],
        cfg.drive_pros.damping["(NECK1|NECK2)"],
        cfg.drive_pros.stiffness[hands],
        cfg.drive_pros.damping[hands],
    ) == (
        _W1_ARM_AND_HEAD_DAMPING,
        _W1_ARM_AND_HEAD_DAMPING,
        _W1_HAND_STIFFNESS,
        _W1_HAND_DAMPING,
    )
