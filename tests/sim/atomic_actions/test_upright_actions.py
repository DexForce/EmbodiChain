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

"""Tests for UprightAction and UprightActionCfg."""

from __future__ import annotations

from unittest.mock import Mock

import torch

from embodichain.lab.sim.atomic_actions.actions import (
    PickUpAction,
    PickUpActionCfg,
    UprightAction,
    UprightActionCfg,
)
from embodichain.lab.sim.atomic_actions.core import AntipodalAffordance, ObjectSemantics

NUM_ENVS = 2
ARM_DOF = 6
HAND_DOF = 2
TOTAL_DOF = ARM_DOF + HAND_DOF


def _make_mock_robot(
    num_envs: int = NUM_ENVS,
    arm_dof: int = ARM_DOF,
    hand_dof: int = HAND_DOF,
) -> Mock:
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.dof = arm_dof + hand_dof

    def get_qpos(name=None):
        if name == "arm":
            return torch.zeros(num_envs, arm_dof)
        if name == "hand":
            return torch.zeros(num_envs, hand_dof)
        return torch.zeros(num_envs, arm_dof + hand_dof)

    robot.get_qpos = get_qpos

    def get_joint_ids(name=None):
        if name == "arm":
            return list(range(arm_dof))
        if name == "hand":
            return list(range(arm_dof, arm_dof + hand_dof))
        return list(range(arm_dof + hand_dof))

    robot.get_joint_ids = get_joint_ids
    return robot


def _make_mock_motion_generator(robot: Mock | None = None) -> Mock:
    mg = Mock()
    mg.robot = robot or _make_mock_robot()
    mg.device = mg.robot.device
    return mg


class TestUprightAction:
    def setup_method(self):
        self.robot = _make_mock_robot()
        self.mg = _make_mock_motion_generator(self.robot)

    def _make_cfg(self, cfg_cls, **overrides):
        defaults = dict(
            hand_open_qpos=torch.tensor([0.0, 0.0]),
            hand_close_qpos=torch.tensor([0.025, 0.025]),
            control_part="arm",
            hand_control_part="hand",
            pre_grasp_distance=0.15,
            lift_height=0.15,
            approach_direction=torch.tensor([0.0, 0.0, -1.0]),
        )
        defaults.update(overrides)
        return cfg_cls(**defaults)

    def test_cfg_inherits_pickup_cfg_and_renames_action(self):
        cfg = self._make_cfg(UprightActionCfg)
        assert isinstance(cfg, PickUpActionCfg)
        assert cfg.name == "upright"

    def test_init_matches_pickup_action_shape_and_joints(self):
        upright_cfg = self._make_cfg(UprightActionCfg)
        pickup_cfg = self._make_cfg(PickUpActionCfg)

        upright = UprightAction(self.mg, cfg=upright_cfg)
        pickup = PickUpAction(self.mg, cfg=pickup_cfg)

        assert isinstance(upright.cfg, UprightActionCfg)
        assert upright.cfg.name == "upright"
        assert upright.hand_joint_ids == pickup.hand_joint_ids
        assert upright.joint_ids == pickup.joint_ids
        assert upright.arm_dof == pickup.arm_dof
        assert upright.dof == TOTAL_DOF
        assert upright.dof == pickup.dof

    def test_build_upright_object_pose_aligns_local_z_to_world_z(self):
        cfg = self._make_cfg(UprightActionCfg, place_clearance=0.01)
        upright = UprightAction(self.mg, cfg=cfg)
        obj_poses = torch.eye(4).unsqueeze(0).repeat(NUM_ENVS, 1, 1)
        obj_poses[:, :3, :3] = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        obj_poses[:, :3, 3] = torch.tensor([[0.5, 0.1, 0.02], [0.6, -0.1, 0.03]])
        mesh_vertices = torch.tensor(
            [
                [-0.03, -0.03, -0.02],
                [0.03, 0.03, 0.18],
            ]
        )
        semantics = ObjectSemantics(
            label="bottle",
            affordance=AntipodalAffordance(),
            geometry={"mesh_vertices": mesh_vertices},
        )

        upright_pose = upright._build_upright_object_pose(semantics, obj_poses)

        expected_z_axis = torch.tensor([0.0, 0.0, 1.0]).repeat(NUM_ENVS, 1)
        assert torch.allclose(upright_pose[:, :3, 2], expected_z_axis)
        assert torch.allclose(upright_pose[:, :2, 3], obj_poses[:, :2, 3])
        assert torch.allclose(
            upright_pose[:, 2, 3],
            torch.full((NUM_ENVS,), 0.03),
        )
