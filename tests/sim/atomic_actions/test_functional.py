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

from __future__ import annotations

from unittest.mock import Mock

import torch

from embodichain.lab.sim.atomic_actions import (
    GripperActionCfg,
    gripper_close,
    gripper_open,
    move,
    pick_up,
    place,
)
from embodichain.lab.sim.atomic_actions import functional


NUM_ENVS = 2
ARM_DOF = 6
HAND_DOF = 2
TOTAL_DOF = ARM_DOF + HAND_DOF


def _make_mock_robot() -> Mock:
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.dof = TOTAL_DOF

    def get_qpos(name=None):
        if name == "arm":
            return torch.zeros(NUM_ENVS, ARM_DOF)
        if name == "hand":
            return torch.zeros(NUM_ENVS, HAND_DOF)
        return torch.zeros(NUM_ENVS, TOTAL_DOF)

    def get_joint_ids(name=None):
        if name == "arm":
            return list(range(ARM_DOF))
        if name == "hand":
            return list(range(ARM_DOF, TOTAL_DOF))
        return list(range(TOTAL_DOF))

    def compute_ik(pose=None, qpos_seed=None, name=None, joint_seed=None):
        seed = joint_seed if joint_seed is not None else qpos_seed
        if seed is None:
            seed = torch.zeros(NUM_ENVS, ARM_DOF)
        return True, seed.clone()

    robot.get_qpos = get_qpos
    robot.get_joint_ids = get_joint_ids
    robot.compute_ik = compute_ik
    return robot


def _make_motion_generator() -> Mock:
    motion_generator = Mock()
    motion_generator.robot = _make_mock_robot()
    motion_generator.device = torch.device("cpu")
    return motion_generator


class _FakeAction:
    calls: list[dict] = []

    def __init__(self, motion_generator, cfg):
        self.motion_generator = motion_generator
        self.cfg = cfg
        self.__class__.calls.append({"motion_generator": motion_generator, "cfg": cfg})

    def execute(self, target=None, start_qpos=None):
        self.__class__.calls[-1]["target"] = target
        self.__class__.calls[-1]["start_qpos"] = start_qpos
        return True, torch.ones(1, 2, 3), [4, 5, 6]


def test_move_passes_cfg_start_qpos_and_return(monkeypatch):
    _FakeAction.calls = []
    monkeypatch.setattr(functional, "MoveAction", _FakeAction)
    motion_generator = _make_motion_generator()
    target = torch.eye(4)
    start_qpos = torch.arange(ARM_DOF, dtype=torch.float32)

    result = move(
        motion_generator=motion_generator,
        target=target,
        start_qpos=start_qpos,
        control_part="arm",
        sample_interval=9,
        velocity_limit=0.5,
    )

    assert result[0] is True
    assert result[1].shape == (1, 2, 3)
    assert result[2] == [4, 5, 6]
    call = _FakeAction.calls[-1]
    assert call["motion_generator"] is motion_generator
    assert call["cfg"].control_part == "arm"
    assert call["cfg"].sample_interval == 9
    assert call["cfg"].velocity_limit == 0.5
    assert call["target"] is target
    assert call["start_qpos"] is start_qpos


def test_pick_up_and_place_pass_cfg(monkeypatch):
    _FakeAction.calls = []
    monkeypatch.setattr(functional, "PickUpAction", _FakeAction)
    monkeypatch.setattr(functional, "PlaceAction", _FakeAction)
    motion_generator = _make_motion_generator()
    hand_open = torch.tensor([0.0, 0.0])
    hand_close = torch.tensor([0.02, 0.02])
    target = torch.eye(4)
    start_qpos = torch.zeros(ARM_DOF)

    pick_up(
        motion_generator=motion_generator,
        target=target,
        start_qpos=start_qpos,
        hand_open_qpos=hand_open,
        hand_close_qpos=hand_close,
        pre_grasp_distance=0.12,
        lift_height=0.18,
        sample_interval=21,
        hand_interp_steps=7,
    )
    place(
        motion_generator=motion_generator,
        target=target,
        start_qpos=start_qpos,
        hand_open_qpos=hand_open,
        hand_close_qpos=hand_close,
        lift_height=0.11,
        sample_interval=19,
        hand_interp_steps=6,
    )

    pickup_cfg = _FakeAction.calls[-2]["cfg"]
    place_cfg = _FakeAction.calls[-1]["cfg"]
    assert pickup_cfg.name == "pick_up"
    assert pickup_cfg.pre_grasp_distance == 0.12
    assert pickup_cfg.lift_height == 0.18
    assert pickup_cfg.sample_interval == 21
    assert pickup_cfg.hand_interp_steps == 7
    assert place_cfg.name == "place"
    assert place_cfg.lift_height == 0.11
    assert place_cfg.sample_interval == 19
    assert place_cfg.hand_interp_steps == 6


def test_move_real_return_shape_and_joint_ids(monkeypatch):
    motion_generator = _make_motion_generator()
    start_qpos = torch.ones(NUM_ENVS, ARM_DOF)

    def fake_plan(self, target_states_list, start_qpos, n_waypoints, arm_dof=None):
        return True, start_qpos.unsqueeze(1).repeat(1, n_waypoints, 1)

    monkeypatch.setattr(functional.MoveAction, "_plan_arm_trajectory", fake_plan)

    is_success, trajectory, joint_ids = move(
        motion_generator=motion_generator,
        target=torch.eye(4),
        start_qpos=start_qpos,
        sample_interval=5,
    )

    assert is_success is True
    assert trajectory.shape == (NUM_ENVS, 5, ARM_DOF)
    assert joint_ids == list(range(ARM_DOF))
    torch.testing.assert_close(trajectory[:, 0], start_qpos)


def test_gripper_open_close_real_targets_and_joint_ids():
    motion_generator = _make_motion_generator()
    start_qpos = torch.tensor([0.02, 0.02])
    open_qpos = torch.tensor([0.0, 0.0])
    close_qpos = torch.tensor([0.025, 0.025])

    open_success, open_traj, open_joint_ids = gripper_open(
        motion_generator=motion_generator,
        open_qpos=open_qpos,
        start_qpos=start_qpos,
        sample_interval=4,
    )
    close_success, close_traj, close_joint_ids = gripper_close(
        motion_generator=motion_generator,
        close_qpos=close_qpos,
        start_qpos=open_qpos,
        sample_interval=4,
    )

    assert open_success is True
    assert close_success is True
    assert open_joint_ids == list(range(ARM_DOF, TOTAL_DOF))
    assert close_joint_ids == list(range(ARM_DOF, TOTAL_DOF))
    torch.testing.assert_close(open_traj[:, 0], start_qpos.repeat(NUM_ENVS, 1))
    torch.testing.assert_close(open_traj[:, -1], open_qpos.repeat(NUM_ENVS, 1))
    torch.testing.assert_close(close_traj[:, -1], close_qpos.repeat(NUM_ENVS, 1))


def test_gripper_can_use_cfg_target_qpos():
    motion_generator = _make_motion_generator()
    start_qpos = torch.tensor([0.02, 0.02])
    target_qpos = torch.tensor([0.01, 0.015])
    cfg = GripperActionCfg(
        control_part="hand",
        target_qpos=target_qpos,
        sample_interval=3,
    )

    is_success, trajectory, joint_ids = gripper_open(
        motion_generator=motion_generator,
        cfg=cfg,
        start_qpos=start_qpos,
    )

    assert is_success is True
    assert joint_ids == list(range(ARM_DOF, TOTAL_DOF))
    assert trajectory.shape == (NUM_ENVS, 3, HAND_DOF)
    torch.testing.assert_close(trajectory[:, 0], start_qpos.repeat(NUM_ENVS, 1))
    torch.testing.assert_close(trajectory[:, -1], target_qpos.repeat(NUM_ENVS, 1))
