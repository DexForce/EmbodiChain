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

"""Tests for fixed-cadence standalone joint trajectory playback."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.sim.motion.execution import (
    JointTrajectoryPlaybackCfg,
    play_joint_trajectory,
)


class _FakeRobot:
    def __init__(self) -> None:
        self.qpos_commands: list[torch.Tensor] = []
        self.qvel_commands: list[torch.Tensor] = []

    def set_qpos(self, qpos: torch.Tensor, *, joint_ids: list[int]) -> None:
        assert joint_ids == [1]
        self.qpos_commands.append(qpos.clone())

    def set_qvel(self, qvel: torch.Tensor, *, joint_ids: list[int]) -> None:
        assert joint_ids == [1]
        self.qvel_commands.append(qvel.clone())


class _FakeSimulation:
    def __init__(self, physics_dt: float) -> None:
        self.sim_config = SimpleNamespace(physics_dt=physics_dt)
        self.updates: list[tuple[float, int]] = []

    def update(self, physics_dt: float, step: int) -> None:
        self.updates.append((physics_dt, step))


def test_playback_uses_fixed_physics_dt_and_integer_substeps() -> None:
    sim = _FakeSimulation(physics_dt=0.02)
    robot = _FakeRobot()

    play_joint_trajectory(
        sim,
        robot,
        positions=torch.tensor([[[0.0], [1.0]]]),
        dt=torch.tensor([[0.0, 0.07]]),
        joint_ids=[1],
        cfg=JointTrajectoryPlaybackCfg(
            control_dt=0.04,
            joint_command_mode="position_velocity",
        ),
    )

    assert sim.updates == [(0.02, 2), (0.02, 2)]
    torch.testing.assert_close(
        torch.cat(robot.qpos_commands, dim=0)[:, 0],
        torch.tensor([0.0, 0.5, 1.0]),
    )
    assert len(robot.qvel_commands) == 3
    assert torch.equal(robot.qvel_commands[-1], torch.zeros(1, 1))


def test_playback_defaults_control_period_to_physics_period() -> None:
    sim = _FakeSimulation(physics_dt=0.02)
    robot = _FakeRobot()

    play_joint_trajectory(
        sim,
        robot,
        positions=torch.tensor([[[0.0], [1.0]]]),
        dt=torch.tensor([[0.0, 0.04]]),
        joint_ids=[1],
    )

    assert sim.updates == [(0.02, 1), (0.02, 1)]
    assert robot.qvel_commands == []


def test_playback_rejects_control_period_not_on_physics_grid() -> None:
    with pytest.raises(ValueError, match="integer multiple"):
        play_joint_trajectory(
            _FakeSimulation(physics_dt=0.02),
            _FakeRobot(),
            positions=torch.tensor([[[0.0], [1.0]]]),
            dt=torch.tensor([[0.0, 0.04]]),
            joint_ids=[1],
            cfg=JointTrajectoryPlaybackCfg(control_dt=0.03),
        )


def test_position_playback_does_not_write_velocity_targets() -> None:
    robot = _FakeRobot()

    play_joint_trajectory(
        _FakeSimulation(physics_dt=0.02),
        robot,
        positions=torch.tensor([[[0.0], [1.0]]]),
        dt=torch.tensor([[0.0, 0.02]]),
        joint_ids=[1],
        cfg=JointTrajectoryPlaybackCfg(joint_command_mode="position"),
    )

    assert robot.qvel_commands == []
