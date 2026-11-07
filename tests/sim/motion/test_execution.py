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


@pytest.mark.parametrize("preserve_source_grid", [False, True])
def test_velocity_playback_keeps_each_rows_start_and_terminal_holds_at_rest(
    preserve_source_grid: bool,
) -> None:
    robot = _FakeRobot()
    positions = torch.tensor(
        [[[0.0], [1.0], [2.0], [3.0]], [[4.0], [5.0], [5.0], [5.0]], [[6.0]] * 4]
    )
    result = play_joint_trajectory(
        _FakeSimulation(physics_dt=0.01),
        robot,
        positions=positions,
        dt=torch.tensor([[0.0, 0.01, 0.01, 0.01], [0.0, 0.01, 0.0, 0.0], [0.0] * 4]),
        joint_ids=[1],
        cfg=JointTrajectoryPlaybackCfg(
            joint_command_mode="position_velocity",
            preserve_source_grid=preserve_source_grid,
        ),
    )

    dispatched = torch.stack(robot.qvel_commands, dim=1)
    torch.testing.assert_close(dispatched, result.velocities)
    assert torch.count_nonzero(dispatched[:, 0]) == 0
    assert torch.count_nonzero(dispatched[0, -1]) == 0
    assert torch.count_nonzero(dispatched[1, 1:]) == 0
    assert torch.count_nonzero(dispatched[2]) == 0
    # The longer row still moves at one joint unit per 10 ms in its interior.
    torch.testing.assert_close(dispatched[0, 1:-1], torch.full((2, 1), 100.0))


def test_velocity_playback_stops_at_explicit_holds_without_changing_the_clock() -> None:
    robot = _FakeRobot()
    sim = _FakeSimulation(physics_dt=0.01)
    observed = []
    positions = torch.arange(5, dtype=torch.float32).reshape(1, 5, 1)
    result = play_joint_trajectory(
        sim,
        robot,
        positions=positions,
        dt=torch.tensor([[0.0, 0.02, 0.02, 0.02, 0.02]]),
        joint_ids=[1],
        cfg=JointTrajectoryPlaybackCfg(
            joint_command_mode="position_velocity", preserve_source_grid=True
        ),
        hold_durations={2: 0.04, 4: 0.02},
        on_step=lambda index, count: observed.append((index, count)),
    )

    dispatched = torch.stack(robot.qvel_commands, dim=1)
    assert torch.count_nonzero(dispatched[:, [0, 2, 4]]) == 0
    assert dispatched[0, 1, 0] > 0 and dispatched[0, 3, 0] > 0
    torch.testing.assert_close(result.velocities[:, :5], dispatched)
    assert torch.count_nonzero(result.velocities[:, -1]) == 0
    assert torch.equal(torch.stack(robot.qpos_commands, dim=1), positions)
    assert observed == [(index, 5) for index in range(5)]
    assert float(result.dt.sum()) == pytest.approx(
        sum(dt * steps for dt, steps in sim.updates)
    )


def test_aligned_playback_preserves_zero_time_joins_callbacks_and_records_holds() -> (
    None
):
    sim = _FakeSimulation(physics_dt=0.01)
    robot = _FakeRobot()
    observed_steps = []
    source = torch.tensor([[[0.0], [0.1], [0.1], [0.2]]])
    result = play_joint_trajectory(
        sim,
        robot,
        positions=source,
        dt=torch.tensor([[0.0, 0.01, 0.0, 0.01]]),
        joint_ids=[1],
        cfg=JointTrajectoryPlaybackCfg(preserve_source_grid=True),
        on_step=lambda index, count: observed_steps.append((index, count)),
        hold_durations={1: 0.02, 3: 0.03},
    )
    assert observed_steps == [(index, 4) for index in range(4)]
    assert torch.equal(torch.stack(robot.qpos_commands, dim=1), source)
    assert sim.updates == [(0.01, 1), (0.01, 2), (0.01, 1), (0.01, 3)]
    assert float(result.dt.sum()) == pytest.approx(
        sum(dt * steps for dt, steps in sim.updates)
    )
    torch.testing.assert_close(result.dt, torch.tensor([[0.0, 0.01, 0.02, 0.01, 0.03]]))
    assert torch.equal(result.positions[:, -1], source[:, -1])


def test_aligned_playback_does_not_add_a_final_command_interval() -> None:
    sim = _FakeSimulation(physics_dt=0.01)
    result = play_joint_trajectory(
        sim,
        _FakeRobot(),
        positions=torch.tensor([[[0.0], [0.1], [0.2]]]),
        dt=torch.tensor([[0.0, 0.01, 0.01]]),
        joint_ids=[1],
        cfg=JointTrajectoryPlaybackCfg(preserve_source_grid=True),
    )
    assert sim.updates == [(0.01, 1), (0.01, 1)]
    assert float(result.dt.sum()) == pytest.approx(0.02)


def test_off_grid_holds_fail_before_dispatch() -> None:
    sim = _FakeSimulation(physics_dt=0.01)
    robot = _FakeRobot()
    with pytest.raises(ValueError, match="hold durations must align"):
        play_joint_trajectory(
            sim,
            robot,
            positions=torch.tensor([[[0.0], [0.1]]]),
            dt=torch.tensor([[0.0, 0.01]]),
            joint_ids=[1],
            cfg=JointTrajectoryPlaybackCfg(preserve_source_grid=True),
            hold_durations={0: 0.015},
        )
    assert not robot.qpos_commands and not sim.updates


def test_source_index_options_cannot_silently_target_resampled_interior_points() -> (
    None
):
    sim = _FakeSimulation(physics_dt=0.01)
    robot = _FakeRobot()
    with pytest.raises(ValueError, match="require preserve_source_grid"):
        play_joint_trajectory(
            sim,
            robot,
            positions=torch.tensor([[[0.0], [1.0]]]),
            dt=torch.tensor([[0.0, 0.02]]),
            joint_ids=[1],
            hold_durations={1: 0.03},
        )
    assert not robot.qpos_commands and not sim.updates
