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

"""Tests for the simulation execution-runner adapter."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    CommandAckStatus,
    JointCommand,
    SceneSnapshot,
    SimulationExecutionAdapter,
    TaskState,
)

BATCH_SIZE = 2
ROBOT_DOF = 3
PHYSICS_DT = 0.01


def _simulation_and_robot() -> tuple[Mock, Mock]:
    simulation = Mock()
    simulation.sim_config.physics_dt = PHYSICS_DT
    robot = Mock()
    robot.get_qpos.return_value = torch.zeros(BATCH_SIZE, ROBOT_DOF)
    robot.get_qvel.return_value = torch.full((BATCH_SIZE, ROBOT_DOF), 0.1)
    robot.get_qf.return_value = torch.full((BATCH_SIZE, ROBOT_DOF), 0.2)
    return simulation, robot


def _command(*, env_ids: torch.Tensor | None = None) -> JointCommand:
    return JointCommand(
        positions=torch.ones(BATCH_SIZE, ROBOT_DOF),
        velocities=torch.full((BATCH_SIZE, ROBOT_DOF), 0.5),
        active_mask=torch.tensor([True, False]),
        env_ids=(
            torch.arange(BATCH_SIZE, dtype=torch.long) if env_ids is None else env_ids
        ),
        hold_duration=torch.full((BATCH_SIZE,), 0.1),
    )


def test_simulation_adapter_observes_full_robot_state() -> None:
    simulation, robot = _simulation_and_robot()
    adapter = SimulationExecutionAdapter(simulation, robot)

    context = adapter.observe(TaskState.empty(BATCH_SIZE, "cpu"))

    assert context.robot.timestamp == 0.0
    assert torch.equal(context.robot.qpos, robot.get_qpos.return_value)
    assert torch.equal(context.robot.qvel, robot.get_qvel.return_value)
    assert torch.equal(context.robot.qeffort, robot.get_qf.return_value)
    assert context.scene.version == 0


@pytest.mark.parametrize("error", [AttributeError, NotImplementedError])
def test_simulation_adapter_treats_unavailable_effort_as_optional(
    error: type[Exception],
) -> None:
    simulation, robot = _simulation_and_robot()
    robot.get_qf.side_effect = error("effort unavailable")
    adapter = SimulationExecutionAdapter(simulation, robot)

    context = adapter.observe(TaskState.empty(BATCH_SIZE, "cpu"))

    assert context.robot.qeffort is None


def test_simulation_adapter_sends_active_rows_and_inactive_holds_together() -> None:
    simulation, robot = _simulation_and_robot()
    adapter = SimulationExecutionAdapter(simulation, robot)
    command = _command()

    acknowledgement = adapter.send(command, timeout=1.0)

    assert acknowledgement.status is CommandAckStatus.ACCEPTED
    sent_qpos = robot.set_qpos.call_args.args[0]
    sent_qvel = robot.set_qvel.call_args.args[0]
    assert torch.equal(sent_qpos, command.positions)
    assert torch.equal(sent_qvel, command.velocities)
    assert robot.set_qpos.call_args.kwargs["env_ids"] == [0, 1]
    assert robot.set_qvel.call_args.kwargs["env_ids"] == [0, 1]


def test_simulation_adapter_keeps_stable_ids_separate_from_robot_indices() -> None:
    simulation, robot = _simulation_and_robot()
    stable_ids = torch.tensor([10, 20], dtype=torch.long)
    adapter = SimulationExecutionAdapter(simulation, robot, env_ids=stable_ids)
    command = _command(env_ids=stable_ids)

    acknowledgement = adapter.send(command, timeout=1.0)

    assert acknowledgement.status is CommandAckStatus.ACCEPTED
    assert robot.set_qpos.call_args.kwargs["env_ids"] == [0, 1]


def test_simulation_adapter_hold_targets_every_environment() -> None:
    simulation, robot = _simulation_and_robot()
    adapter = SimulationExecutionAdapter(simulation, robot)
    command = _command()

    acknowledgement = adapter.hold(command, timeout=1.0)

    assert acknowledgement.status is CommandAckStatus.ACCEPTED
    robot.set_qpos.assert_called_once_with(command.positions, env_ids=[0, 1])
    robot.set_qvel.assert_called_once_with(command.velocities, env_ids=[0, 1])


def test_simulation_adapter_sleep_advances_integral_physics_steps() -> None:
    simulation, robot = _simulation_and_robot()
    adapter = SimulationExecutionAdapter(simulation, robot)

    adapter.sleep(0.025)

    simulation.update.assert_called_once_with(physics_dt=PHYSICS_DT, step=3)
    assert adapter.now() == pytest.approx(0.03)


def test_simulation_adapter_supplies_elapsed_time_to_scene_callback() -> None:
    simulation, robot = _simulation_and_robot()
    timestamps: list[float] = []

    def scene_supplier(timestamp: float) -> SceneSnapshot:
        timestamps.append(timestamp)
        return SceneSnapshot(timestamp=timestamp, version=3)

    adapter = SimulationExecutionAdapter(
        simulation,
        robot,
        scene_supplier=scene_supplier,
    )
    adapter.sleep(PHYSICS_DT)

    context = adapter.observe(TaskState.empty(BATCH_SIZE, "cpu"))

    assert timestamps == pytest.approx([PHYSICS_DT])
    assert context.scene.version == 3


def test_simulation_adapter_rejects_changed_environment_identity() -> None:
    simulation, robot = _simulation_and_robot()
    adapter = SimulationExecutionAdapter(simulation, robot)
    command = _command(env_ids=torch.tensor([1, 0], dtype=torch.long))

    acknowledgement = adapter.send(command, timeout=1.0)

    assert acknowledgement.status is CommandAckStatus.REJECTED
    assert "env_ids" in acknowledgement.message
    robot.set_qpos.assert_not_called()
