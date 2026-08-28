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

from unittest.mock import Mock, patch

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    CommandAckStatus,
    create_simulation_atomic_action_engine,
    EndpointCommand,
    EndpointCommandTransport,
    JointPositionPayload,
    JointPositionTarget,
    RigidObjectSceneProvider,
    RigidObjectSceneProviderCfg,
    RuntimeCommandFrame,
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
    robot.get_proprioception.return_value = {}
    return simulation, robot


def _command(
    *,
    env_ids: torch.Tensor | None = None,
    active_mask: torch.Tensor | None = None,
) -> RuntimeCommandFrame:
    return RuntimeCommandFrame(
        commands=(
            EndpointCommand(
                target=JointPositionTarget(
                    control_part="arm",
                    joint_ids=tuple(range(ROBOT_DOF)),
                ),
                payload=JointPositionPayload(
                    positions=torch.ones(BATCH_SIZE, ROBOT_DOF),
                    velocities=torch.full((BATCH_SIZE, ROBOT_DOF), 0.5),
                ),
            ),
        ),
        active_mask=(
            torch.tensor([True, False]) if active_mask is None else active_mask
        ),
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


def test_simulation_adapter_is_joint_position_transport() -> None:
    simulation, robot = _simulation_and_robot()
    adapter = SimulationExecutionAdapter(simulation, robot)

    assert isinstance(adapter, EndpointCommandTransport)
    assert adapter.transport_id == JointPositionTarget.TRANSPORT_ID
    assert adapter.payload_type is JointPositionPayload


@pytest.mark.parametrize("error", [AttributeError, NotImplementedError])
def test_simulation_adapter_treats_unavailable_effort_as_optional(
    error: type[Exception],
) -> None:
    simulation, robot = _simulation_and_robot()
    robot.get_qf.side_effect = error("effort unavailable")
    adapter = SimulationExecutionAdapter(simulation, robot)

    context = adapter.observe(TaskState.empty(BATCH_SIZE, "cpu"))

    assert context.robot.qeffort is None


def test_simulation_adapter_falls_back_to_proprioception_effort() -> None:
    simulation, robot = _simulation_and_robot()
    robot.get_qf.side_effect = AttributeError("effort unavailable")
    expected_qeffort = torch.full((BATCH_SIZE, ROBOT_DOF), 0.3)
    robot.get_proprioception.return_value = {"qf": expected_qeffort}
    adapter = SimulationExecutionAdapter(simulation, robot)

    context = adapter.observe(TaskState.empty(BATCH_SIZE, "cpu"))

    assert torch.equal(context.robot.qeffort, expected_qeffort)


def test_simulation_adapter_sends_active_rows_and_inactive_holds_together() -> None:
    simulation, robot = _simulation_and_robot()
    adapter = SimulationExecutionAdapter(simulation, robot)
    command = _command()

    acknowledgement = adapter.send(command, timeout=1.0)

    assert acknowledgement.status is CommandAckStatus.ACCEPTED
    sent_qpos = robot.set_qpos.call_args.args[0]
    sent_qvel = robot.set_qvel.call_args.args[0]
    expected_qpos = torch.zeros(BATCH_SIZE, ROBOT_DOF)
    expected_qpos[0] = 1.0
    expected_qvel = torch.zeros(BATCH_SIZE, ROBOT_DOF)
    expected_qvel[0] = 0.5
    assert torch.equal(sent_qpos, expected_qpos)
    assert torch.equal(sent_qvel, expected_qvel)
    endpoint_command = command.commands[0]
    assert isinstance(endpoint_command.target, JointPositionTarget)
    assert endpoint_command.target.target_id == "arm"
    assert endpoint_command.target.joint_ids == tuple(range(ROBOT_DOF))
    assert isinstance(endpoint_command.payload, JointPositionPayload)
    assert robot.set_qpos.call_args.kwargs["env_ids"] == [0, 1]
    assert robot.set_qpos.call_args.kwargs["joint_ids"] == [0, 1, 2]
    assert robot.set_qvel.call_args.kwargs["env_ids"] == [0, 1]
    assert robot.set_qvel.call_args.kwargs["joint_ids"] == [0, 1, 2]


def test_simulation_adapter_writes_disjoint_joint_endpoints_independently() -> None:
    simulation, robot = _simulation_and_robot()
    adapter = SimulationExecutionAdapter(simulation, robot)
    command = RuntimeCommandFrame(
        commands=(
            EndpointCommand(
                target=JointPositionTarget("arm", (0, 2)),
                payload=JointPositionPayload(torch.tensor([[1.0, 3.0], [4.0, 6.0]])),
            ),
            EndpointCommand(
                target=JointPositionTarget("tool", (1,)),
                payload=JointPositionPayload(torch.tensor([[2.0], [5.0]])),
            ),
        ),
        active_mask=torch.ones(BATCH_SIZE, dtype=torch.bool),
        env_ids=torch.arange(BATCH_SIZE, dtype=torch.long),
        hold_duration=torch.zeros(BATCH_SIZE),
    )

    acknowledgement = adapter.send(command, timeout=1.0)

    assert acknowledgement.status is CommandAckStatus.ACCEPTED
    assert robot.set_qpos.call_count == 2
    arm_call, tool_call = robot.set_qpos.call_args_list
    assert torch.equal(
        arm_call.args[0],
        torch.tensor([[1.0, 3.0], [4.0, 6.0]]),
    )
    assert arm_call.kwargs == {"joint_ids": [0, 2], "env_ids": [0, 1]}
    assert torch.equal(tool_call.args[0], torch.tensor([[2.0], [5.0]]))
    assert tool_call.kwargs == {"joint_ids": [1], "env_ids": [0, 1]}
    robot.set_qvel.assert_not_called()


def test_simulation_adapter_neutralizes_inactive_rows_without_velocity_payload() -> (
    None
):
    simulation, robot = _simulation_and_robot()
    robot.get_qvel.return_value = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    adapter = SimulationExecutionAdapter(simulation, robot)
    command = RuntimeCommandFrame(
        commands=(
            EndpointCommand(
                target=JointPositionTarget("arm", (0, 2)),
                payload=JointPositionPayload(torch.ones(BATCH_SIZE, 2)),
            ),
        ),
        active_mask=torch.tensor([True, False]),
        env_ids=torch.arange(BATCH_SIZE, dtype=torch.long),
        hold_duration=torch.zeros(BATCH_SIZE),
    )

    acknowledgement = adapter.send(command, timeout=1.0)

    assert acknowledgement.accepted
    assert torch.equal(
        robot.set_qvel.call_args.args[0],
        torch.tensor([[0.1, 0.3], [0.0, 0.0]]),
    )
    assert robot.set_qvel.call_args.kwargs == {
        "joint_ids": [0, 2],
        "env_ids": [0, 1],
    }


def test_simulation_adapter_send_writes_a_pure_hold_batch() -> None:
    simulation, robot = _simulation_and_robot()
    adapter = SimulationExecutionAdapter(simulation, robot)
    command = _command(active_mask=torch.zeros(BATCH_SIZE, dtype=torch.bool))

    acknowledgement = adapter.send(command, timeout=1.0)

    assert acknowledgement.status is CommandAckStatus.ACCEPTED
    assert torch.equal(
        robot.set_qpos.call_args.args[0],
        torch.zeros(BATCH_SIZE, ROBOT_DOF),
    )
    assert torch.equal(
        robot.set_qvel.call_args.args[0],
        torch.zeros(BATCH_SIZE, ROBOT_DOF),
    )
    assert robot.set_qpos.call_args.kwargs["env_ids"] == [0, 1]
    assert robot.set_qpos.call_args.kwargs["joint_ids"] == [0, 1, 2]
    assert robot.set_qvel.call_args.kwargs["env_ids"] == [0, 1]
    assert robot.set_qvel.call_args.kwargs["joint_ids"] == [0, 1, 2]


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
    observed_positions = torch.full((BATCH_SIZE, ROBOT_DOF), 0.25)
    robot.get_qpos.return_value = observed_positions
    adapter = SimulationExecutionAdapter(simulation, robot)
    command = _command()
    context = adapter.observe(TaskState.empty(BATCH_SIZE, "cpu"))

    acknowledgement = adapter.hold(command.targets, context, timeout=1.0)

    assert acknowledgement.status is CommandAckStatus.ACCEPTED
    assert torch.equal(robot.set_qpos.call_args.args[0], observed_positions)
    assert torch.equal(
        robot.set_qvel.call_args.args[0],
        torch.zeros_like(observed_positions),
    )
    assert robot.set_qpos.call_args.kwargs["env_ids"] == [0, 1]
    assert robot.set_qpos.call_args.kwargs["joint_ids"] == [0, 1, 2]
    assert robot.set_qvel.call_args.kwargs["env_ids"] == [0, 1]
    assert robot.set_qvel.call_args.kwargs["joint_ids"] == [0, 1, 2]


def test_simulation_adapter_hold_scopes_write_to_target_joint_ids() -> None:
    simulation, robot = _simulation_and_robot()
    observed_positions = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    robot.get_qpos.return_value = observed_positions
    adapter = SimulationExecutionAdapter(simulation, robot)
    context = adapter.observe(TaskState.empty(BATCH_SIZE, "cpu"))

    acknowledgement = adapter.hold(
        (JointPositionTarget("tool", (1,)),),
        context,
        timeout=1.0,
    )

    assert acknowledgement.status is CommandAckStatus.ACCEPTED
    assert torch.equal(
        robot.set_qpos.call_args.args[0],
        torch.tensor([[0.2], [0.5]]),
    )
    assert robot.set_qpos.call_args.kwargs == {
        "joint_ids": [1],
        "env_ids": [0, 1],
    }
    assert torch.equal(robot.set_qvel.call_args.args[0], torch.zeros(BATCH_SIZE, 1))


def test_simulation_adapter_cancel_validates_transport_targets() -> None:
    simulation, robot = _simulation_and_robot()
    adapter = SimulationExecutionAdapter(simulation, robot)
    targets = _command().targets

    acknowledgement = adapter.cancel(targets, timeout=1.0)

    assert acknowledgement.status is CommandAckStatus.ACCEPTED
    assert [(target.transport_id, target.target_id) for target in targets] == [
        (JointPositionTarget.TRANSPORT_ID, "arm")
    ]
    robot.set_qpos.assert_not_called()

    invalid = adapter.cancel(
        (JointPositionTarget("invalid", (ROBOT_DOF,)),),
        timeout=1.0,
    )
    assert invalid.status is CommandAckStatus.REJECTED
    assert "outside robot DOF" in invalid.message


def test_simulation_adapter_sleep_advances_integral_physics_steps() -> None:
    simulation, robot = _simulation_and_robot()
    adapter = SimulationExecutionAdapter(simulation, robot)

    adapter.sleep(0.025)

    simulation.update.assert_called_once_with(physics_dt=PHYSICS_DT, step=3)
    assert adapter.now() == pytest.approx(0.03)


def test_simulation_adapter_sleep_absorbs_float32_roundoff() -> None:
    simulation, robot = _simulation_and_robot()
    adapter = SimulationExecutionAdapter(simulation, robot, physics_dt=0.1)
    duration = float(torch.tensor(0.2, dtype=torch.float32))

    adapter.sleep(duration)

    simulation.update.assert_called_once_with(physics_dt=0.1, step=2)
    assert adapter.now() == pytest.approx(0.2)


def test_simulation_adapter_supplies_time_and_ids_to_scene_provider() -> None:
    simulation, robot = _simulation_and_robot()
    timestamps: list[float] = []
    observed_env_ids: list[torch.Tensor] = []

    class RecordingSceneProvider:
        """Record adapter correlation arguments for this test."""

        def snapshot(
            self,
            *,
            timestamp: float,
            env_ids: torch.Tensor,
        ) -> SceneSnapshot:
            timestamps.append(timestamp)
            observed_env_ids.append(env_ids.clone())
            return SceneSnapshot(timestamp=timestamp, version=3)

    adapter = SimulationExecutionAdapter(
        simulation,
        robot,
        scene_provider=RecordingSceneProvider(),
    )
    adapter.sleep(PHYSICS_DT)

    context = adapter.observe(TaskState.empty(BATCH_SIZE, "cpu"))

    assert timestamps == pytest.approx([PHYSICS_DT])
    assert torch.equal(observed_env_ids[0], torch.arange(BATCH_SIZE))
    assert context.scene.version == 3


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


def test_simulation_adapter_rejects_two_scene_sources() -> None:
    simulation, robot = _simulation_and_robot()

    with pytest.raises(ValueError, match="mutually exclusive"):
        SimulationExecutionAdapter(
            simulation,
            robot,
            scene_provider=Mock(),
            scene_supplier=Mock(),
        )


def test_rigid_object_scene_provider_tracks_per_environment_collision_revision() -> (
    None
):
    obstacle = Mock()
    initial_pose = torch.eye(4).repeat(BATCH_SIZE, 1, 1)
    obstacle.get_local_pose.return_value = initial_pose
    provider = RigidObjectSceneProvider(
        {"obstacle": obstacle},
        collision_entity_ids=("obstacle",),
    )
    env_ids = torch.arange(BATCH_SIZE, dtype=torch.long)

    initial = provider.snapshot(timestamp=0.0, env_ids=env_ids)
    moved_pose = initial_pose.clone()
    moved_pose[1, 0, 3] = 0.01
    obstacle.get_local_pose.return_value = moved_pose
    changed = provider.snapshot(timestamp=PHYSICS_DT, env_ids=env_ids)

    assert initial.version == 0
    assert initial.collision_world_revisions(BATCH_SIZE) == (0, 0)
    assert changed.version == 1
    assert changed.collision_world_revisions(BATCH_SIZE) == (0, 1)
    assert changed.collision_entity_ids == ("obstacle",)
    assert torch.equal(changed.entities["obstacle"].pose, moved_pose)


def test_simulation_engine_factory_registers_selected_entity_uids() -> None:
    cube = Mock(uid="cube")
    cube_pose = torch.eye(4).repeat(BATCH_SIZE, 1, 1)
    cube.get_local_pose.return_value = cube_pose
    motion_generator = Mock()
    control_profiles = {"hand": Mock()}
    grasp_pose_generators = {"hand": Mock()}
    tracking_runtime = Mock()

    with patch(
        "embodichain.lab.sim.atomic_actions.sim_adapter.AtomicActionEngine"
    ) as engine_type:
        engine = create_simulation_atomic_action_engine(
            motion_generator,
            (cube,),
            control_profiles,
            grasp_pose_generators,
            load_builtins=False,
            tracking_runtime=tracking_runtime,
        )

    assert engine is engine_type.return_value
    kwargs = engine_type.call_args.kwargs
    assert kwargs["control_profiles"] is control_profiles
    assert kwargs["grasp_pose_generators"] is grasp_pose_generators
    assert kwargs["load_builtins"] is False
    assert kwargs["tracking_runtime"] is tracking_runtime
    scene = kwargs["scene_provider"].snapshot(
        timestamp=0.0,
        env_ids=torch.arange(BATCH_SIZE),
    )
    assert tuple(scene.entities) == ("cube",)
    assert torch.equal(scene.entities["cube"].pose, cube_pose)


def test_simulation_engine_factory_rejects_duplicate_entity_uids() -> None:
    with pytest.raises(ValueError, match="Duplicate scene entity uid 'cube'"):
        create_simulation_atomic_action_engine(
            Mock(),
            (Mock(uid="cube"), Mock(uid="cube")),
        )


@pytest.mark.parametrize(
    "value",
    [-1.0, float("nan"), float("inf"), float("-inf")],
)
def test_rigid_object_scene_provider_cfg_rejects_invalid_translation_threshold(
    value: float,
) -> None:
    with pytest.raises(ValueError, match="translation_threshold"):
        RigidObjectSceneProviderCfg(translation_threshold=value)


@pytest.mark.parametrize(
    "value",
    [-1.0, float("nan"), float("inf"), float("-inf")],
)
def test_rigid_object_scene_provider_cfg_rejects_invalid_rotation_threshold(
    value: float,
) -> None:
    with pytest.raises(ValueError, match="rotation_threshold"):
        RigidObjectSceneProviderCfg(rotation_threshold=value)


def test_rigid_object_scene_provider_filters_subthreshold_pose_noise() -> None:
    obstacle = Mock()
    initial_pose = torch.eye(4).repeat(BATCH_SIZE, 1, 1)
    obstacle.get_local_pose.return_value = initial_pose
    provider = RigidObjectSceneProvider(
        {"obstacle": obstacle},
        collision_entity_ids=("obstacle",),
        cfg=RigidObjectSceneProviderCfg(translation_threshold=0.01),
    )
    env_ids = torch.arange(BATCH_SIZE, dtype=torch.long)
    provider.snapshot(timestamp=0.0, env_ids=env_ids)
    noisy_pose = initial_pose.clone()
    noisy_pose[:, 0, 3] = 0.001
    obstacle.get_local_pose.return_value = noisy_pose

    unchanged = provider.snapshot(timestamp=PHYSICS_DT, env_ids=env_ids)

    assert unchanged.version == 0
    assert unchanged.collision_world_revisions(BATCH_SIZE) == (0, 0)


def test_rigid_object_scene_provider_accumulates_subthreshold_translation() -> None:
    obstacle = Mock()
    pose = torch.eye(4).repeat(BATCH_SIZE, 1, 1)
    obstacle.get_local_pose.return_value = pose
    provider = RigidObjectSceneProvider(
        {"obstacle": obstacle},
        collision_entity_ids=("obstacle",),
        cfg=RigidObjectSceneProviderCfg(translation_threshold=0.01),
    )
    env_ids = torch.arange(BATCH_SIZE, dtype=torch.long)
    provider.snapshot(timestamp=0.0, env_ids=env_ids)

    pose = pose.clone()
    pose[1, 0, 3] = 0.006
    obstacle.get_local_pose.return_value = pose
    first = provider.snapshot(timestamp=PHYSICS_DT, env_ids=env_ids)

    pose = pose.clone()
    pose[1, 0, 3] = 0.012
    obstacle.get_local_pose.return_value = pose
    second = provider.snapshot(timestamp=2 * PHYSICS_DT, env_ids=env_ids)

    assert first.version == 0
    assert first.collision_world_revisions(BATCH_SIZE) == (0, 0)
    assert second.version == 1
    assert second.collision_world_revisions(BATCH_SIZE) == (0, 1)


def test_rigid_object_scene_provider_accumulates_subthreshold_rotation() -> None:
    obstacle = Mock()
    pose = torch.eye(4).repeat(BATCH_SIZE, 1, 1)
    obstacle.get_local_pose.return_value = pose
    provider = RigidObjectSceneProvider(
        {"obstacle": obstacle},
        collision_entity_ids=("obstacle",),
        cfg=RigidObjectSceneProviderCfg(rotation_threshold=0.1),
    )
    env_ids = torch.arange(BATCH_SIZE, dtype=torch.long)
    provider.snapshot(timestamp=0.0, env_ids=env_ids)

    for index, angle in enumerate((0.06, 0.12), start=1):
        cosine = torch.cos(torch.tensor(angle))
        sine = torch.sin(torch.tensor(angle))
        rotated = torch.eye(4).repeat(BATCH_SIZE, 1, 1)
        rotated[0, :2, :2] = torch.tensor(
            [[cosine, -sine], [sine, cosine]], dtype=torch.float32
        )
        obstacle.get_local_pose.return_value = rotated
        snapshot = provider.snapshot(
            timestamp=index * PHYSICS_DT,
            env_ids=env_ids,
        )

    assert snapshot.version == 1
    assert snapshot.collision_world_revisions(BATCH_SIZE) == (1, 0)


def test_simulation_adapter_rejects_changed_environment_identity() -> None:
    simulation, robot = _simulation_and_robot()
    adapter = SimulationExecutionAdapter(simulation, robot)
    command = _command(env_ids=torch.tensor([1, 0], dtype=torch.long))

    acknowledgement = adapter.send(command, timeout=1.0)

    assert acknowledgement.status is CommandAckStatus.REJECTED
    assert "env_ids" in acknowledgement.message
    robot.set_qpos.assert_not_called()
