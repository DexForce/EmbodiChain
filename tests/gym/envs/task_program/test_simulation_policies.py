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

"""Tests for explicit simulation-backed segment policies."""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.task_program import TaskProgramCompiler, decode_task_program
from embodichain.lab.task_program.integrations import (
    RigidizedArticulationAntipodalGraspBinding,
    SimulationArticulationBinding,
    SimulationRigidizedArticulationObjectBinding,
    SimulationRigidObjectBinding,
    SimulationSceneBinding,
)
from embodichain.lab.gym.envs.task_program.bridge import (
    SegmentPostPolicyPort,
    SegmentValidatorPort,
)
from embodichain.lab.task_program.integrations.simulation.policies import (
    SimulationSegmentPolicyPort,
    SimulationSettleMonitorCfg,
)
from embodichain.lab.gym.envs.settling import DynamicSettleMonitorCfg
from embodichain.lab.sim.atomic_actions import EntityState
from embodichain.lab.task_program.semantics.scene import (
    SceneAffordanceRef,
    SceneArticulationRef,
    SceneEntityRegistration,
    SceneObjectRef,
    SceneRegistry,
)


class _StaticStateProvider:
    """Provide an inert object state for provider-free compilation."""

    def observe(
        self,
        *,
        timestamp: float,
        env_ids: torch.Tensor,
    ) -> EntityState:
        del timestamp
        return EntityState(
            torch.eye(4, device=env_ids.device).expand(env_ids.numel(), -1, -1)
        )


class _RigidObject:
    """Small live rigid-object double with mutable velocities and poses."""

    def __init__(self, positions: torch.Tensor) -> None:
        batch_size = positions.shape[0]
        self.is_non_dynamic = False
        self.pose_reads = 0
        self.body_data = SimpleNamespace(
            lin_vel=torch.zeros(batch_size, 3),
            ang_vel=torch.zeros(batch_size, 3),
        )
        self._pose = torch.eye(4).expand(batch_size, -1, -1).clone()
        self._pose[:, :3, 3] = positions

    def get_local_pose(self, *, to_matrix: bool) -> torch.Tensor:
        assert to_matrix
        self.pose_reads += 1
        return self._pose.clone()


class _Robot:
    """Distinct current- and target-qpos source for post-policy holds."""

    def __init__(
        self,
        current_qpos: torch.Tensor,
        target_qpos: torch.Tensor | None = None,
    ) -> None:
        self.current_qpos = current_qpos.clone()
        self.target_qpos = (
            current_qpos.clone() if target_qpos is None else target_qpos.clone()
        )
        self.qpos_reads: list[bool] = []

    def get_qpos(self, target: bool = False) -> torch.Tensor:
        self.qpos_reads.append(target)
        return (self.target_qpos if target else self.current_qpos).clone()

    def get_joint_ids(self, *, name: str) -> list[int]:
        if name != "arm":
            raise KeyError(name)
        return [0]

    def compute_fk(self, *, qpos, name, to_matrix):
        assert name == "arm" and to_matrix
        return torch.eye(4).repeat(qpos.shape[0], 1, 1)


class _Articulation:
    """Small articulation double with one measured prismatic joint."""

    joint_names = ("cabinet_to_drawer",)

    def __init__(
        self, qpos: torch.Tensor, positions: torch.Tensor | None = None
    ) -> None:
        self._qpos = qpos
        self._pose = torch.eye(4).expand(qpos.shape[0], -1, -1).clone()
        if positions is not None:
            self._pose[:, :3, 3] = positions
        self.body_data = SimpleNamespace(
            body_link_vel=torch.zeros(qpos.shape[0], 1, 6),
        )

    def get_qpos(self) -> torch.Tensor:
        """Return the configured batched joint state."""
        return self._qpos.clone()

    def get_local_pose(self, *, to_matrix: bool) -> torch.Tensor:
        assert to_matrix
        return self._pose.clone()


class _Simulation:
    """Resolve only explicitly selected native test entities."""

    def __init__(
        self,
        entity: _RigidObject | None = None,
        reference: _RigidObject | None = None,
        articulation: _Articulation | None = None,
    ) -> None:
        self.entity = entity
        self.reference = reference
        self.articulation = articulation

    def get_rigid_object(self, uid: str) -> _RigidObject | None:
        if uid == "native_cube":
            return self.entity
        if uid == "native_reference":
            return self.reference
        return None

    def get_articulation(self, uid: str) -> _Articulation | None:
        return (
            self.articulation
            if uid in {"native_drawer", "native_cube_articulation"}
            else None
        )


def _compiled_segment(*, settle_preset: str = "fast"):
    """Compile one segment containing both supported policy types."""
    payload = {
        "program_id": "policy_test",
        "integration": {
            "robot_profile": "test_robot",
            "scene_registry": "test_scene",
            "runtime_preset": "safe",
        },
        "targets": {
            "drop": {
                "kind": "cyclic_pose",
                "values": [
                    {
                        "position": [0.0, 0.0, 0.0],
                        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                    }
                ],
            }
        },
        "program": {
            "kind": "segment",
            "name": "place",
            "steps": {
                "kind": "invoke",
                "call": {
                    "kind": "place",
                    "object": "cube",
                    "at": {"kind": "target_ref", "target": "drop"},
                },
            },
            "post": [
                {
                    "kind": "wait_stable",
                    "entity": "cube",
                    "preset": settle_preset,
                }
            ],
            "validators": [
                {
                    "kind": "object_near_target",
                    "object": "cube",
                    "target": "drop",
                    "position_tolerance": 0.05,
                }
            ],
        },
    }
    registry = SceneRegistry(
        (
            SceneEntityRegistration(
                ref=SceneObjectRef("cube"),
                state_provider=_StaticStateProvider(),
            ),
        )
    )
    compiled = TaskProgramCompiler.from_scene_registry(registry).compile(
        decode_task_program(payload)
    )
    return next(compiled.iter_segments())


def _compiled_articulation_segment(*, settle_preset=None):
    """Compile one standard joint-position validator."""
    payload = {
        "program_id": "joint_policy_test",
        "integration": {
            "robot_profile": "test_robot",
            "scene_registry": "test_scene",
            "runtime_preset": "safe",
        },
        "targets": {},
        "program": {
            "kind": "segment",
            "name": "open_drawer",
            "steps": {
                "kind": "invoke",
                "call": {
                    "kind": "registered",
                    "call_id": "example.slide",
                    "arguments": {},
                },
            },
            "validators": [
                {
                    "kind": "articulation_joint_position",
                    "articulation": "drawer",
                    "joint": "cabinet_to_drawer",
                    "minimum_position": 0.10,
                }
            ],
        },
    }
    if settle_preset is not None:
        payload["program"]["post"] = [
            {"kind": "wait_stable", "entity": "drawer", "preset": settle_preset}
        ]
    registry = SceneRegistry(
        (
            SceneEntityRegistration(
                ref=SceneArticulationRef("drawer"),
                state_provider=_StaticStateProvider(),
            ),
        )
    )
    compiled = TaskProgramCompiler.from_scene_registry(registry).compile(
        decode_task_program(payload)
    )
    return next(compiled.iter_segments())


def _compiled_relative_segment():
    """Compile one validator with two late-observed object identities."""
    payload = {
        "program_id": "relative_policy_test",
        "integration": {
            "robot_profile": "test_robot",
            "scene_registry": "test_scene",
            "runtime_preset": "safe",
        },
        "targets": {},
        "program": {
            "kind": "segment",
            "name": "place_relative",
            "steps": {
                "kind": "invoke",
                "call": {"kind": "pick", "object": "cube"},
            },
            "validators": [
                {
                    "kind": "object_near_relative_target",
                    "object": "cube",
                    "reference": "reference",
                    "displacement": [0.0, -0.1, 0.04],
                    "position_tolerance": 0.05,
                }
            ],
        },
    }
    registry = SceneRegistry(
        (
            SceneEntityRegistration(
                ref=SceneObjectRef("cube"),
                state_provider=_StaticStateProvider(),
            ),
            SceneEntityRegistration(
                ref=SceneObjectRef("reference"),
                state_provider=_StaticStateProvider(),
            ),
        )
    )
    compiled = TaskProgramCompiler.from_scene_registry(registry).compile(
        decode_task_program(payload)
    )
    return next(compiled.iter_segments())


def _port(
    positions: torch.Tensor,
    *,
    preset: DynamicSettleMonitorCfg | None = None,
    target_qpos: torch.Tensor | None = None,
    step_dt: float | None = None,
) -> tuple[SimulationSegmentPolicyPort, _RigidObject, _Robot]:
    """Build one policy port and expose its mutable test doubles."""
    entity = _RigidObject(positions)
    robot = _Robot(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        target_qpos=target_qpos,
    )
    port = SimulationSegmentPolicyPort(
        _Simulation(entity),
        robot,
        SimulationSceneBinding(
            registry_id="test_scene",
            rigid_objects=(
                SimulationRigidObjectBinding(
                    entity_id="cube",
                    simulation_uid="native_cube",
                ),
            ),
        ),
        settle_presets={
            "fast": preset
            or DynamicSettleMonitorCfg(
                min_steps=0,
                max_steps=3,
                check_interval_steps=1,
                required_stable_checks=2,
            )
        },
        step_dt=step_dt,
    )
    return port, entity, robot


def test_pose_delta_is_explicit_unresolved_first_and_preserves_partial_rows() -> None:
    cfg = SimulationSettleMonitorCfg(
        measurement="pose_delta",
        min_steps=0,
        max_steps=3,
        check_interval_steps=1,
        required_stable_checks=2,
    )
    port, entity, _ = _port(torch.zeros(2, 3), preset=cfg, step_dt=0.02)
    # Backend velocity-cache values do not select or override explicit pose evidence.
    entity.body_data.lin_vel.fill_(100)
    cfg.measurement = "reported_velocity"
    segment = _compiled_segment()
    policy = segment.post_policies[0]
    actions = port.actions(
        policy, segment=segment, active_mask=torch.ones(2, dtype=torch.bool)
    )
    count = 0
    for action in actions:
        count += 1
        assert action.shape == (2, 2)
        metadata = port.post_policy_metadata(policy, segment=segment)
        if count == 1:
            assert metadata["state"]["max_linear_speed"] == [None, None]
            assert metadata["state"]["settled_mask"] == [False, False]
        entity._pose[1, 0, 3] += 0.1
    assert count == 3
    assert port.post_policy_result(policy, segment=segment).tolist() == [True, False]
    metadata = port.post_policy_metadata(policy, segment=segment)
    assert metadata["measurement_source"] == "pose_delta"
    assert metadata["status"] == "timed_out"
    assert metadata["state"]["max_linear_speed"] == pytest.approx([0.0, 5.0])
    json.dumps(metadata, allow_nan=False)
    # A new episode starts with fresh temporal evidence; the old last pose is not reused.
    fresh = port.new_task_program_session()
    iterator = fresh.actions(
        policy, segment=segment, active_mask=torch.ones(2, dtype=torch.bool)
    )
    next(iterator)
    assert fresh.post_policy_metadata(policy, segment=segment)["state"][
        "max_linear_speed"
    ] == [None, None]
    iterator.close()


def test_pose_delta_requires_cadence_and_preserves_explicit_snapshot() -> None:
    cfg = SimulationSettleMonitorCfg(measurement="pose_delta")
    assert cfg.snapshot().measurement == "pose_delta"
    with pytest.raises(ValueError, match="step_dt"):
        _port(torch.zeros(2, 3), preset=cfg)
    cfg.measurement = "implicit_by_name"
    with pytest.raises(ValueError, match="measurement"):
        cfg.snapshot()


def test_pose_delta_wait_matches_frozen_b0_command_and_evidence_trace() -> None:
    frozen = json.loads(
        (Path(__file__).parent / "data/pose_delta_settle_b0.json").read_text()
    )
    port, entity, _ = _port(
        torch.zeros(2, 3),
        step_dt=0.02,
        preset=SimulationSettleMonitorCfg(
            measurement="pose_delta",
            min_steps=0,
            max_steps=3,
            check_interval_steps=1,
            required_stable_checks=2,
        ),
    )
    entity.body_data.lin_vel.fill_(100)
    segment = _compiled_segment()
    policy = segment.post_policies[0]
    actions = port.actions(
        policy, segment=segment, active_mask=torch.ones(2, dtype=torch.bool)
    )
    commands = []
    for pose, expected in zip(frozen["poses"], frozen["trace"], strict=True):
        entity._pose = torch.tensor(pose)
        try:
            commands.append(next(actions).tolist())
        except StopIteration:
            pass
        # The arbitrary preset name is the only declaration difference.
        expected["preset"] = "fast"
        assert port.post_policy_metadata(policy, segment=segment) == expected
    assert commands == frozen["commands"]
    assert port.post_policy_result(policy, segment=segment).tolist() == frozen["result"]


def _rigidized_port(
    positions: torch.Tensor,
) -> tuple[SimulationSegmentPolicyPort, _Articulation]:
    articulation = _Articulation(
        torch.zeros(positions.shape[0], 1), positions=positions
    )
    port = SimulationSegmentPolicyPort(
        _Simulation(articulation=articulation),
        _Robot(torch.zeros(positions.shape[0], 2)),
        SimulationSceneBinding(
            registry_id="test_scene",
            rigidized_articulations=(
                SimulationRigidizedArticulationObjectBinding(
                    entity_id="cube",
                    simulation_uid="native_cube_articulation",
                    locked_qpos={"cabinet_to_drawer": 0.0},
                ),
            ),
            rigidized_articulation_grasps=(
                RigidizedArticulationAntipodalGraspBinding(
                    entity_id="cube_grasp",
                    object_id="cube",
                    grasp_link="lower_two_layers",
                    native_name="lower_two_layers",
                    revision="1",
                ),
            ),
        ),
        settle_presets={
            "fast": DynamicSettleMonitorCfg(
                min_steps=0,
                max_steps=3,
                check_interval_steps=1,
                required_stable_checks=2,
            )
        },
    )
    return port, articulation


def test_rigidized_articulation_object_can_wait_for_stability() -> None:
    segment = _compiled_segment()
    port, _ = _rigidized_port(torch.zeros(2, 3))
    policy = segment.post_policies[0]

    port.validate_policy(policy, segment=segment)
    list(
        port.actions(
            policy,
            segment=segment,
            active_mask=torch.ones(2, dtype=torch.bool),
        )
    )

    assert port.post_policy_result(policy, segment=segment).tolist() == [True, True]


def test_rigidized_articulation_object_can_validate_target_position() -> None:
    segment = _compiled_segment()
    port, _ = _rigidized_port(torch.tensor([[0.01, 0.0, 0.0], [0.20, 0.0, 0.0]]))

    result = port.validate(segment.validators[0], segment=segment)

    assert result.tolist() == [True, False]


def test_rigidized_articulation_grasp_uses_parent_settle_target() -> None:
    segment = _compiled_segment()
    policy = replace(
        segment.post_policies[0],
        cfg=replace(segment.post_policies[0].cfg, entity="cube_grasp"),
        entity=SceneAffordanceRef("cube_grasp"),
    )
    segment = replace(segment, post_policies=(policy,))
    port, _ = _rigidized_port(torch.zeros(2, 3))

    port.validate_policy(policy, segment=segment)
    list(
        port.actions(
            policy,
            segment=segment,
            active_mask=torch.ones(2, dtype=torch.bool),
        )
    )

    assert port.post_policy_result(policy, segment=segment).tolist() == [True, True]


def test_port_implements_complete_bridge_policy_protocols() -> None:
    """One shared instance serves both complete policy boundaries."""
    port, _, _ = _port(torch.zeros(2, 3))

    assert isinstance(port, SegmentPostPolicyPort)
    assert isinstance(port, SegmentValidatorPort)
    assert port.settle_preset_ids == ("fast",)


def test_default_settle_presets_cover_rigid_objects_and_articulations() -> None:
    """Common entity kinds require no task-local settling configuration."""
    articulation = _Articulation(torch.zeros(2, 1))
    port = SimulationSegmentPolicyPort(
        _Simulation(articulation=articulation),
        _Robot(torch.zeros(2, 2)),
        SimulationSceneBinding(
            registry_id="test_scene",
            articulations=(
                SimulationArticulationBinding(
                    entity_id="drawer",
                    simulation_uid="native_drawer",
                ),
            ),
        ),
    )

    assert port.settle_preset_ids == ("rigid_object", "articulation")


def test_pure_preflight_validates_hooks_without_reading_live_state() -> None:
    """Static hook validation emits no hold and samples no pose or qpos."""
    segment = _compiled_segment()
    port, entity, robot = _port(torch.zeros(2, 3))

    port.validate_policy(segment.post_policies[0], segment=segment)
    port.validate_validator(segment.validators[0], segment=segment)

    assert robot.qpos_reads == [False]
    assert entity.pose_reads == 0


def test_pure_preflight_rejects_unknown_settle_preset_without_observation() -> None:
    """An unknown preset fails before policy iteration can sample live state."""
    segment = _compiled_segment(settle_preset="missing")
    port, entity, robot = _port(torch.zeros(2, 3))

    with pytest.raises(KeyError, match="Unknown settle preset 'missing'"):
        port.validate_policy(segment.post_policies[0], segment=segment)

    assert robot.qpos_reads == [False]
    assert entity.pose_reads == 0


def test_wait_stable_yields_fresh_target_qpos_holds_through_gym() -> None:
    """Settling preserves loaded drive targets with independently owned holds."""
    segment = _compiled_segment()
    target_qpos = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    port, _, robot = _port(
        torch.zeros(2, 3),
        target_qpos=target_qpos,
        preset=DynamicSettleMonitorCfg(
            min_steps=0,
            max_steps=4,
            check_interval_steps=1,
            required_stable_checks=3,
        ),
    )
    actions = port.actions(
        segment.post_policies[0],
        segment=segment,
        active_mask=torch.ones(2, dtype=torch.bool),
    )

    first = next(actions)
    assert torch.equal(first, target_qpos)
    assert not torch.equal(first, robot.current_qpos)
    first.fill_(99.0)

    second = next(actions)
    assert torch.equal(second, target_qpos)
    assert second.data_ptr() != first.data_ptr()
    with pytest.raises(StopIteration):
        next(actions)
    assert torch.equal(
        robot.current_qpos,
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
    assert torch.equal(robot.target_qpos, target_qpos)
    assert robot.qpos_reads == [False, True, True]

    metadata = port.post_policy_metadata(
        segment.post_policies[0],
        segment=segment,
    )
    assert metadata["status"] == "settled"
    assert metadata["preset"] == "fast"
    assert metadata["thresholds"] == {
        "linear_velocity": 0.03,
        "angular_velocity": 0.2,
        "min_steps": 0,
        "max_steps": 4,
        "check_interval_steps": 1,
        "required_stable_checks": 3,
    }
    assert metadata["state"]["elapsed_steps"] == 2
    assert metadata["state"]["settled_mask"] == [True, True]
    assert metadata["state"]["timeout_mask"] == [False, False]
    assert metadata["state"]["max_linear_speed"] == [0.0, 0.0]
    assert port.post_policy_result(
        segment.post_policies[0],
        segment=segment,
    ).tolist() == [True, True]


def test_wait_stable_holds_active_targets_and_inactive_current_qpos() -> None:
    """Initial inactive rows use measured holds while active rows keep preload."""
    segment = _compiled_segment()
    target_qpos = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    port, _, robot = _port(
        torch.zeros(2, 3),
        target_qpos=target_qpos,
        preset=DynamicSettleMonitorCfg(
            min_steps=0,
            max_steps=4,
            check_interval_steps=1,
            required_stable_checks=3,
        ),
    )
    active_mask = torch.tensor([True, False])
    actions = port.actions(
        segment.post_policies[0],
        segment=segment,
        active_mask=active_mask,
    )
    expected = torch.stack((target_qpos[0], robot.current_qpos[1]))

    first = next(actions)
    assert torch.equal(first, expected)
    first.fill_(99.0)

    second = next(actions)
    assert torch.equal(second, expected)
    assert second.data_ptr() != first.data_ptr()
    with pytest.raises(StopIteration):
        next(actions)

    assert robot.qpos_reads == [False, True, False, True, False]
    result = port.post_policy_result(
        segment.post_policies[0],
        segment=segment,
    )
    assert result.tolist() == [True, False]


def test_wait_stable_rejects_wrong_target_width_for_all_active_rows() -> None:
    """All-active settling fails closed on a malformed full target qpos."""
    segment = _compiled_segment()
    port, _, robot = _port(torch.zeros(2, 3))
    robot.target_qpos = torch.zeros(2, 3)
    actions = port.actions(
        segment.post_policies[0],
        segment=segment,
        active_mask=torch.ones(2, dtype=torch.bool),
    )

    with pytest.raises(
        ValueError,
        match="target full qpos must match the construction-time current full qpos",
    ):
        next(actions)

    assert robot.qpos_reads == [False, True]


def test_wait_stable_returns_row_local_timeout_result_and_metadata() -> None:
    """A moving row times out without failing a settled peer or the batch."""
    segment = _compiled_segment()
    port, entity, _ = _port(torch.zeros(2, 3))
    entity.body_data.lin_vel[1, 0] = 1.0
    actions = port.actions(
        segment.post_policies[0],
        segment=segment,
        active_mask=torch.ones(2, dtype=torch.bool),
    )

    assert sum(1 for _ in (next(actions), next(actions), next(actions))) == 3
    with pytest.raises(StopIteration):
        next(actions)

    metadata = port.post_policy_metadata(
        segment.post_policies[0],
        segment=segment,
    )
    assert metadata["status"] == "timed_out"
    assert metadata["state"]["elapsed_steps"] == 3
    assert metadata["state"]["settled_mask"] == [True, False]
    assert metadata["state"]["timeout_mask"] == [False, True]
    assert metadata["state"]["max_linear_speed"] == [0.0, 1.0]
    assert port.post_policy_result(
        segment.post_policies[0],
        segment=segment,
    ).tolist() == [True, False]


def test_in_progress_settling_metadata_uses_json_null_for_unchecked_speeds() -> None:
    segment = _compiled_segment()
    port, _, _ = _port(
        torch.zeros(2, 3),
        preset=DynamicSettleMonitorCfg(
            min_steps=2,
            max_steps=4,
            check_interval_steps=1,
            required_stable_checks=1,
        ),
    )
    actions = port.actions(
        segment.post_policies[0],
        segment=segment,
        active_mask=torch.ones(2, dtype=torch.bool),
    )

    next(actions)
    metadata = port.post_policy_metadata(
        segment.post_policies[0],
        segment=segment,
    )
    actions.close()

    assert metadata["status"] == "running"
    assert metadata["state"]["max_linear_speed"] == [None, None]
    assert metadata["state"]["max_angular_speed"] == [None, None]
    json.dumps(metadata, allow_nan=False, sort_keys=True)


def test_wait_stable_excludes_inactive_moving_row_from_completion() -> None:
    """A failed runtime row cannot block or pass a later settling policy."""
    segment = _compiled_segment()
    port, entity, _ = _port(torch.zeros(2, 3))
    entity.body_data.lin_vel[1, 0] = 1.0
    active_mask = torch.tensor([True, False])
    actions = port.actions(
        segment.post_policies[0],
        segment=segment,
        active_mask=active_mask,
    )

    assert sum(1 for _ in actions) == 1

    metadata = port.post_policy_metadata(
        segment.post_policies[0],
        segment=segment,
    )
    assert metadata["status"] == "settled"
    assert metadata["active_mask"] == [True, False]
    assert metadata["state"]["active_mask"] == [True, False]
    assert metadata["state"]["settled_mask"] == [True, False]
    assert metadata["state"]["timeout_mask"] == [False, False]
    assert metadata["state"]["max_linear_speed"] == [0.0, None]
    assert port.post_policy_result(
        segment.post_policies[0],
        segment=segment,
    ).tolist() == [True, False]


def test_wait_stable_skips_when_no_rows_remain_active() -> None:
    """An empty active cohort completes without an environment hold."""
    segment = _compiled_segment()
    port, _, _ = _port(torch.zeros(2, 3))

    actions = port.actions(
        segment.post_policies[0],
        segment=segment,
        active_mask=torch.zeros(2, dtype=torch.bool),
    )

    assert tuple(actions) == ()
    metadata = port.post_policy_metadata(
        segment.post_policies[0],
        segment=segment,
    )
    assert metadata["status"] == "skipped"
    assert metadata["state"]["settled_mask"] == [False, False]
    assert metadata["state"]["timeout_mask"] == [False, False]
    assert port.post_policy_result(
        segment.post_policies[0],
        segment=segment,
    ).tolist() == [False, False]


def test_object_near_target_validates_rows_independently() -> None:
    """The validator compares explicit native object poses row by row."""
    segment = _compiled_segment()
    port, _, _ = _port(torch.tensor([[0.01, 0.0, 0.0], [0.20, 0.0, 0.0]]))

    result = port.validate(segment.validators[0], segment=segment)

    assert result.dtype == torch.bool
    assert result.tolist() == [True, False]
    metadata = port.validator_metadata(segment.validators[0], segment=segment)
    assert metadata["kind"] == "object_near_target"
    assert metadata["object_id"] == "cube"
    assert metadata["target_id"] == "drop"
    assert metadata["position_tolerance"] == 0.05
    assert metadata["position_error"] == pytest.approx([0.01, 0.20])
    assert metadata["accepted_mask"] == [True, False]


def test_object_near_relative_target_observes_both_rows_independently() -> None:
    """The validator derives each target from the current reference pose."""
    segment = _compiled_relative_segment()
    obj = _RigidObject(torch.tensor([[0.01, -0.1, 0.04], [0.20, -0.1, 0.04]]))
    reference = _RigidObject(torch.tensor([[0.0, 0.0, 0.0], [0.10, 0.0, 0.0]]))
    port = SimulationSegmentPolicyPort(
        _Simulation(entity=obj, reference=reference),
        _Robot(torch.zeros(2, 2)),
        SimulationSceneBinding(
            registry_id="test_scene",
            rigid_objects=(
                SimulationRigidObjectBinding(
                    entity_id="cube",
                    simulation_uid="native_cube",
                ),
                SimulationRigidObjectBinding(
                    entity_id="reference",
                    simulation_uid="native_reference",
                ),
            ),
        ),
    )

    validator = segment.validators[0]
    result = port.validate(validator, segment=segment)
    metadata = port.validator_metadata(validator, segment=segment)

    assert result.tolist() == [True, False]
    assert metadata["kind"] == "object_near_relative_target"
    assert metadata["object_id"] == "cube"
    assert metadata["reference_id"] == "reference"
    assert metadata["displacement"] == pytest.approx([0.0, -0.1, 0.04])
    assert metadata["position_error"] == pytest.approx([0.01, 0.10])
    assert metadata["accepted_mask"] == [True, False]
    assert obj.pose_reads == 1
    assert reference.pose_reads == 1


def test_articulation_joint_position_validates_measured_rows() -> None:
    """The joint validator applies its inclusive bound to each simulator row."""
    segment = _compiled_articulation_segment()
    articulation = _Articulation(
        torch.tensor([[0.11], [0.09], [float("nan")]], dtype=torch.float32)
    )
    robot = _Robot(torch.zeros(3, 2))
    port = SimulationSegmentPolicyPort(
        _Simulation(articulation=articulation),
        robot,
        SimulationSceneBinding(
            registry_id="test_scene",
            articulations=(
                SimulationArticulationBinding(
                    entity_id="drawer",
                    simulation_uid="native_drawer",
                ),
            ),
        ),
    )

    validator = segment.validators[0]
    result = port.validate(validator, segment=segment)
    metadata = port.validator_metadata(validator, segment=segment)

    assert result.tolist() == [True, False, False]
    assert metadata["kind"] == "articulation_joint_position"
    assert metadata["articulation_id"] == "drawer"
    assert metadata["joint"] == "cabinet_to_drawer"
    assert metadata["minimum_position"] == pytest.approx(0.10)
    assert metadata["maximum_position"] is None
    assert metadata["joint_position"][:2] == pytest.approx([0.11, 0.09])
    assert metadata["joint_position"][2] is None
    assert metadata["accepted_mask"] == [True, False, False]
    json.dumps(metadata, allow_nan=False, sort_keys=True)


def test_policy_port_rejects_unbound_native_entities_and_foreign_members() -> None:
    """Bindings and compiled segment ownership are exact fail-closed boundaries."""
    binding = SimulationSceneBinding(
        registry_id="test_scene",
        rigid_objects=(
            SimulationRigidObjectBinding(
                entity_id="missing",
                simulation_uid="unknown",
            ),
        ),
    )
    robot = _Robot(torch.zeros(2, 2))
    with pytest.raises(KeyError, match="unknown"):
        SimulationSegmentPolicyPort(
            _Simulation(_RigidObject(torch.zeros(2, 3))),
            robot,
            binding,
        )

    segment = _compiled_segment()
    other = _compiled_segment()
    port, _, _ = _port(torch.zeros(2, 3))
    with pytest.raises(ValueError, match="does not belong"):
        tuple(
            port.actions(
                other.post_policies[0],
                segment=segment,
                active_mask=torch.ones(2, dtype=torch.bool),
            )
        )


__all__: list[str] = []


def _object_state_payload(validators: list[dict[str, object]]) -> dict[str, object]:
    """Build a segment using the public strict decoder and ordinary calls."""
    return {
        "program_id": "measured_objects",
        "targets": {},
        "integration": {
            "robot_profile": "test_robot",
            "scene_registry": "test_scene",
            "runtime_preset": "safe",
        },
        "program": {
            "kind": "segment",
            "name": "pick",
            "steps": {"kind": "invoke", "call": {"kind": "pick", "object": "cube"}},
            "validators": validators,
        },
    }


def _measured_object_segment(validators: list[dict[str, object]]):
    """Resolve both measured object references without constructing a simulator."""
    registry = SceneRegistry(
        tuple(
            SceneEntityRegistration(
                ref=SceneObjectRef(entity_id), state_provider=_StaticStateProvider()
            )
            for entity_id in ("cube", "reference")
        )
    )
    compiled = TaskProgramCompiler.from_scene_registry(registry).compile(
        decode_task_program(_object_state_payload(validators))
    )
    return next(compiled.iter_segments())


def _measured_object_port(batch_size: int = 2):
    """Keep stable logical IDs distinct from simulator row indices."""
    objects = {
        name: _RigidObject(torch.zeros(batch_size, 3, dtype=torch.float64))
        for name in ("cube", "reference")
    }
    # Use double poses so exact binary tolerance boundaries stay exact.
    for entity in objects.values():
        entity._pose = entity._pose.to(dtype=torch.float64)
    port = SimulationSegmentPolicyPort(
        SimpleNamespace(get_rigid_object=objects.get),
        _Robot(torch.zeros(batch_size, 2)),
        SimulationSceneBinding(
            registry_id="test_scene",
            rigid_objects=tuple(
                SimulationRigidObjectBinding(entity_id=name, simulation_uid=name)
                for name in objects
            ),
        ),
        env_ids=torch.arange(batch_size - 1, -1, -1),
    )
    return port, objects


def test_lift_requires_segment_start_and_uses_observed_row_baselines() -> None:
    segment = _measured_object_segment(
        [{"kind": "object_lifted", "object": "cube", "minimum_lift": 0.05}]
    )
    validator = segment.validators[0]
    port, objects = _measured_object_port()
    with pytest.raises(RuntimeError, match="begin_segment"):
        port.validate(validator, segment=segment)

    objects["cube"]._pose[:, 2, 3] = torch.tensor([0.0, 0.5])
    port.begin_segment(segment=segment, active_mask=None)
    objects["cube"]._pose[:, 2, 3] = torch.tensor([0.05, 0.54], dtype=torch.float64)
    assert port.validate(validator, segment=segment).tolist() == [True, False]
    trace = port.validator_metadata(validator, segment=segment)
    assert trace["initial_height"] == [0.0, 0.5]
    assert trace["env_ids"] == [1, 0]
    assert trace["measured_lift"] == pytest.approx([0.05, 0.04])
    json.dumps(trace, allow_nan=False)


def test_segment_policy_sessions_keep_lift_baselines_independent() -> None:
    segment = _measured_object_segment(
        [{"kind": "object_lifted", "object": "cube", "minimum_lift": 0.05}]
    )
    validator = segment.validators[0]
    owner, objects = _measured_object_port()
    first = owner.new_task_program_session()
    second = owner.new_task_program_session()
    objects["cube"]._pose[:, 2, 3] = 0.0
    first.begin_segment(segment=segment, active_mask=None)
    objects["cube"]._pose[:, 2, 3] = 0.5
    second.begin_segment(segment=segment, active_mask=None)
    objects["cube"]._pose[:, 2, 3] = torch.tensor([0.56, 0.53])
    assert first.validate(validator, segment=segment).tolist() == [True, True]
    assert second.validate(validator, segment=segment).tolist() == [True, False]
    assert first.validator_metadata(validator, segment=segment)["initial_height"] == [
        0.0,
        0.0,
    ]
    assert second.validator_metadata(validator, segment=segment)["initial_height"] == [
        0.5,
        0.5,
    ]
    with pytest.raises(RuntimeError, match="already captured"):
        first.begin_segment(segment=segment, active_mask=None)


def test_lift_rejects_nonfinite_baseline_per_row_and_preserves_inactive_mask() -> None:
    segment = _measured_object_segment([{"kind": "object_lifted", "object": "cube"}])
    port, objects = _measured_object_port(3)
    objects["cube"]._pose[0, 0, 3] = float("nan")
    port.begin_segment(segment=segment, active_mask=torch.tensor([True, True, False]))
    objects["cube"]._pose[:, 0, 3] = 0.0
    objects["cube"]._pose[:, 2, 3] = 0.1
    validator = segment.validators[0]
    assert port.validate(validator, segment=segment).tolist() == [False, True, False]
    json.dumps(port.validator_metadata(validator, segment=segment), allow_nan=False)


def test_relative_position_uses_live_world_axes_ignores_disabled_z_and_is_strict() -> (
    None
):
    segment = _measured_object_segment(
        [
            {
                "kind": "object_relative_position",
                "object": "cube",
                "reference": "reference",
                "position_offset": [0.5, 0.0, 0.0],
                "position_tolerance": [0.125, 0.125, None],
            }
        ]
    )
    port, objects = _measured_object_port()
    port.begin_segment(segment=segment, active_mask=None)
    # The reference moves and rotates after the segment starts; its local X
    # points along world Y, but the declared offset remains along world X.
    objects["reference"]._pose[:, :3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    objects["reference"]._pose[:, :3, 3] = torch.tensor([1.0, 2.0, 3.0])
    objects["cube"]._pose[:, :3, 3] = torch.tensor(
        [[1.5, 2.0, 99.0], [1.625, 2.0, -99.0]], dtype=torch.float64
    )
    assert port.validate(segment.validators[0], segment=segment).tolist() == [
        True,
        False,
    ]
    objects["reference"]._pose[0, 0, 3] = float("nan")
    assert port.validate(segment.validators[0], segment=segment).tolist() == [
        False,
        False,
    ]
    json.dumps(
        port.validator_metadata(segment.validators[0], segment=segment), allow_nan=False
    )


def test_relative_position_parent_frame_follows_live_reference_rotation() -> None:
    segment = _measured_object_segment(
        [
            {
                "kind": "object_relative_position",
                "object": "cube",
                "reference": "reference",
                "reference_frame": "parent",
                "position_offset": [0.5, 0.0, 0.0],
                "position_tolerance": [0.125, 0.125, None],
            }
        ]
    )
    port, objects = _measured_object_port()
    port.begin_segment(segment=segment, active_mask=None)
    # Local X follows world Y after a quarter turn, including after begin_segment.
    objects["reference"]._pose[:, :3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    objects["reference"]._pose[:, :3, 3] = torch.tensor([1.0, 2.0, 3.0])
    objects["cube"]._pose[:, :3, 3] = torch.tensor(
        [[1.0, 2.5, 99.0], [1.0, 2.625, -99.0]], dtype=torch.float64
    )
    validator = segment.validators[0]
    assert port.validate(validator, segment=segment).tolist() == [True, False]
    evidence = port.validator_metadata(validator, segment=segment)
    assert evidence["reference_frame"] == "parent"
    assert evidence["position_error"][0][:2] == [0.0, 0.0]
    objects["reference"]._pose[0, 0, 0] = float("nan")
    assert port.validate(validator, segment=segment).tolist() == [False, False]
    json.dumps(port.validator_metadata(validator, segment=segment), allow_nan=False)


def test_upright_rejects_angle_boundary_and_nonfinite_rows() -> None:
    segment = _measured_object_segment(
        [{"kind": "object_upright", "object": "cube", "maximum_tilt_degrees": 90.0}]
    )
    port, objects = _measured_object_port(3)
    objects["cube"]._pose[1, :3, :3] = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=torch.float64
    )
    objects["cube"]._pose[2, 2, 2] = float("inf")
    assert port.validate(segment.validators[0], segment=segment).tolist() == [
        True,
        False,
        False,
    ]
    json.dumps(
        port.validator_metadata(segment.validators[0], segment=segment), allow_nan=False
    )


def test_measured_validator_rejects_a_different_segment_and_missing_binding() -> None:
    segment = _measured_object_segment(
        [{"kind": "object_lifted", "object": "reference"}]
    )
    port, _, _ = _port(torch.zeros(2, 3))
    with pytest.raises(KeyError, match="reference"):
        port.validate_validator(segment.validators[0], segment=segment)
    other = _measured_object_segment([{"kind": "object_lifted", "object": "cube"}])
    with pytest.raises(ValueError, match="does not belong"):
        port.validate_validator(other.validators[0], segment=segment)


def test_declared_condition_keeps_duration_grid_and_resets_anchor_after_drift():
    from embodichain.lab.task_program.integrations.simulation.stability import (
        SimulationStabilityConditionCfg,
    )

    cfg = SimulationStabilityConditionCfg(
        entity="cube",
        kind="placement",
        target_position=(0.0, 0.0, 0.0),
        duration=0.08,
        timeout=0.24,
        position_tolerance=1.0,
        translation_drift=0.02,
    )
    port, entity, robot = _port(torch.zeros(2, 3), preset=cfg, step_dt=0.04)
    cfg.duration = 9.0  # The port owns a normalized independent snapshot.
    segment = _compiled_segment()
    policy = segment.post_policies[0]
    actions = port.actions(
        policy, segment=segment, active_mask=torch.tensor([True, False])
    )
    next(actions)
    entity._pose[0, 0, 3] = 0.03
    next(actions)
    assert port.post_policy_metadata(policy, segment=segment)["stable_seconds"] == [
        0.0,
        0.0,
    ]
    next(actions)
    with pytest.raises(StopIteration):
        next(actions)
    assert port.post_policy_result(policy, segment=segment).tolist() == [True, False]
    metadata = port.post_policy_metadata(policy, segment=segment)
    assert metadata["elapsed_seconds"] == 0.12
    assert metadata["required_seconds"] == 0.08
    assert metadata["stable_seconds"] == pytest.approx([0.08, 0.0])


def test_held_condition_latches_loss_and_holds_failed_rows_at_observed_qpos():
    from embodichain.lab.task_program.integrations.simulation.stability import (
        SimulationStabilityConditionCfg,
    )

    cfg = SimulationStabilityConditionCfg(
        entity="cube",
        kind="hold",
        target_position=(0.0, 0.0, 0.0),
        motion_parts=("arm",),
        duration=0.12,
        timeout=0.24,
        position_tolerance=0.2,
    )
    port, entity, robot = _port(
        torch.zeros(2, 3), preset=cfg, step_dt=0.04, target_qpos=torch.ones(2, 2)
    )
    segment = _compiled_segment()
    policy = segment.post_policies[0]
    actions = port.actions(
        policy, segment=segment, active_mask=torch.ones(2, dtype=torch.bool)
    )
    assert next(actions).tolist() == [[1.0, 1.0], [1.0, 1.0]]
    entity._pose[0, 0, 3] = 0.03
    command = next(actions)
    assert torch.equal(command[0], robot.current_qpos[0])
    assert command[1].tolist() == [1.0, 1.0]
    entity._pose[0, 0, 3] = 0.0
    next(actions)
    with pytest.raises(StopIteration):
        next(actions)
    assert port.post_policy_result(policy, segment=segment).tolist() == [False, True]
    assert port.post_policy_metadata(policy, segment=segment)["failed_mask"] == [
        True,
        False,
    ]


def test_condition_session_cleanup_stops_incomplete_sampling_without_success():
    from embodichain.lab.task_program.integrations.simulation.stability import (
        SimulationStabilityConditionCfg,
    )

    cfg = SimulationStabilityConditionCfg(
        entity="cube",
        kind="upright",
        local_axis=(0.0, 0.0, 2.0),
        duration=0.08,
        timeout=0.2,
    )
    port, _, _ = _port(torch.zeros(2, 3), preset=cfg, step_dt=0.04)
    session = port.new_task_program_session()
    segment = _compiled_segment()
    policy = segment.post_policies[0]
    actions = session.actions(
        policy, segment=segment, active_mask=torch.ones(2, dtype=torch.bool)
    )
    next(actions)
    port.close_runtime_resources()
    port.close_runtime_resources()
    with pytest.raises(RuntimeError, match="closed"):
        next(actions)
    assert not session.post_policy_result(policy, segment=segment).any()


def test_observation_clock_scope_is_local_idempotent_and_does_not_change_default_execution():
    from embodichain.lab.gym.envs.task_program.bridge import EnvironmentStepClock

    template, _, _ = _port(torch.zeros(2, 3), step_dt=0.04)
    assert template.observation_clock is None
    first = template.new_task_program_session()
    clock = EnvironmentStepClock(0.04)
    first.bind_clock(clock)
    assert first.observation_clock is clock
    first.bind_clock(clock)
    first.validate_observation_scope(clock)
    second = template.new_task_program_session()
    other_clock = EnvironmentStepClock(0.04)
    second.bind_clock(other_clock)
    with pytest.raises(ValueError, match="another runtime/session"):
        first.validate_observation_scope(clock)
    with pytest.raises(ValueError, match="another runtime/session"):
        _ = first.observation_clock
    assert second.observation_clock is other_clock
    independent, _, _ = _port(torch.zeros(2, 3), step_dt=0.04)
    independent.bind_clock(clock)
    assert independent.observation_clock is clock
    segment = _compiled_segment()
    policy = segment.post_policies[0]
    list(
        first.actions(
            policy, segment=segment, active_mask=torch.ones(2, dtype=torch.bool)
        )
    )
    assert first.post_policy_result(policy, segment=segment).all()
    template.close_runtime_resources()
    with pytest.raises(RuntimeError, match="closed"):
        _ = second.observation_clock
    assert independent.observation_clock is clock


def _joint_condition_port(qpos, *, duration=1.0, timeout=5.0, **options):
    from embodichain.lab.task_program.integrations.simulation.stability import (
        SimulationJointStabilityConditionCfg,
    )

    art = _Articulation(qpos)
    robot = _Robot(
        torch.zeros(qpos.shape[0], 2), target_qpos=torch.ones(qpos.shape[0], 2)
    )
    cfg = SimulationJointStabilityConditionCfg(
        entity="drawer",
        joint="cabinet_to_drawer",
        target=0.1,
        tolerance=0.03,
        duration=duration,
        timeout=timeout,
        **options,
    )
    port = SimulationSegmentPolicyPort(
        _Simulation(articulation=art),
        robot,
        SimulationSceneBinding(
            registry_id="test_scene",
            articulations=(
                SimulationArticulationBinding(
                    entity_id="drawer", simulation_uid="native_drawer"
                ),
            ),
        ),
        settle_presets={"joint": cfg},
        step_dt=0.04,
    )
    segment = _compiled_articulation_segment(settle_preset="joint")
    return port, art, robot, segment, segment.post_policies[0]


def test_joint_stability_uses_joint_window_and_original_observed_hold():
    port, art, robot, segment, policy = _joint_condition_port(
        torch.tensor([[0.1], [0.2]])
    )
    commands = []
    for command in port.actions(
        policy, segment=segment, active_mask=torch.tensor([True, True])
    ):
        commands.append(command)
        robot.current_qpos.add_(0.01)
    assert len(commands) == 125
    assert all(torch.equal(command, torch.zeros(2, 2)) for command in commands)
    assert port.post_policy_result(policy, segment=segment).tolist() == [True, False]
    metadata = port.post_policy_metadata(policy, segment=segment)
    assert metadata == {
        "joint": "cabinet_to_drawer",
        "target": 0.1,
        "observed": art.get_qpos()[:, 0].tolist(),
        "tolerance": 0.03,
        "stable_seconds": 1.0,
        "accepted": [True, False],
        "evidence": "joint_position_and_stability",
        "contact_qualification": "not_provided",
    }
    assert not any(robot.qpos_reads)


def test_joint_stability_resets_the_full_range_and_does_not_count_tick_zero():
    port, art, _, segment, policy = _joint_condition_port(torch.tensor([[0.1]]))
    steps = 0
    for _ in port.actions(policy, segment=segment, active_mask=torch.tensor([True])):
        steps += 1
        if steps == 10:
            art._qpos[0, 0] = 0.1011
    assert steps == 35
    assert port.post_policy_result(policy, segment=segment).tolist() == [True]


def test_joint_stability_unavailable_rows_and_inactive_rows_preserve_source_boundary():
    port, art, _, segment, policy = _joint_condition_port(
        torch.tensor([[0.1], [float("nan")]])
    )
    actions = list(
        port.actions(policy, segment=segment, active_mask=torch.tensor([True, False]))
    )
    assert len(actions) == 25
    assert port.post_policy_result(policy, segment=segment).tolist() == [True, False]
    assert port.post_policy_metadata(policy, segment=segment)["observed"][1] is None
    actions = list(
        port.actions(policy, segment=segment, active_mask=torch.tensor([True, True]))
    )
    assert actions == []
    assert not port.post_policy_result(policy, segment=segment).any()


def test_joint_stability_clocked_raw_channel_uses_actual_scope_and_sample_sequence():
    from embodichain.lab.gym.envs.task_program.bridge import EnvironmentStepClock

    port, _, _, segment, policy = _joint_condition_port(torch.tensor([[0.1]]))
    clock = EnvironmentStepClock(0.04)
    port.bind_clock(clock)
    for _ in port.actions(policy, segment=segment, active_mask=torch.tensor([True])):
        clock.advance_after_env_step()
    record = port.post_policy_metadata(policy, segment=segment)
    raw = record["joint_observation"]
    assert raw["evidence_id"] == "articulation.joint_position"
    assert raw["timestamp"] == clock.now() == 1.0
    assert raw["observation_revision"] == 26
    assert raw["valid"] == [True] and raw["env_ids"] == [0]
    assert raw["acquisition_errors"] == [None]
    assert raw["start_step"] == 0 and raw["end_step"] == 25
    assert record["evidence"] == "joint_position_and_stability"
    # No successful receipt from this same compiled policy can survive a
    # later incomplete/cancelled execution.
    actions = port.actions(policy, segment=segment, active_mask=torch.tensor([True]))
    next(actions)
    actions.close()
    with pytest.raises(RuntimeError, match="unavailable"):
        port.post_policy_result(policy, segment=segment)


def test_joint_stability_rejects_unacknowledged_steps_before_claiming_duration():
    from embodichain.lab.gym.envs.task_program.bridge import EnvironmentStepClock

    port, _, _, segment, policy = _joint_condition_port(torch.tensor([[0.1]]))
    clock = EnvironmentStepClock(0.04)
    port.bind_clock(clock)
    actions = port.actions(policy, segment=segment, active_mask=torch.tensor([True]))
    next(actions)
    with pytest.raises(ValueError, match="acknowledged"):
        next(actions)
    with pytest.raises(RuntimeError, match="unavailable"):
        port.post_policy_result(policy, segment=segment)


def test_joint_window_records_distinguish_same_terminal_sample_histories():
    from embodichain.lab.gym.envs.task_program.bridge import EnvironmentStepClock

    records = []
    for transient in (False, True):
        port, art, _, segment, policy = _joint_condition_port(
            torch.tensor([[0.1]]), timeout=1.0
        )
        clock = EnvironmentStepClock(0.04)
        port.bind_clock(clock)
        for _ in port.actions(
            policy, segment=segment, active_mask=torch.tensor([True])
        ):
            clock.advance_after_env_step()
            # A transient stays inside the target tolerance but violates the
            # candidate window's 1 mm range; the final coordinate is unchanged.
            art._qpos[0, 0] = 0.102 if transient and clock.step_index == 10 else 0.1
        records.append(port.post_policy_metadata(policy, segment=segment))
    assert records[0]["joint_observation"] == records[1]["joint_observation"]
    assert [record["accepted"] for record in records] == [[True], [False]]
    good, bad = [record["joint_window"] for record in records]
    assert good["values"] != bad["values"]
    assert good["steps"] == bad["steps"] == list(range(26))
    assert good["record_complete"] and bad["record_complete"]
    assert good["termination"] == "satisfied"
    assert bad["termination"] == "timed_out"
    assert bad["condition_status"] == ["not_satisfied"]
    assert good["policy_source_path"] == list(policy.source_path)
    assert good["native_uid"] == "native_drawer"
    # Independent consumer can recompute the rule from the emitted facts.
    from embodichain.compute.trajectory import PositionRangeMonitor

    for record in records:
        window = record["joint_window"]
        values = torch.tensor(window["values"])
        monitor = PositionRangeMonitor(
            torch.tensor(window["monitor_initial_values"]),
            required_steps=25,
            position_drift=0.001,
        )
        for step in window["steps"]:
            observed = values[:, step]
            replayed = monitor.observe(
                observed,
                (observed - record["target"]).abs() <= record["tolerance"],
                elapsed_steps=step,
            )
        assert replayed.tolist() == record["accepted"]


def test_joint_window_replay_retains_pre_sample_range_anchor():
    from embodichain.compute.trajectory import PositionRangeMonitor
    from embodichain.lab.gym.envs.task_program.bridge import EnvironmentStepClock

    port, art, _, segment, policy = _joint_condition_port(
        torch.tensor([[0.1]]), timeout=1.0
    )
    clock = EnvironmentStepClock(0.04)
    port.bind_clock(clock)
    original = art.get_qpos
    reads = 0

    def anchored_read():
        nonlocal reads
        reads += 1
        return torch.tensor([[0.1008]]) if reads == 1 else original()

    art.get_qpos = anchored_read
    for _ in port.actions(policy, segment=segment, active_mask=torch.tensor([True])):
        clock.advance_after_env_step()
        art._qpos[0, 0] = 0.0997 if clock.step_index >= 10 else 0.1
    record = port.post_policy_metadata(policy, segment=segment)
    window = record["joint_window"]
    values = torch.tensor(window["values"])
    assert window["monitor_initial_values"] != values[:, 0].tolist()
    monitor = PositionRangeMonitor(
        torch.tensor(window["monitor_initial_values"]),
        required_steps=25,
        position_drift=0.001,
    )
    for step, sample in enumerate(values.T):
        accepted = monitor.observe(
            sample,
            (sample - record["target"]).abs() <= record["tolerance"],
            elapsed_steps=step,
        )
    assert accepted.tolist() == record["accepted"] == [False]


def test_joint_window_explicitly_isolates_unavailable_rows():
    from embodichain.lab.gym.envs.task_program.bridge import EnvironmentStepClock

    port, _, _, segment, policy = _joint_condition_port(
        torch.tensor([[0.1], [float("nan")]]), invalid_observation_policy="isolate_rows"
    )
    clock = EnvironmentStepClock(0.04)
    port.bind_clock(clock)
    for _ in port.actions(
        policy, segment=segment, active_mask=torch.tensor([True, True])
    ):
        clock.advance_after_env_step()
    assert clock.step_index == 25
    assert port.post_policy_result(policy, segment=segment).tolist() == [True, False]
    record = port.post_policy_metadata(policy, segment=segment)
    assert record["joint_window"]["condition_status"] == ["satisfied", "unavailable"]
    assert record["joint_observation"]["valid"] == [True, False]
    assert record["joint_window"]["values"][1] == [None] * 26
    json.dumps(record, allow_nan=False)


def test_joint_window_partial_receipt_is_not_success_or_complete_duration():
    from embodichain.lab.gym.envs.task_program.bridge import EnvironmentStepClock

    port, _, _, segment, policy = _joint_condition_port(torch.tensor([[0.1]]))
    clock = EnvironmentStepClock(0.04)
    port.bind_clock(clock)
    actions = port.actions(policy, segment=segment, active_mask=torch.tensor([True]))
    next(actions)
    actions.close()
    with pytest.raises(RuntimeError, match="unavailable"):
        port.post_policy_result(policy, segment=segment)
    window = port.post_policy_metadata(policy, segment=segment)["joint_window"]
    assert window["termination"] == "cancelled"
    assert not window["execution_complete"]
    assert window["condition_status"] == ["pending"]
    assert window["steps"] == [0]


def test_joint_window_rejects_clock_changes_during_acquisition_without_raw_fact():
    from embodichain.lab.gym.envs.task_program.bridge import EnvironmentStepClock

    port, art, _, segment, policy = _joint_condition_port(torch.tensor([[0.1]]))
    clock = EnvironmentStepClock(0.04)
    port.bind_clock(clock)
    original = art.get_qpos
    reads = 0

    def changing_clock():
        nonlocal reads
        reads += 1
        if reads == 2:  # Anchor read is followed by the first policy sample.
            clock.advance_after_env_step()
        return original()

    art.get_qpos = changing_clock
    with pytest.raises(ValueError, match="clock changed during"):
        next(port.actions(policy, segment=segment, active_mask=torch.tensor([True])))
    record = port.post_policy_metadata(policy, segment=segment)
    assert "joint_observation" not in record
    assert record["joint_window"]["sample_count"] == 0
    assert record["joint_window"]["termination"] == "interrupted"


def test_joint_window_trace_capacity_never_silently_certifies_full_record():
    from embodichain.lab.gym.envs.task_program.bridge import EnvironmentStepClock

    port, _, _, segment, policy = _joint_condition_port(
        torch.tensor([[0.1]]), max_observation_samples=3
    )
    clock = EnvironmentStepClock(0.04)
    port.bind_clock(clock)
    for _ in port.actions(policy, segment=segment, active_mask=torch.tensor([True])):
        clock.advance_after_env_step()
    assert port.post_policy_result(policy, segment=segment).tolist() == [True]
    window = port.post_policy_metadata(policy, segment=segment)["joint_window"]
    assert window["sample_count"] == 26 and window["retained_sample_count"] == 3
    assert window["dropped_samples"] == 23 and not window["record_complete"]
    assert window["steps"] == [23, 24, 25]


def _press_control_port(*, program_fingerprint="program", episode_id="episode"):
    from dataclasses import asdict
    from embodichain.lab.sim.scene.press_inputs import PressBinding
    from embodichain.lab.sim.sensors.contact.sampling import ContactEvidenceSampler
    from embodichain.lab.task_program.integrations.simulation.control_observation import (
        ControlContactActors,
    )
    from embodichain.lab.task_program.integrations.simulation.control_policy import (
        ControlOperationRuntime,
        SimulationControlConditionCfg,
    )
    from embodichain.lab.gym.envs.task_program.bridge import EnvironmentStepClock

    art = _Articulation(torch.zeros(1, 1))
    art.num_instances = 1
    art.uid = "native_drawer"
    art.get_qpos_limits = lambda: torch.tensor([[[0.0, 0.01]]])
    robot = _Robot(torch.full((1, 2), 0.2), torch.full((1, 2), 0.7))
    robot.num_instances = 1
    clock = EnvironmentStepClock(0.04)
    time = [0.0]
    hooks = {}
    fetches = [0]
    data = {
        "is_valid": torch.ones(1, 1, dtype=torch.bool),
        "user_ids": torch.tensor([[[1, 2]]]),
        "impulse": torch.zeros(1, 1),
        "position": torch.zeros(1, 1, 3),
        "normal": torch.zeros(1, 1, 3),
        "distance": torch.zeros(1, 1),
    }

    def update():
        fetches[0] += 1

    def add(name, observer):
        assert name not in hooks
        hooks[name] = observer

    def remove(name, *, expected):
        if hooks.get(name) is expected:
            hooks.pop(name)

    sensor = SimpleNamespace(
        uid="contact",
        num_instances=1,
        contact_capabilities=SimpleNamespace(geometry=True, impulse=True),
        dropped_contacts=0,
        update=update,
        get_data=lambda: data,
        add_sampling_observer=add,
        remove_sampling_observer=remove,
    )
    binding = PressBinding(
        "native_drawer",
        "cabinet_to_drawer",
        "button",
        "base",
        "/frozen/button.usda",
        "a" * 64,
        "/button/mesh",
        1.0,
        (0.0, 0.0, -1.0),
        (0.0, 0.01),
        (0.0, 0.0, 0.0),
        0.0,
        0.01,
    )
    payload = {
        "binding": asdict(binding),
        "operation_id": "press0",
        "arm": "right",
        "variant": "source",
        "moving_mass": 0.1,
        "extra_press_distance": 0.004,
    }
    payload["binding"].pop("source_path")
    geometry = SimpleNamespace(
        robot=robot,
        articulation=art,
        press_clearances=lambda *, terminal: {
            "retreat_clearance": 0.05,
            "tcp_clearance": 0.06,
            "terminal_clearance": 0.05 if terminal else None,
        },
    )
    sampler = ContactEvidenceSampler(
        sensor,
        native_revision_reader=lambda: round(time[0] / 0.004),
        topology_revision_reader=lambda: 0,
    )
    operation = ControlOperationRuntime(
        robot,
        art,
        binding,
        geometry,
        sampler,
        ControlContactActors(1, 4, (2,)),
        entity_id="drawer",
        operation_id="press0",
        program_fingerprint=program_fingerprint,
        episode_id=episode_id,
        observer_id="press",
        route_payload=payload,
        control_dt=0.04,
        physics_dt=0.004,
        simulation_time=lambda: time[0],
        validate_scope=lambda: None,
    )
    presets = {
        name: SimulationControlConditionCfg(
            entity="drawer",
            operation_id="press0",
            family="press",
            phase=phase,
            terminal=terminal,
        )
        for name, phase, terminal in (
            ("ready", "ready", False),
            ("released", "released", False),
            ("park", "released", True),
        )
    }
    port = SimulationSegmentPolicyPort(
        _Simulation(articulation=art),
        robot,
        SimulationSceneBinding(
            registry_id="test_scene",
            articulations=(
                SimulationArticulationBinding(
                    entity_id="drawer", simulation_uid="native_drawer"
                ),
            ),
        ),
        settle_presets=presets,
        step_dt=0.04,
        control_operations={"press0": operation},
    )
    port.bind_clock(clock)

    def advance():
        for _ in range(10):
            time[0] += 0.004
            for observer in tuple(hooks.values()):
                token = observer.before_sample()
                sensor.update()
                observer.after_sample(0.004, token)
        clock.advance_after_env_step()

    return SimpleNamespace(
        port=port,
        operation=operation,
        robot=robot,
        art=art,
        clock=clock,
        data=data,
        advance=advance,
        fetches=fetches,
        hooks=hooks,
        sensor=sensor,
        sampler=sampler,
        time=time,
        geometry=geometry,
        segments={},
    )


def _run_control_policy(system, name):
    segment = (
        next(
            segment
            for segment in system.compiled.iter_segments()
            if segment.name == name
        )
        if getattr(system, "compiled", None) is not None
        else _compiled_articulation_segment(settle_preset=name)
    )
    policy = segment.post_policies[0]
    system.segments[name] = (segment, policy)
    commands = []
    for command in system.port.actions(
        policy, segment=segment, active_mask=torch.tensor([True])
    ):
        commands.append(command)
        system.advance()
    return commands, system.port.post_policy_metadata(policy, segment=segment)


def test_standard_control_port_preserves_press_holds_event_retreat_and_terminal_receipt():
    system = _press_control_port()
    commands, ready = _run_control_policy(system, "ready")
    assert len(commands) == 13 and all(
        torch.equal(command, torch.full((1, 2), 0.2)) for command in commands
    )
    assert ready["accepted"] and ready["phase"] == "released_stable"
    assert (
        system.fetches[0] == 132
    )  # 130 native substeps plus the original two arm refreshes.
    system.data["impulse"].fill_(0.1)
    system.advance()
    system.art._qpos.fill_(0.00006)
    system.advance()
    system.data["impulse"].zero_()
    commands, released = _run_control_policy(system, "released")
    assert len(commands) == 5 and all(
        torch.equal(command, torch.full((1, 2), 0.7)) for command in commands
    )
    assert (
        released["accepted"]
        and released["retreat_verified"]
        and not released["terminal"]
    )
    commands, terminal = _run_control_policy(system, "park")
    assert len(commands) == 5 and terminal["accepted"] and terminal["terminal"]
    assert terminal["activation_verified"] is False
    assert terminal["control"]["operation_id"] == "press0"
    assert terminal["control"]["sample_count"] == 121
    retained = system.operation.recorded_window()
    system.art.get_qpos = lambda: (_ for _ in ()).throw(
        AssertionError("historical reads must be detached")
    )
    assert system.operation.recorded_window().samples == retained.samples
    segment, policy = system.segments["park"]
    services = system.port.runtime_service_context(
        program_fingerprint="program", episode_id="episode"
    )
    assert services.control_operation("press0") is system.operation
    with pytest.raises(ValueError, match="another program/episode"):
        system.port.runtime_service_context(
            program_fingerprint="other", episode_id="episode"
        )
    with pytest.raises(TypeError):
        services.control_operations["wrong"] = system.operation
    scope = system.port.control_policy_scope(policy, segment=segment)
    facts = system.port.control_policy_measurements(policy, segment=segment)
    assert facts.phase == "released" and facts.terminal
    assert (
        facts.sample_count == len(retained.samples)
        and facts.generation == retained.generation
    )
    assert facts.clearance.batches["control.terminal_clearance"].values.tolist() == [
        0.05
    ]
    ready_segment, ready_policy = system.segments["ready"]
    ready_facts = system.port.control_policy_measurements(
        ready_policy, segment=ready_segment
    )
    assert len(ready_facts.hold_frames) == 13
    assert ready_facts.hold_end_step - ready_facts.hold_start_step == 13
    assert torch.equal(
        ready_facts.joint_limits.batches["articulation.joint_limits"].values,
        torch.tensor([[0.0, 0.01]]),
    )
    ready_facts.hold_frames[0].batches["articulation.joint_position"].values.fill_(99)
    assert system.port.control_policy_measurements(
        ready_policy, segment=ready_segment
    ).hold_frames[0].batches["articulation.joint_position"].values.tolist() == [0.0]

    assert (
        scope["program_fingerprint"] == "program" and scope["episode_id"] == "episode"
    )
    assert scope["contact_actors"] == {"target": 1, "parent": 4, "hand": [2]}
    assert (
        system.port.control_policy_window(policy, segment=segment).samples
        == retained.samples
    )
    earlier, earlier_policy = system.segments["released"]
    with pytest.raises(ValueError, match="another post-policy"):
        system.port.control_policy_window(earlier_policy, segment=earlier)
    system.port.close_runtime_resources()
    assert not system.hooks
    with pytest.raises(RuntimeError, match="closed"):
        services.control_operation("press0")
    assert (
        system.port.control_policy_window(policy, segment=segment).samples
        == retained.samples
    )


def test_control_port_cancellation_and_foreign_clock_cannot_reuse_success():
    from embodichain.lab.gym.envs.task_program.bridge import EnvironmentStepClock

    system = _press_control_port()
    _, ready = _run_control_policy(system, "ready")
    assert ready["accepted"]
    with pytest.raises(ValueError, match="fresh service"):
        system.port.new_task_program_session().bind_clock(EnvironmentStepClock(0.04))
    assert system.port.observation_clock is system.clock
    segment = _compiled_articulation_segment(settle_preset="released")
    policy = segment.post_policies[0]
    actions = system.port.actions(
        policy, segment=segment, active_mask=torch.tensor([True])
    )
    next(actions)
    other = system.port.actions(
        policy, segment=segment, active_mask=torch.tensor([True])
    )
    with pytest.raises(ValueError, match="executing post-policy"):
        next(other)
    system.advance()
    next(actions)  # Rejected overlap did not cancel the actual owner.
    actions.close()
    with pytest.raises(RuntimeError, match="unavailable"):
        system.port.post_policy_result(policy, segment=segment)
    with pytest.raises(RuntimeError, match="unavailable"):
        system.operation.receipt()


def _twist_control_case(
    *, program_fingerprint="program", compiled=None, initialize=None, retain=False
):
    from dataclasses import asdict
    from embodichain.lab.sim.scene.revolute_inputs import RevoluteControlBinding
    from embodichain.lab.sim.atomic_actions import (
        ActionPlan,
        EndpointCommand,
        JointPositionTarget,
        JointPositionPayload,
        RuntimeCommandFrame,
        TimedCommandSequence,
        RecoveryPolicy,
        TrackingPolicy,
        PlannerDiagnostics,
        TimedTrajectory,
    )
    from embodichain.lab.sim.atomic_actions.primitives.twist_recovery import (
        TwistRecoveryScope,
        TwistPlanningRecovery,
        TwistPlanningOutcome,
    )
    from embodichain.lab.task_program.integrations.simulation.twist_session import (
        TwistOperationSession,
    )
    from embodichain.lab.task_program.integrations.simulation.control_observation import (
        ControlContactActors,
    )
    from embodichain.lab.task_program.integrations.simulation.control_policy import (
        ControlOperationRuntime,
        SimulationControlConditionCfg,
    )

    system = _press_control_port(program_fingerprint=program_fingerprint)
    if compiled is not None:
        system.compiled = compiled
    system.port.close_runtime_resources()
    art, robot = system.art, system.robot
    art._qpos = torch.tensor([[0.0, 0.17]])
    art.joint_names = ("cabinet_to_drawer", "other_joint")
    art.cfg = SimpleNamespace(
        fpath="/frozen/native.usda", body_scale_mass_policy="native"
    )
    art.get_qvel = lambda: torch.zeros_like(art._qpos)
    system.geometry.park_clearance = lambda: 0.05
    binding = RevoluteControlBinding(
        "native_drawer",
        "cabinet_to_drawer",
        "button",
        "base",
        "a" * 64,
        1.0,
        (0.0, 0.0, 1.0),
        1.0,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.02),
        0.01,
        0.02,
        (-1.0, 1.0),
        1,
        0.3,
        calibration_sha256="b" * 64,
    )
    scope = TwistRecoveryScope(program_fingerprint, "twist0", "episode", 0)
    cursor = TwistOperationSession(
        robot,
        SimpleNamespace(robot=robot),
        scope,
        invocation_ids=("chunk",),
        parent_entity_id="drawer",
        child_entity_id="button",
        joint_name=binding.joint,
        validate_execution_scope=lambda: None,
        validate_observation=lambda context: None,
    )
    payload = {
        "binding": asdict(binding),
        "operation_id": "twist0",
        "success_criterion": "contact_backed_coarse_turn",
    }
    operation = ControlOperationRuntime(
        robot,
        art,
        binding,
        system.geometry,
        system.sampler,
        ControlContactActors(1, 4, (2,)),
        entity_id="drawer",
        operation_id="twist0",
        program_fingerprint=program_fingerprint,
        episode_id="episode",
        observer_id="twist",
        route_payload=payload,
        control_dt=0.04,
        physics_dt=0.004,
        simulation_time=lambda: system.time[0],
        validate_scope=lambda: None,
        twist_session=cursor,
        table_actor=10,
        robot_actors=(2, 3),
    )
    operation.start_startup()  # Reuses the actual completed query; no new fetch.
    assert system.fetches[0] == 1
    presets = {
        name: SimulationControlConditionCfg(
            entity="drawer",
            operation_id="twist0",
            family="twist",
            phase=phase,
            terminal=terminal,
        )
        for name, phase, terminal in (
            ("ready", "ready", False),
            ("chunk", "chunk", False),
            ("park", "released", True),
        )
    }
    system.port = SimulationSegmentPolicyPort(
        _Simulation(articulation=art),
        robot,
        SimulationSceneBinding(
            registry_id="test_scene",
            articulations=(
                SimulationArticulationBinding(
                    entity_id="drawer", simulation_uid="native_drawer"
                ),
            ),
        ),
        settle_presets=presets,
        step_dt=0.04,
        control_operations={"twist0": operation},
    )
    system.port.bind_clock(system.clock)
    system.operation = operation
    if initialize is not None:
        initialize(system)
    commands, ready = _run_control_policy(system, "ready")
    assert len(commands) == 13 and ready["accepted"]
    assert ready["startup_check"]["sample_count"] == 131
    assert system.fetches[0] == 133
    state = cursor.snapshot()
    outcome = TwistPlanningOutcome(
        "twist",
        "chunk",
        0,
        "plan",
        replace(
            state,
            version=state.version + 1,
            selected_index=0,
            selected_roll=0.0,
            selected_bite_depth=0.005,
            order=(0,),
            candidates=((0, 0.0, 0.0),),
        ),
    )
    ids = torch.tensor([0])
    positions = torch.zeros(1, 2)
    command = EndpointCommand(
        JointPositionTarget("arm", (0, 1)), JointPositionPayload(positions)
    )
    frame = RuntimeCommandFrame(
        (command,), torch.tensor([True]), ids, torch.tensor([0.04])
    )
    plan = ActionPlan(
        skill_id="twist",
        plan_success=torch.tensor([True]),
        commands=TimedCommandSequence((frame,), ids),
        recovery_policy=RecoveryPolicy(planning_recovery=TwistPlanningRecovery()),
        tracking_policy=TrackingPolicy.timed(),
        planned_scene_version=0,
        planned_collision_world_revision=(0,),
        diagnostics=PlannerDiagnostics("fixture"),
        joint_trajectory=TimedTrajectory.from_uniform_step(
            positions[:, None], env_ids=ids, step_dt=0.04
        ),
        invocation_id="chunk",
        invocation_revision=0,
        planning_recovery_outcome=outcome,
    )
    cursor.commit_planning(plan)
    cursor.adopt_plan(plan)
    system.data["impulse"].fill_(0.1)
    system.advance()
    art._qpos[:, 0].fill_(0.3)
    system.advance()
    system.data["impulse"].zero_()
    commands, chunk = _run_control_policy(system, "chunk")
    assert len(commands) == 13 and chunk["accepted"] and chunk["goal_complete"]
    commands, terminal = _run_control_policy(system, "park")
    assert all(torch.equal(value, torch.full((1, 2), 0.7)) for value in commands)
    assert len(commands) == 13 and terminal["accepted"] and terminal["terminal"]
    assert (
        terminal["stable"] and terminal["others_unchanged"] and terminal["early_stop"]
    )
    terminal_segment, terminal_policy = system.segments["park"]
    facts = system.port.control_policy_measurements(
        terminal_policy, segment=terminal_segment
    )
    assert torch.equal(
        facts.reference_qpos.batches["articulation.joint_positions"].values,
        torch.tensor([[0.0, 0.17]]),
    )
    assert torch.equal(
        facts.current_qpos.batches["articulation.joint_positions"].values,
        torch.tensor([[0.3, 0.17]]),
    )
    assert len(facts.hold_frames) == 13
    assert operation.recorded_window(startup=True).available
    assert operation.recorded_window().available
    if retain:
        return system
    system.port.close_runtime_resources()
    assert not system.hooks


def test_standard_control_port_runs_twist_startup_chunk_and_park_through_shared_cursor():
    _twist_control_case()


def test_support_height_uses_highest_qualified_contact_without_averaging():
    from embodichain.compute.geometry.contact_support import (
        contact_support_heights as _contact_support_heights,
    )

    row = {
        "contacts_world": [[-0.01, 0.0, 0.08], [0.01, 0.0, 0.12]],
        "center_of_mass_world": [0.0, 0.0, 0.2],
        "com_patch_distance_m": 0.0,
        "com_tolerance_m": 0.003,
        "upward_impulse": 0.1,
    }
    heights = _contact_support_heights(
        torch.tensor([True, False]),
        {"scope": "native_contact_support", "rows": [row, row]},
        torch.zeros(2),
    )
    assert heights[0].item() == pytest.approx(0.12)
    assert torch.isnan(heights[1])


def _full_control_program(names=("ready", "released", "park")):
    payload = dict(
        program_id="raw_control_test",
        integration=dict(
            robot_profile="test_robot",
            scene_registry="test_scene",
            runtime_preset="safe",
        ),
        targets={},
        program=dict(
            kind="sequence",
            items=[
                dict(
                    kind="segment",
                    name=name,
                    steps=dict(
                        kind="invoke",
                        call=dict(
                            kind="registered", call_id="example.control", arguments={}
                        ),
                    ),
                    post=[dict(kind="wait_stable", entity="drawer", preset=name)],
                )
                for name in names
            ],
        ),
    )
    registry = SceneRegistry(
        (
            SceneEntityRegistration(
                ref=SceneArticulationRef("drawer"),
                state_provider=_StaticStateProvider(),
            ),
        )
    )
    compiled = TaskProgramCompiler.from_scene_registry(registry).compile(
        decode_task_program(payload)
    )
    return payload, compiled


def _raw_press_observation(*, empty_initial=False):
    from types import SimpleNamespace
    from embodichain.lab.gym.envs.episode_context import EpisodeContext
    from embodichain.lab.gym.envs.task_program.bridge import (
        TaskProgramDemoBridge,
        BufferedGymCommandSink,
        RuntimeCommandFrameEncoder,
    )
    from embodichain.lab.task_program.integrations.invocation_policy import (
        task_program_fingerprint,
    )
    from embodichain.lab.task_program.integrations.simulation.control_observations import (
        SimulationControlObservationCollector,
    )

    payload, compiled = _full_control_program()
    system = _press_control_port(
        program_fingerprint=task_program_fingerprint(compiled), episode_id="0"
    )
    system.compiled = compiled
    runtime = SimpleNamespace(
        result=None,
        status=None,
        start=lambda *a, **k: None,
        step=lambda: None,
        cancel=lambda reason: None,
        adopt_verified_task_state=lambda state: None,
    )
    sink = BufferedGymCommandSink(
        RuntimeCommandFrameEncoder(
            SimpleNamespace(current_qpos=lambda ids: system.robot.get_qpos())
        ),
        system.clock,
    )
    bridge = TaskProgramDemoBridge(
        compiled,
        runtime,
        sink,
        system.clock,
        post_policy_port=system.port,
        isolate_policy_sessions=False,
    )
    epochs = EpisodeContext(1)
    epochs._begin_episode(torch.tensor([0]))
    epochs._finish_initialization(torch.tensor([0]))
    env = SimpleNamespace(
        sim=system.port._simulation, robot=system.robot, episode_context=epochs
    )
    collector = SimulationControlObservationCollector(env, bridge, "press0")
    segments = []
    selectors = []
    for name in ("ready", "released", "park"):
        if name == "released":
            if empty_initial:
                system.data["is_valid"].zero_()
                system.advance()
                system.data["is_valid"].fill_(True)
            system.data["impulse"].fill_(0.1)
            system.advance()
            system.art._qpos.fill_(0.00006)
            system.advance()
            system.data["impulse"].zero_()
        _, result = _run_control_policy(system, name)
        segment, policy = system.segments[name]
        scope = system.port.control_policy_scope(policy, segment=segment)
        selector = dict(
            segment_id=segment.segment_id,
            segment_name=segment.name,
            source_path=list(segment.source_path),
            policy_index=0,
            preset=name,
            phase=scope["phase"],
            terminal=scope["terminal"],
        )
        selectors.append(selector)
        segments.append(
            dict(
                name=segment.name,
                active=[True],
                successes=[True],
                end_steps=[system.clock.step_index],
                metadata=dict(
                    task_program_id=compiled.program_id,
                    program_segment_id=segment.segment_id,
                    program_segment_source_path=list(segment.source_path),
                    runtime=dict(env_ids=[0]),
                    post_policies=[
                        dict(
                            kind="wait_stable",
                            policy_index=0,
                            source_path=list(policy.source_path),
                            result_mask=[True],
                            result=result,
                        )
                    ],
                ),
            )
        )
    system.operation.close()
    system.art.get_qpos = lambda: (_ for _ in ()).throw(
        AssertionError("Control evidence capture cannot read live-later state")
    )
    raw = collector.capture(selectors, terminal_policy=selectors[-1])
    return system, collector, payload, segments, selectors, raw
