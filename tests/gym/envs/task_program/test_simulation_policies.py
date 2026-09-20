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
from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.task_program import TaskProgramCompiler, decode_task_program
from embodichain.lab.task_program.integrations import (
    ControlPartCommandPreset,
    ControlPartEndpointBinding,
    ControlPartResourceBinding,
    SimulationRobotSkillProfileBinding,
    SimulationArticulationBinding,
    SimulationRigidObjectBinding,
    SimulationSceneBinding,
)
from embodichain.lab.gym.envs.task_program.bridge import (
    SegmentPostPolicyPort,
    SegmentValidatorPort,
)
from embodichain.lab.task_program.integrations.simulation.policies import (
    SimulationSegmentPolicyPort,
)
from embodichain.lab.gym.envs.settling import DynamicSettleMonitorCfg
from embodichain.lab.sim.atomic_actions import EntityState
from embodichain.lab.task_program.semantics.scene import (
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

    def get_qpos(self, name: str | None = None, target: bool = False) -> torch.Tensor:
        self.qpos_reads.append(target)
        value = self.target_qpos if target else self.current_qpos
        return (value[:, :1] if name == "hand" else value).clone()


class _Articulation:
    """Small articulation double with one measured prismatic joint."""

    joint_names = ("cabinet_to_drawer",)

    def __init__(self, qpos: torch.Tensor) -> None:
        self._qpos = qpos
        self.body_data = SimpleNamespace(
            body_link_vel=torch.zeros(qpos.shape[0], 1, 6),
        )

    def get_qpos(self) -> torch.Tensor:
        """Return the configured batched joint state."""
        return self._qpos.clone()


class _Simulation:
    """Resolve only explicitly selected native test entities."""

    def __init__(
        self,
        entity: _RigidObject | None = None,
        articulation: _Articulation | None = None,
    ) -> None:
        self.entity = entity
        self.articulation = articulation

    def get_rigid_object(self, uid: str) -> _RigidObject | None:
        return self.entity if uid == "native_cube" else None

    def get_articulation(self, uid: str) -> _Articulation | None:
        return self.articulation if uid == "native_drawer" else None


def _compiled_segment(
    *,
    settle_preset: str = "fast",
    release_resource=None,
    return_program=False,
    mutate_payload=None,
):
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
    if release_resource is not None:
        payload["program"]["validators"][0]["release_resource"] = release_resource
    if mutate_payload is not None:
        mutate_payload(payload)
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
    return compiled if return_program else next(compiled.iter_segments())


def _compiled_articulation_segment():
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


def _port(
    positions: torch.Tensor,
    *,
    preset: DynamicSettleMonitorCfg | None = None,
    target_qpos: torch.Tensor | None = None,
    robot_profile=None,
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
        robot_profile=robot_profile,
        settle_presets={
            "fast": preset
            or DynamicSettleMonitorCfg(
                min_steps=0,
                max_steps=3,
                check_interval_steps=1,
                required_stable_checks=2,
            )
        },
    )
    return port, entity, robot


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


def _release_profile():
    """Declare the real open command independently of commanded drive targets."""
    return SimulationRobotSkillProfileBinding(
        profile_id="test_robot",
        resources=(
            ControlPartResourceBinding(
                resource_id="primary_manipulator",
                endpoints=(
                    ControlPartEndpointBinding(
                        endpoint_id="grasp",
                        control_part="hand",
                        capabilities=frozenset({"interaction.grasp"}),
                        command_preset="jaw_commands",
                    ),
                ),
            ),
        ),
        command_presets=(
            ControlPartCommandPreset(
                preset_id="jaw_commands",
                control_part="hand",
                commands={"open": (0.0,), "grasp": (0.024,)},
            ),
        ),
        defaults={"place": {"primary": "primary_manipulator"}},
    ).declare()


def test_release_validator_reads_actual_gripper_not_open_target() -> None:
    segment = _compiled_segment(release_resource="primary_manipulator")
    port, _, robot = _port(torch.zeros(2, 3), robot_profile=_release_profile())
    robot.current_qpos[:, 0] = torch.tensor([0.001, 0.024])
    robot.target_qpos[:, 0] = 0.0
    result = port.validate(segment.validators[0], segment=segment)
    assert result.tolist() == [True, False]
    assert port.validator_metadata(segment.validators[0], segment=segment)[
        "release_mask"
    ] == [True, False]


def test_measured_acceptance_requires_current_completed_stability_and_release() -> None:
    program = _compiled_segment(
        release_resource="primary_manipulator", return_program=True
    )
    segment = next(program.iter_segments())
    port, _, robot = _port(torch.zeros(2, 3), robot_profile=_release_profile())
    robot.current_qpos[:, 0] = torch.tensor([0.0, 0.024])
    port.validate_measured_acceptance(program)
    assert port.measured_success_mask(program).tolist() == [False, False]
    port.validate(segment.validators[0], segment=segment)
    assert port.measured_success_mask(program).tolist() == [False, False]
    list(
        port.actions(
            segment.post_policies[0],
            segment=segment,
            active_mask=torch.ones(2, dtype=torch.bool),
        )
    )
    assert port.measured_success_mask(program).tolist() == [True, False]
    port.validate_measured_acceptance(program)
    assert port.measured_success_mask(program).tolist() == [False, False]


def test_measured_acceptance_rejects_position_only_program() -> None:
    program = _compiled_segment(return_program=True)
    port, _, _ = _port(torch.zeros(2, 3), robot_profile=_release_profile())
    with pytest.raises(ValueError, match="release"):
        port.validate_measured_acceptance(program)


@pytest.mark.parametrize(
    "field,value",
    [
        ("release_resource", ""),
        ("release_tolerance", 0),
        ("release_tolerance", True),
        ("release_tolerance", float("nan")),
    ],
)
def test_release_validator_rejects_invalid_contract(field, value) -> None:
    from embodichain.lab.task_program.language.schema import (
        ObjectNearTargetValidatorCfg,
    )

    with pytest.raises((TypeError, ValueError)):
        ObjectNearTargetValidatorCfg(object="cube", target="drop", **{field: value})


@pytest.mark.parametrize(
    "failure",
    [
        "missing_settle",
        "wrong_target",
        "wrong_resource",
        "terminal_pick",
        "missing_second_segment",
    ],
)
def test_measured_acceptance_rejects_unqualified_task_outcomes(failure: str) -> None:
    def mutate(payload):
        segment = payload["program"]
        if failure == "missing_settle":
            segment["post"] = []
        elif failure == "wrong_target":
            payload["targets"]["elsewhere"] = {
                "kind": "cyclic_pose",
                "values": [{"position": [1, 0, 0], "quaternion_xyzw": [0, 0, 0, 1]}],
            }
            segment["validators"][0]["target"] = "elsewhere"
        elif failure == "wrong_resource":
            segment["validators"][0]["release_resource"] = "unrelated_hand"
        elif failure == "terminal_pick":
            segment["steps"]["call"] = {"kind": "pick", "object": "cube"}
        else:
            from copy import deepcopy

            second = deepcopy(segment)
            second["validators"] = []
            payload["program"] = {"kind": "sequence", "items": [segment, second]}

    program = _compiled_segment(
        release_resource="primary_manipulator",
        return_program=True,
        mutate_payload=mutate,
    )
    port, _, _ = _port(torch.zeros(2, 3), robot_profile=_release_profile())
    with pytest.raises(ValueError, match="Measured acceptance"):
        port.validate_measured_acceptance(program)
    with pytest.raises(ValueError, match="no current"):
        port.measured_success_mask(program)


def test_measured_acceptance_rejects_nonfinite_release_and_unsettled_rows() -> None:
    program = _compiled_segment(
        release_resource="primary_manipulator", return_program=True
    )
    segment = next(program.iter_segments())
    port, entity, robot = _port(torch.zeros(2, 3), robot_profile=_release_profile())
    robot.current_qpos[:, 0] = torch.tensor([float("nan"), 0.0])
    entity.body_data.lin_vel[1, 0] = 1.0
    port.validate_measured_acceptance(program)
    list(
        port.actions(
            segment.post_policies[0],
            segment=segment,
            active_mask=torch.ones(2, dtype=torch.bool),
        )
    )
    port.validate(segment.validators[0], segment=segment)
    assert port.measured_success_mask(program).tolist() == [False, False]
    metadata = port.validator_metadata(segment.validators[0], segment=segment)
    assert metadata["release_error"] == [None, 0.0]


@pytest.mark.parametrize("empty_scope", ["program", "segment"])
def test_empty_task_cannot_reach_measured_qualification(empty_scope: str) -> None:
    """The strict compiler boundary rejects empty flows before qualification."""

    def mutate(payload):
        empty = {"kind": "sequence", "items": []}
        if empty_scope == "program":
            payload["program"] = empty
        else:
            payload["program"]["steps"] = empty

    with pytest.raises(ValueError):
        _compiled_segment(
            release_resource="primary_manipulator",
            return_program=True,
            mutate_payload=mutate,
        )
