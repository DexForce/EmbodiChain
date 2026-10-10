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

"""Production reset ordering and complete B=1 scene initial-state restoration."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
from embodichain.lab.sim.motion.expansion import ValidationCheck, ValidationResult
from embodichain.lab.sim.scene_expansion import ScenePoseChange, SceneVariant
from embodichain.lab.task_program.integrations.simulation.scene_expansion import (
    SimulationSceneExpansionHost,
)
from embodichain.utils.math import quat_from_matrix

_DEFAULT_CUBE_X = 0.1
_CANDIDATE_CUBE_X = 0.6
_SETTLE_DELTA = 0.03


class _Rigid:
    body_type = "dynamic"

    def __init__(self, x: float) -> None:
        self.value = torch.tensor(
            [[x, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.3]]
        )
        self.default = self.value.clone()

    @property
    def body_state(self) -> torch.Tensor:
        return self.value

    def clear_dynamics(self, *, env_ids: list[int]) -> None:
        assert env_ids == [0]
        self.value[:, 7:] = 0

    def set_local_pose(self, value: torch.Tensor, *, env_ids: list[int]) -> None:
        assert env_ids == [0]
        if value.shape == (1, 4, 4):
            self.value[:, :3] = value[:, :3, 3]
            self.value[:, 3:7] = quat_from_matrix(value[:, :3, :3])
        else:
            self.value[:, :7] = value

    def set_velocity(
        self, *, lin_vel: torch.Tensor, ang_vel: torch.Tensor, env_ids: list[int]
    ) -> None:
        assert env_ids == [0]
        self.value[:, 7:10] = lin_vel
        self.value[:, 10:13] = ang_vel


class _Articulation:
    joint_names = ("joint_a", "joint_b")

    def __init__(self) -> None:
        self.state = {
            "root_pose": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
            "root_lin_vel": torch.tensor([[0.1, 0.2, 0.3]]),
            "root_ang_vel": torch.tensor([[0.4, 0.5, 0.6]]),
            "qpos": torch.tensor([[0.1, 0.2]]),
            "qvel": torch.tensor([[0.3, 0.4]]),
            "target_qpos": torch.tensor([[0.5, 0.6]]),
            "target_qvel": torch.tensor([[0.7, 0.8]]),
            "qf": torch.tensor([[0.9, 1.0]]),
        }
        self.default = deepcopy(self.state)
        self.body_data = SimpleNamespace(fetch_state=lambda: self.state)

    def get_qpos(self, *, target: bool = False) -> torch.Tensor:
        return self.state["target_qpos" if target else "qpos"]

    def get_qvel(self, *, target: bool = False) -> torch.Tensor:
        return self.state["target_qvel" if target else "qvel"]

    def get_qf(self) -> torch.Tensor:
        return self.state["qf"]

    def set_state(
        self,
        *,
        env_ids: list[int],
        clear_dynamics: bool,
        root_velocity: torch.Tensor,
        **kwargs,
    ) -> None:
        assert env_ids == [0] and clear_dynamics
        self.state.update({key: value.clone() for key, value in kwargs.items()})
        self.state["root_lin_vel"] = root_velocity[:, :3].clone()
        self.state["root_ang_vel"] = root_velocity[:, 3:].clone()


class _Simulation:
    device = torch.device("cpu")
    physics_backend = "default"

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.rigid = {"cube": _Rigid(_DEFAULT_CUBE_X), "other": _Rigid(-0.4)}
        self.robots = {"robot": _Articulation()}
        self.articulations = {"drawer": _Articulation()}
        self.groups = []
        self.deformables = []
        self.constraints = []
        self.updated_steps = 0

    def get_rigid_object_uid_list(self) -> list[str]:
        return list(self.rigid)

    def get_robot_uid_list(self) -> list[str]:
        return list(self.robots)

    def get_articulation_uid_list(self) -> list[str]:
        return list(self.articulations)

    def get_rigid_object_group_uid_list(self) -> list[str]:
        return self.groups

    def get_deformable_object_uid_list(self) -> list[str]:
        return self.deformables

    def get_rigid_constraint_uid_list(self) -> list[str]:
        return self.constraints

    def get_rigid_object(self, uid: str) -> _Rigid | None:
        return self.rigid.get(uid)

    def get_robot(self, uid: str) -> _Articulation | None:
        return self.robots.get(uid)

    def get_articulation(self, uid: str) -> _Articulation | None:
        return self.articulations.get(uid)

    def reset_objects_state(self, **kwargs) -> None:
        self.events.append("reset_objects")
        for obj in self.rigid.values():
            obj.value.copy_(obj.default)
        for obj in (*self.robots.values(), *self.articulations.values()):
            obj.state = deepcopy(obj.default)

    def update(self, *, step: int) -> None:
        self.events.append("settle")
        self.updated_steps += step
        self.rigid["cube"].value[:, 0] += _SETTLE_DELTA

    def render_frame(self, **kwargs):
        self.events.append("render")
        return nullcontext()


class _Environment(EmbodiedEnv):
    """Run the actual BaseEnv/EmbodiedEnv reset with CPU scene state doubles."""

    def __init__(self) -> None:
        self.events = []
        self.sim = _Simulation(self.events)
        self._num_envs = 1
        self.cfg = SimpleNamespace(sim_steps_per_control=4)
        self.sim_cfg = SimpleNamespace(physics_dt=0.01)
        self._profiler = SimpleNamespace(section=lambda *a, **k: nullcontext())
        self._detached_uids_for_reset = []
        self._elapsed_steps = torch.ones(1, dtype=torch.int32)
        self.sensors = {
            "contact": SimpleNamespace(
                uid="contact", reset=lambda **k: self.events.append("sensor_reset")
            )
        }
        self.observation_manager = SimpleNamespace(
            reset=lambda **k: self.events.append("observation_reset")
        )
        self.rollout_buffer = TensorDict(
            {"obs": torch.zeros(1, 4, 1)}, batch_size=[1, 4]
        )
        self.rollout_steps = torch.zeros(1, dtype=torch.long)
        self._rollout_buffer_mode = "expert"
        self._max_rollout_steps = 4
        self._traj_buffer = None
        self._active_task_program_bridge = object()

    def is_task_success(self) -> torch.Tensor:
        return torch.zeros(1, dtype=torch.bool)

    def _initialize_episode(self, env_ids, **kwargs) -> None:
        assert "scene_expansion_prepare" not in kwargs
        self.events.append("reset_events")
        self.sim.rigid["cube"].value[:, 0] = 0.2

    def _reset_physical_objective(self, env_ids) -> None:
        self.events.append("objective")
        self.objective_initial_x = self.sim.rigid["cube"].value[0, 0].item()

    def get_obs(self, **kwargs) -> torch.Tensor:
        assert "scene_expansion_prepare" not in kwargs
        self.events.append("observation")
        return self.sim.rigid["cube"].value[:, :1].clone()

    def get_info(self, **kwargs) -> dict:
        return {}

    def _seed_recording_state(self, obs: torch.Tensor, env_ids: torch.Tensor) -> None:
        super()._seed_recording_state(obs, env_ids)
        self.events.append("seed_recording")


def _passed() -> ValidationResult:
    return ValidationResult((ValidationCheck("actual.initial", "passed"),))


def _variant() -> SceneVariant:
    pose = torch.eye(4)
    pose[0, 3] = _CANDIDATE_CUBE_X
    pose[2, 3] = 0.1
    return SceneVariant(
        "asset-revision:1", 7, 0, (ScenePoseChange("cube", pose.tolist()),)
    )


def test_preparation_uses_final_settled_state_for_objective_and_first_recording() -> (
    None
):
    env = _Environment()
    host = SimulationSceneExpansionHost(env, initial_validator=_passed, settle_steps=2)
    result = host(_variant())
    expected = _CANDIDATE_CUBE_X + _SETTLE_DELTA
    assert result.validation.accepted
    assert env.objective_initial_x == pytest.approx(expected)
    assert env.rollout_buffer["obs"][0, 0, 0].item() == pytest.approx(expected)
    assert env._active_task_program_bridge is None
    assert env.sim.updated_steps == 2
    assert env._elapsed_steps.tolist() == [0]
    assert (
        env.events.index("reset_events")
        < env.events.index("settle")
        < env.events.index("objective")
        < env.events.index("observation")
        < env.events.index("seed_recording")
    )


def test_initial_checks_reject_actual_pose_after_settling() -> None:
    env = _Environment()

    def validate() -> ValidationResult:
        allowed = env.sim.rigid["cube"].value[0, 0] <= _CANDIDATE_CUBE_X
        return ValidationResult(
            (ValidationCheck("support.bounds", "passed" if allowed else "failed"),)
        )

    host = SimulationSceneExpansionHost(env, initial_validator=validate, settle_steps=1)
    assert not host(_variant()).validation.accepted
    assert host.initial_state is not None


def test_snapshot_owns_every_entity_and_control_target_after_live_buffers_change() -> (
    None
):
    env = _Environment()
    host = SimulationSceneExpansionHost(env, initial_validator=_passed)
    snapshot = host.capture_initial_state(parent_scene_id="assets:1")
    metadata = deepcopy(snapshot.to_metadata())
    for obj in env.sim.rigid.values():
        obj.value.add_(1)
    for obj in (*env.sim.robots.values(), *env.sim.articulations.values()):
        for value in obj.state.values():
            value.add_(1)
    assert snapshot.to_metadata() == metadata
    states = {entry["uid"]: entry for entry in metadata["entities"]}
    assert set(states) == {"cube", "other", "robot", "drawer"}
    assert states["robot"]["qpos"] != states["robot"]["target_qpos"]
    assert states["robot"]["qvel"] != states["robot"]["target_qvel"]


def test_restore_overwrites_reset_side_effects_for_all_state_and_first_frame() -> None:
    env = _Environment()
    host = SimulationSceneExpansionHost(env, initial_validator=_passed)
    host(_variant())
    snapshot = host.initial_state
    assert snapshot is not None
    for obj in env.sim.rigid.values():
        obj.value.add_(1)
    for obj in (*env.sim.robots.values(), *env.sim.articulations.values()):
        for value in obj.state.values():
            value.add_(1)
    restored = host.restore_initial_state(snapshot)
    assert restored.initial_state_id == snapshot.initial_state_id
    assert host.initial_state.to_metadata() == snapshot.to_metadata()
    assert env.rollout_buffer["obs"][0, 0, 0].item() == pytest.approx(_CANDIDATE_CUBE_X)
    assert env.sim.updated_steps == 0


@pytest.mark.parametrize("delta,accepted", [(1e-7, True), (1e-3, False)])
def test_restore_checks_actual_round_trip_state_and_preserves_actual_identity(
    delta: float,
    accepted: bool,
) -> None:
    env = _Environment()
    host = SimulationSceneExpansionHost(env, initial_validator=_passed)
    snapshot = host.capture_initial_state(parent_scene_id="assets:1")
    cube = env.sim.rigid["cube"]
    original = cube.set_local_pose

    def round_trip(pose: torch.Tensor, *, env_ids: list[int]) -> None:
        original(pose, env_ids=env_ids)
        cube.value[:, 0] += delta

    cube.set_local_pose = round_trip
    result = host.restore_initial_state(snapshot)
    assert result.validation.accepted is accepted
    assert result.validation.checks[0].check_id == "scene.restore_state"
    assert result.initial_state_id != snapshot.initial_state_id
    assert (
        result.metadata["restored_from_initial_state_id"] == snapshot.initial_state_id
    )


def test_restoration_allows_equivalent_quaternion_signs() -> None:
    env = _Environment()
    host = SimulationSceneExpansionHost(env, initial_validator=_passed)
    snapshot = host.capture_initial_state(parent_scene_id="assets:1")
    cube = env.sim.rigid["cube"]
    original = cube.set_local_pose

    def opposite_sign(pose: torch.Tensor, *, env_ids: list[int]) -> None:
        original(pose, env_ids=env_ids)
        cube.value[:, 3:7].neg_()

    cube.set_local_pose = opposite_sign
    assert host.restore_initial_state(snapshot).validation.accepted


@pytest.mark.parametrize(
    "field,delta,accepted",
    [
        ("qpos", 1.2e-5, True),  # Observed native UR5 joint-limit round-trip clamp.
        ("qpos", 1.1e-4, False),
        ("target_qpos", 1.2e-5, False),
        ("qvel", 1.2e-5, False),
        ("target_qvel", 1.2e-5, False),
        ("qf", 1.2e-5, False),
    ],
)
def test_restoration_joint_clamp_tolerance_does_not_relax_other_saved_fields(
    field: str,
    delta: float,
    accepted: bool,
) -> None:
    env = _Environment()
    host = SimulationSceneExpansionHost(env, initial_validator=_passed)
    snapshot = host.capture_initial_state(parent_scene_id="assets:1")
    robot = env.sim.robots["robot"]
    write = robot.set_state

    def native_round_trip(**kwargs) -> None:
        write(**kwargs)
        robot.state[field][0, 0] += delta

    robot.set_state = native_round_trip
    result = host.restore_initial_state(snapshot)
    assert result.validation.accepted is accepted
    metrics = result.validation.checks[0].metrics
    assert metrics["absolute_tolerance"] == 1e-5
    assert metrics["qpos_absolute_tolerance"] == 1e-4
    assert metrics[
        "max_abs_qpos_error" if field == "qpos" else "max_abs_other_state_error"
    ] == pytest.approx(delta, abs=1e-7)


@pytest.mark.parametrize(
    "field,value",
    [
        ("qpos", (0.1,)),
        ("qvel", (float("nan"), 0.1)),
        ("joint_names", ("same", "same")),
        ("pose", (0.0,) * 7),
    ],
)
def test_malformed_snapshot_replacement_is_rejected_before_restore_writes(
    field: str,
    value: tuple,
) -> None:
    env = _Environment()
    host = SimulationSceneExpansionHost(env, initial_validator=_passed)
    snapshot = host.capture_initial_state(parent_scene_id="assets:1")
    articulation = next(
        state for state in snapshot._states if state.kind == "articulation"
    )
    with pytest.raises(ValueError, match="Snapshot"):
        replace(articulation, **{field: value})
    assert env.events == []


def test_foreign_snapshot_is_rejected_before_reset() -> None:
    first_env = _Environment()
    first = SimulationSceneExpansionHost(first_env, initial_validator=_passed)
    snapshot = first.capture_initial_state(parent_scene_id="assets:1")
    second_env = _Environment()
    second = SimulationSceneExpansionHost(second_env, initial_validator=_passed)
    with pytest.raises(ValueError, match="another scene host"):
        second.restore_initial_state(snapshot)
    assert second_env.events == []


def test_valid_but_different_snapshot_joint_names_fail_before_reset() -> None:
    env = _Environment()
    host = SimulationSceneExpansionHost(env, initial_validator=_passed)
    snapshot = host.capture_initial_state(parent_scene_id="assets:1")
    states = tuple(
        (
            replace(state, joint_names=("different_a", "different_b"))
            if state.kind == "robot"
            else state
        )
        for state in snapshot._states
    )
    altered = replace(snapshot, _states=states)
    with pytest.raises(ValueError, match="joint names"):
        host.restore_initial_state(altered)
    assert env.events == []


def test_state_identity_changes_with_unmoved_object_or_controller_targets() -> None:
    env = _Environment()
    host = SimulationSceneExpansionHost(env, initial_validator=_passed)
    first = host.capture_initial_state(parent_scene_id="assets:1").initial_state_id
    env.sim.rigid["other"].value[0, 0] += 0.1
    second = host.capture_initial_state(parent_scene_id="assets:1").initial_state_id
    env.sim.robots["robot"].state["target_qvel"][0, 0] += 0.1
    third = host.capture_initial_state(parent_scene_id="assets:1").initial_state_id
    assert len({first, second, third}) == 3


@pytest.mark.parametrize("collection", ["groups", "deformables", "constraints"])
def test_unsupported_physical_state_fails_before_reset(collection: str) -> None:
    env = _Environment()
    getattr(env.sim, collection).append("unsupported")
    with pytest.raises(ValueError, match="do not support"):
        SimulationSceneExpansionHost(env, initial_validator=_passed)
    assert env.events == []


def test_changed_or_missing_entities_fail_before_any_reset_write() -> None:
    env = _Environment()
    host = SimulationSceneExpansionHost(env, initial_validator=_passed)
    env.sim.rigid["extra"] = _Rigid(0.5)
    with pytest.raises(ValueError, match="entity set"):
        host(_variant())
    assert env.events == []


def test_fixed_robot_or_unknown_entity_pose_change_fails_before_reset() -> None:
    env = _Environment()
    host = SimulationSceneExpansionHost(env, initial_validator=_passed)
    variant = SceneVariant(
        "assets:1", 0, 0, (ScenePoseChange("robot", torch.eye(4).tolist()),)
    )
    with pytest.raises(ValueError, match="existing rigid object"):
        host(variant)
    assert env.events == []


@pytest.mark.parametrize("selection", [[], [1], [False], [0.0], [[0]]])
def test_partial_or_multienv_reset_callback_rejected_before_state_writes(
    selection: list,
) -> None:
    env = _Environment()
    with pytest.raises(ValueError, match="complete B=1"):
        env.reset(
            options={"reset_ids": selection, "scene_expansion_prepare": lambda: None}
        )
    env._num_envs = 2
    with pytest.raises(ValueError, match="complete B=1"):
        env.reset(options={"scene_expansion_prepare": lambda: None})
    assert env.events == []


def test_invalid_callback_rejected_before_seed_or_reset_mutation() -> None:
    env = _Environment()
    env._set_seed = lambda seed: pytest.fail("seed must not run")
    with pytest.raises(TypeError, match="must be callable"):
        env.reset(seed=4, options={"scene_expansion_prepare": 2})
    assert env.events == []


def test_preparation_error_prevents_final_observation_and_recorder_seeding() -> None:
    env = _Environment()

    def fail() -> None:
        raise RuntimeError("scene preparation failed")

    with pytest.raises(RuntimeError, match="scene preparation failed"):
        env.reset(options={"scene_expansion_prepare": fail})
    assert "reset_events" in env.events
    assert "objective" not in env.events
    assert "observation" not in env.events
    assert "seed_recording" not in env.events


def test_reset_does_not_mutate_callers_options() -> None:
    env = _Environment()
    options = {"scene_expansion_prepare": lambda: env.events.append("prepare")}
    env.reset(options=options)
    assert "scene_expansion_prepare" in options
    assert (
        env.events.index("reset_events")
        < env.events.index("prepare")
        < env.events.index("objective")
    )
