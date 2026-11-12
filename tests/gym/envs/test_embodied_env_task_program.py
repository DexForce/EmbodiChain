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

"""Tests for explicit Task Program environment integration hooks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.gym.envs.demo import DemoSegment
from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.task_program import CompiledTaskProgram


class _FakeBridge:
    """Minimal bridge protocol used by the environment adapter test."""

    def __init__(self, segment: DemoSegment) -> None:
        self._segment = segment
        self.iteration_count = 0

    def iter_segments(self):
        """Yield the configured segment lazily."""
        self.iteration_count += 1
        yield self._segment


class _DeclarativeEnv(EmbodiedEnv):
    """Environment stub with explicit compiler and bridge factories."""

    def compile_task_program(self, program):
        self.compiled_input = program
        return self.compiled_program

    def create_task_program_bridge(self, program):
        self.bridge_input = program
        return self.bridge


def _uninitialized_env(cls: type[EmbodiedEnv], task_program: object) -> EmbodiedEnv:
    """Create an environment instance without starting simulation."""
    env = object.__new__(cls)
    env.cfg = SimpleNamespace(task_program=task_program)
    env.sim = SimpleNamespace(device=torch.device("cpu"))
    env._num_envs = 2
    env._task_program_adapter = None
    env._active_task_program_bridge = None
    return env


def test_embodied_env_cfg_disables_task_program_by_default() -> None:
    """Declarative execution remains an explicit opt-in configuration."""
    cfg = EmbodiedEnvCfg()

    assert cfg.task_program is None


def test_create_demo_segments_uses_explicit_compiler_and_bridge_hooks() -> None:
    """Configured programs flow through provider and runtime factories lazily."""
    program = object()
    compiled_program = object()
    expected_segment = DemoSegment(actions=(), name="declarative")
    bridge = _FakeBridge(expected_segment)
    env = _uninitialized_env(_DeclarativeEnv, program)
    env.compiled_program = compiled_program
    env.bridge = bridge

    segments = env.create_demo_segments(debug_mode=True)

    assert bridge.iteration_count == 0
    assert tuple(segments) == (expected_segment,)
    assert bridge.iteration_count == 1
    assert env.compiled_input is program
    assert env.bridge_input is compiled_program
    assert env._active_task_program_bridge is bridge


def test_episode_program_takes_precedence_over_static_configuration() -> None:
    """Callers can select a fresh declarative program for one episode."""
    configured_program = object()
    episode_program = object()
    compiled_program = object()
    bridge = _FakeBridge(DemoSegment(actions=(), name="episode"))
    env = _uninitialized_env(_DeclarativeEnv, configured_program)
    env.compiled_program = compiled_program
    env.bridge = bridge

    segments = env.create_demo_segments(task_program=episode_program)

    assert tuple(segments)[0].name == "episode"
    assert env.compiled_input is episode_program
    assert env.bridge_input is compiled_program


def test_episode_compiled_program_bypasses_recompilation() -> None:
    """A trusted MLLM frontend can pass its canonical compiled snapshot."""
    compiled_program = object.__new__(CompiledTaskProgram)
    bridge = _FakeBridge(DemoSegment(actions=(), name="model"))
    env = _uninitialized_env(_DeclarativeEnv, object())
    env.bridge = bridge

    segments = env.create_demo_segments(task_program=compiled_program)

    assert tuple(segments)[0].name == "model"
    assert not hasattr(env, "compiled_input")
    assert env.bridge_input is compiled_program


def test_configured_program_requires_explicit_adapter() -> None:
    """The base environment never guesses a live scene or runtime provider."""
    env = _uninitialized_env(EmbodiedEnv, object())

    with pytest.raises(NotImplementedError, match="task_program_adapter"):
        env.create_demo_segments()


def test_task_program_success_is_false_before_normal_bridge_completion() -> None:
    """A configured program never inherits the base environment's true default."""
    env = _uninitialized_env(EmbodiedEnv, object())

    assert env.is_task_success().tolist() == [False, False]

    env._active_task_program_bridge = SimpleNamespace(program_completed=False)
    assert env.is_task_success().tolist() == [False, False]


def test_task_program_success_uses_the_completed_bridge_mask() -> None:
    """Normal bridge completion publishes its row-local validator acceptance."""
    env = _uninitialized_env(EmbodiedEnv, object())
    env._active_task_program_bridge = SimpleNamespace(
        program_completed=True,
        completion_mask=torch.tensor([True, False]),
    )

    success = env.is_task_success()

    assert success.dtype == torch.bool
    assert success.device == env.device
    assert success.tolist() == [True, False]


def test_dynamic_task_program_success_does_not_require_static_config() -> None:
    """Episode-injected programs publish the same completed bridge result."""
    env = _uninitialized_env(EmbodiedEnv, None)
    env._active_task_program_bridge = SimpleNamespace(
        program_completed=True,
        completion_mask=torch.tensor([False, True]),
    )

    assert env.is_task_success().tolist() == [False, True]


def test_direct_dynamic_program_planning_binds_current_episode_language() -> None:
    env = _uninitialized_env(
        _DeclarativeEnv, SimpleNamespace(instruction="Static task")
    )
    env.compiled_program = SimpleNamespace(instruction="Selected task")
    env.bridge = _FakeBridge(DemoSegment(actions=(), name="dynamic"))
    env._demo_episode_metadata = [{}, {}]
    tuple(
        env.create_demo_segments(
            task_program=SimpleNamespace(instruction="Selected task")
        )
    )
    assert [item["instruction"] for item in env._demo_episode_metadata] == [
        "Selected task"
    ] * 2
    assert [item["instruction_source"] for item in env._demo_episode_metadata] == [
        "task_program"
    ] * 2


def test_direct_planning_keeps_an_explicit_executor_language_snapshot() -> None:
    env = _uninitialized_env(_DeclarativeEnv, None)
    env.compiled_program = SimpleNamespace(instruction="Program default")
    env.bridge = _FakeBridge(DemoSegment(actions=(), name="dynamic"))
    env._demo_instruction_snapshot = ("Explicit task", "episode")
    env._demo_episode_metadata = [{"instruction": "Explicit task"}]
    tuple(env.create_demo_segments(task_program=object()))
    assert env._demo_episode_metadata[0]["instruction"] == "Explicit task"


class _CompletedDemoBridge(_FakeBridge):
    """Publish a successful bridge only after the one supplied segment."""

    program_completed = False
    completion_mask = torch.ones(2, dtype=torch.bool)
    expansion_records = ()

    def iter_segments(self):
        yield from super().iter_segments()
        self.program_completed = True


def _final_acceptance_env():
    from contextlib import nullcontext
    from unittest.mock import Mock

    from embodichain.lab.evaluation import (
        EvaluationSpec,
        PredicateSpec,
        builtin_predicates,
    )
    from embodichain.lab.gym.envs.episode_context import EpisodeContext
    from embodichain.lab.gym.envs.task_program._final_acceptance import (
        _deployment_final_rules,
    )
    from embodichain.lab.gym.utils.trajectory_state import _evaluation_observation_cfgs
    from embodichain.lab.task_program.integrations.simulation.bindings import (
        SimulationRigidObjectBinding,
        SimulationSceneBinding,
    )

    env = _uninitialized_env(_DeclarativeEnv, object())
    env.cfg.events = None
    env.cfg.observations = None
    env.cfg.rewards = None
    env.cfg.dataset = None
    env.episode_context = EpisodeContext(2)
    env.episode_context._begin_episode([0, 1])
    env.episode_context._finish_initialization([0, 1])
    target = torch.zeros(2, 7)
    target[:, 6] = 1
    position = target.clone()
    position[:, 0] = torch.tensor([0.1, 0.9])
    objects = {
        "payload": SimpleNamespace(get_local_pose=Mock(return_value=position)),
        "support": SimpleNamespace(get_local_pose=Mock(return_value=target)),
    }
    env.sim.asset_uids = objects.keys()
    env.sim.get_asset = objects.get
    env.sim.simulation_time = 0.3
    env.sim.reset_objects_state = Mock()
    env.sim.render_frame = lambda **kwargs: nullcontext()
    env.sensors = {}
    env._detached_uids_for_reset = []
    env._profiler = SimpleNamespace(section=lambda name, **kwargs: nullcontext())
    env._elapsed_steps = torch.zeros(2, dtype=torch.long)
    env._demo_active_segment_ids = torch.zeros(2, dtype=torch.long)
    env._demo_active_mask = torch.ones(2, dtype=torch.bool)
    env._demo_segment_participants = torch.zeros(2, dtype=torch.bool)
    env._demo_steps = torch.zeros(2, dtype=torch.long)
    env.rollout_steps = torch.zeros(2, dtype=torch.long)
    env._recording_provenance = {}
    env._demo_episode_metadata = [{}, {}]
    env.rollout_buffer = None
    env.episode_success_status = torch.zeros(2, dtype=torch.bool)
    env._normalize_demo_action = None
    env._mask_demo_action = None
    env._seed_recording_state = lambda *args: None
    env._record_initial_observation = lambda *args: None
    env.get_obs = lambda **kwargs: {}
    env.get_info = lambda **kwargs: {}
    env.compiled_program = object()
    env.bridge = _CompletedDemoBridge(
        DemoSegment(actions=(torch.zeros(2, 1),), name="move")
    )
    registration = SimpleNamespace(
        scene_binding=SimulationSceneBinding(
            registry_id="test_scene",
            rigid_objects=tuple(
                SimulationRigidObjectBinding(entity_id=uid, simulation_uid=uid)
                for uid in objects
            ),
        ),
        evaluation_conditions={
            "near_support": builtin_predicates().compile(
                EvaluationSpec(
                    {
                        "near_support": PredicateSpec(
                            "near_target",
                            {
                                "position": "rigid_objects/payload/pose",
                                "target": "rigid_objects/support/pose",
                            },
                            {"tolerance": 0.5},
                        )
                    },
                    ("near_support",),
                )
            )
        },
        evaluation_final_refs=("near_support",),
    )
    env._final_acceptance_rules = _deployment_final_rules(registration)
    env._evaluation_recording = _evaluation_observation_cfgs(registration)
    return env, position, objects["payload"].get_local_pose


def test_live_final_acceptance_revokes_after_movement_but_reset_saves_the_terminal_result(
    tmp_path,
):
    import json

    from embodichain.lab.gym.envs.demo import execute_demo_episode

    env, position, read_pose = _final_acceptance_env()
    replacement = object()
    assert env.is_task_success().tolist() == [False, False]
    read_pose.assert_not_called()

    def step(action):
        env._demo_steps += 1
        return (
            {},
            torch.zeros(2),
            torch.zeros(2, dtype=torch.bool),
            torch.zeros(2, dtype=torch.bool),
            {},
        )

    env.step = step
    result = execute_demo_episode(env, task_program=replacement)
    assert env.compiled_input is replacement
    assert env.bridge.program_completed
    assert result.success == (True, False)
    terminal = [env.get_demo_episode_metadata(row) for row in (0, 1)]
    assert terminal[0]["final_acceptance"]["conditions"]["near_support"]["metrics"][
        "near_support.distance"
    ] == pytest.approx(0.1)
    assert terminal[1]["final_acceptance"]["program_accepted"] is True
    assert terminal[1]["final_acceptance"]["physical_satisfied"] is False
    assert terminal[1]["final_acceptance"]["accepted"] is False
    report = terminal[1]["episode_outcome"]
    assert report["execution"]["program_accepted"] is True
    assert report["physical"]["final_acceptance"]["physical_satisfied"] is False
    assert report["acceptance"]["accepted"] is False
    assert report["persistence"] is None
    assert read_pose.call_count == 1

    position[:, 0] = torch.tensor([0.9, 0.1])
    assert env.is_task_success().tolist() == [False, True]
    position[1, 0] = float("nan")
    assert env.is_task_success().tolist() == [False, False]
    position[1, 0] = 0.1
    assert [env.get_demo_episode_metadata(row) for row in (0, 1)] == terminal
    assert read_pose.call_count == 3

    # The ordinary dataset selection ignores both the new live result and
    # a sticky Gym success flag when a deployment terminal result exists.
    env.episode_success_status[:] = True
    artifact = tmp_path / "saved_episodes.json"

    def save(*, mode, env_ids):
        assert mode == "save"
        artifact.write_text(
            json.dumps([env.get_demo_episode_metadata(row) for row in env_ids.tolist()])
        )

    env.dataset_manager = SimpleNamespace(
        available_modes=["save"], save_failed_episodes=False, apply=save
    )
    _, info = env.reset(options={"reset_ids": [0]})
    assert info["reset_episode_outcomes"][0]["persistence"] == {
        "dataset_selected": True,
        "dataset": [],
        "trajectory": None,
    }
    assert env.episode_context.episode_ids == (1, 0)
    assert "final_acceptance" not in env.get_demo_episode_metadata(0)
    assert env.get_demo_episode_metadata(1) == terminal[1]

    _, info = env.reset(options={"reset_ids": [1]})
    assert info["reset_episode_outcomes"][1]["persistence"]["dataset_selected"] is False
    assert read_pose.call_count == 3
    assert json.loads(artifact.read_text()) == [terminal[0]]
    assert env.episode_context.episode_ids == (1, 1)


def test_gym_terminal_success_cannot_bypass_incomplete_final_gated_program():
    from embodichain.lab.gym.envs.demo import execute_demo_episode

    env, _, read_pose = _final_acceptance_env()

    def step(action):
        return (
            {},
            torch.zeros(2),
            torch.ones(2, dtype=torch.bool),
            torch.zeros(2, dtype=torch.bool),
            {"success": torch.ones(2, dtype=torch.bool)},
        )

    env.step = step
    result = execute_demo_episode(env)
    assert result.success == (False, False)
    assert not env.bridge.program_completed
    assert (
        env.get_demo_episode_metadata(0)["final_acceptance"]["conditions"][
            "near_support"
        ]["status"]
        == "not_run"
    )
    read_pose.assert_not_called()


def test_final_roots_reuse_one_predicate_measurement_per_frame():
    from unittest.mock import Mock

    from embodichain.lab.evaluation import (
        BooleanRule,
        EvaluationSpec,
        PredicateDefinition,
        PredicateMeasurement,
        PredicateSpec,
        builtin_predicates,
    )
    from embodichain.lab.gym.envs.task_program._final_acceptance import (
        _measure_final_conditions,
    )

    env, pose, acquisition = _final_acceptance_env()

    def measure(values, params):
        distance = values["position"][:, 0]
        return PredicateMeasurement(
            distance < params["limit"], torch.isfinite(distance), {"distance": distance}
        )

    measured = Mock(side_effect=measure)
    registry = builtin_predicates()
    registry.register("counted", PredicateDefinition(1, ("position",), dict, measured))
    binding = {"position": "rigid_objects/payload/pose"}
    graph = registry.compile(
        EvaluationSpec(
            {
                "near": PredicateSpec("counted", binding, {"limit": 0.5}),
                "alias": PredicateSpec("counted", binding, {"limit": 0.5}),
                "tighter": PredicateSpec("counted", binding, {"limit": 0.05}),
                "first": BooleanRule("all", ("near", "alias")),
                "second": BooleanRule("any", ("near", "tighter")),
            },
            ("first", "second"),
        )
    )
    rules = tuple((root, graph.select(root)) for root in graph.roots)
    result = _measure_final_conditions(env, rules)
    assert result.satisfied.tolist() == [True, False]
    assert measured.call_count == 2
    acquisition.assert_called_once()

    pose[0, 0] = 0.9
    assert _measure_final_conditions(env, rules).satisfied.tolist() == [False, False]
    assert measured.call_count == 4
    assert result.satisfied.tolist() == [True, False]


def test_reset_seals_manually_completed_program_before_persistence_retry():
    from unittest.mock import Mock

    env, position, read_pose = _final_acceptance_env()
    env._begin_demo_episode_recording()
    tuple(env.create_demo_segments())
    env._demo_steps[:] = 1
    save = Mock(side_effect=OSError("unavailable storage"))
    env.dataset_manager = SimpleNamespace(
        available_modes=["save"], save_failed_episodes=False, apply=save
    )

    with pytest.raises(OSError, match="unavailable storage"):
        env.reset(options={"reset_ids": [0]})
    terminal = env.get_demo_episode_metadata(0)
    assert terminal["success"] is True
    assert terminal["length"] == 1
    assert terminal["final_acceptance"]["accepted"] is True
    assert read_pose.call_count == 1

    position[0, 0] = 0.9
    assert env.is_task_success().tolist() == [False, False]
    recorded = []

    def record(*, mode, env_ids):
        assert mode == "save"
        recorded.append(env.get_demo_episode_metadata(env_ids.item()))

    save.side_effect = record
    env.reset(options={"reset_ids": [0]})
    assert recorded == [terminal]
    assert read_pose.call_count == 2


def test_legacy_completion_is_sealed_once_and_subset_execution_keeps_peer():
    from unittest.mock import Mock

    from embodichain.lab.gym.envs.demo import execute_demo_episode

    env, _, _ = _final_acceptance_env()
    env.cfg.task_program = None
    env._final_acceptance_rules = ()
    env._mask_demo_action = lambda action, active: action
    env.is_task_success = Mock(return_value=torch.tensor([True, False]))
    validate = Mock(return_value=torch.tensor([True, False]))

    def step(action):
        env._demo_steps += env._demo_active_mask
        flags = torch.zeros(2, dtype=torch.bool)
        return {}, torch.zeros(2), flags, flags, {}

    env.step = step
    segments = (
        DemoSegment(
            actions=(torch.zeros(2, 1),),
            validator=validate,
            failure_policy="row_independent",
        ),
    )
    execute_demo_episode(env, segments=segments)
    first = env.get_demo_episode_metadata(0)["episode_outcome"]
    failed = env.get_demo_episode_metadata(1)["episode_outcome"]
    assert first["execution"]["source"] == "legacy_expert"
    assert first["execution"]["segments"][0]["validator_accepted"] is True
    assert failed["execution"]["segments"][0]["validator_accepted"] is False
    assert failed["acceptance"]["reason"] == "segment_validation_failed"
    assert first["physical"]["final_acceptance"] is None
    env.get_info = lambda **kwargs: EmbodiedEnv.get_info(env, **kwargs)
    for _ in range(2):
        env.compute_task_state()
        env.get_info()
        env.evaluate()
        assert env.get_demo_episode_metadata(0)["episode_outcome"] == first
    validate.assert_called_once_with()
    env.is_task_success.assert_called_once_with()

    env.reset(options={"reset_ids": [1], "save_data": False})
    execute_demo_episode(env, segments=segments, active_env_ids=[1])
    assert env.get_demo_episode_metadata(0)["episode_outcome"] == first
    retry = env.get_demo_episode_metadata(1)["episode_outcome"]
    assert retry["identity"]["episode"] == 1
    assert retry["identity"]["episode_uuid"] != failed["identity"]["episode_uuid"]
    _, info = env.reset(options={"reset_ids": [0, 1], "save_data": False})
    assert info["reset_episode_outcomes"][0]["acceptance"]["accepted"] is True
    assert validate.call_count == 2
    env.is_task_success.assert_called_once_with()


def test_demo_execution_error_seals_abort_without_replacing_original_error():
    from unittest.mock import Mock

    from embodichain.lab.gym.envs.demo import execute_demo_episode

    env, _, _ = _final_acceptance_env()
    env.step = Mock(side_effect=ValueError("controller failure"))
    with pytest.raises(RuntimeError, match="action generation") as exc_info:
        execute_demo_episode(env)
    assert str(exc_info.value.__cause__) == "controller failure"
    result = env.get_demo_episode_metadata(0)["episode_outcome"]
    assert result["termination"]["reason"] == "execution_error"
    assert result["acceptance"]["accepted"] is False
    assert (
        result["physical"]["final_acceptance"]["conditions"]["near_support"]["status"]
        == "not_run"
    )


def test_demo_error_cannot_seal_an_early_gym_success_or_save_it():
    from unittest.mock import Mock

    from embodichain.lab.gym.envs.demo import execute_demo_episode

    env, _, read_pose = _final_acceptance_env()
    action = torch.zeros(2, 1)
    env.bridge = _CompletedDemoBridge(
        DemoSegment(
            actions=(action, action), name="move", failure_policy="row_independent"
        )
    )
    env._mask_demo_action = lambda action, active: action
    first_row_done = torch.tensor([True, False])
    env.step = Mock(
        side_effect=[
            (
                {},
                torch.zeros(2),
                first_row_done,
                torch.zeros(2, dtype=torch.bool),
                {"success": first_row_done},
            ),
            ValueError("peer controller failure"),
        ]
    )
    save = Mock()
    env.dataset_manager = SimpleNamespace(
        available_modes=["save"], save_failed_episodes=False, apply=save
    )

    with pytest.raises(RuntimeError, match="action generation") as error:
        execute_demo_episode(env)

    assert str(error.value.__cause__) == "peer controller failure"
    first = env.get_demo_episode_metadata(0)
    assert first["terminated"]
    assert not first["success"] and not first["completed"]
    assert not first["final_acceptance"]["accepted"]
    assert not first["final_acceptance"]["program_accepted"]
    assert first["final_acceptance"]["physical_satisfied"] is None
    assert first["episode_outcome"]["acceptance"] == {
        "accepted": False,
        "eligible": False,
        "reason": "execution_error",
    }
    _, info = env.reset()
    assert not info["reset_episode_outcomes"][0]["persistence"]["dataset_selected"]
    save.assert_not_called()
    read_pose.assert_not_called()


def test_explicit_segments_cannot_reuse_completed_program_in_same_episode():
    from unittest.mock import Mock

    from embodichain.lab.gym.envs.demo import execute_demo_episode

    env, position, read_pose = _final_acceptance_env()
    action = torch.zeros(2, 1)
    flags = torch.zeros(2, dtype=torch.bool)
    env.step = Mock(return_value=({}, torch.zeros(2), flags, flags, {}))
    assert execute_demo_episode(env).success == (True, False)
    terminal = [env.get_demo_episode_metadata(row) for row in (0, 1)]

    position[:, 0] = 0.1
    with pytest.raises(RuntimeError, match="reset"):
        execute_demo_episode(
            env, segments=(DemoSegment(actions=(action,)),), attempt_id=1
        )
    assert [env.get_demo_episode_metadata(row) for row in (0, 1)] == terminal
    env.step.assert_called_once()
    read_pose.assert_called_once()

    env.reset(options={"save_data": False})
    result = execute_demo_episode(
        env, segments=(DemoSegment(actions=(action,)),), attempt_id=1
    )
    assert result.success == (False, False)
    assert not env.get_demo_episode_metadata(0)["final_acceptance"]["program_accepted"]


def test_partial_reset_keeps_manual_completion_for_unreset_rows():
    env, position, read_pose = _final_acceptance_env()
    env.cfg.record_trajectory = False
    env.cfg.actions = None
    env.sim.has_native_renderer = True
    env.observation_manager = None
    env._init_sim_state()
    assert not env._capture_observation().fields
    read_pose.assert_not_called()
    position[:, 0] = 0.1
    env._begin_demo_episode_recording()
    tuple(env.create_demo_segments())
    env._demo_steps[:] = 1
    saved = []
    env.dataset_manager = SimpleNamespace(
        available_modes=["save"],
        save_failed_episodes=False,
        apply=lambda *, mode, env_ids: saved.extend(env_ids.tolist()),
    )
    assert env.is_task_success().tolist() == [True, True]
    env.reset(options={"reset_ids": [0]})
    assert env.episode_context.episode_ids == (1, 0)
    assert env.is_task_success().tolist() == [False, True]
    from copy import deepcopy
    from embodichain.lab.gym.envs.demo import execute_demo_episode

    peer = deepcopy(env.get_demo_episode_metadata(1))
    env.bridge = _CompletedDemoBridge(
        DemoSegment(actions=(torch.zeros(2, 1),), name="retry")
    )
    env.bridge.completion_mask = torch.tensor([True, False])
    env._mask_demo_action = lambda action, active: action

    def step(action):
        assert env.is_task_success().tolist() == [False, True]
        env._demo_steps += env._demo_active_mask
        flags = torch.zeros(2, dtype=torch.bool)
        return {}, torch.zeros(2), flags, flags, {}

    env.step = step
    result = execute_demo_episode(env, active_env_ids=[0])
    assert result.success[0]
    assert env.is_task_success().tolist() == [True, True]
    assert env.get_demo_episode_metadata(1) == peer
    env.reset(options={"reset_ids": [1]})
    assert saved == [0, 1]


def test_trajectory_replay_recomputes_the_sealed_capture_after_physics_changes(
    tmp_path,
):
    from tensordict import TensorDict

    from embodichain.lab.gym.envs.demo import execute_demo_episode
    from embodichain.lab.gym.envs.wrapper import ReplayWrapper

    env, position, read_pose = _final_acceptance_env()
    env.cfg.sim_steps_per_control = 4
    env.sim_cfg = SimpleNamespace(physics_dt=0.01)
    env.robot = SimpleNamespace(uid="arm", dof=1)
    env.active_joint_ids = [0]
    env.max_episode_steps = 5
    env.sim._articulations = {}
    env.sim._rigid_objects = {"payload": None, "support": None}
    env._trajectory_action_metadata = lambda: {"action_kind": "policy"}

    def step(action):
        env._demo_steps += 1
        return (
            {},
            torch.zeros(2),
            torch.zeros(2, dtype=torch.bool),
            torch.zeros(2, dtype=torch.bool),
            {},
        )

    env.step = step
    result = execute_demo_episode(env)
    assert result.success == (True, False)
    sealed = [env.get_demo_episode_metadata(i)["final_acceptance"] for i in (1, 0)]

    # Save after a later acquisition at the same simulation timestamp. The
    # source capture still has the original revision and row-local poses.
    position[:, 0] = torch.tensor([0.9, 0.1])
    assert env.is_task_success().tolist() == [False, True]
    env._traj_steps = torch.ones(2, dtype=torch.long)
    env._traj_buffer = TensorDict(
        {
            "states": {"rigid_objects": {"payload": {"pose": position[:, None, :]}}},
            "actions": torch.zeros(2, 1, 1),
        },
        batch_size=[2, 1],
    )
    path = env.save_trajectory(str(tmp_path / "episode.pt"), env_ids=[1, 0])
    replay = ReplayWrapper(env, path, mode="kinematic")
    for report, expected in zip(replay.final_acceptance, sealed, strict=True):
        assert report == {
            **expected,
            "status": "recomputed",
            "recorded_accepted": expected["accepted"],
        }
    assert read_pose.call_count == 2

    payload = torch.load(path, weights_only=False)
    first = payload["meta"]["demo_episodes"][0]["final_acceptance"]
    first["accepted"] = True
    first["physical_satisfied"] = True
    first["conditions"]["near_support"]["metrics"]["near_support.distance"] = 0.0
    report = ReplayWrapper(env, payload, mode="kinematic").final_acceptance[0]
    assert report["recorded_accepted"] is True
    assert report["accepted"] is False
    assert report["conditions"]["near_support"]["metrics"][
        "near_support.distance"
    ] == pytest.approx(0.9)

    second = payload["meta"]["demo_episodes"][1]["final_acceptance"]
    second["execution_eligible"] = False
    report = ReplayWrapper(env, payload, mode="kinematic").final_acceptance[1]
    assert report["physical_satisfied"] is True
    assert report["accepted"] is False

    first["observation_revision"] += 1
    unavailable, peer = ReplayWrapper(env, payload, mode="kinematic").final_acceptance
    assert unavailable["status"] == "unavailable"
    assert unavailable["accepted"] is None
    assert peer["status"] == "recomputed"

    payload["final_acceptance_observations"]["rows"][1]["fields"][
        "rigid_objects/payload/pose"
    ]["descriptor"]["units"] = ("pixels", "unitless")
    report = ReplayWrapper(env, payload, mode="kinematic").final_acceptance[1]
    assert report["status"] == "unavailable"
    assert report["accepted"] is None
    assert report["conditions"]["near_support"]["status"] == "unavailable"

    field = payload["final_acceptance_observations"]["rows"][1]["fields"][
        "rigid_objects/payload/pose"
    ]
    field["descriptor"]["units"] = ("m", "unitless")
    field["values"] = field["values"].long()
    field["descriptor"]["dtype"] = "torch.int64"
    report = ReplayWrapper(env, payload, mode="kinematic").final_acceptance[1]
    assert report["status"] == "unavailable"
    assert report["accepted"] is None


@pytest.mark.gpu
def test_cuda_final_capture_and_cpu_replay_agree_at_inclusive_boundary():
    from embodichain.lab.evaluation import (
        EvaluationSpec,
        PredicateSpec,
        RuleSession,
        builtin_predicates,
        evaluate_recorded_rules,
    )
    from embodichain.lab.evaluation.observations import advance_observation
    from embodichain.lab.gym.envs.episode_context import EpisodeContext
    from embodichain.lab.gym.envs.managers.observation_types import (
        ObservationDescriptor,
        ObservationFrame,
        ObservationSample,
    )
    from embodichain.lab.gym.envs.task_program._final_acceptance import (
        _evaluate_final_frame,
        _export_final_observations,
        _final_acceptance_rows,
        _replay_final_acceptance,
    )

    position, target = "rigid_objects/a/pose", "rigid_objects/b/pose"
    graph = builtin_predicates().compile(
        EvaluationSpec(
            {
                "near": PredicateSpec(
                    "near_target",
                    {"position": position, "target": target},
                    {"axes": (0, 1), "tolerance": 0.7833548188209534},
                )
            },
            ("near",),
        )
    )
    fields = {}
    for field, coordinates in (
        (position, (0.11330324411392212, 0.7751175165176392, 0)),
        (target, (0, 0, 0)),
    ):
        pose = torch.tensor(
            [(*coordinates, 0, 0, 0, 1)], device="cuda", dtype=torch.float32
        )
        descriptor = ObservationDescriptor(
            field.split("/")[1],
            "pose",
            "arena",
            "xyz_xyzw",
            ("m", "unitless"),
            (7,),
            pose.dtype,
        )
        fields[field] = ObservationSample(
            descriptor, pose, torch.tensor([True], device="cuda"), (None,)
        )
    owner = EpisodeContext(1, device="cuda")
    owner._begin_episode([0])
    frame = ObservationFrame(None, fields, (0,), 4, 0.2, episode_owner=owner)
    rules = (("near", graph),)
    measured = _evaluate_final_frame(rules, frame)
    assert measured.satisfied.device.type == "cuda"
    seal = _final_acceptance_rows(
        rules,
        (0,),
        tuple(measured.satisfied.tolist()),
        (True,),
        torch.tensor([True]),
        measured,
    )[0]
    episode = {"env_id": 0, "episode_uuid": "ep", "final_acceptance": seal}
    payload = {
        "meta": {"num_envs": 1, "env_ids": [0], "demo_episodes": [episode]},
        "final_acceptance_observations": _export_final_observations(
            rules, {0: measured.frame}, [episode]
        ),
    }
    assert seal["accepted"] is True
    assert _replay_final_acceptance(rules, payload) == (
        {"status": "recomputed", "recorded_accepted": True, **seal},
    )
    live = RuleSession(
        graph, env_ids=(0,), episode_ids=(0,), scope="segment", device="cuda"
    )
    advance_observation(live, frame, step=0, episode_owner=owner)
    raw = payload["final_acceptance_observations"]["rows"][0]["fields"]
    recording = {
        "schema_version": 1,
        "env_ids": (0,),
        "episode_ids": (0,),
        "revisions": torch.tensor([[frame.revision]]),
        "lengths": torch.tensor([0]),
        "fields": {
            name: {
                **sample,
                "values": sample["values"][:, None],
                "valid": sample["valid"][:, None],
            }
            for name, sample in raw.items()
        },
    }
    replay = evaluate_recorded_rules(
        graph,
        {"raw_observations": recording},
        env_ids=(0,),
        episode_ids=(0,),
        scope="segment",
    )
    live_result, offline_result = live.snapshot().rules["near"], replay.rules["near"]
    torch.testing.assert_close(live_result.status.cpu(), offline_result.status)
    torch.testing.assert_close(
        live_result.metrics["distance"].cpu(),
        offline_result.metrics["distance"],
        rtol=0,
        atol=0,
    )
