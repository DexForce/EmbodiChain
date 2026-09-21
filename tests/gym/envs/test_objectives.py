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

import json
from types import SimpleNamespace

import pytest
import torch
import yaml

from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.gym_utils import config_to_cfg

pytestmark = pytest.mark.no_sim


def _declaration():
    return {
        "type": "ordered_stable_regions",
        "object_uid": "cube",
        "regions": [
            {"name": name, "lower": [x - 0.1, -0.1, -0.1], "upper": [x + 0.1, 0.1, 0.1]}
            for name, x in [("A", 0.0), ("B", 1.0), ("A", 0.0)]
        ],
        "hold_seconds": 0.2,
        "max_linear_speed": 0.05,
        "max_angular_speed": 0.1,
    }


def _runtime(declaration=None):
    from embodichain.lab.gym.envs.objectives import OrderedPlacementObjective
    from embodichain.lab.gym.envs.objectives.config import decode_objective

    return OrderedPlacementObjective(
        decode_objective(declaration or _declaration()), 2, "cpu"
    )


def _state(xs=(0.0, 0.0), linear=0.0, angular=0.0):
    from embodichain.lab.gym.envs.objectives import MeasuredRigidObjectState

    position = torch.zeros(2, 3)
    position[:, 0] = torch.tensor(xs)
    lin = torch.zeros_like(position)
    ang = torch.zeros_like(position)
    lin[:, 0] = linear
    ang[:, 0] = angular
    return MeasuredRigidObjectState(position, lin, ang)


def test_ordered_dwell_transient_final_revocation_and_partial_reset():
    objective = _runtime()
    objective.update(_state(), 0.1)
    objective.update(_state((1.0, 0.0)), 0.1)
    assert objective.snapshot()["progress"].tolist() == [0, 1]
    for xs in [(0.0, 1.0), (0.0, 1.0), (1.0, 0.0), (1.0, 0.0), (0.0, 0.0), (0.0, 0.0)]:
        objective.update(_state(xs), 0.1)
    terminal = objective.snapshot()
    assert terminal["progress"].tolist() == [3, 3]
    assert terminal["success"].tolist() == [True, True]
    objective.update(_state((1.0, 0.0)), 0.1)
    assert objective.snapshot()["success"].tolist() == [False, True]
    assert objective.snapshot()["progress"].tolist() == [3, 3]
    objective.reset([0])
    assert objective.snapshot()["progress"].tolist() == [0, 3]
    assert terminal["success"].tolist() == [True, True]


def test_snapshot_queries_are_pure_detached_and_json_compatible():
    objective = _runtime()
    objective.update(_state(), 0.1)
    first = objective.snapshot()
    second = objective.snapshot()
    assert first["progress"].tolist() == second["progress"].tolist() == [0, 0]
    first["metrics"]["hold_seconds"].zero_()
    assert objective.snapshot()["metrics"]["hold_seconds"].tolist() == pytest.approx(
        [0.1, 0.1]
    )
    row = objective.snapshot_row(0)
    assert row["progress"] == 0
    assert row["predicate"] == "ordered_stable_regions"
    json.dumps(row, allow_nan=False)


@pytest.mark.parametrize(
    "linear, angular", [(0.051, 0.0), (0.0, 0.101), (float("nan"), 0.0)]
)
def test_speed_and_nonfinite_measurements_break_dwell(linear, angular):
    objective = _runtime()
    objective.update(_state(), 0.1)
    objective.update(_state(linear=linear, angular=angular), 0.1)
    objective.update(_state(), 0.1)
    assert objective.snapshot()["progress"].tolist() == [0, 0]
    objective.update(_state(xs=(float("inf"), 0.0)), 0.1)
    assert objective.snapshot()["success"].tolist() == [False, False]
    json.dumps(objective.snapshot_row(0), allow_nan=False)


def test_overlapping_regions_advance_only_once_per_control_step():
    declaration = _declaration()
    declaration["regions"] = [declaration["regions"][0]] * 3
    objective = _runtime(declaration)
    objective.update(_state(), 1.0)
    assert objective.snapshot()["progress"].tolist() == [1, 1]


@pytest.mark.parametrize(
    "patch",
    [
        {"unexpected": True},
        {"type": "release"},
        {"object_uid": ""},
        {"regions": []},
        {"hold_seconds": 0},
        {"hold_seconds": True},
        {"max_linear_speed": float("inf")},
        {"max_angular_speed": -1},
        {"regions": [{"name": "A", "lower": [0, 0], "upper": [1, 1, 1]}]},
        {"regions": [{"name": "A", "lower": [1, 0, 0], "upper": [0, 1, 1]}]},
        {
            "regions": [
                {"name": "A", "lower": [0, 0, 0], "upper": [1, 1, 1], "release": True}
            ]
        },
    ],
)
def test_strict_objective_validation(patch):
    from embodichain.lab.gym.envs.objectives.config import decode_objective

    declaration = _declaration()
    declaration.update(patch)
    with pytest.raises((TypeError, ValueError)):
        decode_objective(declaration)


def _gym_config():
    return {
        "id": "ObjectiveTest-v0",
        "physics": "default",
        "env": {},
        "robot": {"uid": "robot", "robot_type": "URRobot"},
    }


def test_objective_component_decodes_relative_to_deployment(tmp_path):
    component = tmp_path / "objective.yaml"
    component.write_text(yaml.safe_dump(_declaration()))
    config = _gym_config()
    config["objective"] = {"component": "objective.yaml"}
    cfg = config_to_cfg(config, source_path=tmp_path / "task.yaml")
    assert cfg.objective.object_uid == "cube"
    assert cfg.objective.regions[1].name == "B"


@pytest.mark.parametrize(
    "selection",
    [
        None,
        "objective.yaml",
        {},
        {"component": "objective.yaml", "overrides": {}},
        {"component": ""},
    ],
)
def test_objective_selection_is_strict(selection, tmp_path):
    config = _gym_config()
    config["objective"] = selection
    with pytest.raises((TypeError, ValueError)):
        config_to_cfg(config, source_path=tmp_path / "task.yaml")


def test_absent_objective_preserves_info_keys():
    env = object.__new__(EmbodiedEnv)
    env._num_envs = 2
    env.sim = SimpleNamespace(device=torch.device("cpu"))
    env._elapsed_steps = torch.zeros(2, dtype=torch.int32)
    assert EmbodiedEnvCfg().objective is None
    assert set(env.get_info()) == {"success", "fail", "elapsed_steps", "metrics"}


def test_binding_rejects_missing_physical_uid_before_rollout():
    env = object.__new__(EmbodiedEnv)
    env._num_envs = 2
    objective = _runtime()
    env.cfg = SimpleNamespace(objective=objective.cfg)
    env.sim = SimpleNamespace(
        device=torch.device("cpu"), get_rigid_object=lambda uid: None
    )
    with pytest.raises(ValueError, match="cube"):
        env._initialize_physical_objective()


def test_binding_rejects_static_target_without_velocity_measurements():
    from embodichain.lab.sim.objects.rigid_object import RigidObject

    target = object.__new__(RigidObject)
    target.body_type = "static"
    env = object.__new__(EmbodiedEnv)
    env._num_envs = 2
    env.cfg = SimpleNamespace(objective=_runtime().cfg)
    env.sim = SimpleNamespace(
        device=torch.device("cpu"), get_rigid_object=lambda uid: target
    )
    with pytest.raises(ValueError, match="cube.*measured rigid body data"):
        env._initialize_physical_objective()
    assert env.physical_objective is None
    assert env._physical_objective_object is None


class _Profiler:
    def section(self, *args, **kwargs):
        from contextlib import nullcontext

        return nullcontext()


class _LifecycleEnv(EmbodiedEnv):
    """Real Gym loop with only physics and recorder boundaries replaced."""

    def __init__(self):
        self._num_envs = 2
        self._profiler = _Profiler()
        self._elapsed_steps = torch.zeros(2, dtype=torch.int32)
        self._task_success = torch.zeros(2, dtype=torch.bool)
        self.episode_success_status = torch.zeros(2, dtype=torch.bool)
        self._detached_uids_for_reset = []
        self.max_episode_steps = 100
        self.cfg = SimpleNamespace(
            objective=_runtime().cfg,
            sim_steps_per_control=1,
            ignore_terminations=False,
            events=True,
            observations=None,
            rewards=None,
            dataset=None,
            seed=None,
        )
        self.sim_cfg = SimpleNamespace(physics_dt=0.1)
        self.position = torch.zeros(2, 7)
        self.body = SimpleNamespace(
            lin_vel=torch.zeros(2, 3), ang_vel=torch.zeros(2, 3)
        )
        self.object = SimpleNamespace(
            get_local_pose=lambda: self.position, body_data=self.body
        )
        self.sim = SimpleNamespace(
            device=torch.device("cpu"),
            get_rigid_object=lambda uid: self.object,
            update=lambda *args: None,
            reset_objects_state=lambda **kwargs: None,
            sync_render_state=lambda: None,
            capture_visualization_safely=lambda **kwargs: None,
        )
        self.event_manager = SimpleNamespace(
            available_modes=["interval", "reset"],
            _mode_functor_cfgs={},
            apply=self._apply_event,
            reset=lambda **kwargs: {},
        )
        self.interval_x = 0.0
        self.reset_snapshot = None
        self.dataset_manager = None
        self.rollout_buffer = None
        self._traj_buffer = None
        self._demo_steps = torch.zeros(2, dtype=torch.long)
        self._initialize_physical_objective()

    def _apply_event(self, mode, **kwargs):
        if mode == "interval":
            self.position[:, 0] = self.interval_x
        else:
            self.reset_snapshot = self.physical_objective.snapshot()

    def _preprocess_action(self, action):
        return action

    def _step_action(self, action):
        return action

    def _postprocess_action(self, action):
        return action

    def get_obs(self, **kwargs):
        return {}

    def get_reward(self, **kwargs):
        return torch.zeros(2)

    def _extend_reward(self, rewards, **kwargs):
        return rewards

    def check_truncated(self, **kwargs):
        return torch.zeros(2, dtype=torch.bool)

    def _hook_after_sim_step(self, **kwargs):
        pass

    def is_task_success(self, **kwargs):
        return torch.zeros(2, dtype=torch.bool)


def test_control_step_samples_after_interval_events_and_info_is_pure():
    env = _LifecycleEnv()
    env.position[:, 0] = 1.0
    env.step(None)
    info = env.get_info()
    assert info["physical_objective"]["progress"].tolist() == [0, 0]
    assert env.get_info()["physical_objective"]["metrics"][
        "hold_seconds"
    ].tolist() == pytest.approx([0.1, 0.1])
    _, _, terminated, _, info = env.step(None)
    assert info["physical_objective"]["progress"].tolist() == [1, 1]
    assert not terminated.any()
    assert not info["success"].any()


def test_reset_preserves_terminal_snapshot_and_resets_after_recorder_consumption():
    env = _LifecycleEnv()
    for x in [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]:
        env.interval_x = x
        env.step(None)
    terminal = env.get_info()
    saved = []
    env.dataset_manager = SimpleNamespace(
        available_modes=["save"],
        save_failed_episodes=True,
        apply=lambda **kwargs: saved.append(env.physical_objective.snapshot_row(0)),
    )
    _, info = env.reset(options={"reset_ids": torch.tensor([0])})
    assert saved[0]["success"] is True
    assert env.reset_snapshot["success"].tolist() == [True, True]
    assert info["physical_objective"]["progress"].tolist() == [0, 3]
    assert terminal["physical_objective"]["success"].tolist() == [True, True]


def test_demo_metadata_adds_separate_physical_namespace():
    env = _LifecycleEnv()
    env._demo_episode_metadata = [{"segments": [], "accepted": False}] * 2
    metadata = env.get_demo_episode_metadata(0)
    assert metadata["accepted"] is False
    assert metadata["physical_objective"]["goal_count"] == 3
    json.dumps(metadata, allow_nan=False)


def test_dwell_threshold_does_not_require_extra_step_from_roundoff():
    objective = _runtime()
    for _ in range(10):
        objective.update(_state(), 0.02)
    assert objective.snapshot()["progress"].tolist() == [1, 1]


def test_final_reentry_requires_fresh_stable_dwell():
    objective = _runtime()
    for xs in [(0.0, 0.0), (1.0, 1.0), (0.0, 0.0)]:
        objective.update(_state(xs), 0.2)
    objective.update(_state((1.0, 1.0)), 0.1)
    objective.update(_state(), 0.1)
    assert objective.snapshot()["progress"].tolist() == [3, 3]
    assert objective.snapshot()["success"].tolist() == [False, False]
    objective.update(_state(), 0.1)
    assert objective.snapshot()["success"].tolist() == [True, True]


def test_inclusive_bounds_and_speed_limits_use_measurement_precision():
    objective = _runtime()
    objective.update(_state((0.1, -0.1), linear=0.05, angular=0.1), 0.2)
    assert objective.snapshot()["progress"].tolist() == [1, 1]
