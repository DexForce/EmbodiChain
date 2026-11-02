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

"""Configured RL contracts, tested without native simulation."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import yaml

from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.envs.episode_context import EpisodeContext
from embodichain.lab.gym.envs.managers import ConditionCfg, RLTaskCfg, SceneEntityCfg
from embodichain.lab.gym.envs.managers.conditions import (
    object_near_target,
    object_below_height,
)
from embodichain.lab.gym.envs.managers.manager_base import Functor
from embodichain.lab.gym.envs.managers.termination_manager import TerminationManager
from embodichain.lab.gym.envs.objectives import (
    ConditionResult,
    evaluate_near_target,
    evaluate_below_height,
)
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.utils.registration import get_env_spec

_ROOT = Path(__file__).resolve().parents[3]
_CONFIG = (
    _ROOT
    / "extensions/embodichain_tasks/configs/tasks/manipulation/push_cube/env.configured_rl.yaml"
)


def _task():
    return RLTaskCfg(
        success={
            "goal": ConditionCfg(
                func=object_near_target,
                params={
                    "entity_cfg": SceneEntityCfg(uid="cube"),
                    "target_pose_key": "goal_pose",
                    "axes": [0, 1],
                    "tolerance": 0.1,
                },
            )
        },
        failure={
            "fallen": ConditionCfg(
                func=object_below_height,
                params={"entity_cfg": SceneEntityCfg(uid="cube"), "height": -0.1},
            )
        },
    )


def _env():
    env = object.__new__(EmbodiedEnv)
    env._num_envs = 2
    env.cfg = SimpleNamespace(
        rl_task=_task(), task_program=None, ignore_terminations=False
    )
    env.pose = torch.eye(4).repeat(2, 1, 1)
    env.sim = SimpleNamespace(
        device=torch.device("cpu"),
        asset_uids=["cube"],
        get_rigid_object=lambda uid: (
            SimpleNamespace(get_local_pose=lambda **kwargs: env.pose)
            if uid == "cube"
            else None
        ),
    )
    env.goal_pose = torch.eye(4).repeat(2, 1, 1)
    env.termination_manager = TerminationManager(env.cfg.rl_task, env)
    env._task_initial_states = [{}, {}]
    env.episode_context = EpisodeContext(2)
    env._elapsed_steps = torch.zeros(2, dtype=torch.int32)
    env.physical_objective = None
    env.rollout_buffer = None
    env._profiler = SimpleNamespace(section=lambda *args, **kwargs: nullcontext())
    env._write_trajectory_step = lambda: None
    env.episode_success_status = torch.zeros(2, dtype=torch.bool)
    env._active_task_program_bridge = None
    return env


def test_geometry_independent_axes_boundary_height_and_invalid_rows():
    position = torch.tensor([[0.3, 0.4, 8.0], [0.0, 0.0, float("nan")]])
    target = torch.zeros(3)
    strict = evaluate_near_target(position, target, 0.5, axes=(0, 1), inclusive=False)
    assert strict.satisfied.tolist() == [False, False]
    assert strict.valid.tolist() == [True, False]
    assert strict.metrics["distance"][0].item() == pytest.approx(0.5)
    assert evaluate_near_target(
        position, target, 0.5, axes=(0, 1)
    ).satisfied.tolist() == [True, False]
    assert not evaluate_near_target(position[:1], target, 0.5).satisfied.item()
    assert evaluate_below_height(
        torch.tensor([[0.0, 0.0, -0.2], [0.0, 0.0, -0.1]]), -0.1
    ).satisfied.tolist() == [True, False]


def test_outcome_queries_partial_reset_and_invalid_failure():
    env = _env()
    assert env.is_task_success().tolist() == [False, False]
    env.initialize_task_outcome()
    env.termination_manager.compute()
    initial = env.get_info()
    assert initial["success"].tolist() == [True, True]
    for _ in range(3):
        env.get_info()
        env.evaluate()
        env.is_task_success()
    assert env.termination_manager.snapshot()["steps"].tolist() == [1, 1]
    initial["task_outcome"]["accepted"].fill_(False)
    env.termination_manager.terminated.fill_(False)
    assert env.is_task_success().all()
    assert env.termination_manager.terminated.all()
    env.initialize_task_outcome([1])
    assert env.is_task_success().tolist() == [True, False]
    assert env.termination_manager.snapshot()["steps"].tolist() == [1, 0]
    assert env.termination_manager.terminated.tolist() == [True, False]
    env.pose[1, 2, 3] = float("nan")
    env.termination_manager.compute()
    assert env.get_info()["fail"].tolist() == [False, True]
    assert env.get_info()["task_outcome"]["invalid_measurement"].tolist() == [
        False,
        True,
    ]


def test_failure_and_actual_time_limit_override_physical_success():
    env = _env()
    env.pose[1, 2, 3] = -0.2
    env.termination_manager.compute(time_out=torch.tensor([True, False]))
    assert env.is_task_success().tolist() == [False, False]
    assert env.get_info()["task_outcome"]["physical_success"].tolist() == [True, True]
    assert env.get_info()["fail"].tolist() == [False, True]


def test_empty_success_set_never_accepts():
    env = _env()
    manager = TerminationManager(RLTaskCfg(), env)
    manager.compute()
    assert manager.task_state()[0].tolist() == [False, False]


@pytest.mark.parametrize("task_program", [None, object()])
def test_unconfigured_host_keeps_legacy_and_task_program_acceptance(task_program):
    env = _env()
    env.termination_manager = None
    env.cfg = SimpleNamespace(task_program=task_program)
    env.initialize_task_outcome([0])
    info = env.get_info()
    assert "task_outcome" not in info
    assert "task_initial_state" not in info
    assert not info["success"].any()
    assert env.is_task_success().tolist() == [task_program is None] * 2
    assert not env.check_truncated({}, info).any()
    with pytest.raises(ValueError, match="rl_task"):
        env.restore_task_initial_state({})


@pytest.mark.parametrize("integration", ["program", "adapter"])
def test_rl_task_constructor_rejects_task_program_integration(integration):
    cfg = EmbodiedEnvCfg(rl_task=_task())
    kwargs = {}
    if integration == "program":
        cfg.task_program = object()
    else:
        kwargs["task_program_adapter_factory"] = object()
    with pytest.raises(ValueError, match="cannot share"):
        EmbodiedEnv(cfg, **kwargs)


def test_reset_target_is_required_and_initial_context_is_owned():
    env = _env()
    del env.goal_pose
    with pytest.raises(ValueError, match="reset target"):
        env.initialize_task_outcome()
    env.goal_pose = torch.eye(4).repeat(2, 1, 1)
    env.initialize_task_outcome()
    env.goal_pose[0, 0, 3] = 10
    assert env.get_task_initial_state()["targets"]["goal_pose"][0][3] == 0
    exported = env.get_task_initial_state()
    exported["targets"]["goal_pose"][0][3] = 99
    assert env.get_task_initial_state()["targets"]["goal_pose"][0][3] == 0


@pytest.mark.parametrize(
    "change, error, message",
    [
        ("missing_entity", ValueError, "absent"),
        ("bad_param", ValueError, "unexpected"),
        ("bad_threshold", RuntimeError, "tolerance"),
        ("bad_height", RuntimeError, "height"),
        ("bad_target_key", RuntimeError, "target_pose_key"),
    ],
)
def test_conditions_fail_during_binding(change, error, message):
    env = _env()
    task = _task()
    if change == "missing_entity":
        task.success["goal"].params["entity_cfg"].uid = "absent"
    elif change == "bad_param":
        task.success["goal"].params["unexpected"] = 1
    elif change == "bad_threshold":
        task.success["goal"].params["tolerance"] = -1
    elif change == "bad_height":
        task.failure["fallen"].params["height"] = float("nan")
    else:
        task.success["goal"].params["target_pose_key"] = " "
    with pytest.raises(error, match=message):
        TerminationManager(task, env)


def test_config_registers_once_and_rejects_changed_declarations():
    config = yaml.safe_load(_CONFIG.read_text())
    config["id"] = "ConfiguredRLDecodeTest-v1"
    cfg = config_to_cfg(config, source_path=_CONFIG)
    assert isinstance(cfg, EmbodiedEnvCfg)
    assert get_env_spec(config["id"]).cls is EmbodiedEnv
    assert get_env_spec(config["id"]).supports_rl
    runtime_change = deepcopy(config)
    runtime_change.update(num_envs=3, device="cpu", seed=11)
    config_to_cfg(runtime_change, source_path=_CONFIG)
    changed = deepcopy(config)
    changed["rl_task"]["success"]["goal"]["params"]["tolerance"] = 0.2
    with pytest.raises(ValueError, match="different declaration"):
        config_to_cfg(changed, source_path=_CONFIG)


@pytest.mark.parametrize(
    "declaration",
    [{"unknown": {}}, {"success": {"goal": {"func": "object_near_target", "typo": 1}}}],
)
def test_invalid_task_schema_fails_before_registration(declaration):
    config = yaml.safe_load(_CONFIG.read_text())
    config["rl_task"] = declaration
    with pytest.raises((ValueError, TypeError)):
        config_to_cfg(config, source_path=_CONFIG)


def test_rl_and_task_program_are_mutually_exclusive():
    config = yaml.safe_load(_CONFIG.read_text())
    config["task_program"] = {}
    with pytest.raises(ValueError, match="cannot share"):
        config_to_cfg(config, source_path=_CONFIG)


def test_task_context_restore_rejects_missing_metadata_and_restores_goal():
    env = _env()
    env._num_envs = 1
    env.goal_pose = env.goal_pose[:1]
    env.pose = env.pose[:1]
    env.episode_context = EpisodeContext(1)
    env.termination_manager = TerminationManager(env.cfg.rl_task, env)
    env.initialize_task_outcome()
    original = env.get_task_initial_state()
    env.goal_pose[:, 0, 3] = 20
    with pytest.raises(ValueError, match="task_initial_state"):
        env.restore_task_initial_state({})
    env.restore_task_initial_state(original)
    assert env.goal_pose[0, 0, 3] == 0
    assert env.termination_manager.snapshot()["steps"].tolist() == [0]


class _CountingCondition(Functor):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.counts = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def __call__(self, env):
        self.counts += 1
        return ConditionResult(
            self.counts > 0,
            torch.ones_like(self.counts, dtype=torch.bool),
            {"count": self.counts},
        )

    def reset(self, env_ids=None):
        self.counts[env_ids] = 0


def test_stateful_condition_advances_only_on_compute_and_resets_selected_rows():
    env = _env()
    manager = TerminationManager(
        RLTaskCfg(success={"count": ConditionCfg(func=_CountingCondition)}), env
    )
    manager.compute()
    old = manager.snapshot()
    manager.snapshot()
    manager.task_state()
    manager.reset([1])
    manager.compute()
    assert manager.task_state()[2]["success.count.count"].tolist() == [2, 1]
    assert old["conditions"]["success.count"]["metrics"]["count"].tolist() == [1, 1]


@pytest.mark.parametrize("manager", ["actions", "observations"])
def test_registration_preserves_ordered_policy_interface(manager):
    config = yaml.safe_load(_CONFIG.read_text())
    config["id"] = f"ConfiguredRLOrder-{manager}-v1"
    if manager == "actions":
        config["env"][manager] = {
            "first": {"func": "JointPositionAction", "params": {"joint_ids": [0]}},
            "second": {"func": "JointPositionAction", "params": {"joint_ids": [1]}},
        }
    config_to_cfg(config, source_path=_CONFIG)
    config["env"][manager] = dict(reversed(tuple(config["env"][manager].items())))
    with pytest.raises(ValueError, match="different declaration"):
        config_to_cfg(config, source_path=_CONFIG)


@pytest.mark.parametrize("ignore_terminations", [False, True])
def test_step_uses_one_condition_snapshot_for_obs_reward_flags_and_recording(
    monkeypatch, ignore_terminations
):
    env = _env()
    env.cfg.events = None
    env.cfg.ignore_terminations = ignore_terminations
    env.cfg.rl_task.success["count"] = ConditionCfg(func=_CountingCondition)
    env.termination_manager = TerminationManager(env.cfg.rl_task, env)
    env.max_episode_steps = 2
    env._elapsed_steps[1] = 1  # Only the second row expires on this step.
    env._profiler = SimpleNamespace(section=lambda *args, **kwargs: nullcontext())
    env.sim.render_frame = lambda **kwargs: nullcontext()
    env._demo_no_auto_reset = True
    env._advance_physics = lambda: None
    env._preprocess_action = lambda action: action
    env._step_action = lambda action: action
    env._postprocess_action = lambda action: action
    env.get_obs = lambda **kwargs: {"accepted": env.is_task_success()}
    env.get_reward = lambda obs, action, info: info["success"].float() * 10
    env._extend_reward = lambda rewards, **kwargs: rewards
    recorded = []
    monkeypatch.setattr(
        EmbodiedEnv,
        "_update_episode_success_status",
        lambda self, info, dones: recorded.append(info["success"].clone()),
    )
    obs, reward, terminated, truncated, info = env.step(torch.zeros(2, 1))
    assert obs["accepted"].tolist() == [True, False]
    assert reward.tolist() == [10, 0]
    assert terminated.tolist() == [not ignore_terminations, False]
    assert truncated.tolist() == [False, True]
    assert info["success"].tolist() == recorded[0].tolist() == [True, False]
    assert info["task_outcome"]["physical_success"].tolist() == [True, True]
    assert info["task_outcome"]["terminated"].tolist() == terminated.tolist()
    assert info["task_outcome"]["truncation"].tolist() == truncated.tolist()
    assert info["metrics"]["success.count.count"].tolist() == [1, 1]
    assert env.termination_manager.snapshot()["steps"].tolist() == [1, 1]


@pytest.mark.parametrize("ignore_terminations", [False, True])
def test_unconfigured_host_preserves_info_failure_and_time_limit_flags(
    ignore_terminations,
):
    env = _env()
    env.termination_manager = None
    env.cfg.ignore_terminations = ignore_terminations
    env.max_episode_steps = 2
    env._elapsed_steps[:] = torch.tensor([1, 2])
    terminated, truncated = env._get_termination_flags(
        {},
        {"success": torch.tensor([True, False]), "fail": torch.tensor([False, True])},
    )
    assert terminated.tolist() == [not ignore_terminations] * 2
    assert truncated.tolist() == [False, True]


def test_explicit_export_does_not_inherit_previous_episode_success():
    env = _env()
    env.initialize_task_outcome()
    env.pose[0, 0, 3] = 1
    env.termination_manager.compute()
    env._demo_episode_metadata = [{"segments": []}, {"segments": []}]
    env._demo_steps = torch.ones(2, dtype=torch.long)
    env.rollout_buffer = None
    env._task_success = torch.ones(2, dtype=torch.bool)  # Previous reset snapshot.
    env.episode_success_status = torch.zeros(2, dtype=torch.bool)
    metadata = env.get_demo_episode_metadata(0)
    assert metadata["success"] is False
    assert metadata["segments"][0]["success"] is False
    assert metadata["task_outcome"]["accepted"] is False


def test_trajectory_only_export_retains_final_gym_flags():
    env = _env()
    env.initialize_task_outcome()
    env.termination_manager.compute(time_out=torch.tensor([True, False]))
    env._demo_episode_metadata = [{"segments": []}, {"segments": []}]
    env._demo_steps = torch.ones(2, dtype=torch.long)
    env.rollout_buffer = None
    env._task_success = torch.zeros(2, dtype=torch.bool)
    env.episode_success_status = torch.zeros(2, dtype=torch.bool)
    metadata = env.get_demo_episode_metadata(0)
    assert metadata["success"] is False
    assert metadata["truncated"] is True
    assert metadata["terminated"] is False
    assert metadata["terminal_reason"] == "truncated"


def test_configured_rl_composes_existing_physical_and_embodiment_components(tmp_path):
    source = yaml.safe_load(_CONFIG.read_text())
    physical = {
        "environment_id": "configured_rl_composition_fixture",
        "physics": source["physics"],
        "max_episode_steps": source["max_episode_steps"],
        "simulation": {
            key: source[key]
            for key in (
                "light",
                "background",
                "rigid_object",
                "rigid_object_group",
                "articulation",
            )
        },
        "env": source["env"],
    }
    embodiment = {
        "embodiment_id": "configured_rl_robot_fixture",
        "simulation": source["robot"],
        "sensor": source["sensor"],
    }
    (tmp_path / "environment.yaml").write_text(yaml.safe_dump(physical))
    (tmp_path / "embodiment.yaml").write_text(yaml.safe_dump(embodiment))
    deployment = {
        "id": "ConfiguredRLComponentsTest-v1",
        "environment": {"component": "environment.yaml"},
        "embodiment": {"component": "embodiment.yaml"},
        "rl_task": source["rl_task"],
    }
    cfg = config_to_cfg(deployment, source_path=tmp_path / "task.yaml")
    assert isinstance(cfg, EmbodiedEnvCfg)
    assert cfg.robot.uid == "Manipulator"
    assert cfg.rl_task.success["goal"].params["entity_cfg"].uid == "cube"
    assert get_env_spec(deployment["id"]).supports_rl


def test_configured_rl_rejects_mid_episode_virtual_target_resampling():
    config = yaml.safe_load(_CONFIG.read_text())
    config["env"]["events"]["randomize_goal"]["mode"] = "interval"
    with pytest.raises(ValueError, match="sampled in reset mode"):
        config_to_cfg(config, source_path=_CONFIG)


def test_falling_is_failure_only_in_configured_deployment(monkeypatch):
    from embodichain_tasks.manipulation.push_cube import PushCubeEnv

    env = _env()
    env.pose[:, 2, 3] = -0.2
    monkeypatch.setattr(EmbodiedEnv, "__init__", lambda *args, **kwargs: None)
    legacy = PushCubeEnv()
    legacy._num_envs = env.num_envs
    legacy.sim = env.sim
    legacy.goal_pose = env.goal_pose
    old_success, old_failure, _ = legacy.compute_task_state()
    assert old_success.all()  # XY alone is at the goal, as in the old task.
    assert not old_failure.any()
    assert legacy.check_truncated({}, {}).all()
    env.termination_manager.compute()
    info = env.get_info()
    assert info["fail"].all()
    assert not info["success"].any()
    assert not env.check_truncated({}, info).any()
