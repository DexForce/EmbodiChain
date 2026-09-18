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

import hashlib
from pathlib import Path

import pytest


def test_component_snapshot_tracks_referenced_content(tmp_path: Path) -> None:
    from embodichain.lab.scripts.evaluate_task_objective import _component_snapshot

    (tmp_path / "task.yaml").write_text(
        "id: Test-v1\nenvironment: {component: env.yaml}\nobjective: {component: objective.yaml}\n"
    )
    (tmp_path / "env.yaml").write_text("physics: default\n")
    objective = tmp_path / "objective.yaml"
    objective.write_text("object_uid: cube\n")
    snapshot = _component_snapshot(tmp_path / "task.yaml")
    assert set(snapshot) == {"task.yaml", "env.yaml", "objective.yaml"}
    assert (
        snapshot["objective.yaml"]["sha256"]
        == hashlib.sha256(objective.read_bytes()).hexdigest()
    )
    objective.write_text("object_uid: changed\n")
    assert (
        _component_snapshot(tmp_path / "task.yaml")["objective.yaml"]["sha256"]
        != snapshot["objective.yaml"]["sha256"]
    )


def test_report_keeps_execution_physics_and_acceptance_independent() -> None:
    from embodichain.lab.scripts.evaluate_task_objective import _outcomes

    episode = {
        "completed": False,
        "success": [False],
        "terminal_reason": "validation_failed",
        "segments": [{"metadata": {"runtime": {"status": "completed"}}}],
    }
    physical = {"success": True, "completed_regions": 3}
    result = _outcomes(episode, physical, action_source="task_program")
    assert result["execution_outcome"]["status"] == "incomplete"
    assert result["execution_outcome"]["segment_statuses"] == ["completed"]
    assert result["demo_acceptance"]["success"] is False
    assert result["physical_outcome"]["success"] is True
    assert result["persistence_status"]["dataset"] == "not_requested"
    physical["success"] = False
    assert result["physical_outcome"]["success"] is True


def test_replay_report_does_not_invent_demo_acceptance() -> None:
    from embodichain.lab.scripts.evaluate_task_objective import _outcomes

    result = _outcomes(None, None, action_source="dynamic_replay")
    assert result["demo_acceptance"]["status"] == "not_applicable"
    assert result["physical_outcome"]["status"] == "not_evaluated"
    assert result["execution_outcome"]["status"] == "not_available"


def test_run_snapshot_serializes_arrays_and_rejects_unknown_values() -> None:
    import numpy as np
    import torch
    from embodichain.lab.scripts.evaluate_task_objective import _json_value

    assert _json_value(
        {"pose": np.array([1.0, 2.0]), "seed": np.int64(3), "x": torch.tensor([4])}
    ) == {"pose": [1.0, 2.0], "seed": 3, "x": [4]}
    with pytest.raises(TypeError):
        _json_value(object())


def test_snapshot_keeps_dynamic_manager_terms() -> None:
    from embodichain.utils import configclass
    from embodichain.lab.scripts.evaluate_task_objective import _json_value

    @configclass
    class Events:
        pass

    events = Events()
    events.initial_pose = {"range": 0.01}
    assert _json_value(events) == {"initial_pose": {"range": 0.01}}


def test_snapshot_serializes_native_simulation_enums() -> None:
    from embodichain.lab.sim import SimulationManagerCfg
    from embodichain.lab.scripts.evaluate_task_objective import _json_value

    cfg = SimulationManagerCfg()
    assert _json_value(cfg.thread_mode) == {
        "enum": "dexsim.cuda.pybind.types:ThreadMode",
        "name": "RENDER_SHARE_ENGINE",
        "value": 0,
    }


@pytest.fixture
def runner_case(monkeypatch, tmp_path):
    import argparse
    import gymnasium as gym
    from types import SimpleNamespace
    from embodichain.lab.gym.utils import gym_utils, registration
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.scripts import evaluate_task_objective as runner
    from embodichain.utils import configclass

    @configclass
    class Config:
        seed: int = -1
        trajectory_auto_save: bool = True
        init_rollout_buffer: bool = False

    source = tmp_path / "task.yaml"
    source.write_text("id: Fixture-v0\n")
    cfg = Config()
    events = []
    monkeypatch.setattr(registration, "discover_task_packages", lambda: None)
    monkeypatch.setattr(registration, "execute_init_hooks", lambda: None)
    monkeypatch.setattr(
        gym_utils,
        "build_env_cfg_from_args",
        lambda *a, **k: (cfg, {"id": "Fixture-v0"}, None),
    )
    monkeypatch.setattr(
        SimulationManager, "flush_cleanup_queue", lambda: events.append("drain")
    )
    args = argparse.Namespace(
        num_envs=1,
        initial_position_jitter=0.0,
        settle_steps=2,
        gym_config=str(source),
        output_dir=str(tmp_path / "run"),
    )
    return SimpleNamespace(
        runner=runner,
        gym=gym,
        cfg=cfg,
        args=args,
        events=events,
        report=tmp_path / "run" / "report.json",
    )


def test_constructor_traceback_resources_released_before_drain(
    runner_case, monkeypatch
):
    import json
    import weakref
    from embodichain.lab.sim import SimulationManager

    case = runner_case
    resources = []

    class NativeResource:
        pass

    def fail_constructor(**kwargs):
        resource = NativeResource()
        resources.append(weakref.ref(resource))
        raise ValueError("constructor failed")

    def drain():
        assert resources[0]() is None, "constructor traceback still owns resource"
        case.events.append("drain")

    monkeypatch.setattr(case.gym, "make", fail_constructor)
    monkeypatch.setattr(SimulationManager, "flush_cleanup_queue", drain)
    with pytest.raises(ValueError, match="constructor failed"):
        case.runner._run(case.args)
    assert case.events == ["drain"]
    report = json.loads(case.report.read_text())
    assert report["error"]["type"] == "ValueError"
    assert "fail_constructor" in report["error"]["traceback"]


def test_close_failure_still_drains_and_preserves_original_error(
    runner_case, monkeypatch
):
    import json
    from types import SimpleNamespace

    case = runner_case

    def reset(**kwargs):
        raise ValueError("reset failed")

    def close(**kwargs):
        case.events.append("close")
        raise RuntimeError("close failed")

    env = SimpleNamespace(reset=reset, close=close)
    monkeypatch.setattr(case.gym, "make", lambda **k: SimpleNamespace(unwrapped=env))
    with pytest.raises(ValueError, match="reset failed"):
        case.runner._run(case.args)
    assert case.events == ["close", "drain"]
    report = json.loads(case.report.read_text())
    assert report["error"]["message"] == "reset failed"
    assert report["cleanup_errors"][0]["message"] == "close failed"


def test_program_completion_requires_bridge_evidence():
    from embodichain.lab.scripts.evaluate_task_objective import _outcomes

    episode = {
        "completed": False,
        "terminal_reason": "validation_failed",
        "segments": [{"metadata": {"runtime": {"status": "completed"}}}],
    }
    result = _outcomes(
        episode, None, action_source="task_program", program_completed=True
    )
    assert result["execution_outcome"]["status"] == "completed"
    assert result["demo_acceptance"]["success"] is False


@pytest.fixture
def completed_runner(runner_case, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import torch
    import gymnasium as gym
    from tensordict import TensorDict
    from embodichain.lab.gym.envs import EmbodiedEnv, demo
    from embodichain.lab.gym.envs.types import ControllerAction
    from embodichain.lab.gym.envs.expert_trajectory import build_expert_action_spec

    case = runner_case

    class Env(gym.Env):
        _preprocess_action = EmbodiedEnv._preprocess_action
        _prepare_controller_action = EmbodiedEnv._prepare_controller_action
        _mask_controller_demo_action = EmbodiedEnv._mask_controller_demo_action
        _write_trajectory_step = EmbodiedEnv._write_trajectory_step
        _write_episode_rollout_step = EmbodiedEnv._write_episode_rollout_step
        _hook_after_sim_step = EmbodiedEnv._hook_after_sim_step
        _update_episode_success_status = EmbodiedEnv._update_episode_success_status

        def __init__(self):
            self.cfg = case.cfg
            self.cfg.seed = 42
            self.num_envs = 1
            self.device = torch.device("cpu")
            self.active_joint_ids = [0]
            self.max_episode_steps = self._max_rollout_steps = 8
            self.step_dt = 0.04
            self.robot = SimpleNamespace(
                dof=1,
                joint_names=["joint"],
                get_qpos=lambda: torch.ones(1, 1),
                set_qpos=lambda value, target: None,
                set_local_pose=lambda value: None,
            )
            self.expert_action_spec = build_expert_action_spec(
                joint_names=["joint"], joint_command_mode="position"
            )
            self._traj_buffer = TensorDict(
                {
                    "states": TensorDict(
                        {
                            "robot": {
                                "root_pose": torch.zeros(1, 8, 7),
                                "qpos": torch.ones(1, 8, 1),
                            }
                        },
                        batch_size=[1, 8],
                    ),
                    "actions": torch.zeros(1, 8, 1),
                },
                batch_size=[1, 8],
            )
            self.rollout_buffer = TensorDict(
                {"actions": torch.zeros(1, 8, 1), "rewards": torch.zeros(1, 8)},
                batch_size=[1, 8],
                device="cpu",
            )
            self._profiler = SimpleNamespace(section=lambda name: nullcontext())
            self._rollout_buffer_mode = "expert"
            self.action_manager = None
            self._demo_no_auto_reset = False
            self._active_task_program_bridge = SimpleNamespace(program_completed=True)
            self._demo_active_segment_start_steps = torch.zeros(1, dtype=torch.long)
            self.reset_seeds = []
            self.closed_rollout_steps = None
            self.physical_objective = SimpleNamespace(
                cfg=SimpleNamespace(object_uid="cube"),
                snapshot_row=lambda i: {"observed_steps": int(self._elapsed_steps[0])},
            )
            self.sim = SimpleNamespace(
                enable_physics=lambda enabled: None,
                get_rigid_object=lambda uid: SimpleNamespace(
                    get_local_pose=lambda: torch.zeros(1, 7)
                ),
            )

        def reset(self, *, seed=None, options=None):
            self.reset_seeds.append(seed)
            self.cfg.seed = 123
            self._traj_steps = torch.zeros(1, dtype=torch.long)
            self.rollout_steps = torch.zeros(1, dtype=torch.long)
            self._elapsed_steps = torch.zeros(1, dtype=torch.long)
            self._demo_steps = torch.zeros(1, dtype=torch.long)
            self._demo_active_mask = torch.ones(1, dtype=torch.bool)
            self.episode_success_status = torch.zeros(1, dtype=torch.bool)
            return {}, {}

        def get_obs(self):
            return {}

        def get_info(self):
            return {}

        def _seed_trajectory_states(self, env_ids):
            pass

        def _seed_expert_observations(self, obs, env_ids):
            pass

        def step(self, action):
            command = self._preprocess_action(action)
            self._elapsed_steps += 1
            done = torch.zeros(1, dtype=torch.bool)
            self._hook_after_sim_step({}, command, torch.zeros(1), done, {})
            return {}, torch.zeros(1), done, done, {}

        def save_trajectory(self, path):
            length = int(self._traj_steps[0])
            meta = self.expert_action_spec.metadata(step_dt=self.step_dt)
            meta.update(
                lengths=[length],
                num_steps=length,
                num_envs=1,
                active_joint_ids=[0],
                robot_dof=1,
            )
            torch.save(
                {
                    "states": self._traj_buffer["states"][:, :length].clone(),
                    "actions": self._traj_buffer["actions"][:, :length].clone(),
                    "meta": meta,
                },
                path,
            )
            self.saved_rollout_steps = self.rollout_steps.clone()
            self.saved_demo_steps = self._demo_steps.clone()
            self.saved_mask = self._demo_active_mask.clone()

        def close(self, **kwargs):
            case.events.append("close")

    env = Env()
    case.cfg.seed = -1

    def make(**kwargs):
        case.cfg.seed = 42
        return env

    monkeypatch.setattr(case.gym, "make", make)

    def execute(env):
        env.step(ControllerAction(torch.ones(1, 1)))
        env._demo_active_mask.fill_(False)
        return SimpleNamespace(
            to_metadata=lambda: {
                "completed": True,
                "terminal_reason": "completed",
                "length": 1,
                "segments": [
                    {"end_step": 1, "metadata": {"runtime": {"status": "completed"}}}
                ],
            }
        )

    monkeypatch.setattr(demo, "execute_demo_episode", execute)
    case.env = env
    return case


def test_settling_tail_is_recorded_and_replayed_without_extending_demo(
    completed_runner,
):
    import json
    import torch

    case = completed_runner
    report = json.loads(case.runner._run(case.args).read_text())
    trajectory = torch.load(case.report.parent / "expert.pt", weights_only=False)
    assert trajectory["meta"]["lengths"] == [3]
    assert [
        result["physical_outcome"]["observed_steps"] for result in report["results"]
    ] == [3, 3]
    assert report["expert_episode"]["length"] == 1
    assert report["expert_episode"]["segments"][0]["end_step"] == 1
    assert case.env.saved_rollout_steps.tolist() == [1]
    assert case.env.saved_demo_steps.tolist() == [1]
    assert case.env.saved_mask.tolist() == [False]


def test_report_refreshes_effective_seed_and_reuses_it_for_replay(completed_runner):
    import json

    case = completed_runner
    report = json.loads(case.runner._run(case.args).read_text())
    assert report["seed"] == report["resolved_config"]["seed"] == 123
    assert case.env.reset_seeds == [42, 123]


@pytest.mark.parametrize("stage", ["constructor", "close"])
def test_system_exit_preserves_exit_code_and_drains(runner_case, monkeypatch, stage):
    from types import SimpleNamespace

    case = runner_case

    def fail(**kwargs):
        raise SystemExit(7)

    if stage == "constructor":
        monkeypatch.setattr(case.gym, "make", fail)
    else:
        env = SimpleNamespace(reset=fail, close=fail)
        monkeypatch.setattr(
            case.gym, "make", lambda **kwargs: SimpleNamespace(unwrapped=env)
        )
    with pytest.raises(SystemExit) as raised:
        case.runner._run(case.args)
    assert raised.value.code == 7
    assert case.events == ["drain"]


def test_tail_failure_restores_demo_recording_guards(completed_runner, monkeypatch):
    case = completed_runner
    original_step = case.env.step

    def fail_tail(action):
        if int(case.env._elapsed_steps[0]) == 1:
            raise ValueError("settle failed")
        return original_step(action)

    monkeypatch.setattr(case.env, "step", fail_tail)
    with pytest.raises(ValueError, match="settle failed"):
        case.runner._run(case.args)
    assert case.env._demo_active_mask.tolist() == [False]
    assert case.env._demo_no_auto_reset is False
    assert case.env.rollout_steps.tolist() == [1]
    assert case.events == ["close", "drain"]


def test_close_failure_after_completed_run_is_reported_and_drained(
    completed_runner, monkeypatch
):
    import json

    case = completed_runner

    def close(**kwargs):
        case.events.append("close")
        raise RuntimeError("close failed")

    monkeypatch.setattr(case.env, "close", close)
    with pytest.raises(RuntimeError, match="close failed"):
        case.runner._run(case.args)
    report = json.loads(case.report.read_text())
    assert report["status"] == "error"
    assert report["cleanup_errors"][0]["message"] == "close failed"
    assert case.events == ["close", "drain"]
