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

"""Tests for the environment initialization summary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import embodichain.lab.gym.envs.base_env as base_env_module
from embodichain.lab.gym.envs import EmbodiedEnv

pytestmark = pytest.mark.no_sim


class _SummaryEnv(EmbodiedEnv):
    """Environment stub used to exercise summary formatting only."""


class _SummaryCfg:
    """Minimal environment configuration needed by the summary."""

    seed = 42
    sim_steps_per_control = 4
    max_episode_steps = 300


class _RobotStub:
    """Minimal robot carrying the identity shown in the summary."""

    uid = "test_arm"


class _ManagerStub:
    """Manager stub exposing the common active-functor contract."""

    def __init__(self, active_functors: dict[str, list[str]]) -> None:
        self.active_functors = active_functors


class _ActionManagerStub:
    """Action-manager stub exposing terms by processing mode."""

    active_functors = ["delta_qpos", "smooth_action"]

    def get_terms_by_mode(self, mode: str) -> list[tuple[str, object]]:
        terms = {
            "pre": [("delta_qpos", object())],
            "post": [("smooth_action", object())],
        }
        return terms[mode]


def _make_summary_env() -> _SummaryEnv:
    """Create a fully populated environment shell without starting simulation."""
    env = object.__new__(_SummaryEnv)
    env.cfg = _SummaryCfg()
    env.sim_cfg = SimpleNamespace(physics_dt=0.005, headless=True)
    env.sim = SimpleNamespace(device=torch.device("cuda:0"))
    env._num_envs = 8
    env.robot = _RobotStub()
    env.sensors = {"front_camera": object(), "wrist_camera": object()}
    env.metadata = {
        "render_fps": 50.0,
        "task_type": "manipulation",
        "dataset": {
            "instruction": "Pick up the red cube",
            "robot_meta": {"model": "test_arm"},
        },
    }
    env.event_manager = _ManagerStub(
        {
            "startup": ["load_scene"],
            "reset": ["reset_robot", "randomize_objects"],
        }
    )
    env.observation_manager = _ManagerStub(
        {"modify": ["normalize_rgb"], "add": ["task_state"]}
    )
    env.reward_manager = None
    env.action_manager = _ActionManagerStub()
    env.dataset_manager = _ManagerStub({"save": ["record_episode"]})
    return env


def test_summary_includes_runtime_metadata_and_every_manager_functor() -> None:
    """The summary exposes key runtime facts and every configured functor."""
    lines = _make_summary_env()._initialization_summary_lines()
    rendered = "\n".join(lines)
    normalized_lines = {" ".join(line.split()) for line in lines}

    assert rendered.startswith("╭─ Environment initialized: _SummaryEnv")
    assert "├─ Runtime" in rendered
    assert "Config                 _SummaryCfg" in rendered
    assert "Device                 cuda:0" in rendered
    assert "Parallel environments  8" in rendered
    assert "Robot                  _RobotStub (uid=test_arm)" in rendered
    assert "Sensors                2 (front_camera, wrist_camera)" in rendered
    assert "├─ Timing" in rendered
    assert "Physics                0.005 s (200 Hz)" in rendered
    assert "Control                0.02 s (50 Hz, 4 physics steps)" in rendered
    assert "├─ Metadata" in rendered
    assert "dataset                2 keys (instruction, robot_meta)" in rendered
    assert "Pick up the red cube" not in rendered
    assert "├─ Managers (4/5 active, 8 functors)" in rendered
    assert "EventManager           3 functors" in rendered
    assert "│ startup load_scene" in normalized_lines
    assert "│ reset reset_robot, randomize_objects" in normalized_lines
    assert "ObservationManager     2 functors" in rendered
    assert "│ modify normalize_rgb" in normalized_lines
    assert "│ add task_state" in normalized_lines
    assert "RewardManager          disabled" in rendered
    assert "ActionManager          2 functors" in rendered
    assert "│ pre delta_qpos" in normalized_lines
    assert "│ post smooth_action" in normalized_lines
    assert "DatasetManager         1 functor" in rendered
    assert "│ save record_episode" in normalized_lines
    assert rendered.endswith("╰─ Ready")


def test_summary_logs_each_tree_line_as_a_separate_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every tree line retains the timestamp and log columns from the logger."""
    env = _make_summary_env()
    messages: list[str] = []
    monkeypatch.setattr(base_env_module.logger, "log_info", messages.append)

    env._log_initialization_summary()

    assert messages == env._initialization_summary_lines()
    assert all("\n" not in message for message in messages)
