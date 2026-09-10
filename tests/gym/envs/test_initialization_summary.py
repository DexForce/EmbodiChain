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
from functools import partial

import pytest
import torch

import embodichain.lab.gym.envs.base_env as base_env_module
from embodichain.lab.gym.envs import BaseEnv, EmbodiedEnv
from embodichain.lab.sim import SimulationManagerCfg

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
        self.configs = {
            name: SimpleNamespace(func=_summary_functor, params={}, mode=mode)
            for mode, names in active_functors.items()
            for name in names
        }
        self.save_failed_episodes = False

    def get_functor_cfg(self, name: str) -> object:
        return self.configs[name]


def _summary_functor(*args, **kwargs):
    raise AssertionError("Formatting must never execute a functor")


class _ActionTermStub:
    input_key = "action"
    action_dim = 7
    cfg = SimpleNamespace(params={})


class _ActionManagerStub:
    """Action-manager stub exposing terms by processing mode."""

    active_functors = ["delta_qpos", "smooth_action"]

    def get_terms_by_mode(self, mode: str) -> list[tuple[str, object]]:
        terms = {
            "pre": [("delta_qpos", _ActionTermStub())],
            "post": [("smooth_action", _ActionTermStub())],
        }
        return terms[mode]

    def get_term(self, name: str) -> _ActionTermStub:
        return _ActionTermStub()


@pytest.fixture(autouse=True)
def _no_gpu_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Summary tests must never probe CUDA hardware or emit terminal escapes."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("COLUMNS", "120")


def _make_summary_env(mode: str = "full") -> _SummaryEnv:
    """Create a fully populated environment shell without starting simulation."""
    env = object.__new__(_SummaryEnv)
    env.cfg = _SummaryCfg()
    env.sim_cfg = SimulationManagerCfg(
        physics_dt=0.005,
        headless=True,
        device="cpu",
        num_envs=8,
        startup_summary=mode,
    )
    env.sim = SimpleNamespace(
        device=torch.device("cpu"),
        sim_config=env.sim_cfg,
        num_envs=8,
        physics=SimpleNamespace(name="default", solver_type="TGS"),
        _requested_renderer="hybrid",
        _requested_solver="TGS",
        is_window_opened=False,
        spawn_result=SimpleNamespace(topology_revision=1, needs_rebuild=False),
        _ready_spawn_topology_revision=1,
        _robots={"test_arm": object()},
        _articulations={},
        _rigid_objects={"cube": object()},
        _rigid_object_groups={},
        _deformable_objects={},
        _sensors={},
        _default_plane=object(),
        _startup_summary_logged=False,
        _scene_summary_logged=False,
    )
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


def _column_text(rendered: str, column: int) -> str:
    """Reassemble a wrapped details-table column without neighboring cells."""
    return "".join(
        cells[column].strip()
        for line in rendered.splitlines()
        if len(cells := line.split("│")) == 6
    )


def test_summary_includes_shared_runtime_scene_and_task_metadata() -> None:
    """Gym must include the shared rendering/physics/scene facts in one table."""
    rendered = "\n".join(_make_summary_env()._initialization_summary_lines())

    assert rendered.count("Environment initialized: _SummaryEnv") == 1
    for label in (
        "Section",
        "Setting",
        "Value",
        "Renderer",
        "Graphics API",
        "Backend",
        "Solver",
        "Gravity",
        "Robots",
        "Rigid objects",
    ):
        assert label in rendered
    for value in (
        "Default",
        "Constraint Dynamics",
        "_SummaryCfg",
        "cpu",
        "_RobotStub",
        "test_arm",
        "0.02 s",
        "50 Hz",
        "300 control steps",
        "dataset",
        "instruction",
        "robot_meta",
        "READY",
    ):
        assert value in rendered
    assert "render_fps" not in rendered
    assert "Pick up the red cube" not in rendered
    assert "PhysX" not in rendered
    assert "Engine threads" not in rendered
    assert "Stepping" not in rendered


def test_full_summary_preserves_every_manager_count_and_functor() -> None:
    """Full output must retain names, modes, disabled managers, and totals."""
    rendered = "\n".join(_make_summary_env()._initialization_summary_lines())
    normalized = " ".join(rendered.split())

    assert "4/5 active, 8 functors" in normalized
    for manager, count in (
        ("EventManager", "3 functors"),
        ("ObservationManager", "2 functors"),
        ("RewardManager", "disabled"),
        ("ActionManager", "2 functors"),
        ("DatasetManager", "1 functor"),
    ):
        assert any(manager in line and count in line for line in rendered.splitlines())
    for mode, name in (
        ("startup", "load_scene"),
        ("reset", "reset_robot"),
        ("reset", "randomize_objects"),
        ("modify", "normalize_rgb"),
        ("add", "task_state"),
        ("pre", "delta_qpos"),
        ("post", "smooth_action"),
        ("save", "record_episode"),
    ):
        assert any(mode in line and name in line for line in rendered.splitlines())


def test_compact_summary_separates_counts_from_functor_details() -> None:
    """Default output lists every functor in its own table after the main table."""
    rendered = "\n".join(_make_summary_env("compact")._initialization_summary_lines())

    assert "4/5 active, 8 functors" in rendered
    assert "EventManager" in rendered
    assert "3 functors" in rendered
    main, details = rendered.split("EmbodiChain · Functor Details", 1)
    assert "load_scene" not in main
    assert "randomize_objects" not in main
    assert "load_scene" in details
    assert "randomize_objects" in details
    assert "_summary_functor" in _column_text(details, 2)
    assert "input=action" in details
    assert "dim=7" in details
    assert "save failed episodes=OFF" in details


def test_summary_omits_metadata_section_for_render_fps_only() -> None:
    """Gym's internal render cadence does not create a metadata section."""
    env = _make_summary_env()
    env.metadata = {"render_fps": 50.0}

    rendered = "\n".join(env._initialization_summary_lines())

    assert "Metadata" not in rendered


@pytest.mark.parametrize(
    "mode, expected_records", [("compact", 1), ("full", 1), ("off", 0)]
)
def test_summary_logs_once_and_consumes_simulation_summaries(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected_records: int,
) -> None:
    """The ready Gym table must suppress duplicate core, scene, and Gym tables."""
    env = _make_summary_env(mode)
    calls: list[tuple[object, dict[str, object]]] = []

    def capture(message: object, **kwargs: object) -> None:
        calls.append((message, kwargs))

    monkeypatch.setattr(base_env_module.logger, "log_info", capture)
    env._log_initialization_summary()
    env._log_initialization_summary()

    assert len(calls) == expected_records
    if expected_records:
        assert calls[0][1] == {"prefix": False}
        assert "Environment initialized: _SummaryEnv" in str(calls[0][0])
        assert str(calls[0][0]).count("EmbodiChain · Functor Details") == 1
        assert env.sim._startup_summary_logged
        assert env.sim._scene_summary_logged


def test_scene_setup_defers_simulation_summary_until_gym_is_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scene declaration must not print the standalone manager table early."""
    env = _make_summary_env()
    emitted: list[str] = []

    def create_sim(cfg: SimulationManagerCfg, *, defer_startup_summary: bool = False):
        if not defer_startup_summary:
            emitted.append("standalone startup")
        return SimpleNamespace()

    monkeypatch.setattr(base_env_module, "SimulationManager", create_sim)
    monkeypatch.setattr(env, "_declare_robot", lambda **kwargs: None)
    monkeypatch.setattr(env, "_prepare_scene", lambda **kwargs: None)
    BaseEnv._setup_scene(env)

    assert emitted == []


@pytest.mark.parametrize("mode", ["compact", "full"])
def test_functor_details_show_type_specific_settings(mode):
    env = _make_summary_env(mode)
    env.event_manager = _ManagerStub({"interval": ["record"]})
    env.event_manager.configs["record"].interval_step = 10
    env.observation_manager.configs["task_state"].name = "robot/eef_pose"
    env.reward_manager = _ManagerStub({"add": ["reach"]})
    env.reward_manager.configs["reach"].weight = -0.25
    env.dataset_manager.save_failed_episodes = True
    rendered = " ".join("\n".join(env._initialization_summary_lines()).split())
    for expected in (
        "every 10 control steps",
        "output=robot/eef_pose",
        "weight=-0.25",
        "save failed episodes=ON",
    ):
        assert expected in rendered


def test_full_details_expand_parameters_without_evaluating_values():
    from embodichain.lab.gym.envs.managers.cfg import SceneEntityCfg

    class Opaque:
        def __repr__(self):
            raise AssertionError("Do not evaluate arbitrary repr")

    env = _make_summary_env("full")
    cfg = env.event_manager.configs["load_scene"]
    cfg.func = partial(_summary_functor, scale=2.0)
    cfg.params = {
        "target": SceneEntityCfg(uid="cube", body_ids=[1, 3]),
        "large": list(range(1000)),
        "tensor": torch.zeros(100, 3),
        "opaque": Opaque(),
    }
    rendered = "\n".join(env._initialization_summary_lines())
    flattened = _column_text(rendered, 4).replace(" ", "")
    assert "test_initialization_summary._summary_functor" in _column_text(rendered, 2)
    assert 'uid="cube"' in flattened or "uid='cube'" in flattened
    assert "body_ids=[1,3]" in flattened
    assert "list(len=1000)" in flattened
    assert "shape=(100,3)" in flattened
    assert "Opaque" in flattened
    assert "scale=2.0" in flattened
    assert "0,1,2,3,4,5,6,7,8,9" not in flattened
    env.sim_cfg.startup_summary = "compact"
    compact = "\n".join(env._initialization_summary_lines())
    assert "Params" not in compact
    assert "Opaque" not in compact


def test_off_omits_both_tables_and_empty_managers_have_no_details():
    assert _make_summary_env("off")._initialization_summary_lines() == []
    env = _make_summary_env()
    for _, attribute in env._manager_summary_fields:
        setattr(env, attribute, None)
    rendered = "\n".join(env._initialization_summary_lines())
    assert "0/5 active, 0 functors" in rendered
    assert "Functor Details" not in rendered


@pytest.mark.parametrize("width", [64, 80, 120])
def test_details_table_width_and_color_do_not_change_content(width):
    import re
    from wcwidth import wcswidth
    from embodichain.lab.gym.envs._startup_summary import format_functor_summary

    env = _make_summary_env()
    manager = env.event_manager
    groups = [("EventManager", manager, [("reset", ["reset_robot"])])]
    plain = format_functor_summary(groups, full=True, color=False, width=width)
    styled = format_functor_summary(groups, full=True, color=True, width=width)
    assert re.sub(r"\x1b\[[0-9;]*m", "", styled) == plain
    assert max(wcswidth(line) for line in plain.splitlines()) <= width
    assert "\x1b[1;36m" in styled
    assert "\x1b[1;32m" in styled
    assert "\x1b[1;33m" in styled


def test_no_color_overrides_terminal_detection(monkeypatch):
    import sys
    from embodichain.lab.gym.envs._startup_summary import format_functor_summary

    env = _make_summary_env()
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    rendered = format_functor_summary(
        [("EventManager", env.event_manager, [("reset", ["reset_robot"])])], full=False
    )
    assert "\x1b[" not in rendered
