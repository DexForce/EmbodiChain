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

import importlib
from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import DefaultPhysicsCfg, NewtonPhysicsCfg, RenderCfg

pytestmark = pytest.mark.no_sim


def _sim(physics_cfg=None):
    cfg = SimulationManagerCfg(
        headless=True, physics_cfg=physics_cfg, render_cfg=RenderCfg(renderer="hybrid")
    )
    return SimpleNamespace(
        sim_config=cfg,
        device=torch.device(cfg.device),
        num_envs=cfg.num_envs,
        physics=SimpleNamespace(
            name=(
                "newton" if isinstance(cfg.physics_cfg, NewtonPhysicsCfg) else "default"
            ),
            solver_type=(
                "mujoco_warp" if isinstance(cfg.physics_cfg, NewtonPhysicsCfg) else None
            ),
            cuda_graph_status="pending",
            sync_render_state=lambda result: None,
        ),
        _requested_renderer="auto",
        _requested_solver="auto",
        is_window_opened=False,
        _robots={},
        _articulations={},
        _rigid_objects={},
        _rigid_object_groups={},
        _deformable_objects={},
        _sensors={},
        _lights={},
        _constraints={},
        _default_plane=None,
        _world_config=SimpleNamespace(backend=SimpleNamespace(name="VULKAN")),
        spawn_result=None,
    )


def test_startup_configuration_defaults_and_validation():
    cfg = SimulationManagerCfg()
    assert getattr(cfg, "startup_summary", None) == "compact"
    assert cfg.dexsim_startup_info is False
    for mode in ("compact", "full", "off"):
        assert SimulationManagerCfg(startup_summary=mode).startup_summary == mode
    with pytest.raises(ValueError, match="startup_summary"):
        SimulationManagerCfg(startup_summary="verbose")


def test_cpu_physics_does_not_imply_rendering_disabled(monkeypatch):
    summary = importlib.import_module("embodichain.lab.sim._startup_summary")
    monkeypatch.setattr(summary.torch.cuda, "is_available", lambda: False)
    rows = summary.simulation_rows(_sim())
    text = summary.format_summary("Simulation initialized", rows, color=False)
    assert "Default" in text
    assert "Constraint Dynamics" in text
    assert "auto -> hybrid" in text
    assert "CLOSED" in text
    assert "cpu" in text
    assert "10 ms (100 Hz)" in text
    assert "Engine threads" not in text and "Stepping" not in text
    assert "PhysX" not in text
    assert "\x1b" not in text and "\x00" not in text


def test_newton_reports_pending_then_resolved_solver_and_graph(monkeypatch):
    summary = importlib.import_module("embodichain.lab.sim._startup_summary")
    monkeypatch.setattr(summary.torch.cuda, "is_available", lambda: False)
    sim = _sim(NewtonPhysicsCfg())
    sim.physics.solver_type = "auto"
    before = str(summary.simulation_rows(sim))
    assert "PENDING" in before
    sim.spawn_result = SimpleNamespace(topology_revision=1, needs_rebuild=False)
    sim._ready_spawn_topology_revision = 1
    sim.physics.solver_type = "mujoco_warp"
    after = str(summary.scene_rows(sim))
    assert "auto -> mujoco_warp" in after
    assert "PENDING" in after and "CAPTURED" not in after
    sim.physics.cuda_graph_status = "captured"
    assert "CAPTURED" in str(summary.scene_rows(sim))


def test_full_mode_includes_backend_specific_diagnostics(monkeypatch):
    summary = importlib.import_module("embodichain.lab.sim._startup_summary")
    monkeypatch.setattr(summary.torch.cuda, "is_available", lambda: False)
    sim = _sim(DefaultPhysicsCfg(device="cuda:0"))
    compact = str(summary.simulation_rows(sim))
    sim.sim_config.startup_summary = "full"
    full = str(summary.simulation_rows(sim))
    assert "Contact capacity" not in compact
    assert "Contact capacity" in full
    assert "Cache" in full


def test_table_wraps_long_values_and_preserves_visible_alignment():
    summary = importlib.import_module("embodichain.lab.sim._startup_summary")
    rows = [
        ("Runtime", "Config", "/some/very/long/path/" * 8),
        ("Physics", "Solver", "mujoco_warp"),
    ]
    text = summary.format_summary("Simulation initialized", rows, color=False, width=80)
    assert max(map(len, text.splitlines())) <= 80
    assert "mujoco_warp" in text
    assert text.startswith("╭") and text.endswith("╯")
    colored = summary.format_summary(
        "Simulation initialized", rows, color=True, width=80
    )
    assert "\x1b[" in colored


def test_no_color_overrides_terminal_highlighting(monkeypatch):
    summary = importlib.import_module("embodichain.lab.sim._startup_summary")
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(summary.sys, "stderr", SimpleNamespace(isatty=lambda: True))
    assert "\x1b" not in summary.format_summary("Ready", [("Scene", "State", "READY")])


def test_startup_emission_is_once_and_off_is_quiet(monkeypatch):
    from embodichain.utils import logger

    calls = []
    monkeypatch.setattr(logger, "log_info", lambda text, **kwargs: calls.append(text))
    summary = importlib.import_module("embodichain.lab.sim._startup_summary")
    monkeypatch.setattr(summary.torch.cuda, "is_available", lambda: False)
    sim = object.__new__(SimulationManager)
    sim.__dict__.update(vars(_sim()))
    sim._startup_summary_logged = False
    sim._scene_summary_logged = False
    sim._log_startup_summary()
    sim._log_startup_summary()
    assert len(calls) == 1
    sim.sim_config.startup_summary = "off"
    sim._log_scene_summary()
    assert len(calls) == 1


def test_default_solver_does_not_query_native_configuration(monkeypatch):
    import dexsim
    from embodichain.lab.sim.physics.default import DefaultPhysicsBackend

    def unexpected_query():
        pytest.fail("Default solver diagnostics must not query native configuration")

    monkeypatch.setattr(dexsim, "get_physics_config", unexpected_query)
    backend = DefaultPhysicsBackend(_sim())
    assert backend.solver_type is None


@pytest.mark.parametrize("mode", ["compact", "full"])
def test_default_summary_does_not_require_solver_selection(mode):
    summary = importlib.import_module("embodichain.lab.sim._startup_summary")
    sim = _sim()
    sim.sim_config.startup_summary = mode
    del sim.physics.solver_type
    del sim._requested_solver
    text = summary.format_summary(
        "Simulation initialized", summary.simulation_rows(sim), color=False
    )
    assert "Constraint Dynamics" in text


def test_scene_emission_waits_for_readiness_and_is_once(monkeypatch):
    from embodichain.utils import logger

    sim = object.__new__(SimulationManager)
    sim.__dict__.update(vars(_sim()))
    sim._is_constructed = True
    sim._defer_startup_summary = False
    sim._scene_summary_logged = False
    sim._spawn_scene = SimpleNamespace(
        builder=SimpleNamespace(is_finalized=False, has_pending_changes=False)
    )
    calls = []
    monkeypatch.setattr(logger, "log_info", lambda text, **kwargs: calls.append(text))
    sim._log_scene_summary()
    assert not calls
    sim._spawn_scene.builder.is_finalized = True
    sim._spawn_scene.builder.result = SimpleNamespace(
        topology_revision=1, needs_rebuild=False
    )
    sim._ready_spawn_topology_revision = 1
    sim._log_scene_summary()
    sim._log_scene_summary()
    assert len(calls) == 1 and "READY" in calls[0]


def test_newton_scene_emission_waits_for_cuda_graph_capture(monkeypatch):
    from embodichain.utils import logger

    sim = object.__new__(SimulationManager)
    sim.__dict__.update(vars(_sim(NewtonPhysicsCfg())))
    sim._is_constructed = True
    sim._defer_startup_summary = False
    sim._scene_summary_logged = False
    sim._spawn_scene = SimpleNamespace(
        builder=SimpleNamespace(
            is_finalized=True,
            has_pending_changes=False,
            result=SimpleNamespace(topology_revision=1, needs_rebuild=False),
        )
    )
    sim._ready_spawn_topology_revision = 1
    calls = []
    monkeypatch.setattr(logger, "log_info", lambda text, **kwargs: calls.append(text))

    sim._log_scene_summary()
    assert not calls
    assert not sim._scene_summary_logged

    sim.physics.cuda_graph_status = "captured"
    sim._log_scene_summary()
    sim._log_scene_summary()
    assert len(calls) == 1
    assert "CAPTURED" in calls[0]


def test_world_config_receives_startup_switch_before_construction(monkeypatch):
    import embodichain.lab.sim.sim_manager as manager_module

    native_cfg = SimpleNamespace(
        raytrace_config=SimpleNamespace(), postprocess_config=SimpleNamespace()
    )
    monkeypatch.setattr(manager_module.dexsim, "WorldConfig", lambda: native_cfg)
    sim = object.__new__(SimulationManager)
    sim._material_cache_dir = "/tmp/test-material-cache"
    sim.physics = SimpleNamespace(configure_world=lambda *_: None)
    cfg = SimulationManagerCfg(render_cfg=RenderCfg(renderer="hybrid"))
    assert sim._convert_sim_config(cfg).log_startup_info is False
    cfg.dexsim_startup_info = True
    assert sim._convert_sim_config(cfg).log_startup_info is True


@pytest.mark.parametrize(
    "mode,native_info", [("compact", False), ("full", True), ("off", False)]
)
def test_gym_config_preserves_summary_preferences(mode, native_info):
    from embodichain.lab.gym.utils.gym_utils import (
        config_to_cfg,
        DEFAULT_MANAGER_MODULES,
    )

    cfg = config_to_cfg(
        {
            "id": "EmbodiedEnv-v1",
            "physics": "default",
            "env": {},
            "robot": {"uid": "TestRobot"},
            "startup_summary": mode,
            "dexsim_startup_info": native_info,
        },
        manager_modules=DEFAULT_MANAGER_MODULES,
    )
    assert cfg.sim_cfg.startup_summary == mode
    assert cfg.sim_cfg.dexsim_startup_info is native_info


def test_summary_uses_existing_device_metadata_without_runtime_queries(monkeypatch):
    summary = importlib.import_module("embodichain.lab.sim._startup_summary")
    sim = _sim()
    sim._render_device_name = "Existing renderer GPU"

    def unexpected_query(*args, **kwargs):
        pytest.fail("A read-only startup snapshot must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_query)
    monkeypatch.setattr(torch.cuda, "get_device_name", unexpected_query)
    assert "Existing renderer GPU" in str(summary.simulation_rows(sim))


def test_failed_preparation_does_not_report_ready_or_consume_snapshot(monkeypatch):
    from embodichain.utils import logger

    sim = object.__new__(SimulationManager)
    sim.__dict__.update(vars(_sim()))
    sim._is_constructed = True
    sim._defer_startup_summary = False
    sim._scene_summary_logged = False
    result = SimpleNamespace(topology_revision=1, needs_rebuild=False)

    def fail_runtime(_):
        raise RuntimeError("runtime not prepared")

    scene = SimpleNamespace(
        builder=SimpleNamespace(
            is_finalized=True, result=result, has_pending_changes=False
        ),
        prepare_runtime_config=fail_runtime,
        bind=lambda: None,
    )
    sim._spawn_scene = scene
    sim._world = SimpleNamespace(render_camera_group=lambda _: None)
    calls = []
    monkeypatch.setattr(logger, "log_info", lambda text, **kwargs: calls.append(text))
    with pytest.raises(RuntimeError, match="runtime not prepared"):
        sim.prepare()
    sim.render_camera_group([])
    assert not calls
    assert not sim._scene_summary_logged
    scene.prepare_runtime_config = lambda _: None
    sim._prepare_spawn_runtime = lambda _: None
    sim._sync_spawn_render_state = lambda _: None
    sim._attach_parented_cameras = lambda: None
    sim.prepare()
    sim.render_camera_group([])
    assert len(calls) == 1 and "READY" in calls[0]
