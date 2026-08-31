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

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    DefaultPhysicsCfg,
    NewtonPhysicsCfg,
    WindowCameraPoseCfg,
)
from embodichain.lab.sim.physics import NewtonPhysicsBackend
from embodichain.lab.sim.physics import newton as newton_physics
from embodichain.lab.sim import sim_manager


def test_physics_runtime_fields_are_stored_on_physics_cfg() -> None:
    cfg = SimulationManagerCfg(
        headless=True,
        physics_dt=0.02,
        device=torch.device("cpu"),
    )

    assert cfg.physics_dt == 0.02
    assert cfg.device == torch.device("cpu")
    assert cfg.physics_cfg.physics_dt == 0.02
    assert cfg.physics_cfg.device == torch.device("cpu")

    serialized = cfg.to_dict()
    assert "physics_dt" not in serialized
    assert "device" not in serialized
    assert serialized["physics_cfg"]["physics_dt"] == 0.02
    assert serialized["physics_cfg"]["device"] == torch.device("cpu")


def test_simulation_manager_cfg_keeps_legacy_physics_accessors() -> None:
    cfg = SimulationManagerCfg(physics_cfg=NewtonPhysicsCfg())

    cfg.physics_dt = 0.005
    cfg.device = "cuda:0"

    assert cfg.physics_cfg.physics_dt == 0.005
    assert cfg.physics_cfg.device == "cuda:0"


def test_simulation_manager_cfg_initializes_window_camera_pose() -> None:
    window_camera_pose = WindowCameraPoseCfg(
        enable_hotkey=False,
        convert_to_look_at=False,
    )

    cfg = SimulationManagerCfg(window_camera_pose=window_camera_pose)

    assert cfg.window_camera_pose == window_camera_pose


def test_simulation_manager_cfg_has_no_scene_construction_switch() -> None:
    cfg = SimulationManagerCfg()

    assert "scene_construction" not in cfg.to_dict()
    with pytest.raises(TypeError, match="scene_construction"):
        SimulationManagerCfg(scene_construction="legacy")


def test_newton_physics_cfg_uses_device() -> None:
    cfg = NewtonPhysicsCfg(device="cuda:1")

    serialized = cfg.to_dict()
    assert serialized["device"] == "cuda:1"
    assert serialized["physics_dt"] == 1.0 / 100.0
    assert "solver_type" not in serialized


def test_newton_physics_cfg_uses_mujoco_warp_solver_by_default() -> None:
    from dexsim.engine.newton_physics import MJWarpSolverCfg

    cfg = NewtonPhysicsCfg()

    dexsim_cfg = cfg.to_dexsim_cfg(gpu_id=0)

    assert isinstance(dexsim_cfg.solver_cfg, MJWarpSolverCfg)
    assert dexsim_cfg.solver_cfg.solver_type == "mujoco_warp"


def test_newton_physics_cfg_passes_warp_log_suppression() -> None:
    cfg = NewtonPhysicsCfg(suppress_warp_kernel_logs=False)

    dexsim_cfg = cfg.to_dexsim_cfg(gpu_id=0)

    assert dexsim_cfg.suppress_warp_kernel_logs is False


@pytest.mark.parametrize(
    ("physics_cfg", "expect_suppressed"),
    [
        (NewtonPhysicsCfg(), True),
        (NewtonPhysicsCfg(suppress_warp_kernel_logs=False), False),
        (DefaultPhysicsCfg(), False),
    ],
)
def test_warp_runtime_init_honors_newton_log_suppression(
    monkeypatch: pytest.MonkeyPatch,
    physics_cfg,
    expect_suppressed: bool,
) -> None:
    previous_log_level = sim_manager.wp.config.log_level
    observed_log_levels = []

    def fake_init() -> None:
        observed_log_levels.append(sim_manager.wp.config.log_level)

    monkeypatch.setattr(sim_manager.wp, "init", fake_init)
    try:
        sim_manager._initialize_warp_runtime(physics_cfg)
        expected_log_level = (
            sim_manager.wp.LOG_WARNING if expect_suppressed else previous_log_level
        )
        assert observed_log_levels == [expected_log_level]
        assert sim_manager.wp.config.log_level == previous_log_level
    finally:
        sim_manager.wp.config.log_level = previous_log_level


def test_newton_warp_log_suppression_covers_world_update() -> None:
    previous_log_level = sim_manager.wp.config.log_level
    observed_log_levels = []

    class NoopProfiler:
        def section(self, *_args, **_kwargs):
            return nullcontext()

    class World:
        def update(self, _physics_dt: float) -> None:
            observed_log_levels.append(sim_manager.wp.config.log_level)

    manager = SimpleNamespace(
        profiler=NoopProfiler(),
        prepare=lambda: None,
        is_physics_manually_update=True,
        sim_config=SimpleNamespace(
            physics_dt=0.01,
            physics_cfg=NewtonPhysicsCfg(),
            visualization=SimpleNamespace(backend="none"),
        ),
        update_gizmos=lambda: None,
        _world=World(),
        _visualization_sim_step=0,
        _visualization_sim_time=0.0,
        _window_record_state=None,
    )
    try:
        SimulationManager.update(manager, physics_dt=0.01)
        assert observed_log_levels == [sim_manager.wp.LOG_WARNING]
        assert sim_manager.wp.config.log_level == previous_log_level
    finally:
        sim_manager.wp.config.log_level = previous_log_level


def test_newton_backend_exposes_resolved_solver_type() -> None:
    backend = NewtonPhysicsBackend(SimpleNamespace())
    world_config = SimpleNamespace(newton_cfg=None)
    sim_config = SimulationManagerCfg(
        physics_cfg=NewtonPhysicsCfg(
            device="cpu",
            solver_cfg={"solver_type": "xpbd"},
        ),
    )

    backend.configure_world(world_config, sim_config)

    assert backend.solver_type == "xpbd"
    assert world_config.newton_cfg.solver_cfg.solver_type == "xpbd"


def test_newton_teardown_releases_render_views_on_the_resolved_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    render_sync = MagicMock()
    newton_backend = SimpleNamespace(render_sync=render_sync)
    manager = SimpleNamespace(_world=object())
    backend = NewtonPhysicsBackend(manager)
    world_config = SimpleNamespace(newton_cfg=None)
    synchronize_device = MagicMock()
    sim_config = SimulationManagerCfg(
        gpu_id=2,
        physics_cfg=NewtonPhysicsCfg(device="cuda"),
    )
    monkeypatch.setattr(
        newton_physics.wp,
        "synchronize_device",
        synchronize_device,
    )
    from dexsim.engine.newton_physics import backend_registry

    monkeypatch.setattr(
        backend_registry,
        "get_newton_backend",
        lambda world: newton_backend if world is manager._world else None,
    )

    backend.configure_world(world_config, sim_config)
    backend.prepare_for_teardown()

    synchronize_device.assert_called_once_with("cuda:2")
    render_sync.clear.assert_called_once_with()


def test_newton_teardown_skips_cpu_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    render_sync = MagicMock()
    newton_backend = SimpleNamespace(render_sync=render_sync)
    manager = SimpleNamespace(_world=object())
    backend = NewtonPhysicsBackend(manager)
    world_config = SimpleNamespace(newton_cfg=None)
    synchronize_device = MagicMock()
    sim_config = SimulationManagerCfg(physics_cfg=NewtonPhysicsCfg(device="cpu"))
    monkeypatch.setattr(
        newton_physics.wp,
        "synchronize_device",
        synchronize_device,
    )
    from dexsim.engine.newton_physics import backend_registry

    monkeypatch.setattr(
        backend_registry,
        "get_newton_backend",
        lambda world: newton_backend if world is manager._world else None,
    )

    backend.configure_world(world_config, sim_config)
    backend.prepare_for_teardown()

    synchronize_device.assert_not_called()
    render_sync.clear.assert_called_once_with()


def test_newton_backend_syncs_render_state_without_physics_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = object()
    native_backend = SimpleNamespace(
        sync_to_dexsim=MagicMock(),
        sync_particle_fluids=MagicMock(),
    )
    monkeypatch.setattr(
        "dexsim.engine.newton_physics.backend_registry.get_newton_backend",
        lambda candidate: native_backend if candidate is world else None,
    )
    backend = NewtonPhysicsBackend(SimpleNamespace())

    backend.sync_render_state(SimpleNamespace(world=world))

    native_backend.sync_to_dexsim.assert_called_once_with(world)
    native_backend.sync_particle_fluids.assert_called_once_with(world)


def test_newton_physics_cfg_converts_mapping_solver_cfg_to_dexsim_cfg() -> None:
    from dexsim.engine.newton_physics import MJWarpSolverCfg

    cfg = NewtonPhysicsCfg(
        device="cuda",
        solver_cfg={
            "class_type": "MJWarpSolverCfg",
            "iterations": 12,
            "ls_iterations": 4,
            "use_mujoco_contacts": False,
        },
    )

    dexsim_cfg = cfg.to_dexsim_cfg(gpu_id=2)

    assert dexsim_cfg.device == "cuda:2"
    assert isinstance(dexsim_cfg.solver_cfg, MJWarpSolverCfg)
    assert dexsim_cfg.solver_cfg.iterations == 12
    assert dexsim_cfg.solver_cfg.ls_iterations == 4
    assert dexsim_cfg.solver_cfg.use_mujoco_contacts is False


def test_newton_physics_cfg_accepts_mjvbd_solver_alias() -> None:
    from dexsim.engine.newton_physics import MJVBDSolverCfg

    cfg = NewtonPhysicsCfg(solver_cfg={"class_type": "MJVBDSolverCfg"})

    dexsim_cfg = cfg.to_dexsim_cfg(gpu_id=0)

    assert isinstance(dexsim_cfg.solver_cfg, MJVBDSolverCfg)
    assert dexsim_cfg.solver_cfg.solver_type == "mjvbd"


def test_newton_physics_cfg_directly_accepts_dexsim_solver_cfg_object() -> None:
    from dexsim.engine.newton_physics import XPBDSolverCfg

    solver_cfg = XPBDSolverCfg(iterations=8, enable_restitution=True)
    cfg = NewtonPhysicsCfg(solver_cfg=solver_cfg)

    dexsim_cfg = cfg.to_dexsim_cfg(gpu_id=0)

    assert isinstance(dexsim_cfg.solver_cfg, XPBDSolverCfg)
    assert dexsim_cfg.solver_cfg.iterations == 8
    assert dexsim_cfg.solver_cfg.enable_restitution is True
