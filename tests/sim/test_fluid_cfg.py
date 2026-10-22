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

"""Fluid declaration and solver configuration contracts."""

from __future__ import annotations

import numpy as np
import pytest
from embodichain.lab.sim.cfg import (
    FluidObjectCfg,
    FluidSpawnCfg,
    NewtonPhysicsCfg,
)
from embodichain.compute.geometry.particles import sample_volume_particles

pytestmark = pytest.mark.no_sim


def test_fluid_round_trip_keeps_native_render_options():
    cfg = FluidObjectCfg.from_dict(
        {
            "uid": "water",
            "spawn": {"kind": "box", "size": [0.02] * 3},
            "render": {"mode": "mesh_surface", "mesh_surface": {"iso_density": 0.57}},
        }
    )
    restored = FluidObjectCfg.from_dict(cfg.to_dict())
    np.testing.assert_array_equal(cfg.spawn.sample(0.005), restored.spawn.sample(0.005))
    assert restored.render.mesh_surface == {"iso_density": 0.57}
    restored.render.mesh_surface["iso_density"] = 0.8
    assert cfg.render.mesh_surface["iso_density"] == 0.57


def test_native_sph_and_yaml_mapping_use_same_solver():
    from dexsim.engine.newton_physics import SPHSolverCfg

    source = SPHSolverCfg(sph_method="dfsph", viscosity=0.0006)
    native = NewtonPhysicsCfg(solver_cfg=source).to_dexsim_cfg(0)
    mapping = NewtonPhysicsCfg(
        solver_cfg={"solver_type": "sph", "sph_method": "dfsph", "viscosity": 0.0006}
    ).to_dexsim_cfg(0)
    assert type(native.solver_cfg) is SPHSolverCfg
    assert type(mapping.solver_cfg) is SPHSolverCfg
    assert mapping.solver_cfg == source


@pytest.mark.parametrize("kind", ["box", "sphere", "cylinder"])
def test_lattice_is_deterministic_and_inside_volume(kind):
    points = sample_volume_particles(
        kind, 0.01, size=(0.04, 0.04, 0.04), radius=0.02, height=0.04
    )
    np.testing.assert_array_equal(
        points,
        sample_volume_particles(
            kind, 0.01, size=(0.04, 0.04, 0.04), radius=0.02, height=0.04
        ),
    )
    assert np.all(np.abs(points) < 0.02)
    if kind == "sphere":
        assert np.all(np.linalg.norm(points, axis=1) < 0.02)
    if kind == "cylinder":
        assert np.all(np.linalg.norm(points[:, :2], axis=1) < 0.02)


def test_grid_limit_is_checked_before_allocation():
    with pytest.raises(ValueError, match="max_particles"):
        sample_volume_particles("box", 1e-9)


@pytest.mark.parametrize(
    "positions", [[], [[float("nan"), 0, 0]], [[0, 1]], [[[0, 0, 0]]]]
)
def test_invalid_particle_sources_rejected(positions):
    with pytest.raises(ValueError):
        FluidSpawnCfg(positions=positions).sample(0.001)


def test_npz_points_are_loaded_without_pickle(tmp_path):
    path = tmp_path / "fill.npz"
    np.savez(path, positions=np.array([[1, 2, 3]], dtype=np.float32))
    np.testing.assert_array_equal(
        FluidSpawnCfg(file_path=str(path)).sample(0.01), [[1, 2, 3]]
    )


def test_fill_mesh_must_be_closed(tmp_path):
    import trimesh

    path = tmp_path / "open.obj"
    trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]]
    ).export(path)
    with pytest.raises(ValueError, match="closed"):
        sample_volume_particles("mesh_volume", 0.01, mesh_path=str(path))


@pytest.mark.parametrize("backend,num_envs", [("default", 1), ("newton", 2)])
def test_manager_rejects_unsupported_fluid_worlds_before_declaration(backend, num_envs):
    from types import SimpleNamespace
    import torch
    from embodichain.lab.sim import SimulationManager

    manager = SimpleNamespace(
        physics=SimpleNamespace(name=backend),
        device=torch.device("cuda"),
        num_envs=num_envs,
    )
    with pytest.raises(NotImplementedError):
        SimulationManager.add_fluid_object(manager, FluidObjectCfg(uid="water"))


def test_points_source_validates_spacing():
    with pytest.raises(ValueError, match="spacing"):
        FluidSpawnCfg(positions=[[0, 0, 0]]).sample(0)


def test_closed_mesh_sampling_matches_equivalent_box(tmp_path):
    import trimesh

    mesh = tmp_path / "fill.obj"
    trimesh.creation.box(extents=[0.04] * 3).export(mesh)
    actual = sample_volume_particles("mesh_volume", 0.01, mesh_path=str(mesh))
    expected = sample_volume_particles("box", 0.01, size=(0.04,) * 3)
    np.testing.assert_allclose(actual, expected)


def test_profile_source_round_trip_rebuilds_without_a_particle_file():
    cfg = FluidObjectCfg.from_dict(
        {
            "uid": "water",
            "spawn": {
                "kind": "revolved_volume",
                "profile": [[-0.02, 0.01], [0.03, 0.02]],
                "fill_level": 0.8,
                "clearance": 0.001,
                "reference_uid": "cup",
            },
        }
    )
    restored = FluidObjectCfg.from_dict(cfg.to_dict())
    assert restored.spawn.file_path is None
    points = restored.spawn.sample(0.004)
    np.testing.assert_array_equal(points, cfg.spawn.sample(0.004))
    assert points.dtype == np.float32
    assert points[:, 2].max() < -0.02 + 0.8 * 0.05
    assert len(restored.spawn.sample(0.002)) > len(points)


@pytest.mark.parametrize("extra", [{"profile": [[0, 1], [1, 1]]}, {"fill_level": 0.5}])
def test_points_reject_geometry_filters_instead_of_silently_ignoring_them(extra):
    with pytest.raises(ValueError, match="points sources"):
        FluidSpawnCfg(positions=[[0, 0, 0]], **extra).sample(0.001)


def _fluid_manager_for_renderer(renderer):
    from types import SimpleNamespace
    from unittest.mock import Mock
    import torch
    from embodichain.lab.sim import SimulationManagerCfg
    from embodichain.lab.sim.cfg import RenderCfg

    return SimpleNamespace(
        physics=SimpleNamespace(name="newton", configure_world=Mock()),
        device=torch.device("cuda"),
        num_envs=1,
        _active_newton_solver_type="sph",
        spawn_result=None,
        sim_config=SimulationManagerCfg(
            headless=True,
            physics_cfg=NewtonPhysicsCfg(),
            render_cfg=RenderCfg(renderer=renderer),
        ),
        has_native_renderer=renderer != "no-render",
        asset_uids=[],
        _fluid_objects={},
        _rigid_objects={},
        _spawn_scene=SimpleNamespace(declare=Mock()),
        create_visual_material=Mock(),
        _invalidate_fluid_state=Mock(),
        notify_visualization_topology_changed=Mock(),
    )


@pytest.mark.parametrize("mode", ["particles", "mesh_surface", "none"])
@pytest.mark.parametrize(
    "requested,process_renderer",
    [
        ("hybrid", None),
        ("fast-rt", "hybrid"),
        ("rt", "hybrid"),
        ("auto", "hybrid"),
        ("no-render", "hybrid"),
    ],
)
def test_manager_rejects_hybrid_fluid_rendering_before_sampling(
    requested, process_renderer, mode, monkeypatch
):
    from unittest.mock import Mock
    from embodichain.lab.sim import SimulationManager

    monkeypatch.setattr(
        SimulationManager, "_get_process_renderer", lambda: process_renderer
    )
    sample = Mock(side_effect=AssertionError("Fluid must be rejected before sampling"))
    monkeypatch.setattr(FluidSpawnCfg, "sample", sample)
    manager = _fluid_manager_for_renderer(requested)
    cfg = FluidObjectCfg(uid="water", render={"mode": mode})
    with pytest.raises(NotImplementedError, match="fast-rt.*Vulkan.*new process"):
        SimulationManager.add_fluid_object(manager, cfg)
    sample.assert_not_called()
    manager.create_visual_material.assert_not_called()
    manager._spawn_scene.declare.assert_not_called()


@pytest.mark.parametrize("process_renderer", [None, "hybrid"])
def test_auto_selected_hybrid_rejects_fluid_before_sampling(
    process_renderer, monkeypatch, tmp_path
):
    from unittest.mock import Mock
    from embodichain.lab.sim import SimulationManager, cfg as sim_cfg

    monkeypatch.setattr(sim_cfg, "DEFAULT_RENDERER", "auto")
    monkeypatch.setattr(
        SimulationManager, "_get_process_renderer", lambda: process_renderer
    )
    monkeypatch.setattr(
        "embodichain.lab.sim.utility.render_utils.select_default_renderer",
        lambda _: "hybrid",
    )
    manager = _fluid_manager_for_renderer("auto")
    manager._requires_native_renderer = True
    manager._material_cache_dir = tmp_path
    SimulationManager._convert_sim_config(manager, manager.sim_config)
    assert manager.sim_config.render_cfg.renderer == "hybrid"
    sample = Mock(side_effect=AssertionError("Fluid must be rejected before sampling"))
    monkeypatch.setattr(FluidSpawnCfg, "sample", sample)
    with pytest.raises(NotImplementedError, match="fast-rt.*Vulkan"):
        SimulationManager.add_fluid_object(manager, FluidObjectCfg(uid="water"))
    sample.assert_not_called()


@pytest.mark.parametrize(
    "requested,process_renderer,mode",
    [
        ("fast-rt", None, "particles"),
        ("fast-rt", "fast-rt", "mesh_surface"),
        ("rt", "rt", "mesh_surface"),
        ("no-render", "no-render", "none"),
        ("no-render", "no-render", "particles"),
    ],
)
def test_fluid_renderer_gate_allows_supported_backends(
    requested, process_renderer, mode, monkeypatch
):
    from unittest.mock import Mock
    from embodichain.lab.sim import SimulationManager

    monkeypatch.setattr(
        SimulationManager, "_get_process_renderer", lambda: process_renderer
    )
    facade = Mock()
    monkeypatch.setattr(
        "embodichain.lab.sim.sim_manager.FluidObject", Mock(return_value=facade)
    )
    manager = _fluid_manager_for_renderer(requested)
    cfg = FluidObjectCfg(
        uid="water", spawn=FluidSpawnCfg(kind="box"), render={"mode": mode}
    )
    assert SimulationManager.add_fluid_object(manager, cfg) is facade
    assert manager._fluid_objects["water"] is facade
    manager._spawn_scene.declare.assert_called_once()
