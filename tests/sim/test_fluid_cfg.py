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
    RigidObjectCfg,
)
from embodichain.lab.sim.spawn.descriptors import rigid_desc_from_cfg
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
    from dexsim.engine.newton_physics import DexUniSolverCfg, SPHSolverCfg

    if "sph_cfg" not in getattr(DexUniSolverCfg, "__dataclass_fields__", {}):
        pytest.skip("This DexSim build does not compose SPH through DexUni.")

    source = SPHSolverCfg(sph_method="dfsph", viscosity=0.0006)
    native = NewtonPhysicsCfg(solver_cfg=DexUniSolverCfg(sph_cfg=source)).to_dexsim_cfg(
        0
    )
    mapping = NewtonPhysicsCfg(
        solver_cfg={
            "solver_type": "dexuni",
            "sph_cfg": {"sph_method": "dfsph", "viscosity": 0.0006},
        }
    ).to_dexsim_cfg(0)
    assert type(native.solver_cfg) is DexUniSolverCfg
    assert type(mapping.solver_cfg) is DexUniSolverCfg
    assert native.solver_cfg.sph_cfg == source
    assert mapping.solver_cfg.sph_cfg == source


def test_standalone_sph_mapping_points_to_dexuni_migration():
    with pytest.raises(RuntimeError, match="configure DexUni with sph_cfg"):
        NewtonPhysicsCfg(solver_cfg={"solver_type": "sph"}).to_dexsim_cfg(0)


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


def test_rigid_visual_mesh_does_not_replace_sdf_collision():
    cfg = RigidObjectCfg.from_dict(
        {
            "uid": "cup",
            "body_type": "kinematic",
            "shape": {
                "shape_type": "Mesh",
                "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                "triangles": [[0, 1, 2]],
                "collision": {
                    "approximation": "sdf",
                    "force_sdf": False,
                    "sdf_resolution": 192,
                },
            },
            "visual_shape": {"shape_type": "Mesh", "fpath": "cup.glb"},
        }
    )
    desc, _ = rigid_desc_from_cfg(cfg, newton_solver_type="dexuni")
    assert desc.renders[0].file_path == "cup.glb"
    assert desc.collisions[0].newton.force_sdf is False
    assert desc.collisions[0].render_source_index is None


def test_rigid_material_preserves_transmission():
    import json
    from embodichain.lab.sim.shapes import CubeCfg
    from embodichain.lab.sim.material import VisualMaterialCfg

    cfg = RigidObjectCfg(
        uid="glass",
        shape=CubeCfg(
            visual_material=VisualMaterialCfg(parameters={"transmission": 0.8})
        ),
    )
    _, materials = rigid_desc_from_cfg(cfg, newton_solver_type="dexuni")
    assert (
        json.loads(next(iter(materials.values())).parameters_json)["transmission"]
        == 0.8
    )


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


@pytest.mark.parametrize(
    "solver_cfg",
    [None, {"solver_type": "dexuni", "sph_cfg": {"sph_method": "dfsph"}}],
)
def test_fluid_declaration_accepts_dexuni_dispatch(solver_cfg, monkeypatch):
    from unittest.mock import Mock
    from embodichain.lab.sim import SimulationManager

    manager = _fluid_manager_for_renderer("no-render")
    manager._active_newton_solver_type = "dexuni"
    manager.physics.supports_dexuni_fluids = True
    manager.sim_config.physics_cfg.solver_cfg = solver_cfg
    monkeypatch.setattr(SimulationManager, "_get_process_renderer", lambda: None)
    monkeypatch.setattr("embodichain.lab.sim.sim_manager.FluidObject", Mock())
    cfg = FluidObjectCfg(
        uid="water", spawn=FluidSpawnCfg(positions=[[0, 0, 0]]), render={"mode": "none"}
    )

    SimulationManager.add_fluid_object(manager, cfg)

    manager._spawn_scene.declare.assert_called_once()
    manager.add_runtime_control.assert_called_once()


def test_fluid_declaration_preserves_explicit_joint_ownership(monkeypatch):
    from unittest.mock import Mock
    from embodichain.lab.sim import SimulationManager

    manager = _fluid_manager_for_renderer("no-render")
    manager.sim_config.physics_cfg.solver_cfg = {
        "solver_type": "dexuni",
        "mujoco_joints": [0],
        "sph_cfg": {},
    }
    monkeypatch.setattr(SimulationManager, "_get_process_renderer", lambda: None)
    monkeypatch.setattr("embodichain.lab.sim.sim_manager.FluidObject", Mock())
    cfg = FluidObjectCfg(
        uid="water", spawn=FluidSpawnCfg(positions=[[0, 0, 0]]), render={"mode": "none"}
    )
    SimulationManager.add_fluid_object(manager, cfg)
    manager.add_runtime_control.assert_not_called()


def test_dexuni_nested_sph_density_conflict_is_rejected(monkeypatch):
    from embodichain.lab.sim import SimulationManager

    manager = _fluid_manager_for_renderer("no-render")
    manager._active_newton_solver_type = "dexuni"
    manager.physics.supports_dexuni_fluids = True
    manager.sim_config.physics_cfg.solver_cfg = {
        "solver_type": "dexuni",
        "sph_cfg": {"sph_method": "dfsph", "rest_density": 1200.0},
    }
    monkeypatch.setattr(SimulationManager, "_get_process_renderer", lambda: None)

    with pytest.raises(ValueError, match="rest_density conflicts"):
        SimulationManager.add_fluid_object(manager, FluidObjectCfg(uid="water"))
    manager._spawn_scene.declare.assert_not_called()


@pytest.mark.parametrize("solver,capability", [("sph", True), ("dexuni", False)])
def test_fluid_requires_a_dexuni_build_with_sph_support(solver, capability):
    from embodichain.lab.sim import SimulationManager

    manager = _fluid_manager_for_renderer("no-render")
    manager._active_newton_solver_type = solver
    manager.physics.supports_dexuni_fluids = capability
    with pytest.raises(NotImplementedError, match="DexUni"):
        SimulationManager.add_fluid_object(manager, FluidObjectCfg(uid="water"))
    manager._spawn_scene.declare.assert_not_called()


def test_dexuni_fluid_rejects_existing_deformables():
    from embodichain.lab.sim import SimulationManager

    manager = _fluid_manager_for_renderer("no-render")
    manager._deformable_objects = {"cloth": object()}
    with pytest.raises(NotImplementedError, match="cannot coexist"):
        SimulationManager.add_fluid_object(manager, FluidObjectCfg(uid="water"))
    manager._spawn_scene.declare.assert_not_called()


def test_dexuni_deformable_rejects_existing_fluid():
    from types import SimpleNamespace
    import torch
    from embodichain.lab.sim import SimulationManager

    manager = SimpleNamespace(
        physics=SimpleNamespace(name="newton", supports_surface_deformables=True),
        device=torch.device("cuda"),
        _active_newton_solver_type="dexuni",
        _fluid_objects={"water": object()},
    )
    with pytest.raises(NotImplementedError, match="cannot coexist"):
        SimulationManager.add_deformable_object(
            manager, SimpleNamespace(deformable_type="surface")
        )


@pytest.mark.parametrize("has_sph_cfg", [False, True])
def test_dexuni_fluid_capability_tracks_native_config(monkeypatch, has_sph_cfg):
    from types import SimpleNamespace
    import dexsim.engine.newton_physics as native
    from embodichain.lab.sim.physics.newton import NewtonPhysicsBackend

    fields = {"sph_cfg": object()} if has_sph_cfg else {}
    fake_cfg = type("FakeDexUniCfg", (), {"__dataclass_fields__": fields})
    monkeypatch.setattr(native, "DexUniSolverCfg", fake_cfg)
    backend = NewtonPhysicsBackend(SimpleNamespace())
    backend._configured_solver_type = "dexuni"
    assert backend.supports_dexuni_fluids is has_sph_cfg


def test_fluid_reset_invalidates_sph_without_resetting_dexuni_bodies(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    import warp as wp
    from dexsim.engine.newton_physics import backend_registry
    from embodichain.lab.sim.physics.newton import NewtonPhysicsBackend

    wp.init()
    sph = SimpleNamespace(supports_world_reset=True, reset=Mock())
    solid = SimpleNamespace(reset=Mock())
    native = SimpleNamespace(
        solver_type="dexuni",
        sph_solver=sph,
        solver=solid,
        invalidate_cuda_graph=Mock(),
        model=SimpleNamespace(world_count=2, device="cpu"),
        _runtime=SimpleNamespace(current_state=object()),
    )
    monkeypatch.setattr(backend_registry, "get_newton_backend", lambda _: native)
    backend = NewtonPhysicsBackend(SimpleNamespace(_world=object()))

    backend.invalidate_fluid_state([1])

    native.invalidate_cuda_graph.assert_called_once()
    solid.reset.assert_not_called()
    assert sph.reset.call_args.args[0] is native._runtime.current_state
    assert sph.reset.call_args.kwargs["world_mask"].numpy().tolist() == [
        False,
        True,
        False,
    ]
    assert sph.reset.call_args.kwargs["flags"] == 0


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
        physics=SimpleNamespace(
            name="newton", configure_world=Mock(), supports_dexuni_fluids=True
        ),
        device=torch.device("cuda"),
        num_envs=1,
        _active_newton_solver_type="dexuni",
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
        add_runtime_control=Mock(),
        create_visual_material=Mock(),
        _invalidate_fluid_state=Mock(),
        notify_visualization_topology_changed=Mock(),
    )


def test_fluid_joint_ownership_selects_finalized_topology_and_rebuilds(monkeypatch):
    from types import SimpleNamespace
    from embodichain.lab.sim._runtime_controls import _DexUniFluidJointOwnership

    cfg = SimpleNamespace(
        mujoco_articulations=None, mujoco_articulation_paths=None, mujoco_joints=None
    )
    backend = SimpleNamespace(solver_type="dexuni", cfg=SimpleNamespace(solver_cfg=cfg))
    monkeypatch.setattr(
        "dexsim.engine.newton_physics.backend_registry.get_newton_backend",
        lambda world: backend,
    )
    control = _DexUniFluidJointOwnership()
    result = SimpleNamespace(world=object())
    control.initialize_model(SimpleNamespace(joint_count=3), result)
    assert cfg.mujoco_joints == (0, 1, 2)
    control.initialize_model(SimpleNamespace(joint_count=5), result)
    assert cfg.mujoco_joints == (0, 1, 2, 3, 4)
    control.prepare_episode_reset((0,))()
    cfg.mujoco_joints = (1,)
    with pytest.raises(RuntimeError, match="explicit selector"):
        control.initialize_model(SimpleNamespace(joint_count=5), result)


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


@pytest.mark.parametrize("representation", ["native", "mapping", "full_mapping"])
@pytest.mark.parametrize("density,spacing", [(1000.0, 0.0015), (800.0, 0.003)])
def test_solver_defaults_allow_fluid_owned_density_and_spacing(
    representation, density, spacing, monkeypatch
):
    from dataclasses import asdict
    from unittest.mock import Mock
    from dexsim.engine.newton_physics import DexUniSolverCfg, SPHSolverCfg
    from embodichain.lab.sim import SimulationManager

    sph = SPHSolverCfg(sph_method="dfsph")
    if representation == "native":
        if "sph_cfg" not in getattr(DexUniSolverCfg, "__dataclass_fields__", {}):
            pytest.skip("This DexSim build does not compose SPH through DexUni.")
        solver = DexUniSolverCfg(sph_cfg=sph)
    elif representation == "mapping":
        solver = {"solver_type": "dexuni", "sph_cfg": {"sph_method": "dfsph"}}
    else:
        solver = {"solver_type": "dexuni", "sph_cfg": asdict(sph)}
    manager = _fluid_manager_for_renderer("no-render")
    manager.sim_config.physics_cfg.solver_cfg = solver
    monkeypatch.setattr(SimulationManager, "_get_process_renderer", lambda: None)
    monkeypatch.setattr("embodichain.lab.sim.sim_manager.FluidObject", Mock())
    cfg = FluidObjectCfg(
        uid="water",
        density=density,
        particle_spacing=spacing,
        spawn=FluidSpawnCfg(positions=[[0.0, 0.0, 0.0]]),
        render={"mode": "none"},
    )

    SimulationManager.add_fluid_object(manager, cfg)

    descriptor = manager._spawn_scene.declare.call_args.args[2]
    assert descriptor.density == density
    assert descriptor.particle_spacing == spacing
    if representation != "mapping":
        values = asdict(sph) if representation == "native" else solver["sph_cfg"]
        assert values["rest_density"] == SPHSolverCfg().rest_density
        assert values["particle_diameter"] == SPHSolverCfg().particle_diameter


@pytest.mark.parametrize("representation", ["native", "mapping"])
@pytest.mark.parametrize("conflict", ["rest_density", "particle_diameter", None])
def test_solver_nondefault_physical_values_must_match_fluid(
    representation, conflict, monkeypatch
):
    from dataclasses import asdict
    from unittest.mock import Mock
    from dexsim.engine.newton_physics import DexUniSolverCfg, SPHSolverCfg
    from embodichain.lab.sim import SimulationManager

    density, spacing = 800.0, 0.003
    sph = SPHSolverCfg(
        sph_method="dfsph", rest_density=density, particle_diameter=spacing
    )
    if conflict is not None:
        setattr(sph, conflict, 1200.0 if conflict == "rest_density" else 0.02)
    if representation == "mapping":
        solver = {"solver_type": "dexuni", "sph_cfg": asdict(sph)}
    else:
        if "sph_cfg" not in getattr(DexUniSolverCfg, "__dataclass_fields__", {}):
            pytest.skip("This DexSim build does not compose SPH through DexUni.")
        solver = DexUniSolverCfg(sph_cfg=sph)
    manager = _fluid_manager_for_renderer("no-render")
    manager.sim_config.physics_cfg.solver_cfg = solver
    monkeypatch.setattr(SimulationManager, "_get_process_renderer", lambda: None)
    monkeypatch.setattr("embodichain.lab.sim.sim_manager.FluidObject", Mock())
    cfg = FluidObjectCfg(
        uid="water",
        density=density,
        particle_spacing=spacing,
        spawn=FluidSpawnCfg(positions=[[0.0, 0.0, 0.0]]),
        render={"mode": "none"},
    )

    if conflict is None:
        SimulationManager.add_fluid_object(manager, cfg)
        manager._spawn_scene.declare.assert_called_once()
    else:
        with pytest.raises(ValueError, match=f"Fluid {conflict} conflicts"):
            SimulationManager.add_fluid_object(manager, cfg)
        manager._spawn_scene.declare.assert_not_called()


def test_manager_resolves_source_once_without_rewriting_config(monkeypatch):
    from unittest.mock import Mock
    from types import SimpleNamespace
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.cfg import RigidObjectCfg
    from embodichain.lab.sim.spawn.descriptors import _pose_from_cfg

    points = np.array([[0.01, 0.0, 0.0]], dtype=np.float32)
    sampled = []

    def sample(source, spacing):
        sampled.append(spacing)
        return points.copy()

    monkeypatch.setattr(FluidSpawnCfg, "sample", sample)
    monkeypatch.setattr(SimulationManager, "_get_process_renderer", lambda: None)
    factory = Mock()
    monkeypatch.setattr("embodichain.lab.sim.sim_manager.FluidObject", factory)
    manager = _fluid_manager_for_renderer("no-render")
    reference = RigidObjectCfg(
        uid="cup", init_pos=(0.2, 0.3, 0.4), init_rot=(10.0, 20.0, 30.0)
    )
    manager._rigid_objects["cup"] = SimpleNamespace(cfg=reference)
    cfg = FluidObjectCfg(
        uid="water",
        init_pos=(0.01, 0.02, 0.03),
        init_rot=(15.0, 25.0, 35.0),
        spawn=FluidSpawnCfg(kind="box", reference_uid="cup"),
        render={"mode": "none"},
    )
    before = cfg.to_dict()
    expected_pose = _pose_from_cfg(reference) @ _pose_from_cfg(cfg)

    fluid = SimulationManager.add_fluid_object(manager, cfg)

    assert sampled == [cfg.particle_spacing]
    assert cfg.to_dict() == before
    assert len(factory.call_args.args) == 2
    assert isinstance(factory.call_args.args[0], FluidObjectCfg)
    assert factory.call_args.args[1] == manager.device
    assert factory.call_args.kwargs == {}
    fluid._initialize_spawn_source.assert_called_once()
    args = fluid._initialize_spawn_source.call_args
    np.testing.assert_array_equal(args.args[0], points)
    np.testing.assert_allclose(args.args[1], expected_pose[:3, :3])
    assert args.kwargs["on_write"] is manager._invalidate_fluid_state
    descriptor = manager._spawn_scene.declare.call_args.args[2]
    np.testing.assert_array_equal(descriptor.positions, points)
    np.testing.assert_allclose(descriptor.pose, expected_pose)
