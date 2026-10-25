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

from types import SimpleNamespace

import pytest
from dexsim.engine import BoxGeometry
from dexsim.spawn.descs import CollisionDesc, GeometryDesc
from dexsim.types import RigidBodyShape

from embodichain.lab.sim.objects import RigidObject


class _FakePhysicalBody:
    def __init__(self, geometries):
        self.geometries = geometries

    def get_shape_count(self):
        return len(self.geometries)

    def get_shape_geometry(self, shape_idx):
        return self.geometries[shape_idx]

    def get_shape_name(self, shape_idx):
        return f"collision_{shape_idx}"


class _UnavailableGeometryBody(_FakePhysicalBody):
    def get_shape_geometry(self, shape_idx):
        raise RuntimeError("SDF dispatch is unavailable")


class _FakePhysicalEntity:
    def __init__(self, geometries):
        self.physical_body = _FakePhysicalBody(geometries)

    def get_physical_body(self):
        return self.physical_body


class _DescriptorEntity:
    """New-style SpawnedRigidBody shape-query fixture."""

    def __init__(self, collision):
        self.object_desc = SimpleNamespace(
            collisions=[collision],
            body_scale=(1.0, 1.0, 1.0),
        )
        self._physics_binding = None

    def native(self):
        return None


def _box_geometry(half_extents):
    geometry = BoxGeometry()
    geometry.half_extents = half_extents
    return geometry


def test_get_collision_shapes_snapshots_physical_box_geometry():
    rigid_object = RigidObject.__new__(RigidObject)
    rigid_object.uid = "fixture"
    rigid_object._entities = [_FakePhysicalEntity([_box_geometry([0.1, 0.2, 0.3])])]

    shapes = rigid_object.get_collision_shapes()

    assert len(shapes) == 1
    assert shapes[0].name == "collision_0"
    assert shapes[0].shape_type == RigidBodyShape.BOX
    assert shapes[0].half_extents.tolist() == pytest.approx([0.1, 0.2, 0.3])


@pytest.mark.no_sim
def test_prepared_variant_reset_pose_accepts_gym_tensor_indices() -> None:
    """Gym reset IDs must select each row's configured asset initial pose."""
    import torch

    from embodichain.lab.sim.cfg import RigidObjectCfg

    fixture = SimpleNamespace(
        cfg=RigidObjectCfg(
            init_pos=(0.0, 0.0, 1.0),
            env_variants={1: RigidObjectCfg(init_pos=(0.0, 0.0, 2.0))},
        ),
        device=torch.device("cpu"),
    )

    poses = RigidObject._build_cfg_init_pose(fixture, torch.tensor([1, 0]))

    torch.testing.assert_close(
        poses[:, :3, 3], torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 1.0]])
    )


@pytest.mark.gpu
@pytest.mark.requires_sim
def test_native_compound_aperture_and_parent_lifecycle() -> None:
    """A falling peg traverses the aperture in both isolated arenas."""
    import numpy as np
    import torch

    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import (
        MassPropertiesCfg,
        NewtonPhysicsCfg,
        RenderCfg,
        RigidBodyPhysicsCfg,
        RigidObjectCfg,
    )
    from embodichain.lab.sim.shapes import CompoundShapeCfg, CubeCfg, ShapePartCfg

    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            startup_summary="off",
            enable_entity_gizmo=False,
            num_envs=2,
            physics_dt=0.001,
            physics_cfg=NewtonPhysicsCfg(
                use_cuda_graph=False,
                num_substeps=1,
                solver_cfg={"solver_type": "mujoco_warp"},
            ),
            render_cfg=RenderCfg(renderer="hybrid"),
        )
    )
    try:
        fixture = sim.add_rigid_object(
            RigidObjectCfg(
                uid="aperture",
                body_type="static",
                init_pos=(0, 0, 0.1),
                shape=CompoundShapeCfg(
                    parts=[
                        ShapePartCfg(
                            shape=CubeCfg(size=(0.01, 0.06, 0.02)),
                            position=(-0.02, 0, 0),
                        ),
                        ShapePartCfg(
                            shape=CubeCfg(size=(0.01, 0.06, 0.02)),
                            position=(0.02, 0, 0),
                        ),
                    ],
                    # Independent appearance must never become collision geometry.
                    visual_parts=[ShapePartCfg(shape=CubeCfg(size=(0.05, 0.06, 0.02)))],
                ),
            )
        )
        peg = sim.add_rigid_object(
            RigidObjectCfg(
                uid="peg",
                init_pos=(0, 0, 0.2),
                shape=CubeCfg(size=(0.008, 0.008, 0.02)),
                attrs=RigidBodyPhysicsCfg(
                    mass_props=MassPropertiesCfg(mass=0.1, inertia=(1e-5, 1e-5, 1e-5))
                ),
            )
        )
        sim.prepare()
        assert fixture.num_instances == 2
        assert len(fixture.get_collision_shapes()) == 2
        sim.update(step=200)
        # A hull spanning both walls would hold the peg above z=0.12 m.
        assert torch.all(peg.get_local_pose()[:, 2] < 0.05)
        torch.testing.assert_close(
            peg.get_mass(), torch.full((2,), 0.1, device=sim.device)
        )
        original_second_pose = peg.get_local_pose()[1].clone()
        peg.reset(env_ids=[0])
        torch.testing.assert_close(
            peg.get_local_pose()[0, :3],
            torch.tensor([0.0, 0.0, 0.2], device=sim.device),
        )
        torch.testing.assert_close(peg.get_local_pose()[1], original_second_pose)
        sim.remove_asset("aperture")
        sim.prepare()
        assert sim.get_rigid_object("aperture") is None
        assert np.isfinite(peg.get_local_pose().cpu().numpy()).all()
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


@pytest.mark.gpu
@pytest.mark.requires_sim
def test_native_raw_mujoco_contact_parameters_reach_solver() -> None:
    """Native contact arrays preserve raw values through model compilation."""
    import numpy as np
    from dataclasses import fields
    from dexsim.spawn import NewtonCollisionDesc

    if "solref" not in {item.name for item in fields(NewtonCollisionDesc)}:
        pytest.skip("Requires DexSim assembly contact patch.")

    from dexsim.engine.newton_physics.backend_registry import get_newton_backend
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import (
        NewtonCollisionPropertiesCfg,
        NewtonPhysicsCfg,
        RenderCfg,
        RigidBodyPhysicsCfg,
        RigidObjectCfg,
    )
    from embodichain.lab.sim.shapes import CubeCfg

    reference = (0.002, 1.0)
    impedance = (0.8, 0.99, 0.0001, 0.5, 2.0)
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            startup_summary="off",
            enable_entity_gizmo=False,
            physics_cfg=NewtonPhysicsCfg(
                use_cuda_graph=False,
                num_substeps=1,
                solver_cfg={"solver_type": "mujoco_warp"},
            ),
            render_cfg=RenderCfg(renderer="hybrid"),
        )
    )
    try:
        sim.add_rigid_object(
            RigidObjectCfg(
                uid="raw_contact",
                shape=CubeCfg(size=(0.01, 0.01, 0.01)),
                init_pos=(0, 0, 1),
                attrs=RigidBodyPhysicsCfg(
                    collision_props=NewtonCollisionPropertiesCfg(
                        margin=0, gap=0, solref=reference, solimp=impedance, solmix=0.7
                    )
                ),
            )
        )
        sim.prepare()
        backend = get_newton_backend(sim.get_world())
        # One ground geom followed by the cube, in a single MuJoCo world.
        np.testing.assert_allclose(
            backend.solver.mjw_model.geom_solref.numpy()[0, -1], reference, rtol=1e-6
        )
        np.testing.assert_allclose(
            backend.solver.mjw_model.geom_solimp.numpy()[0, -1], impedance, rtol=1e-6
        )
        assert backend.model.mujoco.solref_mode.numpy()[-1] == 1
        sim.update(step=1)
        assert np.isfinite(backend.state_0.body_q.numpy()).all()
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


@pytest.mark.gpu
@pytest.mark.requires_sim
@pytest.mark.parametrize("solver", ["mujoco_warp", "semi_implicit"])
def test_native_prepared_variants_keep_mass_reset_and_rebuild(solver: str) -> None:
    """Prepared variants bind real arena rows, retaining their own bodies."""
    from dataclasses import fields
    import torch
    from dexsim.spawn import ObjectDesc

    if "env_variants" not in {item.name for item in fields(ObjectDesc)}:
        pytest.skip("Requires DexSim assembly variants patch.")

    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import (
        MassPropertiesCfg,
        NewtonPhysicsCfg,
        RenderCfg,
        RigidBodyPhysicsCfg,
        RigidObjectCfg,
    )
    from embodichain.lab.sim.shapes import CompoundShapeCfg, CubeCfg, ShapePartCfg

    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            startup_summary="off",
            enable_entity_gizmo=False,
            num_envs=2,
            physics_cfg=NewtonPhysicsCfg(
                use_cuda_graph=False,
                num_substeps=1,
                solver_cfg={"solver_type": solver},
            ),
            render_cfg=RenderCfg(renderer="hybrid"),
        )
    )
    try:
        variant_shape = CubeCfg(size=(0.02, 0.03, 0.04))
        if solver == "semi_implicit":
            variant_shape = CompoundShapeCfg(
                parts=[
                    ShapePartCfg(shape=variant_shape),
                    ShapePartCfg(
                        shape=CubeCfg(size=(0.01, 0.01, 0.01)), position=(0.05, 0, 0)
                    ),
                ]
            )
        cfg = RigidObjectCfg(
            uid="logical",
            shape=CubeCfg(size=(0.01, 0.01, 0.01)),
            init_pos=(0, 0, 1),
            attrs=RigidBodyPhysicsCfg(mass_props=MassPropertiesCfg(mass=1)),
            env_variants={
                1: RigidObjectCfg(
                    uid="asset_B",
                    shape=variant_shape,
                    init_pos=(0, 0, 2),
                    attrs=RigidBodyPhysicsCfg(mass_props=MassPropertiesCfg(mass=2)),
                )
            },
        )
        body = sim.add_rigid_object(cfg)
        sim.prepare()
        expected_masses = torch.tensor([1.0, 2.0], device=sim.device)
        torch.testing.assert_close(body.get_mass(), expected_masses)
        assert "arena_0/logical" in sim.spawn_result.handles
        assert "arena_1/logical" in sim.spawn_result.handles
        assert len(body.get_collision_shapes(1, allow_heterogeneous=True)) == (
            2 if solver == "semi_implicit" else 1
        )
        sim.update(step=1)
        first_pose = body.get_local_pose()[0].clone()
        body.reset(env_ids=[1])
        torch.testing.assert_close(body.get_local_pose()[0], first_pose)
        assert body.get_local_pose()[1, 2] == 2.0
        sim.add_rigid_object(
            RigidObjectCfg(
                uid="extra", shape=CubeCfg(size=(0.01, 0.01, 0.01)), init_pos=(0, 0, 3)
            )
        )
        sim.prepare()
        torch.testing.assert_close(body.get_mass(), expected_masses)
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


def test_get_collision_shapes_rejects_batched_topology_mismatch():
    rigid_object = RigidObject.__new__(RigidObject)
    rigid_object.uid = "fixture"
    rigid_object._entities = [
        _FakePhysicalEntity([_box_geometry([0.1, 0.2, 0.3])]),
        _FakePhysicalEntity(
            [
                _box_geometry([0.1, 0.2, 0.3]),
                _box_geometry([0.4, 0.5, 0.6]),
            ]
        ),
    ]

    with pytest.raises(ValueError, match="different collision-shape topology"):
        rigid_object.get_collision_shapes()
    assert len(rigid_object.get_collision_shapes(0, allow_heterogeneous=True)) == 1
    assert len(rigid_object.get_collision_shapes(1, allow_heterogeneous=True)) == 2


def test_get_collision_shapes_reports_unavailable_sdf_geometry():
    rigid_object = RigidObject.__new__(RigidObject)
    rigid_object.uid = "sdf_object"
    entity = _FakePhysicalEntity([])
    entity.physical_body = _UnavailableGeometryBody([object()])
    rigid_object._entities = [entity]

    with pytest.raises(RuntimeError, match="canonical collision mesh"):
        rigid_object.get_collision_shapes()


def test_get_collision_shapes_reads_new_spawned_body_descriptor():
    collision = CollisionDesc.from_geometry(GeometryDesc.cube((0.2, 0.4, 0.6)))
    rigid_object = RigidObject.__new__(RigidObject)
    rigid_object.uid = "descriptor_fixture"
    rigid_object._entities = [_DescriptorEntity(collision)]

    shapes = rigid_object.get_collision_shapes()

    assert len(shapes) == 1
    assert shapes[0].shape_type == RigidBodyShape.BOX
    assert shapes[0].half_extents.tolist() == pytest.approx([0.1, 0.2, 0.3])
