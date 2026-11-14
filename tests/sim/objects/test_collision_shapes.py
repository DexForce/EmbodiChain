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
    assert RigidObject._build_cfg_init_pose(
        fixture, torch.tensor([], dtype=torch.long)
    ).shape == (0, 4, 4)


@pytest.mark.gpu
@pytest.mark.requires_sim
@pytest.mark.parametrize("solver", ["mujoco_warp", "dexuni", "default"])
def test_native_compound_aperture_and_parent_lifecycle(solver: str) -> None:
    """A falling peg traverses the aperture in both isolated arenas."""
    import numpy as np
    import torch

    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import (
        MassPropertiesCfg,
        DefaultPhysicsCfg,
        NewtonPhysicsCfg,
        RenderCfg,
        RigidBodyPhysicsCfg,
        RigidObjectCfg,
        VisualElementCfg,
    )
    from embodichain.lab.sim.shapes import CompoundShapeCfg, CubeCfg, ShapePartCfg

    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            startup_summary="off",
            enable_entity_gizmo=False,
            num_envs=2,
            physics_dt=0.001,
            physics_cfg=(
                DefaultPhysicsCfg()
                if solver == "default"
                else NewtonPhysicsCfg(
                    use_cuda_graph=False,
                    num_substeps=1,
                    solver_cfg={"solver_type": solver},
                )
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
                ),
                # Independent appearance must never become collision geometry.
                visuals=[
                    VisualElementCfg(representation=CubeCfg(size=(0.05, 0.06, 0.02)))
                ],
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
@pytest.mark.parametrize("solver", ["mujoco_warp", "dexuni"])
def test_native_raw_mujoco_contact_parameters_reach_solver(solver: str) -> None:
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
        NewtonRigidBodyMaterialCfg,
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
                solver_cfg={"solver_type": solver},
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
                    collision_props=NewtonCollisionPropertiesCfg(margin=0, gap=0),
                    material_props=NewtonRigidBodyMaterialCfg(
                        solref=reference, solimp=impedance, solmix=0.7
                    ),
                ),
            )
        )
        sim.prepare()
        backend = get_newton_backend(sim.get_world())
        assert backend.solver_type == solver
        mujoco_solver = backend.mujoco_solver
        assert mujoco_solver is not None
        # One ground geom followed by the cube, in a single MuJoCo world.
        np.testing.assert_allclose(
            mujoco_solver.mjw_model.geom_solref.numpy()[0, -1], reference, rtol=1e-6
        )
        np.testing.assert_allclose(
            mujoco_solver.mjw_model.geom_solimp.numpy()[0, -1], impedance, rtol=1e-6
        )
        assert backend.model.mujoco.solref_mode.numpy()[-1] == 1
        sim.update(step=1)
        assert np.isfinite(backend.state_0.body_q.numpy()).all()
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


@pytest.mark.gpu
@pytest.mark.requires_sim
@pytest.mark.parametrize("solver", ["mujoco_warp", "semi_implicit", "dexuni"])
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
        RigidObjectGroupCfg,
        VisualElementCfg,
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
        cfg.env_variants[1].visuals = [
            VisualElementCfg(
                id="outer", representation=CubeCfg(size=(0.08, 0.08, 0.08))
            )
        ]
        body = sim.add_rigid_object(cfg)
        member_cfg = cfg.copy()
        member_cfg.init_pos = (1, 0, 1)
        member_cfg.env_variants[1].init_pos = (1, 0, 2)
        group = sim.add_rigid_object_group(
            RigidObjectGroupCfg(uid="group", rigid_objects={"member": member_cfg})
        )
        sim.prepare()
        appearance = body.get_vertices(env_ids=[1], scale=True)
        torch.testing.assert_close(
            appearance.amax(1) - appearance.amin(1),
            torch.full((1, 3), 0.08, device=sim.device),
        )
        semantic, _ = body.get_semantic_mesh([1])
        expected_semantic_width = 0.065 if solver == "semi_implicit" else 0.02
        assert (
            semantic[:, :, 0].amax() - semantic[:, :, 0].amin()
        ).item() == pytest.approx(expected_semantic_width)
        expected_masses = torch.tensor([1.0, 2.0], device=sim.device)
        torch.testing.assert_close(body.get_mass(), expected_masses)
        torch.testing.assert_close(group.get_mass(), expected_masses[:, None])
        torch.testing.assert_close(
            group.get_local_pose()[:, 0, 2],
            torch.tensor([1.0, 2.0], device=sim.device),
        )
        assert "arena_0/logical" in sim.spawn_result.handles
        assert "arena_1/logical" in sim.spawn_result.handles
        assert len(body.get_collision_shapes(1, allow_heterogeneous=True)) == (
            2 if solver == "semi_implicit" else 1
        )
        sim.update(step=1)
        first_pose = body.get_local_pose()[0].clone()
        first_group_pose = group.get_local_pose()[0].clone()
        body.reset(env_ids=[1])
        group.reset(env_ids=[1])
        torch.testing.assert_close(body.get_local_pose()[0], first_pose)
        torch.testing.assert_close(group.get_local_pose()[0], first_group_pose)
        assert body.get_local_pose()[1, 2] == 2.0
        assert group.get_local_pose()[1, 0, 2] == 2.0
        sim.add_rigid_object(
            RigidObjectCfg(
                uid="extra", shape=CubeCfg(size=(0.01, 0.01, 0.01)), init_pos=(0, 0, 3)
            )
        )
        sim.prepare()
        torch.testing.assert_close(body.get_mass(), expected_masses)
        torch.testing.assert_close(group.get_mass(), expected_masses[:, None])
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


@pytest.mark.gpu
@pytest.mark.requires_sim
def test_native_dexuni_compound_and_profiles_in_mixed_scene(tmp_path) -> None:
    """Qualify independent equality profiles with a VBD-owned cylindrical peg."""
    import numpy as np
    import torch
    from dexsim.engine.newton_physics.backend_registry import get_newton_backend
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import (
        ArticulationCfg,
        ArticulationRootPropertiesCfg,
        JointDrivePropertiesCfg,
        MassPropertiesCfg,
        NewtonPhysicsCfg,
        RenderCfg,
        RigidBodyPhysicsCfg,
        RigidObjectCfg,
    )
    from embodichain.lab.sim.shapes import (
        CompoundShapeCfg,
        CubeCfg,
        CylinderCfg,
        ShapePartCfg,
    )

    path = tmp_path / "mimic.urdf"
    path.write_text("""<robot name="mimic_probe">
<link name="base"><inertial><mass value="1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial></link>
<link name="leader_link"><inertial><mass value="0.1"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial><visual><geometry><box size="0.02 0.02 0.02"/></geometry></visual><collision><geometry><box size="0.02 0.02 0.02"/></geometry></collision></link>
<link name="follower_link"><inertial><mass value="0.1"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial><visual><geometry><box size="0.02 0.02 0.02"/></geometry></visual><collision><geometry><box size="0.02 0.02 0.02"/></geometry></collision></link>
<joint name="leader" type="prismatic"><parent link="base"/><child link="leader_link"/><origin xyz="0 -0.1 0"/><axis xyz="1 0 0"/><limit lower="-0.5" upper="0.5" effort="100" velocity="1"/></joint>
<joint name="follower" type="prismatic"><parent link="base"/><child link="follower_link"/><origin xyz="0 0.1 0"/><axis xyz="1 0 0"/><limit lower="-0.5" upper="0.5" effort="100" velocity="1"/><mimic joint="leader" multiplier="1" offset="0"/></joint>
</robot>""")
    profiles = [
        ((0.003, 1.0), (0.8, 0.99, 0.0001, 0.5, 2.0)),
        ((0.007, 0.9), (0.6, 0.95, 0.0003, 0.4, 3.0)),
    ]
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            startup_summary="off",
            enable_entity_gizmo=False,
            scene_node_capacity=512,
            width=64,
            height=64,
            num_envs=2,
            physics_dt=0.001,
            physics_cfg=NewtonPhysicsCfg(
                use_cuda_graph=False,
                num_substeps=1,
                solver_cfg={"solver_type": "dexuni"},
            ),
            render_cfg=RenderCfg(renderer="hybrid"),
        )
    )
    arms = []
    fixture = peg = backend = mujoco = None
    try:
        for index, (reference, impedance) in enumerate(profiles):
            arms.append(
                sim.add_articulation(
                    ArticulationCfg(
                        uid=f"mechanism_{index}",
                        fpath=str(path),
                        asset_physics_mode="overlay",
                        init_pos=(2 + index, 0, 1),
                        init_qpos=[0, 0],
                        root_props=ArticulationRootPropertiesCfg(
                            fixed_base=True,
                            self_collision_enabled=False,
                            newton_mimic_solref=reference,
                            newton_mimic_solimp=impedance,
                        ),
                        joint_drive_props=JointDrivePropertiesCfg(
                            stiffness=20,
                            damping=2,
                            drive_type="force",
                            target_mode="position",
                        ),
                    )
                )
            )
        fixture = sim.add_rigid_object(
            RigidObjectCfg(
                uid="fixture",
                body_type="static",
                init_pos=(0, 0, 0.1),
                shape=CompoundShapeCfg(
                    parts=[
                        ShapePartCfg(
                            shape=CubeCfg(size=(0.01, 0.06, 0.02)), position=(x, 0, 0)
                        )
                        for x in (-0.02, 0.02)
                    ]
                ),
            )
        )
        peg = sim.add_rigid_object(
            RigidObjectCfg(
                uid="peg",
                init_pos=(0, 0, 0.2),
                shape=CylinderCfg(radius=0.004, half_height=0.01),
                attrs=RigidBodyPhysicsCfg(
                    mass_props=MassPropertiesCfg(mass=0.1, inertia=(1e-5, 1e-5, 1e-5))
                ),
            )
        )
        sim.prepare()
        backend = get_newton_backend(sim.get_world())
        assert backend.solver_type == "dexuni"
        features = sim.spawn_result.get_solver_diagnostics()["features"]
        assert features["mujoco_solve_enabled"] and features["vbd_solve_enabled"]
        assert features["vbd_dynamic_body_count"] == 2
        mujoco = backend.mujoco_solver
        reference_rows = mujoco.mjw_model.eq_solref.numpy().reshape(-1, 2)
        impedance_rows = mujoco.mjw_model.eq_solimp.numpy().reshape(-1, 5)
        for reference, impedance in profiles:
            rows = np.all(np.isclose(reference_rows, reference, rtol=1e-6), axis=1)
            assert rows.sum() == 2  # One equality per environment for this mechanism.
            np.testing.assert_allclose(
                impedance_rows[rows], np.tile(impedance, (2, 1)), rtol=1e-6
            )
        gains = backend.model.joint_target_ke.numpy()
        assert np.count_nonzero(np.isclose(gains, 20.0)) == 8
        assert not np.any(np.isclose(gains, 0.2))  # No opt-in follower stabilizer.
        assert len(fixture.get_collision_shapes()) == 2
        sim.update(step=200)
        z = peg.get_local_pose()[:, 2]
        # The peg must cross the aperture and stop on the ground, at half-height.
        assert torch.all((z > 0.005) & (z < 0.05))
        for arm in arms:
            assert torch.isfinite(arm.get_qpos()).all()
    finally:
        arm = fixture = peg = backend = mujoco = None
        arms.clear()
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


def test_compound_cylinder_query_preserves_parts_and_scaled_local_positions() -> None:
    from unittest.mock import Mock

    import numpy as np
    from scipy.spatial import ConvexHull
    from embodichain.lab.sim.cfg import RigidObjectCfg
    from embodichain.lab.sim.shapes import (
        CompoundShapeCfg,
        CubeCfg,
        CylinderCfg,
        ShapePartCfg,
    )
    from embodichain.lab.sim.spawn.descriptors import rigid_desc_from_cfg

    cfg = RigidObjectCfg(
        uid="fixture",
        body_type="static",
        body_scale=(2.0, 2.0, 2.0),
        shape=CompoundShapeCfg(
            parts=[
                ShapePartCfg(shape=CubeCfg(size=(0.2, 0.2, 0.2)), position=(1, 0, 0)),
                ShapePartCfg(
                    shape=CylinderCfg(radius=0.1, half_height=0.2), position=(-1, 0, 0)
                ),
            ]
        ),
    )
    desc, _ = rigid_desc_from_cfg(cfg)
    fallback = Mock(side_effect=AssertionError("compound shapes must not be merged"))
    entity = SimpleNamespace(
        object_desc=desc,
        native=lambda: None,
        _physics_binding=SimpleNamespace(get_collision_mesh=fallback),
    )
    obj = RigidObject.__new__(RigidObject)
    obj.uid = "fixture"
    obj._entities = [entity]
    shapes = obj.get_collision_shapes()

    assert len(shapes) == 2
    assert shapes[0].shape_type == RigidBodyShape.BOX
    assert shapes[1].shape_type == RigidBodyShape.CONVEX
    assert shapes[0].local_pose[0, 3].item() == pytest.approx(2.0)
    assert shapes[1].local_pose[0, 3].item() == pytest.approx(-2.0)
    assert shapes[0].half_extents.tolist() == pytest.approx([0.2, 0.2, 0.2])
    # The planner proxy circumscribes the analytic cylinder rather than
    # underestimating its radius between polygon vertices.
    hull = ConvexHull(shapes[1].vertices[:, :2].numpy())
    distances = -hull.equations[:, 2] / np.linalg.norm(hull.equations[:, :2], axis=1)
    assert distances.min() >= 0.2 - 1e-7
    fallback.assert_not_called()


@pytest.mark.parametrize("enabled", [True, False])
def test_compound_query_refuses_merged_fallback_for_unavailable_part(enabled) -> None:
    from unittest.mock import Mock

    from dexsim.spawn.descs import GeometryType

    missing = SimpleNamespace(
        geometry_type=GeometryType.MESH,
        vertices=None,
        triangles=None,
        enable_collision=enabled,
    )
    second = CollisionDesc.from_geometry(GeometryDesc.cube(size=(1, 1, 1)))
    second.enable_collision = enabled
    fallback = Mock()
    entity = SimpleNamespace(
        object_desc=SimpleNamespace(collisions=[missing, second], body_scale=(1, 1, 1)),
        native=lambda: None,
        _physics_binding=SimpleNamespace(get_collision_mesh=fallback),
    )
    obj = RigidObject.__new__(RigidObject)
    obj.uid, obj._entities = "compound", [entity]
    with pytest.raises(RuntimeError, match="refusing a merged collision mesh"):
        obj.get_collision_shapes()
    fallback.assert_not_called()


def test_disabled_contact_response_keeps_planner_geometry() -> None:
    collision = CollisionDesc.from_geometry(GeometryDesc.cube((0.2, 0.4, 0.6)))
    collision.enable_collision = False
    rigid_object = RigidObject.__new__(RigidObject)
    rigid_object.uid = "disabled_contact"
    rigid_object._entities = [_DescriptorEntity(collision)]
    shapes = rigid_object.get_collision_shapes()
    assert len(shapes) == 1
    assert shapes[0].shape_type == RigidBodyShape.BOX
    assert shapes[0].half_extents.tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_articulation_link_collision_reads_native_physical_shape_not_render_mesh():
    from embodichain.lab.sim.objects.backends import (
        collision_shapes_from_articulation_link,
    )

    geometry = _box_geometry([0.1, 0.2, 0.3])
    native = SimpleNamespace(
        get_physical_body=lambda name: (
            _FakePhysicalBody([geometry]) if name == "arm" else None
        )
    )
    entity = SimpleNamespace(native=lambda: native)
    shapes = collision_shapes_from_articulation_link(entity, "arm", object_uid="robot")
    assert shapes[0].shape_type == RigidBodyShape.BOX
    assert shapes[0].half_extents.tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert (
        collision_shapes_from_articulation_link(entity, "empty", object_uid="robot")
        == []
    )


def test_articulation_link_collision_reads_newton_named_physical_mesh():
    import numpy as np
    from embodichain.lab.sim.objects.backends import (
        collision_shapes_from_articulation_link,
    )

    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    triangles = np.array([[0, 1, 2]])
    entity = SimpleNamespace(
        native=lambda: None,
        _physics_binding=SimpleNamespace(
            get_collision_mesh=lambda name: (vertices, triangles)
        ),
    )
    shapes = collision_shapes_from_articulation_link(entity, "arm", object_uid="robot")
    vertices[:] = 99
    assert shapes[0].vertices[1, 0] == 1
    assert shapes[0].triangles.tolist() == [[0, 1, 2]]


def test_newton_link_keeps_convex_pieces_and_triangle_types():
    import torch
    from types import SimpleNamespace
    import numpy as np
    import newton
    from embodichain.lab.sim.objects.backends.collision import (
        collision_shapes_from_articulation_link,
    )

    mesh = SimpleNamespace(
        vertices=np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
        ),
        indices=np.array([0, 1, 2, 0, 3, 1, 0, 2, 3, 1, 3, 2]),
    )
    model = SimpleNamespace(
        shape_flags=np.array(
            [
                int(newton.ShapeFlags.COLLIDE_SHAPES),
                int(newton.ShapeFlags.COLLIDE_SHAPES),
            ]
        ),
        shape_type=np.array(
            [int(newton.GeoType.CONVEX_MESH), int(newton.GeoType.MESH)]
        ),
        shape_scale=np.array([[2, 3, 4], [1, 1, 1]], dtype=np.float32),
        shape_transform=np.array(
            [[0, 0, 0, 0, 0, 0, 1], [1, 2, 3, 0, 0, 0, 1]], dtype=np.float32
        ),
        shape_source=[mesh, mesh],
    )
    binding = SimpleNamespace(
        _runtime=SimpleNamespace(model=model),
        _links=[SimpleNamespace(name="arm", shape_ids=(0, 1))],
    )
    shapes = collision_shapes_from_articulation_link(
        SimpleNamespace(_physics_binding=binding), "arm", object_uid="robot"
    )
    assert len(shapes) == 2
    assert shapes[0].shape_type.name == "CONVEX" and shapes[1].shape_type.name == "MESH"
    torch.testing.assert_close(shapes[0].vertices[1], torch.tensor([2.0, 0.0, 0.0]))
    torch.testing.assert_close(
        shapes[1].local_pose[:3, 3], torch.tensor([1.0, 2.0, 3.0])
    )

    model.shape_flags[1] = 0
    enabled = collision_shapes_from_articulation_link(
        SimpleNamespace(_physics_binding=binding), "arm", object_uid="robot"
    )
    assert len(enabled) == 1 and enabled[0].shape_type.name == "CONVEX"
