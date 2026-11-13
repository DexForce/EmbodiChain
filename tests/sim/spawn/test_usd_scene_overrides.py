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

"""Whole-scene USD overrides are validated before native declaration."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from dexsim.spawn import (
    MeshObjectDesc,
    ConstraintDesc,
    RigidBodyPhysicsDesc,
    UsdSourceDesc,
    SceneDesc,
    SoftBodyDesc,
    ClothDesc,
    GeometryDesc,
    ArticulationDesc,
    LinkDesc,
)
from dexsim.types import ActorType

from embodichain.lab.sim.sim_manager import SimulationManager
from embodichain.lab.sim.spawn.scene import SpawnScene
from embodichain.lab.sim.cfg import UsdImportCfg, UsdSourceCfg

pytestmark = pytest.mark.no_sim


def _manager():
    manager = object.__new__(SimulationManager)
    manager.device = torch.device("cpu")
    manager.physics = SimpleNamespace(name="default")
    manager.sim_config = SimpleNamespace(render_cfg=SimpleNamespace(renderer="hybrid"))
    for name in (
        "_lights",
        "_sensors",
        "_robots",
        "_visual_objects",
        "_rigid_objects",
        "_rigid_object_groups",
        "_deformable_objects",
        "_fluid_objects",
        "_articulations",
    ):
        setattr(manager, name, {})
    manager.add_sensor = MagicMock()
    scene = object.__new__(SpawnScene)
    scene._assets = {}
    scene._num_envs = 1
    scene.builder = MagicMock(is_finalized=False, result=None)
    scene.builder.add_scene.side_effect = lambda source, **kwargs: list(
        source.iter_entities()
    )
    scene._resolve_materials = lambda materials: materials
    scene.track = MagicMock()
    manager._spawn_scene = scene
    manager.notify_visualization_topology_changed = MagicMock()
    return manager


def _source():
    base = MeshObjectDesc(
        name="base",
        usd=UsdSourceDesc(prim_path="/World/base"),
        physics=RigidBodyPhysicsDesc(actor_type=ActorType.KINEMATIC),
    )
    base.pose[0, 3] = 2
    child = MeshObjectDesc(
        name="child",
        usd=UsdSourceDesc(prim_path="/World/child"),
        physics=RigidBodyPhysicsDesc(actor_type=ActorType.DYNAMIC),
    )
    constraint = ConstraintDesc(name="joint", body0="base", body1="child")
    scene = SceneDesc(
        mesh_objects=[base, child],
        articulations=[],
        constraints=[constraint],
        materials={},
        rigid_collision_filter_pairs=[("base", "child")],
    )
    return scene


@pytest.mark.parametrize("representation", ["mesh", "gaussian_splat"])
def test_source_inventory_compiles_without_world_or_facades(
    monkeypatch, representation
):
    """Source inspection preserves appearance without runtime admission/binding."""
    from dexsim.spawn import GeometryType, RenderDesc
    from embodichain.lab.sim.spawn.usd_import import _compile_usd_import
    import dexsim.kit.usd

    path = "/World/asset"
    render = RenderDesc(
        geometry_type=(
            GeometryType.GAUSSIAN_SPLAT
            if representation == "gaussian_splat"
            else GeometryType.MESH
        ),
        file_path=(
            "appearance.ply" if representation == "gaussian_splat" else "appearance.obj"
        ),
        usd=UsdSourceDesc(prim_path=path + "/appearance"),
    )
    descriptor = MeshObjectDesc(
        name="asset",
        usd=UsdSourceDesc(prim_path=path),
        renders=[render],
        physics=RigidBodyPhysicsDesc(actor_type=ActorType.DYNAMIC, mass=0.25),
    )
    source = SceneDesc(mesh_objects=[descriptor])
    monkeypatch.setattr(dexsim.kit.usd, "parse_usd", lambda *args, **kwargs: source)
    plan = _compile_usd_import(
        None,
        UsdImportCfg(
            name="scene",
            source=UsdSourceCfg(file_path="source.usda"),
            asset_cfgs={path: {"uid": "object"}},
        ),
    )
    assert plan.resource_uids == {path: "object"}
    assert plan.resource_types == {path: "rigid_object"}
    assert plan.declarations[0][3] is None
    assert plan.source.mesh_objects[0].physics.mass == pytest.approx(0.25)
    assert plan.source_visual_elements[path][0]["representation"] == representation
    assert (
        plan.source_visual_elements[path][0]["source_prim_path"] == path + "/appearance"
    )
    assert plan.source.mesh_objects[0].renders == [render]


def test_sparse_override_keeps_actor_and_pose_and_remaps_constraint(monkeypatch):
    import dexsim.kit.usd

    source = _source()
    monkeypatch.setattr(dexsim.kit.usd, "parse_usd", lambda *args, **kwargs: source)
    manager = _manager()

    assets = manager.add_usd(
        UsdImportCfg(
            name="room",
            source=UsdSourceCfg(file_path="scene.usda"),
            asset_cfgs={
                "/World/base": {
                    "asset_physics_mode": "overlay",
                    "attrs": {"mass_props": {"mass": 5}},
                },
                "/World/child": {"uid": "door"},
            },
        )
    )

    assert set(assets) == {"/World/base", "/World/child"}
    assert assets["/World/base"].cfg.body_type == "kinematic"
    assert assets["/World/base"].cfg.init_local_pose[0, 3] == 2
    assert source.mesh_objects[0].physics.mass == 5
    assert assets["/World/child"].uid == "door"
    assert source.constraints[0].body0 == "base"
    assert source.constraints[0].body1 == "door"
    assert source.rigid_collision_filter_pairs == [("base", "door")]
    manager._spawn_scene.builder.add_scene.assert_called_once()


def test_unknown_override_target_is_rejected_before_declaration(monkeypatch):
    import dexsim.kit.usd

    source = _source()
    monkeypatch.setattr(dexsim.kit.usd, "parse_usd", lambda *args, **kwargs: source)
    manager = _manager()

    with pytest.raises(ValueError, match="missing"):
        manager.add_usd(
            UsdImportCfg(
                name="room",
                source=UsdSourceCfg(file_path="scene.usda"),
                asset_cfgs={"/World/missing": {}},
            )
        )
    manager._spawn_scene.builder.add_scene.assert_not_called()


def test_dexsim_scene_parser_keeps_standalone_constraint_after_uid_override(tmp_path):
    path = tmp_path / "joint.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    for name in ("base", "child"):
        body = UsdGeom.Xform.Define(stage, f"/World/{name}")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        cube = UsdGeom.Cube.Define(stage, f"/World/{name}/mesh")
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    joint = UsdPhysics.FixedJoint.Define(stage, "/World/joint")
    joint.CreateBody0Rel().SetTargets(["/World/base"])
    joint.CreateBody1Rel().SetTargets(["/World/child"])
    stage.GetRootLayer().Save()

    manager = _manager()
    imported = manager.add_usd(
        UsdImportCfg(
            name="room",
            source=UsdSourceCfg(file_path=str(path)),
            asset_cfgs={"/World/child": {"uid": "door"}},
        )
    )
    from embodichain.lab.sim import UsdImportResult

    assert isinstance(imported, UsdImportResult)

    scene = manager._spawn_scene.builder.add_scene.call_args.args[0]
    assert len(scene.constraints) == 1
    assert scene.constraints[0].body0 == "base"
    assert scene.constraints[0].body1 == "door"
    assert imported.constraint_uids["/World/joint"] == "joint"
    assert imported.resource_uids["/World/child"] == "door"
    assert imported.resource_types["/World/joint"] == "constraint"
    assert imported.resource_types["/World/child"] == "rigid_object"


@pytest.mark.parametrize(
    "descriptor_type,facade_name",
    [(SoftBodyDesc, "VolumeDeformableObject"), (ClothDesc, "SurfaceDeformableObject")],
)
def test_particle_import_registers_existing_facade_and_preserves_authored_physics(
    monkeypatch, descriptor_type, facade_name
):
    import dexsim.kit.usd

    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    descriptor = descriptor_type(
        name="authored",
        mesh=GeometryDesc.mesh(
            vertices=points, triangles=np.array([[0, 1, 2]], dtype=np.int32)
        ),
        positions=points,
        usd=UsdSourceDesc(prim_path="/World/Soft"),
    )
    descriptor.physics.density = 321.0
    source = SceneDesc(particle_sets=[descriptor])
    monkeypatch.setattr(dexsim.kit.usd, "parse_usd", lambda *args, **kwargs: source)
    manager = _manager()
    manager.device = torch.device("cuda:0")
    manager.physics = SimpleNamespace(name="newton", solver_type="dexuni")
    manager.sim_config = SimpleNamespace(
        physics_cfg=SimpleNamespace(requires_grad=False)
    )

    result = manager.add_usd(
        UsdImportCfg(
            name="test",
            source=UsdSourceCfg(file_path="scene.usda"),
            asset_cfgs={"/World/Soft": {"uid": "renamed", "particle_flags": 0}},
        )
    )

    asset = result["/World/Soft"]
    assert type(asset).__name__ == facade_name
    assert manager._deformable_objects["renamed"] is asset
    assert asset.cfg.source.source_prim_path == "/World/Soft"
    assert descriptor.physics.density == 321.0
    assert descriptor.particle_flags == 0
    assert isinstance(descriptor.particle_flags, int)
    np.testing.assert_array_equal(descriptor.positions, points)


def test_import_recipe_selects_subtree_and_namespaces_runtime_identity(tmp_path):
    path = tmp_path / "two_cells.usda"
    stage = Usd.Stage.CreateNew(str(path))
    for side in ("Left", "Right"):
        body = UsdGeom.Xform.Define(stage, f"/World/{side}/Body")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        mesh = UsdGeom.Cube.Define(stage, f"/World/{side}/Body/mesh")
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    stage.GetRootLayer().Save()
    original = path.read_bytes()
    manager = _manager()

    first = manager.add_usd(
        UsdImportCfg(
            name="left",
            source=UsdSourceCfg(file_path=str(path), root_prim_path="/World/Left"),
            namespace="a",
        )
    )
    second = manager.add_usd(
        UsdImportCfg(
            name="right",
            source=UsdSourceCfg(file_path=str(path), root_prim_path="/World/Right"),
            namespace="b",
        )
    )

    assert list(first) == ["/World/Left/Body"]
    assert list(second) == ["/World/Right/Body"]
    assert first["/World/Left/Body"].uid == "a::Body"
    assert second["/World/Right/Body"].uid == "b::Body"
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="already exists"):
        manager.add_usd(
            UsdImportCfg(
                name="duplicate",
                source=UsdSourceCfg(file_path=str(path), root_prim_path="/World/Left"),
                namespace="a",
            )
        )
    assert manager._spawn_scene.builder.add_scene.call_count == 2


@pytest.mark.parametrize(
    "parent,sensor_uid,renderer,sensor_type,valid",
    [
        ("base", "wrist", "hybrid", "Camera", True),
        ("base", "wrist", "hybrid", "ExternalCamera", True),
        ("missing", "wrist", "hybrid", "Camera", False),
        ("base", "Robot", "hybrid", "Camera", False),
        ("base", "wrist", "no-render", "Camera", False),
        ("base", "wrist", "hybrid", "Unsupported", False),
        ("base", "wrist", "hybrid", "ContactSensor", False),
        ("base", "wrist", "hybrid", "StereoCamera", False),
    ],
)
def test_robot_metadata_preserves_source_identity_and_preflights_sensor_binding(
    monkeypatch, parent, sensor_uid, renderer, sensor_type, valid
):
    import dexsim.kit.usd
    from embodichain.lab.sim.cfg import RobotCfg, ArticulationRootPropertiesCfg
    from embodichain.lab.sim.sensors import (
        Camera,
        CameraCfg,
        SensorRegistration,
        register_sensor,
    )
    from embodichain.lab.sim.sensors import registry
    from embodichain.utils import configclass

    camera_cfg_type = CameraCfg
    if sensor_type == "ExternalCamera":

        @configclass
        class ExternalCameraCfg(CameraCfg):
            sensor_type: str = "ExternalCamera"

        class ExternalCamera(Camera):
            pass

        monkeypatch.setattr(registry, "_registrations", registry._registrations.copy())
        register_sensor(
            SensorRegistration(
                "ExternalCamera", ExternalCameraCfg, ExternalCamera, legacy_factory=True
            )
        )
        camera_cfg_type = ExternalCameraCfg

    descriptor = ArticulationDesc(
        name="Robot",
        links=[
            LinkDesc(name="base", parent_name=None, pose=np.eye(4, dtype=np.float32))
        ],
        fixed_base=False,
        usd=UsdSourceDesc(prim_path="/Robot"),
    )
    descriptor.pose[0, 3] = 2
    monkeypatch.setattr(
        dexsim.kit.usd,
        "parse_usd",
        lambda *args, **kwargs: SceneDesc(articulations=[descriptor]),
    )
    metadata = RobotCfg(
        uid="metadata_uid",
        init_pos=(9, 0, 0),
        root_props=ArticulationRootPropertiesCfg(
            fixed_base=True,
            sleep_threshold=0.123,
            min_position_iters=16,
            min_velocity_iters=4,
        ),
        build_pk_chain=False,
        enable_sensor=True,
        sensors=[
            camera_cfg_type(
                uid=sensor_uid,
                sensor_type=sensor_type,
                extrinsics=CameraCfg.ExtrinsicsCfg(parent=parent),
            )
        ],
        joint_drive_props=None,
    )
    recipe = UsdImportCfg(
        name="test",
        source=UsdSourceCfg(file_path="scene.usda"),
        robot_cfgs={"/Robot": metadata},
    )
    manager = _manager()

    manager.sim_config.render_cfg.renderer = renderer
    if valid:
        result = manager.add_usd(recipe)
        assert result["/Robot"].uid == "Robot"
        assert result["/Robot"].cfg.init_local_pose[0, 3] == 2
        assert result["/Robot"].cfg.root_props.fixed_base is False
        root_props = result["/Robot"].cfg.root_props
        assert (
            root_props.sleep_threshold,
            root_props.min_position_iters,
            root_props.min_velocity_iters,
        ) == (0.123, 16, 4)
        assert manager.add_sensor.call_args.args[0].extrinsics.parent == "Robot/base"
        assert metadata.uid == "metadata_uid"
        assert metadata.sensors[0].extrinsics.parent == "base"
    else:
        with pytest.raises(
            (ValueError, RuntimeError, TypeError),
            match="unknown link|already exists|requires a native renderer|Unsupported mounted|Unsupported sensor type|StereoCamera requires",
        ):
            manager.add_usd(recipe)
        manager._spawn_scene.builder.add_scene.assert_not_called()
        manager.add_sensor.assert_not_called()
        assert manager._robots == {}
        if renderer == "no-render" or sensor_type != "Camera":
            recipe.robot_cfgs["/Robot"].enable_sensor = False
            result = manager.add_usd(recipe)
            assert manager._robots == {"Robot": result["/Robot"]}
            manager.add_sensor.assert_not_called()


@pytest.mark.parametrize("typed", [False, True])
def test_static_collider_overrides_keep_identity_pose_and_relationships(
    monkeypatch, typed
):
    import dexsim.kit.usd
    from dexsim.spawn import CollisionDesc, ConstraintDesc
    from embodichain.lab.sim.cfg import RigidObjectCfg

    descriptor = MeshObjectDesc(
        name="wall", collisions=[CollisionDesc()], usd=UsdSourceDesc(prim_path="/Wall")
    )
    descriptor.body_scale = np.array([2, 3, 4], dtype=np.float32)
    descriptor.pose[0, 3] = 2
    constraint = ConstraintDesc(name="anchor", body0="wall", body1=None)
    source = SceneDesc(mesh_objects=[descriptor], constraints=[constraint])
    monkeypatch.setattr(dexsim.kit.usd, "parse_usd", lambda *args, **kwargs: source)
    override = {
        "uid": "renamed",
        "init_pos": (1, 0, 0),
        "source_pose_mode": "offset",
        "body_type": "dynamic",
        "body_scale": (1, 1, 1),
    }
    manager = _manager()
    result = manager.add_usd(
        UsdImportCfg(
            name="room",
            source=UsdSourceCfg(file_path="scene.usda"),
            asset_cfgs={
                "/Wall": (RigidObjectCfg.from_dict(override) if typed else override)
            },
        )
    )
    assert descriptor.name == result["/Wall"].uid == "renamed"
    assert manager._rigid_objects == {"renamed": result["/Wall"]}
    assert result.resource_uids["/Wall"] == "renamed"
    assert constraint.body0 == "renamed"
    assert descriptor.pose[0, 3] == result["/Wall"].cfg.init_local_pose[0, 3] == 3
    assert descriptor.physics is None
    assert result["/Wall"].cfg.body_type == "static"
    np.testing.assert_array_equal(result["/Wall"].cfg.body_scale, [2, 3, 4])


@pytest.mark.parametrize("representation", ["sparse", "typed", "serialized"])
def test_identity_override_preserves_source_actor_and_nonunit_scale(
    monkeypatch, representation
):
    import dexsim.kit.usd
    from embodichain.lab.sim.cfg import RigidObjectCfg

    descriptor = MeshObjectDesc(
        name="body",
        usd=UsdSourceDesc(prim_path="/Body"),
        physics=RigidBodyPhysicsDesc(actor_type=ActorType.KINEMATIC),
        body_scale=np.array([2, 3, 4], dtype=np.float32),
    )
    monkeypatch.setattr(
        dexsim.kit.usd,
        "parse_usd",
        lambda *args, **kwargs: SceneDesc(mesh_objects=[descriptor]),
    )
    override = (
        {"uid": "renamed"}
        if representation == "sparse"
        else RigidObjectCfg(uid="renamed", body_type=None)
    )
    recipe = UsdImportCfg(
        name="scene",
        source=UsdSourceCfg(file_path="scene.usda"),
        asset_cfgs={"/Body": override},
    )
    if representation == "serialized":
        recipe = UsdImportCfg.from_dict(recipe.to_dict())
    result = _manager().add_usd(recipe)
    assert result["/Body"].cfg.body_type == "kinematic"
    np.testing.assert_array_equal(result["/Body"].cfg.body_scale, [2, 3, 4])
    assert descriptor.physics.actor_type == ActorType.KINEMATIC


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("explicit_matrix", [False, True])
def test_scene_override_rotation_matches_intrinsic_xyz(
    monkeypatch, typed, explicit_matrix
):
    import dexsim.kit.usd
    from scipy.spatial.transform import Rotation
    from embodichain.lab.sim.cfg import RigidObjectCfg

    descriptor = MeshObjectDesc(
        name="body",
        usd=UsdSourceDesc(prim_path="/Body"),
        physics=RigidBodyPhysicsDesc(actor_type=ActorType.DYNAMIC),
    )
    monkeypatch.setattr(
        dexsim.kit.usd,
        "parse_usd",
        lambda *args, **kwargs: SceneDesc(mesh_objects=[descriptor]),
    )
    fields = {"init_rot": (30, 40, 50)}
    expected = np.eye(4)
    expected[:3, :3] = Rotation.from_euler(
        "XYZ", (30, 40, 50), degrees=True
    ).as_matrix()
    if explicit_matrix:
        expected[:3, :3] = Rotation.from_euler("Z", 17, degrees=True).as_matrix()
        expected[:3, 3] = (1, 2, 3)
        fields["init_local_pose"] = expected.tolist()
    override = RigidObjectCfg(**fields) if typed else fields
    result = _manager().add_usd(
        UsdImportCfg(
            name="scene",
            source=UsdSourceCfg(file_path="scene.usda"),
            asset_cfgs={"/Body": override},
        )
    )
    np.testing.assert_allclose(result["/Body"].cfg.init_local_pose, expected, atol=1e-6)


def test_rigid_uid_override_remaps_newton_body_references_only(monkeypatch):
    import dexsim.kit.usd
    from dexsim.spawn import NewtonSceneDesc, NewtonEqualityDesc
    from collections import Counter
    from dexsim.spawn.adapters.newton_mujoco_scene import _validate_mujoco_references

    source = SceneDesc(
        mesh_objects=[
            MeshObjectDesc(
                name=name,
                usd=UsdSourceDesc(prim_path="/" + name),
                physics=RigidBodyPhysicsDesc(actor_type=ActorType.DYNAMIC),
            )
            for name in ["A", "B"]
        ],
        newton=NewtonSceneDesc(
            body_order=["A", "B", "link"],
            equalities=[
                NewtonEqualityDesc(
                    name="weld", equality_type=1, object1="A", object2="B", data=()
                ),
                NewtonEqualityDesc(
                    name="joint", equality_type=2, object1="A", object2="B", data=()
                ),
            ],
        ),
    )
    monkeypatch.setattr(dexsim.kit.usd, "parse_usd", lambda *args, **kwargs: source)
    result = _manager().add_usd(
        UsdImportCfg(
            name="scene",
            source=UsdSourceCfg(file_path="scene.usda"),
            asset_cfgs={"/A": {"uid": "new_A"}},
        )
    )
    assert result["/A"].uid == "new_A"
    assert source.newton.body_order == ["new_A", "B", "link"]
    assert (
        source.newton.equalities[0].object1,
        source.newton.equalities[0].object2,
    ) == ("new_A", "B")
    assert (
        source.newton.equalities[1].object1,
        source.newton.equalities[1].object2,
    ) == ("A", "B")
    _validate_mujoco_references(
        source.newton,
        Counter({"new_A": 1, "B": 1}),
        Counter({"A": 1, "B": 1}),
        Counter(),
    )


def test_visual_rename_does_not_rewrite_newton_articulation_body_names(monkeypatch):
    import dexsim.kit.usd
    from dexsim.spawn import NewtonSceneDesc, NewtonEqualityDesc

    visual = MeshObjectDesc(name="A", usd=UsdSourceDesc(prim_path="/Visual"))
    robot = ArticulationDesc(
        name="Robot",
        links=[
            LinkDesc(
                name="A",
                parent_name=None,
                pose=np.eye(4, dtype=np.float32),
                rigid_body=RigidBodyPhysicsDesc(actor_type=ActorType.DYNAMIC),
            )
        ],
        usd=UsdSourceDesc(prim_path="/Robot"),
    )
    source = SceneDesc(
        mesh_objects=[visual],
        articulations=[robot],
        newton=NewtonSceneDesc(
            body_order=["A"],
            equalities=[
                NewtonEqualityDesc(
                    name="weld", equality_type=1, object1="A", object2=None, data=()
                )
            ],
        ),
    )
    monkeypatch.setattr(dexsim.kit.usd, "parse_usd", lambda *args, **kwargs: source)
    result = _manager().add_usd(
        UsdImportCfg(
            name="scene",
            source=UsdSourceCfg(file_path="scene.usda"),
            asset_cfgs={"/Visual": {"uid": "new_visual"}},
        )
    )
    assert result["/Visual"].uid == "new_visual"
    assert robot.links[0].name == "A"
    assert source.newton.body_order == ["A"]
    assert source.newton.equalities[0].object1 == "A"


@pytest.mark.parametrize(
    "override,preserve_source",
    [
        ({}, True),
        ({"init_pos": (0, 0, 0)}, False),
        ({"init_rot": (0, 0, 0)}, False),
        ({"init_local_pose": np.eye(4)}, False),
        ({"source_pose_mode": "config"}, False),
        ({"source_pose_mode": "preserve", "init_pos": (0, 0, 0)}, True),
    ],
    ids=[
        "omitted",
        "zero-position",
        "zero-rotation",
        "identity-matrix",
        "config-mode",
        "preserve-mode",
    ],
)
def test_sparse_override_preserves_explicit_zero_placement(
    monkeypatch, override, preserve_source
):
    import dexsim.kit.usd
    from scipy.spatial.transform import Rotation

    authored = np.eye(4, dtype=np.float32)
    authored[:3, :3] = Rotation.from_euler("Z", 17, degrees=True).as_matrix()
    authored[0, 3] = 2
    descriptor = MeshObjectDesc(
        name="body",
        pose=authored.copy(),
        usd=UsdSourceDesc(prim_path="/Body"),
        physics=RigidBodyPhysicsDesc(actor_type=ActorType.DYNAMIC),
    )
    monkeypatch.setattr(
        dexsim.kit.usd,
        "parse_usd",
        lambda *args, **kwargs: SceneDesc(mesh_objects=[descriptor]),
    )
    result = _manager().add_usd(
        UsdImportCfg(
            name="scene",
            source=UsdSourceCfg(file_path="scene.usda"),
            asset_cfgs={"/Body": override},
        )
    )
    expected = authored if preserve_source else np.eye(4)
    np.testing.assert_allclose(result["/Body"].cfg.init_local_pose, expected, atol=1e-6)


@pytest.mark.parametrize("curve_count", [1, 2])
def test_curve_prim_resources_keep_all_uids_and_stable_source_keys(
    tmp_path, curve_count
):
    path = tmp_path / "cables.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    curves = UsdGeom.BasisCurves.Define(stage, "/World/Cable")
    curves.CreateTypeAttr("linear")
    curves.CreateWrapAttr("nonperiodic")
    curves.CreateCurveVertexCountsAttr([2] * curve_count)
    curves.CreatePointsAttr(
        [
            (0, i, 1) if end == 0 else (1, i, 1)
            for i in range(curve_count)
            for end in (0, 1)
        ]
    )
    curves.GetPrim().SetMetadata(
        "apiSchemas",
        Sdf.TokenListOp.Create(prependedItems=["PhysicsCurvesDeformableSimAPI"]),
    )
    stage.GetRootLayer().Save()
    expected_keys = (
        ["/World/Cable"]
        if curve_count == 1
        else [f"/World/Cable#{i}" for i in range(curve_count)]
    )
    for namespace in (None, "copy"):
        manager = _manager()
        result = manager.add_usd(
            UsdImportCfg(
                name="scene",
                source=UsdSourceCfg(file_path=str(path)),
                namespace=namespace,
            )
        )
        declared = manager._spawn_scene.builder.add_scene.call_args.args[0].cables
        assert len(declared) == curve_count
        assert list(result.resource_uids) == expected_keys
        assert list(result.resource_uids.values()) == [cable.name for cable in declared]
        assert dict(result.resource_types) == {key: "cable" for key in expected_keys}
        assert dict(result.resource_source_paths) == {
            key: "/World/Cable" for key in expected_keys
        }
        assert len(result) == 0
        with pytest.raises(TypeError):
            result.resource_source_paths[expected_keys[0]] = "/Other"


def test_portable_sparse_articulation_pose_matrix_is_decoded(monkeypatch):
    import json
    import dexsim.kit.usd
    from scipy.spatial.transform import Rotation

    descriptor = ArticulationDesc(
        name="robot",
        links=[LinkDesc(name="base", parent_name=None, pose=np.eye(4))],
        usd=UsdSourceDesc(prim_path="/Robot"),
    )
    monkeypatch.setattr(
        dexsim.kit.usd,
        "parse_usd",
        lambda *args, **kwargs: SceneDesc(articulations=[descriptor]),
    )
    expected = np.eye(4)
    expected[:3, :3] = Rotation.from_euler(
        "XYZ", (20, 30, 40), degrees=True
    ).as_matrix()
    expected[:3, 3] = (1, 2, 3)
    recipe = UsdImportCfg.from_dict(
        json.loads(
            json.dumps(
                {
                    "name": "scene",
                    "source": {"file_path": "robot.usda"},
                    "asset_cfgs": {"/Robot": {"init_local_pose": expected.tolist()}},
                }
            )
        )
    )
    result = _manager().add_usd(recipe)
    np.testing.assert_allclose(
        result["/Robot"].cfg.init_local_pose, expected, atol=1e-6
    )
