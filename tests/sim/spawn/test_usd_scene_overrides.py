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
from pxr import Usd, UsdGeom, UsdPhysics

from dexsim.spawn import (
    MeshObjectDesc,
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


def _source(constraint_type):
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
    constraint = constraint_type()
    constraint.name = "joint"
    constraint.body0 = "base"
    constraint.body1 = "child"
    scene = SceneDesc(
        mesh_objects=[base, child],
        articulations=[],
        constraints=[constraint],
        materials={},
        rigid_collision_filter_pairs=[("base", "child")],
    )
    return scene


def test_sparse_override_keeps_actor_and_pose_and_remaps_constraint(monkeypatch):
    import dexsim.kit.usd
    import dexsim.spawn
    import embodichain.lab.sim.spawn.usd as usd_adapter

    class ConstraintDesc:
        pass

    source = _source(ConstraintDesc)
    monkeypatch.setattr(dexsim.spawn, "ConstraintDesc", ConstraintDesc, raising=False)
    monkeypatch.setattr(dexsim.kit.usd, "parse_usd", lambda *args, **kwargs: source)
    monkeypatch.setattr(usd_adapter, "_require_usd_import_contract", lambda: None)
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
    import dexsim.spawn
    import embodichain.lab.sim.spawn.usd as usd_adapter

    class ConstraintDesc:
        pass

    source = _source(ConstraintDesc)
    monkeypatch.setattr(dexsim.spawn, "ConstraintDesc", ConstraintDesc, raising=False)
    monkeypatch.setattr(dexsim.kit.usd, "parse_usd", lambda *args, **kwargs: source)
    monkeypatch.setattr(usd_adapter, "_require_usd_import_contract", lambda: None)
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
    "parent,sensor_uid,renderer,valid",
    [
        ("base", "wrist", "hybrid", True),
        ("missing", "wrist", "hybrid", False),
        ("base", "Robot", "hybrid", False),
        ("base", "wrist", "no-render", False),
    ],
)
def test_robot_metadata_preserves_source_identity_and_preflights_sensor_binding(
    monkeypatch, parent, sensor_uid, renderer, valid
):
    import dexsim.kit.usd
    from embodichain.lab.sim.cfg import RobotCfg, ArticulationRootPropertiesCfg
    from embodichain.lab.sim.sensors import CameraCfg

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
        root_props=ArticulationRootPropertiesCfg(fixed_base=True),
        build_pk_chain=False,
        enable_sensor=True,
        sensors=[
            CameraCfg(uid=sensor_uid, extrinsics=CameraCfg.ExtrinsicsCfg(parent=parent))
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
        assert manager.add_sensor.call_args.args[0].extrinsics.parent == "Robot/base"
        assert metadata.uid == "metadata_uid"
        assert metadata.sensors[0].extrinsics.parent == "base"
    else:
        with pytest.raises(
            (ValueError, RuntimeError),
            match="unknown link|already exists|requires a native renderer",
        ):
            manager.add_usd(recipe)
        manager._spawn_scene.builder.add_scene.assert_not_called()
        manager.add_sensor.assert_not_called()
        assert manager._robots == {}
        if renderer == "no-render":
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
    descriptor.pose[0, 3] = 2
    constraint = ConstraintDesc(name="anchor", body0="wall", body1=None)
    source = SceneDesc(mesh_objects=[descriptor], constraints=[constraint])
    monkeypatch.setattr(dexsim.kit.usd, "parse_usd", lambda *args, **kwargs: source)
    override = {"uid": "renamed", "init_pos": (1, 0, 0), "source_pose_mode": "offset"}
    manager = _manager()
    result = manager.add_usd(
        UsdImportCfg(
            name="room",
            source=UsdSourceCfg(file_path="scene.usda"),
            asset_cfgs={
                "/Wall": (
                    RigidObjectCfg.from_dict({**override, "body_type": None})
                    if typed
                    else override
                )
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
