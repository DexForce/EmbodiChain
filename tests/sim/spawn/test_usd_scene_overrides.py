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

import pytest
import torch
from pxr import Usd, UsdGeom, UsdPhysics

from dexsim.spawn import MeshObjectDesc, RigidBodyPhysicsDesc, UsdSourceDesc
from dexsim.types import ActorType

from embodichain.lab.sim.sim_manager import SimulationManager
from embodichain.lab.sim.spawn.scene import SpawnScene

pytestmark = pytest.mark.no_sim


def _manager():
    manager = object.__new__(SimulationManager)
    manager.device = torch.device("cpu")
    manager.physics = SimpleNamespace(name="default")
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
    scene = SimpleNamespace(
        mesh_objects=[base, child],
        articulations=[],
        constraints=[constraint],
        materials={},
        rigid_collision_filter_pairs=[("base", "child")],
    )
    scene.iter_entities = lambda: iter([base, child, constraint])
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
        "room",
        "scene.usda",
        asset_cfgs={
            "/World/base": {
                "asset_physics_mode": "overlay",
                "attrs": {"mass_props": {"mass": 5}},
            },
            "/World/child": {"uid": "door"},
        },
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
        manager.add_usd("room", "scene.usda", asset_cfgs={"/World/missing": {}})
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
    manager.add_usd("room", str(path), asset_cfgs={"/World/child": {"uid": "door"}})

    scene = manager._spawn_scene.builder.add_scene.call_args.args[0]
    assert len(scene.constraints) == 1
    assert scene.constraints[0].body0 == "base"
    assert scene.constraints[0].body1 == "door"
