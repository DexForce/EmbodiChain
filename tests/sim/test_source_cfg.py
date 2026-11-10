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

from pathlib import Path

import numpy as np

import pytest
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    ObjectBaseCfg,
    RigidObjectCfg,
    RobotCfg,
    UsdSourceCfg,
    UsdImportCfg,
    VisualObjectCfg,
    VolumeDeformableObjectCfg,
    SurfaceDeformableObjectCfg,
)
from embodichain.lab.sim.cfg.source import _asset_file_path
from embodichain.lab.sim.shapes import CubeCfg, MeshCfg
from embodichain.lab.sim.spawn.usd import rigid_desc_from_usd, visual_desc_from_usd

ASSET_CONFIGS = (VisualObjectCfg, RigidObjectCfg, ArticulationCfg, RobotCfg)
pytestmark = pytest.mark.no_sim


def test_import_recipe_round_trip_keeps_source_selection_and_sparse_overrides():
    data = {
        "name": "cell",
        "source": {
            "file_path": "cell.usda",
            "root_prim_path": "/World/Cell",
            "variants": {"/World/Cell": {"tool": "gripper"}},
        },
        "namespace": "cell_1",
        "asset_cfgs": {"/World/Cell/Box": {"uid": "payload"}},
    }
    cfg = UsdImportCfg.from_dict(data)
    cfg.validate_import()
    restored = UsdImportCfg.from_dict(cfg.to_dict())
    restored.validate_import()
    assert restored.source.parse_kwargs() == {
        "prim_path": "/World/Cell",
        "variants": {"/World/Cell": {"tool": "gripper"}},
    }
    assert restored.asset_cfgs == data["asset_cfgs"]
    assert restored.namespace == "cell_1"


def test_import_recipe_round_trip_retains_typed_robot_override():
    cfg = UsdImportCfg(
        name="scene",
        source=UsdSourceCfg(file_path="scene.usda"),
        asset_cfgs={
            "/Robot": RobotCfg(
                uid="robot", control_parts={"arm": ["joint"]}, build_pk_chain=False
            )
        },
    )

    restored = UsdImportCfg.from_dict(cfg.to_dict())

    assert isinstance(restored.asset_cfgs["/Robot"], RobotCfg)
    assert restored.asset_cfgs["/Robot"].control_parts == {"arm": ["joint"]}
    assert restored.asset_cfgs["/Robot"].init_local_pose is None
    assert (
        restored.asset_cfgs["/Robot"].asset_physics_mode
        == cfg.asset_cfgs["/Robot"].asset_physics_mode
    )


@pytest.mark.parametrize(
    "source",
    [
        UsdSourceCfg(file_path="scene.usda", source_prim_path="/World/Cup"),
        UsdSourceCfg(file_path="scene.usda", pose_mode="offset"),
        UsdSourceCfg(file_path="scene.usda", namespace="misplaced"),
    ],
)
def test_scene_recipe_rejects_singleton_or_import_options_in_source(source):
    with pytest.raises(ValueError, match="source|namespace|placement"):
        UsdImportCfg(name="scene", source=source).validate_import()


def test_source_root_and_legacy_subtree_cannot_disagree():
    with pytest.raises(ValueError, match="root_prim_path"):
        UsdSourceCfg(
            file_path="scene.usda", root_prim_path="/A", prim_path="/B"
        ).validate_source()


@pytest.mark.parametrize("cfg_type", ASSET_CONFIGS)
def test_source_configuration_round_trip(cfg_type):
    data = {
        "uid": "runtime_instance",
        "source": {"file_path": "scene.usda", "source_prim_path": "/World/Cup"},
    }
    cfg = cfg_type.from_dict(data)
    cfg.validate()
    assert isinstance(cfg.source, UsdSourceCfg)
    assert _asset_file_path(cfg) == "scene.usda"
    restored = cfg_type.from_dict(cfg.to_dict())
    assert restored.source.file_path == data["source"]["file_path"]
    assert restored.source.source_prim_path == data["source"]["source_prim_path"]
    assert restored.source.pose_mode == "auto"
    assert restored.uid == "runtime_instance"
    assert restored.init_local_pose is None
    assert not hasattr(cfg, "prim_path")
    assert not hasattr(ObjectBaseCfg(), "source")


def test_usd_source_requires_assembled_file_path():
    source = UsdSourceCfg()
    with pytest.raises(TypeError, match="file_path"):
        source.validate()


@pytest.mark.parametrize("cfg_type", [VisualObjectCfg, RigidObjectCfg])
def test_mesh_asset_requires_geometry_or_source(cfg_type):
    with pytest.raises(ValueError, match="either shape or source"):
        _asset_file_path(cfg_type(uid="empty"))


@pytest.mark.parametrize("cfg_type", ASSET_CONFIGS)
def test_source_rejects_competing_file_owners(cfg_type):
    kwargs = {"source": UsdSourceCfg(file_path="scene.usd")}
    if cfg_type in (VisualObjectCfg, RigidObjectCfg):
        kwargs["shape"] = CubeCfg()
    else:
        kwargs["fpath"] = "robot.urdf"
    with pytest.raises(ValueError, match="cannot be combined"):
        _asset_file_path(cfg_type(**kwargs))


@pytest.mark.parametrize("file_path", ["", "mesh.obj", None, 24])
def test_source_rejects_invalid_usd_file(file_path):
    cfg = VisualObjectCfg(source=UsdSourceCfg(file_path=file_path))
    with pytest.raises(ValueError, match="file_path"):
        _asset_file_path(cfg)


@pytest.mark.parametrize(
    "selector",
    ["", "/", "World/Cup", "/World/Cup.color", "/World/*", "/World/Cup{v=a}"],
)
def test_source_selector_requires_an_exact_absolute_prim(selector):
    cfg = VisualObjectCfg(
        source=UsdSourceCfg(file_path="scene.usda", source_prim_path=selector)
    )
    with pytest.raises(ValueError, match="absolute USD prim path"):
        _asset_file_path(cfg)


@pytest.mark.parametrize("cfg_type", ASSET_CONFIGS)
@pytest.mark.parametrize("old_field", ["prim_path", "source_prim_path"])
def test_top_level_selectors_have_an_actionable_error(cfg_type, old_field):
    with pytest.raises(ValueError, match="source=UsdSourceCfg"):
        cfg_type.from_dict({old_field: "/World/Cup"})


def test_source_rejects_unknown_options():
    with pytest.raises(ValueError, match="Unknown UsdSourceCfg fields"):
        UsdSourceCfg.from_dict({"file_path": "scene.usda", "unknown": "/World/Cup"})


def test_source_options_keep_exact_entity_selection_separate_from_stage_selection():
    source = UsdSourceCfg(
        file_path="scene.usda",
        source_prim_path="/World/Cup",
        prim_path="/World",
        variants={"/World": {"size": "large"}},
        namespace="room",
        pose_mode="offset",
    )
    assert source.parse_kwargs() == {
        "prim_path": "/World",
        "variants": {"/World": {"size": "large"}},
        "namespace": "room",
    }
    assert source.source_prim_path == "/World/Cup"


@pytest.mark.parametrize("placement_owner", ["asset", "legacy_source"])
def test_usd_pose_mode_preserves_or_offsets_authored_transform(placement_owner):
    import numpy as np
    from embodichain.lab.sim.spawn.usd import _source_pose

    authored = np.eye(4, dtype=np.float32)
    authored[0, 3] = 2
    cfg = RigidObjectCfg(
        uid="cup",
        source=UsdSourceCfg(file_path="scene.usda"),
        init_pos=(1, 0, 0),
    )
    if placement_owner == "asset":
        cfg.source_pose_mode = "offset"
    else:
        cfg.source.pose_mode = "offset"
    assert _source_pose(cfg, authored)[0, 3] == 3
    if placement_owner == "asset":
        cfg.source_pose_mode = "preserve"
    else:
        cfg.source.pose_mode = "preserve"
    assert _source_pose(cfg, authored)[0, 3] == 2


def _cups(tmp_path: Path, names: tuple[str, ...]) -> Path:
    path = tmp_path / "cups.usda"
    stage = Usd.Stage.CreateNew(str(path))
    for name in names:
        prim = UsdGeom.Xform.Define(stage, f"/World/{name}").GetPrim()
        prim.CreateAttribute("MeshObject", Sdf.ValueTypeNames.Token)
        mesh = UsdGeom.Cube.Define(stage, f"/World/{name}/mesh").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.CollisionAPI.Apply(mesh)
    stage.GetRootLayer().Save()
    return path


def test_source_exact_path_distinguishes_repeated_leaf_names(tmp_path):
    path = _cups(tmp_path, ("Left/Cup", "Right/Cup"))
    source = UsdSourceCfg(file_path=str(path), source_prim_path="/World/Right/Cup")
    visual, _ = visual_desc_from_usd(VisualObjectCfg(uid="appearance", source=source))
    rigid, _ = rigid_desc_from_usd(RigidObjectCfg(uid="physical", source=source))
    assert visual.usd.prim_path == rigid.usd.prim_path == "/World/Right/Cup"
    assert visual.name == "appearance"
    assert rigid.name == "physical"
    assert visual.physics is None
    assert rigid.physics is not None


def test_visual_source_defaults_preserve_authored_pose_and_scale(tmp_path):
    import numpy as np
    from itertools import product

    path = _cups(tmp_path, ("Only",))
    stage = Usd.Stage.Open(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    body = UsdGeom.Xform(stage.GetPrimAtPath("/World/Only"))
    body.AddTranslateOp().Set((2.0, 0.0, 0.0))
    body.AddScaleOp().Set((2.0, 2.0, 2.0))
    stage.GetRootLayer().Save()
    cfg = VisualObjectCfg.from_dict(
        {
            "uid": "appearance",
            "source": {"file_path": str(path), "root_prim_path": "/World/Only"},
        }
    )

    descriptor, _ = visual_desc_from_usd(cfg)

    np.testing.assert_allclose(descriptor.pose[:3, 3], (2.0, 0.0, 0.0))
    visual = descriptor.renders[0]
    corners = np.array(list(product((-0.5, 0.5), repeat=3))) * np.asarray(visual.size)
    transform = (
        descriptor.pose @ np.diag([*descriptor.body_scale, 1]) @ visual.local_transform
    )
    world = corners @ transform[:3, :3].T + transform[:3, 3]
    np.testing.assert_allclose(world.min(axis=0), (0.0, -2.0, -2.0), atol=1e-6)
    np.testing.assert_allclose(world.max(axis=0), (4.0, 2.0, 2.0), atol=1e-6)
    assert descriptor.physics is None


def test_source_omitted_selector_requires_one_candidate(tmp_path):
    path = _cups(tmp_path, ("Only",))
    descriptor, _ = visual_desc_from_usd(
        VisualObjectCfg(uid="one", source=UsdSourceCfg(file_path=str(path)))
    )
    assert descriptor.usd.prim_path == "/World/Only"


def test_source_omitted_selector_rejects_ambiguity(tmp_path):
    path = _cups(tmp_path, ("One", "Two"))
    with pytest.raises(ValueError, match="exactly one.*found 2"):
        visual_desc_from_usd(
            VisualObjectCfg(uid="ambiguous", source=UsdSourceCfg(file_path=str(path)))
        )


def test_legacy_mesh_file_input_remains_available(tmp_path):
    path = _cups(tmp_path, ("Only",))
    descriptor, _ = rigid_desc_from_usd(
        RigidObjectCfg(uid="legacy", shape=MeshCfg(fpath=str(path)))
    )
    assert descriptor.name == "legacy"
    assert descriptor.physics is not None


@pytest.mark.parametrize(
    "cfg_type,explicit_pose",
    [
        (cfg_type, True)
        for cfg_type in (
            *ASSET_CONFIGS,
            VolumeDeformableObjectCfg,
            SurfaceDeformableObjectCfg,
        )
    ]
    + [(VolumeDeformableObjectCfg, False), (SurfaceDeformableObjectCfg, False)],
)
def test_singleton_usd_round_trip_preserves_pose_intent(cfg_type, explicit_pose):
    pose = np.eye(4)
    pose[:3, 3] = (2.0, -1.0, 3.0)
    cfg = cfg_type(
        uid="asset",
        source=UsdSourceCfg(file_path="scene.usda"),
        init_local_pose=pose if explicit_pose else None,
    )
    restored = cfg_type.from_dict(cfg.to_dict())
    if explicit_pose:
        np.testing.assert_allclose(restored.init_local_pose, pose)
    else:
        assert restored.init_local_pose is None


@pytest.mark.parametrize(
    "kind", ["visual", "rigid", "articulation", "volume", "surface"]
)
def test_singleton_translation_captures_resolved_reset_pose(monkeypatch, kind):
    import embodichain.lab.sim.spawn.usd as adapter
    from dexsim.spawn import (
        ArticulationDesc,
        MeshObjectDesc,
        SoftBodyDesc,
        ClothDesc,
        RigidBodyPhysicsDesc,
        SceneDesc,
        GeometryDesc,
    )
    from dexsim.types import ActorType

    config_type, descriptor_type, translate, collection = {
        "visual": (
            VisualObjectCfg,
            MeshObjectDesc,
            adapter.visual_desc_from_usd,
            "mesh_objects",
        ),
        "rigid": (
            RigidObjectCfg,
            MeshObjectDesc,
            adapter.rigid_desc_from_usd,
            "mesh_objects",
        ),
        "articulation": (
            ArticulationCfg,
            ArticulationDesc,
            adapter.articulation_desc_from_usd,
            "articulations",
        ),
        "volume": (
            VolumeDeformableObjectCfg,
            SoftBodyDesc,
            adapter._deformable_desc_from_usd,
            "particle_sets",
        ),
        "surface": (
            SurfaceDeformableObjectCfg,
            ClothDesc,
            adapter._deformable_desc_from_usd,
            "particle_sets",
        ),
    }[kind]
    if kind in ("volume", "surface"):
        mesh = GeometryDesc.mesh(
            vertices=np.array(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
            ),
            triangles=np.array([[0, 1, 2]], dtype=np.int32),
        )
        descriptor = descriptor_type(name="authored", mesh=mesh)
    else:
        descriptor = descriptor_type(name="authored")
    descriptor.pose[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    descriptor.pose[:3, 3] = (2, -1, 3)
    if kind == "rigid":
        descriptor.physics = RigidBodyPhysicsDesc(actor_type=ActorType.DYNAMIC)
    scene = SceneDesc(**{collection: [descriptor]})
    monkeypatch.setattr(
        adapter, "_parse_singleton", lambda *args, **kwargs: (scene, descriptor)
    )
    cfg = config_type(
        uid="runtime",
        source=UsdSourceCfg(
            file_path="scene.usda",
            pose_mode="offset" if kind in ("volume", "surface") else "auto",
        ),
        source_pose_mode="auto" if kind in ("volume", "surface") else "offset",
        init_pos=(1, 2, -1),
    )
    authored = descriptor.pose.copy()
    expected = authored.copy()
    expected[:3, 3] += (1, 2, -1)
    resolved, _ = translate(cfg)
    np.testing.assert_allclose(resolved.pose, expected)
    np.testing.assert_allclose(cfg.init_local_pose, expected)
    resolved.pose[0, 3] += 10
    np.testing.assert_allclose(cfg.init_local_pose, expected)

    descriptor.pose = authored.copy()
    reused, _ = translate(cfg.copy())
    np.testing.assert_allclose(reused.pose, expected)


def test_source_placement_option_keeps_legacy_positional_shape():
    mesh = MeshCfg(fpath="box.obj")
    cfg = RigidObjectCfg(
        "box", (0, 0, 0), (0, 0, 0), None, mesh, source_pose_mode="offset"
    )
    assert isinstance(cfg.shape, MeshCfg)
    assert cfg.shape.fpath == "box.obj"
    assert cfg.source_pose_mode == "offset"


@pytest.mark.parametrize(
    "mode,expected_actor", [("preserve", "KINEMATIC"), ("overlay", "DYNAMIC")]
)
def test_legacy_saved_rigid_config_retains_preserve_actor(
    tmp_path, mode, expected_actor
):
    import json
    from dexsim.types import ActorType

    path = _cups(tmp_path, ("Body",))
    stage = Usd.Stage.Open(str(path))
    body = stage.GetPrimAtPath("/World/Body")
    UsdPhysics.RigidBodyAPI(body).CreateKinematicEnabledAttr(True)
    stage.GetRootLayer().Save()
    # Previous versions serialized dynamic even when the user selected no actor.
    saved = tmp_path / "legacy.json"
    saved.write_text(
        json.dumps(
            {
                "uid": "body",
                "shape": {"shape_type": "Mesh", "fpath": str(path)},
                "body_type": "dynamic",
                "body_scale": [1, 1, 1],
                "asset_physics_mode": mode,
                "init_pos": [0, 0, 0],
                "init_rot": [0, 0, 0],
                "init_local_pose": None,
            }
        )
    )
    cfg = RigidObjectCfg.from_dict(json.loads(saved.read_text()))
    descriptor, _ = rigid_desc_from_usd(cfg)
    assert descriptor.physics.actor_type == getattr(ActorType, expected_actor)


def test_singleton_constraint_checks_follow_the_selected_asset(tmp_path):
    path = _cups(tmp_path, ("A", "B", "C"))
    stage = Usd.Stage.Open(str(path))
    joint = UsdPhysics.FixedJoint.Define(stage, "/World/joint")
    joint.CreateBody0Rel().SetTargets(["/World/A"])
    joint.CreateBody1Rel().SetTargets(["/World/B"])
    stage.GetRootLayer().Save()
    independent, _ = rigid_desc_from_usd(
        RigidObjectCfg(
            uid="free",
            source=UsdSourceCfg(file_path=str(path), source_prim_path="/World/C"),
        )
    )
    assert independent.usd.prim_path == "/World/C"
    with pytest.raises(ValueError, match="standalone constraints"):
        rigid_desc_from_usd(
            RigidObjectCfg(
                uid="coupled",
                source=UsdSourceCfg(file_path=str(path), source_prim_path="/World/A"),
            )
        )
    visual, _ = visual_desc_from_usd(
        VisualObjectCfg(
            uid="appearance",
            source=UsdSourceCfg(file_path=str(path), source_prim_path="/World/B"),
        )
    )
    assert visual.physics is None
    assert visual.collisions == []


@pytest.mark.parametrize("source_field", ["source", "fpath"])
@pytest.mark.parametrize("configured_scale", [None, (2.0, 3.0, 4.0)])
def test_robot_source_scale_survives_portable_round_trip(
    tmp_path, monkeypatch, source_field, configured_scale
):
    import json
    from copy import deepcopy
    from dexsim.spawn import ArticulationDesc, SceneDesc
    import embodichain.lab.sim.spawn.usd as adapter

    path = tmp_path / "robot.usda"
    path.touch()
    file_fields = (
        {"source": UsdSourceCfg(file_path=str(path))}
        if source_field == "source"
        else {"fpath": str(path)}
    )
    original = RobotCfg(uid="robot", body_scale=configured_scale, **file_fields)
    serialized = json.loads(json.dumps(original.to_dict()))
    restored = RobotCfg.from_dict(serialized)
    if configured_scale is None:
        assert restored.body_scale is None
    else:
        np.testing.assert_allclose(restored.body_scale, configured_scale)
    authored = ArticulationDesc(name="authored", body_scale=np.array([1.5, 1.5, 1.5]))
    monkeypatch.setattr(
        adapter,
        "_parse_singleton",
        lambda *args, **kwargs: (SceneDesc(), deepcopy(authored)),
    )
    for cfg in (original, restored):
        descriptor, _ = adapter.articulation_desc_from_usd(cfg)
        np.testing.assert_allclose(
            descriptor.body_scale,
            [1.5] * 3 if configured_scale is None else configured_scale,
        )


def test_robot_decode_retains_preset_scale_and_urdf_defaults(tmp_path):
    from embodichain.utils import configclass

    @configclass
    class ScaledRobotCfg(RobotCfg):
        body_scale: tuple = (2.0, 3.0, 4.0)

    cfg = ScaledRobotCfg.from_dict({"source": {"file_path": "robot.usda"}})
    assert cfg.body_scale == (2.0, 3.0, 4.0)
    path = tmp_path / "robot.urdf"
    path.touch()
    assert RobotCfg.from_dict({"fpath": str(path)}).body_scale == (1.0, 1.0, 1.0)


def test_locomotion_preset_keeps_source_scale_unset_after_decode(tmp_path):
    from embodichain.robot_registry import resolve_robot_cfg

    path = tmp_path / "g1.usda"
    path.touch()
    cfg = resolve_robot_cfg("UnitreeG1", {"source": {"file_path": str(path)}})
    assert cfg.source.file_path == str(path)
    assert cfg.fpath is None
    assert cfg.body_scale is None
