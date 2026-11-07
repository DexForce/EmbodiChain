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
