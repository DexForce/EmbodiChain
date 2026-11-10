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
    VisualObjectCfg,
)
from embodichain.lab.sim.cfg.source import _asset_file_path
from embodichain.lab.sim.shapes import CubeCfg, MeshCfg
from embodichain.lab.sim.spawn.usd import rigid_desc_from_usd, visual_desc_from_usd

ASSET_CONFIGS = (VisualObjectCfg, RigidObjectCfg, ArticulationCfg, RobotCfg)


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
    assert restored.source.to_dict() == data["source"]
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
        UsdSourceCfg.from_dict({"file_path": "scene.usda", "prim_path": "/World/Cup"})


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
