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

"""A push control must not erase another pullable part in the same USD asset."""

from __future__ import annotations

from pathlib import Path
import hashlib
from types import SimpleNamespace as NS

from pxr import Usd, UsdGeom, UsdPhysics
import pytest

from embodichain.lab.sim.scene.articulation_inputs import (
    discover_prismatic_parts,
)


def _asset(tmp_path: Path, name: str, *, with_handle: bool = False) -> dict:
    path = tmp_path / "assembly.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    base = UsdGeom.Xform.Define(stage, "/base").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(base)
    for part, handled in (("rack", True), (name, with_handle)):
        body = UsdGeom.Xform.Define(stage, f"/{part}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        joint = UsdPhysics.PrismaticJoint.Define(stage, f"/{part}_joint")
        joint.CreateBody0Rel().SetTargets([base.GetPath()])
        joint.CreateBody1Rel().SetTargets([body.GetPath()])
        joint.CreateAxisAttr("X")
        joint.CreateLowerLimitAttr(0.0)
        joint.CreateUpperLimitAttr(0.2)
        if handled:
            grip = UsdGeom.Mesh.Define(stage, f"/{part}/pull_grip").GetPrim()
            UsdPhysics.CollisionAPI.Apply(grip).CreateCollisionEnabledAttr(True)
    stage.GetRootLayer().Save()
    return {"fpath": str(path), "fix_base": True}


@pytest.mark.parametrize(
    "name", ["button", "push_button", "press", "plunger", "switch"]
)
def test_handleless_push_controls_do_not_remove_pullable_parts(
    tmp_path: Path, name: str
) -> None:
    parts = discover_prismatic_parts(_asset(tmp_path, name))
    assert len(parts) == 1
    assert parts[0].link == "rack"


def test_handleless_noncontrol_remains_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one handle"):
        discover_prismatic_parts(_asset(tmp_path, "buttonish"))


def test_named_control_with_a_real_pull_handle_is_not_discarded(tmp_path: Path) -> None:
    parts = discover_prismatic_parts(_asset(tmp_path, "switch", with_handle=True))
    assert {part.link for part in parts} == {"rack", "switch"}


@pytest.mark.parametrize("qualified", [True, False])
def test_mass_only_lineage_preserves_part_identity_but_not_asset_hash(
    tmp_path: Path, monkeypatch, qualified: bool
) -> None:
    from embodichain.lab.sim.scene import (
        articulation_inputs as binding,
    )
    from embodichain.lab.sim.scene import control_mass_source as twist_mass_source

    config = _asset(tmp_path, "button")
    runtime = Path(config["fpath"])
    source = tmp_path / "pristine.usda"
    source.write_bytes(runtime.read_bytes())
    original = discover_prismatic_parts({**config, "fpath": str(source)})
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    stage = Usd.Stage.Open(str(runtime))
    UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath("/base")).CreateMassAttr(0.1)
    stage.GetRootLayer().Save()
    runtime.with_suffix(".mass.json").write_text("{}")

    def qualify(path):
        assert Path(path) == runtime
        if not qualified:
            raise ValueError("stale mass lineage")
        return NS(source_sha256=source_hash)

    monkeypatch.setattr(twist_mass_source, "qualify_mass_source_copy", qualify)
    if not qualified:
        with pytest.raises(ValueError, match="stale mass lineage"):
            discover_prismatic_parts(config)
        return
    parts = discover_prismatic_parts(config)
    assert [p.part_id for p in parts] == [p.part_id for p in original]
    monkeypatch.setattr(binding, "handle_mesh", lambda *args: None)
    bound = binding._binding_from_candidate({**config, "uid": "control"}, parts[0])
    assert bound.source_sha256 == hashlib.sha256(runtime.read_bytes()).hexdigest()
    assert bound.source_sha256 != source_hash


def test_spatial_reference_uses_catalog_body_not_fixed_grasp_child(monkeypatch):
    import numpy as np
    from embodichain.lab.sim.scene import spatial_reference as owner

    body_pose = np.eye(4)
    body_pose[:3, 3] = [1.0, 2.0, 0.0]
    geometry = NS(
        root_link="base",
        link_vertices={
            "drawer": np.array([[1.0, 2.0, 0.0], [2.0, 2.0, 0.0], [1.0, 3.0, 0.0]]),
            "grasp_child": np.full((3, 3), 100.0),
        },
        link_poses={"drawer": body_pose, "grasp_child": np.eye(4)},
    )
    monkeypatch.setattr(owner, "read_articulation_geometry", lambda path: geometry)
    monkeypatch.setattr(
        owner,
        "inspect_prismatic_part",
        lambda config, part: NS(link="grasp_child", axis=(1.0, 0.0, 0.0)),
    )
    monkeypatch.setattr(
        owner,
        "discover_prismatic_parts",
        lambda config: [NS(part_id="selected", link="drawer")],
    )
    config = dict(
        fpath="explicit.usda",
        body_scale=[2.0, 3.0, 1.0],
        init_pos=[10.0, 20.0, 0.0],
        init_rot=[0.0, 0.0, 0.0],
    )
    result = owner.read_spatial_reference(config, "selected")
    assert result.link == "drawer"
    assert np.array_equal(
        result.vertices, [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]
    )
    assert np.array_equal(result.initial_pose[:3, 3], [12.0, 26.0, 0.0])
    assert not result.vertices.flags.writeable
    config["init_rot"] = [0.0, 90.0, 0.0]
    with pytest.raises(ValueError, match="horizontal"):
        owner.read_spatial_reference(config, "selected")


def test_spatial_reference_reads_owned_scaled_root_geometry_from_actual_usd(tmp_path):
    import numpy as np
    from ._control_source_fixture import knob_source
    from embodichain.lab.sim.scene.spatial_reference import read_spatial_reference

    config = knob_source(tmp_path)
    config.update(body_scale=[2.0, 3.0, 1.0], init_pos=[10.0, 20.0, 0.8])
    result = read_spatial_reference(config, None)
    assert result.link == "base"
    assert np.allclose(result.vertices[:, 0].min(), -0.2)
    assert np.allclose(result.vertices[:, 0].max(), 0.2)
    assert np.allclose(result.vertices[:, 1].min(), -0.3)
    assert np.allclose(result.vertices[:, 1].max(), 0.3)
    assert np.array_equal(result.initial_pose[:3, 3], [10.0, 20.0, 0.8])
    assert not result.vertices.flags.writeable
    assert not result.initial_pose.flags.writeable
