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
"""Offline authored-collision transfer; synthetic geometry grants no physics credit."""

from __future__ import annotations

import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh
from pxr import Usd, UsdGeom, UsdPhysics

from embodichain_robots.agibot.g1 import _replay_collisions as recipe


def test_collision_recipe_rejects_other_exports_before_resolving_sources(
    tmp_path, monkeypatch
):
    source = tmp_path / "input.urdf"
    source.write_text('<robot name="another-export"><link name="base"/></robot>')

    def unexpected(*args):
        pytest.fail("Unrecognized exports must not resolve or download official assets")

    monkeypatch.setattr(recipe, "_source_bundle", unexpected)
    with pytest.raises(ValueError, match="pinned #1212"):
        recipe.prepare_replay_collisions(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_official_collision_bundle_checks_the_pinned_source_hashes(tmp_path):
    with pytest.raises(ValueError, match="source SHA256 mismatch"):
        recipe._source_bundle(tmp_path)


@pytest.mark.parametrize("wrong_link_frame", [False, True])
def test_authored_collision_transfer_preserves_input_and_link_frames(
    tmp_path, monkeypatch, wrong_link_frame
):
    source = tmp_path / "source"
    source.mkdir()
    mesh_file = source / "visual.obj"
    mesh_file.write_text("v 0 0 0\nv 0.01 0 0\nv 0 0.01 0\nf 1 2 3\n")
    original = ET.Element("robot", name="synthetic-observed-export")
    official = ET.Element("robot", name="synthetic-official-export")
    bundle = tmp_path / "official"
    bundle.mkdir()
    (bundle / "LICENSE").write_text("Synthetic fixture license\n")
    stage = Usd.Stage.CreateNew(str(bundle / "robot.usda"))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.Xform.Define(stage, "/G1")
    # The production recipe expects 14 arm cylinders and 22 hand meshes.
    for i in range(36):
        name = f"link{i}"
        link = ET.SubElement(original, "link", name=name)
        visual = ET.SubElement(link, "visual")
        ET.SubElement(ET.SubElement(visual, "geometry"), "mesh", filename="visual.obj")
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "mass", value="0.1")
        if i:
            joint = ET.SubElement(original, "joint", name=f"joint{i}", type="fixed")
            ET.SubElement(joint, "parent", link="link0")
            ET.SubElement(joint, "child", link=name)
            ET.SubElement(joint, "origin", xyz="0.1 0.2 0.3", rpy="0 0 0")
        authored = ET.SubElement(official, "link", name=name)
        collision = ET.SubElement(authored, "collision")
        geometry = ET.SubElement(collision, "geometry")
        if i < 14:
            ET.SubElement(geometry, "cylinder", radius="0.04", length="0.12")
            ET.SubElement(collision, "origin", xyz="0 0 0.05", rpy="0 0 0")
        else:
            ET.SubElement(geometry, "mesh", filename="package://official/hand.stl")
        usd_link = UsdGeom.Xform.Define(stage, f"/G1/{name}")
        usd_link.AddTranslateOp().Set((5, 6, 7))
        for kind in ("visuals", "collisions"):
            parent = UsdGeom.Xform.Define(stage, f"/G1/{name}/{kind}")
            if kind == "collisions":
                # Collision-local translation must apply once; link world position must not apply.
                parent.AddTranslateOp().Set((0, 0, 0.02))
                UsdPhysics.MeshCollisionAPI.Apply(
                    parent.GetPrim()
                ).CreateApproximationAttr("convexHull")
            mesh = UsdGeom.Mesh.Define(stage, f"/G1/{name}/{kind}/mesh")
            shift = 0.01 if wrong_link_frame and i == 0 and kind == "visuals" else 0
            mesh.CreatePointsAttr(
                [(shift, 0, 0), (0.01 + shift, 0, 0), (shift, 0.01, 0)]
            )
            mesh.CreateFaceVertexCountsAttr([3])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
            if kind == "collisions":
                mesh.CreateOrientationAttr("leftHanded")
    stage.GetRootLayer().Save()
    source_urdf = source / "robot.urdf"
    ET.ElementTree(original).write(source_urdf, encoding="unicode")
    ET.ElementTree(official).write(bundle / recipe._URDF_FILE, encoding="unicode")
    before = source_urdf.read_bytes()
    monkeypatch.setattr(recipe, "_PREPARED_SHA256", hashlib.sha256(before).hexdigest())
    monkeypatch.setattr(
        recipe, "_source_bundle", lambda _: (bundle, {"synthetic_fixture": True})
    )
    output = tmp_path / "prepared"
    if wrong_link_frame:
        with pytest.raises(ValueError, match="link frames disagree"):
            recipe.prepare_replay_collisions(source_urdf, output)
        assert not output.exists()
    else:
        repaired, report = recipe.prepare_replay_collisions(source_urdf, output)
        result = ET.parse(repaired).getroot()
        assert len(result.findall("./link/collision/geometry/cylinder")) == 14
        assert len(result.findall("./link/collision/geometry/mesh")) == 22
        cylinder = result.find("link[@name='link0']/collision")
        assert cylinder.find("geometry/cylinder").attrib == {
            "radius": "0.04",
            "length": "0.12",
        }
        assert cylinder.find("origin").get("xyz") == "0 0 0.05"
        mesh = trimesh.load(
            output / "collisions/link14_0.obj", force="mesh", process=False
        )
        np.testing.assert_allclose(
            mesh.vertices, [[0, 0, 0.02], [0.01, 0, 0.02], [0, 0.01, 0.02]], atol=1e-9
        )
        np.testing.assert_array_equal(mesh.faces, [[2, 1, 0]])
        for link in result.findall("link"):
            for collision in link.findall("collision"):
                link.remove(collision)
        assert ET.tostring(result) == ET.tostring(original)
        assert (output / "visual.obj").read_bytes() == mesh_file.read_bytes()
        assert (output / "LICENSE.agibot").read_bytes() == (
            bundle / "LICENSE"
        ).read_bytes()
        assert report["derived_from_visuals"] is False
        assert report["native_physics_verified"] is False
    assert source_urdf.read_bytes() == before
