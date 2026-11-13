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
"""Fill missing GLB visuals in the W1 USD without changing its physics."""

from __future__ import annotations

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[8]))

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
from scripts.benchmark.tech_reports.physics.experiments.dexuni.s04_common import OUTPUT

__all__ = ["main"]


def main() -> None:
    """Convert original GLB render meshes into existing physical link frames."""
    usd = OUTPUT / "inputs/w1_usd/robot/robot.usda"
    robot = OUTPUT / "inputs/w1_v030/robot.urdf"
    stage = Usd.Stage.Open(str(usd))
    root = stage.GetDefaultPrim()
    count = 0
    for link in ET.parse(robot).getroot().findall("link"):
        matches = [
            p
            for p in stage.Traverse()
            if p.GetName() == link.attrib["name"]
            and str(p.GetPath()).startswith(str(root.GetPath()) + "/Geometry/")
        ]
        if not matches:
            continue
        body = min(matches, key=lambda p: p.GetPath().pathElementCount)
        for i, visual in enumerate(link.findall("visual")):
            desc = visual.find("geometry/mesh")
            if desc is None or not desc.attrib["filename"].endswith(".glb"):
                continue
            asset = robot.parent / desc.attrib["filename"]
            scene = trimesh.load(asset, force="scene")
            parent = UsdGeom.Xform.Define(
                stage, body.GetPath().AppendChild(f"glb_visual_{i}")
            )
            origin = visual.find("origin")
            xyz = (
                np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
                if origin is not None
                else np.zeros(3)
            )
            rpy = (
                np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
                if origin is not None
                else np.zeros(3)
            )
            from scipy.spatial.transform import Rotation

            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
            transform[:3, 3] = xyz
            parent.MakeMatrixXform().Set(Gf.Matrix4d(transform.T.tolist()))
            scale = np.fromstring(desc.attrib.get("scale", "1 1 1"), sep=" ")
            for j, node in enumerate(scene.graph.nodes_geometry):
                matrix, geometry_name = scene.graph[node]
                geometry = scene.geometry[geometry_name]
                vertices = trimesh.transform_points(geometry.vertices, matrix) * scale
                mesh = UsdGeom.Mesh.Define(
                    stage, parent.GetPath().AppendChild(f"mesh_{j}")
                )
                mesh.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in vertices])
                mesh.CreateFaceVertexCountsAttr([3] * len(geometry.faces))
                mesh.CreateFaceVertexIndicesAttr(geometry.faces.reshape(-1).tolist())
                mesh.CreateSubdivisionSchemeAttr("none")
                mesh.CreateDoubleSidedAttr(True)
                mat = getattr(geometry.visual, "material", None)
                rgba = (
                    np.asarray(
                        getattr(mat, "baseColorFactor", (210, 170, 40, 255)),
                        dtype=float,
                    )
                    if mat is not None
                    else np.array((210, 170, 40, 255))
                )
                if rgba is None or rgba.size < 3:
                    rgba = np.array((210, 170, 40, 255))
                rgb = rgba[:3] / 255 if rgba[:3].max() > 1 else rgba[:3]
                mesh.CreateDisplayColorAttr([Gf.Vec3f(*map(float, rgb))])
                material = UsdShade.Material.Define(
                    stage, parent.GetPath().AppendChild(f"material_{j}")
                )
                shader = UsdShade.Shader.Define(
                    stage, material.GetPath().AppendChild("shader")
                )
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                    Gf.Vec3f(*map(float, rgb))
                )
                material.CreateSurfaceOutput().ConnectToSource(
                    shader.ConnectableAPI(), "surface"
                )
                UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
                count += 1
    stage.GetRootLayer().Save()
    print(
        f"Authored {count} original GLB visual meshes; physical USD layers unchanged."
    )


if __name__ == "__main__":
    main()
