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

"""Prepare the shared, fully authored L-02 inputs without starting a simulator."""

from __future__ import annotations
import argparse
import colorsys
import hashlib
import json
from pathlib import Path
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = []
    for shape, limit in [("cube", 16000), ("sphere", 1000)]:
        if shape == "cube":
            vertices = (
                np.array(
                    [
                        [-1, -1, -1],
                        [1, -1, -1],
                        [1, 1, -1],
                        [-1, 1, -1],
                        [-1, -1, 1],
                        [1, -1, 1],
                        [1, 1, 1],
                        [-1, 1, 1],
                    ],
                    dtype=float,
                )
                * 0.025
            )
            faces = np.array(
                [
                    [0, 2, 1],
                    [0, 3, 2],
                    [4, 5, 6],
                    [4, 6, 7],
                    [0, 1, 5],
                    [0, 5, 4],
                    [1, 2, 6],
                    [1, 6, 5],
                    [2, 3, 7],
                    [2, 7, 6],
                    [3, 0, 4],
                    [3, 4, 7],
                ]
            )
        else:
            # 24 longitudes, 11 interior rings: 2*24*11=528 triangles.
            vertices = [[0, 0, 0.025]]
            for j in range(1, 12):
                phi = np.pi * j / 12
                vertices.extend(
                    [
                        [
                            0.025 * np.sin(phi) * np.cos(2 * np.pi * k / 24),
                            0.025 * np.sin(phi) * np.sin(2 * np.pi * k / 24),
                            0.025 * np.cos(phi),
                        ]
                        for k in range(24)
                    ]
                )
            vertices.append([0, 0, -0.025])
            vertices = np.array(vertices)
            faces = []
            for k in range(24):
                faces.append([0, 1 + k, 1 + (k + 1) % 24])
                for j in range(10):
                    a = 1 + j * 24 + k
                    b = 1 + j * 24 + (k + 1) % 24
                    c = a + 24
                    d = b + 24
                    faces.extend([[a, c, d], [a, d, b]])
                faces.append([265, 241 + (k + 1) % 24, 241 + k])
            faces = np.array(faces)
        assert len(faces) == (12 if shape == "cube" else 528)
        # Correct every face to point away from the origin.
        tri = vertices[faces]
        inward = (
            np.einsum(
                "ij,ij->i",
                np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
                tri.mean(1),
            )
            < 0
        )
        faces[inward] = faces[inward][:, [0, 2, 1]]
        for index in range(-1, limit):
            mode = "S" if index < 0 else "U"
            scale = 1.0 if index < 0 else 0.9 + 0.2 * ((index * 997) % 16000) / 15999
            color = (
                (0.85, 0.23, 0.06)
                if index < 0
                else colorsys.hsv_to_rgb((index * 0.618033988749895) % 1, 0.75, 0.9)
            )
            name = f"{shape}_{mode}" + (f"_{index:05d}" if index >= 0 else "") + ".usda"
            path = args.output / name
            stage = Usd.Stage.CreateNew(str(path))
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            root = UsdGeom.Xform.Define(stage, "/Object").GetPrim()
            stage.SetDefaultPrim(root)
            UsdPhysics.RigidBodyAPI.Apply(root)
            mass = UsdPhysics.MassAPI.Apply(root)
            mass.CreateMassAttr(0.1)
            mass.CreateCenterOfMassAttr(Gf.Vec3f(0))
            mass.CreatePrincipalAxesAttr(Gf.Quatf(1))
            moment = 0.1 * (0.05 * scale) ** 2 / (6 if shape == "cube" else 10)
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(moment))
            mesh = UsdGeom.Mesh.Define(stage, "/Object/Visual")
            mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in vertices * scale])
            mesh.CreateFaceVertexCountsAttr([3] * len(faces))
            mesh.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            if shape == "sphere":
                mesh.CreateNormalsAttr([Gf.Vec3f(*(v / 0.025)) for v in vertices])
                mesh.SetNormalsInterpolation("vertex")
            if shape == "cube":
                collision = UsdGeom.Cube.Define(stage, "/Object/Collision")
                collision.CreateSizeAttr(0.05 * scale)
            else:
                collision = UsdGeom.Sphere.Define(stage, "/Object/Collision")
                collision.CreateRadiusAttr(0.025 * scale)
            collision.CreateVisibilityAttr("invisible")
            UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
            collision.GetPrim().AddAppliedSchema("PhysxCollisionAPI")
            collision.GetPrim().CreateAttribute(
                "physxCollision:restOffset", Sdf.ValueTypeNames.Float
            ).Set(0.0)
            collision.GetPrim().CreateAttribute(
                "physxCollision:contactOffset", Sdf.ValueTypeNames.Float
            ).Set(0.001)
            mat = UsdShade.Material.Define(stage, "/Object/Material")
            phys = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
            phys.CreateStaticFrictionAttr(0.8)
            phys.CreateDynamicFrictionAttr(0.8)
            phys.CreateRestitutionAttr(0.0)
            shader = UsdShade.Shader.Define(stage, "/Object/Material/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*color)
            )
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
            mat.CreateSurfaceOutput().ConnectToSource(
                shader.ConnectableAPI(), "surface"
            )
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
            UsdShade.MaterialBindingAPI.Apply(collision.GetPrim()).Bind(
                mat, materialPurpose="physics"
            )
            stage.GetRootLayer().Save()
            data = path.read_bytes()
            manifest.append(
                dict(
                    file=name,
                    shape=shape,
                    mode=mode,
                    index=index,
                    scale=scale,
                    color=color,
                    mass_kg=0.1,
                    inertia_kg_m2=[moment] * 3,
                    triangles=len(faces),
                    bytes=len(data),
                    sha256=hashlib.sha256(data).hexdigest(),
                    geometry_sha256=hashlib.sha256(
                        (vertices * scale).astype("float32").tobytes()
                        + faces.astype("int32").tobytes()
                    ).hexdigest(),
                )
            )
        print(shape, limit + 1, flush=True)
    positions = np.array(
        [
            [0.1 * (i % 10), 0.1 * ((i // 10) % 10), 0.1 + 0.1 * (i // 100)]
            for i in range(1000)
        ]
    )
    capacity = np.array(
        [
            [0.1 * (i % 25), 0.1 * ((i // 25) % 25), 0.1 + 0.1 * (i // 625)]
            for i in range(16000)
        ]
    )
    np.savez(args.output / "layouts.npz", main=positions, capacity=capacity)
    (args.output / "manifest.json").write_text(
        json.dumps(
            dict(
                source="Procedurally authored for L-02; no downloaded inputs",
                license="Apache-2.0 (authored inputs only)",
                files=manifest,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
