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

"""Author common SI grasp objects; no simulator or external assets are needed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for shape in ["cube", "cylinder"]:
        path = args.output / f"{shape}.usda"
        assert not path.exists(), path
        stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        root = UsdGeom.Xform.Define(stage, "/Object").GetPrim()
        stage.SetDefaultPrim(root)
        UsdPhysics.RigidBodyAPI.Apply(root)
        mass = UsdPhysics.MassAPI.Apply(root)
        mass.CreateMassAttr(0.1)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0))
        inertia = (
            [0.1 * 0.05**2 / 6] * 3
            if shape == "cube"
            else [0.1 * (3 * 0.025**2 + 0.05**2) / 12] * 2 + [0.1 * 0.025**2 / 2]
        )
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(1))
        if shape == "cube":
            geom = UsdGeom.Cube.Define(stage, "/Object/shape")
            geom.CreateSizeAttr(0.05)
        else:
            theta = np.arange(64) * (2 * np.pi / 64)
            ring = np.column_stack([0.025 * np.cos(theta), 0.025 * np.sin(theta)])
            vertices = np.concatenate(
                [np.column_stack([ring, np.full(64, z)]) for z in [-0.025, 0.025]]
            )
            faces = [[j, (j + 1) % 64, (j + 1) % 64 + 64, j + 64] for j in range(64)]
            faces += [[0, j + 1, j] for j in range(1, 63)]
            faces += [[64, 64 + j, 64 + j + 1] for j in range(1, 63)]
            faces = [
                tri
                for face in faces
                for tri in (
                    [face]
                    if len(face) == 3
                    else [[face[0], face[1], face[2]], [face[0], face[2], face[3]]]
                )
            ]
            geom = UsdGeom.Mesh.Define(stage, "/Object/shape")
            geom.CreatePointsAttr([Gf.Vec3f(*p) for p in vertices])
            geom.CreateFaceVertexCountsAttr([len(f) for f in faces])
            geom.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
            geom.CreateSubdivisionSchemeAttr("none")
            UsdPhysics.MeshCollisionAPI.Apply(geom.GetPrim()).CreateApproximationAttr(
                "convexHull"
            )
        geom.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.23, 0.06)])
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
        for prim, schema in [
            (root, "PhysxRigidBodyAPI"),
            (geom.GetPrim(), "PhysxCollisionAPI"),
        ]:
            prim.AddAppliedSchema(schema)
        for name, value in [
            ("physxCollision:restOffset", 0.0),
            ("physxCollision:contactOffset", 0.001),
        ]:
            geom.GetPrim().CreateAttribute(name, Sdf.ValueTypeNames.Float).Set(value)
        for name in ["linearDamping", "angularDamping"]:
            root.CreateAttribute(
                f"physxRigidBody:{name}", Sdf.ValueTypeNames.Float
            ).Set(0.0)
        mat = UsdShade.Material.Define(stage, "/Object/Material")
        physics = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        physics.CreateStaticFrictionAttr(0.8)
        physics.CreateDynamicFrictionAttr(0.8)
        physics.CreateRestitutionAttr(0)
        shader = UsdShade.Shader.Define(stage, "/Object/Material/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.85, 0.23, 0.06)
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(mat)
        UsdShade.MaterialBindingAPI(geom.GetPrim()).Bind(mat, materialPurpose="physics")
        stage.GetRootLayer().Save()
        records.append(
            dict(
                file=path.name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                mass_kg=0.1,
                inertia_kg_m2=inertia,
                source="Authored benchmark geometry",
                license="Apache-2.0",
                cylinder_sections=64 if shape == "cylinder" else None,
            )
        )
    (args.output / "shapes.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
