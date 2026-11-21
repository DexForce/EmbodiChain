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

"""Pinned-source analytic control geometry fixture; no simulator or model calls."""

from __future__ import annotations

import hashlib
import json


def _save_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def knob_source(tmp_path):
    import trimesh
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    path = tmp_path / "knob.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Control")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    for name in ("base", "knob"):
        p = UsdGeom.Xform.Define(stage, "/Control/" + name).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(p)
        mass = UsdPhysics.MassAPI.Apply(p)
        mass.CreateMassAttr(0.02)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.01, 0.02, 0.03))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0)))
    for body, name, position, size in [
        ("base", "panel", (0, 0, -0.01), (0.2, 0.2, 0.02)),
        ("knob", "grip", (0, 0, 0.02), (0.06, 0.06, 0.02)),
        ("knob", "pointer", (0, 0.02, 0.03), (0.002, 0.015, 0.001)),
        ("base", "label0", (0, 0.07, 0), (0.005, 0.005, 0.001)),
        ("base", "label90", (0.07, 0, 0), (0.005, 0.005, 0.001)),
    ]:
        geometry = trimesh.creation.box(extents=size)
        geometry.apply_translation(position)
        mesh = UsdGeom.Mesh.Define(stage, f"/Control/{body}/{name}")
        mesh.CreatePointsAttr(geometry.vertices.tolist())
        mesh.CreateFaceVertexCountsAttr([3] * len(geometry.faces))
        mesh.CreateFaceVertexIndicesAttr(geometry.faces.reshape(-1).tolist())
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Control/knob_rotation")
    joint.CreateBody0Rel().SetTargets(["/Control/base"])
    joint.CreateBody1Rel().SetTargets(["/Control/knob"])
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(-90.0)
    joint.CreateUpperLimitAttr(0.0)
    stage.GetRootLayer().Save()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    _save_json(
        path.with_suffix(".twist.json"),
        {
            "schema": "gen_sim.twist-control/v1",
            "source_sha256": digest,
            "joint": "/Control/knob_rotation",
            "grip": "/Control/knob/grip",
            "pointer": "/Control/knob/pointer",
            "settings": {
                "0": ["/Control/base/label0"],
                "90": ["/Control/base/label90"],
            },
        },
    )
    art = {
        "uid": "control",
        "fpath": str(path),
        "fix_base": True,
        "body_scale": [1.0] * 3,
        "init_pos": [0, 0, 0.8],
        "init_rot": [0, 0, 0],
    }
    return art
