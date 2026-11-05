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

"""Authored drawer regions preserve physical topology, scale and target frames."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import trimesh

pytest.importorskip("pxr")
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from embodichain.lab.sim.scene.articulation_inputs import inspect_prismatic
from embodichain.lab.sim.scene.drawer import (
    drawer_moving_links,
    read_drawer_region,
    source_drawer_container_pose,
)

_FIXTURE = Path(__file__).parent / "fixtures/source_declaration_drawer.usda"


def _config(path: Path, scale: float = 1.0) -> dict[str, Any]:
    return dict(
        uid="cabinet",
        root_props=dict(fixed_base=True),
        fpath=str(path),
        body_scale=[scale] * 3,
        init_pos=[3.0, -2.0, 0.7],
        init_rot=[0.0, 0.0, 90.0],
    )


def _box_body(
    stage: Any, name: str, dimensions: list[float], position: list[float]
) -> None:
    body = UsdGeom.Xform.Define(stage, f"/Cabinet/{name}")
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    body.AddTranslateOp().Set(Gf.Vec3d(*position))
    shape = trimesh.creation.box(extents=dimensions)
    mesh = UsdGeom.Mesh.Define(stage, f"/Cabinet/{name}/collision")
    mesh.CreatePointsAttr(shape.vertices.tolist())
    mesh.CreateFaceVertexCountsAttr([3] * len(shape.faces))
    mesh.CreateFaceVertexIndicesAttr(shape.faces.flatten().tolist())
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_fixed_sibling_floor_is_selected_in_the_drawer_frame_once(
    tmp_path: Path, scale: float
) -> None:
    path = tmp_path / "drawer.usda"
    path.write_bytes(_FIXTURE.read_bytes())
    stage = Usd.Stage.Open(str(path))
    # This floor is a sibling in USD; only the fixed joint makes it move with
    # the drawer. Bodies outside the moving subtree must not enlarge its region.
    _box_body(stage, "remote_floor", [0.4, 0.3, 0.01], [0.05, 0.0, -0.02])
    fixed = UsdPhysics.FixedJoint.Define(stage, "/Cabinet/floor_joint")
    fixed.CreateBody0Rel().SetTargets(["/Cabinet/drawer"])
    fixed.CreateBody1Rel().SetTargets(["/Cabinet/remote_floor"])
    _box_body(stage, "unconnected", [2.0, 2.0, 0.01], [5.0, 0.0, 0.0])
    _box_body(stage, "disabled", [3.0, 3.0, 0.01], [8.0, 0.0, 0.0])
    for name in ("unconnected", "disabled"):
        base_joint = UsdPhysics.FixedJoint.Define(stage, f"/Cabinet/base_{name}_joint")
        base_joint.CreateBody0Rel().SetTargets(["/Cabinet/base"])
        base_joint.CreateBody1Rel().SetTargets([f"/Cabinet/{name}"])
    disabled = UsdPhysics.FixedJoint.Define(stage, "/Cabinet/disabled_joint")
    disabled.CreateBody0Rel().SetTargets(["/Cabinet/drawer"])
    disabled.CreateBody1Rel().SetTargets(["/Cabinet/disabled"])
    disabled.CreateJointEnabledAttr(False)
    stage.GetRootLayer().Save()
    before = path.read_bytes()
    cfg = _config(path, scale)
    declared = deepcopy(cfg)
    binding = inspect_prismatic(cfg)

    assert drawer_moving_links(cfg, binding) == {"drawer", "remote_floor"}
    lower, upper = read_drawer_region(cfg, binding)
    # The 1 cm-thick floor is centered at (5, 0, -2) cm. Its top is -1.5 cm;
    # the handle at x=10 cm is excluded from the structural floor selection.
    np.testing.assert_allclose(
        lower, np.array([-0.15, -0.15, -0.015]) * scale, atol=1e-7
    )
    np.testing.assert_allclose(upper, np.array([0.25, 0.15, -0.015]) * scale, atol=1e-7)

    payload_mesh = trimesh.creation.box(extents=[0.04, 0.06, 0.02])
    payload_mesh.vertices = np.asarray(payload_mesh.vertices)[:, [0, 2, 1]] * [1, 1, -1]
    payload_path = tmp_path / "payload.glb"
    payload_mesh.export(payload_path)
    payload = dict(
        shape=dict(shape_type="Mesh", fpath=str(payload_path)),
        init_pos=[9.0, 8.0, 7.0],
        init_rot=[0.0, 0.0, 0.0],
        body_scale=[1.0] * 3,
    )
    target = source_drawer_container_pose(cfg, binding, payload, lower, upper)
    expected = np.eye(4)
    expected[:3, :3] = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    # A 2 cm payload clears the authored floor by the unchanged 3 mm source gap.
    expected[:3, 3] = [0.05 * scale, 0.0, 0.013 - 0.015 * scale]
    np.testing.assert_allclose(target, expected, atol=1e-7)
    assert cfg == declared and path.read_bytes() == before


def test_handle_only_source_keeps_the_original_finite_region_fallback() -> None:
    before = _FIXTURE.read_bytes()
    cfg = _config(_FIXTURE)
    binding = inspect_prismatic(cfg)
    assert drawer_moving_links(cfg, binding) == {"drawer"}
    lower, upper = read_drawer_region(cfg, binding)
    np.testing.assert_allclose(lower, [0.08, -0.01, -0.01], atol=1e-7)
    np.testing.assert_allclose(upper, [0.12, 0.01, 0.01], atol=1e-7)
    assert _FIXTURE.read_bytes() == before
