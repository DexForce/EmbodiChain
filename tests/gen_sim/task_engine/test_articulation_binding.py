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

from pxr import Usd, UsdGeom, UsdPhysics
import pytest

from embodichain.gen_sim.task_engine._task_program.articulation_binding import (
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
