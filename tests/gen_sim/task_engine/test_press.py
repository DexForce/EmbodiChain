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

from dataclasses import replace
from copy import deepcopy
import pytest
from embodichain.gen_sim.task_engine._task_program.press import (
    PressSample,
    evaluate_press_event,
    inspect_press,
)


@pytest.fixture
def button(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics

    path = tmp_path / "button.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Button")
    stage.SetDefaultPrim(root.GetPrim())
    for name in ("base", "cap"):
        body = UsdGeom.Xform.Define(stage, f"/Button/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
    joint = UsdPhysics.PrismaticJoint.Define(stage, "/Button/press_joint")
    joint.CreateBody0Rel().SetTargets(["/Button/base"])
    joint.CreateBody1Rel().SetTargets(["/Button/cap"])
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(-0.005)
    joint.CreateUpperLimitAttr(0.0)
    mesh = UsdGeom.Mesh.Define(stage, "/Button/cap/contact")
    mesh.CreatePointsAttr(
        [
            (-0.02, -0.02, 0.01),
            (0.02, -0.02, 0.01),
            (0.02, 0.02, 0.01),
            (-0.02, 0.02, 0.01),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    stage.GetRootLayer().Save()
    return {
        "uid": "button",
        "fpath": str(path),
        "fix_base": True,
        "body_scale": [2.0] * 3,
    }


def binding(button):
    return inspect_press(button, joint="press_joint", link="cap")


@pytest.mark.parametrize(
    "kind",
    [
        "articulation_slide",
        "articulation_withdraw",
        "articulation_park",
        "press_prepare",
        "press",
    ],
)
def test_control_lowerer_decode_preserves_binding_and_rejects_drift(button, kind):
    from embodichain.gen_sim.task_engine._task_program.articulation_binding import (
        PrismaticBinding,
    )
    from embodichain.gen_sim.task_engine._task_program.press_binding import PressRoute
    from embodichain.gen_sim.task_engine._task_program.configured import (
        decode_task_lowerer,
    )

    source = binding(button)
    if kind.startswith("press"):
        route = PressRoute(source, 0.02, operation_id="press_once")
        payload = {"kind": kind, "route": route.payload()}
    elif kind == "articulation_park":
        payload = {"kind": kind}
    else:
        slide = PrismaticBinding(
            source.object_id,
            source.joint,
            source.link,
            source.parent,
            source.collision_mesh,
            source.source_sha256,
            source.scale,
            source.axis,
            source.limits,
            closed_position=source.released_position,
        )
        payload = {"kind": kind, "bindings": [slide.payload()]}
    factory = decode_task_lowerer(payload, path="test")
    assert factory.call_id == f"gen_sim.{kind}" + (
        ".press_once" if kind.startswith("press") else ""
    )
    if kind.startswith("press"):
        assert factory.route.payload() == route.payload()
    elif kind != "articulation_park":
        assert factory.bindings == (slide,)
    with pytest.raises(ValueError):
        decode_task_lowerer(dict(payload, unexpected=True), path="test")
    if kind != "articulation_park":
        damaged = deepcopy(payload)
        record = (
            damaged["route"]["binding"]
            if kind.startswith("press")
            else damaged["bindings"][0]
        )
        record["source_sha256"] = "stale"
        with pytest.raises(ValueError):
            decode_task_lowerer(damaged, path="test")


def trace():
    return [
        PressSample(0.0, 0.0, "settle", False),
        PressSample(0.1, 0.0, "contact", False),
        PressSample(0.2, 0.0, "press", True),
        PressSample(0.3, -0.004, "press", True),
        PressSample(0.4, -0.0097, "press", True),
        PressSample(0.5, 0.0, "retract", False),
    ]


def test_transient_contact_press_succeeds_without_persistent_activation(button):
    result = evaluate_press_event(binding(button), trace())
    assert result["accepted"]
    assert result["contact_stroke"] == pytest.approx(0.0097)


@pytest.mark.parametrize(
    "ablation, reason",
    [
        ("no_contact", "insufficient_contact_stroke"),
        ("missing_contact", "invalid_observation"),
        ("prepressed", "not_initially_released"),
        ("passive", "passive_before_press"),
        ("contact_gap", "insufficient_contact_stroke"),
        ("late_contact", "insufficient_contact_stroke"),
        ("short_stroke", "insufficient_contact_stroke"),
        ("stale_time", "invalid_observation"),
        ("nan", "invalid_observation"),
        ("overflow", "invalid_observation"),
        ("out_of_bounds", "joint_out_of_bounds"),
    ],
)
def test_event_evidence_ablations_fail_closed(button, ablation, reason):
    samples = trace()
    if ablation == "no_contact":
        samples = [replace(s, target_contact=False) for s in samples]
    elif ablation == "missing_contact":
        samples[3] = replace(samples[3], target_contact=None)
    elif ablation == "prepressed":
        samples[0] = replace(samples[0], qpos=-0.009)
    elif ablation == "passive":
        samples[1] = replace(samples[1], qpos=-0.004)
    elif ablation == "contact_gap":
        samples[3] = replace(samples[3], qpos=-0.000015, target_contact=False)
        samples[4] = replace(samples[4], qpos=-0.000030)
    elif ablation == "late_contact":
        samples[2] = replace(samples[2], target_contact=False)
    elif ablation == "short_stroke":
        samples[3] = replace(samples[3], qpos=-0.000010)
        samples[4] = replace(samples[4], qpos=-0.000020)
    elif ablation == "stale_time":
        samples[3] = replace(samples[3], timestamp=samples[2].timestamp)
    elif ablation == "nan":
        samples[3] = replace(samples[3], qpos=float("nan"))
    elif ablation == "overflow":
        samples[3] = replace(samples[3], valid=False)
    elif ablation == "out_of_bounds":
        samples[3] = replace(samples[3], qpos=-0.02)
    result = evaluate_press_event(binding(button), samples)
    assert not result["accepted"]
    assert result["reason"] == reason
