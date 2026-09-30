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


def trace():
    return [
        PressSample(0.0, 0.0, "settle", False),
        PressSample(0.1, 0.0, "contact", False),
        PressSample(0.2, 0.0, "press", True),
        PressSample(0.3, -0.004, "press", True),
        PressSample(0.4, -0.0097, "press", True),
        PressSample(0.5, 0.0, "retract", False),
    ]


def test_binding_uses_scaled_collision_surface_and_inward_axis(button):
    selected = binding(button)
    assert selected.stroke == pytest.approx(0.01)
    assert selected.axis == pytest.approx((0.0, 0.0, -1.0))
    assert selected.press_position == pytest.approx((0.0, 0.0, 0.02))
    assert len(selected.source_sha256) == 64


def test_binding_and_derived_gate_use_json_native_scalars(button):
    from dataclasses import asdict
    import json

    selected = binding(button)
    report = {"binding": asdict(selected), "released": 0.0 <= selected.stroke * 0.1}
    assert json.loads(json.dumps(report))["released"] is True


def test_binding_accepts_single_rounded_cap_support_point(button):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(button["fpath"])
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Button/cap/contact"))
    mesh.GetPointsAttr().Set(
        [(-0.02, -0.02, 0.01), (0.02, -0.02, 0.01), (0.0, 0.0, 0.02)]
    )
    mesh.GetFaceVertexCountsAttr().Set([3])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2])
    stage.GetRootLayer().Save()
    assert binding(button).press_position == pytest.approx((0.0, 0.0, 0.04))


@pytest.mark.parametrize(
    "change, error",
    [
        ({"fix_base": False}, "fixed base"),
        ({"body_scale": [1.0, 2.0, 1.0]}, "uniform"),
        ({"body_scale": [float("nan")] * 3}, "uniform"),
        ({"uid": ""}, "identities"),
    ],
)
def test_binding_rejects_unsupported_config(button, change, error):
    with pytest.raises(ValueError, match=error):
        binding({**button, **change})


@pytest.mark.parametrize("joint, link", [("missing", "cap"), ("press_joint", "base")])
def test_binding_rejects_wrong_identity(button, joint, link):
    with pytest.raises(ValueError):
        inspect_press(button, joint=joint, link=link)


@pytest.mark.parametrize("joint_type", ["FixedJoint", "RevoluteJoint"])
def test_binding_rejects_parent_below_another_joint(button, joint_type):
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(button["fpath"])
    joint = getattr(UsdPhysics, joint_type).Define(stage, "/Button/other_joint")
    joint.CreateBody1Rel().SetTargets(["/Button/base"])
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="root-body parent"):
        binding(button)


def test_binding_keeps_explicit_button_with_independent_sibling_joint(button):
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(button["fpath"])
    sibling = UsdGeom.Xform.Define(stage, "/Button/lid").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(sibling)
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Button/lid_hinge")
    joint.CreateBody0Rel().SetTargets(["/Button/base"])
    joint.CreateBody1Rel().SetTargets(["/Button/lid"])
    stage.GetRootLayer().Save()
    assert binding(button).joint == "press_joint"


def test_binding_requires_explicit_collision_selection(button):
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(button["fpath"])
    extra = UsdGeom.Mesh.Define(stage, "/Button/cap/extra")
    UsdPhysics.CollisionAPI.Apply(extra.GetPrim())
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="unambiguous"):
        binding(button)
    selected = inspect_press(
        button, joint="press_joint", link="cap", collision_mesh="/Button/cap/contact"
    )
    assert selected.collision_mesh == "/Button/cap/contact"


@pytest.mark.parametrize("meters", [0.01, 100.0])
def test_binding_rejects_non_metre_assets(button, meters):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(button["fpath"])
    UsdGeom.SetStageMetersPerUnit(stage, meters)
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="metre"):
        binding(button)


def test_binding_rejects_non_endpoint_release(button):
    with pytest.raises(ValueError, match="endpoint"):
        inspect_press(button, joint="press_joint", link="cap", released_position=-0.005)


def test_binding_supports_opposite_explicit_release_endpoint(button):
    selected = inspect_press(
        button, joint="press_joint", link="cap", released_position=-0.01
    )
    assert selected.axis == pytest.approx((0.0, 0.0, 1.0))
    assert selected.pressed_position == 0.0


def test_transient_contact_press_succeeds_without_persistent_activation(button):
    result = evaluate_press_event(binding(button), trace())
    assert result["accepted"]
    assert result["contact_stroke"] == pytest.approx(0.0097)


@pytest.mark.parametrize(
    "travel,accepted", [(0.000249, False), (0.000250, True), (0.000251, True)]
)
def test_contact_press_uses_absolute_quarter_millimetre_threshold(
    button, travel, accepted
):
    samples = [
        PressSample(0.0, 0.0, "prepare", False),
        PressSample(0.1, 0.0, "press", True),
        PressSample(0.2, -travel, "press", True),
        PressSample(0.3, 0.0, "cleanup", False),
    ]
    result = evaluate_press_event(binding(button), samples)
    assert result["accepted"] is accepted
    assert result["minimum_contact_stroke_m"] == 0.00025
    assert result["activation_verified"] is False


def test_separate_subthreshold_contacts_do_not_accumulate(button):
    samples = [
        PressSample(0.0, 0.0, "prepare", False),
        PressSample(0.1, 0.0, "press", True),
        PressSample(0.2, -0.00015, "press", True),
        PressSample(0.3, -0.00015, "press", False),
        PressSample(0.4, -0.00015, "press", True),
        PressSample(0.5, -0.00030, "press", True),
    ]
    result = evaluate_press_event(binding(button), samples)
    assert not result["accepted"]
    assert result["contact_stroke"] == pytest.approx(0.00015)


def test_event_supports_positive_joint_travel(button):
    selected = replace(binding(button), limits=(0.0, 0.01), pressed_position=0.01)
    samples = [replace(s, qpos=-s.qpos) for s in trace()]
    assert evaluate_press_event(selected, samples)["accepted"]


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
        samples[3] = replace(samples[3], qpos=-0.00015, target_contact=False)
        samples[4] = replace(samples[4], qpos=-0.00030)
    elif ablation == "late_contact":
        samples[2] = replace(samples[2], target_contact=False)
    elif ablation == "short_stroke":
        samples[3] = replace(samples[3], qpos=-0.00010)
        samples[4] = replace(samples[4], qpos=-0.00020)
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
