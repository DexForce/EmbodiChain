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

from dataclasses import FrozenInstanceError
import hashlib
import json

import pytest

from embodichain.gen_sim.task_engine._task_program import twist_semantics
from embodichain.gen_sim.task_engine._task_program.twist_semantics import (
    InvalidTwistSemanticsError,
    MissingTwistSemanticsError,
    resolve_twist_semantics,
)


@pytest.fixture
def semantic_asset(tmp_path):
    from pxr import Usd, UsdGeom, UsdPhysics

    path = tmp_path / "unseen_control.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Unrelated42").GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    for name in ("piece_17", "piece_83", "piece_91"):
        body = UsdGeom.Xform.Define(stage, f"/Unrelated42/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
    for body, name, collide in (
        ("piece_83", "surface_a", True),
        ("piece_83", "surface_b", False),
        ("piece_17", "surface_c", False),
        ("piece_17", "surface_d", False),
        ("piece_17", "surface_e", False),
        ("piece_91", "surface_f", True),
    ):
        mesh = UsdGeom.Mesh.Define(stage, f"/Unrelated42/{body}/{name}")
        mesh.CreatePointsAttr([(0, 0, 0), (0.01, 0, 0), (0, 0.01, 0)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
        if collide:
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Unrelated42/constraint_19")
    joint.CreateBody0Rel().SetTargets(["/Unrelated42/piece_17"])
    joint.CreateBody1Rel().SetTargets(["/Unrelated42/piece_83"])
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(-180.0)
    joint.CreateUpperLimitAttr(180.0)
    stage.GetRootLayer().Save()
    data = {
        "schema": "gen_sim.twist-control/v1",
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "joint": "/Unrelated42/constraint_19",
        "grip": "/Unrelated42/piece_83/surface_a",
        "pointer": "/Unrelated42/piece_83/surface_b",
        "settings": {
            "90": ["/Unrelated42/piece_17/surface_e"],
            "0": [
                "/Unrelated42/piece_17/surface_d",
                "/Unrelated42/piece_17/surface_c",
            ],
        },
    }
    return path, stage, data


def _save_sidecar(path, data):
    path.with_suffix(".twist.json").write_text(json.dumps(data), encoding="utf-8")


def _save_changed_source(path, stage, data):
    stage.GetRootLayer().Save()
    data["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _save_sidecar(path, data)


def test_arbitrary_source_identity_and_names_are_accepted(semantic_asset):
    path, _, data = semantic_asset
    _save_sidecar(path, data)
    result = resolve_twist_semantics(path)
    assert result.source_sha256 == data["source_sha256"]
    assert result.joint_path == data["joint"]
    assert result.grip_path == data["grip"]
    assert result.pointer_path == data["pointer"]
    assert (
        result.sidecar_sha256
        == hashlib.sha256(path.with_suffix(".twist.json").read_bytes()).hexdigest()
    )
    assert result.settings == (
        (
            0,
            (
                "/Unrelated42/piece_17/surface_c",
                "/Unrelated42/piece_17/surface_d",
            ),
        ),
        (90, ("/Unrelated42/piece_17/surface_e",)),
    )
    with pytest.raises(FrozenInstanceError):
        result.joint_path = "/Other"


def test_semantic_digest_is_order_independent_but_roles_are_bound(semantic_asset):
    path, _, data = semantic_asset
    _save_sidecar(path, data)
    original = resolve_twist_semantics(path)
    data["settings"] = {
        "0": list(reversed(data["settings"]["0"])),
        "90": data["settings"]["90"],
    }
    _save_sidecar(path, dict(reversed(list(data.items()))))
    assert resolve_twist_semantics(path).semantics_sha256 == original.semantics_sha256
    data["settings"]["90"] = ["/Unrelated42/piece_17/surface_c"]
    data["settings"]["0"] = ["/Unrelated42/piece_17/surface_d"]
    _save_sidecar(path, data)
    assert resolve_twist_semantics(path).semantics_sha256 != original.semantics_sha256


def test_snapshot_sidecar_digest_is_from_qualified_bytes_not_later_read(
    semantic_asset, monkeypatch
):
    path, _, data = semantic_asset
    _save_sidecar(path, data)
    sidecar = path.with_suffix(".twist.json")
    original_bytes = sidecar.read_bytes()
    original_dumps = json.dumps
    mutated = False

    def change_after_qualification(value, *args, **kwargs):
        nonlocal mutated
        result = original_dumps(value, *args, **kwargs)
        if set(value) == {"schema", "joint", "grip", "pointer", "settings"}:
            mutated = True
            sidecar.write_bytes(original_bytes + b"\n")
        return result

    monkeypatch.setattr(twist_semantics.json, "dumps", change_after_qualification)
    result = resolve_twist_semantics(path)
    assert mutated
    assert result.sidecar_sha256 == hashlib.sha256(original_bytes).hexdigest()
    assert result.sidecar_sha256 != hashlib.sha256(sidecar.read_bytes()).hexdigest()
    assert result.source_sha256 == data["source_sha256"]


def test_missing_metadata_does_not_guess_names_or_joint(semantic_asset):
    path, _, _ = semantic_asset
    with pytest.raises(MissingTwistSemanticsError):
        resolve_twist_semantics(path)


def test_unannotated_urdf_is_not_a_twist_semantics_error(tmp_path):
    path = tmp_path / "drawer.urdf"
    path.write_text("<robot name='source'/>", encoding="utf-8")
    with pytest.raises(MissingTwistSemanticsError):
        resolve_twist_semantics(path)


def test_annotated_non_usd_is_rejected_without_guessing(tmp_path):
    path = tmp_path / "drawer.urdf"
    path.write_text("<robot name='source'/>", encoding="utf-8")
    path.with_suffix(".twist.json").write_text("{}", encoding="utf-8")
    with pytest.raises(InvalidTwistSemanticsError, match="USD"):
        resolve_twist_semantics(path)


@pytest.mark.parametrize("change", ["non_metre", "no_default", "external_layer"])
def test_unannotated_non_knobs_skip_without_rotary_stage_requirements(
    semantic_asset, change
):
    from pxr import UsdGeom

    path, stage, _ = semantic_asset
    if change == "non_metre":
        UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    elif change == "no_default":
        stage.ClearDefaultPrim()
    else:
        stage.GetDefaultPrim().GetReferences().AddReference("absent.usda")
    stage.GetRootLayer().Save()
    with pytest.raises(MissingTwistSemanticsError):
        resolve_twist_semantics(path)


def test_stale_source_digest_is_rejected(semantic_asset):
    path, _, data = semantic_asset
    data["source_sha256"] = "0" * 64
    _save_sidecar(path, data)
    with pytest.raises(InvalidTwistSemanticsError, match="source"):
        resolve_twist_semantics(path)


@pytest.mark.parametrize(
    "field", ["schema", "source_sha256", "joint", "grip", "pointer", "settings"]
)
def test_required_fields_are_closed(semantic_asset, field):
    path, _, data = semantic_asset
    del data[field]
    _save_sidecar(path, data)
    with pytest.raises(InvalidTwistSemanticsError):
        resolve_twist_semantics(path)


@pytest.mark.parametrize("field", ["task_id", "qpos", "scale", "ignored"])
def test_extra_fields_are_rejected(semantic_asset, field):
    path, _, data = semantic_asset
    data[field] = "untrusted"
    _save_sidecar(path, data)
    with pytest.raises(InvalidTwistSemanticsError):
        resolve_twist_semantics(path)


@pytest.mark.parametrize(
    "path_value",
    [
        "relative",
        "/Unrelated42/../Other",
        "/",
        "/Unrelated42/constraint_19.foo",
        " /Unrelated42/constraint_19",
    ],
)
def test_paths_must_be_absolute_canonical_prim_paths(semantic_asset, path_value):
    path, _, data = semantic_asset
    data["joint"] = path_value
    _save_sidecar(path, data)
    with pytest.raises(InvalidTwistSemanticsError, match="path"):
        resolve_twist_semantics(path)


@pytest.mark.parametrize("setting", ["01", "+1", "-0", "1.0", " 1", "one"])
def test_settings_require_canonical_integer_strings(semantic_asset, setting):
    path, _, data = semantic_asset
    data["settings"] = {setting: data["settings"]["90"]}
    _save_sidecar(path, data)
    with pytest.raises(InvalidTwistSemanticsError, match="setting"):
        resolve_twist_semantics(path)


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"0": []},
        {"0": "/Unrelated42/piece_17/surface_c"},
        {"0": ["/Unrelated42/piece_17/surface_c"] * 2},
        {
            "0": ["/Unrelated42/piece_17/surface_c"],
            "1": ["/Unrelated42/piece_17/surface_c"],
        },
    ],
)
def test_label_parts_are_nonempty_unique_and_not_shared(semantic_asset, settings):
    path, _, data = semantic_asset
    data["settings"] = settings
    _save_sidecar(path, data)
    with pytest.raises(InvalidTwistSemanticsError, match="setting|label"):
        resolve_twist_semantics(path)


@pytest.mark.parametrize(
    "field,path_value",
    [
        ("joint", "/Unrelated42/piece_17"),
        ("grip", "/Unrelated42/piece_17/surface_c"),
        ("grip", "/Unrelated42/piece_83/surface_b"),
        ("pointer", "/Unrelated42/piece_17/surface_c"),
        ("pointer", "/Unrelated42/missing"),
    ],
)
def test_role_types_and_body_ownership_are_validated(semantic_asset, field, path_value):
    path, _, data = semantic_asset
    data[field] = path_value
    _save_sidecar(path, data)
    with pytest.raises(InvalidTwistSemanticsError):
        resolve_twist_semantics(path)


def test_label_on_rotating_body_is_rejected(semantic_asset):
    path, _, data = semantic_asset
    data["settings"] = {"0": [data["pointer"]]}
    _save_sidecar(path, data)
    with pytest.raises(InvalidTwistSemanticsError, match="label"):
        resolve_twist_semantics(path)


@pytest.mark.parametrize(
    "change",
    [
        "joint_disabled",
        "joint_inactive",
        "body_disabled",
        "collision_disabled",
        "parent_child_equal",
        "multiple_targets",
        "nan_points",
        "nan_transform",
        "singular_transform",
        "animated_points",
        "animated_transform",
        "empty_points",
        "missing_root",
        "non_metre",
    ],
)
def test_usd_state_and_geometry_are_fail_closed(semantic_asset, change):
    from pxr import Gf, UsdGeom, UsdPhysics

    path, stage, data = semantic_asset
    joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(data["joint"]))
    if change == "joint_disabled":
        joint.CreateJointEnabledAttr(False)
    elif change == "joint_inactive":
        joint.GetPrim().SetActive(False)
    elif change == "body_disabled":
        UsdPhysics.RigidBodyAPI(
            stage.GetPrimAtPath("/Unrelated42/piece_83")
        ).CreateRigidBodyEnabledAttr(False)
    elif change == "collision_disabled":
        UsdPhysics.CollisionAPI(
            stage.GetPrimAtPath(data["grip"])
        ).CreateCollisionEnabledAttr(False)
    elif change == "parent_child_equal":
        joint.CreateBody1Rel().SetTargets(["/Unrelated42/piece_17"])
    elif change == "multiple_targets":
        joint.CreateBody1Rel().SetTargets(
            ["/Unrelated42/piece_83", "/Unrelated42/piece_91"]
        )
    elif change == "nan_points":
        UsdGeom.Mesh(stage.GetPrimAtPath(data["pointer"])).CreatePointsAttr(
            [(float("nan"), 0, 0)] * 3
        )
    elif change == "nan_transform":
        UsdGeom.Xformable(stage.GetPrimAtPath(data["pointer"])).AddTranslateOp().Set(
            Gf.Vec3d(float("nan"), 0, 0)
        )
    elif change == "singular_transform":
        UsdGeom.Xformable(stage.GetPrimAtPath(data["pointer"])).AddScaleOp().Set(
            Gf.Vec3f(0, 0, 0)
        )
    elif change == "animated_points":
        UsdGeom.Mesh(stage.GetPrimAtPath(data["pointer"])).GetPointsAttr().Set(
            [(0, 0, 0), (1, 0, 0), (0, 1, 0)], 1
        )
    elif change == "animated_transform":
        UsdGeom.Xformable(
            stage.GetPrimAtPath("/Unrelated42/piece_83")
        ).AddTranslateOp().Set(Gf.Vec3d(0, 0, 1), 1)
    elif change == "empty_points":
        UsdGeom.Mesh(stage.GetPrimAtPath(data["pointer"])).GetPointsAttr().Set([])
    elif change == "missing_root":
        stage.GetDefaultPrim().RemoveAPI(UsdPhysics.ArticulationRootAPI)
    elif change == "non_metre":
        UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    _save_changed_source(path, stage, data)
    with pytest.raises(InvalidTwistSemanticsError):
        resolve_twist_semantics(path)


def test_extra_joint_can_be_selected_explicitly_without_order_guessing(semantic_asset):
    from pxr import UsdGeom, UsdPhysics

    path, stage, data = semantic_asset
    for name in ("marker_93",):
        mesh = UsdGeom.Mesh.Define(stage, f"/Unrelated42/piece_91/{name}")
        mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    extra = UsdPhysics.RevoluteJoint.Define(stage, "/Unrelated42/constraint_03")
    extra.CreateBody0Rel().SetTargets(["/Unrelated42/piece_17"])
    extra.CreateBody1Rel().SetTargets(["/Unrelated42/piece_91"])
    _save_changed_source(path, stage, data)
    assert resolve_twist_semantics(path).joint_path == data["joint"]
    data["joint"] = "/Unrelated42/constraint_03"
    data["grip"] = "/Unrelated42/piece_91/surface_f"
    data["pointer"] = "/Unrelated42/piece_91/marker_93"
    _save_sidecar(path, data)
    assert resolve_twist_semantics(path).joint_path == data["joint"]


def test_additional_incoming_joint_to_selected_body_is_ambiguous(semantic_asset):
    from pxr import UsdPhysics

    path, stage, data = semantic_asset
    extra = UsdPhysics.RevoluteJoint.Define(stage, "/Unrelated42/constraint_03")
    extra.CreateBody0Rel().SetTargets(["/Unrelated42/piece_91"])
    extra.CreateBody1Rel().SetTargets(["/Unrelated42/piece_83"])
    _save_changed_source(path, stage, data)
    with pytest.raises(InvalidTwistSemanticsError, match="root|owned"):
        resolve_twist_semantics(path)


@pytest.mark.parametrize("location", ["root", "settings"])
def test_duplicate_json_keys_are_rejected(semantic_asset, location):
    path, _, data = semantic_asset
    text = json.dumps(data)
    if location == "root":
        text = text.replace(
            '"schema":', '"schema": "gen_sim.twist-control/v1", "schema":', 1
        )
    else:
        text = text.replace(
            '"settings": {',
            '"settings": {"90": ["/Unrelated42/piece_17/surface_e"],',
            1,
        )
    path.with_suffix(".twist.json").write_text(text, encoding="utf-8")
    with pytest.raises(InvalidTwistSemanticsError, match="duplicate"):
        resolve_twist_semantics(path)


def test_embedded_custom_data_can_supply_same_explicit_roles(semantic_asset):
    path, stage, data = semantic_asset
    embedded = {key: value for key, value in data.items() if key != "source_sha256"}
    stage.GetDefaultPrim().SetCustomDataByKey("gen_sim:twistControl", embedded)
    stage.GetRootLayer().Save()
    result = resolve_twist_semantics(path)
    assert result.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result.joint_path == embedded["joint"]
    assert result.sidecar_sha256 is None


def test_embedded_data_cannot_self_author_source_digest(semantic_asset):
    path, stage, data = semantic_asset
    stage.GetDefaultPrim().SetCustomDataByKey("gen_sim:twistControl", data)
    stage.GetRootLayer().Save()
    with pytest.raises(InvalidTwistSemanticsError):
        resolve_twist_semantics(path)


def test_two_sources_must_agree_instead_of_silent_priority(semantic_asset):
    path, stage, data = semantic_asset
    embedded = {key: value for key, value in data.items() if key != "source_sha256"}
    stage.GetDefaultPrim().SetCustomDataByKey("gen_sim:twistControl", embedded)
    _save_changed_source(path, stage, data)
    assert resolve_twist_semantics(path).joint_path == data["joint"]
    data["settings"]["90"] = ["/Unrelated42/piece_17/surface_c"]
    data["settings"]["0"] = ["/Unrelated42/piece_17/surface_d"]
    _save_sidecar(path, data)
    with pytest.raises(InvalidTwistSemanticsError, match="conflict"):
        resolve_twist_semantics(path)


def test_unresolved_external_reference_is_not_self_contained(semantic_asset):
    path, stage, data = semantic_asset
    stage.GetDefaultPrim().GetReferences().AddReference("absent.usda")
    _save_changed_source(path, stage, data)
    with pytest.raises(InvalidTwistSemanticsError, match="self-contained"):
        resolve_twist_semantics(path)


def test_resolution_is_fresh_when_same_usd_path_changes(semantic_asset):
    from pxr import UsdPhysics

    path, stage, data = semantic_asset
    _save_sidecar(path, data)
    assert resolve_twist_semantics(path).source_sha256 == data["source_sha256"]
    UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(data["joint"])).CreateJointEnabledAttr(
        False
    )
    _save_changed_source(path, stage, data)
    with pytest.raises(InvalidTwistSemanticsError, match="enabled"):
        resolve_twist_semantics(path)
