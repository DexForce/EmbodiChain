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
"""Unknown-source binding contracts without a simulator or identity registry."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from embodichain.gen_sim.task_engine._task_program.twist_binding import (
    TwistRoute,
    discover_twist,
    enrich_twist_inventory,
)
from embodichain.gen_sim.task_engine._task_program.twist_geometry import (
    load_twist_geometry,
)
from embodichain.gen_sim.task_engine._task_program.twist_semantics import (
    InvalidTwistSemanticsError,
    MissingTwistSemanticsError,
)


def _box(stage, path, center, size, *, collision=False):
    mesh = UsdGeom.Mesh.Define(stage, path)
    vertices = [
        tuple(center[i] + size[i] * signs[i] / 2 for i in range(3))
        for signs in (
            (-1, -1, -1),
            (1, -1, -1),
            (1, 1, -1),
            (-1, 1, -1),
            (-1, -1, 1),
            (1, -1, 1),
            (1, 1, 1),
            (-1, 1, 1),
        )
    ]
    mesh.CreatePointsAttr(vertices)
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr(
        [0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7]
    )
    if collision:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    return mesh.GetPrim()


def _source(tmp_path: Path, suffix: str):
    path = tmp_path / f"arbitrary_{suffix}.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = "/Device_" + suffix
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    default = UsdGeom.Xform.Define(stage, root).GetPrim()
    stage.SetDefaultPrim(default)
    UsdPhysics.ArticulationRootAPI.Apply(default)
    fixed, moving = root + "/anchor_" + suffix, root + "/turner_" + suffix
    for name in (fixed, moving):
        body = UsdGeom.Xform.Define(stage, name).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
    joint_path = root + "/axis_" + suffix
    joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([fixed])
    joint.CreateBody1Rel().SetTargets([moving])
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(-180.0)
    joint.CreateUpperLimitAttr(0.0)
    _box(
        stage, fixed + "/base_surface", (0, 0, -0.02), (0.2, 0.2, 0.02), collision=True
    )
    grip = moving + "/surface_" + suffix
    pointer = moving + "/indicator_" + suffix
    first, second = fixed + "/mark_a_" + suffix, fixed + "/mark_b_" + suffix
    _box(stage, grip, (0, 0, 0.02), (0.06, 0.06, 0.02), collision=True)
    _box(stage, pointer, (0, 0.02, 0.03), (0.002, 0.015, 0.001))
    _box(stage, first, (0, 0.07, 0), (0.005, 0.005, 0.001))
    _box(stage, second, (0.07, 0, 0), (0.005, 0.005, 0.001))
    stage.GetRootLayer().Save()
    roles = {
        "schema": "gen_sim.twist-control/v1",
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "joint": joint_path,
        "grip": grip,
        "pointer": pointer,
        "settings": {"0": [first], "73": [second]},
    }
    path.with_suffix(".twist.json").write_text(json.dumps(roles))
    config = {
        "uid": "unseen_" + suffix,
        "fpath": str(path),
        "fix_base": True,
        "body_scale": [0.5] * 3,
    }
    return config, stage, roles


@pytest.mark.parametrize("suffix", ("alpha", "beta", "Q731"))
def test_new_names_and_source_sha_bind_inventory_and_collision_geometry(
    tmp_path, suffix
):
    config, _, roles = _source(tmp_path, suffix)
    route = discover_twist(config, 73)
    assert route.binding.target_qpos == pytest.approx(-math.pi / 2)
    assert route.binding.target_setting == 73
    assert route.binding.grip_depth == pytest.approx(0.01)
    assert route.binding.grip_width == pytest.approx(math.sqrt(2) * 0.03)
    assert TwistRoute.decode(json.loads(json.dumps(route.payload()))) == route
    geometry = load_twist_geometry(config["fpath"], route.binding)
    assert geometry.grip.path == roles["grip"]
    item = {**config, "runtime_uid": config["uid"], "role": "articulation"}
    enrich_twist_inventory([item], [config])
    assert item["attributes"]["twist_setting_labels"]["73"] == route.binding.target_qpos
    assert (
        item["attributes"]["twist_semantics_sha256"] == route.binding.calibration_sha256
    )


def test_unrelated_source_metadata_can_change_sha_without_code_registration(tmp_path):
    config, stage, roles = _source(tmp_path, "fresh")
    old = discover_twist(config, 73)
    stage.SetMetadata("comment", "Different unrelated source bytes.")
    stage.GetRootLayer().Save()
    roles["source_sha256"] = hashlib.sha256(
        Path(config["fpath"]).read_bytes()
    ).hexdigest()
    Path(config["fpath"]).with_suffix(".twist.json").write_text(json.dumps(roles))
    new = discover_twist(config, 73)
    assert new.binding.source_sha256 != old.binding.source_sha256
    assert replace(new.binding, source_sha256=old.binding.source_sha256) == old.binding


def test_missing_annotations_do_not_trigger_role_name_guesses(tmp_path):
    config, _, _ = _source(tmp_path, "missing")
    Path(config["fpath"]).with_suffix(".twist.json").unlink()
    with pytest.raises(MissingTwistSemanticsError):
        discover_twist(config, 73)
    item = {**config, "runtime_uid": config["uid"], "role": "articulation"}
    enrich_twist_inventory([item], [config])
    assert "twist_setting_labels" not in item.get("attributes", {})


def test_stale_or_invalid_declared_semantics_cannot_be_silently_skipped(tmp_path):
    config, _, roles = _source(tmp_path, "stale")
    roles["source_sha256"] = "0" * 64
    Path(config["fpath"]).with_suffix(".twist.json").write_text(json.dumps(roles))
    item = {**config, "runtime_uid": config["uid"], "role": "articulation"}
    with pytest.raises(InvalidTwistSemanticsError, match="stale"):
        enrich_twist_inventory([item], [config])


def test_native_joint_basename_does_not_conflict_with_an_unrelated_mesh(tmp_path):
    config, stage, roles = _source(tmp_path, "same_name")
    _box(stage, "/Unrelated/axis_same_name", (0, 0, 0), (1, 1, 1))
    stage.GetRootLayer().Save()
    roles["source_sha256"] = hashlib.sha256(
        Path(config["fpath"]).read_bytes()
    ).hexdigest()
    Path(config["fpath"]).with_suffix(".twist.json").write_text(json.dumps(roles))
    assert discover_twist(config, 73).binding.joint == "axis_same_name"


@pytest.mark.parametrize(
    "field", ("source_sha256", "calibration_sha256", "mass_lineage_sha256")
)
def test_serialized_route_requires_real_digest_format_not_known_membership(
    tmp_path, field
):
    config, _, _ = _source(tmp_path, "decode")
    payload = discover_twist(config, 73).payload()
    changed = deepcopy(payload)
    changed["binding"][field] = "f" * 64
    assert getattr(TwistRoute.decode(changed).binding, field) == "f" * 64
    changed["binding"][field] = "not_a_digest"
    with pytest.raises(ValueError, match="SHA-256"):
        TwistRoute.decode(changed)
