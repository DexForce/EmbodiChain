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
"""CPU-only provenance contracts for E8 deployment mass-source copies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from embodichain.gen_sim.task_engine._task_program.twist_mass_source import (
    adjacent_manifest_path,
    create_mass_source_copy,
    qualify_mass_source_copy,
)
from embodichain.gen_sim.task_engine._task_program.twist_semantics import (
    resolve_twist_semantics,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def mass_source(tmp_path):
    path = tmp_path / "unseen_mass_control.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    root = UsdGeom.Xform.Define(stage, "/Unexpected").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    for name in ("anchor", "rotor"):
        body = UsdGeom.Xform.Define(stage, f"/Unexpected/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        mass = UsdPhysics.MassAPI.Apply(body)
        mass.CreateMassAttr().Set(0.25)
        mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.1, 0.2, 0.3))
        mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(0.01, 0.02, 0.03))
        mass.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, Gf.Vec3f(0)))
    for body, name, collide in (
        ("rotor", "surface", True),
        ("rotor", "indicator", False),
        ("anchor", "mark_zero", False),
        ("anchor", "mark_other", False),
    ):
        mesh = UsdGeom.Mesh.Define(stage, f"/Unexpected/{body}/{name}")
        mesh.CreatePointsAttr([(0, 0, 0), (0.01, 0, 0), (0, 0.01, 0)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
        if collide:
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Unexpected/constraint")
    joint.CreateBody0Rel().SetTargets(["/Unexpected/anchor"])
    joint.CreateBody1Rel().SetTargets(["/Unexpected/rotor"])
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(-180.0)
    joint.CreateUpperLimitAttr(180.0)
    stage.GetRootLayer().Save()
    roles = {
        "schema": "gen_sim.twist-control/v1",
        "source_sha256": _hash(path),
        "joint": "/Unexpected/constraint",
        "grip": "/Unexpected/rotor/surface",
        "pointer": "/Unexpected/rotor/indicator",
        "settings": {
            "0": ["/Unexpected/anchor/mark_zero"],
            "90": ["/Unexpected/anchor/mark_other"],
        },
    }
    path.with_suffix(".twist.json").write_text(json.dumps(roles))
    return path, stage, roles


def _reseal(path: Path, stage, roles) -> None:
    stage.GetRootLayer().Save()
    roles["source_sha256"] = _hash(path)
    path.with_suffix(".twist.json").write_text(json.dumps(roles))


def _copy(mass_source, tmp_path):
    return create_mass_source_copy(mass_source[0], tmp_path / "copies", (0.5,) * 3)


def test_copy_preserves_source_bytes_and_old_probe_mass_operation(
    mass_source, tmp_path
):
    source, stage, _ = mass_source
    original_bytes = source.read_bytes()
    result = _copy(mass_source, tmp_path)
    assert result.source.read_bytes() == source.read_bytes() == original_bytes
    assert result.source_sha256 == _hash(source)
    assert result.runtime_sha256 == _hash(result.runtime)
    assert result.manifest == adjacent_manifest_path(result.runtime)
    runtime_stage = Usd.Stage.Open(Sdf.Layer.OpenAsAnonymous(str(result.runtime)))
    for name in ("anchor", "rotor"):
        before = UsdPhysics.MassAPI(stage.GetPrimAtPath(f"/Unexpected/{name}"))
        after = UsdPhysics.MassAPI(runtime_stage.GetPrimAtPath(f"/Unexpected/{name}"))
        assert after.GetMassAttr().Get() == before.GetMassAttr().Get()
        assert after.GetPrincipalAxesAttr().Get() == before.GetPrincipalAxesAttr().Get()
        assert np.array_equal(
            np.asarray(after.GetCenterOfMassAttr().Get()),
            np.asarray(
                [float(x) * 0.5 for x in before.GetCenterOfMassAttr().Get()],
                dtype=np.float32,
            ),
        )
        assert np.array_equal(
            np.asarray(after.GetDiagonalInertiaAttr().Get()),
            np.asarray(
                [float(x) * 0.5**2 for x in before.GetDiagonalInertiaAttr().Get()],
                dtype=np.float32,
            ),
        )


def test_roles_and_sidecar_snapshots_are_preserved(mass_source, tmp_path):
    source = mass_source[0]
    result = _copy(mass_source, tmp_path)
    original = resolve_twist_semantics(source)
    copied = resolve_twist_semantics(result.runtime)
    assert (
        result.source_semantics_sha256
        == result.runtime_semantics_sha256
        == original.semantics_sha256
    )
    assert copied.settings == original.settings
    assert result.source_sidecar_sha256 == _hash(source.with_suffix(".twist.json"))
    assert result.runtime_sidecar_sha256 == _hash(
        result.runtime.with_suffix(".twist.json")
    )
    assert (
        result.source.with_suffix(".twist.json").read_bytes()
        == source.with_suffix(".twist.json").read_bytes()
    )


def test_content_address_reuse_requires_complete_qualification(mass_source, tmp_path):
    result = _copy(mass_source, tmp_path)
    assert _copy(mass_source, tmp_path) == result
    assert (
        qualify_mass_source_copy(result.runtime, manifest_sha256=result.manifest_sha256)
        == result
    )


@pytest.mark.parametrize(
    "scale",
    [
        True,
        0.5,
        (),
        (0, 0, 0),
        (-1,) * 3,
        (1, 2, 1),
        (float("nan"),) * 3,
        (float("inf"),) * 3,
        (True,) * 3,
        (True, 1, 1),
        np.asarray(0.5),
    ],
)
def test_invalid_uniform_scale_is_rejected(mass_source, tmp_path, scale):
    with pytest.raises(ValueError, match="positive finite uniform"):
        create_mass_source_copy(mass_source[0], tmp_path / "copies", scale)


@pytest.mark.parametrize("backend", ["newton", "dexuni", "unknown"])
def test_other_backends_cannot_double_scale(mass_source, tmp_path, backend):
    with pytest.raises(ValueError, match="Default-only"):
        create_mass_source_copy(
            mass_source[0], tmp_path / "copies", (0.5,) * 3, backend=backend
        )


@pytest.mark.parametrize(
    "name",
    [
        "physics:mass",
        "physics:centerOfMass",
        "physics:diagonalInertia",
        "physics:principalAxes",
    ],
)
def test_time_sampled_mass_is_rejected(mass_source, tmp_path, name):
    source, stage, roles = mass_source
    attr = stage.GetPrimAtPath("/Unexpected/rotor").GetAttribute(name)
    attr.Set(attr.Get(), 1.0)
    _reseal(source, stage, roles)
    with pytest.raises(ValueError, match="static"):
        _copy(mass_source, tmp_path)


@pytest.mark.parametrize(
    "change", ["missing", "density", "inherited", "zero_inertia", "bad_axes", "non_z"]
)
def test_unsupported_source_mass_contract_is_rejected(mass_source, tmp_path, change):
    source, stage, roles = mass_source
    body = stage.GetPrimAtPath("/Unexpected/rotor")
    api = UsdPhysics.MassAPI(body)
    if change == "missing":
        body.RemoveProperty("physics:centerOfMass")
    elif change == "density":
        api.CreateDensityAttr().Set(2.0)
    elif change == "inherited":
        UsdPhysics.MassAPI.Apply(stage.GetDefaultPrim())
    elif change == "zero_inertia":
        api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(0))
    elif change == "bad_axes":
        api.GetPrincipalAxesAttr().Set(Gf.Quatf(0, Gf.Vec3f(0)))
    else:
        UsdGeom.SetStageUpAxis(stage, "Y")
    _reseal(source, stage, roles)
    with pytest.raises(ValueError):
        _copy(mass_source, tmp_path)


@pytest.mark.parametrize(
    "target", ["source", "runtime", "manifest", "source_sidecar", "runtime_sidecar"]
)
def test_byte_drift_fails_qualification_and_cache_reuse(mass_source, tmp_path, target):
    result = _copy(mass_source, tmp_path)
    path = (
        getattr(result, target)
        if "sidecar" not in target
        else getattr(result, target.split("_")[0]).with_suffix(".twist.json")
    )
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        qualify_mass_source_copy(result.runtime, manifest_sha256=result.manifest_sha256)
    with pytest.raises(ValueError):
        _copy(mass_source, tmp_path)


def test_resealed_non_mass_geometry_change_is_still_rejected(mass_source, tmp_path):
    result = _copy(mass_source, tmp_path)
    runtime_stage = Usd.Stage.Open(Sdf.Layer.OpenAsAnonymous(str(result.runtime)))
    mesh = UsdGeom.Mesh(runtime_stage.GetPrimAtPath("/Unexpected/rotor/surface"))
    mesh.GetPointsAttr().Set([(0, 0, 0), (0.02, 0, 0), (0, 0.01, 0)])
    runtime_stage.GetRootLayer().Export(str(result.runtime))
    sidecar = result.runtime.with_suffix(".twist.json")
    roles = json.loads(sidecar.read_text())
    roles["source_sha256"] = _hash(result.runtime)
    sidecar.write_text(json.dumps(roles))
    manifest = json.loads(result.manifest.read_text())
    manifest["runtime_sha256"] = _hash(result.runtime)
    manifest["runtime_sidecar_sha256"] = _hash(sidecar)
    result.manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="allowed MassAPI"):
        qualify_mass_source_copy(result.runtime)


def test_manifest_unknown_fields_duplicate_keys_and_path_escape_rejected(
    mass_source, tmp_path
):
    result = _copy(mass_source, tmp_path)
    original = result.manifest.read_bytes()
    for field, value in [
        ("ignored", True),
        ("source_file", "../outside.usda"),
        ("backend", "newton"),
        ("scale", float("nan")),
    ]:
        data = json.loads(original)
        data[field] = value
        result.manifest.write_text(json.dumps(data))
        with pytest.raises(ValueError):
            qualify_mass_source_copy(result.runtime)
    result.manifest.write_text(original.decode()[:-2] + ', "schema": "duplicate"}')
    with pytest.raises(ValueError):
        qualify_mass_source_copy(result.runtime)


def test_deployment_source_cannot_be_baked_again(mass_source, tmp_path):
    result = _copy(mass_source, tmp_path)
    with pytest.raises(ValueError, match="already"):
        create_mass_source_copy(result.runtime, tmp_path / "again", (0.5,) * 3)


def test_failed_generation_leaves_original_unchanged(mass_source, tmp_path):
    source = mass_source[0]
    before = source.read_bytes()
    directory = tmp_path / "copies"
    result = _copy(mass_source, tmp_path)
    result.runtime.write_bytes(b"unqualified partial artifact")
    with pytest.raises(ValueError):
        create_mass_source_copy(source, directory, (0.5,) * 3)
    assert source.read_bytes() == before
    assert result.runtime.read_bytes() == b"unqualified partial artifact"


@pytest.mark.parametrize(
    "change",
    ["missing_api", "mass_zero", "external_layer", "reference", "animated_non_role"],
)
def test_additional_source_conflicts_are_rejected(mass_source, tmp_path, change):
    source, stage, roles = mass_source
    body = stage.GetPrimAtPath("/Unexpected/rotor")
    if change == "missing_api":
        body.RemoveAPI(UsdPhysics.MassAPI)
    elif change == "mass_zero":
        UsdPhysics.MassAPI(body).GetMassAttr().Set(0.0)
    elif change == "external_layer":
        other = tmp_path / "external.usda"
        Sdf.Layer.CreateNew(str(other)).Save()
        stage.GetRootLayer().subLayerPaths = [str(other)]
    elif change == "reference":
        stage.GetDefaultPrim().GetReferences().AddInternalReference(
            "/Unexpected/anchor"
        )
    else:
        attr = stage.GetDefaultPrim().CreateAttribute(
            "unrelated", Sdf.ValueTypeNames.Float
        )
        attr.Set(1.0, 1.0)
    _reseal(source, stage, roles)
    with pytest.raises(ValueError):
        _copy(mass_source, tmp_path)


def test_embedded_roles_need_no_source_sidecar_but_runtime_roles_are_sealed(
    mass_source, tmp_path
):
    source, stage, roles = mass_source
    embedded = {key: value for key, value in roles.items() if key != "source_sha256"}
    stage.GetDefaultPrim().SetCustomDataByKey("gen_sim:twistControl", embedded)
    stage.GetRootLayer().Save()
    source.with_suffix(".twist.json").unlink()
    result = _copy(mass_source, tmp_path)
    assert result.source_sidecar_sha256 is None
    assert not result.source.with_suffix(".twist.json").exists()
    assert result.runtime_sidecar_sha256 is not None
    assert result.source_semantics_sha256 == result.runtime_semantics_sha256
    assert qualify_mass_source_copy(result.runtime) == result


def test_source_drift_during_generation_is_not_rebaselined(
    mass_source, tmp_path, monkeypatch
):
    from embodichain.gen_sim.task_engine._task_program import twist_mass_source

    source = mass_source[0]
    original = source.read_bytes()
    scale_layer = twist_mass_source._scaled_layer
    changed = False

    def mutate_after_clone(stage, scale):
        nonlocal changed
        layer = scale_layer(stage, scale)
        if not changed:
            changed = True
            source.write_bytes(original + b"\n")
        return layer

    monkeypatch.setattr(twist_mass_source, "_scaled_layer", mutate_after_clone)
    with pytest.raises(ValueError, match="changed"):
        _copy(mass_source, tmp_path)


def test_same_origin_metadata_cannot_authorize_a_changed_runtime(mass_source, tmp_path):
    result = _copy(mass_source, tmp_path)
    runtime_stage = Usd.Stage.Open(Sdf.Layer.OpenAsAnonymous(str(result.runtime)))
    runtime_stage.GetDefaultPrim().SetCustomDataByKey(
        "gen_sim:origin", result.source_sha256
    )
    runtime_stage.GetRootLayer().Export(str(result.runtime))
    sidecar = result.runtime.with_suffix(".twist.json")
    roles = json.loads(sidecar.read_text())
    roles["source_sha256"] = _hash(result.runtime)
    sidecar.write_text(json.dumps(roles))
    manifest = json.loads(result.manifest.read_text())
    manifest["runtime_sha256"] = _hash(result.runtime)
    manifest["runtime_sidecar_sha256"] = _hash(sidecar)
    result.manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="allowed MassAPI"):
        qualify_mass_source_copy(result.runtime)


@pytest.mark.parametrize(
    "location",
    [
        "shader",
        "asset_array",
        "inactive",
        "unselected_variant",
        "nested_custom",
        "layer_custom",
        "inactive_sample",
    ],
)
def test_external_asset_paths_are_rejected_even_outside_active_composition(
    mass_source, tmp_path, location
):
    source, stage, roles = mass_source
    root = stage.GetDefaultPrim()

    def author_texture():
        shader = stage.DefinePrim("/Unexpected/Shader", "Shader")
        if location == "asset_array":
            shader.CreateAttribute("inputs:files", Sdf.ValueTypeNames.AssetArray).Set(
                [Sdf.AssetPath(""), Sdf.AssetPath("external.png")]
            )
        else:
            texture = shader.CreateAttribute("inputs:file", Sdf.ValueTypeNames.Asset)
            texture.Set(Sdf.AssetPath("external.png"))
            if location == "inactive_sample":
                texture.Set(Sdf.AssetPath("animated_external.png"), 1.0)
        if location in {"inactive", "inactive_sample"}:
            shader.SetActive(False)

    if location == "unselected_variant":
        variants = root.GetVariantSets().AddVariantSet("appearance")
        variants.AddVariant("textured")
        variants.AddVariant("plain")
        variants.SetVariantSelection("textured")
        with variants.GetVariantEditContext():
            author_texture()
        variants.SetVariantSelection("plain")
    elif location == "nested_custom":
        root.SetCustomDataByKey(
            "unrelated", {"nested": {"file": Sdf.AssetPath("external.png")}}
        )
    elif location == "layer_custom":
        stage.GetRootLayer().customLayerData = {
            "unrelated": {"file": Sdf.AssetPath("external.png")}
        }
    else:
        author_texture()
    _reseal(source, stage, roles)
    with pytest.raises(ValueError, match="asset paths"):
        _copy(mass_source, tmp_path)


def test_empty_asset_paths_do_not_declare_external_resources(mass_source, tmp_path):
    source, stage, roles = mass_source
    shader = stage.DefinePrim("/Unexpected/Shader", "Shader")
    shader.CreateAttribute("inputs:file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("")
    )
    shader.CreateAttribute("inputs:files", Sdf.ValueTypeNames.AssetArray).Set(
        [Sdf.AssetPath("")]
    )
    _reseal(source, stage, roles)
    assert _copy(mass_source, tmp_path).source_sha256 == _hash(source)
