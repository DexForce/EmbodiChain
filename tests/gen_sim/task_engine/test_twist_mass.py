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

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation
import torch

from embodichain.gen_sim.task_engine._task_program.twist_mass import verify_scaled_mass


class _ReadOnlyArticulation:
    def __init__(
        self,
        source: Path,
        source_descriptor: Any,
        *,
        scale: float,
        rows: int = 2,
        com_factor: float | None = None,
        inertia_factor: float | None = None,
        policy: str = "fixed_mass",
    ) -> None:
        self.cfg = SimpleNamespace(
            fpath=str(source),
            body_scale=[scale] * 3,
            body_scale_mass_policy=policy,
        )
        self.num_instances = rows
        self.getter_calls: list[tuple[str, str]] = []
        self.setter_calls = 0
        self.values: dict[str, dict[str, torch.Tensor]] = {}
        com_factor = scale if com_factor is None else com_factor
        inertia_factor = scale**2 if inertia_factor is None else inertia_factor
        for link in source_descriptor.links:
            body = link.rigid_body
            if body is None or any(
                getattr(body, field) is None
                for field in ("mass", "com_position", "inertia")
            ):
                continue
            tensor = np.asarray(body.inertia, dtype=float) * inertia_factor
            moments, axes = np.linalg.eigh(tensor)
            if np.linalg.det(axes) < 0:
                axes[:, -1] *= -1
            quaternion_xyzw = Rotation.from_matrix(axes).as_quat()
            pose = np.r_[np.asarray(body.com_position) * com_factor, quaternion_xyzw]
            self.values[link.name] = {
                "mass": torch.full((rows, 1), float(body.mass), dtype=torch.float32),
                "inertia": torch.tensor(moments, dtype=torch.float32).repeat(
                    rows, 1, 1
                ),
                "com": torch.tensor(pose, dtype=torch.float32).repeat(rows, 1, 1),
            }

    def get_mass(self, link: str) -> torch.Tensor:
        self.getter_calls.append(("mass", link))
        return self.values[link]["mass"]

    def get_inertia(self, link: str) -> torch.Tensor:
        self.getter_calls.append(("inertia", link))
        return self.values[link]["inertia"]

    def get_com_pose(self, link: str) -> torch.Tensor:
        self.getter_calls.append(("com", link))
        return self.values[link]["com"]

    def set_mass(self, *_args: Any, **_kwargs: Any) -> None:
        self.setter_calls += 1
        raise AssertionError("Readback verification cannot write mass properties.")

    set_inertia = set_mass
    set_com_pose = set_mass
    set_link_physical_attr = set_mass


@dataclass
class _Source:
    path: Path
    descriptor: Any
    sha256: str

    def art(self, *, scale: float = 0.5, **kwargs: Any) -> _ReadOnlyArticulation:
        return _ReadOnlyArticulation(self.path, self.descriptor, scale=scale, **kwargs)

    def binding(self, *, scale: float = 0.5) -> SimpleNamespace:
        return SimpleNamespace(source_sha256=self.sha256, scale=scale)


def _source(path: Path) -> _Source:
    from dexsim.kit.usd import parse_usd

    scene = parse_usd(str(path))
    assert len(scene.articulations) == 1
    return _Source(
        path,
        scene.articulations[0],
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )


@pytest.fixture
def source(tmp_path: Path) -> _Source:
    path = tmp_path / "control.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Control")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    for index, name in enumerate(("housing", "knob")):
        link = UsdGeom.Xform.Define(stage, f"/Control/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(link)
        mass = UsdPhysics.MassAPI.Apply(link)
        mass.CreateMassAttr(0.8 + index * 0.3)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0.1, -0.2, 0.04))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.004, 0.007, 0.011))
        xyzw = Rotation.from_euler("xyz", [0.41, -0.32, 0.67]).as_quat()
        mass.CreatePrincipalAxesAttr(Gf.Quatf(float(xyzw[3]), Gf.Vec3f(*xyzw[:3])))
    fixed = UsdPhysics.FixedJoint.Define(stage, "/Control/fixed_base")
    fixed.CreateBody1Rel().SetTargets(["/Control/housing"])
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Control/axis")
    joint.CreateBody0Rel().SetTargets(["/Control/housing"])
    joint.CreateBody1Rel().SetTargets(["/Control/knob"])
    joint.CreateAxisAttr("Y")
    joint.CreateLowerLimitAttr(-180.0)
    joint.CreateUpperLimitAttr(180.0)
    stage.GetRootLayer().Save()
    return _source(path)


@pytest.mark.parametrize("scale", (0.5, 1.0, 1.8))
@pytest.mark.parametrize("rows", (1, 2))
def test_real_usd_parser_and_independent_eigendecomposition_readbacks(
    source, scale, rows
):
    art = source.art(scale=scale, rows=rows)
    before = {
        name: {key: value.clone() for key, value in values.items()}
        for name, values in art.values.items()
    }
    result = verify_scaled_mass(art, source.binding(scale=scale))
    assert result["verified"] is True
    assert result["scale"] == scale
    assert result["source_sha256"] == source.sha256
    assert len(result["links"]) == len(art.values) * rows
    assert {record["env_id"] for record in result["links"]} == set(range(rows))
    assert art.setter_calls == 0
    for link in source.descriptor.links:
        if link.name not in art.values:
            continue
        for record in result["links"]:
            if record["link"] == link.name:
                np.testing.assert_allclose(
                    record["inertia_body_about_com"],
                    np.asarray(link.rigid_body.inertia) * scale**2,
                    rtol=1e-5,
                    atol=1e-9,
                )
        for key in before[link.name]:
            assert torch.equal(art.values[link.name][key], before[link.name][key])
    assert hashlib.sha256(source.path.read_bytes()).hexdigest() == source.sha256


@pytest.mark.parametrize(
    "task,asset,scale,known_sha",
    [
        (
            "task1130",
            "knob_panel_001",
            2.0 / 3.0,
            "52c558ed024f20a3904adfeb2272d0c09079764319bfdb2ef5dd3996d0240bb3",
        ),
        (
            "task1139",
            "hot_plate_001",
            0.470352381468,
            "a8a859f7b22e076bea2c0e1ae9c5cac725cb0fe2faae247e3f7cde9c480d72e3",
        ),
        (
            "task1149",
            "dimmer_switch_001",
            12.5 / 7.2,
            "48df433eec8aa98a7630e2d012e4f79f5f40a6596fa4dfbd34dddb5d6276a01d",
        ),
    ],
)
@pytest.mark.parametrize("override_scale", (None, 0.5, 1.0))
def test_cached_calibrated_assets_fixed_mass_readback_contract(
    task, asset, scale, known_sha, override_scale
):
    root = Path(__file__).resolve().parents[3]
    path = (
        root
        / "gym_project/task100/new2"
        / task
        / "scene_export/articulated_assets"
        / asset
        / (asset + ".usdc")
    )
    if not path.is_file():
        pytest.skip("Optional external knob asset cache is not present.")
    parsed = _source(path)
    assert parsed.sha256 == known_sha
    assert (
        len([link for link in parsed.descriptor.links if link.rigid_body is not None])
        >= 2
    )
    factor = scale if override_scale is None else override_scale
    art = parsed.art(scale=factor)
    result = verify_scaled_mass(art, parsed.binding(scale=factor))
    assert result["verified"] is True
    assert len(result["links"]) == len(art.values) * 2
    assert art.setter_calls == 0


@pytest.mark.parametrize(
    "task,asset,scale",
    [
        ("task1130", "knob_panel_001", 2.0 / 3.0),
        ("task1139", "hot_plate_001", 0.470352381468),
        ("task1149", "dimmer_switch_001", 12.5 / 7.2),
    ],
)
@pytest.mark.parametrize("override_scale", (None, 0.5, 1.0))
def test_cached_assets_accept_actual_sdk_principal_frame_roundtrip(
    task, asset, scale, override_scale
):
    from dexsim.spawn._inertia import principal_inertia_pair

    root = Path(__file__).resolve().parents[3]
    path = (
        root
        / "gym_project/task100/new2"
        / task
        / "scene_export/articulated_assets"
        / asset
        / (asset + ".usdc")
    )
    if not path.is_file():
        pytest.skip("Optional external knob asset cache is not present.")
    parsed = _source(path)
    factor = scale if override_scale is None else override_scale
    art = parsed.art(scale=factor)
    for link in parsed.descriptor.links:
        if link.name not in art.values:
            continue
        expected = np.asarray(link.rigid_body.inertia) * factor**2
        diagonal, quaternion_wxyz = principal_inertia_pair(expected)
        quaternion_xyzw = np.asarray(quaternion_wxyz)[[1, 2, 3, 0]]
        art.values[link.name]["inertia"] = torch.tensor(
            diagonal, dtype=torch.float32
        ).repeat(art.num_instances, 1, 1)
        art.values[link.name]["com"][..., 3:] = torch.tensor(
            quaternion_xyzw, dtype=torch.float32
        )
        rotation = Rotation.from_quat(quaternion_xyzw).as_matrix()
        reconstructed = rotation @ np.diag(diagonal) @ rotation.T
        tensor_scale = max(float(np.max(np.abs(expected))), np.finfo(np.float32).eps)
        assert np.max(np.abs(reconstructed - expected)) <= tensor_scale * 1.01e-5
    assert verify_scaled_mass(art, parsed.binding(scale=factor))["verified"] is True
    assert art.setter_calls == 0


def test_dropping_meaningful_body_frame_offdiagonals_is_still_rejected(source):
    art = source.art()
    name = next(iter(art.values))
    source_body = next(
        link.rigid_body for link in source.descriptor.links if link.name == name
    )
    expected = np.asarray(source_body.inertia) * 0.5**2
    offdiagonal = expected.copy()
    offdiagonal[np.diag_indices(3)] = 0.0
    assert np.max(np.abs(offdiagonal)) > np.max(np.abs(expected)) * 0.01
    art.values[name]["inertia"] = torch.tensor(
        np.diag(expected), dtype=torch.float32
    ).repeat(art.num_instances, 1, 1)
    art.values[name]["com"][..., 3:] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="readback mismatch"):
        verify_scaled_mass(art, source.binding())
    assert art.setter_calls == 0


@pytest.mark.parametrize("missing_policy", (False, True))
def test_native_policy_skips_source_and_readbacks_without_writes(
    source, missing_policy
):
    art = source.art(policy="native")
    art.cfg.fpath = "/unreadable/not_requested.usdc"
    if missing_policy:
        del art.cfg.body_scale_mass_policy
    result = verify_scaled_mass(art, SimpleNamespace())
    assert result == {"policy": "native", "verified": False, "scope": "not_requested"}
    assert art.getter_calls == []
    assert art.setter_calls == 0


@pytest.mark.parametrize(
    "com_factor,inertia_factor",
    ((1.0, None), (None, 1.0), (0.25, None), (None, 0.0625)),
)
def test_unscaled_or_double_scaled_readbacks_are_rejected(
    source, com_factor, inertia_factor
):
    art = source.art(com_factor=com_factor, inertia_factor=inertia_factor)
    with pytest.raises(ValueError, match="readback mismatch"):
        verify_scaled_mass(art, source.binding())
    assert art.setter_calls == 0


@pytest.mark.parametrize("field", ("mass", "inertia", "com"))
@pytest.mark.parametrize("bad_value", (float("nan"), float("inf"), -float("inf")))
@pytest.mark.parametrize("env", (0, 1))
def test_nonfinite_public_getter_values_are_rejected(source, field, bad_value, env):
    art = source.art()
    name = next(iter(art.values))
    art.values[name][field][env].reshape(-1)[0] = bad_value
    with pytest.raises(ValueError, match="finite"):
        verify_scaled_mass(art, source.binding())
    assert art.setter_calls == 0


@pytest.mark.parametrize("factor", (0.0, 0.5, 2.0))
def test_unnormalized_xyzw_com_quaternion_is_rejected(source, factor):
    art = source.art()
    name = next(iter(art.values))
    art.values[name]["com"][0, 0, 3:] *= factor
    with pytest.raises(ValueError, match="normalized"):
        verify_scaled_mass(art, source.binding())
    assert art.setter_calls == 0


def test_quaternion_component_order_is_not_silently_reinterpreted(source):
    art = source.art()
    name = next(iter(art.values))
    quaternion = art.values[name]["com"][0, 0, 3:].clone()
    art.values[name]["com"][0, 0, 3:] = quaternion[[3, 0, 1, 2]]
    with pytest.raises(ValueError, match="readback mismatch"):
        verify_scaled_mass(art, source.binding())
    assert art.setter_calls == 0


@pytest.mark.parametrize(
    "field,shape", (("mass", (0, 1)), ("inertia", (0, 1, 3)), ("com", (0, 1, 7)))
)
def test_empty_or_mismatched_public_getter_rows_are_rejected(source, field, shape):
    art = source.art()
    name = next(iter(art.values))
    art.values[name][field] = torch.empty(shape, dtype=torch.float32)
    with pytest.raises(ValueError):
        verify_scaled_mass(art, source.binding())
    assert art.setter_calls == 0


def test_missing_environment_row_is_rejected_even_if_all_field_shapes_agree(source):
    art = source.art(rows=2)
    name = next(iter(art.values))
    for field in art.values[name]:
        art.values[name][field] = art.values[name][field][:1]
    with pytest.raises(ValueError):
        verify_scaled_mass(art, source.binding())
    assert art.setter_calls == 0


@pytest.mark.parametrize(
    "field,shape",
    (("mass", (2,)), ("inertia", (2, 3)), ("com", (2, 7)), ("com", (2, 7, 1))),
)
def test_malformed_public_getter_layouts_are_not_flattened_into_valid_rows(
    source, field, shape
):
    art = source.art()
    name = next(iter(art.values))
    art.values[name][field] = art.values[name][field].reshape(shape)
    with pytest.raises(ValueError):
        verify_scaled_mass(art, source.binding())
    assert art.setter_calls == 0


@pytest.mark.parametrize("field", ("mass", "com_position", "inertia"))
def test_missing_authored_mass_properties_are_rejected(source, monkeypatch, field):
    import dexsim.kit.usd

    descriptor = deepcopy(source.descriptor)
    body = next(
        link.rigid_body for link in descriptor.links if link.rigid_body is not None
    )
    setattr(body, field, None)
    monkeypatch.setattr(
        dexsim.kit.usd,
        "parse_usd",
        lambda _path: SimpleNamespace(articulations=[descriptor]),
    )
    art = source.art()
    with pytest.raises(ValueError, match="authored mass, COM and inertia"):
        verify_scaled_mass(art, source.binding())
    assert art.setter_calls == 0


@pytest.mark.parametrize("bad_mass", (0.0, -1.0))
def test_nonpositive_authored_and_matching_readback_mass_are_rejected(
    source, monkeypatch, bad_mass
):
    import dexsim.kit.usd

    descriptor = deepcopy(source.descriptor)
    body = next(
        link.rigid_body for link in descriptor.links if link.rigid_body is not None
    )
    body.mass = bad_mass
    monkeypatch.setattr(
        dexsim.kit.usd,
        "parse_usd",
        lambda _path: SimpleNamespace(articulations=[descriptor]),
    )
    art = _ReadOnlyArticulation(source.path, descriptor, scale=0.5)
    with pytest.raises(ValueError):
        verify_scaled_mass(art, source.binding())
    assert art.setter_calls == 0


@pytest.mark.parametrize(
    "principal", ((0.0, 0.0, 0.0), (-0.005, 0.007, 0.011), (0.0, 0.007, 0.011))
)
def test_nonpositive_authored_and_matching_readback_inertia_are_rejected(
    source, monkeypatch, principal
):
    import dexsim.kit.usd

    descriptor = deepcopy(source.descriptor)
    body = next(
        link.rigid_body for link in descriptor.links if link.rigid_body is not None
    )
    body.inertia = np.diag(principal)
    monkeypatch.setattr(
        dexsim.kit.usd,
        "parse_usd",
        lambda _path: SimpleNamespace(articulations=[descriptor]),
    )
    art = _ReadOnlyArticulation(source.path, descriptor, scale=0.5)
    with pytest.raises(ValueError):
        verify_scaled_mass(art, source.binding())
    assert art.setter_calls == 0


@pytest.mark.parametrize("articulation_count", (0, 2))
def test_ambiguous_or_absent_source_articulation_is_rejected(
    source, monkeypatch, articulation_count
):
    import dexsim.kit.usd

    monkeypatch.setattr(
        dexsim.kit.usd,
        "parse_usd",
        lambda _path: SimpleNamespace(
            articulations=[source.descriptor] * articulation_count
        ),
    )
    art = source.art()
    with pytest.raises(ValueError, match="one source articulation"):
        verify_scaled_mass(art, source.binding())
    assert art.setter_calls == 0


def test_source_with_no_authored_physical_links_is_rejected(source, monkeypatch):
    import dexsim.kit.usd

    descriptor = deepcopy(source.descriptor)
    for link in descriptor.links:
        link.rigid_body = None
    monkeypatch.setattr(
        dexsim.kit.usd,
        "parse_usd",
        lambda _path: SimpleNamespace(articulations=[descriptor]),
    )
    art = source.art()
    with pytest.raises(ValueError, match="no authored physical links"):
        verify_scaled_mass(art, source.binding())
    assert art.setter_calls == 0


def test_source_hash_drift_is_rejected_before_getters(source):
    art = source.art()
    binding = source.binding()
    binding.source_sha256 = "0" * 64
    with pytest.raises(ValueError, match="source hash differs"):
        verify_scaled_mass(art, binding)
    assert art.getter_calls == []
    assert art.setter_calls == 0


def test_source_path_drift_to_other_asset_is_rejected(source, tmp_path):
    art = source.art()
    other = tmp_path / "other.usda"
    other.write_text("#usda 1.0\n")
    art.cfg.fpath = str(other)
    with pytest.raises(ValueError, match="source hash differs"):
        verify_scaled_mass(art, source.binding())
    assert art.getter_calls == []
    assert art.setter_calls == 0


def test_identical_source_bytes_at_another_path_keep_content_bound_equivalence(
    source, tmp_path
):
    art = source.art()
    relocated = tmp_path / "relocated.usda"
    relocated.write_bytes(source.path.read_bytes())
    art.cfg.fpath = str(relocated)
    assert verify_scaled_mass(art, source.binding())["verified"] is True
    assert art.setter_calls == 0
