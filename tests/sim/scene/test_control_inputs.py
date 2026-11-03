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
"""Source controls retain explicit identity, geometry and mass lineage."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from embodichain.lab.sim.scene.articulation_geometry import read_articulation_geometry
from embodichain.lab.sim.scene.control_mass_source import (
    create_mass_source_copy,
    qualify_mass_source_copy,
)
from embodichain.lab.sim.scene.control_semantics import (
    resolve_twist_semantics,
    InvalidTwistSemanticsError,
)
from embodichain.lab.sim.scene.revolute_inputs import inspect_revolute_control
from embodichain.lab.sim.scene.revolute_geometry import load_twist_geometry
from ._control_source_fixture import knob_source


def test_declared_control_targets_and_geometry_remain_separate_from_scene_pose(
    tmp_path,
):
    config = knob_source(tmp_path)
    binding = inspect_revolute_control(config, 90)
    assert binding.target_qpos == pytest.approx(-math.pi / 2)
    assert binding.axis_sign == -1
    assert binding.axis == (0.0, 0.0, -1.0)
    assert binding.origin == (0.0, 0.0, 0.0)
    assert binding.grip_width == pytest.approx(math.sqrt(0.06**2 * 2))
    geometry = load_twist_geometry(config["fpath"], binding)
    assert geometry.grip.path == "/Control/knob/grip"
    assert geometry.source_sha256 == binding.source_sha256
    assert len(geometry.target_collisions) == 2
    assert len(geometry.parent_collisions) == 3
    assert geometry.grip.vertices.flags.writeable is False
    layout = read_articulation_geometry(config["fpath"])
    copy = deepcopy(layout)
    assert np.array_equal(copy.vertices, layout.vertices)
    assert not np.shares_memory(copy.vertices, layout.vertices)
    assert np.allclose(layout.world_vertices(config) - layout.vertices, [0, 0, 0.8])
    # New declarations retain authored IDs; a different requested label is not guessed.
    with pytest.raises(ValueError, match="not calibrated"):
        inspect_revolute_control(config, 45)


def test_mass_preparation_retains_original_bytes_and_seals_only_the_declared_operation(
    tmp_path,
):
    config = knob_source(tmp_path)
    path = Path(config["fpath"])
    original = path.read_bytes()
    original_annotation = path.with_suffix(".twist.json").read_bytes()
    prepared = create_mass_source_copy(path, tmp_path / "derived", (2.0, 2.0, 2.0))
    assert path.read_bytes() == original
    assert path.with_suffix(".twist.json").read_bytes() == original_annotation
    assert (
        qualify_mass_source_copy(
            prepared.runtime, manifest_sha256=prepared.manifest_sha256
        )
        == prepared
    )
    from embodichain.lab.sim.scene.control_source import capture_revolute_source_seal

    derived = dict(config, fpath=str(prepared.runtime), body_scale=[2.0] * 3)
    binding = inspect_revolute_control(derived, 90)
    seal = capture_revolute_source_seal(prepared.runtime, binding)
    seal.assert_current(source_path=prepared.runtime, body_scale=[2.0] * 3)
    with pytest.raises(ValueError, match="Lab conversion"):
        seal.assert_current(
            source_path=prepared.runtime,
            body_scale=[2.0] * 3,
            body_scale_mass_policy="geometric",
        )
    pristine = prepared.source.read_bytes()
    prepared.source.write_bytes(pristine + b"\n")
    with pytest.raises(ValueError, match="changed"):
        seal.assert_current(source_path=prepared.runtime, body_scale=[2.0] * 3)
    prepared.source.write_bytes(pristine)
    # Keep the stage alive while querying its prim.
    stage = Usd.Stage.Open(str(prepared.runtime))
    mass = UsdPhysics.MassAPI(stage.GetPrimAtPath("/Control/knob"))
    assert tuple(mass.GetDiagonalInertiaAttr().Get()) == pytest.approx(
        (0.04, 0.08, 0.12)
    )
    assert mass.GetMassAttr().Get() == pytest.approx(0.02)
    mass.CreateMassAttr(0.5)
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError):
        qualify_mass_source_copy(prepared.runtime)
    with pytest.raises(ValueError, match="Default-only"):
        create_mass_source_copy(
            path, tmp_path / "unsupported", (1.0, 1.0, 1.0), backend="newton"
        )


def test_annotation_replacement_does_not_reuse_cached_usd_identity(tmp_path):
    config = knob_source(tmp_path)
    path = Path(config["fpath"])
    first = resolve_twist_semantics(path)
    assert first.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = path.with_suffix(".twist.json")
    sidecar.write_text(
        sidecar.read_text().replace("/Control/knob/grip", "/Control/knob/missing")
    )
    with pytest.raises(InvalidTwistSemanticsError):
        resolve_twist_semantics(path)


def test_source_seal_rejects_changed_sidecar_presence_and_scale(tmp_path):
    from embodichain.lab.sim.scene.control_source import capture_revolute_source_seal

    config = knob_source(tmp_path)
    binding = inspect_revolute_control(config, 90)
    path = Path(config["fpath"])
    seal = capture_revolute_source_seal(path, binding)
    seal.assert_current(source_path=path, body_scale=[1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="scale"):
        seal.assert_current(source_path=path, body_scale=[1.0, 1.0, 1.01])
    path.with_suffix(".mass.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        seal.assert_current(source_path=path, body_scale=[1.0, 1.0, 1.0])


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


def test_press_binding_preserves_authored_stroke_and_contact_frame(button):
    from embodichain.lab.sim.scene.press_inputs import inspect_press

    binding = inspect_press(button, joint="press_joint", link="cap")
    assert binding.axis == (0.0, 0.0, -1.0)
    assert binding.limits == pytest.approx((-0.01, 0.0))
    assert binding.stroke == pytest.approx(0.01)
    assert binding.press_position == pytest.approx((0.0, 0.0, 0.02))
    assert binding.released_position == 0.0
    assert binding.pressed_position == pytest.approx(-0.01)
    with pytest.raises(ValueError, match="endpoint"):
        inspect_press(button, joint="press_joint", link="cap", released_position=-0.003)
    with pytest.raises(ValueError, match="missing or ambiguous"):
        inspect_press(button, joint="press_joint", link="guessed_button")
