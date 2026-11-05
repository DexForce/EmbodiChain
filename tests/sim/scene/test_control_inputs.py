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
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from embodichain.lab.sim.spawn.source_geometry import read_usd_source

from embodichain.lab.sim.scene.articulation_geometry import read_articulation_geometry
from embodichain.lab.sim.scene.control_mass_source import (
    create_mass_source_copy,
    qualify_mass_source_copy,
)
from embodichain.lab.sim.scene.control_semantics import (
    resolve_twist_semantics,
    InvalidTwistSemanticsError,
    MissingTwistSemanticsError,
)
from embodichain.lab.sim.scene.revolute_inputs import inspect_revolute_control
from embodichain.lab.sim.scene.revolute_geometry import load_twist_geometry
from ._control_source_fixture import knob_source


def test_declared_control_targets_and_geometry_remain_separate_from_scene_pose(
    tmp_path,
):
    config = knob_source(tmp_path)
    source = read_usd_source(config["fpath"])
    binding = inspect_revolute_control(config, 90, source=source)
    assert binding.target_qpos == pytest.approx(-math.pi / 2)
    from embodichain.lab.sim.scene.revolute_inputs import revolute_setting_attributes

    attributes = revolute_setting_attributes(config)
    assert attributes["twist_setting_labels"] == pytest.approx(
        {"0": 0.0, "90": -math.pi / 2}
    )
    assert attributes["twist_control_joint"] == binding.joint
    assert attributes["twist_source_sha256"] == binding.source_sha256
    assert revolute_setting_attributes({"fix_base": False}) == {}
    assert binding.axis_sign == -1
    assert binding.axis == (0.0, 0.0, -1.0)
    assert binding.origin == (0.0, 0.0, 0.0)
    assert binding.grip_width == pytest.approx(math.sqrt(0.06**2 * 2))
    geometry = load_twist_geometry(source, binding)
    assert geometry.grip.path == "/Control/knob/grip"
    assert geometry.source_sha256 == binding.source_sha256
    assert len(geometry.target_collisions) == 2
    assert len(geometry.parent_collisions) == 3
    assert geometry.grip.vertices.flags.writeable is False
    layout = read_articulation_geometry(source)
    copy = deepcopy(layout)
    assert np.array_equal(copy.vertices, layout.vertices)
    assert not np.shares_memory(copy.vertices, layout.vertices)
    assert np.allclose(layout.world_vertices(config) - layout.vertices, [0, 0, 0.8])
    # New declarations retain authored IDs; a different requested label is not guessed.
    with pytest.raises(ValueError, match="not calibrated"):
        inspect_revolute_control(config, 45)

    from embodichain.lab.sim.scene.control_source import capture_revolute_source_seal

    seal = capture_revolute_source_seal(source, binding)
    seal.assert_current(source_path=config["fpath"], body_scale=[1.0] * 3)
    with pytest.raises(ValueError, match="selection changed"):
        inspect_revolute_control(dict(config, fpath="other.usda"), 90, source=source)


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


@pytest.mark.parametrize("reader", ["press", "object"])
def test_source_readers_ignore_stale_usd_layers(button, reader):
    """Updated bytes must produce updated geometry even while a cached layer lives."""
    from embodichain.lab.sim.scene.press_inputs import inspect_press
    from embodichain.lab.sim.objects.backends._collision_inputs import (
        _usd_collision_meshes,
    )

    path = Path(button["fpath"])
    cached = Usd.Stage.Open(str(path))
    layer = Sdf.Layer.OpenAsAnonymous(str(path))
    updated = Usd.Stage.Open(layer)
    joint = UsdPhysics.PrismaticJoint(updated.GetPrimAtPath("/Button/press_joint"))
    joint.GetLowerLimitAttr().Set(-0.01)
    mesh = UsdGeom.Mesh(updated.GetPrimAtPath("/Button/cap/contact"))
    mesh.GetPointsAttr().Set([(x, y, 0.02) for x, y, _ in mesh.GetPointsAttr().Get()])
    layer.Export(str(path))
    assert UsdPhysics.PrismaticJoint(
        cached.GetPrimAtPath("/Button/press_joint")
    ).GetLowerLimitAttr().Get() == pytest.approx(-0.005)
    if reader == "press":
        binding = inspect_press(button, joint="press_joint", link="cap")
        assert binding.stroke == pytest.approx(0.02)
        assert binding.press_position == pytest.approx((0.0, 0.0, 0.04))
        assert binding.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        meshes = _usd_collision_meshes(button)
        assert list(meshes) == [("cap", "/Button/cap/contact")]
        assert meshes[("cap", "/Button/cap/contact")].vertices[:, 2] == pytest.approx(
            [0.04] * 4
        )


def test_shared_source_snapshot_rejects_changed_bytes(button):
    from embodichain.lab.sim.scene.press_inputs import inspect_press

    path = Path(button["fpath"])
    source = read_usd_source(path)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source changed"):
        inspect_press(button, joint="press_joint", link="cap", source=source)


def test_missing_control_annotation_precedes_geometry_qualification(button):
    stage = Usd.Stage.Open(button["fpath"])
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    stage.GetRootLayer().Save()
    with pytest.raises(MissingTwistSemanticsError):
        resolve_twist_semantics(button["fpath"])
    with pytest.raises(ValueError, match="metre-authored"):
        read_usd_source(button["fpath"]).require_geometry()


def test_prismatic_and_object_inputs_share_link_frames_and_source_topology(button):
    from embodichain.lab.sim.scene.articulation_inputs import (
        inspect_prismatic,
        handle_mesh,
    )
    from embodichain.lab.sim.objects.backends._collision_inputs import (
        _usd_collision_meshes,
    )

    stage = Usd.Stage.Open(button["fpath"])
    layer = stage.GetRootLayer()
    path = "/Button/cap/pull_grip"
    Sdf.CopySpec(layer, "/Button/cap/contact", layer, path)
    stage.RemovePrim("/Button/cap/contact")
    layer.Save()
    source = read_usd_source(button["fpath"])
    binding = inspect_prismatic(button, source=source)
    vertices, faces = handle_mesh(binding, source)
    assert binding.handle_path == path
    assert binding.axis == (0.0, 0.0, 1.0)
    assert binding.target("closed") == 0.0
    assert binding.target("open") == pytest.approx(-0.01)
    assert vertices.dtype == np.float64
    np.testing.assert_allclose(
        vertices,
        [
            [-0.04, -0.04, 0.02],
            [0.04, -0.04, 0.02],
            [0.04, 0.04, 0.02],
            [-0.04, 0.04, 0.02],
        ],
    )
    np.testing.assert_array_equal(faces, [[0, 1, 2], [0, 2, 3]])
    object_mesh = _usd_collision_meshes(button)[("cap", path)]
    np.testing.assert_allclose(object_mesh.bounds, [vertices.min(0), vertices.max(0)])
    with pytest.raises(ValueError, match="does not belong"):
        source.link_mesh(path, "/Button/base", scale=2.0)


def test_control_mass_readback_preserves_fixed_mass_scaling_and_rejects_double_conversion(
    tmp_path,
):
    import torch
    from types import SimpleNamespace
    from embodichain.lab.sim.scene.control_mass_observation import (
        verify_scaled_control_mass,
    )

    config = knob_source(tmp_path)
    config["body_scale"] = (2.0, 2.0, 2.0)
    binding = inspect_revolute_control(config, 90)
    art = SimpleNamespace(
        cfg=SimpleNamespace(fpath=config["fpath"], body_scale_mass_policy="fixed_mass"),
        num_instances=1,
        get_mass=lambda name: torch.tensor([[0.02]]),
        get_inertia=lambda name: torch.tensor([[[0.04, 0.08, 0.12]]]),
        get_com_pose=lambda name: torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]]),
    )
    result = verify_scaled_control_mass(art, binding)
    assert result["verified"] and len(result["links"]) == 2
    art.get_inertia = lambda name: torch.tensor([[[0.16, 0.32, 0.48]]])
    with pytest.raises(ValueError, match="readback mismatch"):
        verify_scaled_control_mass(art, binding)


def test_native_control_geometry_keeps_axial_and_park_metrics_distinct():
    import torch
    from types import SimpleNamespace
    from embodichain.lab.sim.scene.press_inputs import PressBinding
    from embodichain.lab.sim.scene.control_measurement import NativeControlGeometry

    pose = torch.eye(4)[None]
    faces = torch.empty(0, 3, dtype=torch.long)
    finger = torch.tensor([[0.05, -0.01, -0.01], [0.06, 0.01, 0.01]])
    target = torch.tensor([[-0.01, -0.01, -0.01], [0.01, 0.01, 0.01]])
    calls = []
    robot = SimpleNamespace(
        num_instances=1,
        link_names=("finger",),
        get_link_pose=lambda name, **kwargs: pose.clone(),
        get_link_vert_face=lambda name: (finger, faces),
        get_qpos=lambda **kwargs: torch.zeros(1, 1),
        compute_fk=lambda **kwargs: pose.clone(),
    )

    def mesh(name):
        calls.append(name)
        return (target if name == "button" else torch.empty(0, 3)), faces

    art = SimpleNamespace(
        num_instances=1,
        link_names=("button", "base"),
        get_link_vert_face=mesh,
        get_link_pose=lambda name, **kwargs: pose.clone(),
    )
    binding = PressBinding(
        "control",
        "joint",
        "button",
        "base",
        "source.usda",
        "a" * 64,
        "/mesh",
        1.0,
        (1.0, 0.0, 0.0),
        (0.0, 0.01),
        (0.1, 0.0, 0.0),
        0.0,
        0.01,
    )
    geometry = NativeControlGeometry(
        robot,
        art,
        finger_link_names=("finger",),
        motion_part="arm",
        press_binding=binding,
    )
    retreat = geometry.press_clearances(terminal=False)
    assert retreat["retreat_clearance"] == pytest.approx(0.04)
    assert retreat["tcp_clearance"] == pytest.approx(0.1)
    assert retreat["terminal_clearance"] is None and not calls
    terminal = geometry.press_clearances(terminal=True)
    assert terminal["terminal_clearance"] == pytest.approx(0.04)
    assert calls == ["button", "base"]
