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


"""Physical control preparation retains full original support and source geometry."""

from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pytest
from ._control_source_fixture import knob_source
from embodichain.lab.sim.scene.revolute_inputs import inspect_revolute_control
from embodichain.lab.sim.scene.control_preparation import (
    ControlSourceScene,
    ControlSourceResizeRequest,
    prepare_control_source_scene,
)
from embodichain.lab.sim.scene.source_measurement import measure_source_geometry

CASES = json.loads(
    (Path(__file__).parent / "fixtures/control_preparation_b0.json").read_text()
)["cases"]


@pytest.mark.parametrize("case", CASES)
def test_full_source_control_preparation_matches_original_results(tmp_path, case):
    art = knob_source(tmp_path)
    art["body_scale"] = [case["scale"]] * 3
    binding = inspect_revolute_control(art, 90)
    table = dict(
        uid="table",
        shape=dict(shape_type="Cube", size=[1.0, 1.0, 0.1]),
        init_pos=[0.0, 0.0, 0.65],
        init_rot=[0.0, 0.0, 0.0],
        body_scale=[1.0, 1.0, 1.0],
    )
    source = ControlSourceScene((table,), (), (art,))
    before = deepcopy(art)
    result, records = prepare_control_source_scene(
        source,
        (ControlSourceResizeRequest(binding),),
        interaction_kind="revolute",
        support_lift_m=case["lift"],
    )
    expected = deepcopy(case["records"])
    expected[0]["source_sha256"] = binding.source_sha256
    expected[0]["support_clearance"]["source_sha256"] = binding.source_sha256
    assert records == expected
    assert result.articulations[0]["init_pos"] == records[0]["adapted_init_pos"]
    assert art == before
    assert result.articulations[0]["body_scale_mass_policy"] == "fixed_mass"


def test_source_measurement_keeps_primitive_units_and_unknown_shape_unavailable():
    cube = dict(
        shape=dict(shape_type="Cube", size=[2.0, 4.0, 6.0]),
        body_scale=[2.0, 1.0, 0.5],
        init_pos=[1.0, 2.0, 3.0],
        init_rot=[0.0, 0.0, 0.0],
    )
    measured = measure_source_geometry(cube, convert_y_up=True)
    np.testing.assert_array_equal(
        measured["bounds"], [[-1.0, 0.0, 1.5], [3.0, 4.0, 4.5]]
    )
    np.testing.assert_array_equal(measured["local_extents"], [4.0, 4.0, 3.0])
    assert (
        measure_source_geometry(
            dict(shape=dict(shape_type="Unsupported")), convert_y_up=False
        )
        is None
    )


@pytest.fixture
def authored_press(tmp_path):
    from pxr import Usd, UsdGeom, UsdPhysics

    path = tmp_path / "button.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    root = UsdGeom.Xform.Define(stage, "/Button")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    for name in ("base", "cap"):
        body = UsdGeom.Xform.Define(stage, f"/Button/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        UsdPhysics.MassAPI.Apply(body).CreateMassAttr(0.02)
    joint = UsdPhysics.PrismaticJoint.Define(stage, "/Button/press_joint")
    joint.CreateBody0Rel().SetTargets(["/Button/base"])
    joint.CreateBody1Rel().SetTargets(["/Button/cap"])
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(-0.005)
    joint.CreateUpperLimitAttr(0.0)
    mesh = UsdGeom.Mesh.Define(stage, "/Button/cap/button_surface")
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
        "body_scale": [1.0, 1.0, 1.0],
        "init_pos": [0.0, 0.0, 0.8],
        "init_rot": [0.0, 0.0, 0.0],
    }


def test_authored_press_discovery_and_explicit_declaration_calibration(authored_press):
    from embodichain.lab.sim.scene.press_inputs import discover_press_source
    from embodichain.lab.sim.scene.press_calibration import (
        PressSourceCalibrationRequest,
        calibrate_press_source_declarations,
    )
    from embodichain.lab.sim.scene.control_mass_source import (
        requires_derived_revolute_mass_source,
    )
    from embodichain.lab.sim.cfg import ArticulationCfg

    before = deepcopy(authored_press)
    asset = Path(authored_press["fpath"])
    original = asset.read_bytes()
    binding, mass = discover_press_source(authored_press)
    assert (
        binding.joint == "press_joint"
        and binding.link == "cap"
        and mass == pytest.approx(0.02)
    )
    owned, records = calibrate_press_source_declarations(
        [authored_press], [PressSourceCalibrationRequest(binding, binding, mass, mass)]
    )
    assert owned[0]["qpos_limits"] == {binding.joint: list(binding.limits)}
    assert owned[0]["joint_drive_props"]["target_mode"] == {
        binding.joint: "position_velocity"
    }
    assert (
        records[0]["runtime_mass"] == pytest.approx(0.02)
        and records[0]["source_edited"] is False
    )
    assert authored_press == before and asset.read_bytes() == original
    with pytest.raises(ValueError, match="authored runtime drive/limit"):
        calibrate_press_source_declarations(
            owned, [PressSourceCalibrationRequest(binding, binding, mass, mass)]
        )
    assert requires_derived_revolute_mass_source() is (
        "body_scale_mass_policy" not in ArticulationCfg.__dataclass_fields__
    )


@pytest.fixture(autouse=True)
def no_runtime_creation(monkeypatch):
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.lab.sim.objects import Robot

    def forbidden(*args, **kwargs):
        pytest.fail("static source control preparation created runtime")

    monkeypatch.setattr(SimulationManager, "__init__", forbidden)
    monkeypatch.setattr(Robot, "__init__", forbidden)
