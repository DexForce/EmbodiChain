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

import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from embodichain.gen_sim.assembly import import_mjcf
from embodichain.gen_sim.assembly.probes import (
    AnalyticPegGeometry,
    ConstrainedProbeCfg,
    prepare_probe,
    run_reference_probe,
)


def _square_asset():
    # Independent four-box plate: exact square hole of half-width 8 mm,
    # plate half-width 24 mm, thickness 15 mm, and a 40 mm primitive peg.
    xml = """<mujoco><option timestep="0.001"/><default><geom solref="0.002 1" solimp="0.99 0.999 0.001 0.5 2"/></default><worldbody>
    <body name="plate" pos="0 0 0.0075">
      <geom type="box" pos="-0.016 0 0" size="0.008 0.024 0.0075"/>
      <geom type="box" pos="0.016 0 0" size="0.008 0.024 0.0075"/>
      <geom type="box" pos="0 -0.016 0" size="0.008 0.008 0.0075"/>
      <geom type="box" pos="0 0.016 0" size="0.008 0.008 0.0075"/>
    </body><body name="peg" pos="0 0 0.03"><freejoint name="peg_joint"/>
      <geom type="box" size="0.0079 0.0079 0.020"/>
    </body></worldbody></mujoco>"""
    return import_mjcf(
        xml,
        config={
            "kind": "peg",
            "hole": {"type": "rectangle", "width": 0.016, "height": 0.016},
            "clearance": 0.0001,
            "plate_size": 0.048,
            "plate_thickness": 0.015,
            "peg_length": 0.040,
        },
    )


def test_independent_analytic_translation_and_rotated_bounds() -> None:
    circle = AnalyticPegGeometry(shape="circle", clearance=20e-6)
    square = AnalyticPegGeometry(shape="square", clearance=20e-6)
    diagonal = np.array([1.0, 1.0]) / math.sqrt(2)
    assert circle.lateral_support(diagonal) == pytest.approx(20e-6)
    assert square.lateral_support(diagonal) == pytest.approx(20e-6 * math.sqrt(2))
    for geometry in (circle, square):
        angle = geometry.tilt_limit()
        # At first contact, the tilted side projection and the plate half
        # thickness consume exactly the available hole radius/half-width.
        contact_radius = geometry.peg_half_size / math.cos(
            angle
        ) + geometry.thickness / 2 * math.tan(angle)
        assert contact_radius == pytest.approx(geometry.hole_half_size, abs=1e-12)
    yaw = square.yaw_limit()
    assert square.peg_half_size * (math.cos(yaw) + math.sin(yaw)) == pytest.approx(
        square.hole_half_size, abs=1e-12
    )
    with pytest.raises(ValueError, match="circular peg"):
        circle.yaw_limit()


def test_static_gap_scan_detects_exact_square_contacts() -> None:
    result = run_reference_probe(
        _square_asset(),
        ConstrainedProbeCfg(probe="S0", num_directions=4, gaps_um=(20.0, -20.0)),
    )
    assert result["status"] == "passed"
    assert len(result["rows"]) == 8
    for row in result["rows"]:
        assert row["contact"] == (row["true_wall_gap_m"] < 0)


@pytest.mark.parametrize("probe", ["S2", "S3", "S4"])
def test_constrained_probe_integrates_without_pose_drives_and_retains_joint_export(
    probe, tmp_path
) -> None:
    asset = _square_asset()
    cfg = ConstrainedProbeCfg(probe=probe, num_directions=4, duration=0.3)
    definition = prepare_probe(asset, cfg)
    model_xml = ET.fromstring(definition.source_mjcf)
    peg = model_xml.find("worldbody/body[@name='peg']")
    assert [joint.get("type") for joint in peg.findall("joint")] == (
        ["slide", "slide"] if probe == "S2" else ["slide", "slide", "hinge"]
    )
    # The source asset remains unchanged; the constrained probe has no actuator.
    assert model_xml.find("actuator") is None
    assert "freejoint" in asset.source_mjcf
    path = definition.write_urdf(tmp_path)
    urdf = ET.parse(path).getroot()
    assert [joint.get("type") for joint in urdf.findall("joint")] == (
        ["prismatic", "prismatic"]
        if probe == "S2"
        else ["prismatic", "prismatic", "revolute"]
    )
    assert len(urdf.findall("link/collision")) == 1
    result = run_reference_probe(asset, cfg)
    assert result["status"] == "passed"
    assert all(row["contact_samples"] > 0 for row in result["rows"])
    assert all(
        abs(row["wall_error_m"]) <= result["wall_error_tolerance_m"]
        for row in result["rows"]
    )


@pytest.mark.parametrize(
    "errors",
    [
        np.array([-10.0, 10.0] * 8) * 1e-6,
        np.array([-4.0] * 4 + [0.0] * 8 + [4.0] * 4) * 1e-6,
    ],
)
def test_quasistatic_acceptance_rejects_oscillation_or_drift_with_zero_median(errors):
    from embodichain.gen_sim.assembly.probes import _settled_error_statistics

    statistics = _settled_error_statistics(errors, 5e-6)
    assert statistics["wall_error_m"] == pytest.approx(0.0)
    assert statistics["finite"]
    assert not statistics["passed"]
    assert (
        statistics["wall_worst_error_m"] > 5e-6
        or abs(statistics["wall_drift_m"]) > 5e-6
    )


def test_initial_overlap_only_counts_penetrating_plate_peg_contacts():
    from embodichain.gen_sim.assembly.probes import _contact_overlap_worlds

    # World0 has a separated plate/peg pair and an unrelated overlap;
    # world1 has one real plate/peg overlap; world2 starts with no contacts.
    overlap = _contact_overlap_worlds(
        np.array([0, 1, 2, 3]),
        np.array([[1, 2], [1, 3], [2, 1]]),
        np.array([0, 0, 1]),
        np.array([2e-6, -1e-3, -2e-6]),
        3,
        1,
        2,
    )
    np.testing.assert_array_equal(overlap, [False, True, False])


def test_mjfit_nested_default_contact_profile_is_retained_on_analytic_probe_peg():
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("mjfit")
    from embodichain.gen_sim.assembly import build_assembly

    asset = build_assembly("circle")
    source_root = ET.fromstring(asset.source_mjcf)
    assert any(
        default.get("class") == "mjfit" for default in source_root.iter("default")
    )
    definition = prepare_probe(asset, ConstrainedProbeCfg(probe="S2"))
    compiled = mujoco.MjModel.from_xml_string(definition.source_mjcf)
    plate_id = compiled.body("plate").id
    peg_id = compiled.geom("analytic_peg").id
    plate_geoms = np.flatnonzero(
        (compiled.geom_bodyid == plate_id) & (compiled.geom_contype != 0)
    )
    assert len(plate_geoms) > 1
    # These are mjfit's declared contact time constant and damping ratio,
    # independent of how the probe builder copies or resolves default classes.
    np.testing.assert_allclose(
        compiled.geom_solref[peg_id], [0.002, 1.0], atol=0, rtol=0
    )
    for plate_geom in plate_geoms:
        for field in ("geom_solref", "geom_solimp", "geom_solmix", "geom_priority"):
            values = getattr(compiled, field)
            np.testing.assert_array_equal(values[peg_id], values[plate_geom])
