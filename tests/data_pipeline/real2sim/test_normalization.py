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

"""Unit/frame/time normalization without DROID files or experimental code."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from embodichain.data_pipeline.real2sim import (
    EefPoseSamples,
    MimicJoint,
    Real2SimTrajectory,
    RobotJointLayout,
    SourcePhase,
)
from embodichain.data_pipeline.real2sim.normalization import (
    Real2SimNormalizationCfg,
    normalize_real2sim,
)


def source():
    return Real2SimTrajectory(
        source_id="fixture",
        source_revision="raw-v1",
        unit_id="pick-0",
        robot_id="test_robot",
        robot_revision="urdf-v1",
        joint_names=("slide", "angle"),
        joint_units=("mm", "deg"),
        positions=np.array(
            [[0.0, 0.0], [1.0, 5.0], [2.0, 10.0], [3.0, 15.0], [4.0, 20.0]]
        ),
        timestamps=np.array([1000, 1071, 1142, 1210, 1281]),
        time_unit="ms",
        calibration_revision="cal-v1",
        annotation_revision="phase-v1",
        calibration_status="verified",
        annotation_status="verified",
        phases=(
            SourcePhase("approach", 0, 3, "free", ("joint_residual",)),
            SourcePhase("contact", 3, 5, "contact"),
        ),
    )


def layout():
    return RobotJointLayout(
        "test_robot",
        "urdf-v1",
        ("angle", "slide", "follower", "unused"),
        ("rad", "m", "rad", "rad"),
        np.array([0.0, 0.0, 0.2, 0.4]),
        ("angle",),
        (MimicJoint("follower", "angle", -1.0, 0.2),),
    )


def test_mapping_units_mimics_hold_and_clock_are_explicit():
    raw = source()
    target = layout()
    unit = normalize_real2sim(
        raw,
        target,
        Real2SimNormalizationCfg(control_dt=0.01, hold_joint_names=("unused",)),
    )
    np.testing.assert_allclose(
        unit.positions[-1], [np.deg2rad(20), 0.004, 0.2 - np.deg2rad(20), 0.4]
    )
    np.testing.assert_array_equal(unit.positions[:, 2], 0.2 - unit.positions[:, 0])
    np.testing.assert_allclose(
        unit.velocities[:, 2], -unit.velocities[:, 0], atol=1e-12
    )
    np.testing.assert_array_equal(unit.positions[:, 3], 0.4)
    assert unit.eligible
    assert unit.report["held_positions"]["unused"] == 0.4
    assert unit.phases[1].kind == "contact"
    assert unit.phases[1].stop - unit.phases[1].start > 1
    assert unit.time_map.execution_times[-1] == pytest.approx(0.29)
    assert (
        unit.revision
        == normalize_real2sim(
            raw,
            target,
            Real2SimNormalizationCfg(control_dt=0.01, hold_joint_names=("unused",)),
        ).revision
    )


def test_missing_joint_requires_hold_and_mimic_cannot_be_expanded():
    with pytest.raises(ValueError, match="Missing independent joints"):
        normalize_real2sim(
            source(), layout(), Real2SimNormalizationCfg(control_dt=0.01)
        )
    with pytest.raises(ValueError, match="independent"):
        replace(layout(), expandable_joint_names=("follower",))


def test_gripper_encoding_cannot_be_guessed():
    with pytest.raises(ValueError, match="explicit supported unit"):
        replace(source(), joint_units=("mm", "normalized"))


def test_immutable_data_and_revision_changes():
    raw = source()
    before = raw.fingerprint()
    with pytest.raises(ValueError):
        raw.positions.setflags(write=True)
    assert before != replace(raw, annotation_revision="phase-v2").fingerprint()
    cfg = Real2SimNormalizationCfg(control_dt=0.01, hold_joint_names=("unused",))
    a = normalize_real2sim(raw, layout(), cfg)
    b = normalize_real2sim(
        raw, replace(layout(), expandable_joint_names=("slide",)), cfg
    )
    assert a.revision != b.revision
    with pytest.raises(TypeError):
        a.report["held_positions"]["unused"] = 1.0


def test_unverified_data_is_diagnostic_only_and_rejected_data_always_fails():
    raw = replace(source(), calibration_status="unverified")
    cfg = Real2SimNormalizationCfg(control_dt=0.01, hold_joint_names=("unused",))
    with pytest.raises(ValueError, match="diagnostic"):
        normalize_real2sim(raw, layout(), cfg)
    cfg.allow_unverified = True
    assert not normalize_real2sim(raw, layout(), cfg).eligible
    with pytest.raises(ValueError, match="Rejected"):
        normalize_real2sim(replace(raw, calibration_status="rejected"), layout(), cfg)


def test_source_pose_conversion_does_not_claim_independent_measurement():
    raw = source()
    poses = np.tile([1000.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], (5, 1))
    reference = np.eye(4)
    reference[1, 3] = 2.0
    tcp = np.eye(4)
    tcp[2, 3] = 0.15
    eef = EefPoseSamples(
        poses, "camera", "flange", "tcp", reference, tcp, "fk_derived", "wxyz", "mm"
    )
    unit = normalize_real2sim(
        replace(raw, eef=eef),
        layout(),
        Real2SimNormalizationCfg(control_dt=0.01, hold_joint_names=("unused",)),
    )
    np.testing.assert_allclose(
        unit.eef_poses[:, :3], np.tile([1.0, 2.0, 0.15], (len(unit.positions), 1))
    )
    assert unit.report["independent_eef_measurement"] is False
    description = unit.report_dict()["source_description"]["eef"]
    assert description["reference_frame"] == "camera"
    assert description["quaternion_order"] == "wxyz"
    assert description["position_unit"] == "mm"
    np.testing.assert_array_equal(description["link_to_tcp"], tcp)
    np.testing.assert_array_equal(description["reference_to_base"], reference)


def test_report_preserves_source_declarations_and_array_descriptors():
    raw = replace(
        source(), provenance={"corrections": [{"reason": "unit declaration"}]}
    )
    unit = normalize_real2sim(
        raw,
        layout(),
        Real2SimNormalizationCfg(control_dt=0.01, hold_joint_names=("unused",)),
    )
    report = json.loads(json.dumps(unit.report_dict()))
    assert report["source_revision"] == raw.source_revision
    assert report["source_description"]["joint_names"] == ["slide", "angle"]
    assert report["source_description"]["joint_units"] == ["mm", "deg"]
    assert report["source_description"]["time_unit"] == "ms"
    assert report["source_description"]["provenance"]["corrections"][0] == {
        "reason": "unit declaration"
    }
    for name, descriptor in report["time_mapping_arrays"].items():
        array = getattr(unit.time_map, name)
        assert descriptor == {
            "key": "map_" + name,
            "shape": list(array.shape),
            "dtype": array.dtype.str,
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }


def test_nan_duplicate_names_duplicate_clock_and_mimic_cycles_fail():
    with pytest.raises(ValueError):
        replace(source(), joint_names=("angle", "angle"))
    with pytest.raises(ValueError):
        replace(source(), timestamps=np.array([0, 1, 1, 2, 3]))
    with pytest.raises(ValueError):
        replace(source(), positions=np.full((5, 2), np.nan))
    with pytest.raises(ValueError, match="cycle"):
        replace(
            layout(),
            mimics=(MimicJoint("angle", "follower"), MimicJoint("follower", "angle")),
            expandable_joint_names=(),
        )


def test_initial_state_and_robot_revision_mismatch_fail():
    cfg = Real2SimNormalizationCfg(control_dt=0.01, hold_joint_names=("unused",))
    with pytest.raises(ValueError, match="same robot"):
        normalize_real2sim(source(), replace(layout(), robot_revision="other"), cfg)
    with pytest.raises(ValueError, match="initial positions"):
        normalize_real2sim(
            source(),
            replace(layout(), initial_positions=np.array([0.1, 0, 0.1, 0.4])),
            cfg,
        )


def test_integer_nanosecond_epoch_is_subtracted_before_conversion():
    raw = replace(
        source(),
        time_unit="ns",
        timestamps=1_700_000_000_000_000_000
        + np.array([0, 71, 142, 210, 281], dtype=np.int64) * 1_000_000,
    )
    np.testing.assert_allclose(
        raw.seconds, [0, 0.071, 0.142, 0.210, 0.281], rtol=0, atol=1e-15
    )


@pytest.mark.parametrize("authority", ["eef_pose", "hybrid"])
def test_other_authorities_require_a_real_planning_provider(authority):
    with pytest.raises(NotImplementedError, match="IK/replanning"):
        normalize_real2sim(
            source(),
            layout(),
            Real2SimNormalizationCfg(control_dt=0.01, authority=authority),
        )
