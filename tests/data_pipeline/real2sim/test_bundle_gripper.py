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

"""Source bundle admission and explicit gripper calibration contracts."""

from __future__ import annotations
from dataclasses import replace
import hashlib
import json
import numpy as np
import pytest
from embodichain.data_pipeline.real2sim import (
    Real2SimTrajectory,
    RobotJointLayout,
    SourcePhase,
)
from embodichain.data_pipeline.real2sim.normalization import Real2SimNormalizationCfg
from embodichain.data_pipeline.real2sim.bundle import (
    load_real2sim_bundle,
    write_real2sim_bundle,
)
from embodichain.data_pipeline.real2sim.gripper import GripperPositionCalibration


def _bundle(tmp_path, verified=True):
    source = Real2SimTrajectory(
        "fixture",
        "raw1",
        "pick0",
        "robot",
        "robot1",
        ("arm", "finger"),
        ("rad", "rad"),
        np.zeros((12, 2)),
        np.arange(12) * 0.01,
        "s",
        "cal1",
        "ann1",
        "verified" if verified else "unverified",
        "verified",
        (
            SourcePhase("approach", 0, 6, "free", ("joint_residual",)),
            SourcePhase("close", 6, 9, "contact"),
            SourcePhase("lift", 9, 12, "hold"),
        ),
    )
    layout = RobotJointLayout(
        "robot", "robot1", source.joint_names, source.joint_units, np.zeros(2), ("arm",)
    )
    return write_real2sim_bundle(
        source,
        layout,
        Real2SimNormalizationCfg(control_dt=0.01, allow_unverified=not verified),
        tmp_path / "bundle",
    )


def test_bundle_round_trip_and_data_hash_rejection(tmp_path):
    path = _bundle(tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    unit = load_real2sim_bundle(path, expected_sha256=digest)
    assert unit.eligible and unit.phases[1].kind == "contact"
    data = path.parent / "source.npz"
    data.write_bytes(data.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="array digest"):
        load_real2sim_bundle(path, expected_sha256=digest)


def test_bundle_diagnostic_input_is_never_formally_admitted(tmp_path):
    path = _bundle(tmp_path, verified=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert not load_real2sim_bundle(
        path, expected_sha256=digest, require_verified=False
    ).eligible
    with pytest.raises(ValueError, match="unverified"):
        load_real2sim_bundle(path, expected_sha256=digest)


def test_bundle_rejects_unknown_declarations_and_sidecar_traversal(tmp_path):
    path = _bundle(tmp_path)
    data = json.loads(path.read_text())
    data["arrays"] = "../source.npz"
    path.write_text(json.dumps(data))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="sidecar path"):
        load_real2sim_bundle(path, expected_sha256=digest)
    data["arrays"] = "source.npz"
    data["executable"] = "arbitrary.import"
    path.write_text(json.dumps(data))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="exactly"):
        load_real2sim_bundle(path, expected_sha256=digest)


def test_declared_gripper_encodings_have_consistent_physical_widths():
    calibration = GripperPositionCalibration(
        "normalized_closure",
        0.08,
        (0.0, 0.04, 0.08),
        (0.7, 0.4, 0.0),
        "fixture1",
        "measured lookup",
        status="verified",
    )
    np.testing.assert_array_equal(
        calibration.decode(np.array([0.0, 0.5, 1.0])), [0.0, 0.4, 0.7]
    )
    np.testing.assert_array_equal(
        replace(calibration, encoding="width_m").decode(np.array([0.08, 0.04, 0.0])),
        [0.0, 0.4, 0.7],
    )
    with pytest.raises(ValueError, match="domain"):
        calibration.decode(np.array([1.1]))
    with pytest.raises(ValueError, match="explicitly"):
        replace(calibration, encoding="guess")
    with pytest.raises(ValueError, match="Rejected"):
        replace(calibration, status="rejected").decode(np.array([0.5]))
