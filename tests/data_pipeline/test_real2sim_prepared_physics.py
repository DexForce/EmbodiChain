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
"""Verify physical package revisions preserve source observations and identity."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from embodichain.data_pipeline.real2sim._geometry import _robot_geometry_sha256
from embodichain.data_pipeline.real2sim.adaptation._prepared_physics import (
    prepare_physical_inputs,
    record_native_qualification,
    seal_prepared_checksums,
    verify_prepared_checksums,
)


@pytest.fixture
def package(tmp_path: Path) -> Path:
    root = tmp_path / "original"
    sample = root / "samples/example"
    robot = sample / "robot/robot.urdf"
    robot.parent.mkdir(parents=True)
    robot.write_text('<robot name="fixture"><link name="pad"/></robot>')
    files = {
        root
        / "manifest.json": {
            "samples": [{"id": "example", "prepared_source": "samples/example"}]
        },
        root
        / "input_validation.json": {
            "cases": [{"id": "example", "robot_geometry_unchanged": True}],
            "relocation_check": {"status": "passed", "files": 8},
        },
        sample
        / "robot.json": {
            "urdf": "robot/robot.urdf",
            "gripper_calibrations": [{"pad_links": ["pad"]}],
        },
        sample
        / "input.json": {
            "prepared_source": {
                "robot_geometry_sha256": _robot_geometry_sha256(robot, ())
            }
        },
    }
    for path, value in files.items():
        path.write_text(json.dumps(value))
    (sample / "trajectory_original.npz").write_bytes(b"immutable recorded samples")
    (sample / "original_front.mp4").write_bytes(b"immutable video")
    (sample / "calibration.json").write_text('{"extrinsics":"unchanged"}')
    seal_prepared_checksums(root)
    return root


def _physical_robot(source: Path, output: Path) -> tuple[Path, dict]:
    output.mkdir()
    tree = ET.parse(source)
    collision = ET.SubElement(tree.getroot().find("link"), "collision")
    ET.SubElement(ET.SubElement(collision, "geometry"), "box", size="0.01 0.01 0.01")
    target = output / "robot.urdf"
    tree.write(target)
    provenance = {"collision_links": ["pad"], "source": "fixture"}
    (output / "collision_provenance.json").write_text(json.dumps(provenance))
    return target, provenance


def _native_report(output: Path, geometry: str) -> dict:
    native = output / "samples/example/physical_validation/native"
    native.mkdir()
    trace = b"fixture native trace bytes"
    (native / "trace.npz").write_bytes(trace)
    result = {
        "schema": "prepared-robot-native-qualification/v1",
        "status": "passed",
        "robot_geometry_sha256": geometry,
        "task_success_verified": False,
        "traces": {"path": "trace.npz", "sha256": hashlib.sha256(trace).hexdigest()},
    }
    (native / "report.json").write_text(json.dumps(result))
    return result


def test_physical_package_preserves_recording_and_updates_identity(
    package: Path, tmp_path: Path
) -> None:
    original = {
        p.relative_to(package): p.read_bytes()
        for p in package.rglob("*")
        if p.is_file()
    }
    output = tmp_path / "physical"
    receipt = prepare_physical_inputs(
        package, output, sample_id="example", prepare_robot=_physical_robot
    )
    assert receipt["source_robot_geometry_sha256"] != receipt["robot_geometry_sha256"]
    assert receipt["collision_count"] == 1
    assert receipt["native_qualification"] == "not_run"
    assert not receipt["task_success_verified"]
    verify_prepared_checksums(output)
    for name, data in original.items():
        assert (package / name).read_bytes() == data
    for name in (
        "trajectory_original.npz",
        "original_front.mp4",
        "calibration.json",
        "robot/robot.urdf",
    ):
        assert (output / "samples/example" / name).read_bytes() == (
            package / "samples/example" / name
        ).read_bytes()
    meta = json.loads((output / "samples/example/robot.json").read_text())
    assert meta["urdf"] == "robot_physical/robot.urdf"
    assert meta["gripper_calibrations"] == [{"pad_links": ["pad"]}]
    audit = json.loads((output / "input_validation.json").read_text())
    assert audit["relocation_check"]["status"] == "not_run"
    assert audit["source_package_relocation_check"]["files"] == 8
    record_native_qualification(
        output, "example", _native_report(output, receipt["robot_geometry_sha256"])
    )
    verify_prepared_checksums(output)
    updated = json.loads(
        (output / "samples/example/physical_validation/preparation.json").read_text()
    )
    assert updated["native_qualification"] == "passed"
    assert not updated["task_success_verified"]


def test_corrupt_source_rejected_before_preparation(
    package: Path, tmp_path: Path
) -> None:
    (package / "samples/example/original_front.mp4").write_bytes(b"changed")
    output = tmp_path / "physical"
    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare_physical_inputs(
            package, output, sample_id="example", prepare_robot=_physical_robot
        )
    assert not output.exists()


def test_native_report_must_match_prepared_geometry(
    package: Path, tmp_path: Path
) -> None:
    output = tmp_path / "physical"
    prepare_physical_inputs(
        package, output, sample_id="example", prepare_robot=_physical_robot
    )
    with pytest.raises(ValueError, match="different physical robot"):
        record_native_qualification(
            output,
            "example",
            {"status": "passed", "robot_geometry_sha256": "different"},
        )
    verify_prepared_checksums(output)


def test_native_attachment_cannot_reseal_changed_observations(
    package: Path, tmp_path: Path
) -> None:
    output = tmp_path / "physical"
    receipt = prepare_physical_inputs(
        package, output, sample_id="example", prepare_robot=_physical_robot
    )
    result = _native_report(output, receipt["robot_geometry_sha256"])
    video = output / "samples/example/original_front.mp4"
    video.write_bytes(b"changed during native check")
    with pytest.raises(ValueError, match="checksum mismatch"):
        record_native_qualification(output, "example", result)


def test_native_attachment_requires_persisted_report_and_trace(
    package: Path, tmp_path: Path
) -> None:
    output = tmp_path / "physical"
    receipt = prepare_physical_inputs(
        package, output, sample_id="example", prepare_robot=_physical_robot
    )
    result = {
        "status": "passed",
        "robot_geometry_sha256": receipt["robot_geometry_sha256"],
    }
    with pytest.raises(ValueError, match="matching persisted report"):
        record_native_qualification(output, "example", result)
    result = _native_report(output, receipt["robot_geometry_sha256"])
    (output / "samples/example/physical_validation/native/trace.npz").write_bytes(
        b"changed"
    )
    with pytest.raises(ValueError, match="trace checksum mismatch"):
        record_native_qualification(output, "example", result)
