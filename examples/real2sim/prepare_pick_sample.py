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

"""Audit one original DROID HDF5 and prepare a conservative Pick source bundle.

This optional example never marks a model estimate as a measured calibration.
Its bundle remains diagnostic until mount/TCP and camera alignment are reviewed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pytorch_kinematics as pk
import torch
import trimesh

from embodichain.data import get_data_path
from embodichain.data_pipeline.real2sim import (
    Real2SimTrajectory,
    RobotJointLayout,
    SourcePhase,
    MimicJoint,
    EefPoseSamples,
)
from embodichain.data_pipeline.real2sim.bundle import write_real2sim_bundle
from embodichain.data_pipeline.real2sim.normalization import Real2SimNormalizationCfg
from embodichain.data_pipeline.real2sim.gripper import GripperPositionCalibration

__all__: list[str] = []
UPSTREAM = "https://github.com/droid-dataset/droid/blob/"  # pin supplied on CLI


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_calibration(hand_path: Path, evidence: str) -> GripperPositionCalibration:
    root = ET.parse(hand_path).getroot()
    joints = [j for j in root.findall("joint") if j.get("type") != "fixed"]
    chain = pk.build_chain_from_urdf(hand_path.read_bytes()).to(dtype=torch.float64)
    vertices = {}
    for link in ("left_inner_finger", "right_inner_finger"):
        collision = root.find(f"link[@name='{link}']/collision")
        if collision.find("origin").get("xyz") != "0 0 0":
            raise ValueError("Unexpected finger collision offset")
        mesh = trimesh.load(
            hand_path.parent / collision.find("geometry/mesh").get("filename"),
            force="mesh",
        )
        vertices[link] = mesh.vertices[mesh.vertices[:, 2] > 0.02]
    angles = np.linspace(0.0, 0.725, 80)
    widths = []
    for angle in angles:
        values = {
            j.get("name"): torch.tensor(
                [
                    angle
                    * (
                        float(j.find("mimic").get("multiplier", "1"))
                        if j.find("mimic") is not None
                        else 1.0
                    )
                ],
                dtype=torch.float64,
            )
            for j in joints
        }
        transforms = chain.forward_kinematics(values)
        points = {
            k: trimesh.transform_points(v, transforms[k].get_matrix()[0].numpy())
            for k, v in vertices.items()
        }
        widths.append(
            float(
                points["left_inner_finger"][:, 0].min()
                - points["right_inner_finger"][:, 0].max()
            )
        )
    if np.any(np.diff(widths) >= 0) or widths[-1] <= 0 or widths[0] >= 0.085:
        raise ValueError(
            "Unexpected Robotiq contact-pad geometry; review lookup extraction"
        )
    # The nominal 85 mm hardware range extends beyond this mesh's open-pad span.
    # Preserve the discrepancy explicitly; endpoint extensions are estimates.
    return GripperPositionCalibration(
        "normalized_closure",
        0.085,
        tuple([0.0, *widths[::-1], 0.085]),
        tuple([0.725, *angles[::-1], 0.0]),
        "robotiq2f85-model-pad-lookup-v1",
        evidence,
        status="unverified",
    )


def main() -> None:
    """Prepare one source slice and retain unresolved calibration prerequisites."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream-revision", required=True)
    parser.add_argument("--close-start", type=int, required=True)
    parser.add_argument("--lift-start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    args = parser.parse_args()
    source_path = args.hdf5.resolve()
    output = args.output.resolve()
    if output == source_path.parent or source_path.parent in output.parents:
        parser.error("Output must be outside the original source directory")
    if len(args.upstream_revision) != 40 or any(
        c not in "0123456789abcdef" for c in args.upstream_revision
    ):
        parser.error("Pin the upstream source-code commit SHA")
    output.mkdir(parents=True, exist_ok=False)
    arm_path = Path(get_data_path("Franka/Panda/Panda.urdf"))
    hand_path = Path(get_data_path("Robotiq2F85/Robotiq2F85.urdf"))
    evidence = UPSTREAM + args.upstream_revision + "/droid/franka/robot.py"
    calibration = _model_calibration(hand_path, evidence)
    (output / "gripper_calibration.json").write_text(
        json.dumps(asdict(calibration), indent=2) + "\n"
    )
    with h5py.File(source_path, "r") as f:
        q = f["joint_positions"][:]
        times = f["timestamps"][:]
        scalar = f["gripper_positions"][:]
        poses = f["gripper_pose"][:]
        arrays = {
            "raw_gripper_scalar": scalar,
            "raw_qpos": q,
            "timestamps_ms": times,
            "source_flange_pose": poses,
        }
        camera_metrics = {}
        for camera in f["intrinsic"]:
            intrinsic = f[f"intrinsic/{camera}"][:]
            extrinsic = f[f"extrinsic/{camera}_optimized"][:]
            arrays[f"intrinsic_{camera}"] = intrinsic
            arrays[f"extrinsic_{camera}"] = extrinsic
            camera_metrics[camera] = {
                "rotation_determinant": float(np.linalg.det(extrinsic[:3, :3])),
                "alignment_status": "unverified",
                "direction": "base_to_camera candidate; independent reprojection pending",
            }
        pipeline_metadata = json.loads(
            bytes(f["extrinsic/_pipeline_metadata"][:]).decode()
        )
    if not 0 < args.close_start < args.lift_start < args.stop <= len(q):
        parser.error("Invalid Pick phase boundaries")
    if (
        np.any(scalar[: args.close_start] > 0.02)
        or np.ptp(scalar[args.lift_start : args.stop]) > 0.02
    ):
        raise ValueError("Phase assumptions disagree with the recorded gripper events")
    driver = calibration.decode(scalar)
    hand = ET.parse(hand_path).getroot()
    moving = [j for j in hand.findall("joint") if j.get("type") != "fixed"]
    arm_names = tuple(f"fr3_joint{i}" for i in range(1, 8))
    names = (*arm_names, *(j.get("name") for j in moving))
    mimics = tuple(
        MimicJoint(
            j.get("name"),
            j.find("mimic").get("joint"),
            float(j.find("mimic").get("multiplier", "1")),
            float(j.find("mimic").get("offset", "0")),
        )
        for j in moving
        if j.find("mimic") is not None
    )
    full = np.c_[
        q,
        *[
            driver
            * (
                float(j.find("mimic").get("multiplier", "1"))
                if j.find("mimic") is not None
                else 1.0
            )
            for j in moving
        ],
    ]
    model_revision = hashlib.sha256(
        (_sha(arm_path) + _sha(hand_path)).encode()
    ).hexdigest()
    source = Real2SimTrajectory(
        "droid_original_hdf5",
        _sha(source_path),
        "pick_source_samples_0_" + str(args.stop),
        "franka_robotiq2f85",
        model_revision,
        names,
        ("rad",) * len(names),
        full[: args.stop],
        times[: args.stop],
        "ms",
        "partial-model-calibration-v1",
        "manual-event-annotation-v1",
        phases=(
            SourcePhase("approach", 0, args.close_start, "free"),
            SourcePhase("close", args.close_start, args.lift_start, "contact"),
            SourcePhase("lift", args.lift_start, args.stop, "hold"),
        ),
        eef=EefPoseSamples(
            poses[: args.stop],
            "robot_base",
            "fr3_link8",
            "observed_flange",
            np.eye(4),
            np.eye(4),
            "fk_derived",
        ),
        provenance={
            "gripper_mapping": asdict(calibration),
            "original_scalar_sha256": hashlib.sha256(scalar.tobytes()).hexdigest(),
            "annotation_note": "Manual conservative Pick phases supported by source gripper transition and lifting motion; contact time and camera offset remain unverified.",
            "source_sample_range": [0, args.stop],
            "source_pose_is_independent_measurement": False,
        },
    )
    layout = RobotJointLayout(
        source.robot_id,
        model_revision,
        names,
        ("rad",) * len(names),
        full[0],
        arm_names,
        mimics,
    )
    path = write_real2sim_bundle(
        source,
        layout,
        Real2SimNormalizationCfg(control_dt=0.01, allow_unverified=True),
        output / "pick_bundle",
    )
    chain = pk.build_serial_chain_from_urdf(arm_path.read_bytes(), "fr3_link8").to(
        dtype=torch.float64
    )
    fk = chain.forward_kinematics(torch.from_numpy(q)).get_matrix().numpy()
    rmse = float(np.sqrt(np.mean(np.sum((fk[:, :3, 3] - poses[:, :3]) ** 2, axis=1))))
    np.savez_compressed(
        output / "calibration_evidence.npz", **arrays, model_driver_rad=driver
    )
    report = {
        "bundle": str(path.relative_to(output)),
        "bundle_sha256": _sha(path),
        "source_sha256": _sha(source_path),
        "gripper_encoding": "normalized_closure",
        "gripper_convention_reference": evidence,
        "lookup_status": "model-derived estimate; hardware calibration pending",
        "flange_position_consistency_rmse_m": rmse,
        "independent_pose_accuracy_test": False,
        "camera_candidates": camera_metrics,
        "input_calibration_pipeline_metadata": pipeline_metadata,
        "pick_source_segments": {
            "approach": [0, args.close_start],
            "close": [args.close_start, args.lift_start],
            "lift": [args.lift_start, args.stop],
        },
        "formal_admission": "rejected_until_reviewed",
        "unresolved": [
            "real flange-to-gripper mount and TCP",
            "hardware jaw-width/driver calibration",
            "independent camera reprojection and N-1 video offset",
            "target simulator full-joint order and initial scene registration",
        ],
    }
    (output / "calibration_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
