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
"""Read prepared real robot recordings and registered scene artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from embodichain.data_pipeline.real2sim import RobotDemonstration
from embodichain.data_pipeline.real2sim._geometry import _robot_geometry_sha256
from embodichain.data_pipeline.real2sim.adaptation.inputs import (
    CameraObservation,
    ReconstructionInput,
    RegisteredScene,
    RobotModel,
)
from embodichain.data_pipeline.real2sim.io import _digest as digest

__all__ = ["load_prepared_source", "load_registered_scene"]


def load_prepared_source(root: Path, instruction: str) -> ReconstructionInput:
    """Load unchanged joints and explicit source frame mappings without inference."""
    root = root.resolve()
    meta = json.loads((root / "robot.json").read_text())
    urdf = (root / meta["urdf"]).resolve()
    physical = ET.parse(urdf).getroot()
    joints = {joint.get("name"): joint for joint in physical.findall("joint")}
    meshes = tuple(
        (urdf.parent / mesh.get("filename")).resolve()
        for mesh in physical.findall("./link/collision/geometry/mesh")
    )
    trajectory = root / "trajectory_original.npz"
    with np.load(trajectory, allow_pickle=False) as raw:
        arrays = {key: raw[key].copy() for key in raw.files}
    timestamps = arrays["timestamps"]
    if (
        meta.get("robot_type", "").startswith("rh20t-")
        and "acquisition_timestamps" in arrays
    ):
        acquisition = np.asarray(arrays["acquisition_timestamps"], dtype=float)
        if (
            acquisition.shape != (len(arrays["qpos"]),)
            or not len(acquisition)
            or not np.isfinite(acquisition).all()
            or not np.all(np.diff(acquisition) > 0)
        ):
            raise ValueError(
                "RH20T acquisition timestamps must be finite, match samples and increase"
            )
        timestamps = acquisition - acquisition[0]
    names = tuple(arrays["joint_names"].tolist())
    demonstration = RobotDemonstration(
        source_id="prepared_robot_recording",
        source_revision=digest(trajectory),
        unit_id=root.name,
        joint_names=names,
        joint_units=tuple(
            "m" if joints[n].get("type") == "prismatic" else "rad" for n in names
        ),
        qpos=arrays["qpos"],
        timestamps=timestamps,
        time_unit="s",
        robot_geometry_sha256=_robot_geometry_sha256(urdf, meshes),
        recording_sha256=digest(trajectory),
    )
    visuals = json.loads((root / "robot_visuals.json").read_text())["visuals"]
    for visual in visuals:
        visual["path"] = str((root / visual["path"]).resolve())
    sides = tuple(meta.get("side_names", meta.get("prefixes", ())))
    robot = RobotModel(
        urdf=urdf,
        joint_names=names,
        side_names=sides,
        tcp_links=tuple(meta.get("tcp_links", [side + "_link6" for side in sides])),
        flange_to_tcp=np.asarray(meta["flange_to_tcp"]),
        visuals=tuple(visuals),
        pad_center_tcp_z_m=float(meta.get("pad_center_tcp_z_m", -0.035)),
        max_half_open_m=float(meta.get("max_half_open_m", 0.065)),
        arm_joint_groups=tuple(
            tuple(group) for group in meta.get("arm_joint_groups", ())
        ),
        gripper_calibrations=tuple(meta.get("gripper_calibrations", ())),
    )
    cameras = {}
    frame_map = arrays.get("source_frame_indices", np.arange(len(demonstration.qpos)))
    for view in ("front", "left", "right"):
        video = root / f"original_{view}.mp4"
        if not video.is_file() or view + "_K" not in arrays:
            continue
        first = root / f"first_{view}_original.png"
        if not first.is_file():
            first = root / f"first_{view}.png"
        depth_path = root / (
            "depth_first.npy" if view == "front" else f"depth_{view}_first.npy"
        )
        cameras[view] = CameraObservation(
            video,
            first if first.is_file() else None,
            frame_map,
            arrays[view + "_K"],
            arrays[view + "_camera"][0],
            np.load(depth_path, allow_pickle=False) if depth_path.is_file() else None,
        )
    return ReconstructionInput(demonstration, robot, cameras, "front", instruction)


def load_registered_scene(root: Path) -> RegisteredScene:
    """Load the registered scene and cached world vertices without regeneration."""
    root = root.resolve()
    description = json.loads((root / "scene.json").read_text())
    with np.load(root / "scene_vertices.npz", allow_pickle=False) as data:
        vertices = {key: data[key].copy() for key in data.files}
    for obj in description["objects"]:
        obj["path"] = str((root / obj["path"]).resolve())
        if obj["id"] not in vertices:
            raise ValueError(f"Missing registered vertices: {obj['id']}")
    return RegisteredScene(description, vertices)
