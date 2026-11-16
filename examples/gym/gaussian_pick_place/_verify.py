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

"""Offline FR3 physical-task, two-camera and wrist-mount qualification."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from embodichain.data_pipeline.recording.media import (
    MediaFrameReader,
    verify_media_manifest,
)
from _prepare import digest, write_json
from _recording import CAMERAS, CAMERA, STEP_DT, validate_capture

__all__ = ["verify_episode", "main"]


def verify_episode(
    directory: Path, steps: int, *, require_marker: bool = False
) -> dict[str, Any]:
    if require_marker:
        marker = json.loads((directory / "episode.json").read_text())
        if (
            marker["status"] != "complete"
            or marker["episode_uuid"] != directory.name
            or marker["steps"] != steps
        ):
            raise ValueError("Invalid completion marker")
        for name, expected in marker["files"].items():
            if digest(directory / name) != expected:
                raise ValueError(f"Artifact hash mismatch: {name}")
    result = json.loads((directory / "execution_result.json").read_text())
    if not (
        result["completed"]
        and result["success"] == [True]
        and result["lengths"] == [steps]
        and len(result["segments"]) == 2
        and all(s["success"] for s in result["segments"])
    ):
        raise ValueError("Both measured Pick/Place segments must succeed")
    trajectory = torch.load(
        directory / "trajectory.pt", map_location="cpu", weights_only=False
    )
    meta = trajectory["meta"]
    if (
        meta["episode_uuid"] != directory.name
        or list(meta["lengths"]) != [steps]
        or trajectory["actions"].shape[:2] != (1, steps)
        or not math.isclose(meta["step_dt"], STEP_DT)
    ):
        raise ValueError("Native trajectory identity/length/cadence mismatch")
    poses = np.concatenate(
        [
            trajectory["states"]["rigid_objects"]["cube"]["pose"][0].numpy(),
            trajectory["terminal_states"]["rigid_objects"]["cube"]["pose"][0].numpy()[
                None
            ],
        ],
        axis=0,
    )
    if (
        poses.shape != (steps + 1, 7)
        or not np.isfinite(poses).all()
        or not np.allclose(np.linalg.norm(poses[:, 3:], axis=1), 1, atol=1e-4)
    ):
        raise ValueError("Invalid measured dynamic-cube states")
    root = directory.parents[1]
    run = json.loads((root / "run.json").read_text())
    environment = yaml.safe_load(
        (root / f"resolved/env.{run['scene']}.yaml").read_text()
    )
    scene = run["scene"]
    program = yaml.safe_load(
        (root / f"resolved/task_program/program.{scene}.yaml").read_text()
    )
    support = next(
        item
        for item in environment["simulation"]["background"]
        if item["uid"] == "support_surface"
    )
    target = np.asarray(program["targets"]["rest_pose"]["position"])
    lift = float(poses[:, 2].max() - poses[0, 2])
    error = float(np.linalg.norm(poses[-1, :3] - target))
    tilt = float(
        np.degrees(
            np.arccos(np.clip(1 - 2 * (poses[-1, 3] ** 2 + poses[-1, 4] ** 2), -1, 1))
        )
    )
    times = np.arange(steps + 1) * STEP_DT
    manifest = verify_media_manifest(
        directory / "media/manifest.json",
        expected_frames=steps + 1,
        parent_episode_id=directory.name,
        expected_timestamps=times,
    )
    expected = {
        f"{name}/rgb": {"shape": [540, 960, 3], "dtype": "|u1"} for name in CAMERAS
    }
    if manifest["features"] != expected:
        raise ValueError("Expected both native RGB streams")
    mount = np.asarray(manifest["calibration"]["wrist_mount_pose"])
    intrinsics = np.asarray(
        manifest["calibration"]["cameras"]["cam_wrist"]["intrinsics"][0]
    )
    if mount[2, 1] < 0.9 or mount[1, 0] < 0.99:
        raise ValueError("Wrist image orientation places the gripper beside the target")

    def project_hand(point: list[float]) -> np.ndarray:
        gl = mount[:3, :3].T @ (np.asarray(point) - mount[:3, 3])
        optical = np.array([gl[0], -gl[1], -gl[2]])
        if optical[2] <= 0:
            raise ValueError("Wrist framing landmark is behind the camera")
        projected = intrinsics @ optical
        return projected[:2] / projected[2]

    tcp_pixel = project_hand([0, 0, 0.1034])
    gripper_pixel = project_hand([0, 0, 0.04])
    if gripper_pixel[1] < tcp_pixel[1] + 50 or abs(gripper_pixel[0] - tcp_pixel[0]) > 1:
        raise ValueError("Gripper must project below the working point")
    previous = {name: -1 for name in CAMERAS}
    groups, physical, wrist_poses = {}, [], []
    first_clock = None
    with (
        (directory / "captures.jsonl").open() as stream,
        MediaFrameReader(directory / "media/manifest.json") as reader,
    ):
        for index in range(steps + 1):
            row = json.loads(stream.readline())
            if (
                row["episode_uuid"] != directory.name
                or row["step"] != index
                or not math.isclose(row["timestamp"], times[index], abs_tol=1e-7)
            ):
                raise ValueError("Capture row identity/time mismatch")
            if index == 0:
                first_clock = row["captures"][CAMERA]["simulation_time"]
            for name in CAMERAS:
                capture = row["captures"][name]
                validate_capture(capture)
                if (
                    capture["camera_id"] != name
                    or capture["frame_id"] <= previous[name]
                    or not math.isclose(
                        capture["simulation_time"] - first_clock,
                        times[index],
                        abs_tol=1e-7,
                    )
                ):
                    raise ValueError("Two-camera identity/clock mismatch")
                if index == 0:
                    groups[name] = capture["status"]["group_id"]
                if groups[name] != capture["status"]["group_id"] or not np.allclose(
                    capture["intrinsics"],
                    manifest["calibration"]["cameras"][name]["intrinsics"],
                    atol=1e-7,
                    rtol=0,
                ):
                    raise ValueError("Camera identity/intrinsics changed")
                if name == CAMERA and not np.allclose(
                    capture["arena_poses"],
                    manifest["calibration"]["cameras"][name]["arena_poses"],
                    atol=1e-7,
                    rtol=0,
                ):
                    raise ValueError("Third-person camera moved")
                previous[name] = capture["frame_id"]
            state = np.asarray(row["physical"]["cube_state"])
            if (
                state.shape != (13,)
                or not np.isfinite(state).all()
                or not np.allclose(state[:7], poses[index], atol=2e-6)
            ):
                raise ValueError("Captured physical state and native trajectory differ")
            hand = np.asarray(row["physical"]["hand_pose"])
            wrist = np.asarray(row["captures"]["cam_wrist"]["arena_poses"][0])
            expected_wrist = hand @ np.asarray(
                manifest["calibration"]["wrist_mount_pose"]
            )
            if not np.allclose(expected_wrist, wrist, atol=5e-4, rtol=0):
                raise ValueError(f"Wrist camera is not rigidly attached at s{index}")
            timestamp, frames = reader.read(index)
            if not math.isclose(timestamp, times[index], abs_tol=1e-7):
                raise ValueError("Raw RGB timestamp differs")
            for name in CAMERAS:
                if frames[f"{name}/rgb"].std() < 2:
                    raise ValueError(f"Empty RGB from {name}")
            physical.append(row["physical"])
            wrist_poses.append(wrist[:3, 3])
        if stream.readline():
            raise ValueError("Extra capture metadata")
    final = np.asarray(physical[-1]["cube_state"])
    linear, angular = float(np.linalg.norm(final[7:10])), float(
        np.linalg.norm(final[10:13])
    )
    stable = float(
        np.linalg.norm(np.diff(poses[-6:, :3], axis=0), axis=1).max() / STEP_DT
    )
    open_error = float(np.max(np.abs(np.asarray(physical[-1]["eef_qpos"]) - 0.04)))
    closed = max(float(np.max(0.04 - np.asarray(p["eef_qpos"]))) for p in physical)
    wrist_motion = float(
        np.linalg.norm(np.asarray(wrist_poses) - wrist_poses[0], axis=1).max()
    )
    metrics = dict(
        status="PASS",
        steps=steps,
        frames=steps + 1,
        rgb_images=2 * (steps + 1),
        simulation_duration_s=steps * STEP_DT,
        lift_m=lift,
        position_error_m=error,
        tilt_degrees=tilt,
        final_linear_speed_m_s=linear,
        final_angular_speed_rad_s=angular,
        last_0p2s_max_pose_speed_m_s=stable,
        final_gripper_open_error_m=open_error,
        peak_gripper_displacement_m=closed,
        wrist_translation_span_m=wrist_motion,
        max_wrist_mount_translation_error_m=max(
            p["wrist_mount_translation_error_m"] for p in physical
        ),
        max_wrist_mount_rotation_error=max(
            p["wrist_mount_rotation_error"] for p in physical
        ),
        max_lift_frame=int(np.argmax(poses[:, 2])),
        target_center=target.tolist(),
    )
    if (
        lift < 0.04 - 1e-6
        or error > 0.02
        or tilt > 15
        or linear > 0.01
        or angular > 0.1
        or stable > 0.01
        or open_error > 0.002
        or closed < 0.004
        or wrist_motion < 0.05
    ):
        metrics["status"] = "FAIL"
        write_json(directory / "physical_metrics.json", metrics)
        raise ValueError(f"Physical/two-camera qualification failed: {metrics}")
    metrics.update(
        wrist_framing="gripper_below_working_point",
        projected_tcp_uv=tcp_pixel.tolist(),
        projected_gripper_body_uv=gripper_pixel.tolist(),
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = json.loads((args.run / "run.json").read_text())
    if run["status"] != "complete":
        raise ValueError("Run is incomplete")
    episode = args.run / run["episode"]
    marker = json.loads((episode / "episode.json").read_text())
    print(
        json.dumps(
            verify_episode(episode, marker["steps"], require_marker=True), indent=2
        )
    )


if __name__ == "__main__":
    main()
