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

"""Bounded RGB and measured-state capture at normal environment boundaries."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from embodichain.data_pipeline.recording.media import MediaSpool

__all__ = ["Capture", "validate_capture", "json_value", "CAMERAS", "CAMERA", "STEP_DT"]

STEP_DT = 0.04
CAMERA = "cam_third"
CAMERAS = ("cam_third", "cam_wrist")


def json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def validate_capture(capture: dict[str, Any]) -> None:
    status, contract = capture.get("status", {}), capture.get("output_contract", {})
    if not (
        capture.get("valid") is True
        and status.get("valid") is True
        and status.get("completed") is True
        and status.get("outcome") == "SUCCESS"
        and contract.get("completed") is True
        and contract.get("same_frame_identity") is True
        and capture.get("frame_id")
        == status.get("frame_index")
        == contract.get("frame_index")
        and status.get("group_id") == contract.get("group_id")
        and capture.get("camera_id") in CAMERAS
        and capture.get("env_ids") == [0]
        and capture.get("outputs") == ["color"]
        and capture.get("gaussian_approach") == "RASTER_CUDA"
        and len(status.get("cameras", [])) == 1
        and status["cameras"][0]["selected_nodes"] > 0
    ):
        raise ValueError("RGB is not from a completed same-frame Gaussian capture")
    for key, shape in [("intrinsics", (1, 3, 3)), ("arena_poses", (1, 4, 4))]:
        data = np.asarray(capture[key])
        if data.shape != shape or not np.isfinite(data).all():
            raise ValueError(f"Invalid {key}")


def snapshot(
    env: Any, observation: Any
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    frames, captures = {}, {}
    for name in CAMERAS:
        rgb = np.array(
            observation["sensor"][name]["color"][0, ..., :3].detach().cpu(), copy=True
        )
        if rgb.shape != (540, 960, 3) or rgb.dtype != np.uint8:
            raise ValueError(f"Expected 960x540 uint8 native RGB: {name}")
        capture = json_value(env.sensors[name].get_capture_metadata())
        validate_capture(capture)
        if capture["camera_id"] != name or not math.isclose(
            capture["simulation_time"], env.sim.simulation_time, abs_tol=1e-7
        ):
            raise ValueError(
                "Camera and measured physical state have different clocks/identity"
            )
        frames[f"{name}/rgb"], captures[name] = rgb, capture
    cube = env.sim.get_rigid_object("cube")
    hand = env.robot.get_link_pose("fr3_hand", to_matrix=True)[0].detach().cpu().numpy()
    mount = (
        env.sensors["cam_wrist"].cfg.extrinsics.transformation.detach().cpu().numpy()
    )
    wrist = np.asarray(captures["cam_wrist"]["arena_poses"][0])
    physical = {
        "cube_state": json_value(cube.body_state[0]),
        "joint_positions": json_value(env.robot.get_qpos()[0]),
        "eef_qpos": json_value(env.robot.get_qpos(name="eef")[0]),
        "hand_pose": hand.tolist(),
        "wrist_mount_translation_error_m": float(
            np.linalg.norm((hand @ mount)[:3, 3] - wrist[:3, 3])
        ),
        "wrist_mount_rotation_error": float(
            np.max(np.abs((hand @ mount)[:3, :3] - wrist[:3, :3]))
        ),
    }
    return frames, captures, physical


class Capture:
    """Store s0 through sT without issuing camera or object-state writes."""

    def __init__(self, root: Path, env: Any, initial: Any) -> None:
        self.root, self.env = root, env
        self.initial = snapshot(env, initial)
        for name in CAMERAS:
            Image.fromarray(self.initial[0][f"{name}/rgb"]).save(
                root / f"initial_{name}.png"
            )
        self.start_time = self.initial[1][CAMERA]["simulation_time"]
        self.directory: Path | None = None
        self.episode_uuid: str | None = None
        self.spool: MediaSpool | None = None
        self.stream: Any = None
        self.count = 0
        self.last_frame = {name: -1 for name in CAMERAS}
        self.calibration: dict[str, Any] = {}
        self.peak_buffer_bytes = 0
        self.capture_seconds = 0.0
        self.error: str | None = None
        self.phase = "pick_cube"

    def observe(self, active: tuple[bool, ...], action: Any, observation: Any) -> None:
        if self.error:
            return
        started = time.perf_counter()
        try:
            if active != (True,):
                raise ValueError("Expected one active physical row")
            episode_uuid = self.env.get_demo_episode_metadata(0)["episode_uuid"]
            if self.spool is None:
                self.episode_uuid = episode_uuid
                self.directory = self.root / "episodes" / episode_uuid
                self.calibration = {
                    "cameras": {
                        name: {
                            key: self.initial[1][name][key]
                            for key in [
                                "intrinsics",
                                "arena_poses",
                                "camera_pose_convention",
                                "length_unit",
                            ]
                        }
                        for name in CAMERAS
                    },
                    "wrist_mount_pose": json_value(
                        self.env.sensors["cam_wrist"].cfg.extrinsics.transformation
                    ),
                    "wrist_parent": "FrankaPanda/fr3_hand",
                }
                self.spool = MediaSpool(
                    self.directory / "media",
                    max_buffer_bytes=64 * 1024 * 1024,
                    parent_episode_id=episode_uuid,
                    calibration=self.calibration,
                )
                self.stream = (self.directory / "captures.jsonl").open("x")
                self.append(*self.initial)
                self.initial = None
            if episode_uuid != self.episode_uuid:
                raise ValueError("Physical episode identity changed")
            self.append(*snapshot(self.env, observation))
            if (self.count - 1) % 100 == 0:
                print(
                    f"CAPTURE {self.count} frames ({(self.count-1)*STEP_DT:.2f}s simulation)",
                    flush=True,
                )
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.capture_seconds += time.perf_counter() - started

    def append(
        self,
        frames: dict[str, np.ndarray],
        captures: dict[str, Any],
        physical: dict[str, Any],
    ) -> None:
        timestamp = captures[CAMERA]["simulation_time"] - self.start_time
        if not math.isclose(timestamp, self.count * STEP_DT, abs_tol=1e-7):
            raise ValueError("Missing or duplicated control-boundary frame")
        for name in CAMERAS:
            capture = captures[name]
            if capture["frame_id"] <= self.last_frame[name] or not math.isclose(
                capture["simulation_time"] - self.start_time, timestamp, abs_tol=1e-7
            ):
                raise ValueError("Camera pair clock/identity did not advance together")
            if not np.allclose(
                capture["intrinsics"],
                self.calibration["cameras"][name]["intrinsics"],
                atol=1e-7,
                rtol=0,
            ):
                raise ValueError("Sensor intrinsics changed")
            if name == CAMERA and not np.allclose(
                capture["arena_poses"],
                self.calibration["cameras"][name]["arena_poses"],
                atol=1e-7,
                rtol=0,
            ):
                raise ValueError("Third-person camera moved")
            self.last_frame[name] = capture["frame_id"]
        self.spool.append(timestamp, frames)
        self.stream.write(
            json.dumps(
                {
                    "episode_uuid": self.episode_uuid,
                    "step": self.count,
                    "timestamp": timestamp,
                    "phase": self.phase,
                    "captures": captures,
                    "physical": physical,
                },
                allow_nan=False,
            )
            + "\n"
        )
        self.count += 1
        self.peak_buffer_bytes = max(self.peak_buffer_bytes, self.spool.buffered_bytes)

    def segment_completed(self, result: Any) -> None:
        metadata = result.to_metadata()
        summary = {
            key: metadata[key]
            for key in ["name", "start_step", "end_step", "success", "failure_reason"]
        }
        print("SEGMENT " + json.dumps(summary), flush=True)
        self.phase = "place_cube"

    def finish(self, steps: int | None = None) -> None:
        if (
            self.error
            or self.spool is None
            or (steps is not None and self.count != steps + 1)
        ):
            raise ValueError(f"Incomplete RGB recording: {self.error or self.count}")
        self.close()
        self.spool.finish()

    def close(self) -> None:
        if self.stream is not None:
            try:
                self.stream.flush()
                os.fsync(self.stream.fileno())
            finally:
                self.stream.close()
                self.stream = None
