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

"""Example-local capture ownership and durable artifact helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np

from embodichain.data_pipeline.recording.media import MediaFrameReader, MediaSpool

CAMERA = "cam_high"
STEP_DT = 0.04
BUFFER_BYTES = 64 * 1024 * 1024

__all__: list[str] = []


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_value(value: Any) -> Any:
    """Copy native metadata; invalid numeric backgrounds are JSON null."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(json_value(value), stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def validate_capture(capture: dict[str, Any]) -> None:
    """Require a completed, single-camera Gaussian RGB/view-Z observation."""
    status = capture.get("status", {})
    contract = capture.get("output_contract", {})
    if (
        capture.get("valid") is not True
        or status.get("valid") is not True
        or status.get("completed") is not True
        or status.get("outcome") != "SUCCESS"
        or contract.get("completed") is not True
        or contract.get("same_frame_identity") is not True
        or contract.get("group_id") != status.get("group_id")
        or contract.get("frame_index") != status.get("frame_index")
        or capture.get("frame_id") != status.get("frame_index")
        or type(capture.get("frame_id")) is not int
        or capture.get("camera_id") != CAMERA
        or capture.get("env_ids") != [0]
        or not {"color", "depth"}.issubset(capture.get("outputs", ()))
        or capture.get("length_unit") != "meter"
        or contract.get("depth", {}).get("is_view_z") is not True
    ):
        raise ValueError(
            "Capture is not a completed same-frame Gaussian RGB/depth pair"
        )
    for name, shape in (("intrinsics", (1, 3, 3)), ("arena_poses", (1, 4, 4))):
        values = np.asarray(capture.get(name), dtype=float)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"Invalid capture {name}")
    if not isinstance(
        capture.get("simulation_time"), (int, float)
    ) or not math.isfinite(capture["simulation_time"]):
        raise ValueError("Capture has no finite simulation time")


def snapshot(
    env: Any, observation: Any
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    data = observation["sensor"][CAMERA]
    frames = {
        f"{CAMERA}/rgb": np.array(data["color"][0, ..., :3].detach().cpu(), copy=True),
        f"{CAMERA}/depth": np.array(data["depth"][0].detach().cpu(), copy=True),
    }
    if (
        frames[f"{CAMERA}/rgb"].shape != (540, 960, 3)
        or frames[f"{CAMERA}/rgb"].dtype != np.uint8
    ):
        raise ValueError("Expected 960x540 uint8 RGB")
    depth = frames[f"{CAMERA}/depth"]
    if depth.shape != (540, 960) or depth.dtype != np.float32:
        raise ValueError("Expected 960x540 float32 view-Z depth")
    if (
        np.isnan(depth).any()
        or np.isneginf(depth).any()
        or not np.any(np.isfinite(depth) & (depth > 0))
    ):
        raise ValueError("Depth contains invalid values or no finite foreground")
    capture = json_value(env.sensors[CAMERA].get_capture_metadata())
    if not isinstance(capture, dict):
        raise ValueError("Camera has no completed capture metadata")
    validate_capture(capture)
    if not math.isclose(
        capture["simulation_time"], env.sim.simulation_time, abs_tol=1e-7
    ):
        raise ValueError("Camera capture is stale relative to the physical state")
    return frames, capture


class EpisodeCapture:
    """One B=1 demo observer; media follows the executor's canonical UUID."""

    def __init__(self, root: Path, env: Any, initial_observation: Any) -> None:
        self.root = root
        self.env = env
        self.initial = snapshot(env, initial_observation)
        self.start_time = self.initial[1]["simulation_time"]
        self.directory: Path | None = None
        self.episode_uuid: str | None = None
        self.spool: MediaSpool | None = None
        self.stream: Any = None
        self.count = 0
        self.last_frame = -1
        self.calibration: dict[str, Any] = {}
        self.peak_buffer_bytes = 0
        self.capture_seconds = 0.0
        self.error: str | None = None

    def observe(self, active: tuple[bool, ...], action: Any, observation: Any) -> None:
        # After the first error, let the executor finish its ordinary safe stop.
        if self.error is not None:
            return
        started = time.perf_counter()
        try:
            if active != (True,):
                raise ValueError("Library recording supports one active environment")
            episode_uuid = self.env.get_demo_episode_metadata(0)["episode_uuid"]
            if self.spool is None:
                self.episode_uuid = episode_uuid
                self.directory = self.root / episode_uuid
                self.calibration = {
                    key: self.initial[1][key]
                    for key in (
                        "intrinsics",
                        "arena_poses",
                        "camera_pose_convention",
                        "length_unit",
                    )
                }
                self.spool = MediaSpool(
                    self.directory / "media",
                    max_buffer_bytes=BUFFER_BYTES,
                    parent_episode_id=episode_uuid,
                    calibration=self.calibration,
                )
                self.stream = (self.directory / "captures.jsonl").open("x")
                self._append(*self.initial)
                self.initial = None
            if episode_uuid != self.episode_uuid:
                raise ValueError("Demo episode changed while capturing")
            self._append(*snapshot(self.env, observation))
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
            raise
        finally:
            self.capture_seconds += time.perf_counter() - started

    def _append(self, frames: dict[str, np.ndarray], capture: dict[str, Any]) -> None:
        timestamp = capture["simulation_time"] - self.start_time
        if not math.isclose(timestamp, self.count * STEP_DT, abs_tol=1e-7):
            raise ValueError("Missing or duplicated control-boundary capture")
        if capture["frame_id"] <= self.last_frame:
            raise ValueError("Camera frame identity did not advance")
        for key in ("intrinsics", "arena_poses"):
            if not np.allclose(capture[key], self.calibration[key], atol=1e-7, rtol=0):
                raise ValueError("Fixed camera calibration changed")
        self.spool.append(timestamp, frames)
        self.stream.write(
            json.dumps(
                {
                    "episode_uuid": self.episode_uuid,
                    "step": self.count,
                    "timestamp": timestamp,
                    "simulation_time": capture["simulation_time"],
                    "capture": capture,
                },
                allow_nan=False,
            )
            + "\n"
        )
        self.last_frame = capture["frame_id"]
        self.count += 1
        self.peak_buffer_bytes = max(self.peak_buffer_bytes, self.spool.buffered_bytes)

    def finish(self, steps: int) -> None:
        if self.error or self.spool is None or self.count != steps + 1:
            raise ValueError(f"Incomplete RGB-D recording: {self.error or self.count}")
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


def preview(directory: Path, result: dict[str, Any]) -> None:
    from PIL import Image, ImageDraw

    steps = result["length"]
    lifted = result["segments"][0]["end_step"]
    indices = (0, lifted, max(lifted, steps - 15), steps)
    canvas = Image.new("RGB", (1920, 1120))
    labels = ("Initial", "Lifted", "Release", "Terminal")
    with MediaFrameReader(directory / "media/manifest.json") as reader:
        for tile, (index, label) in enumerate(zip(indices, labels, strict=True)):
            timestamp, values = reader.read(index)
            x, y = (tile % 2) * 960, (tile // 2) * 560
            canvas.paste(Image.fromarray(values[f"{CAMERA}/rgb"]), (x, y + 20))
            ImageDraw.Draw(canvas).text(
                (x + 6, y + 3), f"{label}: s{index}, {timestamp:.2f}s"
            )
    canvas.save(directory / "preview.png")
