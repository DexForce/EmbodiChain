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

"""Stream and audit a Library recording without a World, GPU or asset server."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from _recording import CAMERA, STEP_DT, digest, validate_capture
from embodichain.data_pipeline.recording.media import (
    MediaFrameReader,
    verify_media_manifest,
)

__all__ = ["verify_episode", "verify_run", "main"]


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON mapping: {path}")
    return value


def _local(root: Path, name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Artifact path must be contained and relative: {name}")
    selected = (root / path).resolve(strict=True)
    if not selected.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact escapes recording directory: {name}")
    return selected


def verify_episode(
    directory: str | Path, record: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Verify an episode, or a pending completion record before publication.

    Args:
        directory: Episode directory containing media and a native trajectory.
        record: Caller-owned pending completion record; CLI always reads disk.
    Returns:
        Episode identity, lengths and independently measured physical metrics.
    """
    import torch

    root = Path(directory).resolve()
    record = _read(root / "episode.json") if record is None else record
    steps = record.get("steps")
    if (
        record.get("schema_version") != 1
        or record.get("status") != "complete"
        or type(steps) is not int
        or steps <= 0
        or record.get("episode_uuid") != root.name
        or not math.isclose(record.get("step_dt", 0), STEP_DT, abs_tol=1e-12)
    ):
        raise ValueError("Invalid or incomplete episode marker")
    expected_files = {
        "trajectory.pt",
        "execution_result.json",
        "captures.jsonl",
        "media/manifest.json",
        "preview.png",
    }
    if set(record.get("files", {})) != expected_files:
        raise ValueError("Episode inventory is incomplete")
    for name, expected in record["files"].items():
        if digest(_local(root, name)) != expected:
            raise ValueError(f"Artifact hash mismatch: {name}")
    result = _read(root / "execution_result.json")
    if (
        result.get("completed") is not True
        or result.get("success") != [True]
        or result.get("lengths") != [steps]
        or len(result.get("segments", ())) != 2
        or any(segment.get("success") is not True for segment in result["segments"])
    ):
        raise ValueError("Physical Task Program did not complete successfully")

    trajectory = torch.load(
        root / "trajectory.pt", map_location="cpu", weights_only=False
    )
    meta = trajectory["meta"]
    if (
        meta.get("episode_uuid") != root.name
        or list(meta.get("lengths", ())) != [steps]
        or trajectory["actions"].shape[:2] != (1, steps)
        or "terminal_states" not in trajectory
        or not math.isclose(meta["step_dt"], STEP_DT, abs_tol=1e-12)
        or meta.get("rgbd_recording")
        != {
            "schema_version": 1,
            "media": "media/manifest.json",
            "captures": "captures.jsonl",
            "media_sha256": record["files"]["media/manifest.json"],
            "captures_sha256": record["files"]["captures.jsonl"],
        }
    ):
        raise ValueError("Trajectory and RGB-D episode identities/layouts differ")

    poses = trajectory["states"]["rigid_objects"]["cube"]["pose"][0].numpy()
    final = trajectory["terminal_states"]["rigid_objects"]["cube"]["pose"][0].numpy()
    if (
        poses.shape != (steps, 7)
        or final.shape != (7,)
        or not np.isfinite(poses).all()
        or not np.isfinite(final).all()
        or not torch.isfinite(trajectory["actions"]).all()
        or not np.allclose(np.linalg.norm(poses[:, 3:], axis=1), 1, atol=1e-4)
        or not np.isclose(np.linalg.norm(final[3:]), 1, atol=1e-4)
    ):
        raise ValueError("Trajectory contains invalid physical states or actions")
    lift = float(max(poses[:, 2].max(), final[2]) - poses[0, 2])
    error = float(np.linalg.norm(final[:3] - [0.55, 0.05, 0.775]))
    quaternion = final[3:]
    cosine = 1 - 2 * float(quaternion[0] ** 2 + quaternion[1] ** 2)
    tilt = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    if lift < 0.04 - 1e-6 or error > 0.02 or tilt > 15:
        raise ValueError(
            f"Measured task failed: lift={lift}, error={error}, tilt={tilt}"
        )

    times = np.arange(steps + 1, dtype=float) * STEP_DT
    manifest = verify_media_manifest(
        root / "media/manifest.json",
        expected_frames=steps + 1,
        parent_episode_id=root.name,
        expected_timestamps=times,
    )
    expected_schema = {
        f"{CAMERA}/rgb": {"shape": [540, 960, 3], "dtype": np.dtype("uint8").str},
        f"{CAMERA}/depth": {"shape": [540, 960], "dtype": np.dtype("float32").str},
    }
    if manifest["features"] != expected_schema:
        raise ValueError("Unexpected RGB/depth schema")
    previous_id = -1
    group_id = None
    start_time = None
    with (
        (root / "captures.jsonl").open() as stream,
        MediaFrameReader(root / "media/manifest.json") as reader,
    ):
        for index in range(steps + 1):
            line = stream.readline()
            if not line.endswith("\n"):
                raise ValueError("Missing or incomplete capture metadata")
            row = json.loads(line)
            capture = row["capture"]
            validate_capture(capture)
            if index == 0:
                start_time = row["simulation_time"]
                group_id = capture["status"]["group_id"]
            if (
                row["episode_uuid"] != root.name
                or row["step"] != index
                or not math.isclose(row["timestamp"], times[index], abs_tol=1e-7)
                or not math.isclose(
                    row["simulation_time"] - start_time, times[index], abs_tol=1e-7
                )
                or not math.isclose(
                    capture["simulation_time"], row["simulation_time"], abs_tol=1e-7
                )
                or capture["frame_id"] <= previous_id
                or capture["status"]["group_id"] != group_id
            ):
                raise ValueError(f"Capture clock/identity mismatch at s{index}")
            for key in ("intrinsics", "arena_poses"):
                if not np.allclose(
                    capture[key], manifest["calibration"][key], atol=1e-7, rtol=0
                ):
                    raise ValueError(f"Camera calibration changed at s{index}")
            timestamp, frames = reader.read(index)
            depth = frames[f"{CAMERA}/depth"]
            if (
                not math.isclose(timestamp, times[index], abs_tol=1e-7)
                or np.isnan(depth).any()
                or np.isneginf(depth).any()
            ):
                raise ValueError(f"Invalid media values/clock at s{index}")
            previous_id = capture["frame_id"]
        if stream.readline():
            raise ValueError("Unexpected trailing capture metadata")
    return {
        "episode_uuid": root.name,
        "steps": steps,
        "frames": steps + 1,
        "lift_m": lift,
        "position_error_m": error,
        "tilt_degrees": tilt,
    }


def verify_run(directory: str | Path) -> dict[str, Any]:
    """Verify a relocated run using only its local artifacts.

    Args:
        directory: Run directory printed by run.py.
    Returns:
        A summary of verified episodes and measured task results.
    """
    root = Path(directory).resolve()
    run = _read(root / "run.json")
    requested = run.get("requested_episodes")
    if (
        run.get("schema_version") != 1
        or run.get("status") != "complete"
        or type(requested) is not int
        or requested < 1
        or len(run.get("episodes", ())) != requested
    ):
        raise ValueError("Run is incomplete or contains failed attempts")
    for name, expected in run["config_files"].items():
        if digest(_local(root, name)) != expected:
            raise ValueError(f"Resolved configuration changed: {name}")
    episodes = []
    for item in run["episodes"]:
        episode = _local(root, item["path"])
        if digest(episode / "episode.json") != item["sha256"]:
            raise ValueError("Episode completion marker changed")
        episodes.append(verify_episode(episode))
    if len({row["episode_uuid"] for row in episodes}) != len(episodes):
        raise ValueError("Run contains duplicate episodes")
    return {"valid": True, "episodes": episodes}


def main() -> None:
    """Verify a complete recording from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_run(args.run_directory), indent=2))


if __name__ == "__main__":
    main()
