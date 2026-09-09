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

"""Inspect actual recorded motion and decodable video, separately from claims."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from .catalog import write_json

__all__ = ["inspect_attempt", "motion_summary"]


def motion_summary(samples: list[dict]) -> dict:
    """Summarize recorded motion without choosing a task-specific pass threshold."""
    if not samples:
        return {"samples": 0, "task_success": None}
    qpos = np.asarray([sample["qpos"] for sample in samples], dtype=float)
    times = np.asarray([sample["time"] for sample in samples], dtype=float)
    dt = np.diff(times)
    dq = np.diff(qpos, axis=0)
    advancing = dt > 0
    velocities = dq[advancing] / dt[advancing, None, None]
    objects = {}
    for uid in samples[0]["objects"]:
        xyz = np.asarray([sample["objects"][uid] for sample in samples])[..., :3]
        excursion = np.linalg.norm(xyz - xyz[0], axis=-1)
        objects[uid] = {
            "max_displacement_m": float(excursion.max()),
            "final_displacement_m": float(excursion[-1].max()),
            "max_z_increase_m": float((xyz[..., 2] - xyz[0, ..., 2]).max()),
        }
    articulations = {}
    for uid in samples[0].get("articulations", {}):
        positions = np.asarray(
            [sample["articulations"][uid]["qpos"] for sample in samples], dtype=float
        )
        articulations[uid] = {
            "initial_qpos": positions[0].tolist(),
            "final_qpos": positions[-1].tolist(),
            "max_abs_qpos_change": float(np.abs(positions - positions[0]).max()),
        }
    return {
        "samples": len(samples),
        "simulation_seconds": float(times[-1] - times[0]),
        "finite_qpos": bool(np.isfinite(qpos).all()),
        "max_qpos_excursion_rad": float(np.abs(qpos - qpos[0]).max()),
        "sampled_max_joint_speed_rad_s": (
            float(np.abs(velocities).max()) if velocities.size else 0.0
        ),
        "objects": objects,
        "articulations": articulations,
        "task_success": None,
    }


def inspect_attempt(attempt: Path) -> dict:
    """Decode all frames and inspect telemetry; save evidence, not a success label."""
    telemetry = attempt / "telemetry.jsonl"
    samples = (
        [json.loads(line) for line in telemetry.read_text().splitlines()]
        if telemetry.is_file()
        else []
    )
    report = motion_summary(samples)
    video = attempt / "video.mp4"
    report["video"] = str(video.resolve())
    if video.is_file():
        reader = imageio.get_reader(video)
        try:
            count = 0
            difference = 0.0
            previous = None
            for frame in reader:
                count += 1
                if previous is not None:
                    difference = max(
                        difference,
                        float(np.abs(frame.astype(float) - previous).mean()),
                    )
                previous = frame
            report["video_frames_decoded"] = count
            report["max_frame_mean_pixel_change"] = difference
        finally:
            reader.close()
    else:
        report["video_frames_decoded"] = 0
    write_json(attempt / "inspection.json", report)
    return report
