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

"""Shared camera path and visual reference marks, with no physical components."""

from __future__ import annotations
import json
import hashlib
import shutil
import subprocess
from pathlib import Path
import numpy as np
from scripts.benchmark.tech_reports.physics.experiments.p02_protocol import (
    scene_geometry,
)


def view(
    cfg: dict, scene: dict, inputs: dict, initial: int, t: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Use a fixed stack view or the same analytic slope camera path for both engines."""
    g = scene_geometry(cfg, scene, inputs, initial)
    if scene["kind"] == "slope":
        theta = np.deg2rad(scene["angle_deg"])
        sliding = np.tan(theta) > scene["mu_static"]
        a = (
            9.81 * (np.sin(theta) - scene["mu_dynamic"] * np.cos(theta))
            if sliding
            else 0
        )
        target = g["body_poses"][0, :3] + g["downhill"] * (0.5 * a * t * t + 0.1)
        eye = (
            target
            + g["downhill"] * 0.15
            + np.array([0, -1.0, 0])
            + g["surface_normal"] * 0.55
        )
        return eye, target, 1.0 if scene["angle_deg"] >= 30 else 10.0
    height = cfg["capture"].get("stack_view_layers", scene["layers"]) * 0.05
    target = np.array([0, 0, cfg["stack_support_top_z_m"] + height / 2])
    distance = max(0.55, 2.8 * height)
    eye = target + np.array([0.43 * distance, -distance, 0.25 * distance])
    return eye, target, cfg["capture"].get("stack_duration_s", 12.0)


def marks(cfg: dict, scene: dict) -> list[dict]:
    """Render-only rulers; these have no collision, mass or solver participation."""
    result = []
    if scene["kind"] == "slope":
        theta = np.deg2rad(scene["angle_deg"])
        normal = np.array([np.sin(theta), 0, np.cos(theta)])
        along = np.array([np.cos(theta), 0, -np.sin(theta)])
        for i, s in enumerate(np.arange(-6, 6.001, 0.25)):
            result.append(
                dict(
                    uid=f"ruler_{i}",
                    pos=(
                        np.array(cfg["slope_top_origin_m"])
                        + along * s
                        + normal * 0.0006
                    ).tolist(),
                    rot=[0, scene["angle_deg"], 0],
                    size=[0.0015, 0.9, 0.0003],
                )
            )
    else:
        z = cfg["stack_support_top_z_m"] + 0.0006
        for i, x in enumerate(np.arange(-0.5, 0.501, 0.05)):
            result.append(
                dict(
                    uid=f"grid_x{i}",
                    pos=[x, 0, z],
                    rot=[0, 0, 0],
                    size=[0.0005, 1, 0.0003],
                )
            )
            result.append(
                dict(
                    uid=f"grid_y{i}",
                    pos=[0, x, z],
                    rot=[0, 0, 0],
                    size=[1, 0.0005, 0.0003],
                )
            )
    return result


def freeze_capture_sources(output: Path, source: Path) -> None:
    files = [source, Path(__file__)]
    hashes = {}
    for p in files:
        shutil.copy2(p, output / "config" / p.name)
        hashes[str(p.resolve())] = hashlib.sha256(p.read_bytes()).hexdigest()
    (output / "config" / "capture-sources.json").write_text(
        json.dumps(hashes, indent=2) + "\n"
    )


def encode(output: Path, frames: list[dict], cfg: dict, source: str) -> dict:
    """Keep original frames and exact encoding command alongside the video."""
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(cfg["fps"]),
        "-i",
        str(output / "frame_%04d.png"),
        "-frames:v",
        str(len(frames)),
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output / "native_1x.mp4"),
    ]
    subprocess.run(cmd, check=True)
    meta = dict(
        source=source,
        common_intrinsics=cfg["intrinsics"],
        width=cfg["width"],
        height=cfg["height"],
        fps=cfg["fps"],
        playback_speed=1,
        physics_pose_writes_during_rollout=False,
        first_frame_time_s=frames[0]["time_s"],
        last_frame_time_s=frames[-1]["time_s"],
        encoder_command=cmd,
        frames=frames,
    )
    (output / "capture.json").write_text(json.dumps(meta, indent=2) + "\n")
    return {k: v for k, v in meta.items() if k != "frames"}
