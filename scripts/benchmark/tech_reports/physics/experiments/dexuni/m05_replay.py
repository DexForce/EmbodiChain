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
"""Frozen-input loading and physical recording for M05 platform previews.

No timings or formal performance samples are collected here.
"""

from __future__ import annotations

from collections.abc import Callable
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from scripts.benchmark.tech_reports.physics.experiments.dexuni.m05_common import (
    BLOCK_HALF,
    OUTPUT,
    PHASES,
    TABLE_Z,
    load_metrics,
    phase,
)

__all__ = ["load_inputs", "record_preview", "sha256", "write_json"]


def sha256(path: Path) -> str:
    """Return a file's content digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    """Save metadata, converting NumPy values at the recording boundary."""

    def convert(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Cannot serialize {type(item).__name__}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=convert) + "\n")


def load_inputs(run: Path = OUTPUT) -> dict[str, Any]:
    """Load and verify the accepted native trajectory and physical mesh.

    Args:
        run: Effective M05 output directory.

    Returns:
        Frozen scene configuration, named joint samples, and mesh arrays.
    """
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("M05 requires a completed native reference")
    for name, expected in manifest["input_sha256"].items():
        if sha256(run / "inputs" / name) != expected:
            raise ValueError(f"M05 frozen input changed: {name}")
    with np.load(run / "inputs/joint_commands.npz") as archive:
        joints = {name: archive[name].copy() for name in archive.files}
    with np.load(run / "inputs/bag_mesh.npz") as archive:
        mesh = {name: archive[name].copy() for name in archive.files}
    if [list(row) for row in PHASES] != manifest["scene"]["phases"]:
        raise ValueError("Current phase helper differs from frozen M05")
    if manifest["scene"]["block_size_m"] != 2 * BLOCK_HALF:
        raise ValueError("Block metric helper differs from frozen M05")
    return dict(run=run, manifest=manifest, joints=joints, mesh=mesh)


def record_preview(
    platform: str,
    inputs: dict[str, Any],
    output: Path,
    step: Callable[[], None],
    readings: Callable[[], dict[str, np.ndarray]],
    capture: Callable[[], np.ndarray] | None,
    diagnostics: Callable[[], dict[str, Any]],
    audit: dict[str, Any],
    *,
    until: float | None = None,
) -> None:
    """Record actual clock, named joints, contact geometry and one main view.

    Args:
        platform: Platform label.
        inputs: Verified native inputs from :func:`load_inputs`.
        output: One current platform preview directory.
        step: Advance exactly one outer step without rendering.
        readings: Read actual physical state without advancing simulation.
        capture: Render one main camera or omit video for diagnostics.
        diagnostics: Read effective initialized solver information.
        audit: Initial topology, contact and physical-property audit.
        until: Optional diagnostic truncation time; never a completed preview.
    """
    output.mkdir(parents=True, exist_ok=True)
    native = inputs["manifest"]
    joints, mesh = inputs["joints"], inputs["mesh"]
    total = len(joints["time"]) - 1
    if until is not None:
        if until <= 0:
            raise ValueError("Diagnostic truncation time must be positive")
        total = min(total, round(until * 60))
    manifest = dict(
        task="M05",
        platform=platform,
        scope="scene_preview_only; no performance measurements",
        status="recording",
        user_scene_approved=False,
        performance_measurements_started=False,
        reference_manifest_sha256=sha256(inputs["run"] / "manifest.json"),
        input_sha256=native["input_sha256"],
        scene=native["scene"],
        camera=native["camera"],
        executable=sys.executable,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        initial_model=audit,
        joint_state_order=joints["joint_names"].tolist(),
    )
    rows: list[dict[str, Any]] = []
    state: dict[str, list[np.ndarray]] = {}
    writer = None
    try:
        if capture is not None:
            import imageio.v2 as imageio
            from PIL import Image, ImageDraw, ImageFont

            font = ImageFont.truetype("DejaVuSans.ttf", 20)
            writer = imageio.get_writer(
                str(output / "video.mp4"),
                fps=30,
                codec="libx264",
                quality=8,
                macro_block_size=2,
            )
        still_times = (2.0, 6.0, 14.6, 24.0, 35.0)
        still_frames = {round(t * 60) for t in still_times}
        for frame in range(total + 1):
            if frame:
                step()
            sample = readings()
            time = float(sample["time"])
            if abs(time - joints["time"][frame]) > 1e-7:
                raise RuntimeError(f"Actual clock mismatch at frame {frame}: {time}")
            for name, value in sample.items():
                if not np.isfinite(value).all():
                    raise RuntimeError(f"Nonfinite {name} at frame {frame}")
            if not np.allclose(
                sample["qpos"], joints["targets"][frame], atol=2e-5, rtol=0
            ):
                raise RuntimeError(f"Named joint replay mismatch at frame {frame}")
            bag, pose = sample["bag"], sample["block_pose"]
            row = load_metrics(bag, pose, mesh["top_centre_group"])
            a, b, c = (bag[mesh["triangles"][:, i]] for i in range(3))
            volume = abs(
                float(
                    np.einsum("ij,ij->i", a - a[0], np.cross(b - a[0], c - a[0])).sum()
                    / 6
                )
            )
            row.update(
                frame=frame,
                time=time,
                phase=phase(time),
                tcp=sample["tcp"],
                tcp_error_m=np.linalg.norm(
                    sample["tcp"] - joints["tcp_baked"][frame], axis=1
                ),
                volume_ratio=volume / float(mesh["rest_volume"]),
                block_tilt_deg=float(
                    np.rad2deg(np.arccos(np.clip(np.abs(pose[2, :3]).max(), 0, 1)))
                ),
            )
            rows.append(row)
            sample["target_q"] = joints["targets"][frame]
            for name, value in sample.items():
                state.setdefault(name, []).append(np.array(value, copy=True))
            if capture is not None and (frame % 2 == 0 or frame in still_frames):
                image = Image.fromarray(capture())
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, image.width, 42), fill="black")
                draw.text(
                    (14, 10),
                    f"{platform} | M05 inflatable bag + rigid load (one-way) | t={time:.2f}s | "
                    f"{row['phase']} | bag h {row['bag_height_m'] * 1e3:.1f} mm | "
                    f"block bottom {(row['block_bottom_z_m'] - TABLE_Z) * 1e3:.1f} mm",
                    font=font,
                    fill="white",
                )
                if frame % 2 == 0:
                    writer.append_data(np.asarray(image))
                if frame in still_frames:
                    image.save(output / f"still_{time:05.2f}s.jpg")
            if frame % 120 == 0:
                print(
                    f"frame={frame}/{total} t={time:.2f} bag={bag.mean(0).round(3)} "
                    f"block={pose[:3,3].round(3)}",
                    flush=True,
                )
        effective = diagnostics()
        if effective["cuda_graph_status"] != "captured":
            raise RuntimeError("M05 CUDA Graph was not captured")
        if effective["features"]["pneumatic_cavity_count"] != 1:
            raise RuntimeError("M05 must have exactly one active gas cavity")
        if effective["vbd"]["iterations"] != 24 or effective["num_substeps"] != 8:
            raise RuntimeError(
                "M05 effective solver budget differs from accepted scene"
            )
        manifest.update(
            status=(
                "completed"
                if total == len(joints["time"]) - 1
                else "diagnostic_truncated"
            ),
            frames=total,
            actual_simulation_time_s=time,
            named_joint_replay_verified=True,
            effective=effective,
        )
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        if writer is not None:
            writer.close()
        if state:
            np.savez_compressed(
                output / "states.npz", **{k: np.stack(v) for k, v in state.items()}
            )
        write_json(output / "metrics.json", rows)
        manifest["ended_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(output / "manifest.json", manifest)
