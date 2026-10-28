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
"""Shared frozen inputs, beam metrics and preview recording for M01 (task 5)."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

OUTPUT = (
    Path(__file__).resolve().parents[3].parents[2]
    / "outputs/benchmarks/tech_reports/dexuni/m01_soft_beam_bend"
)
TABLE_Z = 0.86
# (end time in seconds, phase name) of the approved DexSim motion.
PHASES = (
    (0.5, "home"),
    (2.0, "approach"),
    (3.0, "descend"),
    (4.0, "close"),
    (4.4, "settle"),
    (6.0, "lift"),
    (6.5, "reference"),
    (8.0, "bend"),
    (10.0, "hold"),
    (11.5, "restore"),
    (13.5, "carry"),
    (14.5, "lower"),
    (15.2, "release"),
    (16.0, "retreat"),
)
from scripts.benchmark.tech_reports.physics.experiments.dexuni.reference_inputs import (
    verify_reference,
)

__all__ = [
    "OUTPUT",
    "PHASES",
    "TABLE_Z",
    "beam_metrics",
    "load_inputs",
    "phase",
    "record_preview",
    "static_boxes",
]


def phase(time: float) -> str:
    """Name of the scheduled phase at ``time``."""
    for end, name in PHASES:
        if time <= end + 1e-9:
            return name
    return "observe"


def beam_metrics(particles: np.ndarray, groups: dict) -> dict:
    """Sag of the middle node group relative to the line through the end groups.

    Args:
        particles: Current node positions, shape (N, 3).
        groups: Node indices of the ``left``, ``right`` and ``middle`` groups.

    Returns:
        Signed sag (positive downward), end-group distance, lowest node height
        and group heights, all in metres.
    """
    left = particles[groups["left"]].mean(axis=0)
    right = particles[groups["right"]].mean(axis=0)
    middle = particles[groups["middle"]].mean(axis=0)
    axis = left - right
    axis /= np.linalg.norm(axis)
    closest = right + np.dot(middle - right, axis) * axis
    sag = float(np.linalg.norm(middle - closest) * np.sign(closest[2] - middle[2]))
    return {
        "sag_m": sag,
        "end_distance_m": float(np.linalg.norm(left - right)),
        "lowest_z_m": float(particles[:, 2].min()),
        "middle_z_m": float(middle[2]),
        "end_z_m": [float(left[2]), float(right[2])],
    }


def load_inputs(root: Path = OUTPUT) -> dict:
    """Load the approved config, tetrahedral mesh and named joint commands.

    Args:
        root: The single M01 output directory.

    Returns:
        Config, mesh arrays (with measured node groups), joint names, targets
        and their time grid.
    """
    from .download_artifacts import ensure_task_inputs

    ensure_task_inputs("M01", root)
    config = json.loads((root / "inputs/m01_config.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest.get("user_scene_approved"):
        raise ValueError("Native M01 scene approval is required")
    verify_reference(root, config["native_preview_sha256"])
    mesh = dict(np.load(root / "inputs/soft_mesh.npz"))
    commands = dict(np.load(root / "inputs/joint_commands.npz"))
    names = commands["joint_names"].tolist()
    targets = commands["targets"]
    if len(set(names)) != len(names) or targets.shape != (
        config["steps"] + 1,
        len(names),
    ):
        raise ValueError("Invalid named joint command layout")
    if not np.isfinite(targets).all():
        raise ValueError("Joint command targets must be finite")
    if not np.allclose(
        commands["time"], np.arange(config["steps"] + 1) * config["dt"], atol=1e-9
    ):
        raise ValueError("Joint command time grid differs from the approved scene")
    groups = {k: mesh[f"group_{k}"] for k in ("left", "right", "middle")}
    return {
        "config": config,
        "mesh": mesh,
        "groups": groups,
        "names": names,
        "targets": targets,
        "time": commands["time"],
    }


def static_boxes(config: dict) -> list[tuple[str, tuple, tuple, tuple]]:
    """Static worktop, legs and the two flat marks of the approved scene.

    Returns:
        (name, centre, full size, RGB colour) per box, in metres.
    """
    boxes = [
        ("table", (0.46, 0.0, TABLE_Z - 0.025), (0.6, 1.64, 0.05), (0.5, 0.5, 0.5)),
        *[
            (
                f"leg_{i}_{j}",
                (x, y, (TABLE_Z - 0.05) / 2),
                (0.05, 0.05, TABLE_Z - 0.05),
                (0.5, 0.5, 0.5),
            )
            for i, x in enumerate((0.21, 0.71))
            for j, y in enumerate((-0.74, 0.74))
        ],
    ]
    for mark in config["marks"]:
        boxes.append(
            (
                mark["name"],
                tuple(mark["center"]),
                tuple(mark["size"]),
                tuple(mark["color"]),
            )
        )
    return boxes


def record_preview(
    platform: str,
    inputs: dict,
    step: Callable[[int], None],
    readings: Callable[[], dict[str, np.ndarray]],
    capture: Callable[[], np.ndarray],
    diagnostics: Callable[[], dict],
) -> None:
    """Record actual dynamic states, beam metrics and one video (no timing).

    Args:
        platform: Platform label used for output and captions.
        inputs: Frozen scene and named joint command arrays.
        step: Advance one outer step with the indexed joint target.
        readings: Read ``qpos``, ``particles`` and ``tcp`` without changing state.
        capture: Render the current state to an RGB array.
        diagnostics: Read initialized physics settings and graph state.
    """
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    output = OUTPUT / "preview" / platform
    output.mkdir(parents=True, exist_ok=True)
    config = inputs["config"]
    manifest = {
        "platform": platform,
        "mode": "scene_preview_only",
        "status": "recording",
        "performance_measurements_started": False,
        "executable": sys.executable,
        "config": config,
        "input_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (OUTPUT / "inputs").glob("*")
            if p.is_file()
        },
    }
    states: dict[str, list] = {}
    metrics = []
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    try:
        with imageio.get_writer(
            str(output / "video.mp4"),
            fps=30,
            codec="libx264",
            quality=8,
            macro_block_size=2,
        ) as writer:
            for frame in range(config["steps"] + 1):
                if frame:
                    step(frame)
                time = frame * config["dt"]
                row = readings()
                row.update(time=np.asarray(time), target_q=inputs["targets"][frame])
                for name, value in row.items():
                    array = np.asarray(value)
                    if not np.isfinite(array).all():
                        raise ValueError(f"Nonfinite {name} at frame {frame}")
                    states.setdefault(name, []).append(array.copy())
                beam = beam_metrics(np.asarray(row["particles"]), inputs["groups"])
                beam.update(frame=frame, time=time, phase=phase(time))
                metrics.append(beam)
                if frame % 2 == 0:
                    image = Image.fromarray(
                        np.asarray(capture(), dtype=np.uint8)[..., :3]
                    )
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((0, 0, image.width, 42), fill="black")
                    draw.text(
                        (14, 10),
                        f"{platform} | M01 soft beam bend | t={time:.2f}s | {beam['phase']} | sag {beam['sag_m'] * 1e3:.1f} mm",
                        font=font,
                        fill="white",
                    )
                    writer.append_data(np.asarray(image))
                    if frame == 0:
                        image.save(output / "first_frame.jpg")
                if frame % 60 == 0:
                    print(f"{platform}: {frame}/{config['steps']} frames", flush=True)
        manifest.update(
            status="completed", frames=config["steps"], effective=diagnostics()
        )
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        if states:
            np.savez_compressed(
                output / "states.npz",
                joint_names=np.asarray(inputs["names"]),
                **{key: np.stack(values) for key, values in states.items()},
            )
        if metrics:
            (output / "metrics.json").write_text(json.dumps(metrics) + "\n")
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=lambda v: np.asarray(v).tolist())
            + "\n"
        )
