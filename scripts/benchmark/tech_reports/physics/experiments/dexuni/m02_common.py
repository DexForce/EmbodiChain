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
"""Shared M02 (task 6) cloth geometry, node groups, phases and fold metrics.

The sheet is picked up along its centre line (the fold line), lifted so both
halves hang down, and laid back on the worktop as two layers.
"""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

OUTPUT = (
    Path(__file__).resolve().parents[3].parents[2]
    / "outputs/benchmarks/tech_reports/dexuni/m02_cloth_single_fold"
)
TABLE_Z = 0.86
CLOTH_SIZE = np.array((0.40, 0.48))  # X (folded across), Y (fold line)
CLOTH_SPACING = 0.01  # same node spacing as S03
CLOTH_CENTER = np.array((0.40, 0.0, 0.868))
GROUP_HALF = 0.01  # node groups: nodes within 10 mm (X and Y) of a point
PHASES = (
    (0.5, "home"),
    (2.0, "approach"),
    (3.0, "descend"),
    (4.0, "close"),
    (4.4, "settle"),
    (7.0, "lift"),
    (8.0, "hang"),
    (9.5, "move to lay start"),
    (12.5, "lay down"),
    (13.5, "release"),
    (14.5, "retreat"),
    (16.5, "observe"),
)
from scripts.benchmark.tech_reports.physics.experiments.dexuni.reference_inputs import (
    verify_reference,
)

__all__ = [
    "CLOTH_CENTER",
    "CLOTH_SIZE",
    "OUTPUT",
    "PHASES",
    "TABLE_Z",
    "cloth_mesh",
    "fold_metrics",
    "load_inputs",
    "node_groups",
    "phase",
    "record_preview",
]


def phase(time: float) -> str:
    """Name of the scheduled phase at ``time``."""
    for end, name in PHASES:
        if time <= end + 1e-9:
            return name
    return "observe"


def cloth_mesh() -> dict[str, np.ndarray]:
    """Flat sheet with S03's node spacing, triangulated symmetrically.

    The cell diagonal flips per quadrant, so the mesh is mirror-symmetric about
    both centre lines (one uniform diagonal folded the two sides differently).
    Winding is +Z.
    """
    nx, ny = (np.round(CLOTH_SIZE / CLOTH_SPACING).astype(int) + 1).tolist()
    xs = np.linspace(-CLOTH_SIZE[0] / 2, CLOTH_SIZE[0] / 2, nx)
    ys = np.linspace(-CLOTH_SIZE[1] / 2, CLOTH_SIZE[1] / 2, ny)
    grid = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
    vertices = np.c_[grid, np.zeros(len(grid))] + CLOTH_CENTER
    triangles = []
    for i in range(nx - 1):  # x index (id = i * ny + j)
        for j in range(ny - 1):  # y index
            a = i * ny + j
            if (i < (nx - 1) // 2) != (j < (ny - 1) // 2):
                triangles.extend(((a, a + ny, a + ny + 1), (a, a + ny + 1, a + 1)))
            else:
                triangles.extend(((a, a + ny, a + 1), (a + ny, a + ny + 1, a + 1)))
    return {
        "vertices": vertices.astype(np.float32),
        "triangles": np.asarray(triangles, dtype=np.int32),
    }


def node_groups(rest: np.ndarray) -> dict[str, np.ndarray]:
    """Material node groups picked once from the rest sheet.

    ``near_*`` (robot side, small X) and ``far_*`` are the two free edges that
    should lie on top of each other after the fold; ``left`` is +Y.
    """
    x0, x1 = rest[:, 0].min(), rest[:, 0].max()
    y0, y1 = rest[:, 1].min(), rest[:, 1].max()
    points = {
        "near_left": (x0, y1),
        "near_right": (x0, y0),
        "near_mid": (x0, 0.0),
        "far_left": (x1, y1),
        "far_right": (x1, y0),
        "far_mid": (x1, 0.0),
    }
    groups = {}
    for name, (x, y) in points.items():
        ids = np.flatnonzero(
            (np.abs(rest[:, 0] - x) <= GROUP_HALF + 1e-6)
            & (np.abs(rest[:, 1] - y) <= GROUP_HALF + 1e-6)
        )
        if ids.size == 0:
            raise ValueError(f"Empty node group {name}")
        groups[name] = ids
    return groups


def fold_metrics(particles: np.ndarray, groups: dict) -> dict:
    """XY alignment of the two free-edge groups, plus heights (metres)."""
    centre = {k: particles[v].mean(axis=0) for k, v in groups.items()}
    result = {}
    for side in ("left", "right", "mid"):
        near, far = centre[f"near_{side}"], centre[f"far_{side}"]
        result[f"{side}_xy_error_m"] = float(np.linalg.norm(near[:2] - far[:2]))
        result[f"near_{side}_z_m"] = float(near[2])
        result[f"far_{side}_z_m"] = float(far[2])
    result["lowest_z_m"] = float(particles[:, 2].min())
    result["highest_z_m"] = float(particles[:, 2].max())
    return result


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(root: Path = OUTPUT) -> dict:
    """Load the approved config, cloth mesh and named joint commands.

    Raises if approval is missing or the available reference evidence changed.
    """
    from .download_artifacts import ensure_task_inputs

    ensure_task_inputs("M02", root)
    config = json.loads((root / "inputs/m02_config.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest.get("user_scene_approved"):
        raise ValueError("Native M02 scene approval is required")
    verify_reference(root, config["native_preview_sha256"])
    for name, expected in config["input_sha256"].items():
        if _sha(root / "inputs" / name) != expected:
            raise ValueError(f"Frozen M02 input changed: {name}")
    mesh = dict(np.load(root / "inputs/cloth_mesh.npz"))
    commands = dict(np.load(root / "inputs/joint_commands.npz"))
    names = commands["joint_names"].tolist()
    targets = commands["targets"]
    if len(set(names)) != len(names) or targets.shape != (
        config["steps"] + 1,
        len(names),
    ):
        raise ValueError("Invalid named joint command layout")
    if not np.allclose(
        commands["time"], np.arange(config["steps"] + 1) * config["dt"], atol=1e-9
    ):
        raise ValueError("Joint command time grid differs from the approved scene")
    groups = {
        key[len("group_") :]: value
        for key, value in mesh.items()
        if key.startswith("group_")
    }
    return {
        "config": config,
        "mesh": mesh,
        "groups": groups,
        "names": names,
        "targets": targets,
        "time": commands["time"],
        "tcp_baked": commands["tcp_baked"],
    }


def record_preview(
    platform: str,
    inputs: dict,
    step: Callable[[int], None],
    readings: Callable[[], dict[str, np.ndarray]],
    capture: Callable[[], np.ndarray],
    diagnostics: Callable[[], dict],
) -> None:
    """Record actual states, fold metrics and one video (no timing).

    Args:
        platform: Platform label used for output and captions.
        inputs: Frozen scene and named joint command arrays.
        step: Advance one outer step to the indexed frame.
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
            p.name: _sha(p) for p in (OUTPUT / "inputs").glob("*") if p.is_file()
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
                fold = fold_metrics(np.asarray(row["particles"]), inputs["groups"])
                fold.update(
                    frame=frame,
                    time=time,
                    phase=phase(time),
                    tcp_error_m=np.linalg.norm(
                        np.asarray(row["tcp"]) - inputs["tcp_baked"][frame], axis=1
                    ),
                )
                metrics.append(fold)
                if frame % 2 == 0:
                    image = Image.fromarray(
                        np.asarray(capture(), dtype=np.uint8)[..., :3]
                    )
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((0, 0, image.width, 42), fill="black")
                    draw.text(
                        (14, 10),
                        f"{platform} | M02 cloth single fold (one-way) | t={time:.2f}s | {fold['phase']} | "
                        f"corner XY error L {fold['left_xy_error_m'] * 1e3:.0f} / R {fold['right_xy_error_m'] * 1e3:.0f} mm",
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
            (output / "metrics.json").write_text(
                json.dumps(metrics, default=lambda v: np.asarray(v).tolist()) + "\n"
            )
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=lambda v: np.asarray(v).tolist())
            + "\n"
        )
