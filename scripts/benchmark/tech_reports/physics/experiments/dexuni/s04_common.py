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
"""Shared frozen inputs and preview recording for the approved S04 scene."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

OUTPUT = (
    Path(__file__).resolve().parents[3].parents[2]
    / "outputs/benchmarks/tech_reports/dexuni/s04_rigid_soft_squeeze_grasp"
)
from scripts.benchmark.tech_reports.physics.experiments.dexuni.reference_inputs import (
    verify_reference,
)

__all__ = ["OUTPUT", "load_inputs", "lame_to_youngs", "static_boxes", "record_preview"]


def load_inputs(root: Path = OUTPUT) -> dict:
    """Load named commands and the unchanged approved tetrahedral mesh.

    Args:
        root: The single S04 output directory.

    Returns:
        Config, mesh arrays, named joint targets and their time grid.
    """
    config = json.loads((root / "inputs/s04_config.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest["user_scene_approved"]:
        raise ValueError("Native S04 scene approval is required")
    verify_reference(root, config["native_preview_sha256"])
    mesh = dict(np.load(root / "inputs/soft_mesh.npz"))
    commands = dict(np.load(root / "inputs/joint_commands.npz"))
    names = commands["joint_names"].tolist()
    if len(set(names)) != len(names) or commands["targets"].shape != (
        config["steps"] + 1,
        len(names),
    ):
        raise ValueError("Invalid named joint command layout")
    if not np.isfinite(commands["targets"]).all():
        raise ValueError("Joint command targets must be finite")
    if not np.allclose(
        commands["time"],
        np.arange(config["steps"] + 1) * config["dt"],
        rtol=0,
        atol=1e-9,
    ):
        raise ValueError("Joint command time grid differs from the approved scene")
    return {
        "config": config,
        "mesh": mesh,
        "names": names,
        "targets": commands["targets"],
        "time": commands["time"],
    }


def lame_to_youngs(mu: float, lam: float) -> tuple[float, float]:
    """Convert the approved Lamé coefficients to Young's modulus and Poisson ratio.

    Args:
        mu: Shear modulus in Pa.
        lam: First Lamé parameter in Pa.

    Returns:
        Young's modulus in Pa and dimensionless Poisson ratio.
    """
    return mu * (3 * lam + 2 * mu) / (lam + mu), lam / (2 * (lam + mu))


def static_boxes() -> list[tuple[str, tuple, tuple]]:
    """Return the table and legs from the approved native reference."""
    return [
        ("table", (0.46, 0.0, 0.835), (0.6, 1.64, 0.05)),
        *[
            (f"leg_{i}_{j}", (x, y, 0.405), (0.05, 0.05, 0.81))
            for i, x in enumerate((0.21, 0.71))
            for j, y in enumerate((-0.74, 0.74))
        ],
    ]


def record_preview(
    platform: str,
    inputs: dict,
    step: Callable[[int], None],
    readings: Callable[[], dict[str, np.ndarray]],
    capture: Callable[[], np.ndarray],
    diagnostics: Callable[[], dict],
) -> None:
    """Record actual dynamic states and one video without measuring performance.

    Args:
        platform: Platform label used for output and captions.
        inputs: Frozen scene and named joint command arrays.
        step: Advance one outer step with the indexed joint target.
        readings: Read positions without changing simulation state.
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
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    phases = (
        (2.8, "approach"),
        (4.3, "close"),
        (5.8, "lift"),
        (7.3, "hold"),
        (8.8, "lower"),
        (9.6, "release"),
        (10.6, "retreat"),
        (12.01, "observe"),
    )
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
                row = readings()
                row.update(
                    time=np.asarray(frame * config["dt"]),
                    target_q=inputs["targets"][frame],
                )
                for name, value in row.items():
                    array = np.asarray(value)
                    if not np.isfinite(array).all():
                        raise ValueError(f"Nonfinite {name} at frame {frame}")
                    states.setdefault(name, []).append(array.copy())
                if frame % 2 == 0:
                    image = Image.fromarray(
                        np.asarray(capture(), dtype=np.uint8)[..., :3]
                    )
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((0, 0, image.width, 42), fill="black")
                    phase = next(
                        name for end, name in phases if frame * config["dt"] <= end
                    )
                    draw.text(
                        (14, 10),
                        f"{platform} | S04 rigid / soft grasp | t={frame * config['dt']:.2f}s | {phase}",
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
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
