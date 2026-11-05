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
"""Approved M03 inputs, exact supports and untimed cross-platform recording."""

from __future__ import annotations
from collections.abc import Callable
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
import numpy as np
from scripts.benchmark.tech_reports.physics.platforms.dexuni.dexsim.m03 import (
    observations,
    phase,
)
from scripts.benchmark.tech_reports.reporting.dexuni.build_m03_preview import quality

OUTPUT = (
    Path(__file__).resolve().parents[3].parents[2]
    / "outputs/benchmarks/tech_reports/dexuni/m03_four_layer_cloth_pick_place"
)
CAMERAS = {
    "perspective": ((960, 720), (1.7, -1.45, 1.65), (0.4, 0, 1.04)),
    "detail": ((720, 720), (0.68, -0.68, 1.40), (0.44, 0, 1.08)),
}
__all__ = [
    "OUTPUT",
    "CAMERAS",
    "load_inputs",
    "boxes",
    "record_preview",
    "write_json",
    "digest",
    "audit_layers",
]


def digest(path: Path) -> str:
    """Hash an input or recorded artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Write NumPy-safe detached metadata."""
    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            default=lambda v: np.asarray(v).tolist(),
        )
        + "\n"
    )


def load_inputs() -> dict[str, Any]:
    """Verify the single current input set for an untimed, unapproved preview."""
    root = OUTPUT
    manifest = json.loads((root / "inputs/manifest.json").read_text())
    for path, expected in manifest["sha256"].items():
        if digest(root / "inputs" / path) != expected:
            raise ValueError(f"Current input changed: {path}")
    native = json.loads((root / "manifest.json").read_text())
    if (
        native["status"] != "completed"
        or not native["full_trajectory"]
        or native["performance_measurements_started"]
        or not native["observations"]["all_states_finite"]
    ):
        raise ValueError("A complete native scene diagnostic is required")
    if native["config_sha256"] != digest(root / "inputs/config.json"):
        raise ValueError("Inputs changed after the current native diagnostic")
    config = json.loads((root / "inputs/config.json").read_text())
    with np.load(root / "inputs/cloth_mesh.npz") as data:
        mesh = {k: data[k] for k in data.files}
    with np.load(root / "inputs/joint_commands.npz") as data:
        commands = {k: data[k] for k in data.files}
    names = commands["joint_names"].tolist()
    steps = round(config["motion"]["duration"] / config["solver"]["dt"])
    assert len(names) == len(set(names)) == 24 and commands["targets"].shape == (
        steps + 1,
        24,
    )
    assert np.allclose(
        commands["time"], np.arange(steps + 1) * config["solver"]["dt"], atol=1e-10
    )
    return dict(
        output_root=root,
        user_scene_approved=False,
        config=config,
        mesh=mesh,
        commands=commands,
        names=names,
        targets=commands["targets"],
        steps=steps,
    )


def boxes(config: dict[str, Any]) -> list[tuple]:
    """Exact native supports and visual target markers, without retuning."""
    s = config["support"]
    tz, pz = s["table_z"], s["pickup_z"]
    result = [
        ("table", (0.49, 0, tz - 0.025), (0.68, 1.2, 0.05), (0.45, 0.44, 0.41), True)
    ]
    for i, x in enumerate((0.2, 0.78)):
        for j, y in enumerate((-0.53, 0.53)):
            result.append(
                (
                    f"leg_{i}_{j}",
                    (x, y, (tz - 0.05) / 2),
                    (0.05, 0.05, tz - 0.05),
                    (0.22, 0.24, 0.27),
                    True,
                )
            )
    xe = s["channel_end_x"]
    result.append(
        (
            "pickup_rear",
            ((xe + 0.79) / 2, 0, (tz + pz) / 2),
            (0.79 - xe, 0.36, pz - tz),
            (0.62, 0.58, 0.48),
            True,
        )
    )
    half = s["channel_width"] / 2
    gy = config["motion"]["grasp_y"]
    for i, (a, b) in enumerate(
        ((-0.18, -gy - half), (-gy + half, gy - half), (gy + half, 0.18))
    ):
        result.append(
            (
                f"pickup_front_{i}",
                ((0.265 + xe) / 2, (a + b) / 2, (tz + pz) / 2),
                (xe - 0.265, b - a, pz - tz),
                (0.62, 0.58, 0.48),
                True,
            )
        )
    cx, cy = s["target_center_xy"]
    hs = s["target_size"] / 2
    for i, y in enumerate((cy - hs, cy + hs)):
        result.append(
            (
                f"target_x_{i}",
                (cx, y, pz + 0.0005),
                (2 * hs, 0.004, 0.001),
                (0.12, 0.85, 0.75),
                False,
            )
        )
    for i, x in enumerate((cx - hs, cx + hs)):
        result.append(
            (
                f"target_y_{i}",
                (x, cy, pz + 0.0005),
                (0.004, 2 * hs, 0.001),
                (0.12, 0.85, 0.75),
                False,
            )
        )
    return result


def audit_layers(
    output: Path,
    inputs: dict,
    positions: np.ndarray,
    inverse_mass: np.ndarray,
    **extra: Any,
) -> dict:
    """Verify four independent fully dynamic 18g layers before simulation."""
    assert positions.shape == inputs["mesh"]["initial_positions"].shape == (4, 961, 3)
    if not np.allclose(positions, inputs["mesh"]["initial_positions"], atol=1e-6):
        raise ValueError("Initial cloth node order or coordinates differ")
    if not np.isfinite(inverse_mass).all() or np.any(inverse_mass <= 0):
        raise ValueError("Cloth nodes must all be dynamic")
    mass = (1 / inverse_mass).sum(1)
    if not np.allclose(mass, 0.018, atol=1e-7):
        raise ValueError(f"Each layer must weigh 18g: {mass}")
    np.savez_compressed(
        output / "initial_model.npz",
        positions=positions,
        inverse_mass=inverse_mass,
        triangles=inputs["mesh"]["triangles"],
        **extra,
    )
    return dict(
        nodes_per_layer=[961] * 4,
        triangles_per_layer=[1800] * 4,
        layer_mass_kg=mass,
        all_inverse_mass_positive=True,
        no_attachments=True,
    )


def record_preview(
    platform: str,
    inputs: dict,
    step: Callable,
    readings: Callable,
    capture: Callable | None,
    diagnostics: Callable,
    model_audit: dict,
    source: Path,
) -> None:
    """Record a complete actual-time episode; collect no performance samples."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    from contextlib import ExitStack

    root = inputs.get("output_root", OUTPUT)
    output = root / "preview" / platform
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        task="M03",
        platform=platform,
        status="running",
        scope="scene_preview_only",
        performance_measurements_started=False,
        python=sys.executable,
        source_sha256=digest(source),
        common_source_sha256=digest(Path(__file__)),
        user_scene_approved=False,
        input_manifest_sha256=digest(root / "inputs/manifest.json"),
        initial_model=model_audit,
        requested=inputs["config"],
    )
    write_json(output / "manifest.json", manifest)
    rows = {}
    times = []
    font = ImageFont.truetype("DejaVuSans.ttf", 21)
    try:
        with ExitStack() as stack:
            writer = (
                stack.enter_context(
                    imageio.get_writer(
                        str(output / "video.mp4"),
                        fps=30,
                        codec="libx264",
                        quality=8,
                        macro_block_size=2,
                    )
                )
                if capture
                else None
            )
            for frame in range(inputs["steps"] + 1):
                if frame:
                    step(frame)
                row = readings()
                t = float(row["time"])
                if abs(t - inputs["commands"]["time"][frame]) > 1e-6:
                    raise ValueError(f"Clock mismatch at {frame}: {t}")
                if not np.allclose(row["qpos"], inputs["targets"][frame], atol=1e-5):
                    raise ValueError(f"Joint replay mismatch at {frame}")
                for name, value in row.items():
                    value = np.asarray(value)
                    if not np.isfinite(value).all():
                        raise ValueError(f"Nonfinite {name} at {frame}")
                    rows.setdefault(name, []).append(value.copy())
                if writer and frame % 2 == 0:
                    views = capture()
                    image = Image.fromarray(
                        np.concatenate(views, axis=1).astype(np.uint8)
                    )
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((0, 0, image.width, 42), fill="black")
                    policy = (
                        "surface-fast"
                        if platform == "embodichain"
                        else f"stock VBD {inputs.get('stock_iterations', 10)} iterations"
                    )
                    draw.text(
                        (14, 10),
                        f"{platform} | M03 four independent cloths | one-way | {policy} | t={t:.2f}s | {phase(t,inputs['config'])}",
                        font=font,
                        fill="white",
                    )
                    writer.append_data(np.asarray(image))
                    times.append(t)
                    if frame in (0, 480, 960, 1170, 1230, 1410):
                        image.save(output / f"stage_{frame:04d}.jpg")
                if frame % 120 == 0:
                    print(f"{platform}: {frame}/{inputs['steps']}", flush=True)
        states = {k: np.stack(v) for k, v in rows.items()}
        obs = observations(states, inputs["config"])
        obs.update(
            scope="single platform preview; descriptive candidate criteria",
            max_joint_error=float(np.abs(states["qpos"] - inputs["targets"]).max()),
            max_tcp_replay_error_m=float(
                np.linalg.norm(
                    states["tcp"] - inputs["commands"]["tcp_baked"], axis=2
                ).max()
            ),
        )
        if obs["max_tcp_replay_error_m"] > 0.004:
            raise ValueError("TCP replay error exceeds tolerance")
        write_json(output / "observations.json", obs)
        q = quality(inputs["config"], states, inputs["mesh"])
        q["scope"] = "single platform geometry diagnostics; no performance samples"
        write_json(output / "quality.json", q)
        effective = diagnostics()
        if effective.get("cuda_graph_status") != "captured":
            raise ValueError("CUDA Graph was not captured")
        manifest.update(
            status="completed",
            outer_steps=inputs["steps"],
            actual_end_time=float(states["time"][-1]),
            effective=effective,
        )
        if writer:
            manifest["video_sha256"] = digest(output / "video.mp4")
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        if rows:
            np.savez_compressed(
                output / "states.npz", **{k: np.stack(v) for k, v in rows.items()}
            )
        write_json(output / "frame_times.json", times)
        write_json(output / "manifest.json", manifest)
