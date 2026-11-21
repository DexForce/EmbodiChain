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
"""Load approved M04 inputs and record untimed platform scene previews."""

from __future__ import annotations

from collections.abc import Callable
import datetime as dt
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.benchmark.tech_reports.physics.experiments.dexuni.m04_common import (
    OUTPUT,
    digest,
    phase,
    write_json,
)

__all__ = [
    "CAMERA",
    "load_inputs",
    "finger_schedule",
    "shape_damping",
    "texture_uv",
    "link_matrices",
    "audit_cloth",
    "record_preview",
]
CAMERA = dict(
    resolution=(1280, 720),
    eye=(1.45, -1.05, 1.85),
    target=(0.47, 0, 1.05),
    horizontal_fov_deg=42.0,
)


def load_inputs() -> dict[str, Any]:
    """Verify and load the complete user-approved native scene and commands."""
    native = json.loads((OUTPUT / "manifest.json").read_text())
    hashes = json.loads((OUTPUT / "analysis/input_integrity.json").read_text())
    if native["status"] != "completed":
        raise ValueError("M04 needs a completed approved native reference")
    for name, expected in hashes.items():
        if digest(OUTPUT / "inputs" / name) != expected:
            raise ValueError(f"Approved M04 input changed: {name}")
    config = json.loads((OUTPUT / "inputs/config.json").read_text())
    with np.load(OUTPUT / "inputs/joint_commands.npz") as data:
        commands = {name: data[name].copy() for name in data.files}
    with np.load(OUTPUT / "inputs/cloth_mesh.npz") as data:
        mesh = {name: data[name].copy() for name in data.files}
    return dict(
        config=config,
        commands=commands,
        mesh=mesh,
        native=native,
        hashes=hashes,
        names=commands["joint_names"].tolist(),
    )


def shape_damping(stiffness: float) -> float:
    """Match the approved reference's effective shape damping coefficient."""
    return 0.5 * (0.05 + 1e-6) * (3e5 + stiffness) - 0.05


def finger_schedule(config: dict[str, Any]) -> tuple[tuple[float, float], ...]:
    """Reproduce the approved 60 Hz step-keyed quintic friction release."""
    c = config["contact"]
    ends = {name: end for end, name in config["motion"]["phases"]}
    start, ramp = ends["hold folded hem"], c["release_ramp_s"]
    if not 0 < ramp <= ends["release"] - start:
        raise ValueError("Release ramp must fit inside the finger opening")
    fractions = np.linspace(0, 1, max(2, round(ramp / config["solver"]["dt"]) + 1))
    return (
        (0.0, c["open_mu"]),
        (ends["descend"], c["grasp_mu"]),
        *(
            (
                start + float(u) * ramp,
                float(
                    c["grasp_mu"]
                    + (c["release_mu"] - c["grasp_mu"])
                    * u**3
                    * (10 - 15 * u + 6 * u * u)
                ),
            )
            for u in fractions
        ),
    )


def texture_uv(vertices: np.ndarray) -> np.ndarray:
    """Use the same local planar FabricGray coordinates as the native preview."""
    uv = vertices[:, :2] - vertices[:, :2].min(0)
    uv /= np.maximum(uv.max(0), 1e-8)
    uv[:, 1] = 1 - uv[:, 1]
    return uv * 2


def link_matrices(poses: np.ndarray) -> np.ndarray:
    """Convert public xyz+xyzw link vectors to world matrices once."""
    result = np.broadcast_to(np.eye(4), (len(poses), 4, 4)).copy()
    result[:, :3, :3] = Rotation.from_quat(poses[:, 3:]).as_matrix()
    result[:, :3, 3] = poses[:, :3]
    return result


def audit_cloth(
    inputs: dict[str, Any], positions: np.ndarray, inverse_mass: np.ndarray
) -> dict[str, Any]:
    """Require unchanged topology, node order, mass and no pinned particles."""
    expected = inputs["mesh"]["initial_positions"]
    error = float(np.abs(positions - expected).max())
    with np.load(OUTPUT / "preview/initial_model.npz") as data:
        native_inverse = data["inverse_mass"]
    if positions.shape != (6436, 3) or error > 2e-6 or np.any(inverse_mass <= 0):
        raise ValueError(f"M04 cloth initialization mismatch: max error {error}")
    if not np.allclose(inverse_mass, native_inverse, rtol=2e-4, atol=0):
        raise ValueError("M04 initialized particle masses differ from native")
    return dict(
        nodes=len(positions),
        triangles=len(inputs["mesh"]["triangles"]),
        initial_max_position_error_m=error,
        pinned_nodes=0,
        mass_kg=float((1 / inverse_mass).sum()),
        max_relative_inverse_mass_error=float(
            np.max(np.abs(inverse_mass / native_inverse - 1))
        ),
    )


def record_preview(
    platform: str,
    inputs: dict[str, Any],
    output: Path,
    step: Callable[[], None],
    readings: Callable[[], dict[str, np.ndarray]],
    capture: Callable[[], np.ndarray] | None,
    diagnostics: Callable[[], dict[str, Any]],
    audit: dict[str, Any],
    link_names: list[str],
    *,
    until: float | None = None,
) -> None:
    """Record actual physical states and 30 fps media without timing samples.

    Args:
        platform: Platform name.
        inputs: Verified approved native inputs.
        output: Platform preview directory, separate from the native preview.
        step: Advance one outer physics step.
        readings: Read detached physical state without advancing it.
        capture: Render the common main camera, or omit video.
        diagnostics: Read the effective solver settings.
        audit: Initialized cloth and material audit.
        link_names: Actual recorded link order.
        until: Optional scratch diagnostic end time.
    """
    output.mkdir(parents=True, exist_ok=True)
    commands, cfg = inputs["commands"], inputs["config"]
    last = len(commands["time"]) - 1
    if until is not None:
        last = min(last, round(until / cfg["solver"]["dt"]))
    manifest = dict(
        task="M04",
        platform=platform,
        status="recording",
        scope="scene preview only; no performance measurements",
        native_scene_user_approved=True,
        user_scene_approved=False,
        performance_measurements_started=False,
        input_sha256=inputs["native"]["input_sha256"],
        config=cfg,
        initial_model=audit,
        camera=CAMERA,
        joint_state_order=inputs["names"],
        link_pose_order=link_names,
        executable=sys.executable,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
    )
    write_json(output / "manifest.json", manifest)
    rows: dict[str, list[np.ndarray]] = {}
    writer = None
    video_frames = 0
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
        for frame in range(last + 1):
            if frame:
                step()
            sample = readings()
            t = float(sample["time"])
            if abs(t - commands["time"][frame]) > 1e-7:
                raise RuntimeError(f"Actual simulation clock mismatch: {t}")
            if any(not np.isfinite(v).all() for v in sample.values()):
                raise RuntimeError(f"Nonfinite state at {t}")
            if not np.allclose(
                sample["qpos"], commands["targets"][frame], atol=2e-5, rtol=0
            ):
                raise RuntimeError(f"Named joint replay mismatch at {t}")
            if (
                np.linalg.norm(
                    sample["tcp"] - commands["tcp_baked"][frame], axis=1
                ).max()
                > 2e-5
            ):
                raise RuntimeError(f"Actual TCP mismatch at {t}")
            for key, value in sample.items():
                rows.setdefault(key, []).append(np.array(value, copy=True))
            if (
                capture is not None
                and frame % 2 == 0
                and t >= cfg["motion"]["pre_roll_s"]
            ):
                image = Image.fromarray(capture())
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, image.width, 42), fill="black")
                draw.text(
                    (12, 10),
                    f"{platform} | M04 T-shirt fold | t={t-4:.2f}s | {phase(t,cfg)}",
                    font=font,
                    fill="white",
                )
                writer.append_data(np.asarray(image))
                video_frames += 1
                if any(abs(t - key) < 1e-7 for key in (4, 14, 18, 26, 28, 32)):
                    image.save(output / f"stage_{t:05.2f}.jpg")
            if frame % 120 == 0:
                print(
                    f"frame={frame}/{last} t={t:.2f} center={sample['particles'].mean(0).round(4)}",
                    flush=True,
                )
        effective = diagnostics()
        if effective["cuda_graph_status"] != "captured":
            raise RuntimeError("M04 CUDA Graph must be captured")
        manifest.update(
            status=(
                "completed"
                if last == len(commands["time"]) - 1
                else "diagnostic_truncated"
            ),
            outer_steps=last,
            actual_simulation_time_s=t,
            all_states_finite=True,
            named_joint_and_tcp_replay_verified=True,
            video_frames=video_frames,
            effective=effective,
        )
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        if writer is not None:
            writer.close()
        if rows:
            np.savez_compressed(
                output / "states.npz", **{k: np.stack(v) for k, v in rows.items()}
            )
        manifest["ended_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(output / "manifest.json", manifest)
