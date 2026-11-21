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
"""Record the user-requested 24x12 C02 episode despite shape failures.

Run with the existing Isaac Python environment; no arguments required.
"""

from __future__ import annotations

from collections.abc import Callable
import datetime as dt
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[7]
sys.path.insert(0, str(REPO))
from scripts.benchmark.tech_reports.physics.experiments.dexuni.c02_replay import (
    _sample,
    sha,
    write_json,
)

__all__ = ["main"]


def _record(
    platform: str,
    inputs: dict[str, Any],
    output: Path,
    step: Callable,
    readings: Callable,
    capture: Callable | None,
    diagnostics: Callable,
    audit: dict[str, Any],
    until: float | None = None,
) -> None:
    """Record physical outer states and an optional full 30 fps video."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    shutil.copy2(Path(__file__), output / "source" / Path(__file__).name)
    shape_failures: list[int] = []
    upright_failures: list[int] = []
    platform += " | VBD 24x12"
    native = inputs["manifest"]
    joints = inputs["joints"]
    total = 3840 if until is None else min(3840, round(until * 60))
    assert total > 0
    preview = output / "preview"
    preview.mkdir()
    states = {}
    writer = None
    manifest = dict(
        task="C02",
        platform=platform,
        status="recording",
        performance_measurements=False,
        reference_user_approved=True,
        reference_manifest_sha256=sha(inputs["reference"] / "manifest.json"),
        joint_mode="kinematic",
        coupling="one_way",
        controller="Original W1BagController with observed bag feedback; reference lift tail; external process provides IK only",
        joint_state_order=native["joint_state_order"],
        link_state_order=native["link_state_order"],
        table_centre=native["table_centre"],
        table_half=native["table_half"],
        initial_model=audit,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        executable=sys.executable,
    )
    trajectory = output / "inputs/controller_trajectory.json"
    if trajectory.exists():
        manifest["controller_trajectory"] = json.loads(trajectory.read_text())
        manifest["controller"] += "; explicit platform-specific robot TCP correction"
    try:
        if capture is not None:
            writer = imageio.get_writer(
                preview / "video.mp4",
                fps=30,
                codec="libx264",
                quality=8,
                macro_block_size=None,
            )
        for frame in range(total + 1):
            if frame:
                step()
            sample = _sample(readings(), inputs, frame)
            assert abs(float(sample["time"]) - frame / 60) < 1e-7, (
                frame,
                sample["time"],
            )
            assert (
                max(
                    abs(
                        sample["qpos"]
                        - sample.get("joint_command", joints["targets"][frame])
                    )
                )
                < 2e-5
            ), (
                frame,
                "joint replay",
            )
            assert all(np.isfinite(v).all() for v in sample.values()), frame
            for key, v in sample.items():
                states.setdefault(key, []).append(np.array(v, copy=True))
            # User explicitly accepts geometry failures for this recording.
            # Keep actual physics/state and record every failing frame.
            time_value = float(sample["time"])
            limit = 0.04 if time_value < 11 - 1e-8 else 0.025
            if float(sample["shape_error"]) > limit:
                shape_failures.append(frame)
            if time_value >= 11 - 1e-8 and float(sample["bag_rotation"][2, 2]) < np.cos(
                np.radians(30)
            ):
                upright_failures.append(frame)
            if writer is not None and frame % 2 == 0:
                img = Image.fromarray(capture().astype(np.uint8))
                draw = ImageDraw.Draw(img)
                draw.rectangle((0, 0, 1280, 36), fill="black")
                draw.text(
                    (12, 10),
                    f"C02 | {platform} | original paper bag + two rigid goods | {frame/60:.2f}s",
                    fill="white",
                )
                writer.append_data(np.asarray(img))
                if frame in (0, 600, 1200, 2100, 2820, 3000, 3360, 3480, 3600, 3840):
                    img.save(preview / f"still_{frame/60:05.2f}s.jpg")
            if frame % 30 == 0:
                print(
                    f'C02 {platform} frame={frame}/{total} loaded={sample["loaded"].tolist()} '
                    f'shape={float(sample["shape_error"]):.5f} '
                    f'tilt_deg={np.degrees(np.arccos(np.clip(sample["bag_rotation"][2, 2], -1, 1))):.2f}',
                    flush=True,
                )
        manifest.update(
            status="completed",
            actual_steps=total,
            actual_time_s=total / 60,
            reference_paper_shape_limits_passed=not shape_failures,
            reference_bag_upright_passed=not upright_failures,
        )
    except BaseException as exc:
        manifest.update(
            status="failed",
            error=repr(exc),
            actual_steps=len(states.get("time", [])) - 1,
        )
        raise
    finally:
        if writer is not None:
            writer.close()
        if "joint_command" in states:
            np.savez_compressed(
                output / "inputs/joint_commands.npz",
                time=np.asarray(states["time"]),
                joint_names=joints["joint_names"],
                targets=np.stack(states["joint_command"]),
            )
        if states:
            np.savez_compressed(
                preview / "states.npz", **{k: np.stack(v) for k, v in states.items()}
            )
        manifest.update(
            recording_policy="User requested complete24x12 video even when shape fails; shape/upright are observed, not fatal; finite/clock/robot checks remain fatal",
            completion_scope="Recording duration only; see measured geometry and task results separately",
            quality_failure_frames=dict(shape=shape_failures, upright=upright_failures),
            effective=diagnostics(),
            ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            input_sha256={
                str(p.relative_to(output / "inputs")): sha(p)
                for p in (output / "inputs").rglob("*")
                if p.is_file()
            },
            source_sha256={
                str(p.relative_to(output / "source")): sha(p)
                for p in (output / "source").rglob("*")
                if p.is_file() and "__pycache__" not in p.parts
            },
        )
        write_json(output / "manifest.json", manifest)


def main() -> None:
    """Use original physics/controller with advisory paper geometry checks."""
    from scripts.benchmark.tech_reports.physics.platforms.dexuni.isaaclab import c02

    root = REPO / "outputs/benchmarks/tech_reports/dexuni/c02_bag_frame_packing"
    output = root / "budget_reduction/i24_s12_video"
    if output.exists():
        raise FileExistsError(output)
    c02.record = _record
    sys.argv = [
        str(c02.__file__),
        "--visualizer",
        "none",
        "--iterations",
        "24",
        "--substeps",
        "12",
        "--trajectory",
        str(root / "platforms/isaaclab/inputs/controller_trajectory.json"),
        "--out",
        str(output),
    ]
    c02.main()


if __name__ == "__main__":
    main()
