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
"""Verify and record the accepted C01 through a platform's real simulation.

This preview harness records actual state and clock, without performance timing.
Its output layout also supports C01's existing offline physical/FCL analyzers.
"""

from __future__ import annotations

from collections.abc import Callable
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

from scripts.benchmark.tech_reports.physics.experiments.dexuni.c01_common import (
    GOODS,
    GOOD_ARMS,
    OUTPUT,
    PHASES,
    packing_metrics,
    phase,
)

__all__ = ["load_inputs", "freeze_inputs", "record_preview", "sha256", "write_json"]


def sha256(path: Path) -> str:
    """Return the content digest of a recorded input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    """Save recording metadata with explicit NumPy serialization."""

    def convert(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Cannot serialize {type(item).__name__}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=convert) + "\n")


def load_inputs(run: Path = OUTPUT) -> dict[str, Any]:
    """Verify frozen native geometry, command bytes and phase/arm semantics."""
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("C01 needs a complete native reference")
    for name, expected in manifest["input_sha256"].items():
        if sha256(run / "inputs" / name) != expected:
            raise ValueError(f"Frozen C01 input changed: {name}")
    if manifest["scene"]["phases"] != [list(row) for row in PHASES]:
        raise ValueError("C01 phase helper differs from the accepted reference")
    if manifest["scene"]["goods"] != list(GOODS):
        raise ValueError("C01 active goods differ from the reference")
    if manifest["scene"]["good_arms"] != GOOD_ARMS:
        raise ValueError("C01 arm assignment differs from the reference")
    locked = run / "analysis/platform_port/native_reference_sha256.json"
    if locked.exists():
        for name, expected in json.loads(locked.read_text()).items():
            if name.startswith("inputs/") or name == "manifest.json":
                if sha256(run / name) != expected:
                    raise ValueError(f"Accepted C01 reference changed: {name}")
    arrays = {}
    for key, filename in [
        ("joints", "joint_commands.npz"),
        ("mesh", "goods_meshes.npz"),
    ]:
        with np.load(run / "inputs" / filename) as archive:
            arrays[key] = {name: archive[name].copy() for name in archive.files}
    asset = json.loads((run / "inputs/shopping_bag.json").read_text())
    return dict(run=run, manifest=manifest, asset=asset, **arrays)


def freeze_inputs(inputs: dict[str, Any], output: Path, worker: Path) -> None:
    """Save complete input and source payloads before platform initialization."""
    if output.resolve() == inputs["run"].resolve():
        raise ValueError("A platform preview must not overwrite the native reference")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(inputs["run"] / "inputs", output / "inputs")
    source = output / "source"
    source.mkdir()
    shutil.copy2(worker, source / worker.name)
    shutil.copy2(Path(__file__), source / "c01_replay.py")
    shutil.copy2(Path(__file__).with_name("c01_common.py"), source / "c01_common.py")
    shutil.copy2(inputs["run"] / "manifest.json", source / "native_manifest.json")
    shutil.copytree(inputs["run"] / "source/reference", source / "reference")


def record_preview(
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
    """Record actual physics, matched commands and a fixed centered camera."""
    native, joints, mesh = inputs["manifest"], inputs["joints"], inputs["mesh"]
    total = len(joints["time"]) - 1
    if until is not None:
        if until <= 0:
            raise ValueError("Diagnostic duration must be positive")
        total = min(total, round(until * 60))
    manifest = dict(
        task="C01",
        platform="EmbodiChain",
        mode="platform_scene_preview_only",
        status="recording",
        user_scene_approved=False,
        reference_user_approved=True,
        performance_measurements_started=False,
        reference_manifest_sha256=sha256(inputs["run"] / "manifest.json"),
        input_sha256={
            str(p.relative_to(output / "inputs")): sha256(p)
            for p in sorted((output / "inputs").rglob("*"))
            if p.is_file()
        },
        reference_input_sha256=native["input_sha256"],
        execution_source_sha256={
            str(p.relative_to(output / "source")): sha256(p)
            for p in sorted((output / "source").rglob("*"))
            if p.is_file()
        },
        scene=native["scene"],
        camera=native["camera"],
        coupling=native["coupling"],
        joint_mode=native["joint_mode"],
        executable=sys.executable,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        initial_model=audit,
        joint_state_order=joints["joint_names"].tolist(),
        trajectory_checks=native["trajectory_checks"],
    )
    preview = output / "preview"
    preview.mkdir()
    rows: list[dict[str, Any]] = []
    states: dict[str, list[np.ndarray]] = {}
    writer = None
    try:
        if capture is not None:
            import imageio.v2 as imageio
            from PIL import Image, ImageDraw, ImageFont

            font = ImageFont.truetype("DejaVuSans.ttf", 20)
            writer = imageio.get_writer(
                str(preview / "video.mp4"),
                fps=30,
                codec="libx264",
                quality=8,
                macro_block_size=2,
            )
        ends = dict((name, end) for end, name in PHASES)
        stills = {0, 120, total} | {
            round(ends[f"{good} {stage}"] * 60)
            for good in GOODS
            for stage in ("settle", "carry", "release")
        }
        groups = dict(mouth=mesh["bag_mouth_group"], bottom=mesh["bag_bottom_group"])
        for frame in range(total + 1):
            if frame:
                step()
            sample = readings()
            time = float(sample["time"])
            if abs(time - float(joints["time"][frame])) > 1e-7:
                raise RuntimeError(
                    f"C01 actual clock mismatch at frame {frame}: {time}"
                )
            if any(not np.isfinite(value).all() for value in sample.values()):
                raise RuntimeError(f"C01 nonfinite physical state at frame {frame}")
            if not np.allclose(
                sample["qpos"], joints["targets"][frame], atol=2e-5, rtol=0
            ):
                raise RuntimeError(f"C01 named joint replay mismatch at frame {frame}")
            error = np.linalg.norm(sample["tcp"] - joints["tcp_baked"][frame], axis=1)
            if error.max() > 2e-5:
                raise RuntimeError(f"C01 actual TCP replay mismatch at frame {frame}")
            row = packing_metrics(
                sample["bag"],
                {name: sample[name].mean(0) for name in GOODS},
                groups,
            )
            row.update(
                frame=frame,
                time=time,
                phase=phase(time),
                tcp=sample["tcp"],
                tcp_error_m=error,
            )
            rows.append(row)
            sample["target_q"] = joints["targets"][frame]
            for name, value in sample.items():
                states.setdefault(name, []).append(np.array(value, copy=True))
            if capture is not None and (frame % 2 == 0 or frame in stills):
                image = Image.fromarray(capture())
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, image.width, 42), fill="black")
                arm = GOOD_ARMS.get(row["phase"].split()[0])
                label = row["phase"] + (f" ({arm})" if arm else "")
                packed = sum(row[f"{name}_in_bag"] for name in GOODS)
                draw.text(
                    (14, 10),
                    f"EmbodiChain | C01 supermarket packing (one-way) | t={time:5.2f}s | {label} | goods in bag: {packed}/2",
                    font=font,
                    fill="white",
                )
                if frame % 2 == 0:
                    writer.append_data(np.asarray(image))
                if frame in stills:
                    image.save(preview / f"still_{time:05.2f}s.jpg")
            if frame % 120 == 0:
                print(f"frame={frame}/{total} t={time:.2f}", flush=True)
        effective = diagnostics()
        if effective["cuda_graph_status"] != "captured":
            raise RuntimeError("C01 platform preview did not capture CUDA Graph")
        manifest.update(
            status=(
                "completed"
                if total == len(joints["time"]) - 1
                else "diagnostic_truncated"
            ),
            frames=total,
            effective=effective,
        )
    except Exception as error:
        manifest.update(status="failed", error=repr(error), effective=diagnostics())
        raise
    finally:
        if writer is not None:
            writer.close()
        if states:
            np.savez_compressed(
                preview / "states.npz", **{k: np.stack(v) for k, v in states.items()}
            )
        write_json(preview / "metrics.json", rows)
        manifest["ended_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(output / "manifest.json", manifest)
