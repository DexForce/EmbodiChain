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

"""Portable, simulator-independent trajectory artifacts and physical evidence."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["write_recording", "load_trajectory", "physical_metrics", "json_value"]


def json_value(value: Any) -> Any:
    """Convert protocol dataclasses and arrays to strict JSON values."""
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def _validate(arrays: dict[str, np.ndarray]) -> None:
    required = ("timestamps", "qpos", "tcp_position", "object_position")
    if any(k not in arrays for k in required):
        raise ValueError(f"Trajectory requires {required}.")
    t = arrays["timestamps"]
    if t.ndim != 1 or len(t) < 2 or not np.all(np.diff(t) > 0):
        raise ValueError(
            "Timestamps must be a strictly increasing vector with at least two samples."
        )
    for key, value in arrays.items():
        if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
            raise ValueError(f"{key}: nonnumeric or nonfinite samples.")
        expected = len(t) - 1 if key == "actions" else len(t)
        if value.ndim == 0 or len(value) != expected:
            raise ValueError(f"{key}: samples must align to timestamps.")
    if arrays["qpos"].ndim != 2 or arrays["qpos"].shape[1] < 1:
        raise ValueError("qpos must have shape (T, D).")
    for key in ("tcp_position", "object_position"):
        if arrays[key].shape != (len(t), 3):
            raise ValueError(f"{key} must have shape (T, 3).")
    if "scene_positions" in arrays:
        n = arrays["scene_positions"].shape
        if len(n) != 3 or n[2] != 3:
            raise ValueError("scene_positions must have shape (T, N, 3).")
        if arrays.get("scene_wxyz", np.empty(0)).shape != (len(t), n[1], 4):
            raise ValueError("scene_wxyz must align with scene positions.")
        if arrays.get("scene_visible", np.empty(0)).shape != (len(t), n[1]):
            raise ValueError("scene_visible must align with scene positions.")


def load_trajectory(path: str | Path) -> dict[str, np.ndarray]:
    """Read validated numeric samples with pickle explicitly disabled."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    _validate(arrays)
    return arrays


def write_recording(
    directory: str | Path,
    arrays: dict[str, np.ndarray],
    scene: dict[str, Any],
    *,
    camera: np.ndarray | None = None,
) -> dict[str, str]:
    """Publish a complete episode directory atomically without overwriting data.

    Camera frames, when present, share the trajectory timeline and use uint8 RGB.
    A failed write leaves no visible episode directory.
    """
    _validate(arrays)
    if camera is not None and (
        camera.dtype != np.uint8
        or camera.ndim != 4
        or camera.shape[0] != len(arrays["timestamps"])
        or camera.shape[-1] != 3
    ):
        raise ValueError(
            "Camera frames must be uint8 (T, H, W, 3), aligned to timestamps."
        )
    directory = Path(directory).resolve()
    if directory.exists():
        raise FileExistsError(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".recording-", dir=directory.parent))
    names = {"trajectory": "trajectory.npz", "scene": "scene.json"}
    try:
        np.savez_compressed(temporary / names["trajectory"], **arrays)
        (temporary / names["scene"]).write_text(
            json.dumps(json_value(scene), allow_nan=False), encoding="utf-8"
        )
        if camera is not None:
            names["camera"] = "camera.npz"
            np.savez_compressed(
                temporary / names["camera"],
                frames=camera,
                timestamps=arrays["timestamps"],
            )
        os.rename(temporary, directory)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {key: str(directory / name) for key, name in names.items()}


def physical_metrics(
    arrays: dict[str, np.ndarray],
    *,
    target_xy: tuple[float, float],
    lift_threshold: float = 0.05,
    placement_tolerance: float = 0.06,
) -> dict[str, Any]:
    """Measure lift and final placement; this is a task-specific geometric check.

    This does not certify contact stability or every intermediate pick/place.
    """
    _validate(arrays)
    positions = arrays["object_position"]
    lift = float(positions[:, 2].max() - positions[0, 2])
    error = float(np.linalg.norm(positions[-1, :2] - target_xy))
    settled = bool(abs(positions[-1, 2] - positions[0, 2]) < 0.03)
    path = float(np.linalg.norm(np.diff(arrays["tcp_position"], axis=0), axis=1).sum())
    return {
        "lift_height_m": lift,
        "final_xy_error_m": error,
        "physical_success": bool(
            lift >= lift_threshold and error <= placement_tolerance and settled
        ),
        "physical_check": "lift >= 0.05m; final XY error <= 0.06m; final height within 0.03m of initial",
        "tcp_path_length_m": path,
        "duration_s": float(arrays["timestamps"][-1] - arrays["timestamps"][0]),
    }
