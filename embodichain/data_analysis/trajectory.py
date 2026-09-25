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

"""Offline phase-aligned trajectory comparison and plotting."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from .recording import load_trajectory

__all__ = ["phase_options", "trajectory_comparison", "trajectory_plot"]

_CURVE_SAMPLES = 101
_GEOMETRY_SAMPLES = 101


def _segments(record: Mapping[str, Any]) -> list[tuple[str, str, int, int]]:
    raw_segments = record.get("segments", [])
    if raw_segments is None:
        return []
    if not isinstance(raw_segments, list):
        raise ValueError("Record segments must be a list.")
    occurrences: defaultdict[str, int] = defaultdict(int)
    result = []
    for index, segment in enumerate(raw_segments):
        if not isinstance(segment, Mapping):
            raise ValueError(f"Segment {index} must be a mapping.")
        name = segment.get("name")
        start = segment.get("start_step")
        end = segment.get("end_step")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Segment {index} requires a non-empty name.")
        if type(start) is not int or type(end) is not int:
            raise ValueError(
                f"Segment {index} requires integer start_step and end_step."
            )
        occurrences[name] += 1
        key = f"{name} [{occurrences[name]}]"
        result.append((key, name, start, end))
    return result


def phase_options(record: Mapping[str, Any]) -> list[str]:
    """Return stable whole-record and segment name-occurrence phase keys.

    Repeated segment names are keyed by their one-based occurrence in record
    order, for example ``move_cube [1]`` and ``move_cube [2]``.
    """
    return ["whole", *(key for key, _name, _start, _end in _segments(record))]


def _artifact_path(record: Mapping[str, Any], base_dir: str | Path | None) -> Path:
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Record requires an artifacts mapping.")
    value = artifacts.get("trajectory")
    if not isinstance(value, str) or not value:
        raise ValueError("Record requires a trajectory artifact path.")
    path = Path(value)
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    if not path.is_file():
        raise FileNotFoundError(f"Trajectory artifact does not exist: {path}")
    return path


def _phase_bounds(
    record: Mapping[str, Any], phase: str, sample_count: int, *, role: str
) -> tuple[int, int]:
    if phase == "whole":
        return 0, sample_count - 1
    matches = [segment for segment in _segments(record) if segment[0] == phase]
    if not matches:
        raise ValueError(
            f"The {role} record does not provide phase {phase!r}; it is not available."
        )
    _key, _name, start, end = matches[0]
    if start < 0 or end < start or end >= sample_count:
        raise ValueError(
            f"Phase {phase!r} bounds {start}..{end} are outside trajectory samples 0..{sample_count - 1}."
        )
    if end == start:
        raise ValueError(f"Phase {phase!r} must contain at least two state samples.")
    return start, end


def _slice(
    record: Mapping[str, Any], phase: str, base_dir: str | Path | None, *, role: str
) -> dict[str, np.ndarray]:
    arrays = load_trajectory(_artifact_path(record, base_dir))
    start, end = _phase_bounds(record, phase, len(arrays["timestamps"]), role=role)
    return {
        "timestamps": arrays["timestamps"][start : end + 1],
        "qpos": arrays["qpos"][start : end + 1],
        "tcp_position": arrays["tcp_position"][start : end + 1],
    }


def _normalized_curves(
    arrays: Mapping[str, np.ndarray], target: np.ndarray
) -> dict[str, Any]:
    timestamps = arrays["timestamps"]
    duration = float(timestamps[-1] - timestamps[0])
    source = (timestamps - timestamps[0]) / duration
    tcp = arrays["tcp_position"]
    qpos = arrays["qpos"]
    interval_speed = np.linalg.norm(np.diff(tcp, axis=0), axis=1) / np.diff(timestamps)
    speed_time = (source[:-1] + source[1:]) * 0.5
    return {
        "tcp_position": np.column_stack(
            [np.interp(target, source, tcp[:, axis]) for axis in range(3)]
        ).tolist(),
        "tcp_speed_m_s": np.interp(target, speed_time, interval_speed).tolist(),
        "qpos": np.column_stack(
            [
                np.interp(target, source, qpos[:, joint])
                for joint in range(qpos.shape[1])
            ]
        ).tolist(),
    }


def _summary(arrays: Mapping[str, np.ndarray]) -> dict[str, int | float]:
    timestamps = arrays["timestamps"]
    tcp = arrays["tcp_position"]
    duration = float(timestamps[-1] - timestamps[0])
    lengths = np.linalg.norm(np.diff(tcp, axis=0), axis=1)
    speeds = lengths / np.diff(timestamps)
    path_length = float(lengths.sum())
    return {
        "sample_count": len(timestamps),
        "duration_s": duration,
        "tcp_path_length_m": path_length,
        "mean_tcp_speed_m_s": path_length / duration,
        "peak_tcp_speed_m_s": float(speeds.max()),
        "joint_count": int(arrays["qpos"].shape[1]),
    }


def _arc_resample(path: np.ndarray, samples: int = _GEOMETRY_SAMPLES) -> np.ndarray:
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    if cumulative[-1] == 0.0:
        return np.repeat(path[:1], samples, axis=0)
    cumulative /= cumulative[-1]
    target = np.linspace(0.0, 1.0, samples)
    return np.column_stack(
        [np.interp(target, cumulative, path[:, axis]) for axis in range(3)]
    )


def _geometric_distance(primary: np.ndarray, comparison: np.ndarray) -> float:
    primary_resampled = _arc_resample(primary)
    comparison_resampled = _arc_resample(comparison)
    return float(
        np.linalg.norm(primary_resampled - comparison_resampled, axis=1).mean()
    )


def trajectory_comparison(
    record: Mapping[str, Any],
    other: Mapping[str, Any] | None = None,
    *,
    phase: str = "whole",
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build JSON-friendly normalized curves and physical trajectory summaries.

    Segment ``start_step..end_step`` bounds include both endpoint states.
    Speeds and physical summaries use original timestamps. Only display curves
    are interpolated to normalized phase time. When comparing two records,
    the requested name-occurrence phase must exist in both.

    Args:
        record: Primary catalog episode record.
        other: Optional comparison episode record.
        phase: ``whole`` or a key returned by :func:`phase_options`.
        base_dir: Optional root for relative artifact paths.

    Returns:
        Plain mappings, lists and scalars suitable for JSON and UI consumers.

    Raises:
        ValueError: If phase metadata or robot/joint contracts are incompatible.
        FileNotFoundError: If a trajectory artifact is absent.
    """
    if not isinstance(phase, str) or not phase:
        raise ValueError("phase must be a non-empty string.")
    primary = _slice(record, phase, base_dir, role="primary")
    comparison = None
    if other is not None:
        if record.get("robot_id") != other.get("robot_id"):
            raise ValueError(
                f"Cannot compare joints for different robot IDs: {record.get('robot_id')!r} and {other.get('robot_id')!r}."
            )
        comparison = _slice(other, phase, base_dir, role="comparison")
        if primary["qpos"].shape[1] != comparison["qpos"].shape[1]:
            raise ValueError(
                "Cannot compare trajectories with different joint dimensions: "
                f"{primary['qpos'].shape[1]} and {comparison['qpos'].shape[1]}."
            )

    normalized_time = np.linspace(0.0, 1.0, _CURVE_SAMPLES)
    curves: dict[str, Any] = {
        "normalized_time": normalized_time.tolist(),
        "primary": _normalized_curves(primary, normalized_time),
        "comparison": None,
    }
    summary: dict[str, Any] = {
        "primary": _summary(primary),
        "comparison": None,
        "geometric_distance_m": None,
    }
    if comparison is not None:
        curves["comparison"] = _normalized_curves(comparison, normalized_time)
        summary["comparison"] = _summary(comparison)
        summary["geometric_distance_m"] = _geometric_distance(
            primary["tcp_position"], comparison["tcp_position"]
        )
    return {
        "phase": phase,
        "primary_episode_id": record.get("episode_id"),
        "comparison_episode_id": None if other is None else other.get("episode_id"),
        "curves": curves,
        "summary": summary,
    }


def trajectory_plot(result: Mapping[str, Any]) -> Figure:
    """Render normalized TCP position, speed and joint curves into a Figure."""
    curves = result["curves"]
    normalized_time = np.asarray(curves["normalized_time"], dtype=np.float64)
    figure = Figure(figsize=(9.0, 8.0), constrained_layout=True)
    position_axis, speed_axis, joint_axis = figure.subplots(3, 1, sharex=True)
    for key, label, style in (
        ("primary", "Primary", "-"),
        ("comparison", "Comparison", "--"),
    ):
        curve = curves.get(key)
        if curve is None:
            continue
        tcp = np.asarray(curve["tcp_position"])
        for axis, coordinate in enumerate("xyz"):
            position_axis.plot(
                normalized_time,
                tcp[:, axis],
                linestyle=style,
                label=f"{label} {coordinate}",
            )
        speed_axis.plot(
            normalized_time,
            curve["tcp_speed_m_s"],
            linestyle=style,
            label=label,
        )
        qpos = np.asarray(curve["qpos"])
        for joint in range(qpos.shape[1]):
            joint_axis.plot(
                normalized_time,
                qpos[:, joint],
                linestyle=style,
                label=f"{label} q{joint}",
            )
    position_axis.set_ylabel("TCP position (m)")
    speed_axis.set_ylabel("TCP speed (m/s)")
    joint_axis.set_ylabel("Joint position")
    joint_axis.set_xlabel("Normalized phase time")
    for axis in (position_axis, speed_axis, joint_axis):
        axis.grid(alpha=0.25)
        axis.legend(fontsize="small", ncol=3)
    return figure
