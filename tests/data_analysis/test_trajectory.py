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

from __future__ import annotations

from pathlib import Path

import matplotlib.figure
import numpy as np
import pytest

from embodichain.data_analysis.trajectory import (
    phase_options,
    trajectory_comparison,
    trajectory_plot,
)


def _record(
    tmp_path: Path,
    episode_id: str,
    timestamps: np.ndarray,
    path: np.ndarray,
    *,
    robot_id: str = "robot",
    joint_count: int = 2,
    segments: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    artifact = tmp_path / f"{episode_id}.npz"
    np.savez(
        artifact,
        timestamps=timestamps,
        qpos=np.column_stack([np.linspace(0.0, 1.0, len(timestamps))] * joint_count),
        tcp_position=path,
        object_position=np.zeros((len(timestamps), 3), dtype=np.float64),
    )
    return {
        "episode_id": episode_id,
        "robot_id": robot_id,
        "segments": segments or [],
        "artifacts": {"trajectory": str(artifact)},
    }


def test_phase_options_assign_stable_occurrences_and_slice_endpoints_inclusively(
    tmp_path: Path,
) -> None:
    timestamps = np.arange(7, dtype=np.float64)
    path = np.column_stack([timestamps, np.zeros((7, 2))])
    segments = [
        {"name": "move_cube", "start_step": 0, "end_step": 2},
        {"name": "wait", "start_step": 2, "end_step": 3},
        {"name": "move_cube", "start_step": 3, "end_step": 6},
    ]
    record = _record(tmp_path, "repeated", timestamps, path, segments=segments)

    assert phase_options(record) == [
        "whole",
        "move_cube [1]",
        "wait [1]",
        "move_cube [2]",
    ]
    result = trajectory_comparison(record, phase="move_cube [2]")
    assert result["summary"]["primary"]["sample_count"] == 4
    assert result["summary"]["primary"]["duration_s"] == pytest.approx(3.0)
    assert result["curves"]["primary"]["tcp_position"][0] == [3.0, 0.0, 0.0]
    assert result["curves"]["primary"]["tcp_position"][-1] == [6.0, 0.0, 0.0]


def test_retimed_identical_path_has_zero_geometry_difference_and_different_speed(
    tmp_path: Path,
) -> None:
    fast_time = np.array([5.0, 5.5, 6.0])
    slow_time = np.array([20.0, 21.0, 22.0])
    path = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
    fast = _record(tmp_path, "fast", fast_time, path)
    slow = _record(tmp_path, "slow", slow_time, path)

    result = trajectory_comparison(fast, slow)
    assert result["curves"]["normalized_time"][0] == 0.0
    assert result["curves"]["normalized_time"][-1] == 1.0
    assert result["summary"]["geometric_distance_m"] == pytest.approx(0.0, abs=1e-12)
    assert result["summary"]["primary"]["duration_s"] == pytest.approx(1.0)
    assert result["summary"]["comparison"]["duration_s"] == pytest.approx(2.0)
    assert result["summary"]["primary"]["mean_tcp_speed_m_s"] == pytest.approx(1.0)
    assert result["summary"]["comparison"]["mean_tcp_speed_m_s"] == pytest.approx(0.5)


def test_bent_path_has_nonzero_arc_length_resampled_distance(tmp_path: Path) -> None:
    timestamps = np.array([0.0, 1.0, 2.0])
    straight = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
    bent = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [1.0, 0.0, 0.0]])
    result = trajectory_comparison(
        _record(tmp_path, "straight", timestamps, straight),
        _record(tmp_path, "bent", timestamps, bent),
    )
    assert result["summary"]["geometric_distance_m"] > 0.1


def test_missing_segments_only_support_whole_and_unmatched_phase_is_rejected(
    tmp_path: Path,
) -> None:
    timestamps = np.array([0.0, 1.0])
    path = np.zeros((2, 3))
    record = _record(tmp_path, "no-segments", timestamps, path)
    assert phase_options(record) == ["whole"]
    with pytest.raises(ValueError, match="not available"):
        trajectory_comparison(record, phase="move_cube [1]")


def test_phase_must_match_name_and_occurrence_across_records(tmp_path: Path) -> None:
    timestamps = np.arange(4, dtype=np.float64)
    path = np.zeros((4, 3))
    segments = [{"name": "move_cube", "start_step": 0, "end_step": 3}]
    primary = _record(tmp_path, "primary", timestamps, path, segments=segments)
    comparison = _record(tmp_path, "comparison", timestamps, path)
    with pytest.raises(ValueError, match=r"comparison.*move_cube \[1\]"):
        trajectory_comparison(primary, comparison, phase="move_cube [1]")


@pytest.mark.parametrize(
    ("robot_id", "joint_count", "message"),
    [("other", 2, "robot"), ("robot", 3, "joint dimension")],
)
def test_comparison_rejects_incompatible_joint_contracts(
    tmp_path: Path, robot_id: str, joint_count: int, message: str
) -> None:
    timestamps = np.array([0.0, 1.0])
    path = np.zeros((2, 3))
    primary = _record(tmp_path, "primary", timestamps, path)
    other = _record(
        tmp_path, "other", timestamps, path, robot_id=robot_id, joint_count=joint_count
    )
    with pytest.raises(ValueError, match=message):
        trajectory_comparison(primary, other)


def test_invalid_artifact_and_segment_bounds_have_readable_errors(
    tmp_path: Path,
) -> None:
    missing = {
        "episode_id": "missing",
        "robot_id": "robot",
        "segments": [],
        "artifacts": {"trajectory": "missing.npz"},
    }
    with pytest.raises(FileNotFoundError, match="missing.npz"):
        trajectory_comparison(missing, base_dir=tmp_path)

    timestamps = np.array([0.0, 1.0])
    path = np.zeros((2, 3))
    invalid = _record(
        tmp_path,
        "invalid-bounds",
        timestamps,
        path,
        segments=[{"name": "move", "start_step": 0, "end_step": 2}],
    )
    with pytest.raises(ValueError, match="outside trajectory"):
        trajectory_comparison(invalid, phase="move [1]")


def test_plot_returns_figure_without_pyplot_state(tmp_path: Path) -> None:
    timestamps = np.array([0.0, 1.0, 2.0])
    path = np.column_stack([timestamps, np.zeros((3, 2))])
    result = trajectory_comparison(_record(tmp_path, "plot", timestamps, path))
    figure = trajectory_plot(result)
    assert isinstance(figure, matplotlib.figure.Figure)
    assert len(figure.axes) == 3
