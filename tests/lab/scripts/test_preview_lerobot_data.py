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

"""Tests for the LeRobot dataset preview script."""

from __future__ import annotations

import os

from pathlib import Path

import numpy as np
import pytest

from embodichain.lab.scripts import preview_lerobot_data as preview_module
from embodichain.lab.scripts.preview_lerobot_data import (
    REQUIRED_FEATURES,
    build_episode_preview,
    resolve_dataset_root,
)

FPS = 25
FRAMES_PER_SEGMENT = 2
NUM_SEGMENTS = 3
STATE_DIM = 2
TASK = "Pick up and place the cube three times."
VALIDATION_FAILURE_EXIT_CODE = 1


def _build_samples() -> list[dict]:
    """Create a valid three-segment episode using scalar NumPy fields."""
    samples: list[dict] = []
    for frame_index in range(NUM_SEGMENTS * FRAMES_PER_SEGMENT):
        segment_id = frame_index // FRAMES_PER_SEGMENT
        segment_step = frame_index % FRAMES_PER_SEGMENT
        samples.append(
            {
                "observation.state": np.full(STATE_DIM, frame_index, dtype=np.float32),
                "action": np.full(STATE_DIM, segment_id, dtype=np.float32),
                "subtask_index": np.int64(segment_id),
                "annotation.episode_step": np.int64(frame_index),
                "annotation.segment_id": np.int64(segment_id),
                "annotation.segment_step": np.int64(segment_step),
                "annotation.segment_start": np.int64(segment_step == 0),
                "annotation.segment_end": np.int64(
                    segment_step == FRAMES_PER_SEGMENT - 1
                ),
                "annotation.terminated": np.int64(0),
                "annotation.truncated": np.int64(0),
                "timestamp": np.float32(frame_index / FPS),
                "frame_index": np.int64(frame_index),
                "episode_index": np.int64(0),
                "task_index": np.int64(0),
                "task": TASK,
                "subtask": f"Move cube to target {segment_id + 1}.",
            }
        )
    return samples


def _build_info() -> dict:
    """Create the minimal LeRobot info required by the preview."""
    return {
        "codebase_version": "v3.0",
        "robot_type": "ur5",
        "fps": FPS,
        "total_episodes": 1,
        "total_frames": NUM_SEGMENTS * FRAMES_PER_SEGMENT,
        "features": {key: {} for key in REQUIRED_FEATURES},
    }


def _build_sidecar() -> dict:
    """Create sidecar metadata matching the synthetic samples."""
    return {
        "length": NUM_SEGMENTS * FRAMES_PER_SEGMENT,
        "instruction": TASK,
        "success": True,
        "terminated": False,
        "truncated": False,
        "segments": [
            {
                "segment_id": segment_id,
                "start_step": segment_id * FRAMES_PER_SEGMENT,
                "end_step": (segment_id + 1) * FRAMES_PER_SEGMENT,
                "instruction": f"Move cube to target {segment_id + 1}.",
            }
            for segment_id in range(NUM_SEGMENTS)
        ],
    }


def test_build_episode_preview_accepts_consistent_three_segment_episode(
    tmp_path: Path,
) -> None:
    """A consistent task/subtask/segment layout passes every check."""
    preview = build_episode_preview(
        dataset_root=tmp_path,
        info=_build_info(),
        episode_index=0,
        samples=_build_samples(),
        expected_segments=NUM_SEGMENTS,
        sidecar=_build_sidecar(),
    )

    assert preview.is_valid
    assert preview.task == TASK
    assert [segment.length for segment in preview.segments] == [
        FRAMES_PER_SEGMENT
    ] * NUM_SEGMENTS
    assert [segment.subtask_index for segment in preview.segments] == [0, 1, 2]
    assert preview.sidecar_success is True


def test_build_episode_preview_rejects_invalid_segment_end_marker(
    tmp_path: Path,
) -> None:
    """A missing final marker is reported against the affected segment."""
    samples = _build_samples()
    samples[-1]["annotation.segment_end"] = np.int64(0)

    preview = build_episode_preview(
        dataset_root=tmp_path,
        info=_build_info(),
        episode_index=0,
        samples=samples,
        expected_segments=NUM_SEGMENTS,
        sidecar=_build_sidecar(),
    )

    assert not preview.is_valid
    assert preview.errors == ("Segment 2 has invalid segment_end markers.",)


def test_build_episode_preview_accepts_structurally_valid_failed_episode(
    tmp_path: Path,
) -> None:
    """A failed episode is valid data when terminal annotations match its sidecar."""
    samples = _build_samples()
    samples[-1]["annotation.terminated"] = np.int64(1)
    sidecar = _build_sidecar()
    sidecar["success"] = False
    sidecar["terminated"] = True

    preview = build_episode_preview(
        dataset_root=tmp_path,
        info=_build_info(),
        episode_index=0,
        samples=samples,
        expected_segments=NUM_SEGMENTS,
        sidecar=sidecar,
    )

    assert preview.is_valid
    assert preview.sidecar_success is False


def test_build_episode_preview_rejects_early_terminal_annotation(
    tmp_path: Path,
) -> None:
    """A terminal flag before the saved row's final frame is inconsistent."""
    samples = _build_samples()
    samples[0]["annotation.truncated"] = np.int64(1)

    preview = build_episode_preview(
        dataset_root=tmp_path,
        info=_build_info(),
        episode_index=0,
        samples=samples,
        expected_segments=NUM_SEGMENTS,
        sidecar=_build_sidecar(),
    )

    assert not preview.is_valid
    assert preview.errors == (
        "annotation.truncated may be set only on the final frame.",
    )


def test_build_episode_preview_accepts_legacy_null_segment_instruction(
    tmp_path: Path,
) -> None:
    """Older schema-v2 sidecars inherit their episode-level instruction."""
    samples = _build_samples()[:FRAMES_PER_SEGMENT]
    for frame_index, sample in enumerate(samples):
        sample["subtask"] = TASK
        sample["annotation.segment_end"] = np.int64(
            frame_index == FRAMES_PER_SEGMENT - 1
        )
    sidecar = {
        "schema_version": 2,
        "length": FRAMES_PER_SEGMENT,
        "instruction": TASK,
        "success": True,
        "segments": [
            {
                "segment_id": 0,
                "name": "legacy",
                "start_step": 0,
                "end_step": FRAMES_PER_SEGMENT,
                "instruction": None,
            }
        ],
    }

    preview = build_episode_preview(
        dataset_root=tmp_path,
        info=_build_info(),
        episode_index=0,
        samples=samples,
        expected_segments=1,
        sidecar=sidecar,
    )

    assert preview.is_valid
    assert preview.errors == ()
    assert "using the episode instruction" in preview.warnings[0]


def test_build_episode_preview_rejects_null_instruction_for_modern_segments(
    tmp_path: Path,
) -> None:
    """Compatibility fallback must not hide malformed segmented metadata."""
    samples = _build_samples()
    sidecar = _build_sidecar()
    sidecar["schema_version"] = 2
    sidecar["segments"][0]["name"] = "pick"
    sidecar["segments"][0]["instruction"] = None

    preview = build_episode_preview(
        dataset_root=tmp_path,
        info=_build_info(),
        episode_index=0,
        samples=samples,
        expected_segments=NUM_SEGMENTS,
        sidecar=sidecar,
    )

    assert not preview.is_valid
    assert preview.warnings == ()
    assert any("Sidecar metadata does not match" in error for error in preview.errors)


def test_resolve_dataset_root_selects_newest_child(tmp_path: Path) -> None:
    """--latest resolves the most recently modified child dataset."""
    older = tmp_path / "dataset_000" / "meta"
    newer = tmp_path / "dataset_001" / "meta"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    (older / "info.json").write_text("{}", encoding="utf-8")
    (newer / "info.json").write_text("{}", encoding="utf-8")
    os.utime(older.parent, ns=(1, 1))
    os.utime(newer.parent, ns=(2, 2))

    resolved = resolve_dataset_root(tmp_path, latest=True)

    assert resolved == newer.parent.resolve()


def test_cli_propagates_validation_failure_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The unified CLI must preserve a validation failure for shell callers."""
    monkeypatch.setattr(
        preview_module,
        "main",
        lambda argv: VALIDATION_FAILURE_EXIT_CODE,
    )

    with pytest.raises(SystemExit) as exc_info:
        preview_module.cli(["dataset"])

    assert exc_info.value.code == VALIDATION_FAILURE_EXIT_CODE
