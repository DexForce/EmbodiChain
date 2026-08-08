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

"""Preview and validate an EmbodiChain LeRobot dataset episode."""

from __future__ import annotations

import argparse
import json
import sys

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

__all__ = [
    "EpisodePreview",
    "SegmentPreview",
    "build_episode_preview",
    "cli",
    "inspect_dataset",
    "main",
    "resolve_dataset_root",
]

REQUIRED_FEATURES = {
    "observation.state",
    "action",
    "subtask_index",
    "annotation.episode_step",
    "annotation.segment_id",
    "annotation.segment_step",
    "annotation.segment_start",
    "annotation.segment_end",
    "timestamp",
    "frame_index",
    "episode_index",
    "task_index",
}


@dataclass(frozen=True)
class SegmentPreview:
    """Summary of one contiguous demonstration segment."""

    segment_id: int
    start_frame: int
    end_frame: int
    subtask_index: int
    description: str

    @property
    def length(self) -> int:
        """Return the number of frames in this half-open segment range."""
        return self.end_frame - self.start_frame


@dataclass(frozen=True)
class EpisodePreview:
    """Human-readable episode summary and validation result."""

    dataset_root: Path
    episode_index: int
    codebase_version: str
    robot_type: str
    fps: int
    total_episodes: int
    total_frames: int
    episode_frames: int
    task: str
    state_shape: tuple[int, ...]
    action_shape: tuple[int, ...]
    state_range: tuple[float, float]
    action_range: tuple[float, float]
    segments: tuple[SegmentPreview, ...]
    sidecar_success: bool | None
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def duration_seconds(self) -> float:
        """Return episode duration derived from frame count and FPS."""
        return self.episode_frames / self.fps if self.fps > 0 else 0.0

    @property
    def is_valid(self) -> bool:
        """Return whether all structural checks passed."""
        return not self.errors


def _scalar(value: Any) -> Any:
    """Convert a scalar tensor or array to its Python value."""
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected one scalar value, got shape {array.shape}.")
    return array.reshape(-1)[0].item()


def _feature_matrix(samples: Sequence[dict[str, Any]], key: str) -> np.ndarray:
    """Stack one numeric feature across samples as a NumPy array."""
    return np.stack([np.asarray(sample[key]) for sample in samples])


def _segment_ranges(segment_ids: Sequence[int]) -> list[tuple[int, int, int]]:
    """Return ``(segment_id, start, end)`` for contiguous ID runs."""
    if not segment_ids:
        return []

    ranges: list[tuple[int, int, int]] = []
    start = 0
    active_id = segment_ids[0]
    for position, segment_id in enumerate(segment_ids[1:], start=1):
        if segment_id == active_id:
            continue
        ranges.append((active_id, start, position))
        active_id = segment_id
        start = position
    ranges.append((active_id, start, len(segment_ids)))
    return ranges


def _read_episode_sidecar(
    dataset_root: Path, episode_index: int
) -> dict[str, Any] | None:
    """Read one matching EmbodiChain episode-sidecar record when available."""
    sidecar_path = dataset_root / "meta" / "embodichain_episodes.jsonl"
    if not sidecar_path.exists():
        return None

    for line in sidecar_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        record_index = record.get("lerobot_episode_index", record.get("episode_index"))
        if int(record_index) == episode_index:
            return record
    return None


def build_episode_preview(
    *,
    dataset_root: Path,
    info: dict[str, Any],
    episode_index: int,
    samples: Sequence[dict[str, Any]],
    expected_segments: int | None = None,
    sidecar: dict[str, Any] | None = None,
) -> EpisodePreview:
    """Build and validate a preview from already loaded LeRobot samples.

    Args:
        dataset_root: Dataset directory containing ``meta/info.json``.
        info: Parsed LeRobot dataset info.
        episode_index: Episode selected for inspection.
        samples: Official ``LeRobotDataset`` samples for that episode.
        expected_segments: Optional exact segment count requirement.
        sidecar: Optional matching EmbodiChain episode metadata.

    Returns:
        A structured preview containing summaries, errors, and warnings.
    """
    errors: list[str] = []
    warnings: list[str] = []
    features = set(info.get("features", {}))
    missing_features = sorted(REQUIRED_FEATURES - features)
    if missing_features:
        errors.append(f"Missing required features: {', '.join(missing_features)}")

    if not samples:
        errors.append(f"Episode {episode_index} contains no frames.")
        return EpisodePreview(
            dataset_root=dataset_root,
            episode_index=episode_index,
            codebase_version=str(info.get("codebase_version", "unknown")),
            robot_type=str(info.get("robot_type", "unknown")),
            fps=int(info.get("fps", 0)),
            total_episodes=int(info.get("total_episodes", 0)),
            total_frames=int(info.get("total_frames", 0)),
            episode_frames=0,
            task="",
            state_shape=(),
            action_shape=(),
            state_range=(float("nan"), float("nan")),
            action_range=(float("nan"), float("nan")),
            segments=(),
            sidecar_success=None,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    frame_indices = [int(_scalar(sample["frame_index"])) for sample in samples]
    expected_frame_indices = list(range(len(samples)))
    if frame_indices != expected_frame_indices:
        errors.append("frame_index is not contiguous from zero within the episode.")

    episode_indices = {int(_scalar(sample["episode_index"])) for sample in samples}
    if episode_indices != {episode_index}:
        errors.append(
            f"Expected only episode_index={episode_index}, got {sorted(episode_indices)}."
        )

    episode_steps = [
        int(_scalar(sample["annotation.episode_step"])) for sample in samples
    ]
    if episode_steps != expected_frame_indices:
        errors.append("annotation.episode_step does not match frame_index.")

    fps = int(info.get("fps", 0))
    timestamps = np.asarray(
        [float(_scalar(sample["timestamp"])) for sample in samples],
        dtype=np.float64,
    )
    if fps <= 0:
        errors.append(f"Dataset FPS must be positive, got {fps}.")
    elif not np.allclose(
        timestamps,
        np.arange(len(samples), dtype=np.float64) / fps,
        atol=1e-5,
        rtol=0.0,
    ):
        errors.append("timestamp does not match frame_index / fps.")

    tasks = {str(sample.get("task", "")) for sample in samples}
    if len(tasks) != 1:
        errors.append(f"Episode must have one constant task, got {len(tasks)} values.")
    task = next(iter(tasks), "")

    segment_ids = [int(_scalar(sample["annotation.segment_id"])) for sample in samples]
    ranges = _segment_ranges(segment_ids)
    range_ids = [segment_id for segment_id, _, _ in ranges]
    if len(range_ids) != len(set(range_ids)):
        errors.append("A segment_id appears in multiple non-contiguous ranges.")
    if range_ids != sorted(range_ids):
        errors.append(f"segment_id order is not monotonic: {range_ids}.")
    if expected_segments is not None and len(ranges) != expected_segments:
        errors.append(f"Expected {expected_segments} segments, found {len(ranges)}.")

    segment_previews: list[SegmentPreview] = []
    subtask_descriptions: dict[int, str] = {}
    for segment_id, start, end in ranges:
        group = samples[start:end]
        segment_steps = [
            int(_scalar(sample["annotation.segment_step"])) for sample in group
        ]
        if segment_steps != list(range(end - start)):
            errors.append(
                f"Segment {segment_id} has non-contiguous segment_step values."
            )

        start_flags = [
            int(_scalar(sample["annotation.segment_start"])) for sample in group
        ]
        expected_start_flags = [1, *([0] * (len(group) - 1))]
        if start_flags != expected_start_flags:
            errors.append(f"Segment {segment_id} has invalid segment_start markers.")

        end_flags = [int(_scalar(sample["annotation.segment_end"])) for sample in group]
        expected_end_flags = [*([0] * (len(group) - 1)), 1]
        if end_flags != expected_end_flags:
            errors.append(f"Segment {segment_id} has invalid segment_end markers.")

        subtask_indices = {int(_scalar(sample["subtask_index"])) for sample in group}
        if len(subtask_indices) != 1:
            errors.append(
                f"Segment {segment_id} references multiple subtask_index values: "
                f"{sorted(subtask_indices)}."
            )
        subtask_index = min(subtask_indices, default=-1)

        descriptions = {str(sample.get("subtask", "")) for sample in group}
        if "" in descriptions:
            errors.append(
                f"Segment {segment_id} has no resolved subtask description; "
                "check meta/subtasks.parquet."
            )
        if len(descriptions) != 1:
            errors.append(
                f"Segment {segment_id} resolves to multiple descriptions: "
                f"{sorted(descriptions)}."
            )
        description = next(iter(descriptions), "")
        existing_description = subtask_descriptions.setdefault(
            subtask_index, description
        )
        if existing_description != description:
            errors.append(
                f"subtask_index {subtask_index} maps to inconsistent descriptions."
            )

        segment_previews.append(
            SegmentPreview(
                segment_id=segment_id,
                start_frame=frame_indices[start],
                end_frame=frame_indices[end - 1] + 1,
                subtask_index=subtask_index,
                description=description,
            )
        )

    sidecar_success: bool | None = None
    if sidecar is None:
        warnings.append("No matching meta/embodichain_episodes.jsonl record found.")
    else:
        sidecar_success = bool(sidecar.get("success", False))
        if int(sidecar.get("length", -1)) != len(samples):
            errors.append("Sidecar episode length does not match LeRobot frames.")
        if str(sidecar.get("instruction", "")) != task:
            errors.append("Sidecar instruction does not match LeRobot task.")

        sidecar_segments = sidecar.get("segments", [])
        if len(sidecar_segments) != len(segment_previews):
            errors.append("Sidecar segment count does not match frame annotations.")
        for preview, metadata in zip(segment_previews, sidecar_segments, strict=False):
            expected = (
                preview.segment_id,
                preview.start_frame,
                preview.end_frame,
                preview.description,
            )
            actual = (
                int(metadata.get("segment_id", -1)),
                int(metadata.get("start_step", -1)),
                int(metadata.get("end_step", -1)),
                str(metadata.get("instruction", "")),
            )
            if actual != expected:
                errors.append(
                    f"Sidecar metadata does not match segment {preview.segment_id}."
                )

    state = _feature_matrix(samples, "observation.state")
    action = _feature_matrix(samples, "action")
    return EpisodePreview(
        dataset_root=dataset_root,
        episode_index=episode_index,
        codebase_version=str(info.get("codebase_version", "unknown")),
        robot_type=str(info.get("robot_type", "unknown")),
        fps=fps,
        total_episodes=int(info.get("total_episodes", 0)),
        total_frames=int(info.get("total_frames", 0)),
        episode_frames=len(samples),
        task=task,
        state_shape=tuple(state.shape),
        action_shape=tuple(action.shape),
        state_range=(float(np.min(state)), float(np.max(state))),
        action_range=(float(np.min(action)), float(np.max(action))),
        segments=tuple(segment_previews),
        sidecar_success=sidecar_success,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def resolve_dataset_root(path: str | Path, *, latest: bool = False) -> Path:
    """Resolve a dataset root, optionally selecting the newest child dataset."""
    candidate = Path(path).expanduser().resolve()
    if (candidate / "meta" / "info.json").is_file():
        return candidate
    if not latest:
        raise FileNotFoundError(
            f"{candidate} is not a LeRobot dataset root (meta/info.json missing)."
        )
    if not candidate.is_dir():
        raise FileNotFoundError(f"Dataset parent directory does not exist: {candidate}")

    datasets = [
        child
        for child in candidate.iterdir()
        if child.is_dir() and (child / "meta" / "info.json").is_file()
    ]
    if not datasets:
        raise FileNotFoundError(f"No LeRobot datasets found below {candidate}.")
    return max(datasets, key=lambda child: (child.stat().st_mtime_ns, child.name))


def inspect_dataset(
    dataset_root: str | Path,
    *,
    episode_index: int = 0,
    expected_segments: int | None = None,
    latest: bool = False,
) -> EpisodePreview:
    """Load one episode through LeRobot and return its validated preview."""
    root = resolve_dataset_root(dataset_root, latest=latest)
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    total_episodes = int(info.get("total_episodes", 0))
    if episode_index < 0 or episode_index >= total_episodes:
        raise IndexError(
            f"episode_index {episode_index} is outside [0, {total_episodes})."
        )

    dataset = LeRobotDataset(
        repo_id=root.name,
        root=root,
        episodes=[episode_index],
    )
    samples = [dataset[index] for index in range(len(dataset))]
    return build_episode_preview(
        dataset_root=root,
        info=info,
        episode_index=episode_index,
        samples=samples,
        expected_segments=expected_segments,
        sidecar=_read_episode_sidecar(root, episode_index),
    )


def _print_preview(preview: EpisodePreview) -> None:
    """Print a concise terminal representation of a preview."""
    print("LeRobot dataset preview")
    print(f"  Dataset : {preview.dataset_root}")
    print(
        f"  Format  : {preview.codebase_version} | robot={preview.robot_type} "
        f"| fps={preview.fps}"
    )
    print(
        f"  Dataset : {preview.total_episodes} episode(s), "
        f"{preview.total_frames} total frame(s)"
    )
    print(
        f"  Episode : {preview.episode_index} | {preview.episode_frames} frame(s) "
        f"| {preview.duration_seconds:.2f}s"
    )
    print(f"  Task    : {preview.task}")
    print(
        f"  State   : shape={preview.state_shape}, "
        f"range=[{preview.state_range[0]:.5f}, {preview.state_range[1]:.5f}]"
    )
    print(
        f"  Action  : shape={preview.action_shape}, "
        f"range=[{preview.action_range[0]:.5f}, {preview.action_range[1]:.5f}]"
    )
    print("  Segments:")
    for segment in preview.segments:
        print(
            f"    #{segment.segment_id}: frames "
            f"[{segment.start_frame}, {segment.end_frame}) "
            f"({segment.length}), subtask_index={segment.subtask_index}"
        )
        print(f"       {segment.description}")

    if preview.sidecar_success is not None:
        print(f"  Sidecar : success={preview.sidecar_success}")
    for warning in preview.warnings:
        print(f"  [WARN] {warning}")
    for error in preview.errors:
        print(f"  [FAIL] {error}")
    if preview.is_valid:
        print("  [PASS] Dataset structure and segment metadata are consistent.")
    else:
        print(f"  [FAIL] Validation found {len(preview.errors)} error(s).")


def _create_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="embodichain preview_lerobot_data",
        description="Preview and validate an EmbodiChain LeRobot dataset episode.",
    )
    parser.add_argument(
        "dataset_root",
        help="LeRobot dataset root, or a parent directory when --latest is used.",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index to inspect (default: 0).",
    )
    parser.add_argument(
        "--expect-segments",
        type=int,
        default=None,
        help="Fail validation unless the episode has exactly this many segments.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Select the newest dataset directly below dataset_root.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the preview CLI and return a process exit code."""
    args = _create_parser().parse_args(argv)
    try:
        preview = inspect_dataset(
            args.dataset_root,
            episode_index=args.episode,
            expected_segments=args.expect_segments,
            latest=args.latest,
        )
    except (FileNotFoundError, IndexError, OSError, RuntimeError, ValueError) as error:
        print(f"Failed to preview dataset: {error}", file=sys.stderr)
        return 2

    _print_preview(preview)
    return 0 if preview.is_valid else 1


def cli(argv: Sequence[str] | None = None) -> None:
    """Run the preview through the unified ``embodichain`` CLI.

    Args:
        argv: Arguments excluding the command name. Uses ``sys.argv`` when
            omitted.

    Raises:
        SystemExit: If loading fails or dataset validation reports errors.
    """
    exit_code = main(argv)
    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    cli()
