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

"""Inspect persisted LeRobot data and create provenance-safe split manifests.

The inspector reads files directly and never imports LeRobot or simulation
modules. Numeric frame data is scanned in bounded Arrow batches; decoding
videos is an explicit, optional expense. No operation modifies source data.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Literal

import numpy as np
import pyarrow.parquet as pq

__all__ = [
    "DatasetValidationIssue",
    "DatasetValidationReport",
    "create_split_manifest",
    "main",
    "validate_dataset",
]


@dataclass(frozen=True)
class DatasetValidationIssue:
    """One actionable diagnostic with a file, episode, or frame location.

    Args:
        severity: ``error`` for corrupt data, ``warning`` for limited evidence.
        code: Stable machine-readable diagnostic category.
        location: Dataset-relative file and, when known, row or episode.
        message: Description of the violated invariant.
    """

    severity: Literal["error", "warning"]
    code: str
    location: str
    message: str


@dataclass
class DatasetValidationReport:
    """Result of a file-only dataset scan, including capped issue details.

    Args:
        root: Absolute dataset directory.
        episodes: Number of declared unique episodes.
        frames: Number of frame rows scanned.
        errors: Total error count, including diagnostics beyond the detail cap.
        warnings: Total warning count, including diagnostics beyond the cap.
        issues: Up to 1,000 detailed diagnostics.
    """

    root: str
    episodes: int = 0
    frames: int = 0
    errors: int = 0
    warnings: int = 0
    issues: list[DatasetValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return whether the scan found no errors."""
        return self.errors == 0

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible diagnostics and aggregate counts.

        Returns:
            Report mapping suitable for CLI output or automated gates.
        """
        return {"ok": self.ok, **asdict(self)}

    def raise_for_errors(self) -> None:
        """Raise a concise error when data failed validation.

        Raises:
            ValueError: If the scan found one or more errors.
        """
        if not self.ok:
            first = next(
                (issue for issue in self.issues if issue.severity == "error"), None
            )
            detail = (
                f"{first.location}: {first.message}"
                if first
                else "See error counts (issue detail cap reached)"
            )
            raise ValueError(f"Dataset has {self.errors} errors: {detail}")


def _issue(
    report: DatasetValidationReport,
    code: str,
    location: str,
    message: str,
    *,
    warning: bool = False,
) -> None:
    if warning:
        report.warnings += 1
    else:
        report.errors += 1
    if len(report.issues) < 1000:
        report.issues.append(
            DatasetValidationIssue(
                "warning" if warning else "error", code, location, message
            )
        )


def _integer(value: Any) -> int | None:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return value if type(value) is int else None


def _scalar(value: Any) -> Any:
    return value[0] if isinstance(value, (list, tuple)) and len(value) == 1 else value


def _json_mapping(path: Path, report: DatasetValidationReport) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        return value
    except (OSError, ValueError) as error:
        _issue(report, "metadata", str(path), str(error))
        return {}


def _jsonl_rows(
    path: Path, report: DatasetValidationReport
) -> Iterator[tuple[str, dict[str, Any]]]:
    try:
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                location = f"{path}:{line_number}"
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("Expected a JSON object")
                    yield location, value
                except ValueError as error:
                    _issue(report, "metadata", location, str(error))
    except OSError as error:
        _issue(report, "metadata", str(path), str(error))


def _parquet_rows(
    paths: Sequence[Path], report: DatasetValidationReport, batch_size: int
) -> Iterator[tuple[str, dict[str, Any]]]:
    for path in paths:
        try:
            row_number = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
                for row in batch.to_pylist():
                    yield f"{path}:{row_number}", row
                    row_number += 1
        except (OSError, ValueError, TypeError) as error:
            _issue(report, "parquet", str(path), str(error))


def _episode_rows(
    root: Path, report: DatasetValidationReport, batch_size: int
) -> dict[int, dict[str, Any]]:
    paths = sorted((root / "meta/episodes").rglob("*.parquet"))
    rows = (
        _parquet_rows(paths, report, batch_size)
        if paths
        else _jsonl_rows(root / "meta/episodes.jsonl", report)
    )
    episodes: dict[int, dict[str, Any]] = {}
    for location, row in rows:
        index = _integer(row.get("episode_index"))
        if index is None or index < 0:
            _issue(report, "episode_index", location, "Invalid episode_index")
            continue
        if index in episodes:
            _issue(report, "duplicate_episode", location, f"Duplicate episode {index}")
        episodes[index] = row
    return episodes


def _sidecar_rows(
    root: Path, report: DatasetValidationReport
) -> dict[int, dict[str, Any]]:
    path = root / "meta/embodichain_episodes.jsonl"
    if not path.exists():
        return {}
    sidecars: dict[int, dict[str, Any]] = {}
    identifiers: dict[str, set[str]] = defaultdict(set)
    for location, row in _jsonl_rows(path, report):
        index = _integer(row.get("lerobot_episode_index", row.get("episode_index")))
        if index is None or index < 0:
            _issue(report, "episode_index", location, "Invalid sidecar episode index")
            continue
        if index in sidecars:
            _issue(report, "duplicate_episode", location, f"Duplicate sidecar {index}")
        sidecars[index] = row
        for key in (
            "source_episode_uuid",
            "source_episode_uuids",
            "parent_episode_uuid",
        ):
            try:
                _lineage_values(row, key)
            except ValueError as error:
                _issue(report, "identity", location, str(error))
        try:
            _expansion_lineage_tokens(row)
        except ValueError as error:
            _issue(report, "identity", location, str(error))
        for key in ("episode_uuid", "fragment_id"):
            value = row.get(key)
            if value is None:
                continue
            if not isinstance(value, str) or not value:
                _issue(report, "identity", location, f"{key} must be nonempty text")
            elif value in identifiers[key]:
                _issue(
                    report, "duplicate_identity", location, f"Duplicate {key}: {value}"
                )
            else:
                identifiers[key].add(value)
    return sidecars


def _index_table(
    root: Path,
    key: str,
    required: bool,
    report: DatasetValidationReport,
    batch_size: int,
) -> dict[int, str]:
    path = root / f"meta/{key}s.parquet"
    legacy_path = root / f"meta/{key}s.jsonl"
    if not path.exists() and not legacy_path.exists():
        if required:
            _issue(report, "language_index", str(path), f"Missing {key} table")
        return {}
    rows = (
        _parquet_rows([path], report, batch_size)
        if path.exists()
        else _jsonl_rows(legacy_path, report)
    )
    indices: dict[int, str] = {}
    for position, (location, row) in enumerate(rows):
        index = _integer(row.get(f"{key}_index"))
        if index is None or index < 0 or index in indices:
            _issue(report, "language_index", location, f"Invalid/duplicate {key}_index")
        elif index != position:
            _issue(
                report,
                "language_index",
                location,
                f"{key}_index {index} differs from table position {position}",
            )
        text_columns = [
            column
            for column in row
            if column != f"{key}_index" and isinstance(row[column], str)
        ]
        if not text_columns or not row[text_columns[0]].strip():
            _issue(report, "language_text", location, f"Missing {key} description")
        if index is not None:
            indices[index] = row[text_columns[0]] if text_columns else ""
    return indices


def _segments(
    sidecar: Mapping[str, Any],
    length: int,
    location: str,
    report: DatasetValidationReport,
) -> list[dict[str, Any]]:
    raw = sidecar.get("segments", [])
    if not isinstance(raw, list):
        _issue(report, "segment_span", location, "segments must be a list")
        return []
    valid: list[dict[str, Any]] = []
    identities: set[tuple[int, int, int]] = set()
    for position, segment in enumerate(raw):
        where = f"{location}/segments/{position}"
        if not isinstance(segment, dict):
            _issue(report, "segment_span", where, "Segment must be a mapping")
            continue
        start, end = _integer(segment.get("start_step")), _integer(
            segment.get("end_step")
        )
        if start is None or end is None or not 0 <= start <= end <= length:
            _issue(
                report,
                "segment_span",
                where,
                f"Invalid half-open span [{start}, {end})",
            )
            continue
        identity = tuple(
            _integer(segment.get(key, fallback))
            for key, fallback in (
                ("segment_id", position),
                ("attempt_id", sidecar.get("attempt_id", 0)),
                ("continuity_id", sidecar.get("continuity_id", 0)),
            )
        )
        if any(value is None or value < 0 for value in identity):
            _issue(
                report,
                "segment_identity",
                where,
                "Invalid segment/attempt/continuity id",
            )
            continue
        if identity in identities:
            _issue(
                report,
                "segment_identity",
                where,
                f"Duplicate segment identity {identity}",
            )
        identities.add(identity)
        if end > start:
            valid.append(
                {
                    "attempt_id": sidecar.get("attempt_id", 0),
                    "continuity_id": sidecar.get("continuity_id", 0),
                    **segment,
                }
            )
    valid.sort(key=lambda segment: (segment["start_step"], segment["end_step"]))
    previous_end = 0
    for segment in valid:
        start, end = segment["start_step"], segment["end_step"]
        if start < previous_end:
            _issue(report, "segment_span", location, "Segment spans overlap")
        if start > previous_end:
            _issue(
                report,
                "segment_gap",
                location,
                f"Unannotated frames [{previous_end}, {start})",
                warning=True,
            )
        previous_end = max(previous_end, end)
    return valid


def _action_slices(
    features: Mapping[str, Any], report: DatasetValidationReport
) -> list[tuple[int, int]]:
    action = features.get("action", {})
    info = action.get("info", {})
    if not isinstance(info, dict):
        _issue(
            report,
            "action_contract",
            "meta/info.json/action",
            "Action info must be a mapping",
        )
        return []
    contract = info.get("embodichain.action_contract", {})
    if not isinstance(contract, dict):
        _issue(
            report,
            "action_contract",
            "meta/info.json/action",
            "Action contract must be a mapping",
        )
        return []
    terms = info.get("embodichain.action_terms", contract.get("action_terms", []))
    shape = action.get("shape", [])
    width = shape[0] if isinstance(shape, (list, tuple)) and len(shape) == 1 else None
    slices: list[tuple[int, int]] = []
    previous_stop = 0
    if not isinstance(terms, list):
        _issue(
            report,
            "action_contract",
            "meta/info.json/action",
            "action_terms must be a list",
        )
        return []
    for descriptor in terms:
        if not isinstance(descriptor, Mapping):
            _issue(
                report,
                "action_contract",
                "meta/info.json/action",
                "Invalid action descriptor",
            )
            continue
        span, term = descriptor.get("slice"), descriptor.get("term", {})
        if (
            not isinstance(span, (list, tuple))
            or len(span) != 2
            or not all(type(value) is int for value in span)
        ):
            _issue(
                report,
                "action_contract",
                "meta/info.json/action",
                "Invalid action slice",
            )
            continue
        start, stop = span
        if start != previous_stop or stop <= start or width is None or stop > width:
            _issue(
                report,
                "action_contract",
                "meta/info.json/action",
                f"Invalid action slice {span}",
            )
        previous_stop = stop
        if not isinstance(term, Mapping) or term.get("action_dim") != stop - start:
            _issue(
                report,
                "action_contract",
                "meta/info.json/action",
                "Action term width mismatch",
            )
            continue
        if term.get("representation") == "parallel_gripper":
            slices.append((start, stop))
    if terms and previous_stop != width:
        _issue(
            report,
            "action_contract",
            "meta/info.json/action",
            "Action slices do not cover feature width",
        )
    if contract.get("representation") == "joint_position_velocity":
        qpos, qvel = contract.get("qpos_slice", []), contract.get("qvel_slice", [])
        if not (
            isinstance(qpos, (list, tuple))
            and isinstance(qvel, (list, tuple))
            and len(qpos) == len(qvel) == 2
            and all(type(value) is int for value in (*qpos, *qvel))
            and qpos[0] == 0
            and qpos[1] == qvel[0]
            and qvel[1] == width
            and qpos[1] == qvel[1] - qvel[0]
        ):
            _issue(
                report,
                "action_contract",
                "meta/info.json/action",
                "Invalid qpos/qvel layout",
            )
    return slices


def _numeric_feature(
    value: Any,
    feature: Mapping[str, Any],
    location: str,
    report: DatasetValidationReport,
) -> np.ndarray | None:
    try:
        array = np.asarray(value)
        shape = tuple(feature.get("shape", []))
        if array.shape != shape and not (array.shape == () and shape == (1,)):
            _issue(
                report,
                "feature_shape",
                location,
                f"Expected shape {shape}, got {array.shape}",
            )
        if array.dtype.kind not in "biuf":
            _issue(
                report, "feature_dtype", location, "Feature contains nonnumeric data"
            )
            return None
        if not np.all(np.isfinite(array)):
            _issue(report, "nonfinite", location, "Feature contains NaN or infinity")
        if str(feature.get("dtype", "")).startswith(("int", "uint")) and not np.all(
            array == np.floor(array)
        ):
            _issue(
                report,
                "feature_dtype",
                location,
                "Integer feature contains fractional values",
            )
        return array
    except (TypeError, ValueError) as error:
        _issue(report, "feature_shape", location, str(error))
        return None


def _check_frame_annotations(
    row: Mapping[str, Any],
    frame: int,
    segment: Mapping[str, Any] | None,
    location: str,
    report: DatasetValidationReport,
) -> None:
    expected = {"episode_step": frame}
    if segment is not None:
        expected.update(
            {
                "segment_id": segment.get("segment_id", 0),
                "segment_step": frame - segment["start_step"],
                "segment_start": frame == segment["start_step"],
                "segment_end": frame == segment["end_step"] - 1,
                "segment_attempt_id": segment.get("attempt_id", 0),
                "continuity_id": segment.get("continuity_id", 0),
                "segment_accepted": segment.get("success", True),
            }
        )
        if "segment_id" not in segment:
            expected.pop("segment_id")
    for key, value in expected.items():
        column = f"annotation.{key}"
        if column in row and _scalar(row[column]) != value:
            _issue(
                report,
                "frame_annotation",
                f"{location}/{column}",
                f"Expected {value!r}, got {row[column]!r}",
            )


def _media_path(
    root: Path, value: Any, report: DatasetValidationReport, location: str
) -> Path | None:
    if not isinstance(value, str) or not value:
        _issue(report, "media_path", location, "Missing media file path")
        return None
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        _issue(report, "media_path", location, "Media file escapes dataset directory")
        return None
    if not path.is_file():
        _issue(report, "media_missing", location, f"Media file does not exist: {value}")
        return None
    return path


def _video_count(
    path: Path, report: DatasetValidationReport
) -> tuple[int, float] | None:
    try:
        import av

        with av.open(str(path)) as container:
            stream = next(iter(container.streams.video), None)
            if stream is None:
                raise ValueError("No video stream")
            frames = sum(1 for _ in container.decode(stream))
            rate = (
                float(stream.average_rate) if stream.average_rate is not None else 0.0
            )
            return frames, rate
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        _issue(report, "media_decode", str(path), str(error))
        return None


def _check_depth(
    root: Path,
    episodes: Mapping[int, Mapping[str, Any]],
    report: DatasetValidationReport,
    *,
    check_media: bool,
    media_frame_counts: bool,
) -> None:
    path = root / "depth_meta.json"
    if not path.exists():
        return
    metadata = _json_mapping(path, report)
    sensors = metadata.get("sensors", {})
    if not isinstance(sensors, dict):
        _issue(report, "depth_metadata", str(path), "sensors must be a mapping")
        return
    for key, sensor in sensors.items():
        location = f"{path}/sensors/{key}"
        if not isinstance(sensor, dict):
            _issue(
                report, "depth_metadata", location, "Sensor metadata must be a mapping"
            )
            continue
        # "auto" describes the producer's dtype-based input interpretation
        # (float metres / integer millimetres). Stored quantization bounds and
        # decoded outputs still have explicit metric units.
        for unit, accepted in (
            ("video.input_unit", {"auto", "m", "mm"}),
            ("video.output_unit", {"m", "mm"}),
        ):
            if sensor.get(unit) not in accepted:
                _issue(
                    report,
                    "depth_unit",
                    location,
                    f"{unit} must be one of {sorted(accepted)}",
                )
        minimum, maximum = sensor.get("video.depth_min"), sensor.get("video.depth_max")
        if not (
            isinstance(minimum, (int, float))
            and isinstance(maximum, (int, float))
            and math.isfinite(minimum)
            and math.isfinite(maximum)
            and 0 <= minimum < maximum
        ):
            _issue(report, "depth_range", location, "Invalid depth quantization range")
        shape = sensor.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or shape[-1] != 1
            or any(type(item) is not int or item <= 0 for item in shape)
        ):
            _issue(
                report,
                "depth_shape",
                location,
                "Depth shape must be [height, width, 1]",
            )
        recorded = sensor.get("episodes", {})
        if not isinstance(recorded, dict):
            _issue(report, "depth_metadata", location, "episodes must be a mapping")
            continue
        for episode_index, episode in episodes.items():
            entry = recorded.get(str(episode_index))
            where = f"{location}/episodes/{episode_index}"
            if not isinstance(entry, dict):
                _issue(report, "depth_missing", where, "Missing episode depth metadata")
                continue
            expected = _integer(episode.get("length"))
            if _integer(entry.get("frame_count")) != expected:
                _issue(
                    report,
                    "depth_length",
                    where,
                    f"Depth frame_count differs from episode length {expected}",
                )
            if check_media or media_frame_counts:
                video = _media_path(root, entry.get("file"), report, where)
                if video is not None and media_frame_counts:
                    count = _video_count(video, report)
                    if count is not None and count[0] != expected:
                        _issue(
                            report,
                            "depth_length",
                            where,
                            f"Decoded {count[0]} frames, expected {expected}",
                        )
        for index in recorded:
            if not index.isdigit() or int(index) not in episodes:
                _issue(
                    report,
                    "depth_orphan",
                    location,
                    f"Depth refers to undeclared episode {index}",
                )


def _check_videos(
    root: Path,
    info: Mapping[str, Any],
    episodes: Mapping[int, Mapping[str, Any]],
    report: DatasetValidationReport,
    *,
    media_frame_counts: bool,
) -> None:
    spans: dict[Path, list[tuple[int, int, str]]] = defaultdict(list)
    fps = info.get("fps", 0)
    for key, feature in info.get("features", {}).items():
        if feature.get("dtype") != "video":
            continue
        for episode_index, episode in episodes.items():
            location = f"episode[{episode_index}]/videos/{key}"
            prefix = f"videos/{key}"
            template = info.get("video_path")
            try:
                filename = template.format(
                    video_key=key,
                    chunk_index=episode[f"{prefix}/chunk_index"],
                    file_index=episode[f"{prefix}/file_index"],
                    episode_index=episode_index,
                )
                begin = float(episode.get(f"{prefix}/from_timestamp", 0.0))
                end = float(
                    episode.get(
                        f"{prefix}/to_timestamp", begin + episode["length"] / fps
                    )
                )
                if (
                    not math.isfinite(begin)
                    or not math.isfinite(end)
                    or begin < 0
                    or end <= begin
                ):
                    raise ValueError("Invalid video timestamp span")
                expected = episode["length"]
                if not math.isclose((end - begin) * fps, expected, abs_tol=0.1):
                    _issue(
                        report,
                        "video_span",
                        location,
                        f"Video span differs from {expected} frames",
                    )
                path = _media_path(root, filename, report, location)
                if path is not None:
                    spans[path].append((round(begin * fps), round(end * fps), location))
            except (
                AttributeError,
                KeyError,
                TypeError,
                ValueError,
                ZeroDivisionError,
            ) as error:
                _issue(report, "video_metadata", location, str(error))
    if not media_frame_counts:
        return
    for path, ranges in spans.items():
        measured = _video_count(path, report)
        if measured is None:
            continue
        count, rate = measured
        if rate and not math.isclose(rate, fps, rel_tol=0.01):
            _issue(
                report, "video_fps", str(path), f"Video FPS {rate} differs from {fps}"
            )
        if count != max(end for _, end, _ in ranges):
            _issue(
                report,
                "video_length",
                str(path),
                f"Decoded {count} frames, expected final span end {max(end for _, end, _ in ranges)}",
            )


def _check_journals(
    root: Path,
    episodes: Mapping[int, Mapping[str, Any]],
    sidecars: Mapping[int, Mapping[str, Any]],
    report: DatasetValidationReport,
) -> None:
    # Reuse the journal owner's schema/identity validation, while reusing this
    # scan's episode evidence rather than scanning Parquet once again.
    from embodichain.data_pipeline.recording import RecordingJournal

    journal = RecordingJournal(root)
    for path in sorted((root / "meta/embodichain_commits").glob("*.json")):
        raw = _json_mapping(path, report)
        if raw.get("phase") != "complete":
            _issue(
                report,
                "pending_commit",
                str(path),
                f"Incomplete commit phase {raw.get('phase')!r}; inspect/recover before consuming",
            )
        try:
            record = journal.get(path.stem)
        except (OSError, TypeError, ValueError) as error:
            _issue(report, "journal_metadata", str(path), str(error))
            continue
        if record is None or record["phase"] != "complete":
            continue
        target = record["sidecar"]
        index, length = _integer(target.get("lerobot_episode_index")), _integer(
            target.get("length")
        )
        if index not in episodes or length != _integer(episodes[index].get("length")):
            _issue(
                report,
                "journal_episode",
                str(path),
                "Complete journal disagrees with SDK episode index or length",
            )
        if index not in sidecars or sidecars[index] != target:
            _issue(
                report,
                "journal_snapshot",
                str(path),
                "Complete journal disagrees with persisted episode sidecar",
            )


def validate_dataset(
    root: str | Path,
    *,
    check_media: bool = False,
    media_frame_counts: bool = False,
    batch_size: int = 4096,
) -> DatasetValidationReport:
    """Validate on-disk LeRobot numeric, semantic, and persistence invariants.

    Args:
        root: Dataset root containing ``meta/info.json`` and frame Parquet files.
        check_media: Check referenced RGB/image/depth files exist.
        media_frame_counts: Also decode every referenced video once to check its
            frame count and RGB cadence. Implies ``check_media``; requires PyAV.
        batch_size: Maximum Arrow frame rows loaded per batch.

    Returns:
        Diagnostic report. Missing optional EmbodiChain provenance is a warning;
        malformed metadata, pending commits, and broken links are errors.

    Raises:
        ValueError: If ``batch_size`` is not a positive integer.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    root = Path(root).resolve()
    report = DatasetValidationReport(str(root))
    info = _json_mapping(root / "meta/info.json", report)
    if not info:
        if report.errors == 0:
            _issue(report, "metadata", "meta/info.json", "Dataset info is empty")
        return report
    features = info.get("features", {})
    if not isinstance(features, dict) or not all(
        isinstance(item, dict) for item in features.values()
    ):
        _issue(
            report,
            "features",
            "meta/info.json",
            "features must map names to feature objects",
        )
        return report
    for key in ("timestamp", "episode_index", "frame_index", "index", "task_index"):
        if key not in features:
            _issue(
                report, "features", "meta/info.json", f"Missing standard feature {key}"
            )
    fps = info.get("fps")
    if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
        _issue(report, "fps", "meta/info.json", "fps must be finite and positive")
        return report
    episodes = _episode_rows(root, report, batch_size)
    sidecars = _sidecar_rows(root, report)
    report.episodes = len(episodes)
    tasks = _index_table(root, "task", True, report, batch_size)
    subtasks = _index_table(
        root, "subtask", "subtask_index" in features, report, batch_size
    )
    gripper_slices = _action_slices(features, report)
    segments: dict[int, list[dict[str, Any]]] = {}
    data_paths: dict[int, str] = {}
    for index, episode in episodes.items():
        location = f"episode[{index}]"
        length = _integer(episode.get("length"))
        if length is None or length < 1:
            _issue(
                report, "episode_length", location, "Episode length must be positive"
            )
            length = 0
        if "data/chunk_index" in episode and "data/file_index" in episode:
            try:
                data_paths[index] = str(
                    (
                        root
                        / info["data_path"].format(
                            chunk_index=episode["data/chunk_index"],
                            file_index=episode["data/file_index"],
                            episode_index=index,
                        )
                    ).resolve()
                )
            except (KeyError, TypeError, ValueError, AttributeError) as error:
                _issue(report, "data_path", location, str(error))
        if index in sidecars:
            sidecar = sidecars[index]
            if _integer(sidecar.get("length")) != length:
                _issue(
                    report,
                    "episode_length",
                    location,
                    "Sidecar length differs from episode metadata",
                )
            segments[index] = _segments(sidecar, length, location, report)
            if not sidecar.get("source_episode_uuid") and not sidecar.get(
                "source_episode_uuids"
            ):
                _issue(
                    report,
                    "legacy_provenance",
                    location,
                    "Missing stable source identity; split grouping must be conservative",
                    warning=True,
                )
            artifact = sidecar.get("replay_artifact")
            if artifact is not None:
                if not isinstance(artifact, dict) or not isinstance(
                    artifact.get("path"), str
                ):
                    _issue(
                        report,
                        "replay_artifact",
                        location,
                        "Invalid replay_artifact path",
                    )
                else:
                    replay = Path(artifact["path"])
                    replay_path = (root / replay).resolve()
                    if not replay_path.is_relative_to(root) and not (
                        replay.is_absolute() and artifact.get("external") is True
                    ):
                        _issue(
                            report,
                            "replay_path",
                            location,
                            "External replay path requires explicit external=true and an absolute path",
                        )
                    if not replay_path.is_file():
                        _issue(
                            report,
                            "replay_missing",
                            location,
                            f"Replay artifact does not exist: {artifact['path']}",
                        )
                    expected_source = sidecar.get(
                        "parent_episode_uuid",
                        sidecar.get("episode_uuid", sidecar.get("source_episode_uuid")),
                    )
                    if artifact.get("source_episode_uuid") != expected_source:
                        _issue(
                            report,
                            "replay_identity",
                            location,
                            "Replay source identity differs from episode source",
                        )
                    initial_step = _integer(artifact.get("initial_state_step", 0))
                    if (
                        initial_step is None
                        or initial_step < 0
                        or (
                            artifact.get("state_alignment") == "source_episode"
                            and initial_step != 0
                        )
                    ):
                        _issue(
                            report,
                            "replay_state",
                            location,
                            "initial_state_step must be nonnegative and zero for source_episode alignment",
                        )
        elif sidecars:
            _issue(
                report,
                "sidecar_missing",
                location,
                "Missing EmbodiChain episode sidecar",
            )
    for index in sidecars.keys() - episodes.keys():
        _issue(
            report,
            "sidecar_orphan",
            f"episode[{index}]",
            "Sidecar references an undeclared episode",
        )
    _check_journals(root, episodes, sidecars, report)
    counts: dict[int, int] = defaultdict(int)
    first_indices: dict[int, int] = {}
    final_indices: dict[int, int | None] = {}
    cursors: dict[int, int] = defaultdict(int)
    paths = sorted((root / "data").rglob("*.parquet"))
    if not paths:
        _issue(report, "data_missing", str(root / "data"), "No frame Parquet files")
    for location, row in _parquet_rows(paths, report, batch_size):
        index = _integer(row.get("episode_index"))
        frame = _integer(row.get("frame_index"))
        global_index = _integer(row.get("index"))
        if global_index != report.frames:
            _issue(
                report,
                "frame_index",
                location,
                f"Expected global index {report.frames}, got {global_index}",
            )
        report.frames += 1
        if index is None or index not in episodes:
            _issue(report, "episode_reference", location, f"Undeclared episode {index}")
            continue
        if index in data_paths and location.rsplit(":", 1)[0] != data_paths[index]:
            _issue(
                report,
                "data_path",
                location,
                f"Episode metadata points to {data_paths[index]}",
            )
        expected_frame = counts[index]
        if frame != expected_frame:
            _issue(
                report,
                "frame_index",
                location,
                f"Expected episode frame {expected_frame}, got {frame}",
            )
        counts[index] += 1
        first_indices.setdefault(index, global_index)
        final_indices[index] = global_index
        timestamp = _scalar(row.get("timestamp"))
        if (
            not isinstance(timestamp, (int, float))
            or not math.isfinite(timestamp)
            or not math.isclose(
                timestamp,
                expected_frame / fps,
                abs_tol=max(
                    1e-5,
                    0.001 / fps,
                    float(np.spacing(np.float32(expected_frame / fps))) * 1.1,
                ),
            )
        ):
            _issue(
                report,
                "timestamp",
                location,
                f"Expected timestamp {expected_frame / fps:.8f}, got {timestamp!r}",
            )
        for key, feature in features.items():
            dtype = feature.get("dtype")
            if dtype == "video":
                continue
            if key not in row:
                _issue(
                    report,
                    "feature_missing",
                    f"{location}/{key}",
                    "Declared feature missing from row",
                )
                continue
            if dtype == "image":
                if check_media or media_frame_counts:
                    image = row[key]
                    if isinstance(image, dict) and image.get("bytes") is None:
                        _media_path(
                            root, image.get("path"), report, f"{location}/{key}"
                        )
                    elif not isinstance(image, dict) or not image.get("bytes"):
                        _issue(
                            report,
                            "image_missing",
                            f"{location}/{key}",
                            "Image has neither bytes nor path",
                        )
                continue
            if dtype in {"string", "language", "json", "binary", "list", "struct"}:
                continue
            array = _numeric_feature(row[key], feature, f"{location}/{key}", report)
            if key == "action" and array is not None:
                for start, stop in gripper_slices:
                    grip = array.reshape(-1)[start:stop]
                    if np.any((grip < -1) | (grip > 1)):
                        _issue(
                            report,
                            "gripper_bounds",
                            f"{location}/action",
                            "Normalized parallel gripper lies outside [-1, 1]",
                        )
        for key, valid_indices in (("task_index", tasks), ("subtask_index", subtasks)):
            if key in row and _integer(row[key]) not in valid_indices:
                _issue(
                    report,
                    "language_reference",
                    f"{location}/{key}",
                    f"Unknown {key}: {row[key]}",
                )
        spans = segments.get(index, [])
        cursor = cursors[index]
        while cursor < len(spans) and spans[cursor]["end_step"] <= expected_frame:
            cursor += 1
        cursors[index] = cursor
        segment = (
            spans[cursor]
            if cursor < len(spans) and spans[cursor]["start_step"] <= expected_frame
            else None
        )
        _check_frame_annotations(row, expected_frame, segment, location, report)
        sidecar = sidecars.get(index, {})
        overall = sidecar.get("instruction")
        if (
            isinstance(overall, str)
            and tasks.get(_integer(row.get("task_index"))) != overall
        ):
            _issue(
                report,
                "task_description",
                location,
                "Frame task differs from overall episode instruction",
            )
        instruction = segment.get("instruction") if segment is not None else None
        expected_subtask = instruction or overall
        if (
            isinstance(expected_subtask, str)
            and "subtask_index" in row
            and subtasks.get(_integer(row["subtask_index"])) != expected_subtask.strip()
        ):
            _issue(
                report,
                "subtask_description",
                location,
                "Frame subtask differs from active segment instruction",
            )
    for index, episode in episodes.items():
        if counts[index] != _integer(episode.get("length")):
            _issue(
                report,
                "episode_length",
                f"episode[{index}]",
                f"Scanned {counts[index]} frames, declared {episode.get('length')}",
            )
        for key, actual in (
            ("dataset_from_index", first_indices.get(index)),
            (
                "dataset_to_index",
                None if final_indices.get(index) is None else final_indices[index] + 1,
            ),
        ):
            if key in episode and _integer(episode[key]) != actual:
                _issue(
                    report,
                    "episode_range",
                    f"episode[{index}]/{key}",
                    f"Expected {actual}, got {episode[key]}",
                )
    for key, actual in (
        ("total_episodes", report.episodes),
        ("total_frames", report.frames),
        ("total_tasks", len(tasks)),
    ):
        if key in info and _integer(info[key]) != actual:
            _issue(
                report,
                "dataset_totals",
                "meta/info.json",
                f"{key}={info[key]}, scanned {actual}",
            )
    _check_depth(
        root,
        episodes,
        report,
        check_media=check_media,
        media_frame_counts=media_frame_counts,
    )
    if check_media or media_frame_counts:
        _check_videos(
            root, info, episodes, report, media_frame_counts=media_frame_counts
        )
    return report


def _lineage_values(record: Mapping[str, Any], key: str) -> list[str]:
    values: list[str] = []
    for mapping in (
        record,
        *(record.get(field, {}) for field in ("metadata", "provenance", "lineage")),
    ):
        if isinstance(mapping, Mapping):
            value = mapping.get(key)
            if value is None:
                continue
            if key == "source_episode_uuids":
                if (
                    not isinstance(value, list)
                    or not value
                    or not all(isinstance(item, str) and item.strip() for item in value)
                ):
                    raise ValueError(
                        "source_episode_uuids must be a nonempty list of nonempty strings"
                    )
                values.extend(value)
            elif isinstance(value, str) and value.strip():
                values.append(value)
            elif isinstance(value, list):
                if key in {
                    "episode_uuid",
                    "source_episode_uuid",
                    "parent_episode_uuid",
                }:
                    raise ValueError(f"{key} must be a nonempty string")
                if not value or not all(
                    isinstance(item, str) and item.strip() for item in value
                ):
                    raise ValueError(f"{key} values must be nonempty strings")
                values.extend(value)
            elif key in {"episode_uuid", "source_episode_uuid", "parent_episode_uuid"}:
                raise ValueError(f"{key} must be a nonempty string")
    return values


def _expansion_lineage_tokens(record: Mapping[str, Any]) -> set[str]:
    tokens: set[str] = set()
    expansions = record.get("expansion", [])
    if not isinstance(expansions, list):
        raise ValueError("expansion must be a list")
    for expansion in expansions:
        if not isinstance(expansion, Mapping):
            raise ValueError("expansion records must be mappings")
        lineage = expansion.get("selected_candidate_lineage")
        if lineage is None:
            continue
        fields = ("scene_case_id", "initial_state_id", "source_id", "source_revision")
        if not isinstance(lineage, Mapping) or not all(
            isinstance(lineage.get(key), str) and lineage[key].strip() for key in fields
        ):
            raise ValueError(
                "selected_candidate_lineage requires nonempty scene/state/source/revision strings"
            )
        reference = [lineage[key] for key in fields]
        tokens.add("reference:" + json.dumps(reference, separators=(",", ":")))
        scope = [
            lineage["scene_case_id"],
            lineage["source_id"],
            lineage["source_revision"],
        ]
        for key, kind in (
            ("candidate_id", "candidate"),
            ("parent_id", "candidate"),
            ("geometry_family_id", "family"),
        ):
            value = lineage.get(key)
            if value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"selected_candidate_lineage {key} must be nonempty text"
                )
            tokens.add(kind + ":" + json.dumps([*scope, value], separators=(",", ":")))
    return tokens


def _group_episodes(
    records: Mapping[int, Mapping[str, Any]],
    *,
    group_by: str,
    scene_field: str,
    allow_legacy_independent: bool,
) -> tuple[dict[int, str], list[str]]:
    parents = {index: index for index in records}

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def unite(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        parents[max(first_root, second_root)] = min(first_root, second_root)

    seen: dict[str, int] = {}
    run_members: dict[str, list[int]] = defaultdict(list)
    fingerprint_members: dict[str, list[int]] = defaultdict(list)
    unknown: list[int] = []
    notes: list[str] = []
    tokens_by_index: dict[int, set[str]] = {}
    for index, record in records.items():
        tokens = {
            f"episode:{value}"
            for key in (
                "episode_uuid",
                "source_episode_uuid",
                "source_episode_uuids",
                "parent_episode_uuid",
            )
            for value in _lineage_values(record, key)
        }
        tokens.update(_expansion_lineage_tokens(record))
        has_source = bool(
            _lineage_values(record, "source_episode_uuid")
            or _lineage_values(record, "source_episode_uuids")
            or _lineage_values(record, "parent_episode_uuid")
        )
        if record.get("lineage_unknown") is True:
            has_source = False
        fingerprint = record.get("source_dataset_fingerprint")
        if isinstance(fingerprint, str) and fingerprint:
            fingerprint_members[fingerprint].append(index)
        run = record.get("run_uuid")
        if isinstance(run, str) and run:
            run_members[run].append(index)
        if group_by == "scene":
            scene = _lineage_values(record, scene_field)
            if not scene:
                raise ValueError(
                    f"episode {index} has no {scene_field!r}; explicit scene grouping requires it"
                )
            tokens.update(f"scene:{value}" for value in scene)
        if not has_source:
            if allow_legacy_independent:
                notes.append(
                    f"episode {index}: missing source identity; caller explicitly assumes independence"
                )
            else:
                unknown.append(index)
        tokens.add(f"dataset_episode:{index}")
        tokens_by_index[index] = tokens
        for token in tokens:
            if token in seen:
                unite(index, seen[token])
            else:
                seen[token] = index
    for index in unknown:
        run = records[index].get("run_uuid")
        fingerprint = records[index].get("source_dataset_fingerprint")
        peers = (
            fingerprint_members.get(fingerprint, [])
            if isinstance(fingerprint, str)
            else []
        )
        if not peers:
            peers = run_members.get(run, []) if isinstance(run, str) else []
        if not peers:
            peers = list(records)
        for peer in peers:
            unite(index, peer)
        notes.append(
            f"episode {index}: missing source identity; conservatively grouped with {len(peers)} episodes"
        )
    groups: dict[int, list[int]] = defaultdict(list)
    for index in records:
        groups[find(index)].append(index)
    identities: dict[int, str] = {}
    for members in groups.values():
        tokens = sorted(
            {token for index in members for token in tokens_by_index[index]}
        )
        identity = hashlib.sha256("\n".join(tokens).encode()).hexdigest()
        for index in members:
            identities[index] = identity
    return identities, notes


def _physical_valid(record: Mapping[str, Any]) -> bool | None:
    if type(record.get("physical_valid")) is bool:
        return record["physical_valid"]
    physical = record.get("physical_validation")
    if isinstance(physical, Mapping):
        for key in ("accepted", "success", "valid"):
            if type(physical.get(key)) is bool:
                return physical[key]
    objective = record.get("physical_objective")
    if (
        isinstance(objective, Mapping)
        and objective.get("predicate") == "ordered_stable_regions"
    ):
        metrics = objective.get("metrics")
        if (
            isinstance(metrics, Mapping)
            and type(metrics.get("measurement_valid")) is bool
            and type(objective.get("success")) is bool
        ):
            return objective["success"] and metrics["measurement_valid"]
    return None


def _recovery_count(record: Mapping[str, Any]) -> int | None:
    value = _integer(record.get("recovery_count"))
    if value is not None and value >= 0:
        return value
    recoveries = record.get("recoveries")
    if isinstance(recoveries, list):
        return len(recoveries)
    # Canonical Task Program results contain complete event histories and row
    # identities. "replanned" counts an actual retry or local replan once;
    # counting both action_retry and replanned would double-count a retry.
    segments = record.get("segments")
    env_id = _integer(record.get("env_id", record.get("source_env_id")))
    if env_id is None or not isinstance(segments, list) or not segments:
        return None
    total = 0
    for segment in segments:
        metadata = segment.get("metadata", {}) if isinstance(segment, Mapping) else {}
        runtime = metadata.get("runtime") if isinstance(metadata, Mapping) else None
        count = _runtime_recoveries(runtime, env_id)
        if count is None:
            return None
        total += count
    return total


def _runtime_recoveries(runtime: Any, env_id: int) -> int | None:
    if not isinstance(runtime, Mapping):
        return None
    if (
        runtime.get("kind") == "parallel_skill_result"
        and runtime.get("schema_version") == 1
    ):
        branches = runtime.get("branches")
        if not isinstance(branches, Mapping) or not branches:
            return None
        counts = [_runtime_recoveries(branch, env_id) for branch in branches.values()]
        return None if any(count is None for count in counts) else sum(counts)
    if runtime.get("kind") != "skill_result" or runtime.get("schema_version") != 2:
        return None
    env_ids = runtime.get("env_ids")
    if not isinstance(env_ids, list) or env_id not in env_ids:
        return None
    row = env_ids.index(env_id)
    events, recoveries = runtime.get("events"), runtime.get("workflow_recoveries")
    if not isinstance(events, list) or not isinstance(recoveries, list):
        return None
    total = 0
    for event in events:
        if not isinstance(event, Mapping):
            return None
        if event.get("kind") == "replanned":
            mask = event.get("env_mask")
            if (
                not isinstance(mask, list)
                or len(mask) != len(env_ids)
                or type(mask[row]) is not bool
            ):
                return None
            total += int(mask[row])
    cycles: set[tuple[int, int]] = set()
    for recovery in recoveries:
        if not isinstance(recovery, Mapping):
            return None
        trigger, attempt = _integer(recovery.get("trigger_call_index")), _integer(
            recovery.get("attempt_index")
        )
        masks = recovery.get("masks")
        mask = masks.get("entered") if isinstance(masks, Mapping) else None
        if (
            trigger is None
            or attempt is None
            or not isinstance(mask, list)
            or len(mask) != len(env_ids)
            or type(mask[row]) is not bool
        ):
            return None
        if mask[row]:
            cycles.add((trigger, attempt))
    return total + len(cycles)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(filename)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _check_output_path(root: Path, output: Path) -> None:
    if (
        output == root
        or (output.exists() and root in output.parents and output.suffix != ".json")
        or (
            root in output.parents
            and output.parts[len(root.parts)]
            in {"meta", "data", "videos", "depth_videos", "images"}
        )
        or output == root / "depth_meta.json"
    ):
        raise ValueError(
            "output must be a separate manifest/report, outside dataset-owned files"
        )


def create_split_manifest(
    root: str | Path,
    output: str | Path,
    *,
    fractions: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 0,
    group_by: Literal["lineage", "scene"] = "lineage",
    scene_field: str = "scene_id",
    success_only: bool = False,
    failure_only: bool = False,
    max_recoveries: int | None = None,
    require_physical_valid: bool = False,
    allow_legacy_independent: bool = False,
) -> dict[str, Any]:
    """Write an atomic split manifest without modifying dataset metadata.

    Shared source, parent, and episode identities form transitive groups before
    quality filtering. Scene grouping adds scene identities to these groups;
    it cannot separate two derivatives of the same episode. Unknown provenance
    conservatively groups the recording run, or the whole dataset if no run is
    known. Fractions are targets: indivisible groups may leave a split empty.

    Args:
        root: Dataset directory with episode metadata and optional sidecars.
        output: Separate JSON manifest destination; existing source files cannot
            be overwritten.
        fractions: Train, validation, and test fractions summing to one.
        seed: Seed for deterministic group assignment.
        group_by: Group by lineage, optionally also by explicit scene identity.
        scene_field: Sidecar field containing a scene identifier.
        success_only: Select only explicitly successful episodes.
        failure_only: Select only explicitly failed episodes.
        max_recoveries: Maximum explicit recovery count; unknown counts are
            excluded when this filter is requested. Canonical Task Program traces
            also support counting row-local replanned events plus distinct
            workflow recovery cycles; retries are counted once.
        require_physical_valid: Require an explicit accepted physical validation
            or an ordered-stable-regions objective with successful valid measured
            evidence. Declaring an objective without measurements is insufficient.
        allow_legacy_independent: Explicitly assume unidentified legacy episodes
            are independent; recorded as an assumption in the manifest.

    Returns:
        Manifest containing split indices, group identities, exclusions, counts,
        quality selection, and provenance assumptions.

    Raises:
        ValueError: If arguments or episode metadata are invalid.
    """
    if (
        len(fractions) != 3
        or any(
            not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
            for value in fractions
        )
        or not math.isclose(sum(fractions), 1.0, abs_tol=1e-9)
    ):
        raise ValueError(
            "fractions must be three finite nonnegative values summing to one"
        )
    if type(seed) is not int or group_by not in {"lineage", "scene"}:
        raise ValueError(
            "seed must be an integer and group_by must be lineage or scene"
        )
    if success_only and failure_only:
        raise ValueError("success_only and failure_only are mutually exclusive")
    if max_recoveries is not None and (
        type(max_recoveries) is not int or max_recoveries < 0
    ):
        raise ValueError("max_recoveries must be a nonnegative integer")
    root, output = Path(root).resolve(), Path(output).resolve()
    _check_output_path(root, output)
    report = DatasetValidationReport(str(root))
    episodes = _episode_rows(root, report, 4096)
    sidecars = _sidecar_rows(root, report)
    report.raise_for_errors()
    if not episodes:
        raise ValueError("Dataset has no episodes")
    for index, episode in episodes.items():
        length = _integer(episode.get("length"))
        if length is None or length < 1:
            raise ValueError(f"episode {index} has an invalid length")
    _check_journals(root, episodes, sidecars, report)
    if any(issue.code == "pending_commit" for issue in report.issues):
        raise ValueError(
            "Incomplete recording commit; inspect/recover before splitting"
        )
    report.raise_for_errors()
    records = {
        index: {**episode, **sidecars.get(index, {})}
        for index, episode in episodes.items()
    }
    identities, notes = _group_episodes(
        records,
        group_by=group_by,
        scene_field=scene_field,
        allow_legacy_independent=allow_legacy_independent,
    )
    excluded: list[dict[str, Any]] = []
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in records.items():
        reasons: list[str] = []
        success = record.get("success")
        if success_only and success is not True:
            reasons.append("success_not_confirmed")
        if failure_only and success is not False:
            reasons.append("failure_not_confirmed")
        recoveries = _recovery_count(record)
        if max_recoveries is not None and (
            recoveries is None or recoveries > max_recoveries
        ):
            reasons.append(
                "recovery_count_unknown" if recoveries is None else "recovery_limit"
            )
        if require_physical_valid and _physical_valid(record) is not True:
            reasons.append("physical_validation_not_confirmed")
        if reasons:
            excluded.append({"episode_index": index, "reasons": reasons})
        else:
            groups[identities[index]].append(index)
    if not groups:
        raise ValueError("No episodes satisfy the quality filters")
    names = ("train", "validation", "test")
    splits: dict[str, list[int]] = {name: [] for name in names}
    frame_counts = {name: 0 for name in names}
    selected = sum(len(members) for members in groups.values())
    targets = [float(fraction) * selected for fraction in fractions]
    # Place largest indivisible groups first; seed breaks ties reproducibly.
    ordered = sorted(
        groups,
        key=lambda identity: (
            -len(groups[identity]),
            hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest(),
        ),
    )
    assignments: dict[str, str] = {}
    for identity in ordered:
        split = max(
            (position for position in range(3) if fractions[position] > 0),
            key=lambda position: (
                targets[position] - len(splits[names[position]]),
                -position,
            ),
        )
        name = names[split]
        splits[name].extend(groups[identity])
        frame_counts[name] += sum(
            int(episodes[index]["length"]) for index in groups[identity]
        )
        assignments[identity] = name
    manifest = {
        "schema_version": 1,
        "dataset_root": str(root),
        "seed": seed,
        "group_by": group_by,
        "scene_field": scene_field if group_by == "scene" else None,
        "fractions": dict(zip(names, fractions)),
        "quality": {
            "success_only": success_only,
            "failure_only": failure_only,
            "max_recoveries": max_recoveries,
            "require_physical_valid": require_physical_valid,
        },
        "splits": {name: sorted(indices) for name, indices in splits.items()},
        "episode_counts": {name: len(indices) for name, indices in splits.items()},
        "frame_counts": frame_counts,
        "groups": [
            {
                "group_id": identity,
                "split": assignments[identity],
                "episode_indices": sorted(groups[identity]),
            }
            for identity in sorted(groups)
        ],
        "excluded": sorted(excluded, key=lambda item: item["episode_index"]),
        "assumptions": notes,
        "allow_legacy_independent": allow_legacy_independent,
    }
    _atomic_json(output, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone dataset validation/split command.

    Args:
        argv: Arguments excluding the executable name; defaults to process args.

    Returns:
        Zero on success, one for validation errors, two for invalid requests.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser(
        "validate", help="Scan dataset files without simulation"
    )
    validate.add_argument("root")
    validate.add_argument("--check-media", action="store_true")
    validate.add_argument("--media-frame-counts", action="store_true")
    validate.add_argument("--batch-size", type=int, default=4096)
    validate.add_argument("--output")
    split = commands.add_parser(
        "split", help="Create a lineage-safe quality-filtered split"
    )
    split.add_argument("root")
    split.add_argument("output")
    split.add_argument("--fractions", type=float, nargs=3, default=(0.8, 0.1, 0.1))
    split.add_argument("--seed", type=int, default=0)
    split.add_argument("--group-by", choices=("lineage", "scene"), default="lineage")
    split.add_argument("--scene-field", default="scene_id")
    quality = split.add_mutually_exclusive_group()
    quality.add_argument("--success-only", action="store_true")
    quality.add_argument("--failure-only", action="store_true")
    split.add_argument("--max-recoveries", type=int)
    split.add_argument("--require-physical-valid", action="store_true")
    split.add_argument("--allow-legacy-independent", action="store_true")
    arguments = vars(parser.parse_args(argv))
    command, output = arguments.pop("command"), arguments.pop("output")
    try:
        if command == "validate":
            report = validate_dataset(**arguments)
            if output:
                report_path = Path(output).resolve()
                _check_output_path(Path(report.root), report_path)
                _atomic_json(report_path, report.to_dict())
            print(json.dumps(report.to_dict(), indent=2))
            return 0 if report.ok else 1
        result = create_split_manifest(output=output, **arguments)
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError) as error:
        print(json.dumps({"error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
