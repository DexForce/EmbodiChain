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

"""File-only migration of v3 recordings to the LeRobot 0.6.1 consumer contract."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import uuid

import av
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.configs.recipe import TrainingRecipe
from lerobot.configs.video import DepthEncoderConfig
from lerobot.datasets.compute_stats import (
    RunningQuantileStats,
    aggregate_stats,
    get_feature_stats,
)
from lerobot.datasets.depth_utils import dequantize_depth
from lerobot.datasets.io_utils import write_table_one_row_group_per_episode
from lerobot.datasets.language import (
    LANGUAGE_PERSISTENT,
    language_feature_info,
    language_persistent_arrow_type,
)
from lerobot.datasets.language_render import render_sample
from lerobot.datasets.lerobot_dataset import LeRobotDataset

__all__ = ["export_dataset"]

_SIDECAR = "meta/embodichain_episodes.jsonl"
_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
_RECIPE = {
    "messages": [
        {"role": "user", "content": "${task}", "stream": "high_level"},
        {
            "role": "assistant",
            "content": "${subtask}",
            "stream": "low_level",
            "target": True,
        },
    ]
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _portable_path(root: Path, value: str) -> Path:
    path = root / value
    if Path(value).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Dataset path escapes its root: {value!r}")
    return path


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    files: dict[str, tuple[int, int]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Dataset contains a symlink: {path}")
        if path.is_file():
            stat = path.stat()
            files[path.relative_to(root).as_posix()] = (stat.st_size, stat.st_mtime_ns)
    return files


def _fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted((root / "meta").rglob("*")) + sorted(
        (root / "data").rglob("*.parquet")
    )
    for path in paths:
        if not path.is_file():
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


def _lookup(root: Path, name: str) -> dict[int, str]:
    path = root / "meta" / f"{name}s.parquet"
    if not path.exists():
        return {}
    table = pd.read_parquet(path)
    return {int(row[f"{name}_index"]): str(text) for text, row in table.iterrows()}


def _sidecars(root: Path) -> dict[int, dict[str, Any]]:
    path = root / _SIDECAR
    records = {}
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                index = int(record["lerobot_episode_index"])
                if index in records:
                    raise ValueError(f"Duplicate sidecar for episode {index}")
                records[index] = record
    return records


def _archive_journals(
    stage: Path,
    sidecars: dict[int, dict[str, Any]],
    episodes: list[dict[str, Any]],
    fingerprint: str,
) -> int:
    """Reject unresolved commits and retain source evidence outside live recovery."""
    journal_root = stage / "meta/embodichain_commits"
    if not journal_root.exists():
        return 0
    count = 0
    depth_path = stage / "depth_meta.json"
    depth = _read_json(depth_path) if depth_path.exists() else {}
    for path in sorted(journal_root.glob("*.json")):
        record = _read_json(path)
        identity = str(uuid.UUID(path.stem))
        if (
            record.get("schema_version") != 1
            or record.get("episode_uuid") != identity
            or record.get("phase") != "complete"
        ):
            raise ValueError(
                f"Unfinished or invalid source recording journal: {path.name}"
            )
        sidecar = record.get("sidecar", {})
        index = sidecar.get("lerobot_episode_index")
        if (
            sidecar.get("episode_uuid") != identity
            or sidecars.get(index) != sidecar
            or not isinstance(index, int)
            or not 0 <= index < len(episodes)
            or sidecar.get("length") != episodes[index]["length"]
        ):
            raise ValueError(
                f"Source journal conflicts with episode sidecar: {path.name}"
            )
        sensors = record.get("depth_sensors")
        if not isinstance(sensors, list) or any(
            sensor not in depth.get("sensors", {}) for sensor in sensors
        ):
            raise ValueError(
                f"Source journal references missing depth sensors: {path.name}"
            )
        replay = sidecar.get("replay_artifact")
        if replay:
            name = Path(replay["path"])
            if name.is_absolute():
                if replay.get("external") is not True:
                    raise ValueError("Absolute replay path requires external=true")
                artifact = name
            else:
                artifact = _portable_path(stage, name.as_posix())
            if not artifact.is_file() or artifact.stat().st_size == 0:
                raise ValueError(f"Source replay artifact is missing: {name}")
        count += 1
    archive = stage / "meta/embodichain_source_commits" / fingerprint
    archive.parent.mkdir(parents=True, exist_ok=True)
    journal_root.rename(archive)
    return count


def _episode_frames(root: Path) -> dict[int, dict[str, list[Any]]]:
    """Read only small annotation columns, with no RGB/depth/action materialization."""
    episodes: dict[int, dict[str, list[Any]]] = {}
    for path in sorted((root / "data").rglob("*.parquet")):
        parquet = pq.ParquetFile(path)
        available = parquet.schema_arrow.names
        columns = [
            key
            for key in (
                "episode_index",
                "index",
                "frame_index",
                "timestamp",
                "task_index",
                "subtask_index",
                "segment_id",
            )
            if key in available
        ]
        for batch in parquet.iter_batches(columns=columns):
            for row in batch.to_pylist():
                index = int(row["episode_index"])
                episode = episodes.setdefault(index, {key: [] for key in columns})
                for key, value in row.items():
                    if isinstance(value, list) and len(value) == 1:
                        value = value[0]
                    episode[key].append(value)
    return episodes


def _annotations(
    frames: dict[str, list[Any]],
    metadata: dict[str, Any],
    tasks: dict[int, str],
    subtasks: dict[int, str],
    fps: int,
) -> tuple[str, list[dict[str, Any]], float]:
    count = len(frames["timestamp"])
    if not count or frames["frame_index"] != list(range(count)):
        raise ValueError("Episode frame_index must be contiguous from zero")
    timestamps = np.asarray(frames["timestamp"], dtype=np.float64)
    origin = float(timestamps[0])
    if not np.isfinite(timestamps).all() or not np.allclose(
        timestamps - origin, np.arange(count) / fps, atol=1e-4, rtol=0
    ):
        raise ValueError("Episode timestamps do not match the dataset FPS")
    task = metadata.get("instruction")
    if not isinstance(task, str) or not task.strip():
        descriptions = {tasks[int(index)] for index in frames["task_index"]}
        if len(descriptions) != 1:
            raise ValueError(
                "Changing frame tasks require an overall sidecar instruction"
            )
        task = descriptions.pop()
    labels = [task] * count
    boundaries = {0}
    occupied = [False] * count
    if "subtask_index" in frames:
        labels = [subtasks[int(index)] for index in frames["subtask_index"]]
    for segment in metadata.get("segments", []):
        start, end = segment.get("start_step", 0), segment.get("end_step", 0)
        if type(start) is not int or type(end) is not int:
            raise ValueError("Segment boundaries must be integers")
        if not 0 <= start <= end <= count:
            raise ValueError(f"Segment range [{start}, {end}) is outside its episode")
        if start == end:
            continue
        if any(occupied[start:end]):
            raise ValueError("Segment ranges overlap")
        occupied[start:end] = [True] * (end - start)
        boundaries.update((start, end))
        instruction = segment.get("instruction")
        if instruction:
            labels[start:end] = [instruction] * (end - start)
    segment_ids = frames.get("segment_id", [None] * count)
    rows = []
    for step, label in enumerate(labels):
        if (
            step in boundaries
            or label != labels[step - 1]
            or (step and segment_ids[step] != segment_ids[step - 1])
        ):
            rows.append(
                {
                    "role": "assistant",
                    "content": label,
                    "style": "subtask",
                    "timestamp": float(np.float32(timestamps[step] - origin)),
                    "camera": None,
                    "tool_calls": None,
                }
            )
    return task, rows, origin


def _numeric_stats(values: np.ndarray) -> dict[str, Any]:
    return {
        key: value.tolist()
        for key, value in get_feature_stats(values, axis=0, keepdims=True).items()
    }


def _rebase_rgb_video(
    stage: Path,
    info: dict[str, Any],
    episode: dict[str, Any],
    key: str,
    origin: float,
) -> None:
    start = float(episode[f"videos/{key}/from_timestamp"]) + origin
    end = float(episode[f"videos/{key}/to_timestamp"]) + origin
    duration = episode["length"] / info["fps"]
    if (
        not math.isfinite(start)
        or not math.isfinite(end)
        or start < 0
        or end < start
        or not math.isclose(end - start, duration, abs_tol=1e-4)
    ):
        raise ValueError(
            f"RGB video interval for {key} differs from its episode length"
        )
    path = _portable_path(
        stage,
        info["video_path"].format(
            video_key=key,
            chunk_index=episode[f"videos/{key}/chunk_index"],
            file_index=episode[f"videos/{key}/file_index"],
        ),
    )
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.average_rate is None or float(stream.average_rate) != info["fps"]:
            raise ValueError(f"RGB video FPS for {key} differs from its episode")
        if (
            stream.duration is None
            or end > float(stream.duration * stream.time_base) + 1e-4
        ):
            raise ValueError(
                f"RGB video interval for {key} exceeds its stream duration"
            )
    episode[f"videos/{key}/from_timestamp"] = start
    episode[f"videos/{key}/to_timestamp"] = end


def _rewrite_frames(
    stage: Path,
    annotations: dict[int, tuple[str, list[dict[str, Any]], float]],
    task_indices: dict[str, int],
    depth_keys: set[str],
) -> None:
    for path in sorted((stage / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        indices = table["episode_index"].to_pylist()
        table = table.replace_schema_metadata(None)
        for key in depth_keys:
            if key in table.column_names:
                table = table.drop_columns(key)
        if LANGUAGE_PERSISTENT in table.column_names:
            raise ValueError(
                "Source already contains language_persistent; refusing to replace it"
            )
        table = table.append_column(
            LANGUAGE_PERSISTENT,
            # Infer first: Arrow cannot construct nested JSON extension arrays
            # from Python lists directly. Casting then enforces the official type.
            pa.array([annotations[index][1] for index in indices]).cast(
                language_persistent_arrow_type()
            ),
        )
        table = table.set_column(
            table.column_names.index("task_index"),
            "task_index",
            pa.array(
                [task_indices[annotations[index][0]] for index in indices], pa.int64()
            ),
        )
        timestamps = np.asarray(table["timestamp"])
        timestamps = timestamps - np.asarray(
            [annotations[index][2] for index in indices]
        )
        table = table.set_column(
            table.column_names.index("timestamp"),
            "timestamp",
            pa.array(timestamps, pa.float32()),
        )
        write_table_one_row_group_per_episode(table, path)


def _depth_stats(
    path: Path, sensor: dict[str, Any], count: int, fps: int
) -> dict[str, Any]:
    """Validate coded frames and compute exact decoded population moments in metres."""
    height, width, channels = sensor["shape"]
    if channels != 1 or sensor.get("video.quant_bits", 12) != 12:
        raise ValueError(
            "Only official 12-bit, single-channel depth sidecars are supported"
        )
    if sensor.get("video.qmax", 4095) != 4095:
        raise ValueError("Unexpected depth quantization code range")
    # EmbodiChain stores the FFmpeg encoder name; upstream expects its codec name.
    video_info = dict(sensor)
    if video_info.get("video.codec") == "libx265":
        video_info["video.codec"] = "hevc"
    config = DepthEncoderConfig.from_video_info(video_info)
    if config.pix_fmt != "gray12le":
        raise ValueError("Only gray12le depth sidecars are supported")
    if (
        not all(
            math.isfinite(value)
            for value in (config.depth_min, config.depth_max, config.shift)
        )
        or not 0 <= config.depth_min < config.depth_max
        or not isinstance(config.use_log, bool)
        or (config.use_log and config.depth_min + config.shift <= 0)
    ):
        raise ValueError("Invalid depth quantization parameters")
    running = RunningQuantileStats()
    seen = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.codec_context.name != "hevc" or float(stream.average_rate) != fps:
            raise ValueError("Depth stream must use HEVC at the dataset FPS")
        if stream.codec_context.pix_fmt != "gray12le":
            raise ValueError("Depth stream pixel format differs from gray12le metadata")
        for frame in container.decode(stream):
            if frame.pts is None or not math.isclose(
                float(frame.pts * frame.time_base), seen / fps, abs_tol=1e-4
            ):
                raise ValueError("Depth video timestamps must be contiguous from zero")
            codes = frame.to_ndarray(format="gray12le")
            if codes.shape != (height, width) or codes.max() > 4095:
                raise ValueError(
                    "Depth stream shape or code range differs from its metadata"
                )
            depth = dequantize_depth(
                codes,
                depth_min=config.depth_min,
                depth_max=config.depth_max,
                shift=config.shift,
                use_log=config.use_log,
                output_unit="m",
                output_tensor=False,
            )
            running.update(depth.reshape(-1, 1))
            seen += 1
    if seen != count:
        raise ValueError(
            f"Depth frame count {seen} differs from episode length {count}"
        )
    feature_stats = (
        running.get_statistics()
        if count * height * width > 1
        else get_feature_stats(depth.reshape(-1, 1), axis=0, keepdims=True)
    )
    return {
        key: [count] if key == "count" else value.reshape(1, 1, 1).tolist()
        for key, value in feature_stats.items()
    }


def _migrate_depth(
    stage: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    depth: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    stats: dict[int, dict[str, Any]] = {int(ep["episode_index"]): {} for ep in episodes}
    if not depth:
        return stats
    fps = info["fps"]
    if depth.get("fps") != fps:
        raise ValueError("Depth sidecar FPS differs from the LeRobot dataset")
    info["video_path"] = info.get("video_path") or _VIDEO_PATH
    for sensor_key, sensor in depth["sensors"].items():
        key = f"observation.depth.{sensor_key}"
        if sensor_key in {"", ".", ".."} or "/" in sensor_key or "\\" in sensor_key:
            raise ValueError(f"Invalid depth sensor key: {sensor_key!r}")
        quantizer = {
            name: sensor[name]
            for name in (
                "video.depth_min",
                "video.depth_max",
                "video.shift",
                "video.use_log",
            )
        }
        info["features"][key] = {
            "dtype": "video",
            "shape": list(sensor["shape"]),
            "names": ["height", "width", "channels"],
            "info": {
                "is_depth_map": True,
                "depth_unit": "m",
                "video.codec": "hevc",
                "video.pix_fmt": "gray12le",
                "video.fps": fps,
                "video.channels": 1,
                "video.height": sensor["shape"][0],
                "video.width": sensor["shape"][1],
                "video.quant_bits": 12,
                "video.qmax": 4095,
                "video.lossless": sensor.get("video.lossless", False),
                **quantizer,
            },
        }
        for episode in episodes:
            index = int(episode["episode_index"])
            entry = sensor["episodes"].get(str(index))
            if entry is None or int(entry["frame_count"]) != episode["length"]:
                raise ValueError(
                    f"Missing or incomplete depth for episode {index}, sensor {sensor_key}"
                )
            source = _portable_path(stage, entry["file"])
            stats[index][key] = _depth_stats(source, sensor, episode["length"], fps)
            chunk, file_index = divmod(index, info.get("chunks_size", 1000))
            destination = _portable_path(
                stage,
                info["video_path"].format(
                    video_key=key, chunk_index=chunk, file_index=file_index
                ),
            )
            if destination.exists():
                raise ValueError(f"Depth destination already exists: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            episode.update(
                {
                    f"videos/{key}/chunk_index": chunk,
                    f"videos/{key}/file_index": file_index,
                    f"videos/{key}/from_timestamp": 0.0,
                    f"videos/{key}/to_timestamp": episode["length"] / fps,
                }
            )
    return stats


def _check_official(
    stage: Path,
    annotations: dict[int, tuple[str, list[dict[str, Any]], float]],
    episodes: list[dict[str, Any]],
) -> int:
    dataset = LeRobotDataset(
        "local/embodichain-export",
        root=stage,
        video_backend="pyav",
        depth_output_unit="m",
    )
    recipe = TrainingRecipe.from_dict(_RECIPE)
    checked = 0
    for episode in episodes:
        index = int(episode["episode_index"])
        task, language, _ = annotations[index]
        offsets = {0, episode["length"] - 1}
        offsets.update(round(row["timestamp"] * dataset.fps) for row in language)
        for offset in sorted(offsets):
            sample_index = episode["dataset_from_index"] + offset
            sample = dataset[sample_index]
            if sample["task"] != task:
                raise ValueError(
                    f"Official reader changed overall task in episode {index}"
                )
            rendered = render_sample(
                recipe=recipe,
                persistent=sample[LANGUAGE_PERSISTENT],
                events=None,
                t=float(sample["timestamp"]),
                sample_idx=sample_index,
                task=sample["task"],
            )
            expected = language[
                max(
                    i
                    for i, row in enumerate(language)
                    if row["timestamp"] <= offset / dataset.fps + 1e-6
                )
            ]["content"]
            if rendered is None or rendered["messages"][-1]["content"] != expected:
                raise ValueError(
                    f"Official recipe changed subtask in episode {index}, frame {offset}"
                )
            checked += 1
    return checked


def export_dataset(source: str | Path, destination: str | Path) -> dict[str, Any]:
    """Export a finalized recording without changing its source or Python runtime.

    Args:
        source: Existing LeRobot v3.0 dataset produced by EmbodiChain 0.4.x recording.
        destination: New directory, published atomically after official consumer checks.

    Returns:
        Conversion manifest containing episode counts, source fingerprint and checks.

    Raises:
        ValueError: If metadata, annotations or depth streams are inconsistent.
        FileExistsError: If the destination exists.
        RuntimeError: If the installed LeRobot version differs from the pinned version.
    """
    if importlib.metadata.version("lerobot") != "0.6.1":
        raise RuntimeError("Run this tool in its isolated LeRobot 0.6.1 environment")
    source, destination = Path(source).resolve(), Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if destination.resolve().is_relative_to(source):
        raise ValueError("Destination must be outside the source dataset")
    snapshot = _snapshot(source)
    fingerprint = _fingerprint(source)
    info = _read_json(source / "meta/info.json")
    if info.get("codebase_version") != "v3.0":
        raise ValueError(
            "Only LeRobot v3.0 input is supported; migrate older storage first"
        )
    fps = info["fps"]
    if not isinstance(fps, int) or fps <= 0:
        raise ValueError("Dataset FPS must be a positive integer")
    if info.get("total_episodes", 0) <= 0:
        raise ValueError("Source contains no finalized episodes")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        shutil.copytree(source, stage, dirs_exist_ok=True)
        if snapshot != _snapshot(source) or fingerprint != _fingerprint(stage):
            raise RuntimeError(
                "Source changed during export; finalize recording before converting"
            )
        tasks, subtasks = _lookup(stage, "task"), _lookup(stage, "subtask")
        sidecars = _sidecars(stage)
        frames = _episode_frames(stage)
        episode_files = sorted((stage / "meta/episodes").rglob("*.parquet"))
        episodes = [
            row for path in episode_files for row in pq.read_table(path).to_pylist()
        ]
        if [int(ep["episode_index"]) for ep in episodes] != list(
            range(info["total_episodes"])
        ):
            raise ValueError("Episode metadata must cover contiguous episode indices")
        if (
            set(frames) != set(range(info["total_episodes"]))
            or sum(len(ep["timestamp"]) for ep in frames.values())
            != info["total_frames"]
        ):
            raise ValueError("Frame counts differ from dataset metadata")
        expected_start = 0
        for episode in episodes:
            index, count = int(episode["episode_index"]), int(episode["length"])
            if (
                count <= 0
                or episode["dataset_from_index"] != expected_start
                or episode["dataset_to_index"] != expected_start + count
                or frames[index]["index"]
                != list(range(expected_start, expected_start + count))
            ):
                raise ValueError(
                    f"Episode {index} has inconsistent global frame indices"
                )
            expected_start += count
            if (
                "length" in sidecars.get(index, {})
                and sidecars[index]["length"] != count
            ):
                raise ValueError(
                    f"Episode {index} sidecar length differs from its frame data"
                )
        source_commits = _archive_journals(stage, sidecars, episodes, fingerprint)
        annotations = {
            index: _annotations(data, sidecars.get(index, {}), tasks, subtasks, fps)
            for index, data in frames.items()
        }
        task_indices = {text: index for index, text in tasks.items()}
        for task, _, _ in annotations.values():
            task_indices.setdefault(task, len(task_indices))
        pd.DataFrame(
            {"task_index": list(task_indices.values())}, index=list(task_indices)
        ).to_parquet(stage / "meta/tasks.parquet")
        depth_path = stage / "depth_meta.json"
        depth = _read_json(depth_path) if depth_path.exists() else {}
        depth_keys = {
            f"observation.depth.{sensor}" for sensor in depth.get("sensors", {})
        }
        _rewrite_frames(stage, annotations, task_indices, depth_keys)
        changed_stats = _migrate_depth(stage, info, episodes, depth)
        for episode in episodes:
            index = int(episode["episode_index"])
            if episode["length"] != len(frames[index]["timestamp"]):
                raise ValueError(f"Episode {index} length differs from frame data")
            task, _, origin = annotations[index]
            episode["tasks"] = [task]
            changed_stats[index]["task_index"] = _numeric_stats(
                np.full(episode["length"], task_indices[task])
            )
            changed_stats[index]["timestamp"] = _numeric_stats(
                np.arange(episode["length"], dtype=np.float32) / fps
            )
            for key in info["features"]:
                if key in depth_keys or info["features"][key]["dtype"] != "video":
                    continue
                _rebase_rgb_video(stage, info, episode, key, origin)
            for key, feature_stats in changed_stats[index].items():
                for existing in list(episode):
                    if existing.startswith(f"stats/{key}/"):
                        del episode[existing]
                for stat, value in feature_stats.items():
                    episode[f"stats/{key}/{stat}"] = value
        cursor = 0
        for path in episode_files:
            count = pq.ParquetFile(path).metadata.num_rows
            pq.write_table(
                pa.Table.from_pylist(episodes[cursor : cursor + count]), path
            )
            cursor += count
        stats = _read_json(stage / "meta/stats.json")
        aggregated = aggregate_stats(
            [
                {
                    key: {stat: np.asarray(value) for stat, value in values.items()}
                    for key, values in record.items()
                }
                for record in changed_stats.values()
            ]
        )
        stats.update(
            {
                key: {stat: value.tolist() for stat, value in values.items()}
                for key, values in aggregated.items()
            }
        )
        _write_json(stage / "meta/stats.json", stats)
        info["features"][LANGUAGE_PERSISTENT] = language_feature_info()[
            LANGUAGE_PERSISTENT
        ]
        info["total_tasks"] = len(task_indices)
        _write_json(stage / "meta/info.json", info)
        with (stage / _SIDECAR).open("w", encoding="utf-8") as stream:
            for episode in episodes:
                index = int(episode["episode_index"])
                record = dict(sidecars.get(index, {}))
                identity = str(
                    uuid.uuid5(uuid.NAMESPACE_URL, f"embodichain:{fingerprint}:{index}")
                )
                record.setdefault("episode_uuid", identity)
                owners = (record, record.get("lineage"), record.get("provenance"))
                known_source = any(
                    owner.get(key)
                    for owner in owners
                    if isinstance(owner, dict)
                    for key in (
                        "source_episode_uuid",
                        "source_episode_uuids",
                        "parent_episode_uuid",
                    )
                )
                if not known_source:
                    record["lineage_unknown"] = True
                    record["source_dataset_fingerprint"] = fingerprint
                record.update(
                    {
                        "lerobot_episode_index": index,
                        "instruction": annotations[index][0],
                        "length": int(episode["length"]),
                    }
                )
                record["lerobot_export"] = {
                    "source_fingerprint": fingerprint,
                    "timestamp_origin": annotations[index][2],
                    "target_sdk": "0.6.1",
                }
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        _write_json(stage / "meta/task_subtask_recipe.json", _RECIPE)
        checked = _check_official(stage, annotations, episodes)
        manifest = {
            "schema_version": 1,
            "source_fingerprint": fingerprint,
            "target_sdk": "0.6.1",
            "storage_version": "v3.0",
            "total_episodes": len(episodes),
            "total_frames": info["total_frames"],
            "depth_features": sorted(depth_keys),
            "official_samples_checked": checked,
            "source_commit_records": source_commits,
        }
        _write_json(stage / "meta/embodichain_export.json", manifest)
        if depth:
            shutil.rmtree(stage / "depth_videos", ignore_errors=True)
            depth_path.replace(stage / "meta/embodichain_source_depth.json")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        os.rename(stage, destination)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)
