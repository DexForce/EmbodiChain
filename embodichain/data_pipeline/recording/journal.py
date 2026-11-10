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

"""Durable recording commit evidence and conservative offline sidecar repair.

The journal cannot make LeRobot, RGB/depth videos and JSONL one transaction.
``lerobot_committing`` is deliberately ambiguous after a crash. Recovery never
replays buffered frames, removes files, or loads a pickle trajectory.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any
from uuid import UUID

__all__ = ["RecordingJournal", "inspect_recording", "recover_recording"]

_JOURNAL_PATH = Path("meta/embodichain_commits")
_SIDECAR_PATH = Path("meta/embodichain_episodes.jsonl")
_PHASES = (
    "prepared",
    "lerobot_committing",
    "lerobot_committed",
    "depth_committed",
    "complete",
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _local_file(root: Path, name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Unsafe dataset-relative artifact path: {name!r}.")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact path escapes dataset root: {name!r}.")
    return resolved


def _sidecars(
    root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[int, dict[str, Any]], list[str]]:
    records: dict[str, dict[str, Any]] = {}
    indices: dict[int, dict[str, Any]] = {}
    errors: list[str] = []
    path = _local_file(root, _SIDECAR_PATH.as_posix())
    if not path.exists():
        return records, indices, errors
    contents = path.read_text(encoding="utf-8")
    if contents and not contents.endswith("\n"):
        errors.append("Episode sidecar has an unterminated final record.")
    for number, line in enumerate(contents.splitlines(), 1):
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("record is not a mapping")
            index = record.get("lerobot_episode_index", record.get("episode_index"))
            if type(index) is not int or index < 0:
                raise ValueError("episode index must be a non-negative integer")
            if index in indices:
                errors.append(f"Duplicate sidecar episode index {index}.")
            indices[index] = record
            identity = record.get("episode_uuid")
            if isinstance(identity, str):
                if identity in records:
                    errors.append(f"Duplicate sidecar episode_uuid {identity}.")
                records[identity] = record
        except (ValueError, TypeError) as error:
            errors.append(f"Sidecar line {number}: {error}")
    return records, indices, errors


def _episode_evidence(
    root: Path,
) -> tuple[dict[int, int], dict[int, dict[str, Any]], dict[str, Any], list[str]]:
    import pyarrow.parquet as parquet

    counts: dict[int, int] = {}
    first_indices: dict[int, int] = {}
    final_indices: dict[int, int] = {}
    global_index = 0
    errors: list[str] = []
    for path in sorted((root / "data").rglob("*.parquet")):
        try:
            for batch in parquet.ParquetFile(path).iter_batches(
                batch_size=4096, columns=["episode_index", "frame_index", "index"]
            ):
                for row in batch.to_pylist():
                    index = row.get("episode_index")
                    if type(index) is not int or index < 0:
                        raise ValueError("Invalid frame episode_index")
                    frame_index = row.get("frame_index")
                    if type(frame_index) is not int or frame_index != counts.get(
                        index, 0
                    ):
                        raise ValueError(
                            f"Episode {index} has non-contiguous frame_index."
                        )
                    actual_index = row.get("index")
                    if type(actual_index) is not int or actual_index != global_index:
                        raise ValueError(
                            f"Expected global frame index {global_index}, got {actual_index!r}."
                        )
                    previous = final_indices.get(index)
                    if previous is not None and previous + 1 != global_index:
                        raise ValueError(
                            f"Episode {index} does not occupy a contiguous global frame range."
                        )
                    counts[index] = counts.get(index, 0) + 1
                    first_indices.setdefault(index, global_index)
                    final_indices[index] = global_index
                    global_index += 1
        except Exception as error:
            errors.append(f"Unreadable frame parquet {path.relative_to(root)}: {error}")
    episodes: dict[int, dict[str, Any]] = {}
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        try:
            for row in parquet.read_table(path).to_pylist():
                index = int(row["episode_index"])
                if index in episodes:
                    errors.append(f"Duplicate SDK episode metadata for index {index}.")
                episodes[index] = row
                if counts.get(index, 0):
                    begin, end = row.get("dataset_from_index"), row.get(
                        "dataset_to_index"
                    )
                    if (
                        type(begin) is not int
                        or type(end) is not int
                        or begin != first_indices[index]
                        or end != final_indices[index] + 1
                        or end - begin != counts[index]
                    ):
                        errors.append(
                            f"SDK episode {index} range conflicts with its frame data."
                        )
        except Exception as error:
            errors.append(
                f"Unreadable episode parquet {path.relative_to(root)}: {error}"
            )
    try:
        info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
        if not isinstance(info, dict):
            raise ValueError("info.json is not a mapping")
        if type(info.get("total_episodes")) is not int or info["total_episodes"] < 0:
            raise ValueError("info.json total_episodes must be a non-negative integer")
        if not isinstance(info.get("features"), dict):
            raise ValueError("info.json features must be a mapping")
    except (OSError, ValueError) as error:
        info = {}
        errors.append(f"Invalid LeRobot metadata: {error}")
    return counts, episodes, info, errors


def _decoded_video(
    path: Path, cache: dict[Path, tuple[int, float] | str]
) -> tuple[int, float]:
    """Decode each shard once, retaining only count/FPS or its failure."""
    cached = cache.get(path)
    if isinstance(cached, str):
        raise ValueError(cached)
    if cached is not None:
        return cached
    try:
        import av

        with av.open(str(path)) as container:
            stream = next(iter(container.streams.video), None)
            if stream is None or stream.average_rate is None:
                raise ValueError("Missing video stream or FPS")
            fps = float(stream.average_rate)
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError("Invalid video FPS")
            count = 0
            for frame in container.decode(stream):
                if frame.time is None or not math.isclose(
                    frame.time, count / fps, rel_tol=0, abs_tol=1e-4
                ):
                    raise ValueError(
                        "Video presentation timestamps are not contiguous from zero"
                    )
                count += 1
            if count == 0:
                raise ValueError("Video contains no decodable frames")
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        cache[path] = f"{type(error).__name__}: {error}"
        raise ValueError(cache[path]) from error
    cache[path] = count, fps
    return count, fps


def _rgb_evidence(
    root: Path,
    episode: Mapping[str, Any],
    info: Mapping[str, Any],
    video_cache: dict[Path, tuple[int, float] | str],
) -> list[str]:
    issues: list[str] = []
    for name, feature in info.get("features", {}).items():
        if not isinstance(feature, Mapping):
            issues.append(f"Invalid SDK feature metadata for {name!r}.")
            continue
        if feature.get("dtype") != "video":
            continue
        try:
            relative = info["video_path"].format(
                video_key=name,
                chunk_index=int(episode[f"videos/{name}/chunk_index"]),
                file_index=int(episode[f"videos/{name}/file_index"]),
            )
            path = _local_file(root, relative)
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError("missing/empty video")
            fps = float(info["fps"])
            length = int(episode["length"])
            begin = float(episode[f"videos/{name}/from_timestamp"])
            end = float(episode[f"videos/{name}/to_timestamp"])
            if (
                not math.isfinite(fps)
                or fps <= 0
                or not math.isfinite(begin)
                or not math.isfinite(end)
                or begin < 0
                or end <= begin
                or not math.isclose(end - begin, length / fps, rel_tol=0, abs_tol=1e-4)
                or not math.isclose(
                    begin, round(begin * fps) / fps, rel_tol=0, abs_tol=1e-4
                )
            ):
                raise ValueError("Video span does not match episode cadence/length")
            count, rate = _decoded_video(path, video_cache)
            if not math.isclose(rate, fps, rel_tol=1e-6) or count < round(end * fps):
                raise ValueError(
                    "Decoded video does not cover the episode span at its FPS"
                )
        except (OSError, ValueError, KeyError, TypeError) as error:
            issues.append(f"RGB artifact {name!r} incomplete: {error}")
    return issues


def _depth_evidence(
    root: Path,
    record: Mapping[str, Any],
    video_cache: dict[Path, tuple[int, float] | str],
) -> list[str]:
    sensors = record.get("depth_sensors", [])
    if not sensors:
        return []
    try:
        metadata = json.loads((root / "depth_meta.json").read_text(encoding="utf-8"))
        episode_index = str(record["sidecar"]["lerobot_episode_index"])
        expected_length = int(record["sidecar"]["length"])
        for sensor in sensors:
            episode = metadata["sensors"][sensor]["episodes"][episode_index]
            path = _local_file(root, episode["file"])
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"missing depth video for sensor {sensor!r}")
            if int(episode["frame_count"]) != expected_length:
                raise ValueError(f"depth frame count mismatch for sensor {sensor!r}")
            count, rate = _decoded_video(path, video_cache)
            if count != expected_length or not math.isclose(
                rate, float(metadata["fps"]), rel_tol=1e-6
            ):
                raise ValueError(
                    f"decoded depth frame count/FPS mismatch for sensor {sensor!r}"
                )
    except (OSError, ValueError, KeyError, TypeError) as error:
        return [f"Depth artifacts incomplete: {error}"]
    return []


def _replay_evidence(root: Path, record: Mapping[str, Any]) -> list[str]:
    sidecar = record["sidecar"]
    replay = sidecar.get("replay_artifact")
    if replay is None:
        return []
    try:
        # Explicit external trajectory directories are permitted. They are
        # inspected only, never deserialized, rewritten or removed by recovery.
        name = replay["path"]
        if not isinstance(name, str) or not name or "\x00" in name:
            raise ValueError("invalid replay path")
        path = Path(name)
        if path.is_absolute():
            if replay.get("external") is not True:
                raise ValueError("absolute replay path requires external=true")
        else:
            path = _local_file(root, name)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError("missing/empty replay artifact")
        source = sidecar.get("parent_episode_uuid", sidecar["episode_uuid"])
        if replay.get("source_episode_uuid") != source:
            raise ValueError("replay source episode identity mismatch")
        if replay.get("initial_state_step") != 0:
            raise ValueError("replay initial state must refer to source episode step 0")
    except (OSError, ValueError, TypeError, KeyError) as error:
        return [f"Replay artifact incomplete: {error}"]
    return []


class RecordingJournal:
    """One atomic, fsynced commit record per stable episode UUID.

    Args:
        root: Dataset directory. A recorder's single writer owns mutations;
            offline inspection/repair must run after that writer has stopped.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, episode_uuid: str) -> Path:
        return _local_file(
            self.root, (_JOURNAL_PATH / f"{UUID(episode_uuid)}.json").as_posix()
        )

    def get(self, episode_uuid: str) -> dict[str, Any] | None:
        """Read a persisted commit record.

        Args:
            episode_uuid: Canonical episode identity, never a filesystem path.

        Returns:
            Commit evidence, or ``None`` if no write has been prepared.
        """
        path = self._path(episode_uuid)
        if not path.exists():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or record.get("schema_version") != 1:
            raise ValueError(f"Invalid recording journal schema in {path}.")
        if record.get("episode_uuid") != str(UUID(episode_uuid)):
            raise ValueError(f"Journal identity mismatch in {path}.")
        if record.get("phase") not in _PHASES:
            raise ValueError(f"Invalid journal phase in {path}.")
        if not isinstance(record.get("sidecar"), dict):
            raise ValueError(f"Invalid journal sidecar in {path}.")
        if record["sidecar"].get("episode_uuid") != record["episode_uuid"]:
            raise ValueError(f"Journal sidecar identity mismatch in {path}.")
        sensors = record.get("depth_sensors")
        if not isinstance(sensors, list) or not all(
            isinstance(item, str) for item in sensors
        ):
            raise ValueError(f"Invalid journal depth sensor list in {path}.")
        return record

    def prepare(
        self, sidecar: Mapping[str, Any], depth_sensors: list[str] | None = None
    ) -> None:
        """Persist intended sidecar before submitting any episode frames.

        Args:
            sidecar: Complete target episode metadata with ``episode_uuid``.
            depth_sensors: Depth streams that must exist before repair succeeds.

        Raises:
            RuntimeError: If this identity already crossed the SDK commit boundary.
        """
        identity = str(UUID(sidecar["episode_uuid"]))
        existing = self.get(identity)
        if existing is not None and existing["phase"] != "prepared":
            raise RuntimeError(
                f"Episode {identity} already has phase {existing['phase']}; "
                "refusing a duplicate SDK write. Inspect/recover the recording."
            )
        record = {
            "schema_version": 1,
            "episode_uuid": identity,
            "phase": "prepared",
            "sidecar": copy.deepcopy(dict(sidecar)),
            "depth_sensors": list(depth_sensors or []),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(self._path(identity), record)

    def advance(self, episode_uuid: str, phase: str) -> None:
        """Durably advance a commit phase without moving backwards.

        Args:
            episode_uuid: Prepared episode identity.
            phase: Next phase in the commit sequence.
        """
        record = self.get(episode_uuid)
        if record is None or phase not in _PHASES:
            raise ValueError("A prepared journal record and valid phase are required.")
        previous_index = _PHASES.index(record["phase"])
        next_index = _PHASES.index(phase)
        if next_index < previous_index or next_index > previous_index + 1:
            raise ValueError(f"Invalid journal transition {record['phase']} → {phase}.")
        record["phase"] = phase
        record.pop("error", None)
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(self._path(episode_uuid), record)

    def record_error(self, episode_uuid: str, error: BaseException) -> None:
        """Keep the last confirmed phase and persist the error.

        Args:
            episode_uuid: Prepared episode identity.
            error: Exception raised while committing this episode.
        """
        record = self.get(episode_uuid)
        if record is None:
            return
        record["error"] = f"{type(error).__name__}: {error}"
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(self._path(episode_uuid), record)


def inspect_recording(root: str | Path) -> dict[str, Any]:
    """Diagnose persisted phases against frame/depth/episode-sidecar evidence.

    Args:
        root: Stopped recording's dataset directory.

    Returns:
        ``ok``, global ``errors``, and per-episode ``commits``. ``prepared``
        without frames is safely retryable; ``lerobot_committing`` without
        matching frames remains unknown. Even ``complete`` records must have
        matching frame identities and contiguous SDK ranges on disk: SDK
        buffering is not crash durability. Referenced videos are decoded with
        PyAV; unreadable or incomplete media stays unresolved.
    """
    root = Path(root)
    try:
        sidecars, sidecar_indices, errors = _sidecars(root)
    except (OSError, ValueError) as error:
        sidecars, sidecar_indices, errors = (
            {},
            {},
            [f"Invalid episode sidecar: {error}"],
        )
    counts, episodes, info, parquet_errors = _episode_evidence(root)
    errors.extend(parquet_errors)
    commits: list[dict[str, Any]] = []
    video_cache: dict[Path, tuple[int, float] | str] = {}
    journal = RecordingJournal(root)
    for path in sorted((root / _JOURNAL_PATH).glob("*.json")):
        try:
            record = journal.get(path.stem)
            if record is None:
                continue
            sidecar = record["sidecar"]
            index = int(sidecar["lerobot_episode_index"])
            expected = int(sidecar["length"])
            frame_count = counts.get(index, 0)
            identity = record["episode_uuid"]
            issues = _depth_evidence(root, record, video_cache)
            issues.extend(_replay_evidence(root, record))
            sdk_episode = episodes.get(index, {})
            if int(sdk_episode.get("length", -1)) != expected:
                issues.append(
                    "SDK episode metadata is missing or has a different length."
                )
            if index >= int(info.get("total_episodes", 0)):
                issues.append("SDK info.json does not include this episode.")
            issues.extend(_rgb_evidence(root, sdk_episode, info, video_cache))
            matching_frames = expected > 0 and frame_count == expected
            if frame_count != expected:
                issues.append(f"Expected {expected} frames; found {frame_count}.")
            sidecar_exists = identity in sidecars
            if sidecar_exists and sidecars[identity] != sidecar:
                issues.append("Episode sidecar conflicts with journal metadata.")
            existing = sidecar_indices.get(index)
            if existing is not None and existing.get("episode_uuid") != identity:
                issues.append("Episode sidecar index is owned by a different identity.")
            phase = record["phase"]
            if phase == "prepared" and frame_count == 0 and not parquet_errors:
                status = "retryable"
            elif phase == "prepared" or not matching_frames or issues or parquet_errors:
                status = "unresolved"
            elif sidecar_exists:
                status = "complete"
            else:
                status = "repairable"
            commits.append(
                {
                    "episode_uuid": identity,
                    "lerobot_episode_index": index,
                    "phase": phase,
                    "status": status,
                    "sidecar_exists": sidecar_exists,
                    "issues": issues,
                    "error": record.get("error"),
                }
            )
        except (ValueError, KeyError, TypeError, OSError) as error:
            errors.append(f"Journal {path.name}: {error}")
    return {
        "ok": not errors and all(item["status"] == "complete" for item in commits),
        "root": str(root.resolve()),
        "errors": errors,
        "commits": commits,
    }


def recover_recording(root: str | Path, *, repair: bool = False) -> dict[str, Any]:
    """Inspect or repair missing episode sidecars using committed disk evidence.

    Args:
        root: Stopped recording's dataset directory.
        repair: Explicitly permit sidecar append and journal completion. The
            default is read-only. Stop all recorder processes before repair.

    Returns:
        Inspection report with ``repaired`` UUIDs. Incomplete or conflicting
        frames/media, malformed JSONL and unknown SDK writes stay unresolved.
        No Parquet, video, replay artifact, or SDK metadata is rewritten.
    """
    root = Path(root)
    report = inspect_recording(root)
    repaired: list[str] = []
    if repair and not report["errors"]:
        journal = RecordingJournal(root)
        for item in report["commits"]:
            if item["status"] not in {"repairable", "complete"}:
                continue
            identity = item["episode_uuid"]
            record = journal.get(identity)
            if record is None:
                continue
            if item["status"] == "repairable":
                sidecar_path = _local_file(root, _SIDECAR_PATH.as_posix())
                sidecar_path.parent.mkdir(parents=True, exist_ok=True)
                # A failed append can leave a truncated last line. Never append
                # another record to a malformed stream; inspection blocks that.
                with sidecar_path.open("a", encoding="utf-8") as stream:
                    json.dump(
                        record["sidecar"], stream, ensure_ascii=False, allow_nan=False
                    )
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                repaired.append(identity)
            while record["phase"] != "complete":
                next_phase = _PHASES[_PHASES.index(record["phase"]) + 1]
                journal.advance(identity, next_phase)
                record = journal.get(identity)
        report = inspect_recording(root)
    report["repaired"] = repaired
    return report
