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


def _sidecars(root: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    records: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    path = _local_file(root, _SIDECAR_PATH.as_posix())
    if not path.exists():
        return records, errors
    contents = path.read_text(encoding="utf-8")
    if contents and not contents.endswith("\n"):
        errors.append("Episode sidecar has an unterminated final record.")
    for number, line in enumerate(contents.splitlines(), 1):
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("record is not a mapping")
            identity = record.get("episode_uuid")
            if isinstance(identity, str):
                if identity in records:
                    errors.append(f"Duplicate sidecar episode_uuid {identity}.")
                records[identity] = record
        except (ValueError, TypeError) as error:
            errors.append(f"Sidecar line {number}: {error}")
    return records, errors


def _episode_evidence(
    root: Path,
) -> tuple[dict[int, int], dict[int, dict[str, Any]], dict[str, Any], list[str]]:
    import pyarrow.compute as compute
    import pyarrow.parquet as parquet

    counts: dict[int, int] = {}
    errors: list[str] = []
    for path in sorted((root / "data").rglob("*.parquet")):
        try:
            table = parquet.read_table(path, columns=["episode_index"])
            for item in compute.value_counts(table.column("episode_index")).to_pylist():
                index = int(item["values"])
                counts[index] = counts.get(index, 0) + int(item["counts"])
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


def _rgb_evidence(
    root: Path, episode: Mapping[str, Any], info: Mapping[str, Any]
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
        except (OSError, ValueError, KeyError, TypeError) as error:
            issues.append(f"RGB artifact {name!r} incomplete: {error}")
    return issues


def _depth_evidence(root: Path, record: Mapping[str, Any]) -> list[str]:
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
        matching frame counts on disk: SDK buffering is not crash durability.
    """
    root = Path(root)
    try:
        sidecars, errors = _sidecars(root)
    except (OSError, ValueError) as error:
        sidecars, errors = {}, [f"Invalid episode sidecar: {error}"]
    counts, episodes, info, parquet_errors = _episode_evidence(root)
    errors.extend(parquet_errors)
    commits: list[dict[str, Any]] = []
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
            issues = _depth_evidence(root, record)
            issues.extend(_replay_evidence(root, record))
            sdk_episode = episodes.get(index, {})
            if int(sdk_episode.get("length", -1)) != expected:
                issues.append(
                    "SDK episode metadata is missing or has a different length."
                )
            if index >= int(info.get("total_episodes", 0)):
                issues.append("SDK info.json does not include this episode.")
            issues.extend(_rgb_evidence(root, sdk_episode, info))
            matching_frames = expected > 0 and frame_count == expected
            if frame_count != expected:
                issues.append(f"Expected {expected} frames; found {frame_count}.")
            sidecar_exists = identity in sidecars
            if sidecar_exists and sidecars[identity] != sidecar:
                issues.append("Episode sidecar conflicts with journal metadata.")
            phase = record["phase"]
            if phase == "prepared" and frame_count == 0 and not parquet_errors:
                status = "retryable"
            elif phase == "prepared" or not matching_frames or issues:
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
        frames/depth, malformed JSONL and unknown SDK writes stay unresolved.
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
