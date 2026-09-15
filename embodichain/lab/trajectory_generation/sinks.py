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

"""Synchronous, host-independent persistence of accepted expert episodes."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from embodichain.lab.sim.motion.expansion import (
    CommitReceipt,
    ExpertEpisode,
)

__all__ = ["LeRobotEpisodeSink"]

_NUMERIC_DTYPES = {torch.float32, torch.float64, torch.int32, torch.int64, torch.uint8}
_FEATURE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")


def _plain(value: Any) -> Any:
    """Convert immutable contract metadata to JSON-native containers."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [_plain(item) for item in sorted(value)]
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _atomic_bytes(path: Path, data: bytes) -> None:
    """Replace one required metadata file only after its complete write."""
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = stream.name
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


class LeRobotEpisodeSink:
    """Seal and read back one LeRobot dataset shard per accepted episode.

    Each shard contains exactly ``T`` causal training frames. Required sidecars
    retain the terminal observation, all ``T+1`` measured timestamps, candidate
    lineage, validation, phases, and user metadata. Only shards listed in
    ``manifest.json`` are committed. Every successful receipt follows writer
    finalization, dataset/media decoding, sidecar verification, and manifest
    replacement/readback. This is a synchronous, single-caller sink; ``drain``
    has no deferred receipts.

    Numeric observations must be nonempty vectors or matrices of float32,
    float64, int32, int64, or uint8. Matrices are flattened in C order into
    LeRobot vectors; ``observation_shapes`` in episode.json retains their source
    layout, and terminal.npz preserves the original shape. RGB observations
    must be uint8 arrays shaped ``(T+1,H,W,3)``.
    Feature names are preserved below ``observation.``; already prefixed names
    are retained. Unsupported fields are rejected before any episode write.
    Actions must be float32 or float64 vectors. Training timestamps are the
    nominal relative LeRobot clock; exact measured timestamps, including their
    origin, are authoritative in the required sidecar.

    One episode is the bounded shard unit. Retrying an identical commit reuses
    a readable sealed shard; incomplete uncommitted data is rebuilt at the same
    path. Changed payloads under the same commit ID are rejected. The sink owns
    a new/empty output directory and does not implement process restart or
    power-loss recovery. Do not concurrently modify its files.

    Args:
        root: New or empty output directory for the collection manifest/shards.
        fps: Integer control frequency. All measured intervals and relative
            timestamps must agree with this clock within ``timestamp_tolerance``.
        repo_id: Local LeRobot repository identity; nothing is uploaded.
        max_episode_bytes: Maximum raw tensor plus metadata bytes per episode.
        timestamp_tolerance: Absolute seconds allowed between the measured and
            nominal clocks; relative tolerance is zero.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        fps: int,
        repo_id: str = "embodichain/trajectory-generation",
        max_episode_bytes: int = 256 * 1024 * 1024,
        timestamp_tolerance: float = 1e-6,
    ) -> None:
        if type(fps) is not int or fps < 1:
            raise ValueError("fps must be a positive integer.")
        if type(max_episode_bytes) is not int or max_episode_bytes < 1:
            raise ValueError("max_episode_bytes must be a positive integer.")
        if not isinstance(repo_id, str) or not repo_id:
            raise ValueError("repo_id must be a nonempty string.")
        if (
            isinstance(timestamp_tolerance, bool)
            or not isinstance(timestamp_tolerance, (int, float))
            or not math.isfinite(timestamp_tolerance)
            or timestamp_tolerance < 0
        ):
            raise ValueError("timestamp_tolerance must be a finite nonnegative number.")
        self.root = Path(root).resolve()
        self.fps = fps
        self.repo_id = repo_id
        self.max_episode_bytes = max_episode_bytes
        self.timestamp_tolerance = float(timestamp_tolerance)
        self._closed = False
        self._fingerprints: dict[str, str] = {}
        self._committed: dict[str, dict[str, Any]] = {}
        self.root.mkdir(parents=True, exist_ok=True)
        if any(self.root.iterdir()):
            raise ValueError(
                "LeRobotEpisodeSink requires a new or empty root directory."
            )
        self._lock = self.root / ".writer.lock"
        self._lock.touch(exist_ok=False)

    def _prepare(self, episode: ExpertEpisode) -> tuple[ExpertEpisode, dict, dict, str]:
        """Validate, bound, and own the complete submitted evidence."""
        if not isinstance(episode, ExpertEpisode):
            raise TypeError("episode must be an ExpertEpisode.")
        if not episode.validation.accepted:
            raise ValueError("Only accepted episodes may enter the expert dataset.")
        metadata = {
            "identity": asdict(episode.identity),
            "episode_id": episode.episode_id,
            "commit_id": episode.commit_id,
            "action_representation": episode.action_representation,
            "validation": [
                {
                    "check_id": check.check_id,
                    "status": check.status,
                    "detail": check.detail,
                    "metrics": dict(check.metrics),
                }
                for check in episode.validation.checks
            ],
            "phases": [asdict(phase) for phase in episode.phases],
            "metadata": _plain(episode.metadata),
            "fps": self.fps,
            "steps": episode.actions.shape[0],
            "observation_features": {},
            "observation_shapes": {},
        }
        # Reject unknown layouts before the CPU ownership copy or filesystem work.
        features = {}
        for key, value in episode.observations.items():
            if not _FEATURE_NAME.fullmatch(key):
                raise ValueError(f"Unsupported observation feature name: {key!r}.")
            name = key if key.startswith("observation.") else f"observation.{key}"
            if name in features:
                raise ValueError(f"Observation feature names collide at {name!r}.")
            if (
                value.ndim in (2, 3)
                and min(value.shape[1:]) > 0
                and value.dtype in _NUMERIC_DTYPES
            ):
                features[name] = {
                    "dtype": str(value.dtype).removeprefix("torch."),
                    "shape": (math.prod(value.shape[1:]),),
                    "names": None,
                }
            elif (
                value.ndim == 4
                and value.dtype == torch.uint8
                and value.shape[-1] == 3
                and min(value.shape[1:3]) > 0
            ):
                features[name] = {
                    "dtype": "image",
                    "shape": (3, *value.shape[1:3]),
                    "names": ["channel", "height", "width"],
                }
            else:
                raise ValueError(
                    f"Unsupported observation shape/dtype for {key!r}: {tuple(value.shape)}, {value.dtype}."
                )
            metadata["observation_features"][key] = name
            metadata["observation_shapes"][key] = list(value.shape[1:])
        if episode.actions.dtype not in {torch.float32, torch.float64}:
            raise ValueError("Actions must use float32 or float64.")
        features["action"] = {
            "dtype": str(episode.actions.dtype).removeprefix("torch."),
            "shape": tuple(episode.actions.shape[1:]),
            "names": None,
        }
        metadata["features"] = features
        encoded = _json_bytes(metadata)
        tensors = (episode.actions, episode.timestamps, *episode.observations.values())
        byte_count = len(encoded) + sum(
            value.numel() * value.element_size() for value in tensors
        )
        if byte_count > self.max_episode_bytes:
            raise ValueError("Episode exceeds max_episode_bytes.")
        owned = ExpertEpisode(
            identity=episode.identity,
            observations={
                key: value.detach().cpu() for key, value in episode.observations.items()
            },
            actions=episode.actions.detach().cpu(),
            timestamps=episode.timestamps.detach().cpu(),
            action_representation=episode.action_representation,
            validation=episode.validation,
            episode_id=episode.episode_id,
            commit_id=episode.commit_id,
            phases=episode.phases,
            metadata=episode.metadata,
        )
        times = owned.timestamps.to(torch.float64)
        relative = times - times[0]
        nominal = torch.arange(times.numel(), dtype=torch.float64) / self.fps
        if not torch.allclose(
            relative, nominal, atol=self.timestamp_tolerance, rtol=0
        ) or not torch.allclose(
            times.diff(),
            torch.full_like(times[1:], 1 / self.fps),
            atol=self.timestamp_tolerance,
            rtol=0,
        ):
            raise ValueError(
                "Measured timestamps do not match the configured fixed control clock."
            )
        digest = hashlib.sha256(encoded)
        for name, value in [
            ("actions", owned.actions),
            ("timestamps", owned.timestamps),
            *sorted(owned.observations.items()),
        ]:
            digest.update(_json_bytes([name, str(value.dtype), list(value.shape)]))
            digest.update(value.contiguous().numpy().tobytes())
        return owned, features, json.loads(encoded), digest.hexdigest()

    def _write_dataset(
        self, path: Path, episode: ExpertEpisode, features: dict, metadata: dict
    ) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            root=path,
            fps=self.fps,
            features=features,
            use_videos=False,
            metadata_buffer_size=1,
            image_writer_threads=0,
            image_writer_processes=0,
        )
        try:
            for index, action in enumerate(episode.actions):
                frame = {
                    "action": action.numpy(),
                    "task": str(
                        episode.metadata.get("task", episode.identity.source_id)
                    ),
                }
                for key, value in episode.observations.items():
                    name = metadata["observation_features"][key]
                    sample = value[index]
                    if features[name]["dtype"] == "image":
                        sample = sample.permute(2, 0, 1)
                    else:
                        sample = sample.reshape(-1)
                    frame[name] = sample.numpy()
                dataset.add_frame(frame)
            # LeRobot 0.4.4 validates (1,) numeric arrays but serializes them as
            # scalar HF Values. Match that schema without NumPy 2 scalar casts.
            for name, feature in features.items():
                if feature["shape"] == (1,) and feature["dtype"] != "image":
                    dataset.episode_buffer[name] = [
                        value.reshape(-1)[0] for value in dataset.episode_buffer[name]
                    ]
            dataset.save_episode()
        finally:
            dataset.finalize()

    def _verify_dataset(
        self, path: Path, episode: ExpertEpisode, features: dict, metadata: dict
    ) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        import pyarrow.parquet as pq

        # Check local completeness before LeRobot's constructor can attempt a
        # Hub fallback for missing metadata. Footer reads also require sealing.
        required = [
            path / "meta/info.json",
            path / "meta/stats.json",
            path / "meta/tasks.parquet",
        ]
        data_files = list((path / "data").rglob("*.parquet"))
        episode_files = list((path / "meta/episodes").rglob("*.parquet"))
        if (
            not all(file.is_file() for file in required)
            or not data_files
            or not episode_files
        ):
            raise ValueError("LeRobot shard is not sealed and locally complete.")
        for file in [*data_files, *episode_files, required[-1]]:
            pq.ParquetFile(file).metadata
        dataset = LeRobotDataset(repo_id=self.repo_id, root=path, download_videos=False)
        if (
            dataset.num_episodes != 1
            or len(dataset) != len(episode.actions)
            or dataset.fps != self.fps
        ):
            raise ValueError(
                "LeRobot shard has incorrect episode, frame, or clock metadata."
            )
        # LeRobot's torch row transform constructs Python float lists using the
        # default float32 dtype. Read declared float64 columns from their sealed
        # Arrow representation to verify precision without that reader cast.
        precise = {
            name: []
            for name, feature in features.items()
            if feature["dtype"] == "float64"
        }
        if precise:
            for file in sorted(data_files):
                table = pq.read_table(file, columns=list(precise))
                for name in precise:
                    precise[name].extend(table[name].to_pylist())
        for index, action in enumerate(episode.actions):
            frame = dataset[index]  # Decodes every required image as well.
            observed_action = (
                (
                    torch.as_tensor(precise["action"][index], dtype=action.dtype)
                    if "action" in precise
                    else frame["action"]
                )
                .reshape(action.shape)
                .to(action.dtype)
            )
            if not torch.equal(observed_action, action):
                raise ValueError(
                    "LeRobot action readback differs from submitted commands."
                )
            if not torch.equal(
                frame["timestamp"],
                torch.tensor(index / self.fps, dtype=frame["timestamp"].dtype),
            ):
                raise ValueError(
                    "LeRobot frame timestamps do not match the fixed clock."
                )
            for key, value in episode.observations.items():
                name = metadata["observation_features"][key]
                current = frame[name]
                expected = value[index]
                if features[name]["dtype"] == "image":
                    current = (current * 255).round().to(torch.uint8).permute(1, 2, 0)
                else:
                    if name in precise:
                        current = torch.as_tensor(
                            precise[name][index], dtype=expected.dtype
                        )
                    current = current.reshape(expected.shape).to(expected.dtype)
                if not torch.equal(current, expected):
                    raise ValueError(
                        f"LeRobot observation readback differs for {key!r}."
                    )

    def _write_evidence(
        self, path: Path, episode: ExpertEpisode, metadata: dict
    ) -> None:
        _atomic_bytes(path / "episode.json", _json_bytes(metadata))
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path, prefix=".terminal.", delete=False
            ) as stream:
                temporary = stream.name
                arrays = {"timestamps": episode.timestamps.numpy()}
                arrays.update(
                    {
                        f"observation.{key}": value[-1].numpy()
                        for key, value in episode.observations.items()
                    }
                )
                np.savez(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path / "terminal.npz")
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)

    def _verify_evidence(
        self, path: Path, episode: ExpertEpisode, metadata: dict
    ) -> None:
        if json.loads((path / "episode.json").read_text()) != metadata:
            raise ValueError("Episode lineage/validation metadata readback differs.")
        expected = {"timestamps": episode.timestamps.numpy()}
        expected.update(
            {
                f"observation.{key}": value[-1].numpy()
                for key, value in episode.observations.items()
            }
        )
        with np.load(path / "terminal.npz", allow_pickle=False) as arrays:
            if set(arrays.files) != set(expected):
                raise ValueError("Terminal evidence is incomplete.")
            for key, value in expected.items():
                if (
                    arrays[key].dtype != value.dtype
                    or arrays[key].shape != value.shape
                    or not np.array_equal(arrays[key], value)
                ):
                    raise ValueError(f"Terminal evidence readback differs for {key!r}.")

    def _write_manifest(self, record: dict) -> None:
        updated = {**self._committed, record["commit_id"]: record}
        manifest = {
            "format": "embodichain.expert_shards",
            "fps": self.fps,
            "episodes": list(updated.values()),
        }
        _atomic_bytes(self.root / "manifest.json", _json_bytes(manifest))
        self._verify_manifest(updated)
        self._committed = updated

    def _verify_manifest(self, records: dict) -> None:
        expected = {
            "format": "embodichain.expert_shards",
            "fps": self.fps,
            "episodes": list(records.values()),
        }
        if json.loads((self.root / "manifest.json").read_text()) != expected:
            raise ValueError("Collection manifest readback differs.")

    def submit(
        self, episode: ExpertEpisode, *, submission_id: int = 0
    ) -> CommitReceipt:
        """Persist accepted evidence and return its final synchronous receipt.

        Invalid/unsupported input and changed payloads under an existing commit
        raise before episode writes. Persistence failures return an unconfirmed
        receipt. Retrying uses the same ``commit_id`` and a new submission ID.
        """
        if self._closed:
            raise RuntimeError("LeRobotEpisodeSink is closed.")
        if type(submission_id) is not int or submission_id < 0:
            raise ValueError("submission_id must be a nonnegative integer.")
        owned, features, metadata, fingerprint = self._prepare(episode)
        previous = self._fingerprints.setdefault(owned.commit_id, fingerprint)
        if previous != fingerprint:
            raise ValueError(
                "The same commit_id cannot identify a changed episode payload."
            )
        path = self.root / hashlib.sha256(owned.commit_id.encode()).hexdigest()
        dataset_path = path / "dataset"
        confirmed, error = False, ""
        try:
            if owned.commit_id in self._committed:
                self._verify_dataset(dataset_path, owned, features, metadata)
                self._verify_evidence(path, owned, metadata)
                self._verify_manifest(self._committed)
                return CommitReceipt(
                    episode_id=owned.episode_id,
                    candidate_id=owned.identity.candidate_id,
                    attempt_id=owned.identity.attempt_id,
                    storage_id=str(path),
                    commit_id=owned.commit_id,
                    scene_case_id=owned.identity.scene_case_id,
                    submission_id=submission_id,
                )
            path.mkdir(exist_ok=True)
            if dataset_path.exists():
                try:
                    self._verify_dataset(dataset_path, owned, features, metadata)
                except Exception:
                    if owned.commit_id in self._committed:
                        raise
                    shutil.rmtree(dataset_path)
            if not dataset_path.exists():
                self._write_dataset(dataset_path, owned, features, metadata)
            self._verify_dataset(dataset_path, owned, features, metadata)
            self._write_evidence(path, owned, metadata)
            self._verify_evidence(path, owned, metadata)
            self._write_manifest(
                {
                    "commit_id": owned.commit_id,
                    "episode_id": owned.episode_id,
                    "candidate_id": owned.identity.candidate_id,
                    "fingerprint": fingerprint,
                    "shard": path.name,
                }
            )
            confirmed = True
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"[:1024]
        return CommitReceipt(
            episode_id=owned.episode_id,
            candidate_id=owned.identity.candidate_id,
            attempt_id=owned.identity.attempt_id,
            storage_id=str(path),
            commit_id=owned.commit_id,
            scene_case_id=owned.identity.scene_case_id,
            confirmed=confirmed,
            error=error,
            submission_id=submission_id,
        )

    def drain(self) -> tuple[CommitReceipt, ...]:
        """Return no deferred work: each submit already returns its final receipt."""
        return ()

    def close(self) -> None:
        """Release the writer lease; all successful submissions are already sealed."""
        self._closed = True
        self._lock.unlink(missing_ok=True)

    def __enter__(self) -> LeRobotEpisodeSink:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
