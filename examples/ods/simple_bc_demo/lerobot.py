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

"""Private numeric LeRobot transport for the PickPlace learning experiment.

The official SDK writes and reads the dataset. After integrity validation the
learner samples resident CPU tensors, preserving the existing causal-history
sampler. This measures offline learning behavior and startup loading cost;
it does not measure a LeRobot data loader's per-minibatch throughput.
"""

from __future__ import annotations

import copy
import ctypes
import hashlib
from importlib.metadata import version
import json
import math
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from embodichain.data_pipeline.datasets import LeRobotDataSource, LeRobotFieldMappingCfg

from examples.ods.simple_bc_demo.data import (
    _content_manifest,
    _json_bytes,
    _validate_tensors,
)

__all__ = ["DirectLeRobotWriter", "load_lerobot_corpus"]

_FORMAT = "ods-pick-place-lerobot-corpus-v1"
_SIDECAR = "meta/embodichain_ods_corpus.json"


def _validate_fps(schema: dict[str, Any], fps: int) -> None:
    if type(fps) is not int or fps < 1:
        raise ValueError("fps must be a positive integer")
    step_dt = schema.get("action", {}).get("step_dt")
    if step_dt is not None and (
        not isinstance(step_dt, (float, int))
        or isinstance(step_dt, bool)
        or not math.isfinite(step_dt)
        or not math.isclose(step_dt, 1.0 / fps, rel_tol=1e-6, abs_tol=1e-9)
    ):
        raise ValueError("LeRobot fps must match the policy action step_dt")


def _publish_without_overwrite(source: Path, destination: Path) -> None:
    """Atomically rename on Linux without replacing even an empty directory."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise RuntimeError("Direct writer atomic publication requires Linux renameat2")
    rename.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    rename.restype = ctypes.c_int
    # AT_FDCWD=-100, RENAME_NOREPLACE=1. Never fall back to replacing rename().
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


class DirectLeRobotWriter:
    """Stream immutable CPU episode rows to the SDK through a bounded queue.

    One worker thread owns all SDK operations. At most ``queue_size + 2``
    numeric rows are retained by this adapter (queued, writing and submitting),
    apart from the SDK's own buffers. Hashes include zeroed invalid padding and
    use the same canonical manifest as the SDK reader. Metadata grows with episode count.
    The creator thread alone may submit/finalize/abort. Progress callbacks run
    on that thread while backpressured, allowing producer events to be drained.

    Finalization publishes atomically without an extra SDK read-back. The
    learner's subsequent ``load_lerobot_corpus`` validates the published data.
    Failures retain a hidden temporary directory marked ``FAILED.json``; they
    never publish a usable destination. Queue handoff and finalization have a
    finite operation deadline; abort has a separate shutdown deadline. SDK calls
    cannot be forcibly interrupted, so timeout cleanup reports an unjoined daemon
    thread explicitly rather than preventing process exit.
    """

    def __init__(
        self,
        destination: Path | str,
        *,
        schema: dict[str, Any],
        episode_count: int,
        max_episode_steps: int = 600,
        repo_id: str = "local/ods-pick-place",
        fps: int = 30,
        queue_size: int = 2,
        provenance: dict[str, Any] | None = None,
        operation_timeout: float = 1800.0,
        shutdown_timeout: float = 30.0,
    ) -> None:
        for name, value in (
            ("episode_count", episode_count),
            ("max_episode_steps", max_episode_steps),
            ("queue_size", queue_size),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            ("operation_timeout", operation_timeout),
            ("shutdown_timeout", shutdown_timeout),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        self.operation_timeout = float(operation_timeout)
        self.shutdown_timeout = float(shutdown_timeout)
        _validate_fps(schema, fps)
        # Validate ordered schema before starting a thread or creating artifacts.
        for name in ("observation", "action"):
            names = schema.get(name, {}).get("names")
            if (
                not isinstance(names, list)
                or not names
                or not all(isinstance(n, str) and n for n in names)
                or len(set(names)) != len(names)
            ):
                raise ValueError(
                    f"Schema {name}.names must contain ordered unique names"
                )
        self.schema = json.loads(_json_bytes(schema))
        self.provenance = json.loads(_json_bytes(provenance or {}))
        self.destination = Path(destination)
        if self.destination.exists():
            raise FileExistsError(self.destination)
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{self.destination.name}-", dir=self.destination.parent
            )
        )
        self.staging = self.temporary / "dataset"
        self.episode_count, self.max_episode_steps = episode_count, max_episode_steps
        self.repo_id, self.fps = repo_id, fps
        self._owner = threading.get_ident()
        self._queue: queue.Queue[dict[str, torch.Tensor] | None] = queue.Queue(
            queue_size
        )
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._submitted = 0
        self._identities: set[tuple[int, int]] = set()
        self._closed = False
        self._sidecar: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._stats: dict[str, Any] = {
            "submitted_rows": 0,
            "accepted_rows": 0,
            "valid_frames": 0,
            "queue_capacity_rows": queue_size,
            "max_queued_rows": 0,
            "queue_wait_seconds": 0.0,
            "clone_validation_seconds": 0.0,
            "sdk_create_seconds": 0.0,
            "persist_seconds": 0.0,
            "hash_seconds": 0.0,
            "sdk_finalize_seconds": 0.0,
            "publish_seconds": 0.0,
            "sidecar_seconds": 0.0,
            "finalize_wait_seconds": 0.0,
            "published": False,
            "worker_joined": False,
            "operation_timeout_seconds": self.operation_timeout,
            "shutdown_timeout_seconds": self.shutdown_timeout,
            "scope": "writer CPU wall components; overlapping costs are not additive; "
            "excludes generation, training and audited learner load",
        }
        self._thread = threading.Thread(
            target=self._run, name="direct-lerobot-writer", daemon=True
        )
        self._thread.start()

    @property
    def stats(self) -> dict[str, Any]:
        """Return a consistent snapshot of measured writer-local costs."""
        with self._lock:
            return copy.deepcopy(self._stats)

    def _add(self, name: str, value: float | int) -> None:
        with self._lock:
            self._stats[name] += value

    def _check(self) -> None:
        if threading.get_ident() != self._owner:
            raise RuntimeError("DirectLeRobotWriter requires its creator thread")
        if self._error is not None:
            raise RuntimeError("Direct LeRobot writer failed") from self._error
        if self._closed:
            raise RuntimeError("Direct LeRobot writer is closed")

    def _put(
        self,
        item: dict[str, torch.Tensor] | None,
        progress: Callable[[], None] | None,
        *,
        deadline: float | None = None,
    ) -> None:
        started = time.monotonic()
        deadline = started + self.operation_timeout if deadline is None else deadline
        try:
            while True:
                self._check()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Direct LeRobot queue handoff timed out")
                try:
                    self._queue.put(item, timeout=min(0.05, remaining))
                    break
                except queue.Full:
                    if progress is not None:
                        progress()
        except BaseException as error:
            self._fail(error)
            raise
        finally:
            self._add("queue_wait_seconds", time.monotonic() - started)
        with self._lock:
            self._stats["max_queued_rows"] = max(
                self._stats["max_queued_rows"], self._queue.qsize()
            )

    def submit(
        self,
        row: dict[str, torch.Tensor],
        *,
        progress: Callable[[], None] | None = None,
    ) -> None:
        """Clone one CPU row; block only when the bounded queue is full.

        Required fields are observations[T,O], actions[T,A], valid[T],
        frame_ids[T,3] and generations[T], with T=max_episode_steps.
        Each submitted row must contain at least one valid frame.
        """
        self._check()
        if self._submitted >= self.episode_count:
            raise ValueError("More rows submitted than episode_count")
        started = time.monotonic()
        if set(row) != {"observations", "actions", "valid", "frame_ids", "generations"}:
            raise ValueError("Direct writer requires all five numeric fields")
        if not all(
            isinstance(v, torch.Tensor) and v.device.type == "cpu" for v in row.values()
        ):
            raise ValueError("Direct writer requires CPU tensors")
        owned = {key: value.detach().contiguous().clone() for key, value in row.items()}
        batched = {key: value.unsqueeze(0) for key, value in owned.items()}
        _validate_tensors(batched, self.schema)
        if owned["valid"].shape != (self.max_episode_steps,):
            raise ValueError("Row length must equal max_episode_steps")
        mask = owned["valid"]
        identity = tuple(owned["frame_ids"][mask][0, :2].tolist())
        if identity in self._identities:
            raise ValueError("Duplicate source episode identity")
        for key in set(owned) - {"valid"}:
            owned[key][~mask] = 0
        self._add("clone_validation_seconds", time.monotonic() - started)
        self._put(owned, progress)
        self._identities.add(identity)
        self._submitted += 1
        self._add("submitted_rows", 1)

    def _run(self) -> None:
        dataset = None
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            features = {
                "observation.state": {
                    "dtype": "float32",
                    "shape": (len(self.schema["observation"]["names"]),),
                    "names": self.schema["observation"]["names"],
                },
                "action": {
                    "dtype": "float32",
                    "shape": (len(self.schema["action"]["names"]),),
                    "names": self.schema["action"]["names"],
                },
                "ods.provenance": {
                    "dtype": "int64",
                    "shape": (4,),
                    "names": [
                        "run_id",
                        "episode_id",
                        "source_frame_index",
                        "generation",
                    ],
                },
            }
            started = time.monotonic()
            dataset = LeRobotDataset.create(
                repo_id=self.repo_id,
                fps=self.fps,
                features=features,
                root=self.staging,
                robot_type="franka_panda",
                use_videos=False,
            )
            self._add("sdk_create_seconds", time.monotonic() - started)
            digests: dict[str, Any] = {}
            specs: dict[str, Any] = {}
            episodes: list[dict[str, Any]] = []
            while not self._stop.is_set():
                try:
                    row = self._queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if row is None:
                    break
                started = time.monotonic()
                manifest = _content_manifest(
                    {key: value.unsqueeze(0) for key, value in row.items()}, self.schema
                )
                episodes.extend(manifest["episodes"])
                for key, value in sorted(row.items()):
                    if key not in digests:
                        digests[key] = hashlib.sha256()
                        specs[key] = {
                            "shape": [self.episode_count, *value.shape],
                            "dtype": str(value.dtype),
                        }
                    digests[key].update(value.numpy().tobytes())
                self._add("hash_seconds", time.monotonic() - started)
                started = time.monotonic()
                steps = row["valid"].nonzero().flatten().tolist()
                for step in steps:
                    if self._stop.is_set():
                        raise RuntimeError("Direct LeRobot writing aborted")
                    provenance = torch.cat(
                        (row["frame_ids"][step], row["generations"][step].reshape(1))
                    )
                    dataset.add_frame(
                        {
                            "observation.state": row["observations"][step].clone(),
                            "action": row["actions"][step].clone(),
                            "ods.provenance": provenance,
                            "task": "Pick up the cube and place it in the fixed goal region.",
                        }
                    )
                dataset.save_episode()
                self._add("persist_seconds", time.monotonic() - started)
                self._add("accepted_rows", 1)
                self._add("valid_frames", len(steps))
                del row
            if self._stop.is_set():
                raise RuntimeError("Direct LeRobot writing aborted")
            if len(episodes) != self.episode_count:
                raise ValueError("Direct writer episode count mismatch")
            started = time.monotonic()
            finalizing, dataset = dataset, None
            finalizing.finalize()
            self._add("sdk_finalize_seconds", time.monotonic() - started)
            schema_hash = hashlib.sha256(_json_bytes(self.schema)).hexdigest()
            source_manifest = {
                "format": "ods-pick-place-cpu-corpus-v1",
                "schema": self.schema,
                "provenance": self.provenance,
                "schema_sha256": schema_hash,
                "content_id": hashlib.sha256(
                    _json_bytes({"schema_sha256": schema_hash, "episodes": episodes})
                ).hexdigest(),
                "episodes": episodes,
                "tensors": {
                    key: {**spec, "sha256": digests[key].hexdigest()}
                    for key, spec in specs.items()
                },
            }
            self._sidecar = {
                "format": _FORMAT,
                "repo_id": self.repo_id,
                "fps": self.fps,
                "lerobot_version": version("lerobot"),
                "source_manifest": source_manifest,
                "sampling": "resident_cpu_after_official_lerobot_row_load",
                "per_minibatch_lerobot_io": False,
                "direct_streaming": True,
                "prepublication_readback": False,
            }
            started = time.monotonic()
            (self.staging / _SIDECAR).write_text(
                json.dumps(self._sidecar, indent=2, allow_nan=False) + "\n"
            )
            self._add("sidecar_seconds", time.monotonic() - started)
        except BaseException as error:
            self._fail(error)
        finally:
            if dataset is not None:
                try:
                    dataset.finalize()
                except BaseException as cleanup_error:
                    with self._lock:
                        self._stats["cleanup_error"] = repr(cleanup_error)
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

    def _fail(self, error: BaseException) -> None:
        with self._lock:
            if self._error is not None:
                return
            self._error = error
            self._stats["error"] = repr(error)
        self._stop.set()
        # Retain partial output for diagnosis; never mask the primary error.
        try:
            (self.temporary / "FAILED.json").write_text(
                json.dumps({"error_type": type(error).__name__, "error": str(error)})
                + "\n"
            )
        except OSError:
            pass

    def _join(
        self, progress: Callable[[], None] | None = None, *, deadline: float
    ) -> None:
        try:
            while self._thread.is_alive():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "Direct LeRobot worker did not stop before deadline"
                    )
                self._thread.join(timeout=min(0.05, remaining))
                if progress is not None and self._thread.is_alive():
                    progress()
        except BaseException as error:
            self._fail(error)
            raise
        finally:
            with self._lock:
                self._stats["worker_joined"] = not self._thread.is_alive()

    def finalize(self, *, progress: Callable[[], None] | None = None) -> dict[str, Any]:
        """Drain all rows, finalize SDK metadata, join and return the sidecar."""
        self._check()
        if self._submitted != self.episode_count:
            raise ValueError("Submitted row count differs from episode_count")
        started = time.monotonic()
        deadline = started + self.operation_timeout
        try:
            self._put(None, progress, deadline=deadline)
            self._join(progress, deadline=deadline)
        finally:
            self._add("finalize_wait_seconds", time.monotonic() - started)
        self._check()
        # Only the owner can publish, after all progress callbacks succeeded.
        started = time.monotonic()
        try:
            _publish_without_overwrite(self.staging, self.destination)
        except BaseException as error:
            self._fail(error)
            self._check()
        self._add("publish_seconds", time.monotonic() - started)
        with self._lock:
            self._stats["published"] = True
        shutil.rmtree(self.temporary, ignore_errors=True)
        self._closed = True
        assert self._sidecar is not None
        return copy.deepcopy(self._sidecar)

    def abort(self) -> None:
        """Stop accepting work and join within the configured shutdown deadline.

        A stuck SDK call leaves a daemon thread and raises ``TimeoutError``.
        Stats explicitly report unjoined cleanup; no destination can be published.
        """
        if threading.get_ident() != self._owner:
            raise RuntimeError("DirectLeRobotWriter requires its creator thread")
        self._stop.set()
        try:
            self._join(deadline=time.monotonic() + self.shutdown_timeout)
        except BaseException as cleanup_error:
            with self._lock:
                self._stats["cleanup_error"] = repr(cleanup_error)
            raise
        finally:
            if self._error is None and not self.stats["published"]:
                self._fail(RuntimeError("Direct LeRobot writing aborted"))
            self._closed = True

    def __enter__(self) -> DirectLeRobotWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if not self._closed:
            try:
                self.abort()
            except BaseException as cleanup_error:
                if exc is None:
                    raise
                exc.add_note(
                    f"Direct LeRobot cleanup failed: {cleanup_error!r}; "
                    f"worker_joined={self.stats['worker_joined']}"
                )


def load_lerobot_corpus(
    path: Path | str,
    *,
    expected_schema: dict[str, Any],
    read_batch_size: int = 1024,
    memory_limit_bytes: int = 512 * 1024 * 1024,
) -> LeRobotDataSource:
    """Audit showcase hashes around the reusable bounded numeric loader.

    The example owns its sidecar/content hashes; SDK loading, temporal metadata,
    resident storage and context sampling belong to the public data source.
    """
    path = Path(path)
    sidecar = json.loads((path / _SIDECAR).read_text())
    if sidecar.get("format") != _FORMAT:
        raise ValueError("Unsupported PickPlace LeRobot corpus format")
    manifest = sidecar["source_manifest"]
    if manifest.get("schema") != expected_schema:
        raise ValueError("LeRobot observation/action schema mismatch")
    _validate_fps(expected_schema, sidecar["fps"])
    specs = manifest["tensors"]
    if set(specs) - {"observations", "actions", "valid", "frame_ids", "generations"}:
        raise ValueError("Unexpected numeric field in LeRobot sidecar")
    rows, steps = specs["valid"]["shape"]
    source = LeRobotDataSource(
        path,
        schema=expected_schema,
        memory_limit_bytes=memory_limit_bytes,
        source_id=str(
            manifest.get("provenance", {}).get("source_run_id", manifest["content_id"])
        ),
        mapping=LeRobotFieldMappingCfg(provenance_key="ods.provenance"),
        repo_id=sidecar["repo_id"],
        max_episode_steps=steps,
        episode_frames=[episode["valid_steps"] for episode in manifest["episodes"]],
        read_batch_size=read_batch_size,
    )
    if source.data.batch_size != (rows, steps):
        raise ValueError("LeRobot numeric corpus shape differs from source manifest")
    if "generations" not in specs:
        if (source.data["generations"][source.data["valid"]] != 0).any():
            raise ValueError("Unexpected generation in LeRobot provenance")
        del source.data["generations"]
    tensors = {
        ("observations" if key == "obs" else key): value
        for key, value in source.data.items()
    }
    _validate_tensors(tensors, expected_schema)
    actual = _content_manifest(tensors, expected_schema)
    if any(manifest.get(key) != value for key, value in actual.items()):
        raise ValueError("LeRobot numeric content/hash mismatch")
    storage = source.metadata["storage"]
    storage["sampling"] = sidecar["sampling"]
    source.metadata = {**copy.deepcopy(manifest), "storage": storage}
    return source
