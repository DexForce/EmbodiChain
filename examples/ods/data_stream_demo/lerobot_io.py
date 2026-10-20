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

"""Standard SDK recording and genuinely decoded, bounded LeRobot consumption.

The parent owns exposure reservations. Spawned DataLoader workers own their
SDK Dataset/CPU decoder and never receive a copy of the exposure ledger.
"""

from __future__ import annotations

import copy
import json
import math
import os
import queue
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, default_collate

from embodichain.utils import configclass

from .payload import numeric_episode_digest

__all__ = [
    "LeRobotReaderCfg",
    "StandardLeRobotWriter",
    "StandardLeRobotReader",
    "audit_video_quality",
]

_SIDECAR = "meta/ods_data_stream.json"
_PROVENANCE = "ods.provenance"


def _check_versions() -> dict[str, str]:
    found = {name: version(name) for name in ("lerobot", "torchcodec")}
    if found != {"lerobot": "0.4.4", "torchcodec": "0.5"}:
        raise RuntimeError(
            f"This protocol requires LeRobot0.4.4/TorchCodec0.5: {found}"
        )
    return found


def _feature_names(metadata: Mapping[str, Any]) -> list[str]:
    return [
        f"{'/'.join(feature['path'])}/{index}"
        for feature in metadata["state_features"]
        for index in range(feature["width"])
    ]


class StandardLeRobotWriter:
    """Write accepted CPU episodes through one bounded SDK-owner thread.

    Args:
        destination: New output directory; existing destinations are rejected.
        metadata: Canonical payload description.
        episode_count: Exact accepted count required before finalization.
        fps: Simulation observation frequency.
        queue_size: Queued raw episodes, excluding the active writer/submission.
        operation_timeout: Bound on handoff/finalization and worker join.

    Real video features use the pinned SDK's ordinary AV1 per-episode encoding
    path. A failed/partial corpus remains marked INCOMPLETE and is not readable
    through this adapter. The calling run must treat an unjoined native writer
    as a failure; Python cannot forcibly interrupt an in-process encoder.
    """

    def __init__(
        self,
        destination: Path | str,
        metadata: Mapping[str, Any],
        *,
        episode_count: int,
        fps: int = 30,
        queue_size: int = 2,
        operation_timeout: float = 1800.0,
    ) -> None:
        for name, value in (
            ("episode_count", episode_count),
            ("fps", fps),
            ("queue_size", queue_size),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(operation_timeout) or operation_timeout <= 0:
            raise ValueError("operation_timeout must be finite and positive")
        _check_versions()
        self.root = Path(destination)
        if self.root.exists():
            raise FileExistsError(self.root)
        self.root.mkdir(parents=True)
        (self.root / "INCOMPLETE").write_text("Accepted episodes are not finalized.\n")
        self.metadata = copy.deepcopy(dict(metadata))
        self.episode_count, self.fps = episode_count, fps
        self.operation_timeout = float(operation_timeout)
        self._owner = (os.getpid(), threading.get_ident())
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(queue_size)
        self._abort = threading.Event()
        self._error: BaseException | None = None
        self._closed = False
        self._submitted = 0
        self._identities: set[tuple[int, int]] = set()
        self._stats: dict[str, Any] = {
            "submitted_episodes": 0,
            "written_episodes": 0,
            "valid_frames": 0,
            "queue_capacity": queue_size,
            "peak_queued_episodes": 0,
            "queue_wait_seconds": 0.0,
            "sdk_create_seconds": 0.0,
            "sdk_write_encode_seconds": 0.0,
            "sdk_finalize_seconds": 0.0,
            "finalized": False,
            "worker_joined": False,
        }
        self._thread = threading.Thread(
            target=self._run, name="stream-lerobot-writer", daemon=True
        )
        self._thread.start()

    def _check(self) -> None:
        if self._owner != (os.getpid(), threading.get_ident()):
            raise RuntimeError("Writer operations require the creator process/thread")
        if self._error is not None:
            raise RuntimeError("Standard LeRobot recording failed") from self._error
        if self._closed:
            raise RuntimeError("Standard LeRobot writer is closed")

    @property
    def stats(self) -> dict[str, Any]:
        """Return detached counters; component wall times can overlap generation."""
        return copy.deepcopy(self._stats)

    def _put(
        self, item: dict[str, Any] | None, progress: Callable[[], None] | None
    ) -> None:
        started = time.monotonic()
        deadline = started + self.operation_timeout
        while True:
            self._check()
            if progress is not None:
                progress()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("LeRobot writer queue handoff timed out")
            try:
                self._queue.put(item, timeout=min(remaining, 0.05))
                break
            except queue.Full:
                continue
        self._stats["queue_wait_seconds"] += time.monotonic() - started
        self._stats["peak_queued_episodes"] = max(
            self._stats["peak_queued_episodes"], self._queue.qsize()
        )

    def submit(
        self, episode: dict[str, Any], *, progress: Callable[[], None] | None = None
    ) -> None:
        """Transfer an owned packed episode to the bounded writer queue.

        Args:
            episode: ``pack_episode`` output; caller must not mutate it afterwards.
            progress: Creator-thread cancellation/ODS-event-draining callback.
        """
        self._check()
        if self._submitted >= self.episode_count:
            raise ValueError("More accepted episodes than the declared corpus size")
        identity = tuple(int(v) for v in episode["frame_ids"][0, :2])
        if identity in self._identities:
            raise ValueError("Duplicate immutable source episode")
        self._put(episode, progress)
        self._identities.add(identity)
        self._submitted += 1
        self._stats["submitted_episodes"] = self._submitted

    def _run(self) -> None:
        dataset = None
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            features = {
                "observation.state": {
                    "dtype": "float32",
                    "shape": (self.metadata["state_width"],),
                    "names": _feature_names(self.metadata),
                },
                "action": {
                    "dtype": "float32",
                    "shape": (self.metadata["action_width"],),
                    "names": [
                        f"action/{i}" for i in range(self.metadata["action_width"])
                    ],
                },
                _PROVENANCE: {
                    "dtype": "int64",
                    "shape": (6,),
                    "names": [
                        "run_id",
                        "episode_id",
                        "frame_index",
                        "generation",
                        "continuity_id",
                        "published_at_monotonic_ns",
                    ],
                },
            }
            for name, feature in self.metadata["cameras"].items():
                h, w, _ = feature["shape"]
                features[f"observation.images.{name}"] = {
                    "dtype": "video",
                    "shape": (h, w, 3),
                    "names": ["height", "width", "channels"],
                }
            started = time.monotonic()
            # create() requires a nonexistent root; retain the explicit partial marker.
            (self.root / "INCOMPLETE").unlink()
            self.root.rmdir()
            dataset = LeRobotDataset.create(
                "local/ods-data-stream",
                fps=self.fps,
                features=features,
                root=self.root,
                robot_type="franka_panda",
                use_videos=bool(self.metadata["cameras"]),
                tolerance_s=1e-4,
                video_backend="torchcodec",
                batch_encoding_size=1,
                vcodec="libsvtav1",
                encoder_threads=1,
            )
            (self.root / "INCOMPLETE").write_text(
                "Accepted episodes are not finalized.\n"
            )
            self._stats["sdk_create_seconds"] = time.monotonic() - started
            manifest: dict[str, Any] = {
                "format": "ods-data-stream-lerobot-v1",
                "payload": self.metadata,
                "fps": self.fps,
                "versions": _check_versions(),
                "episodes": [],
                "references": [],
                "video": {
                    "codec": "libsvtav1",
                    "pix_fmt": "yuv420p",
                    "gop": 2,
                    "crf": 30,
                    "encoder_threads": 1,
                },
            }
            offset = 0
            while not self._abort.is_set():
                try:
                    episode = self._queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if episode is None:
                    break
                started = time.monotonic()
                length = len(episode["valid"])
                if not episode["valid"].all():
                    raise ValueError("Writer requires trimmed accepted frames")
                provenance = torch.cat(
                    [
                        episode["frame_ids"],
                        episode["generations"][:, None],
                        episode["continuity_id"][:, None],
                        episode["published_at_monotonic_ns"][:, None],
                    ],
                    -1,
                )
                for step in range(length):
                    if self._abort.is_set():
                        raise RuntimeError("LeRobot recording aborted")
                    frame = {
                        "observation.state": episode["state"][step],
                        "action": episode["actions"][step],
                        _PROVENANCE: provenance[step],
                        "task": "Pick up the cube and place it in the fixed goal region.",
                    }
                    frame.update(
                        {
                            f"observation.images.{name}": image[step]
                            for name, image in episode["images"].items()
                        }
                    )
                    dataset.add_frame(frame)
                dataset.save_episode(parallel_encoding=False)
                if not manifest["episodes"]:
                    refs = self.root / "meta" / "ods_raw_references"
                    refs.mkdir(exist_ok=True)
                    for name, images in episode["images"].items():
                        for step in sorted({0, length // 2, length - 1}):
                            filename = f"{name}-{step}.npy"
                            np.save(
                                refs / filename,
                                images[step].numpy(),
                                allow_pickle=False,
                            )
                            manifest["references"].append(
                                {
                                    "camera": name,
                                    "index": offset + step,
                                    "path": f"meta/ods_raw_references/{filename}",
                                }
                            )
                manifest["episodes"].append(
                    {
                        "index": len(manifest["episodes"]),
                        "offset": offset,
                        "length": length,
                        "source_id": episode["frame_ids"][0, :2].tolist(),
                        "generation": int(episode["generations"][0]),
                        "numeric_sha256": numeric_episode_digest(episode),
                    }
                )
                offset += length
                self._stats["written_episodes"] += 1
                self._stats["valid_frames"] += length
                self._stats["sdk_write_encode_seconds"] += time.monotonic() - started
                del episode
            if self._abort.is_set():
                raise RuntimeError("LeRobot recording aborted")
            if len(manifest["episodes"]) != self.episode_count:
                raise ValueError("Final accepted corpus episode count mismatch")
            started = time.monotonic()
            dataset.finalize()
            dataset = None
            self._stats["sdk_finalize_seconds"] = time.monotonic() - started
            manifest["required_files"] = [
                str(path.relative_to(self.root))
                for path in sorted(self.root.rglob("*"))
                if path.is_file() and path.name != "INCOMPLETE"
            ]
            (self.root / _SIDECAR).write_text(
                json.dumps(manifest, indent=2, allow_nan=False) + "\n"
            )
            (self.root / "INCOMPLETE").unlink()
            self._stats["finalized"] = True
        except BaseException as error:
            self._error = error
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / "INCOMPLETE").write_text(repr(error) + "\n")
        finally:
            if dataset is not None:
                try:
                    dataset.finalize()
                except BaseException as error:
                    self._stats["cleanup_error"] = repr(error)

    def _join(self, progress: Callable[[], None] | None, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while self._thread.is_alive():
            if progress is not None:
                progress()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "Native LeRobot writer did not join within its deadline"
                )
            self._thread.join(min(0.05, remaining))
        self._stats["worker_joined"] = True

    def finalize(self, *, progress: Callable[[], None] | None = None) -> dict[str, Any]:
        """Drain encoding/finalization and return detached component counters.

        Args:
            progress: Owner-thread event/cancellation callback while waiting.

        Returns:
            Recording counters. A failed corpus is never exposed as finalized.
        """
        self._check()
        if self._submitted != self.episode_count:
            raise ValueError(
                "Cannot finalize before all accepted episodes are submitted"
            )
        self._put(None, progress)
        self._join(progress, self.operation_timeout)
        self._check()
        self._closed = True
        return self.stats

    def abort(self) -> None:
        """Cancel pending episodes and bound native writer shutdown to 30 seconds."""
        if self._owner != (os.getpid(), threading.get_ident()):
            raise RuntimeError("Abort requires the creator process/thread")
        if self._closed:
            return
        self._abort.set()
        self._join(None, min(30.0, self.operation_timeout))
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._closed = True


@configclass
class LeRobotReaderCfg:
    """Bounded, standard SDK reader settings used by all visual conditions."""

    batch_size: int = 32
    history_length: int = 2
    prediction_horizon: int = 32
    episode_budget: int | None = 128
    seed: int = 29001
    num_workers: int = 4
    prefetch_factor: int = 1
    timeout: float = 120.0

    def __post_init__(self) -> None:
        for name in (
            "batch_size",
            "history_length",
            "prediction_horizon",
            "num_workers",
            "prefetch_factor",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.episode_budget is not None and (
            type(self.episode_budget) is not int or self.episode_budget < 1
        ):
            raise ValueError("episode_budget must be a positive integer or None")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be finite and positive")


class _DecodedContexts(Dataset):
    def __init__(
        self, root: Path, manifest: Mapping[str, Any], cfg: LeRobotReaderCfg
    ) -> None:
        self.root, self.manifest, self.cfg = root, manifest, cfg
        self._dataset: Any = None

    def __len__(self) -> int:
        return sum(episode["length"] for episode in self.manifest["episodes"])

    def __getitem__(self, key: tuple[int, int]) -> dict[str, Any]:
        reservation, index = key
        started = time.monotonic()
        if self._dataset is None:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            offsets = [
                step / self.manifest["fps"]
                for step in range(
                    1 - self.cfg.history_length, self.cfg.prediction_horizon
                )
            ]
            fields = ["observation.state", "action", _PROVENANCE]
            fields.extend(
                f"observation.images.{name}"
                for name in self.manifest["payload"]["cameras"]
            )
            self._dataset = LeRobotDataset(
                "local/ods-data-stream",
                root=self.root,
                delta_timestamps={field: offsets for field in fields},
                tolerance_s=1e-4,
                video_backend="torchcodec",
            )
        item = self._dataset[index]  # Public SDK read, including real video decoding.
        provenance = item[_PROVENANCE].to(dtype=torch.int64)
        anchor = self.cfg.history_length - 1
        padding = item[f"{_PROVENANCE}_is_pad"].bool()
        continuity = provenance[:, 4]
        compatible = (~padding) & (continuity == continuity[anchor])
        # Stop at the nearest continuity break; do not reconnect matching IDs.
        positions = torch.arange(len(compatible))
        lower = int(
            torch.where((~compatible) & (positions < anchor), positions + 1, 0).max()
        )
        upper = int(
            torch.where(
                (~compatible) & (positions > anchor), positions - 1, len(compatible) - 1
            ).min()
        )
        valid = compatible & (positions >= lower) & (positions <= upper)
        result: dict[str, Any] = {
            "state": item["observation.state"].float(),
            "actions": item["action"].float(),
            "frame_ids": provenance[:, :3].clone(),
            "generations": provenance[:, 3].clone(),
            "continuity_id": continuity.clone(),
            "published_at_monotonic_ns": provenance[:, 5].clone(),
            "valid": valid,
            "loss_mask": valid & (positions >= anchor),
            "images": {},
            "_reservation_id": reservation,
        }
        for name in self.manifest["payload"]["cameras"]:
            field = f"observation.images.{name}"
            if not torch.equal(item[f"{field}_is_pad"].bool(), padding):
                raise ValueError("SDK image/numeric timestamp padding differs")
            image = item[field]
            if image.ndim == 3 and len(valid) == 1:
                image = image.unsqueeze(0)
            if image.dtype != torch.float32 or image.ndim != 4 or image.shape[1] != 3:
                raise ValueError("SDK decoder must return temporal RGB float32 CHW")
            result["images"][name] = image.masked_fill(~valid[:, None, None, None], 0)
        for name in ("state", "actions"):
            result[name] = result[name].masked_fill(~valid[:, None], 0)
        for name in (
            "frame_ids",
            "generations",
            "continuity_id",
            "published_at_monotonic_ns",
        ):
            mask = valid[:, None] if name == "frame_ids" else valid
            result[name].masked_fill_(~mask, -1)
        result["_service_seconds"] = time.monotonic() - started
        result["_worker_pid"] = os.getpid()
        return result


def _initialize_worker(worker_id: int) -> None:
    torch.set_num_threads(1)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def _finish_loader(
    iterator: Any, diagnostics: dict[str, Any], *, timeout: float
) -> None:
    """Drain a finite dispatch stream before requiring clean worker shutdown."""
    workers = list(getattr(iterator, "_workers", ()))
    started = time.monotonic()
    drained = 0
    error: BaseException | None = None
    cleanup_errors: list[str] = []
    forced = False
    try:
        while True:
            remaining = started + timeout - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "LeRobot prefetch drain missed its shutdown deadline"
                )
            iterator._timeout = min(timeout, remaining)
            try:
                batch = next(iterator)
            except StopIteration:
                break
            drained += len(batch["valid"])
            del batch
    except BaseException as failure:
        error = failure
    finally:
        try:
            # PyTorch has no public DataLoader.close(). Reading remains through
            # the public SDK; this private call is isolated to lifecycle cleanup.
            iterator._shutdown_workers()
        except BaseException as failure:
            cleanup_errors.append(repr(failure))
            if error is None:
                error = failure
        for worker in workers:
            for stop in (None, worker.terminate, worker.kill):
                try:
                    if stop is not None and worker.is_alive():
                        forced = True
                        stop()
                    worker.join(timeout=5)
                    if not worker.is_alive():
                        break
                except BaseException as failure:
                    cleanup_errors.append(repr(failure))
                    if error is None:
                        error = failure
        codes = [worker.exitcode for worker in workers]
        forced |= any(code in (-15, -9) for code in codes)
        diagnostics.update(
            worker_pids=[worker.pid for worker in workers],
            worker_exitcodes=codes,
            workers_joined=all(not worker.is_alive() for worker in workers),
            forced_termination=forced,
            discarded_prefetch_chunks=drained,
            shutdown_seconds=time.monotonic() - started,
            cleanup_errors=cleanup_errors,
        )
    if error is None and (
        forced or not diagnostics["workers_joined"] or any(code != 0 for code in codes)
    ):
        error = RuntimeError(f"LeRobot worker shutdown was not clean: {diagnostics}")
    if error is not None:
        diagnostics["error"] = repr(error)
        raise error


class _ReservationSampler:
    """Parent-only episode-uniform index selection and shared exposure ledger."""

    def __init__(self, episodes: list[dict[str, Any]], cfg: LeRobotReaderCfg) -> None:
        self.episodes, self.cfg = episodes, cfg
        self.delivered = [0] * len(episodes)
        self.reserved = [0] * len(episodes)
        self.pending: dict[int, list[int]] = {}
        self.generator = torch.Generator().manual_seed(cfg.seed)
        self.next_reservation = 0
        self.cancelled_chunks = 0
        self.dispatch_enabled = True

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        while self.dispatch_enabled:
            available = [
                index
                for index in range(len(self.episodes))
                if self.cfg.episode_budget is None
                or self.delivered[index] + self.reserved[index]
                < self.cfg.episode_budget
            ]
            if not available:
                return
            if self.cfg.episode_budget is not None:
                remaining = sum(
                    self.cfg.episode_budget - self.delivered[i] - self.reserved[i]
                    for i in available
                )
                if remaining < self.cfg.batch_size:
                    return
            selected: list[int] = []
            indices: list[int] = []
            for _ in range(self.cfg.batch_size):
                row = available[
                    int(torch.randint(len(available), (), generator=self.generator))
                ]
                selected.append(row)
                self.reserved[row] += 1
                episode = self.episodes[row]
                anchor = int(
                    torch.randint(episode["length"], (), generator=self.generator)
                )
                indices.append(episode["offset"] + anchor)
                if (
                    self.cfg.episode_budget is not None
                    and self.delivered[row] + self.reserved[row]
                    == self.cfg.episode_budget
                ):
                    available.remove(row)
            reservation = self.next_reservation
            self.next_reservation += 1
            self.pending[reservation] = selected
            yield [(reservation, index) for index in indices]

    def commit(self, reservation: int) -> None:
        for row in self.pending.pop(reservation):
            self.reserved[row] -= 1
            self.delivered[row] += 1

    def cancel_outstanding(self) -> None:
        self.cancelled_chunks += sum(self.reserved)
        self.pending.clear()
        self.reserved = [0] * len(self.episodes)


class StandardLeRobotReader:
    """Deliver decoded contexts using spawned SDK workers and parent reservations.

    Args:
        root: Finalized SDK corpus with the showcase sidecar.
        episodes: Optional logical episode indices, e.g. the first C diagnostic.
        **kwargs: :class:`LeRobotReaderCfg` fields.

    K accounts for successfully returned complete chunks, with outstanding
    prefetch separately reserved. No decoded corpus is cached by this wrapper.
    A decoder/worker error is terminal. ``restart`` preserves delivered credits
    and cancels only prefetched work after all old workers have stopped.
    """

    def __init__(
        self, root: Path | str, *, episodes: list[int] | None = None, **kwargs: Any
    ) -> None:
        _check_versions()
        self.root = Path(root)
        if (self.root / "INCOMPLETE").exists():
            raise RuntimeError("Cannot read an incomplete LeRobot corpus")
        self.manifest = json.loads((self.root / _SIDECAR).read_text())
        for filename in self.manifest["required_files"]:
            if not (self.root / filename).is_file():
                raise FileNotFoundError(f"Incomplete local corpus: {filename}")
        self.metadata = copy.deepcopy(self.manifest["payload"])
        self.cfg = LeRobotReaderCfg(**kwargs)
        self._owner = (os.getpid(), threading.get_ident())
        all_episodes = self.manifest["episodes"]
        if episodes is not None and (
            not episodes
            or len(set(episodes)) != len(episodes)
            or any(
                type(i) is not int or i < 0 or i >= len(all_episodes) for i in episodes
            )
        ):
            raise ValueError("episodes must contain unique logical episode indices")
        selected = (
            all_episodes if episodes is None else [all_episodes[i] for i in episodes]
        )
        self._sampler = _ReservationSampler(selected, self.cfg)
        self._dataset = _DecodedContexts(self.root, self.manifest, self.cfg)
        self._loader: DataLoader | None = None
        self._iterator: Any = None
        self._failed = False
        self._closed = False
        self._shutdown: dict[str, Any] = {}
        self._shutdown_error: BaseException | None = None
        self._stats: dict[str, Any] = {
            "delivered_batches": 0,
            "sdk_getitem_calls": 0,
            "returned_valid_camera_frames": 0,
            "requested_camera_timestamps": 0,
            "worker_fetch_seconds": 0.0,
            "worker_pids": [],
            "reader": "official LeRobotDataset.__getitem__ + delta_timestamps",
            "video_backend": "torchcodec CPU",
            "decoded_corpus_preloaded": False,
            "prefetch_batches": self.cfg.num_workers * self.cfg.prefetch_factor,
            "persistent_workers": True,
            "ordered_delivery": True,
            "pinned_memory": False,
        }
        self._start()

    def _check(self) -> None:
        if self._owner != (os.getpid(), threading.get_ident()):
            raise RuntimeError("Reader operations require the creator process/thread")
        if self._closed or self._failed:
            raise RuntimeError("Reader is closed or terminally failed")

    def _start(self) -> None:
        self._sampler.dispatch_enabled = True
        self._loader = DataLoader(
            self._dataset,
            batch_sampler=self._sampler,
            num_workers=self.cfg.num_workers,
            multiprocessing_context="spawn",
            prefetch_factor=self.cfg.prefetch_factor,
            persistent_workers=True,
            in_order=True,
            pin_memory=False,
            worker_init_fn=_initialize_worker,
            timeout=self.cfg.timeout,
            generator=torch.Generator().manual_seed(self.cfg.seed),
            collate_fn=default_collate,
        )
        self._iterator = iter(self._loader)

    def next_batch(self) -> dict[str, Any]:
        """Return one decoded batch; StopIteration denotes a depleted fixed corpus.

        Returns:
            Canonical CPU payload with ``_service`` worker timing diagnostics.
        """
        self._check()
        try:
            batch = next(self._iterator)
            reservations = batch.pop("_reservation_id")
            if not (reservations == reservations[0]).all():
                raise RuntimeError("Mixed DataLoader exposure reservations")
            reservation = int(reservations[0])
            service = batch.pop("_service_seconds")
            worker_pids = batch.pop("_worker_pid").unique().tolist()
            selected = self._sampler.pending[reservation]
            expected_ids = torch.tensor(
                [self._sampler.episodes[row]["source_id"] for row in selected],
                dtype=torch.int64,
            )
            anchor = self.cfg.history_length - 1
            if not batch["valid"][:, anchor].all() or not torch.equal(
                batch["frame_ids"][:, anchor, :2], expected_ids
            ):
                raise RuntimeError(
                    "Decoded batch source identities differ from its reservation"
                )
            # Collation, SDK decode and normalization all succeeded before debit.
            self._sampler.commit(reservation)
            self._stats["delivered_batches"] += 1
            self._stats["sdk_getitem_calls"] += self.cfg.batch_size
            self._stats["worker_fetch_seconds"] += service.sum().item()
            cameras = len(self.metadata["cameras"])
            self._stats["returned_valid_camera_frames"] += (
                int(batch["valid"].sum()) * cameras
            )
            self._stats["requested_camera_timestamps"] += (
                batch["valid"].numel() * cameras
            )
            self._stats["worker_pids"] = sorted(
                set(self._stats["worker_pids"]) | set(worker_pids)
            )
            batch["_service"] = {
                "worker_fetch_seconds": service.sum().item(),
                "worker_item_seconds": service.tolist(),
                "scope": "worker SDK fetch including decode, temporal conversion and padding; first item includes Dataset startup",
            }
            return batch
        except StopIteration:
            raise
        except BaseException:
            self._failed = True
            raise

    @property
    def stats(self) -> dict[str, Any]:
        """Return delivery-local decode evidence; it excludes unused prefetch.

        Worker service is aggregate worker time, not additive critical-path wall
        time. Requested timestamps include clamped boundary duplicates.
        """
        return copy.deepcopy(self._stats)

    def snapshot_budget(self) -> dict[str, Any]:
        """Return delivery and live-prefetch counts for every immutable source."""
        return {
            "episode_budget": self.cfg.episode_budget,
            "delivered_chunks": sum(self._sampler.delivered),
            "reserved_chunks": sum(self._sampler.reserved),
            "cancelled_prefetch_chunks": self._sampler.cancelled_chunks,
            "pending_batches": len(self._sampler.pending),
            "remaining_chunks": (
                None
                if self.cfg.episode_budget is None
                else len(self._sampler.episodes) * self.cfg.episode_budget
                - sum(self._sampler.delivered)
                - sum(self._sampler.reserved)
            ),
            "episodes": [
                {
                    "source_id": episode["source_id"],
                    "assigned": self.cfg.episode_budget,
                    "delivered": delivered,
                    "reserved": reserved,
                    "remaining": (
                        None
                        if self.cfg.episode_budget is None
                        else self.cfg.episode_budget - delivered - reserved
                    ),
                }
                for episode, delivered, reserved in zip(
                    self._sampler.episodes,
                    self._sampler.delivered,
                    self._sampler.reserved,
                )
            ],
        }

    def _stop_workers(self) -> None:
        if self._iterator is None:
            if self._shutdown_error is not None:
                raise self._shutdown_error
            return
        self._sampler.dispatch_enabled = False
        self._shutdown = {"unconsumed_prefetch": self.snapshot_budget()}
        try:
            # Discard already reserved outputs without committing exposure or
            # dispatching replacements. Workers become idle before teardown.
            _finish_loader(
                self._iterator, self._shutdown, timeout=min(30.0, self.cfg.timeout)
            )
        except BaseException as error:
            self._failed = True
            self._shutdown_error = error
            raise
        finally:
            self._iterator, self._loader = None, None

    def restart(self) -> None:
        """Restart worker iteration without resetting delivered exposure credits."""
        self._check()
        try:
            self._stop_workers()
        finally:
            self._sampler.cancel_outstanding()
        self._start()

    def close(self) -> dict[str, Any]:
        """Drain unused prefetch without delivery, then require clean worker exits.

        Returns:
            Shutdown status and the unconsumed reservation snapshot.
        """
        if self._owner != (os.getpid(), threading.get_ident()):
            raise RuntimeError("Reader close requires the creator process/thread")
        if not self._closed:
            try:
                self._stop_workers()
            finally:
                self._sampler.cancel_outstanding()
                self._closed = True
        if self._shutdown_error is not None:
            raise self._shutdown_error
        return copy.deepcopy(self._shutdown)

    def __enter__(self) -> StandardLeRobotReader:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def audit_video_quality(root: Path | str) -> dict[str, Any]:
    """Decode small raw-reference samples outside timed consumption.

    Args:
        root: Finalized corpus containing raw first-episode RGB samples.

    Returns:
        Per-camera MAE and PSNR; lossless matches use null PSNR plus a flag.
        No raw images beyond the fixed references are retained.
    """
    root = Path(root)
    manifest = json.loads((root / _SIDECAR).read_text())
    refs = manifest["references"]
    if not refs:
        return {"samples": [], "scope": "state-only; no video"}
    cfg = LeRobotReaderCfg(
        batch_size=1,
        history_length=1,
        prediction_horizon=1,
        episode_budget=None,
        num_workers=1,
    )
    dataset = _DecodedContexts(root, manifest, cfg)
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=[(i, ref["index"]) for i, ref in enumerate(refs)],
        num_workers=1,
        multiprocessing_context="spawn",
        prefetch_factor=1,
        worker_init_fn=_initialize_worker,
        timeout=cfg.timeout,
    )
    iterator = iter(loader)
    samples = []
    shutdown: dict[str, Any] = {}
    try:
        for index, batch in enumerate(iterator):
            ref = refs[index]
            decoded = batch["images"][ref["camera"]][0, 0].movedim(0, -1).double() * 255
            raw = torch.from_numpy(
                np.load(root / ref["path"], allow_pickle=False)
            ).double()
            diff = decoded - raw
            mse = float(diff.square().mean())
            samples.append(
                {
                    "camera": ref["camera"],
                    "index": ref["index"],
                    "mae_uint8": float(diff.abs().mean()),
                    "psnr_db": 10 * math.log10(255**2 / mse) if mse else None,
                    "exact": mse == 0,
                }
            )
    finally:
        _finish_loader(iterator, shutdown, timeout=min(30.0, cfg.timeout))
    return {
        "samples": samples,
        "worker_cleanup": shutdown,
        "scope": "fixed raw RGB references versus standard CPU video decoding; excluded from timed consumption",
    }
