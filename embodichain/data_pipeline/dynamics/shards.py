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

"""Committed HDF5 trajectories and bounded, worker-local window reads."""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from collections.abc import Sequence
import operator

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = [
    "DynamicsShardWriter",
    "DynamicsWindowDataset",
    "trajectory_split",
    "load_manifest",
    "dataset_fingerprint",
]


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".json.pending")
    with temporary.open("wb") as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trajectory_split(trajectory_id: str, seed: int = 0) -> str:
    """Assign a stable trajectory-level 80/10/10 split independent of order."""
    fraction = (
        int.from_bytes(
            hashlib.sha256(f"{seed}:{trajectory_id}".encode()).digest()[:8], "big"
        )
        / 2**64
    )
    return "train" if fraction < 0.8 else "validation" if fraction < 0.9 else "test"


def load_manifest(root: str | Path, *, verify: bool = False) -> dict[str, Any]:
    """Read the committed dataset snapshot, optionally checking shard hashes."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported dynamics dataset schema")
    for shard in manifest["shards"]:
        name = shard["file"]
        if Path(name).name != name or not (root / name).is_file():
            raise ValueError(f"Missing or unsafe committed shard: {name}")
        if verify and _sha256(root / name) != shard["sha256"]:
            raise ValueError(f"Shard checksum mismatch: {name}")
    return manifest


def dataset_fingerprint(manifest: dict[str, Any]) -> str:
    """Hash metadata, schema, immutable shard hashes and split assignments."""
    return hashlib.sha256(_json_bytes(manifest)).hexdigest()


class DynamicsShardWriter:
    """Write trajectories directly to disk; only complete shards are visible.

    This is a single-writer API. Each caller-owned trajectory is written before
    ``add_trajectory`` returns; no unbounded staging queue is retained. Resume
    ignores files absent from the manifest, including an interrupted final shard.
    """

    def __init__(
        self,
        root: str | Path,
        metadata: dict[str, Any],
        trajectories_per_shard: int = 64,
        *,
        resume: bool = False,
        split_seed: int = 0,
    ) -> None:
        if trajectories_per_shard < 1:
            raise ValueError("trajectories_per_shard must be positive")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.capacity = trajectories_per_shard
        self._file: h5py.File | None = None
        self._pending: list[dict[str, Any]] = []
        self._closed = False
        path = self.root / "manifest.json"
        metadata = json.loads(_json_bytes(metadata))
        if path.exists():
            if not resume:
                raise FileExistsError(f"Dataset already exists: {path}")
            self.manifest = load_manifest(self.root, verify=True)
            if (
                self.manifest["metadata"] != metadata
                or self.manifest["split_seed"] != split_seed
            ):
                raise ValueError("Resume metadata or split seed does not match")
        else:
            if resume:
                raise FileNotFoundError(path)
            self.manifest = {
                "schema_version": 1,
                "metadata": metadata,
                "split_seed": split_seed,
                "fields": None,
                "shards": [],
            }
            _atomic_json(path, self.manifest)
        self._ids = {
            row["id"]
            for shard in self.manifest["shards"]
            for row in shard["trajectories"]
        }

    @property
    def committed_ids(self) -> frozenset[str]:
        """IDs persisted in the current manifest, excluding pending writes."""
        return frozenset(
            row["id"]
            for shard in self.manifest["shards"]
            for row in shard["trajectories"]
        )

    def add_trajectory(
        self,
        trajectory_id: str,
        states: dict[str, np.ndarray],
        transitions: dict[str, np.ndarray],
        *,
        split: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append one finite trajectory, with T+1 states and T transitions."""
        if self._closed:
            raise RuntimeError("Writer is closed")
        if not trajectory_id or trajectory_id in self._ids:
            raise ValueError(f"Empty or duplicate trajectory ID: {trajectory_id}")
        if not {"q", "qd"}.issubset(states) or not transitions:
            raise ValueError("states requires q/qd and transitions must be nonempty")
        arrays = {
            "states": {k: np.asarray(v) for k, v in states.items()},
            "transitions": {k: np.asarray(v) for k, v in transitions.items()},
        }
        length = len(next(iter(arrays["transitions"].values())))
        if length < 1:
            raise ValueError("Trajectory must contain at least one transition")
        schema: dict[str, Any] = {}
        for group, fields in arrays.items():
            schema[group] = {}
            for name, value in fields.items():
                if (
                    not name
                    or "/" in name
                    or value.ndim < 2
                    or value.shape[0] != length + (group == "states")
                ):
                    raise ValueError(f"Invalid trajectory shape/name: {group}/{name}")
                if (
                    value.dtype.kind not in "fiu b".replace(" ", "")
                    or not np.isfinite(value).all()
                ):
                    raise ValueError(f"Non-finite/non-numeric trajectory field: {name}")
                schema[group][name] = {
                    "shape": list(value.shape[1:]),
                    "dtype": value.dtype.str,
                }
        if self.manifest["fields"] is not None and schema != self.manifest["fields"]:
            raise ValueError("Trajectory fields differ from committed schema")
        split = split or trajectory_split(trajectory_id, self.manifest["split_seed"])
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {split}")
        trajectory_metadata = json.loads(_json_bytes(metadata or {}))
        if self._file is None:
            self._name = f"shard-{len(self.manifest['shards']):06d}.h5"
            self._temporary = self.root / (self._name + ".pending")
            self._file = h5py.File(self._temporary, "w")
        group_name = str(len(self._pending))
        target = self._file.create_group(group_name)
        for group, fields in arrays.items():
            sub = target.create_group(group)
            for name, value in fields.items():
                # Zero-width contact features cannot be compressed/chunked.
                opts = {"chunks": True, "compression": "lzf"} if value.size else {}
                sub.create_dataset(name, data=value, **opts)
        self.manifest["fields"] = schema
        self._pending.append(
            {
                "id": trajectory_id,
                "group": group_name,
                "length": length,
                "split": split,
                "metadata": trajectory_metadata,
            }
        )
        self._ids.add(trajectory_id)
        if len(self._pending) >= self.capacity:
            self.commit()

    def commit(self) -> None:
        """Publish the current shard atomically, including a content checksum."""
        if self._file is None:
            return
        self._file.flush()
        self._file.close()
        self._file = None
        with self._temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        destination = self.root / self._name
        os.replace(self._temporary, destination)
        shard = {
            "file": self._name,
            "sha256": _sha256(destination),
            "trajectories": self._pending,
        }
        self.manifest["shards"].append(shard)
        _atomic_json(self.root / "manifest.json", self.manifest)
        self._pending = []

    def close(self) -> None:
        """Commit the final partial shard and close the writer."""
        if not self._closed:
            self.commit()
            self._closed = True

    def abort(self) -> None:
        """Discard uncommitted work, retaining every previously committed shard."""
        if self._file is not None:
            self._file.close()
            self._file = None
        self._closed = True

    def __enter__(self) -> DynamicsShardWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close() if exc_type is None else self.abort()


class DynamicsWindowDataset(Dataset):
    """Read bounded windows with O(trajectories) indexing and LRU file handles.

    Default full windows match the official sequence dataset. ``include_prefix``
    also samples startup windows, padded on the right with a validity mask.
    No window ever spans a trajectory, reset or split boundary.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        history_length: int,
        *,
        include_prefix: bool = False,
        max_open_shards: int = 4,
        read_page_steps: int = 256,
        verify: bool = False,
    ) -> None:
        if (
            history_length < 1
            or max_open_shards < 1
            or read_page_steps < 1
            or split not in {"train", "validation", "test"}
        ):
            raise ValueError("Invalid split, history length or handle limit")
        self.root = Path(root)
        self.manifest = load_manifest(self.root, verify=verify)
        self.fingerprint = dataset_fingerprint(self.manifest)
        self.split = split
        self.history_length = history_length
        self.include_prefix = include_prefix
        self.max_open_shards = max_open_shards
        self.read_page_steps = read_page_steps
        self._rows: list[dict[str, Any]] = []
        self._ends: list[int] = []
        total = 0
        for shard in self.manifest["shards"]:
            for row in shard["trajectories"]:
                count = (
                    row["length"]
                    if include_prefix
                    else max(0, row["length"] - history_length + 1)
                )
                if row["split"] == split and count:
                    self._rows.append({**row, "file": shard["file"]})
                    total += count
                    self._ends.append(total)
        self._pid = os.getpid()
        self._handles: OrderedDict[str, h5py.File] = OrderedDict()
        self._resident = None

    def enable_device_cache(self, device: str | torch.device, max_bytes: int) -> int:
        """Cache raw transitions within an explicit single-process byte budget.

        Args:
            device: CPU or CUDA device used by this learner.
            max_bytes: Maximum resident raw-field allocation. Insufficient
                budgets fail before allocating; no data are dropped or quantized.

        Returns:
            Actual resident field bytes. The original shard fingerprint and
            statistical/window weighting remain unchanged.
        """
        from ._resident import _ResidentWindows

        if self._resident is not None:
            raise ValueError("A resident dynamics cache is already enabled")
        cache = _ResidentWindows(self, device, max_bytes)
        self._resident = cache
        return cache.nbytes

    def __len__(self) -> int:
        return self._ends[-1] if self._ends else 0

    def _handle(self, name: str) -> h5py.File:
        if os.getpid() != self._pid:
            self.close()
            self._pid = os.getpid()
        if name not in self._handles:
            if len(self._handles) >= self.max_open_shards:
                _, handle = self._handles.popitem(last=False)
                handle.close()
            self._handles[name] = h5py.File(self.root / name, "r")
        self._handles.move_to_end(name)
        return self._handles[name]

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        row_index = bisect_right(self._ends, index)
        row = self._rows[row_index]
        offset = index - (self._ends[row_index - 1] if row_index else 0)
        end = offset + 1 if self.include_prefix else offset + self.history_length
        start = max(0, end - self.history_length)
        length = end - start
        source = self._handle(row["file"])[row["group"]]
        result: dict[str, Any] = {
            "trajectory_id": row["id"],
            "start": start,
            "length": length,
            "valid_mask": torch.arange(self.history_length) < length,
        }
        for group in ("states", "transitions"):
            result[group] = {}
            extra = int(group == "states")
            for name in self.manifest["fields"][group]:
                value = source[group][name][start : end + extra]
                padding = self.history_length - length
                if padding:
                    value = np.concatenate(
                        (value, np.repeat(value[-1:], padding, axis=0)), axis=0
                    )
                result[group][name] = torch.from_numpy(value.copy())
        return result

    @staticmethod
    def _read_block(
        group: h5py.Group,
        name: str,
        start: int,
        stop: int,
        shape: tuple[int, ...],
        dtype: np.dtype,
    ) -> np.ndarray:
        """Read a contiguous, exactly sized hyperslab without Dataset wrappers."""
        block = np.empty((stop - start, *shape), dtype=dtype)
        dataset = h5py.h5d.open(group.id, name.encode("utf-8"))
        source_space = memory_space = None
        try:
            source_space = dataset.get_space()
            actual = source_space.get_simple_extent_dims()
            if tuple(actual[1:]) != shape or not 0 <= start < stop <= actual[0]:
                raise ValueError(f"Committed field shape/range mismatch: {name}")
            if not block.size:
                return block
            source_space.select_hyperslab((start, *([0] * len(shape))), block.shape)
            # The memory space is constructed from this owned C-contiguous
            # allocation, never from untrusted file dimensions. H5Dread therefore
            # cannot write beyond the allocated destination buffer.
            memory_space = h5py.h5s.create_simple(block.shape)
            dataset.read(memory_space, source_space, block)
        finally:
            if memory_space is not None:
                memory_space.close()
            if source_space is not None:
                source_space.close()
            dataset.close()
        return block

    def get_batch(
        self, indices: Sequence[int] | np.ndarray | torch.Tensor
    ) -> dict[str, Any]:
        """Read an owned batch in caller order using bounded grouped HDF5 reads.

        Requests are grouped by shard, trajectory and a fixed time page. The
        largest source read is ``read_page_steps + history_length`` frames;
        no trajectory/data cache grows with dataset size. Storage consists of the
        output batch, O(batch size) indices, one bounded source block and a
        per-field batched gather at a time. Repeated requests occupy independent rows in the returned tensors.
        Sampling and prefetch state are not touched by this synchronous method.

        Args:
            indices: Nonempty CPU integer window indices. Negative indices have
                the same meaning as ``__getitem__``.

        Returns:
            The same nested tensors/metadata as PyTorch's default collation of
            individual windows, including right-padded startup masks.
        """
        if isinstance(indices, torch.Tensor) and indices.device.type != "cpu":
            raise ValueError("Dynamics batch indices must be on CPU")
        requests = [operator.index(index) for index in indices]
        if not requests:
            raise ValueError("A dynamics batch must contain at least one window")
        if self._resident is not None:
            return self._resident.get_batch(requests)
        size = len(requests)
        grouped: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
        identifiers: list[str] = []
        starts = np.empty(size, dtype=np.int64)
        lengths = np.empty(size, dtype=np.int64)
        for position, index in enumerate(requests):
            if index < 0:
                index += len(self)
            if not 0 <= index < len(self):
                raise IndexError(index)
            row_index = bisect_right(self._ends, index)
            row = self._rows[row_index]
            offset = index - (self._ends[row_index - 1] if row_index else 0)
            end = offset + 1 if self.include_prefix else offset + self.history_length
            start = max(0, end - self.history_length)
            starts[position], lengths[position] = start, end - start
            identifiers.append(row["id"])
            grouped.setdefault((row_index, start // self.read_page_steps), []).append(
                (position, start, end)
            )
        output = {
            kind: {
                name: np.empty(
                    (
                        size,
                        self.history_length + int(kind == "states"),
                        *field["shape"],
                    ),
                    dtype=np.dtype(field["dtype"]),
                )
                for name, field in fields.items()
            }
            for kind, fields in self.manifest["fields"].items()
        }
        # Sort only the I/O schedule, never the sample sequence presented to the
        # learner. Processing all requests for one shard prevents LRU thrashing.
        keys = sorted(grouped, key=lambda key: (self._rows[key[0]]["file"], key))
        last_row = None
        source_groups = None
        for key in keys:
            row_index, _ = key
            row = self._rows[row_index]
            if row_index != last_row:
                source = self._handle(row["file"])[row["group"]]
                source_groups = {kind: source[kind] for kind in output}
                last_row = row_index
            entries = np.asarray(grouped[key], dtype=np.int64)
            positions, first, end = entries[:, 0], entries[:, 1], entries[:, 2]
            low, high = int(first.min()), int(end.max())
            relative = first - low
            for kind, fields in output.items():
                extra = int(kind == "states")
                offsets = np.arange(self.history_length + extra)
                take = relative[:, None] + np.minimum(
                    offsets[None, :], (end - first - 1 + extra)[:, None]
                )
                for name, array in fields.items():
                    block = self._read_block(
                        source_groups[kind],
                        name,
                        low,
                        high + extra,
                        tuple(array.shape[2:]),
                        array.dtype,
                    )
                    array[positions] = block[take]
                    del block
        return {
            "trajectory_id": identifiers,
            "start": torch.from_numpy(starts),
            "length": torch.from_numpy(lengths),
            "valid_mask": torch.arange(self.history_length)[None, :]
            < torch.from_numpy(lengths)[:, None],
            **{
                kind: {name: torch.from_numpy(value) for name, value in fields.items()}
                for kind, fields in output.items()
            },
        }

    def close(self) -> None:
        """Release resident fields and file handles; returned batches stay owned."""
        self._resident = None
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = OrderedDict()
        state["_resident"] = None
        return state

    def __del__(self) -> None:
        # h5py internals may already be finalized during interpreter shutdown.
        # Explicit close() remains strict so operational failures are visible.
        try:
            if hasattr(self, "_handles"):
                self.close()
        except Exception:
            pass
