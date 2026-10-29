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

"""Replayable author MOVi packed sequences for unified supervised learning."""

from __future__ import annotations
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any
import hashlib
import json
import random
import numpy as np
import torch
from embodichain.learning.sl import SupervisedBatch
from .supervised import rigidformer_batch

__all__ = ["MOViSequenceSource"]


def _check_manifest_coverage(root: Path, manifest: Mapping[str, Any]) -> None:
    """Require every readable geometry payload to belong to the committed set."""
    payloads = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.suffix == ".npy"
            or path.name in {"splits.json", "object_scales.json", "physics_params.json"}
        )
    }
    missing = payloads - manifest["files"].keys()
    if missing:
        raise ValueError(
            f"Prepared geometry is not committed by manifest: {sorted(missing)[0]}"
        )


def _verified_manifest(root: Path) -> dict[str, Any]:
    """Read a committed manifest and verify its actual prepared payloads."""
    path = root / "manifest.json"
    if not path.is_file():
        raise ValueError("Prepared data requires manifest.json with file hashes")
    manifest = json.loads(path.read_text())
    _check_manifest_coverage(root, manifest)
    for rel, expected in manifest["files"].items():
        h = hashlib.sha256()
        with (root / rel).open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        if h.hexdigest() != expected:
            raise ValueError(f"Prepared geometry hash differs: {rel}")
    return manifest


class MOViSequenceSource:
    """Read author packed scenes and checkpoint a rank-local shuffled traversal.

    Args:
        root: Converted MOVi-Sphere, MOVi-A or MOVi-B directory.
        split_file: Frozen author scene split JSON.
        object_scale_file: Frozen author size sidecar for unscaled stored vertices.
        split: train, val or test.
        data_config: Author sampling and augmentation settings.
        seed: Epoch-shuffle seed; per-batch augmentation RNG is caller-owned.
        max_batch_bytes: Bound before collating a padded geometry batch.

    Register this source in SupervisedTrainer.training_states so sampling is
    committed after successful updates. Core SL never imports this adapter.
    """

    def __init__(
        self,
        root: str | Path,
        split_file: str | Path,
        object_scale_file: str | Path,
        *,
        split: str = "train",
        data_config: Mapping[str, Any] | None = None,
        seed: int = 42,
        max_batch_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        from ._native.data.dataset import (
            MultiObjectDynamics,
            _cached_numpy_load,
            make_collate_fn,
        )
        from ._native.data.scene_frames import clear_cache

        self.root = Path(root)
        self.split_file, self.scale_file = Path(split_file), Path(object_scale_file)
        self.config = deepcopy(dict(data_config or {}))
        self.seed, self.max_batch_bytes = seed, max_batch_bytes
        if split not in ("train", "val", "test") or max_batch_bytes < 1:
            raise ValueError("Invalid split or batch memory limit")
        for name in ("skip_step_size", "stride"):
            if name in self.config and (
                type(self.config[name]) is not int or self.config[name] < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        if self.config.get("partial_view", {}).get("enable", False) or self.config.get(
            "if_point_cloud", {}
        ).get("enable", False):
            raise ValueError("This recipe supports full mesh-vertex point inputs")
        self.rank = (
            torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        )
        self.world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        self.manifest = _verified_manifest(self.root)
        # A previous source may have cached the old inode or per-frame arrays
        # at these paths before the newly verified manifest was committed.
        clear_cache(self.root)
        _cached_numpy_load.cache_clear()
        identity = {
            "manifest": self.manifest,
            "split": split,
            "split_sha": hashlib.sha256(self.split_file.read_bytes()).hexdigest(),
            "scale_sha": hashlib.sha256(self.scale_file.read_bytes()).hexdigest(),
            "sampling": self.config,
            "seed": seed,
        }
        self.source_id = (
            "rigidformer-movi-v1:"
            + hashlib.sha256(
                json.dumps(identity, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
        )
        options = {
            k: self.config[k]
            for k in (
                "sequence_length",
                "stride",
                "skip_first_frames",
                "skip_step_size",
                "random_step_size",
                "augmentation",
                "anchor_fps_start",
                "fps_order",
                "max_vertices_per_object",
                "max_objects_per_frame",
            )
            if k in self.config
        }
        self.dataset = MultiObjectDynamics(
            str(self.root),
            split=split,
            split_file=str(self.split_file),
            object_scale_file=str(self.scale_file),
            anchor_k=self.config.get("anchor_k", 4),
            if_phys_param=True,
            compact_index=True,
            require_prepared_cache=True,
            **options,
        )
        # The author loader can generate FPS caches. Prepared sources must
        # already commit those caches, rather than acquire untracked inputs.
        _check_manifest_coverage(self.root, self.manifest)
        maximum = self.config.get("max_objects_per_frame")
        if maximum is not None and any(
            len(counts) > maximum
            for counts in self.dataset._mesh_nv_by_dataset.values()
        ):
            raise ValueError(
                "max_objects_per_frame cannot truncate physical scene objects"
            )
        self.collate = make_collate_fn(
            max_objects_per_frame=self.config.get("max_objects_per_frame", 16),
            max_vertices_per_object=self.config.get("max_vertices_per_object"),
        )
        self.epoch = 0
        self.cursor = 0
        self._pending: tuple[int, int] | None = None
        self._order_epoch = -1
        self._order = torch.empty(0, dtype=torch.long)
        self._per_rank = (len(self.dataset) + self.world_size - 1) // self.world_size

    def _verify_files(self) -> None:
        if _verified_manifest(self.root) != self.manifest:
            raise ValueError("Prepared geometry manifest differs")

    def _indices(self, epoch: int) -> torch.Tensor:
        if self._order_epoch != epoch:
            generator = torch.Generator().manual_seed(self.seed + epoch)
            order = torch.randperm(len(self.dataset), generator=generator)
            total = self._per_rank * self.world_size
            if total > len(order):
                padding = total - len(order)
                repeats = (padding + len(order) - 1) // len(order)
                order = torch.cat((order, order.repeat(repeats)[:padding]))
            self._order = order[self.rank : total : self.world_size]
            self._order_epoch = epoch
        return self._order

    def updates_per_epoch(self, batch_size: int) -> int:
        """Return full-batch updates per rank, matching drop-last training.

        Args:
            batch_size: Number of windows in each rank's optimizer update.

        Returns:
            Number of complete rank-local batches in an epoch.
        """
        if type(batch_size) is not int or batch_size < 1 or self._per_rank < batch_size:
            raise ValueError("Every rank needs at least one complete batch")
        return self._per_rank // batch_size

    def sample(self, batch_size: int, *, generator: torch.Generator) -> SupervisedBatch:
        """Stage one bounded batch; on_update commits its traversal position.

        Args:
            batch_size: Number of windows on this rank.
            generator: Caller-owned sampling/augmentation RNG.

        Returns:
            Owned CPU shifted geometry supervision.
        """
        if self._pending is not None:
            raise RuntimeError("Previous sampled batch has not been committed")
        self.updates_per_epoch(batch_size)
        epoch, cursor = self.epoch, self.cursor
        if cursor + batch_size > self._per_rank:
            epoch, cursor = epoch + 1, 0
        indices = self._indices(epoch)[cursor : cursor + batch_size]
        names = [self.dataset.all_sequences[int(i)]["dataset"] for i in indices]
        objects = self.config.get("max_objects_per_frame", 16)
        if objects is None:
            objects = max(len(self.dataset._mesh_nv_by_dataset[name]) for name in names)
        points = self.config.get("max_vertices_per_object") or max(
            self.dataset._max_nv_by_dataset[name] for name in names
        )
        frames = self.dataset.sequence_length
        estimate = batch_size * (
            objects * (points * (frames * 13 + 24) + 9 * self.dataset.anchor_k + 128)
            + 64
        )
        if estimate > self.max_batch_bytes:
            raise ValueError(
                "Padded geometry exceeds max_batch_bytes before reading payloads"
            )
        py, npstate, ts = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
        )
        seed = int(torch.randint(0, 2**31, (1,), generator=generator))
        try:
            random.seed(seed)
            np.random.seed(seed)
            torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
            values = [self.dataset[int(i)] for i in indices]
        finally:
            random.setstate(py)
            np.random.set_state(npstate)
            torch.set_rng_state(ts)
        batch = self.collate(values)
        self._pending = (epoch, cursor + batch_size)
        return rigidformer_batch(
            batch,
            metadata={
                "source_id": self.source_id,
                "scenes": batch["dataset_name"],
                "frame_ids": batch["frame_ids"],
                "physical_frame_rate": 240,
                "model_dt": 1,
            },
        )

    def on_update(self, learner: Any) -> None:
        """Commit the staged window traversal after a successful optimizer step.

        Args:
            learner: Host update context; progress comes from the staged batch.
        """
        if self._pending is None:
            raise RuntimeError("No geometry batch is pending")
        self.epoch, self.cursor = self._pending
        self._pending = None

    def state_dict(self) -> dict[str, Any]:
        """Return the committed rank-local sampling boundary.

        Returns:
            Owned source identity, epoch, cursor and rank/world-size values.
        """
        return {
            "source_id": self.source_id,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "rank": self.rank,
            "world_size": self.world_size,
        }

    def validate_state_dict(self, state: Mapping[str, Any]) -> None:
        """Reject incompatible geometry or traversal before training mutation.

        Args:
            state: Candidate committed sampling snapshot.
        """
        if (
            state.get("source_id") != self.source_id
            or state.get("rank") != self.rank
            or state.get("world_size") != self.world_size
            or type(state.get("epoch")) is not int
            or state["epoch"] < 0
            or type(state.get("cursor")) is not int
            or not 0 <= state["cursor"] <= self._per_rank
        ):
            raise ValueError("MOVi sampling state differs")
        self._verify_files()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a prevalidated committed traversal and discard staged reads.

        Args:
            state: Snapshot accepted by validate_state_dict.
        """
        self.epoch, self.cursor = state["epoch"], state["cursor"]
        self._pending = None
        self._order_epoch = -1
