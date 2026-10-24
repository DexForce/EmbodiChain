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

"""Fixed episode-HDF5 source with bounded handles and causal image decoding."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import torch

__all__ = ["BCWindowDataset"]


class BCWindowDataset:
    """Sample complete history/target windows from a committed offline corpus.

    Args:
        manifest: JSON with ``complete=true``, ordered camera names, and
            episode records containing relative ``path``, ``length`` and sha256.
        history_length: Number of measured input frames including the present.
        prediction_horizon: Number of target frames including the present.
        observation_key: Numeric HDF5 dataset containing measured observations.
        observation_slice: Optional feature slice selected before model input.
        target_key: HDF5 physical targets, when no encoder is supplied.
        target_encoder: Optional task-owned transform of aligned numeric fields.
        target_fields: HDF5 inputs required by the target encoder.
        max_open_files: Positive bound on cached HDF5 handles.
        verify_checksums: Verify each file once before first reading it.

    Sampling is episode-uniform with replacement and uses the caller's CPU RNG.
    Only history camera frames are read; future geometry may construct targets
    but is never placed in the input tree. Every returned tensor owns storage.
    """

    def __init__(
        self,
        manifest: str | Path,
        *,
        history_length: int = 1,
        prediction_horizon: int = 1,
        observation_key: str = "observations",
        observation_slice: slice | None = None,
        target_key: str = "targets",
        target_encoder: (
            Callable[[Mapping[str, torch.Tensor]], torch.Tensor] | None
        ) = None,
        target_fields: tuple[str, ...] = (),
        max_open_files: int = 32,
        verify_checksums: bool = True,
    ) -> None:
        self.path = Path(manifest)
        self.manifest = json.loads(self.path.read_text())
        if self.manifest.get("complete") is not True:
            raise ValueError("BC training requires a completed fixed offline corpus")
        if any(
            type(value) is not int or value < 1
            for value in (history_length, prediction_horizon, max_open_files)
        ):
            raise ValueError("History, prediction and handle capacity must be positive")
        self.history_length, self.prediction_horizon = (
            history_length,
            prediction_horizon,
        )
        self.observation_key, self.observation_slice = (
            observation_key,
            observation_slice,
        )
        self.target_key, self.target_encoder, self.target_fields = (
            target_key,
            target_encoder,
            target_fields,
        )
        if target_encoder is not None and not target_fields:
            raise ValueError("Encoded BC targets require explicit numeric fields")
        self.cameras = tuple(self.manifest.get("cameras", ()))
        if len(set(self.cameras)) != len(self.cameras):
            raise ValueError("Offline camera names must be distinct and ordered")
        self.episodes = self.manifest["episodes"]
        if not self.episodes or any(
            record["length"] < history_length + prediction_horizon - 1
            for record in self.episodes
        ):
            raise ValueError(
                "Every BC episode must contain a complete requested window"
            )
        # The manifest binds content hashes and all physical/camera metadata;
        # source identity cannot be changed by reusing a display source_id.
        canonical = json.dumps(self.manifest, sort_keys=True, allow_nan=False)
        self.source_id = "bc-hdf5:" + hashlib.sha256(canonical.encode()).hexdigest()
        self.max_open_files, self.verify_checksums = max_open_files, verify_checksums
        self._handles: OrderedDict[int, h5py.File] = OrderedDict()
        self._verified: set[int] = set()

    def _handle(self, ordinal: int) -> h5py.File:
        handle = self._handles.pop(ordinal, None)
        if handle is None:
            record = self.episodes[ordinal]
            path = self.path.parent / record["path"]
            if self.verify_checksums and ordinal not in self._verified:
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    while payload := stream.read(1024 * 1024):
                        digest.update(payload)
                if digest.hexdigest() != record["sha256"]:
                    raise ValueError(
                        "Offline BC episode checksum differs from manifest"
                    )
                self._verified.add(ordinal)
            handle = h5py.File(path, "r")
            if len(handle[self.observation_key]) != record["length"]:
                handle.close()
                raise ValueError("Offline BC episode length differs from manifest")
            if len(self._handles) >= self.max_open_files:
                _, retired = self._handles.popitem(last=False)
                retired.close()
        self._handles[ordinal] = handle
        return handle

    def read_episode(
        self, ordinal: int, fields: tuple[str, ...]
    ) -> dict[str, torch.Tensor]:
        """Read owned numeric episode fields for explicit training statistics.

        Args:
            ordinal: Ordered episode index in the committed manifest.
            fields: Numeric HDF5 datasets to load; camera decoding is separate.

        Returns:
            Independently owned tensors, preserving stored numeric dtypes.
        """
        handle = self._handle(ordinal)
        return {name: torch.from_numpy(handle[name][:]) for name in fields}

    def sample(self, batch_size: int, *, generator: torch.Generator) -> dict[str, Any]:
        """Draw complete numeric windows and causal RGB histories.

        Args:
            batch_size: Positive number of independent windows.
            generator: Explicit CPU sampling RNG, checkpointed by the trainer.

        Returns:
            Owned CPU batch. Numeric input history is ``[B,H,D]``; images are
            FP32 ``[B,H,3,256,256]`` in ``[0,1]``; physical targets are ``[B,F,K]``.
        """
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("BC batch size must be a positive integer")
        ordinals = torch.randint(len(self.episodes), (batch_size,), generator=generator)
        anchor_draws = torch.rand(batch_size, generator=generator)
        anchors = []
        observations, targets = [], []
        images: dict[str, list[torch.Tensor]] = {camera: [] for camera in self.cameras}
        for ordinal, draw in zip(ordinals.tolist(), anchor_draws.tolist()):
            length = self.episodes[ordinal]["length"]
            count = length - self.history_length - self.prediction_horizon + 2
            anchor = self.history_length - 1 + int(draw * count)
            anchors.append(anchor)
            start = anchor - self.history_length + 1
            stop = anchor + self.prediction_horizon
            handle = self._handle(ordinal)
            observation = torch.from_numpy(
                handle[self.observation_key][start : anchor + 1]
            )
            if self.observation_slice is not None:
                observation = observation[..., self.observation_slice]
            observations.append(observation.float())
            if self.target_encoder is None:
                target = torch.from_numpy(handle[self.target_key][anchor:stop]).float()
            else:
                fields = {
                    name: torch.from_numpy(handle[name][anchor:stop]).float()
                    for name in self.target_fields
                }
                target = self.target_encoder(fields)
            targets.append(target)
            for camera in self.cameras:
                rgb = torch.from_numpy(handle["images"][camera][start : anchor + 1])
                if rgb.dtype != torch.uint8 or rgb.shape != (
                    self.history_length,
                    256,
                    256,
                    3,
                ):
                    raise ValueError(
                        "Offline BC images must be uint8 THWC RGB at 256 pixels"
                    )
                images[camera].append(rgb.permute(0, 3, 1, 2).float().div_(255))
        inputs: dict[str, Any] = {"history": torch.stack(observations)}
        if images:
            inputs["images"] = {
                camera: torch.stack(values) for camera, values in images.items()
            }
        return {
            "inputs": inputs,
            "targets": torch.stack(targets),
            "masks": {
                "target_valid": torch.ones(
                    batch_size, self.prediction_horizon, dtype=torch.bool
                )
            },
            "metadata": {
                "source_id": self.source_id,
                "episode_ids": ordinals,
                "anchors": torch.tensor(anchors),
                "cameras": self.cameras,
            },
        }

    def close(self) -> None:
        """Close cached handles; the immutable source can subsequently reopen."""
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
