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
from typing import Any, Literal

import h5py
import torch

__all__ = ["BCWindowDataset"]


class BCWindowDataset:
    """Sample causal histories and physical targets from a committed corpus.

    Args:
        manifest: JSON with ``complete=true``, ordered camera names, and
            episode records containing relative ``path``, ``length`` and sha256.
        history_length: Number of measured input frames including the present.
        prediction_horizon: Number of target frames including the present.
        include_partial_targets: Allow episode-tail anchors with complete input
            history and at least one target. Missing future targets are finite
            zeros with ``target_valid=False``. The default preserves complete
            future windows and their historical source identity/sampling.
        sampling_mode: Draw an episode uniformly first (the historical default),
            or draw uniformly among all eligible frames across episodes. Both
            modes sample with replacement and obey the same window boundaries.
        observation_key: Numeric HDF5 dataset containing measured observations.
        observation_slice: Optional feature slice selected before model input.
        target_key: HDF5 physical targets, when no encoder is supplied.
        target_encoder: Optional pure deterministic task-owned transform of
            aligned numeric fields with leading ``[batch,time]`` axes. Partial
            mode groups equal valid lengths before encoding, so the encoder
            never sees synthetic padding; ``time`` may be shorter than F.
        target_fields: HDF5 inputs required by the target encoder.
        target_encoder_id: Explicit versioned target-transform identity, required
            when an encoder is supplied. Change it when transform semantics change.
        max_open_files: Positive bound on cached HDF5 handles.
        verify_checksums: Verify each file once before first reading it.

    Sampling uses the caller's CPU RNG. Frame-uniform sampling stores one
    cumulative count per episode, rather than materializing every frame index.
    Only history camera frames are read; future geometry may construct targets
    but is never placed in the input tree. Every returned tensor owns storage.
    """

    def __init__(
        self,
        manifest: str | Path,
        *,
        history_length: int = 1,
        prediction_horizon: int = 1,
        include_partial_targets: bool = False,
        sampling_mode: Literal["episode_uniform", "frame_uniform"] = "episode_uniform",
        observation_key: str = "observations",
        observation_slice: slice | None = None,
        target_key: str = "targets",
        target_encoder: (
            Callable[[Mapping[str, torch.Tensor]], torch.Tensor] | None
        ) = None,
        target_fields: tuple[str, ...] = (),
        target_encoder_id: str | None = None,
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
        if type(include_partial_targets) is not bool:
            raise ValueError("include_partial_targets must be a Boolean")
        if sampling_mode not in ("episode_uniform", "frame_uniform"):
            raise ValueError(
                "BC sampling mode must be episode_uniform or frame_uniform"
            )
        self.include_partial_targets = include_partial_targets
        self.sampling_mode = sampling_mode
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
        if target_encoder is not None and (
            not isinstance(target_encoder_id, str) or not target_encoder_id.strip()
        ):
            raise ValueError(
                "Encoded BC targets require an explicit versioned transform identity"
            )
        self.cameras = tuple(self.manifest.get("cameras", ()))
        if len(set(self.cameras)) != len(self.cameras):
            raise ValueError("Offline camera names must be distinct and ordered")
        self.episodes = self.manifest["episodes"]
        minimum_length = (
            history_length
            if include_partial_targets
            else history_length + prediction_horizon - 1
        )
        if not self.episodes or any(
            record["length"] < minimum_length for record in self.episodes
        ):
            raise ValueError(
                "Every BC episode must contain a complete requested history"
                if include_partial_targets
                else "Every BC episode must contain a complete requested window"
            )
        self._frame_ends: torch.Tensor | None = None
        if sampling_mode == "frame_uniform":
            self._frame_ends = torch.tensor(
                [record["length"] - minimum_length + 1 for record in self.episodes],
                dtype=torch.int64,
            ).cumsum(0)
        # The manifest binds content hashes and all physical/camera metadata;
        # source identity cannot be changed by reusing a display source_id.
        canonical = json.dumps(self.manifest, sort_keys=True, allow_nan=False)
        self.dataset_id = hashlib.sha256(canonical.encode()).hexdigest()
        selection = (
            None
            if observation_slice is None
            else (
                observation_slice.start,
                observation_slice.stop,
                observation_slice.step,
            )
        )
        view = {
            "history_length": history_length,
            "prediction_horizon": prediction_horizon,
            "observation_key": observation_key,
            "observation_slice": selection,
            "target_key": target_key if target_encoder is None else None,
            "target_fields": target_fields,
            "target_encoder_id": target_encoder_id,
            "cameras": self.cameras,
            "numeric_cast": "float32",
            "image_transform": "uint8-thwc-rgb-to-float32-nchw-div255@1",
            "sampling": "episode-uniform-complete-windows@1",
        }
        if include_partial_targets:
            view.update(
                include_partial_targets=True,
                target_padding="finite-zero-right-padding-valid-mask@1",
                sampling="episode-uniform-complete-history-partial-targets@1",
            )
        if sampling_mode == "frame_uniform":
            view["sampling"] = (
                "frame-uniform-complete-history-partial-targets@1"
                if include_partial_targets
                else "frame-uniform-complete-windows@1"
            )
        self.view_id = hashlib.sha256(
            json.dumps(view, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        self.source_id = f"bc-hdf5:{self.dataset_id}:{self.view_id}"
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

    def _encode_partial_targets(
        self, fields: dict[str, list[torch.Tensor]], lengths: list[int]
    ) -> torch.Tensor:
        assert self.target_encoder is not None
        groups: dict[int, list[int]] = {}
        for row, length in enumerate(lengths):
            groups.setdefault(length, []).append(row)
        encoded = None
        for length, rows in groups.items():
            values = self.target_encoder(
                {
                    name: torch.stack([field_rows[row] for row in rows])
                    for name, field_rows in fields.items()
                }
            )
            if (
                not isinstance(values, torch.Tensor)
                or values.ndim != 3
                or values.shape[:2] != (len(rows), length)
                or values.shape[-1] < 1
                or values.dtype != torch.float32
                or (encoded is not None and values.shape[-1] != encoded.shape[-1])
            ):
                raise ValueError("Partial BC target encoding must return FP32 [B,T,K]")
            if encoded is None:
                encoded = values.new_zeros(
                    len(lengths), self.prediction_horizon, values.shape[-1]
                )
            for offset, row in enumerate(rows):
                encoded[row, :length].copy_(values[offset])
        assert encoded is not None
        return encoded

    @torch.no_grad()
    def sample(self, batch_size: int, *, generator: torch.Generator) -> dict[str, Any]:
        """Draw complete causal histories and optionally masked tail targets.

        Args:
            batch_size: Positive number of independent windows.
            generator: Explicit CPU sampling RNG, checkpointed by the trainer.

        Returns:
            Owned CPU batch. Numeric input history is ``[B,H,D]``; images are
            FP32 ``[B,H,3,256,256]`` in ``[0,1]``; physical targets are ``[B,F,K]``.
            In partial mode each row has at least one valid target and finite
            zero padding. ``target_valid[B,F]`` excludes unavailable futures.
        """
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("BC batch size must be a positive integer")
        if self._frame_ends is None:
            ordinals = torch.randint(
                len(self.episodes), (batch_size,), generator=generator
            )
            anchor_draws = torch.rand(batch_size, generator=generator)
            anchors = []
            for ordinal, draw in zip(ordinals.tolist(), anchor_draws.tolist()):
                length = self.episodes[ordinal]["length"]
                count = length - self.history_length - self.prediction_horizon + 2
                if self.include_partial_targets:
                    count = length - self.history_length + 1
                anchors.append(self.history_length - 1 + int(draw * count))
        else:
            draws = torch.randint(
                int(self._frame_ends[-1]), (batch_size,), generator=generator
            )
            ordinals = torch.searchsorted(self._frame_ends, draws, right=True)
            previous = torch.cat((self._frame_ends.new_zeros(1), self._frame_ends[:-1]))
            anchors = (draws - previous[ordinals] + self.history_length - 1).tolist()
        target_lengths = []
        observations, targets = [], []
        fields: dict[str, list[torch.Tensor]] = {
            name: [] for name in self.target_fields
        }
        images = {
            camera: torch.empty(
                batch_size, self.history_length, 3, 256, 256, dtype=torch.float32
            )
            for camera in self.cameras
        }
        for row, (ordinal, anchor) in enumerate(zip(ordinals.tolist(), anchors)):
            length = self.episodes[ordinal]["length"]
            start = anchor - self.history_length + 1
            stop = min(anchor + self.prediction_horizon, length)
            target_length = stop - anchor
            target_lengths.append(target_length)
            handle = self._handle(ordinal)
            observation = torch.from_numpy(
                handle[self.observation_key][start : anchor + 1]
            )
            if self.observation_slice is not None:
                observation = observation[..., self.observation_slice]
            observations.append(observation.float())
            if self.target_encoder is None:
                target = torch.from_numpy(handle[self.target_key][anchor:stop]).float()
                if self.include_partial_targets:
                    if len(target) != target_length:
                        raise ValueError("BC targets must align with episode frames")
                    padded = target.new_zeros(
                        self.prediction_horizon, *target.shape[1:]
                    )
                    padded[:target_length].copy_(target)
                    target = padded
                targets.append(target)
            else:
                for name in self.target_fields:
                    value = torch.from_numpy(handle[name][anchor:stop]).float()
                    if self.include_partial_targets and len(value) != target_length:
                        raise ValueError(
                            "BC target fields must align with episode frames"
                        )
                    fields[name].append(value)
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
                images[camera][row].copy_(rgb.permute(0, 3, 1, 2))
                images[camera][row].div_(255)
        inputs: dict[str, Any] = {"history": torch.stack(observations)}
        if images:
            inputs["images"] = images
        if self.target_encoder is None:
            encoded = torch.stack(targets)
        elif self.include_partial_targets:
            encoded = self._encode_partial_targets(fields, target_lengths)
        else:
            encoded = self.target_encoder(
                {name: torch.stack(values) for name, values in fields.items()}
            )
        if (
            not isinstance(encoded, torch.Tensor)
            or encoded.ndim != 3
            or encoded.shape[:2] != (batch_size, self.prediction_horizon)
            or encoded.shape[-1] < 1
            or encoded.dtype != torch.float32
        ):
            raise ValueError("BC target encoding must return FP32 [B,F,K] vectors")
        if self.target_encoder is not None:
            encoded = encoded.detach().clone()
        target_valid = torch.ones(batch_size, self.prediction_horizon, dtype=torch.bool)
        if self.include_partial_targets:
            target_valid = (
                torch.arange(self.prediction_horizon)[None, :]
                < torch.tensor(target_lengths)[:, None]
            )
        return {
            "inputs": inputs,
            "targets": encoded,
            "masks": {"target_valid": target_valid},
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
