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

"""Bounded HDF5 episode windows for visual dynamics training."""

from __future__ import annotations

import bisect
import hashlib
import json
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
import torch

from embodichain.learning.sl import SupervisedBatch

__all__ = ["LeWMSequenceSource"]


class LeWMSequenceSource:
    """Sample complete windows from a single immutable, split-owned corpus.

    Args:
        path: HDF5 corpus with metadata and an ``episodes`` group.
        split: Required corpus split: train, validation or test.
        history_size: Number of observed frames in each window.
        horizon: Number of future frames after the observed history.
        action_block: Raw timesteps between observed frames. All intervening
            actions are concatenated in time order into one model action.
        cache_device: Optional device for an immutable uint8 image/action cache.
        cache_max_bytes: Maximum persistent cache bytes per corpus. Required
            when caching; batch outputs and model activations are additional.

    Each episode has ``pixels`` as uint8 HWC frames and ``actions`` as raw
    float32 controls. It contains exactly one more frame than action, and every
    action leaves the corresponding frame. Windows never cross episodes.
    """

    def __init__(
        self,
        path: str | Path,
        split: str,
        history_size: int = 3,
        horizon: int = 1,
        *,
        action_block: int = 1,
        cache_device: str | None = None,
        cache_max_bytes: int = 0,
    ) -> None:
        self.path = Path(path).resolve()
        if (
            split not in {"train", "validation", "test"}
            or type(history_size) is not int
            or type(horizon) is not int
            or type(action_block) is not int
            or history_size < 1
            or horizon < 1
            or action_block < 1
        ):
            raise ValueError("invalid split or temporal window")
        self.split, self.history_size, self.horizon = split, history_size, horizon
        self.action_block = action_block
        digest = hashlib.sha256()
        with self.path.open("rb") as file:
            for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
                digest.update(block)
        self.source_id = (
            f"lewm-hdf5:{digest.hexdigest()}:{split}:{history_size}:{horizon}"
        )
        if action_block != 1:
            self.source_id += f":block={action_block}"
        self.episodes: list[str] = []
        self.ends: list[int] = []
        self.seeds: set[int] = set()
        self.metadata: dict = {}
        with h5py.File(self.path, "r") as file:
            if (
                file.attrs.get("format") != "embodichain.lewm.trajectory.v1"
                or file.attrs.get("split") != split
            ):
                raise ValueError(
                    "HDF5 format or split differs from the requested corpus"
                )
            self.metadata = json.loads(file.attrs["metadata"])
            if not self.metadata.get("action_semantics") or not 0 < float(
                self.metadata.get("step_dt", 0)
            ) < float("inf"):
                raise ValueError(
                    "corpus requires explicit action semantics and positive step_dt"
                )
            total = 0
            image_shape = None
            for key in sorted(file["episodes"]):
                episode = file["episodes"][key]
                pixels, actions = episode["pixels"], episode["actions"]
                if (
                    pixels.ndim != 4
                    or pixels.dtype != np.uint8
                    or pixels.shape[-1] != 3
                ):
                    raise ValueError(f"episode {key} pixels must be uint8 RGB HWC")
                if image_shape is None:
                    image_shape = pixels.shape[1:]
                elif pixels.shape[1:] != image_shape:
                    raise ValueError("all episode frames must have the same RGB shape")
                if (
                    actions.ndim != 2
                    or actions.dtype != np.float32
                    or len(pixels) != len(actions) + 1
                ):
                    raise ValueError(
                        f"episode {key} must have T actions and T+1 frames"
                    )
                if actions.shape[1] != self.metadata.get("action_dim"):
                    raise ValueError(
                        f"episode {key} action width differs from metadata"
                    )
                if "timestamp" in episode:
                    times = np.asarray(episode["timestamp"], dtype=np.float64)
                    if (
                        times.shape != (len(pixels),)
                        or not np.isfinite(times).all()
                        or not np.allclose(
                            np.diff(times), self.metadata["step_dt"], rtol=0, atol=1e-8
                        )
                    ):
                        raise ValueError(
                            f"episode {key} timestamps differ from step_dt"
                        )
                seed = int(episode.attrs["seed"])
                if seed in self.seeds:
                    raise ValueError("episode seeds must be unique within a split")
                self.seeds.add(seed)
                count = len(actions) - (history_size + horizon - 1) * action_block + 1
                if count > 0:
                    self.episodes.append(key)
                    total += count
                    self.ends.append(total)
        if not self.ends:
            raise ValueError("corpus has no complete windows")
        self.model_action_dim: int = self.metadata["action_dim"] * action_block
        if action_block != 1:
            self.metadata["action_block"] = action_block
        self.cache_bytes = 0
        self._cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        if cache_device is not None:
            self._load_cache(cache_device, cache_max_bytes)

    def _load_cache(self, device: str, max_bytes: int) -> None:
        """Preflight resident storage, then stream bounded chunks into it."""
        with h5py.File(self.path, "r") as file:
            episodes = file["episodes"]
            total_frames = sum(len(episodes[key]["pixels"]) for key in self.episodes)
            shape = episodes[self.episodes[0]]["pixels"].shape[1:]
            raw_dim = self.metadata["action_dim"]
            pixel_bytes = int(np.prod(shape))
            required = total_frames * (pixel_bytes + raw_dim * 4) + len(self) * 8
            if type(max_bytes) is not int or max_bytes < required:
                raise ValueError(f"LeWM cache budget needs at least {required} bytes")
            pixels = torch.empty(
                (total_frames, *shape), dtype=torch.uint8, device=device
            )
            controls = torch.zeros(
                (total_frames, raw_dim), dtype=torch.float32, device=device
            )
            starts = []
            offset = 0
            width = (self.history_size + self.horizon - 1) * self.action_block
            for key in self.episodes:
                episode = episodes[key]
                count = len(episode["pixels"])
                for start in range(0, count, 64):
                    chunk = torch.from_numpy(episode["pixels"][start : start + 64])
                    pixels[offset + start : offset + start + len(chunk)].copy_(chunk)
                actions = episode["actions"][:]
                if not np.isfinite(actions).all():
                    raise ValueError("corpus actions must be finite")
                controls[offset : offset + len(actions)].copy_(
                    torch.from_numpy(actions)
                )
                starts.append(torch.arange(offset, offset + len(actions) - width + 1))
                offset += count
            self._cache = pixels, controls, torch.cat(starts).to(device)
            self.cache_bytes = required

    def __len__(self) -> int:
        """Return the number of complete windows."""
        return self.ends[-1]

    def _batch(self, indices: list[int]) -> SupervisedBatch:
        images, actions = [], []
        width = self.history_size + self.horizon - 1
        raw_width = width * self.action_block
        if self._cache is not None:
            pixels, controls, starts = self._cache
            selected = starts[torch.tensor(indices, device=starts.device)][:, None]
            frame_offsets = (
                torch.arange(width + 1, device=starts.device) * self.action_block
            )
            action_offsets = torch.arange(raw_width, device=starts.device)
            return SupervisedBatch(
                inputs={
                    "pixels": pixels[selected + frame_offsets],
                    "actions": controls[selected + action_offsets].reshape(
                        len(indices), width, self.model_action_dim
                    ),
                },
                targets=None,
                metadata={"source_id": self.source_id},
            )
        with h5py.File(self.path, "r") as file:
            for index in indices:
                position = bisect.bisect_right(self.ends, index)
                start = index - (self.ends[position - 1] if position else 0)
                episode = file["episodes"][self.episodes[position]]
                images.append(
                    episode["pixels"][start : start + raw_width + 1 : self.action_block]
                )
                window_actions = episode["actions"][start : start + raw_width]
                if not np.isfinite(window_actions).all():
                    raise ValueError("corpus actions must be finite")
                actions.append(window_actions.reshape(width, self.model_action_dim))
        return SupervisedBatch(
            inputs={
                "pixels": torch.from_numpy(np.stack(images)),
                "actions": torch.from_numpy(np.stack(actions)),
            },
            targets=None,
            metadata={"source_id": self.source_id},
        )

    def sample(self, batch_size: int, *, generator: torch.Generator) -> SupervisedBatch:
        """Draw independently owned windows using the caller's CPU RNG.

        Args:
            batch_size: Positive number of windows.
            generator: Explicit CPU sampling generator.

        Returns:
            A batch with images and aligned raw actions.
        """
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be positive")
        return self._batch(
            torch.randint(len(self), (batch_size,), generator=generator).tolist()
        )

    def iter_batches(self, batch_size: int) -> Iterator[SupervisedBatch]:
        """Iterate each complete window once in deterministic episode order.

        Args:
            batch_size: Maximum windows in one batch.

        Yields:
            A batch of owned image/action windows.
        """
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be positive")
        for start in range(0, len(self), batch_size):
            yield self._batch(list(range(start, min(start + batch_size, len(self)))))

    def action_statistics(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute finite sample moments from raw actions in this train split.

        Returns:
            CPU float32 mean and standard deviation for each action component.
        """
        if self.split != "train":
            raise ValueError("action moments may only be computed from training data")
        count = 0
        sums = np.zeros(self.metadata["action_dim"], dtype=np.float64)
        squares = np.zeros_like(sums)
        with h5py.File(self.path, "r") as file:
            for key in file["episodes"]:
                data = file["episodes"][key]["actions"]
                for start in range(0, len(data), 4096):
                    values = np.asarray(data[start : start + 4096], dtype=np.float64)
                    if not np.isfinite(values).all():
                        raise ValueError("training actions must be finite")
                    count += len(values)
                    sums += values.sum(axis=0)
                    squares += (values * values).sum(axis=0)
        if count < 2:
            raise ValueError("at least two actions are required for normalization")
        mean = sums / count
        variance = np.maximum((squares - count * mean * mean) / (count - 1), 0)
        # A constant action dimension carries no scale information.
        std = np.sqrt(variance)
        std[std < 1e-6] = 1.0
        return (
            torch.from_numpy(mean.astype(np.float32)).repeat(self.action_block),
            torch.from_numpy(std.astype(np.float32)).repeat(self.action_block),
        )
