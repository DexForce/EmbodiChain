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

"""Owned causal point-cloud windows from a fixed resident geometry corpus."""

from __future__ import annotations

import math

import torch

__all__ = ["RigidBodySequenceSource"]


class RigidBodySequenceSource:
    """Sample previous/current/next geometry without crossing episode bounds.

    Args:
        positions: Float CPU points with axes ``[episode, time, object, point, 3]``.
        properties: Float CPU attributes ``[episode, object, attributes]`` or
            ``[episode, object, point, attributes]``.
        step_dt: Seconds between stored frames.
        source_id: Immutable corpus identity recorded with downstream batches.
        stride: Number of stored frames between model input steps.
        episode_lengths: Optional number of real frames per episode.
        object_lens: Optional number of real objects per episode.
        object_point_lens: Optional number of real points per episode/object.
        max_resident_bytes: Bound on cloned resident tensors.
        max_batch_bytes: Bound on returned geometry payloads.

    Points retain their material identities across time. Every returned tensor
    owns its data independently of the source and later sampling. Sampler RNG
    is caller-owned and can be restored with a learner checkpoint.
    """

    def __init__(
        self,
        positions: torch.Tensor,
        properties: torch.Tensor,
        *,
        step_dt: float,
        source_id: str,
        stride: int = 1,
        episode_lengths: torch.Tensor | None = None,
        object_lens: torch.Tensor | None = None,
        object_point_lens: torch.Tensor | None = None,
        max_resident_bytes: int = 512 * 1024 * 1024,
        max_batch_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        if positions.ndim != 5 or positions.shape[-1] != 3:
            raise ValueError("positions must be [episode, time, object, point, 3]")
        if min(positions.shape[:-1]) < 1 or not positions.is_floating_point():
            raise ValueError("positions must be nonempty floating-point tensors")
        if (
            properties.ndim not in (3, 4)
            or properties.shape[:2] != (positions.shape[0], positions.shape[2])
            or (properties.ndim == 4 and properties.shape[2] != positions.shape[3])
            or not properties.is_floating_point()
        ):
            raise ValueError("properties must match the episode/object/point layout")
        if type(stride) is not int or stride < 1:
            raise ValueError("stride must be a positive integer")
        if not math.isfinite(step_dt) or step_dt <= 0 or not source_id:
            raise ValueError("Positive finite step_dt and source_id are required")
        tensors = [positions, properties]
        tensors += [
            v
            for v in (episode_lengths, object_lens, object_point_lens)
            if v is not None
        ]
        if any(v.device.type != "cpu" for v in tensors):
            raise ValueError("Resident geometry must be on CPU")
        lengths = (
            torch.full((positions.shape[0],), positions.shape[1], dtype=torch.long)
            if episode_lengths is None
            else episode_lengths
        )
        if (
            lengths.shape != positions.shape[:1]
            or lengths.dtype != torch.long
            or (lengths > positions.shape[1]).any()
            or (lengths < 2 * stride + 1).any()
        ):
            raise ValueError("Every episode needs enough valid frames for one window")
        if object_lens is not None and (
            object_lens.shape != positions.shape[:1]
            or object_lens.dtype != torch.long
            or (object_lens < 1).any()
            or (object_lens > positions.shape[2]).any()
        ):
            raise ValueError("Invalid object_lens")
        if object_point_lens is not None and (
            object_point_lens.shape != (positions.shape[0], positions.shape[2])
            or object_point_lens.dtype != torch.long
            or (object_point_lens < 1).any()
            or (object_point_lens > positions.shape[3]).any()
        ):
            raise ValueError("Invalid object_point_lens")
        resident = [positions, properties, lengths]
        resident += [v for v in (object_lens, object_point_lens) if v is not None]
        required = sum(v.numel() * v.element_size() for v in resident)
        if required > max_resident_bytes or max_batch_bytes < 1:
            raise ValueError("Geometry exceeds the configured memory budget")
        self.positions, self.properties = positions.clone(), properties.clone()
        self.episode_lengths = lengths.clone()
        self.object_lens = None if object_lens is None else object_lens.clone()
        self.object_point_lens = (
            None if object_point_lens is None else object_point_lens.clone()
        )
        self.step_dt, self.stride, self.source_id = float(step_dt), stride, source_id
        self.max_batch_bytes = max_batch_bytes

    def get_batch(
        self, episode_indices: torch.Tensor, anchors: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return owned windows at explicit current-frame anchors.

        Args:
            episode_indices: CPU int64 episode indices, ``[B]``.
            anchors: CPU int64 current-frame indices, ``[B]``.

        Returns:
            Upstream-compatible previous/current/reference/target geometry.
        """
        if (
            episode_indices.ndim != 1
            or episode_indices.numel() < 1
            or anchors.shape != episode_indices.shape
            or any(
                v.dtype != torch.long or v.device.type != "cpu"
                for v in (episode_indices, anchors)
            )
        ):
            raise ValueError(
                "Episode and anchor indices must be nonempty CPU int64 vectors"
            )
        if (episode_indices < 0).any() or (
            episode_indices >= self.positions.shape[0]
        ).any():
            raise ValueError("Episode index out of range")
        lengths = self.episode_lengths[episode_indices]
        if (anchors < self.stride).any() or (anchors + self.stride >= lengths).any():
            raise ValueError("Geometry windows cannot cross an episode boundary")
        batch = episode_indices.numel()
        point_bytes = self.positions[0, 0].numel() * self.positions.element_size()
        property_bytes = self.properties[0].numel() * self.properties.element_size()
        row_bytes = (
            4 * point_bytes + property_bytes + self.positions.element_size() + 16
        )
        row_bytes += 8 if self.object_lens is not None else 0
        row_bytes += (
            8 * self.positions.shape[2] if self.object_point_lens is not None else 0
        )
        if batch * row_bytes > self.max_batch_bytes:
            raise ValueError("Returned geometry exceeds max_batch_bytes")
        previous = self.positions[episode_indices, anchors - self.stride]
        result = {
            "object_pos_prev": previous,
            "object_pos": self.positions[episode_indices, anchors],
            "object_pos_next": self.positions[episode_indices, anchors + self.stride],
            "object_first_frame_pos": previous.clone(),
            "vertex_properties": self.properties[episode_indices],
            "delta_times": torch.full(
                (batch,), self.step_dt * self.stride, dtype=self.positions.dtype
            ),
            "episode_index": episode_indices.clone(),
            "frame_index": anchors.clone(),
        }
        for key in ("object_lens", "object_point_lens"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value[episode_indices]
        return result

    def sample_batch(
        self, batch_size: int, *, generator: torch.Generator | None = None
    ) -> dict[str, torch.Tensor]:
        """Sample episode-uniform windows with a caller-owned CPU RNG.

        Args:
            batch_size: Positive number of windows.
            generator: Optional CPU sampling generator.

        Returns:
            Owned geometry windows in the same layout as :meth:`get_batch`.
        """
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be positive")
        episodes = torch.randint(
            self.positions.shape[0], (batch_size,), generator=generator
        )
        count = self.episode_lengths[episodes] - 2 * self.stride
        anchors = (
            self.stride + (torch.rand(batch_size, generator=generator) * count).long()
        )
        return self.get_batch(episodes, anchors)
