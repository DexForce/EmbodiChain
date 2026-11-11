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

"""Read-only segment state from DexSim's public cable batch."""

from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Sequence

import torch

if TYPE_CHECKING:
    from dexsim.scene import Scene, SpawnedCable

__all__ = ["CableObjectData"]


class CableObjectData:
    """Independent world-state snapshots for one replicated cable asset.

    Segment poses use EmbodiChain order ``xyz + xyzw``; velocities are world
    COM linear then angular. World positions already include Arena origins.
    All instances of one asset must share the same segment count. Reading
    state never advances physics and never exposes writable simulation views.

    Args:
        entities: Replicated cable handles in environment order.
        scene: Finalized Scene owning those handles.
        device: Device of the Newton cable state.
    """

    def __init__(
        self,
        entities: Sequence[SpawnedCable],
        scene: Scene,
        device: torch.device,
    ) -> None:
        self.entities = list(entities)
        if not self.entities:
            raise ValueError("A cable batch cannot be empty.")
        counts = tuple(entity.segment_count for entity in self.entities)
        if min(counts) <= 0 or len(set(counts)) != 1:
            raise ValueError(
                "Instances of one cable asset require equal positive segment "
                f"counts, got {counts}."
            )
        self.scene = scene
        self.device = torch.device(device)
        self.num_instances = len(self.entities)
        self.num_segments = counts[0]
        self.batch = scene.create_cable_batch(self.entities)
        self._pose_buffer = torch.empty(
            (self.num_instances, self.num_segments, 7),
            dtype=torch.float32,
            device=self.device,
        )
        self._velocity_buffer = torch.empty(
            (self.num_instances, self.num_segments, 6),
            dtype=torch.float32,
            device=self.device,
        )
        self._state_step = 0
        self.batch.fetch_default_state(self._pose_buffer, self._velocity_buffer)
        self._default_segment_state_w = torch.cat(
            (
                self._pose_buffer[..., 4:],
                self._pose_buffer[..., :4],
                self._velocity_buffer,
            ),
            -1,
        )

    def _fetch_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._state_step = self.batch.fetch_state(
            self._pose_buffer, self._velocity_buffer
        )
        pose = torch.cat((self._pose_buffer[..., 4:], self._pose_buffer[..., :4]), -1)
        return pose, self._velocity_buffer.clone()

    @property
    def state_step(self) -> int:
        """Outer simulation step of the most recently fetched snapshot."""
        return self._state_step

    @property
    def segment_pose_w(self) -> torch.Tensor:
        """World segment poses ``(B, S, 7)`` in ``xyz + xyzw`` order."""
        return self._fetch_state()[0]

    @property
    def segment_pos_w(self) -> torch.Tensor:
        """World segment-centre positions ``(B, S, 3)`` [m]."""
        return self.segment_pose_w[..., :3]

    @property
    def segment_quat_w(self) -> torch.Tensor:
        """World material-frame quaternions ``(B, S, 4)`` in xyzw order."""
        return self.segment_pose_w[..., 3:]

    @property
    def segment_vel_w(self) -> torch.Tensor:
        """World COM linear/angular velocities ``(B, S, 6)``."""
        return self._fetch_state()[1]

    @property
    def segment_lin_vel_w(self) -> torch.Tensor:
        """World COM linear velocities ``(B, S, 3)`` [m/s]."""
        return self.segment_vel_w[..., :3]

    @property
    def segment_ang_vel_w(self) -> torch.Tensor:
        """World angular velocities ``(B, S, 3)`` [rad/s]."""
        return self.segment_vel_w[..., 3:]

    @property
    def segment_state_w(self) -> torch.Tensor:
        """One coherent ``[xyz, xyzw, linear, angular]`` snapshot ``(B,S,13)``."""
        pose, velocity = self._fetch_state()
        return torch.cat((pose, velocity), -1)

    @property
    def default_segment_state_w(self) -> torch.Tensor:
        """Immutable creation state; this is not a solver-history checkpoint."""
        return self._default_segment_state_w.clone()
