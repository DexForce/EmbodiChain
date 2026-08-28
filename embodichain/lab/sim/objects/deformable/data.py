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

"""Nodal data contract and Newton particle-set state adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

import torch

if TYPE_CHECKING:
    from dexsim.scene import Scene, SpawnedParticleSet

__all__ = ["DeformableObjectData"]


class DeformableObjectData(ABC):
    """Common nodal-state view for volume and surface deformables.

    Positions and velocities use the simulation world frame. Consumers can
    rely on a stable ``(num_instances, num_nodes, 3)`` contract.
    """

    @property
    @abstractmethod
    def nodal_pos_w(self) -> torch.Tensor:
        """Return current simulation-node positions in world frame."""

    @property
    @abstractmethod
    def nodal_vel_w(self) -> torch.Tensor:
        """Return current simulation-node velocities in world frame."""

    @property
    @abstractmethod
    def default_nodal_state_w(self) -> torch.Tensor:
        """Return default nodal state ``[position, velocity]`` in world frame."""

    @property
    def nodal_state_w(self) -> torch.Tensor:
        """Return current nodal state ``[position, velocity]`` in world frame."""
        return torch.cat((self.nodal_pos_w, self.nodal_vel_w), dim=-1)

    @property
    def root_pos_w(self) -> torch.Tensor:
        """Return the mean nodal position for each deformable instance."""
        return self.nodal_pos_w.mean(dim=1)

    @property
    def root_vel_w(self) -> torch.Tensor:
        """Return the mean nodal velocity for each deformable instance."""
        return self.nodal_vel_w.mean(dim=1)


class _ParticleSetData(DeformableObjectData):
    """Packed state adapter over DexSim 0.5 particle-set handles."""

    def __init__(
        self,
        entities: Sequence[SpawnedParticleSet],
        scene: Scene,
        device: torch.device,
    ) -> None:
        self.entities = list(entities)
        if not self.entities:
            raise ValueError("A deformable particle-set batch cannot be empty.")

        self.scene = scene
        self.device = device
        self.num_instances = len(self.entities)
        particle_counts = tuple(int(entity.particle_count) for entity in self.entities)
        if any(count <= 0 for count in particle_counts):
            raise ValueError("Deformable particle sets must contain particles.")
        if len(set(particle_counts)) != 1:
            raise ValueError(
                "All instances of one deformable asset must have the same "
                f"particle count, got {particle_counts}."
            )

        self.n_nodes = particle_counts[0]
        self.batch = scene.create_particle_set_batch(self.entities)
        self._position_buffer = torch.empty(
            (self.num_instances, self.n_nodes, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self._velocity_buffer = torch.empty_like(self._position_buffer)
        default_positions = self.nodal_pos_w
        default_velocities = self.nodal_vel_w
        self._default_nodal_state_w = torch.cat(
            (default_positions, default_velocities),
            dim=-1,
        )

    @staticmethod
    def _check_batch_status(status: int | None, operation: str) -> None:
        if status is not None and int(status) < 0:
            raise RuntimeError(
                f"DexSim particle batch failed to {operation}: status {status}."
            )

    @property
    def nodal_pos_w(self) -> torch.Tensor:
        """Return current Newton particle positions in world frame."""
        status = self.batch.fetch_particle_positions(
            self._position_buffer.reshape(-1, 3)
        )
        self._check_batch_status(status, "fetch positions")
        return self._position_buffer.clone()

    @property
    def nodal_vel_w(self) -> torch.Tensor:
        """Return current Newton particle velocities in world frame."""
        status = self.batch.fetch_particle_velocities(
            self._velocity_buffer.reshape(-1, 3)
        )
        self._check_batch_status(status, "fetch velocities")
        return self._velocity_buffer.clone()

    @property
    def default_nodal_state_w(self) -> torch.Tensor:
        """Return the particle state captured when Spawn was bound."""
        return self._default_nodal_state_w.clone()

    def _apply_nodal_state(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        env_ids: Sequence[int],
    ) -> None:
        """Apply packed state to selected particle-set instances."""
        env_ids = [int(env_id) for env_id in env_ids]
        if not env_ids:
            return
        if len(set(env_ids)) != len(env_ids):
            raise ValueError(f"env_ids must not contain duplicates, got {env_ids}.")

        expected_shape = (len(env_ids), self.n_nodes, 3)
        if tuple(positions.shape) != expected_shape:
            raise ValueError(
                f"positions must have shape {expected_shape}, got "
                f"{tuple(positions.shape)}."
            )
        if tuple(velocities.shape) != expected_shape:
            raise ValueError(
                f"velocities must have shape {expected_shape}, got "
                f"{tuple(velocities.shape)}."
            )

        if env_ids == list(range(self.num_instances)):
            batch = self.batch
        else:
            batch = self.scene.create_particle_set_batch(
                [self.entities[env_id] for env_id in env_ids]
            )
        packed_positions = (
            positions.to(
                device=self.device,
                dtype=torch.float32,
            )
            .contiguous()
            .reshape(-1, 3)
        )
        packed_velocities = (
            velocities.to(
                device=self.device,
                dtype=torch.float32,
            )
            .contiguous()
            .reshape(-1, 3)
        )
        position_status = batch.apply_particle_positions(packed_positions)
        self._check_batch_status(position_status, "apply positions")
        velocity_status = batch.apply_particle_velocities(packed_velocities)
        self._check_batch_status(velocity_status, "apply velocities")
