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

"""Backend-neutral data contract for deformable simulation objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import torch
from dexsim.scene import Scene, SpawnedParticleSet

__all__ = ["DeformableObjectData"]


class DeformableObjectData(ABC):
    """Common nodal-state view for volume and surface deformables.

    Positions and velocities use the simulation world frame. Concrete
    backends own how the buffers are fetched; consumers can rely on a stable
    ``(num_instances, num_nodes, 3)`` contract.
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

    @abstractmethod
    def apply_nodal_state_w(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
    ) -> None:
        """Write simulation-node positions and velocities in world frame."""

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
    """Packed DexSim particle-set state behind the common nodal contract."""

    def __init__(
        self,
        entities: Sequence[SpawnedParticleSet],
        scene: Scene,
        device: torch.device,
    ) -> None:
        if not isinstance(scene, Scene):
            raise TypeError("Particle-set data requires a finalized DexSim Scene.")
        if not entities:
            raise ValueError("Particle-set data requires at least one Spawn handle.")

        self.entities = tuple(entities)
        self.scene = scene
        self.device = device
        self.num_instances = len(self.entities)
        self.particle_batch = scene.create_particle_set_batch(list(self.entities))
        particle_counts = self.particle_batch.particle_counts
        if len(set(particle_counts)) != 1:
            raise ValueError(
                "A deformable batch requires each SpawnedParticleSet to have "
                f"the same node count, got {particle_counts!r}."
            )

        self.num_nodes = particle_counts[0]
        self._nodal_position = torch.empty(
            (self.num_instances, self.num_nodes, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self._nodal_velocity = torch.empty_like(self._nodal_position)
        self.particle_batch.fetch_particle_positions(
            self._nodal_position.reshape(-1, 3)
        )
        self._default_nodal_state_w = torch.cat(
            (self._nodal_position, torch.zeros_like(self._nodal_position)),
            dim=-1,
        )

    @property
    def nodal_pos_w(self) -> torch.Tensor:
        """Return current simulation-node positions in world frame."""
        self.particle_batch.fetch_particle_positions(
            self._nodal_position.reshape(-1, 3)
        )
        return self._nodal_position.clone()

    @property
    def nodal_vel_w(self) -> torch.Tensor:
        """Return current simulation-node velocities in world frame."""
        self.particle_batch.fetch_particle_velocities(
            self._nodal_velocity.reshape(-1, 3)
        )
        return self._nodal_velocity.clone()

    @property
    def default_nodal_state_w(self) -> torch.Tensor:
        """Return the state captured immediately after Spawn materialization."""
        return self._default_nodal_state_w.clone()

    def apply_nodal_state_w(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
    ) -> None:
        """Write a complete, uniformly-shaped particle-set batch state."""
        expected_shape = (self.num_instances, self.num_nodes, 3)
        for name, value in (("positions", positions), ("velocities", velocities)):
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"Expected {name} shape {expected_shape}, got {tuple(value.shape)}."
                )
        self.particle_batch.apply_particle_positions(positions.reshape(-1, 3))
        self.particle_batch.apply_particle_velocities(velocities.reshape(-1, 3))
