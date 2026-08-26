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

import torch

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
