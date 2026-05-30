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
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence
from functools import cached_property

import torch

__all__ = ["RigidBodyViewBase"]


class RigidBodyViewBase(ABC):
    """Abstract interface for physics-backend rigid body data access.

    All pose/velocity/acceleration data uses EmbodiChain convention:
    ``(x, y, z, qx, qy, qz, qw)``.
    """

    # -- Lifecycle & State --------------------------------------------------

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Whether the backend simulation is finalized and data can be accessed."""
        ...

    # -- Body ID Management -------------------------------------------------

    @cached_property
    @abstractmethod
    def body_ids(self) -> list[int]:
        """Backend body IDs for all managed entities."""
        ...

    @cached_property
    @abstractmethod
    def body_ids_tensor(self) -> torch.Tensor:
        """Body IDs as an int32 tensor on ``device``."""
        ...

    @abstractmethod
    def select_body_ids(self, indices: Sequence[int] | torch.Tensor) -> torch.Tensor:
        """Return body IDs for the given entity indices."""
        ...

    # -- Pose ---------------------------------------------------------------

    @abstractmethod
    def fetch_pose(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        """Fetch poses into ``data`` as ``(N, 7)`` in ``(x, y, z, qx, qy, qz, qw)``."""
        ...

    @abstractmethod
    def apply_pose(self, pose: torch.Tensor, body_ids: torch.Tensor) -> None:
        """Apply poses from ``(N, 7)`` tensor in ``(x, y, z, qx, qy, qz, qw)``."""
        ...

    # -- Center of Mass (local) ---------------------------------------------

    @abstractmethod
    def fetch_com_local_pose(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        """Fetch center-of-mass local poses into ``data`` as ``(N, 7)``."""
        ...

    @abstractmethod
    def apply_com_local_pose(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        """Apply center-of-mass local poses from ``(N, 7)`` tensor."""
        ...

    # -- Velocity -----------------------------------------------------------

    @abstractmethod
    def fetch_linear_velocity(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        """Fetch linear velocities into ``data`` as ``(N, 3)``."""
        ...

    @abstractmethod
    def fetch_angular_velocity(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        """Fetch angular velocities into ``data`` as ``(N, 3)``."""
        ...

    @abstractmethod
    def apply_linear_velocity(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        """Set linear velocities from ``(N, 3)`` tensor."""
        ...

    @abstractmethod
    def apply_angular_velocity(
        self, data: torch.Tensor, body_ids: torch.Tensor
    ) -> None:
        """Set angular velocities from ``(N, 3)`` tensor."""
        ...

    # -- Acceleration -------------------------------------------------------

    @abstractmethod
    def fetch_linear_acceleration(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        """Fetch linear accelerations into ``data`` as ``(N, 3)``."""
        ...

    @abstractmethod
    def fetch_angular_acceleration(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        """Fetch angular accelerations into ``data`` as ``(N, 3)``."""
        ...

    # -- Force & Torque -----------------------------------------------------

    @abstractmethod
    def apply_force(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        """Apply external forces ``(N, 3)``.  One-shot — consumed on next step."""
        ...

    @abstractmethod
    def apply_torque(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        """Apply external torques ``(N, 3)``.  One-shot — consumed on next step."""
        ...

    # -- Physical Properties -------------------------------------------------

    @abstractmethod
    def fetch_mass(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        """Fetch masses into ``data`` as ``(N, 1)``."""
        ...

    @abstractmethod
    def apply_mass(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        """Apply masses from ``(N, 1)`` tensor."""
        ...

    @abstractmethod
    def fetch_inertia_diagonal(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        """Fetch inertia diagonals into ``data`` as ``(N, 3)``."""
        ...

    @abstractmethod
    def apply_inertia_diagonal(
        self, data: torch.Tensor, body_ids: torch.Tensor
    ) -> None:
        """Apply inertia diagonals from ``(N, 3)`` tensor."""
        ...

    @abstractmethod
    def fetch_friction(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        """Fetch friction coefficients into ``data`` as ``(N, 1)``."""
        ...

    @abstractmethod
    def apply_friction(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        """Apply friction coefficients from ``(N, 1)`` tensor."""
        ...

    @abstractmethod
    def fetch_restitution(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        """Fetch restitution coefficients into ``data`` as ``(N, 1)``."""
        ...

    @abstractmethod
    def apply_restitution(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        """Apply restitution coefficients from ``(N, 1)`` tensor."""
        ...
