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

    @property
    @abstractmethod
    def body_ids(self) -> list[int]:
        """Backend body IDs for all managed entities."""
        ...

    @property
    @abstractmethod
    def body_ids_tensor(self) -> torch.Tensor:
        """Body IDs as an int32 tensor on ``device``."""
        ...

    @abstractmethod
    def select_body_ids(self, indices: Sequence[int] | torch.Tensor) -> list[int]:
        """Return body IDs for the given entity indices."""
        ...

    # -- Pose ---------------------------------------------------------------

    @abstractmethod
    def fetch_pose(self, body_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Fetch poses as ``(N, 7)`` tensor in ``(x, y, z, qx, qy, qz, qw)``."""
        ...

    @abstractmethod
    def apply_pose(self, pose: torch.Tensor, body_ids: Sequence[int]) -> None:
        """Apply poses from ``(N, 7)`` tensor in ``(x, y, z, qx, qy, qz, qw)``."""
        ...

    # -- Velocity -----------------------------------------------------------

    @abstractmethod
    def fetch_linear_velocity(
        self, body_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        """Fetch linear velocities as ``(N, 3)`` tensor."""
        ...

    @abstractmethod
    def fetch_angular_velocity(
        self, body_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        """Fetch angular velocities as ``(N, 3)`` tensor."""
        ...

    @abstractmethod
    def apply_linear_velocity(
        self, data: torch.Tensor, body_ids: Sequence[int]
    ) -> None:
        """Set linear velocities from ``(N, 3)`` tensor."""
        ...

    @abstractmethod
    def apply_angular_velocity(
        self, data: torch.Tensor, body_ids: Sequence[int]
    ) -> None:
        """Set angular velocities from ``(N, 3)`` tensor."""
        ...

    # -- Acceleration -------------------------------------------------------

    @abstractmethod
    def fetch_linear_acceleration(
        self, body_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        """Fetch linear accelerations as ``(N, 3)`` tensor."""
        ...

    @abstractmethod
    def fetch_angular_acceleration(
        self, body_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        """Fetch angular accelerations as ``(N, 3)`` tensor."""
        ...

    # -- Force & Torque -----------------------------------------------------

    @abstractmethod
    def apply_force(self, data: torch.Tensor, body_ids: Sequence[int]) -> None:
        """Apply external forces ``(N, 3)``.  One-shot — consumed on next step."""
        ...

    @abstractmethod
    def apply_torque(self, data: torch.Tensor, body_ids: Sequence[int]) -> None:
        """Apply external torques ``(N, 3)``.  One-shot — consumed on next step."""
        ...
