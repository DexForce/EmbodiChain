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

__all__ = ["RigidBodyViewBase", "ArticulationViewBase"]


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

    @property
    def can_apply_pose(self) -> bool:
        """Whether world poses can be written through the backend view."""
        return self.is_ready

    @property
    def can_fetch_pose(self) -> bool:
        """Whether world poses can be read through the backend view."""
        return self.is_ready

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

    @abstractmethod
    def fetch_contact_offset(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        """Fetch contact offsets into ``data`` as ``(N, 1)``."""
        ...

    @abstractmethod
    def apply_contact_offset(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        """Apply contact offsets from ``(N, 1)`` tensor."""
        ...


class ArticulationViewBase(ABC):
    """Abstract interface for physics-backend articulation data access.

    Public root/link poses use EmbodiChain convention:
    ``(x, y, z, qx, qy, qz, qw)``.
    """

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Whether backend runtime data can be accessed through batch APIs."""
        ...

    @property
    def is_newton_backend(self) -> bool:
        """Whether this view targets the DexSim Newton backend."""
        return False

    @property
    @abstractmethod
    def articulation_ids_tensor(self) -> torch.Tensor | None:
        """Backend articulation ids as an int32 tensor, if the backend uses ids."""
        ...

    @abstractmethod
    def select_articulation_ids(
        self, env_ids: Sequence[int] | torch.Tensor
    ) -> torch.Tensor:
        """Return backend articulation ids for the given environment ids."""
        ...

    @abstractmethod
    def fetch_root_pose(self, data: torch.Tensor) -> torch.Tensor:
        """Fetch root poses into ``data`` and return a view/result tensor."""
        ...

    @abstractmethod
    def fetch_root_linear_velocity(self, data: torch.Tensor) -> torch.Tensor:
        """Fetch root linear velocities into ``data`` and return a tensor."""
        ...

    @abstractmethod
    def fetch_root_angular_velocity(self, data: torch.Tensor) -> torch.Tensor:
        """Fetch root angular velocities into ``data`` and return a tensor."""
        ...

    @abstractmethod
    def fetch_qpos(self, data: torch.Tensor) -> torch.Tensor:
        """Fetch current joint positions into ``data``."""
        ...

    @abstractmethod
    def fetch_target_qpos(self, data: torch.Tensor) -> torch.Tensor:
        """Fetch target joint positions into ``data``."""
        ...

    @abstractmethod
    def fetch_qvel(self, data: torch.Tensor) -> torch.Tensor:
        """Fetch current joint velocities into ``data``."""
        ...

    @abstractmethod
    def fetch_target_qvel(self, data: torch.Tensor) -> torch.Tensor:
        """Fetch target joint velocities into ``data``."""
        ...

    @abstractmethod
    def fetch_qacc(self, data: torch.Tensor) -> torch.Tensor:
        """Fetch current joint accelerations into ``data``."""
        ...

    @abstractmethod
    def fetch_qf(self, data: torch.Tensor) -> torch.Tensor:
        """Fetch current joint forces into ``data``."""
        ...

    @abstractmethod
    def fetch_link_pose(self, data: torch.Tensor) -> torch.Tensor:
        """Fetch link poses into ``data``."""
        ...

    @abstractmethod
    def fetch_link_velocity(
        self,
        data: torch.Tensor,
        linear_data: torch.Tensor,
        angular_data: torch.Tensor,
    ) -> torch.Tensor:
        """Fetch link velocities into ``data`` using provided scratch buffers."""
        ...

    @abstractmethod
    def apply_root_pose(
        self, pose: torch.Tensor, env_ids: Sequence[int] | torch.Tensor
    ) -> None:
        """Apply root poses from ``(N, 7)`` or equivalent backend convention."""
        ...

    @abstractmethod
    def apply_qpos(
        self,
        qpos: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor,
        joint_ids: Sequence[int] | torch.Tensor,
        *,
        target: bool,
    ) -> None:
        """Apply joint positions for selected envs and joints."""
        ...

    @abstractmethod
    def apply_qvel(
        self,
        qvel: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor,
        joint_ids: Sequence[int] | torch.Tensor,
        *,
        target: bool,
    ) -> None:
        """Apply joint velocities for selected envs and joints."""
        ...

    @abstractmethod
    def apply_qf(
        self,
        qf: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor,
        joint_ids: Sequence[int] | torch.Tensor,
    ) -> None:
        """Apply joint forces for selected envs and joints."""
        ...

    @abstractmethod
    def clear_dynamics(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """Clear joint velocities, target velocities, and forces."""
        ...

    @abstractmethod
    def compute_kinematics(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """Refresh articulation kinematics if required by the backend."""
        ...
