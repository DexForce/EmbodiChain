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

"""Explicit runtime inputs and public state services for sensor factories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol
import weakref

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from embodichain.lab.sim.sim_manager import SimulationManager

__all__ = ["SensorClock", "SensorStateProvider", "SensorContext"]


class SensorClock(Protocol):
    """Simulation acquisition clock; reads never advance the simulation."""

    def now(self) -> float:
        """Read elapsed manager simulation time.

        Returns:
            Elapsed simulation time in seconds.
        """
        ...

    @property
    def physics_step_index(self) -> int:
        """Number of completed manager physics substeps, unaffected by reset."""
        ...


class SensorStateProvider(Protocol):
    """Backend-neutral state acquisition for external sensors."""

    def get_joint_state(self, asset_uid: str) -> TensorDict:
        """Read owned joint-position and joint-velocity snapshots.

        Args:
            asset_uid: Registered robot or articulation UID.

        Returns:
            ``qpos`` and ``qvel`` tensors with shape ``(num_instances, dof)``
            in articulation joint order, on the context device. Angular joints
            use radians and rad/s; linear joints use metres and m/s. These are
            generalized coordinates, not six-axis reaction measurements.

        Raises:
            ValueError: The UID does not identify a joint-state provider.
        """
        ...


@dataclass(frozen=True)
class SensorContext:
    """Runtime dependencies passed explicitly to a registered sensor factory.

    This is a runtime value, not a serialized configuration. Providers can be
    replaced with lightweight fakes without creating a simulation. Manager
    adapters hold weak ownership and fail after the manager is destroyed.

    Args:
        device: Device for sensor outputs.
        num_instances: Number of simulation environments.
        clock: Simulation clock and completed-substep identity.
        state: Public joint-state acquisition service.
    """

    device: torch.device
    num_instances: int
    clock: SensorClock
    state: SensorStateProvider

    @classmethod
    def from_manager(cls, manager: SimulationManager) -> SensorContext:
        """Build services owned by one explicitly selected manager.

        Args:
            manager: Manager whose prepared assets the sensor will observe.

        Returns:
            Context without exposing native World or backend objects.
        """
        services = _ManagerSensorServices(manager)
        return cls(manager.device, manager.num_envs, services, services)


class _ManagerSensorServices:
    def __init__(self, manager: SimulationManager) -> None:
        self._manager_ref = weakref.ref(manager)

    def _manager(self) -> SimulationManager:
        manager = self._manager_ref()
        if manager is None or getattr(manager, "_is_pending_kill", False) is True:
            raise RuntimeError("The sensor's owning SimulationManager is unavailable.")
        return manager

    def now(self) -> float:
        return self._manager().simulation_time

    @property
    def physics_step_index(self) -> int:
        return self._manager().physics_step_index

    def get_joint_state(self, asset_uid: str) -> TensorDict:
        manager = self._manager()
        # Physical assets and sensors may deliberately share a UID. Generic
        # get_asset() prefers sensor/light namespaces and cannot resolve state.
        asset = manager._robots.get(asset_uid)
        if asset is None:
            asset = manager._articulations.get(asset_uid)
        if asset is None or not all(
            callable(getattr(asset, name, None)) for name in ("get_qpos", "get_qvel")
        ):
            raise ValueError(f"Asset {asset_uid!r} does not provide joint state.")
        return TensorDict(
            {"qpos": asset.get_qpos().clone(), "qvel": asset.get_qvel().clone()},
            batch_size=[manager.num_envs],
            device=manager.device,
        )
