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
from collections.abc import Callable
import weakref

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from embodichain.lab.sim.sim_manager import SimulationManager
    from .contact import ContactSensor, ContactSensorCfg

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

    def get_joint_names(self, asset_uid: str) -> tuple[str, ...]:
        """Return joint names in the same order as get_joint_state()."""
        ...

    def get_frame_state(
        self, asset_uid: str, link_name: str | None = None
    ) -> TensorDict:
        """Read owned arena-relative pose and velocities for an asset or link.

        Args:
            asset_uid: Physical asset identity.
            link_name: Articulation link, or None for a rigid body/root.

        Returns:
            Pose in metres and xyzw, linear velocity in m/s and angular
            velocity in rad/s, with one row per environment.
        """
        ...

    def get_gravity(self) -> torch.Tensor:
        """Return arena-frame gravitational acceleration in m/s²."""
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
        state: Public physical state acquisition service.
        contact_factory: Optional factory for a bound, sensor-owned contact reader.
    """

    device: torch.device
    num_instances: int
    clock: SensorClock
    state: SensorStateProvider
    contact_factory: Callable[[ContactSensorCfg], ContactSensor] | None = None

    @classmethod
    def from_manager(cls, manager: SimulationManager) -> SensorContext:
        """Build services owned by one explicitly selected manager.

        Args:
            manager: Manager whose prepared assets the sensor will observe.

        Returns:
            Context without exposing native World or backend objects.
        """
        services = _ManagerSensorServices(manager)
        return cls(
            manager.device,
            manager.num_envs,
            services,
            services,
            services.create_contact_reader,
        )


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
        asset = self._physical_asset(asset_uid, require_joints=True)
        return TensorDict(
            {"qpos": asset.get_qpos().clone(), "qvel": asset.get_qvel().clone()},
            batch_size=[self._manager().num_envs],
            device=self._manager().device,
        )

    def get_joint_names(self, asset_uid: str) -> tuple[str, ...]:
        return tuple(self._physical_asset(asset_uid, require_joints=True).joint_names)

    def _physical_asset(self, uid: str, *, require_joints: bool = False):
        manager = self._manager()
        asset = manager._robots.get(uid)
        if asset is None:
            asset = manager._articulations.get(uid)
        if asset is None and not require_joints:
            asset = getattr(manager, "_rigid_objects", {}).get(uid)
        if asset is None or (
            require_joints and not callable(getattr(asset, "get_qpos", None))
        ):
            raise ValueError(
                f"Asset {uid!r} does not provide {'joint' if require_joints else 'frame'} state."
            )
        return asset

    def get_frame_state(
        self, asset_uid: str, link_name: str | None = None
    ) -> TensorDict:
        manager = self._manager()
        asset = self._physical_asset(asset_uid)
        if link_name is not None:
            if link_name not in getattr(asset, "link_names", ()):
                raise ValueError(f"Unknown link {link_name!r} on {asset_uid!r}.")
            index = asset.link_names.index(link_name)
            state = asset.body_state[:, index]
            pose, linear, angular = state[:, :7], state[:, 7:10], state[:, 10:13]
        elif getattr(type(asset), "root_state", None) is not None:
            state = asset.root_state
            pose, linear, angular = state[:, :7], state[:, 7:10], state[:, 10:13]
        else:
            state = asset.body_state
            pose, linear, angular = state[:, :7], state[:, 7:10], state[:, 10:13]
        return TensorDict(
            {
                "pose": pose.expand(manager.num_envs, -1).clone(),
                "linear_velocity": linear.expand(manager.num_envs, -1).clone(),
                "angular_velocity": angular.expand(manager.num_envs, -1).clone(),
            },
            batch_size=[manager.num_envs],
            device=manager.device,
        )

    def get_gravity(self) -> torch.Tensor:
        manager = self._manager()
        return torch.as_tensor(
            manager.sim_config.physics_cfg.gravity,
            dtype=torch.float32,
            device=manager.device,
        ).clone()

    def create_contact_reader(self, cfg: ContactSensorCfg) -> ContactSensor:
        from .contact import ContactSensor

        manager = self._manager()
        if not manager.physics.supports_contact_sensor:
            raise NotImplementedError(
                f"Contact acquisition unavailable on {manager.physics_backend}/"
                f"{getattr(manager.physics, 'solver_type', None)} ({manager.device})."
            )
        return ContactSensor(cfg, manager.device, owner=manager)
