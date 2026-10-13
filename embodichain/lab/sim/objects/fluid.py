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

"""Declared fixed-count fluids with Scene-owned particle state."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Sequence
import numpy as np
import torch

from embodichain.lab.sim.cfg.fluid import FluidObjectCfg
from embodichain.lab.sim.objects.deformable.data import DeformableObjectData
from embodichain.utils.math import matrix_from_quat

__all__ = ["FluidObject", "FluidObjectData"]


class FluidObjectData:
    """Detached world-frame state snapshots, shaped ``(E, N, 3|6)``.

    The shared Scene particle batch adapter is composed without inheriting
    any deformable connectivity or fixed render topology.
    """

    def __init__(
        self, entities: Sequence[Any], scene: Any, device: torch.device
    ) -> None:
        self._state = DeformableObjectData(entities, scene, device)

    @property
    def particle_pos_w(self) -> torch.Tensor:
        """Current positions in metres; modifying the result does not write state."""
        return self._state.nodal_pos_w

    @property
    def particle_vel_w(self) -> torch.Tensor:
        """Current world-frame velocities in metres per second."""
        return self._state.nodal_vel_w

    @property
    def particle_state_w(self) -> torch.Tensor:
        """Concatenated positions and velocities, with shape ``(E, N, 6)``."""
        return self._state.nodal_state_w

    @property
    def default_particle_state_w(self) -> torch.Tensor:
        """Independent copy of the finalized initial state."""
        return self._state.default_nodal_state_w


class FluidObject:
    """One logical fluid declared and bound by SimulationManager.

    Particle count is fixed. There is no physical rigid root pose or persistent
    attachment to the initial container. Pose reset transforms the rest template.
    """

    def __init__(
        self,
        cfg: FluidObjectCfg,
        positions: np.ndarray,
        pose: np.ndarray,
        device: torch.device,
        *,
        on_write: Callable[[], None],
    ) -> None:
        self.cfg = deepcopy(cfg)
        self.uid = cfg.uid
        self.device = device
        self._positions = torch.as_tensor(positions.copy(), device=device)
        self._initial_pose = torch.as_tensor(pose.copy(), device=device)
        self._entities: list[Any] = []
        self._spawn_result = None
        self._data: FluidObjectData | None = None
        self._num_instances = 0
        self._on_write = on_write

    def _initialize_spawn_declaration(self, num_instances: int) -> None:
        self._num_instances = num_instances

    @property
    def is_declared(self) -> bool:
        """Whether state is still unavailable pending ``prepare()``."""
        return self._data is None

    @property
    def num_instances(self) -> int:
        """Number of scene-owned replicas."""
        return self._num_instances

    @property
    def num_particles(self) -> int:
        """Fixed particle count per replica."""
        return len(self._positions)

    @property
    def particle_mass(self) -> float:
        """Uniform particle mass in kilograms."""
        return self.cfg.density * self.cfg.particle_spacing**3

    @property
    def nominal_volume(self) -> float:
        """Rest volume in cubic metres, independent of render reconstruction."""
        return self.num_particles * self.cfg.particle_spacing**3

    @property
    def data(self) -> FluidObjectData:
        """Return prepared particle data, or raise before binding."""
        if self._data is None:
            raise RuntimeError("Fluid state requires SimulationManager.prepare().")
        return self._data

    def attach_spawn_handles(self, entities: Sequence[Any]) -> None:
        """Attach the replicas materialized by Spawn."""
        if len(entities) != self.num_instances:
            raise RuntimeError("Fluid replica count does not match the declaration.")
        self._entities = list(entities)

    def bind_spawn(self, scene: Any) -> None:
        """Bind Scene batch state and capture configured initial velocities."""
        if self._data is not None:
            return
        data = FluidObjectData(self._entities, scene, self.device)
        if data._state.n_nodes != self.num_particles:
            raise RuntimeError("Spawn changed the fluid particle count.")
        state = data.particle_state_w
        velocity = torch.as_tensor(
            self.cfg.spawn.initial_velocity, device=self.device, dtype=torch.float32
        )
        state[..., 3:] = velocity @ self._initial_pose[:3, :3].T
        data._state._apply_nodal_state(
            state[..., :3], state[..., 3:], list(range(self.num_instances))
        )
        data._state._default_nodal_state_w = state.clone()
        self._spawn_result = scene
        self._data = data

    def _indices(self, env_ids: Sequence[int] | None) -> list[int]:
        if env_ids is None:
            return list(range(self.num_instances))
        ids = torch.as_tensor(env_ids)
        if ids.ndim != 1 or (
            ids.numel() and ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("env_ids must be a one-dimensional integer selection.")
        result = ids.tolist()
        if len(set(result)) != len(result) or any(
            i < 0 or i >= self.num_instances for i in result
        ):
            raise ValueError("env_ids must be unique valid instance indices.")
        return result

    def write_particle_state_to_sim(
        self,
        state: torch.Tensor,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Write selected world-frame positions and velocities without stepping.

        Args:
            state: Finite floating tensor ``(selected E, N, 6)``.
            env_ids: Unique instance indices; None selects all instances.
        """
        data = self.data
        ids = self._indices(env_ids)
        if not isinstance(state, torch.Tensor) or not state.is_floating_point():
            raise TypeError("Fluid state must be a floating Torch tensor.")
        if tuple(state.shape) != (len(ids), self.num_particles, 6):
            raise ValueError("Fluid state must have shape (selected E, N, 6).")
        if not torch.isfinite(state).all():
            raise ValueError("Fluid state must be finite.")
        if not ids:
            return
        self._on_write()
        data._state._apply_nodal_state(state[..., :3], state[..., 3:], ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Restore selected initial particle states without advancing time."""
        ids = self._indices(env_ids)
        self.write_particle_state_to_sim(self.data.default_particle_state_w[ids], ids)

    def reset_to_pose(
        self, pose_w: torch.Tensor, env_ids: Sequence[int] | None = None
    ) -> None:
        """Place the local rest template at world poses ``xyz + xyzw``.

        Args:
            pose_w: Finite ``(selected E, 7)`` poses with unit quaternions.
            env_ids: Unique instance indices, or None for all.
        """
        ids = self._indices(env_ids)
        if tuple(pose_w.shape) != (len(ids), 7) or not torch.isfinite(pose_w).all():
            raise ValueError("pose_w must be finite (selected E, 7).")
        pose = pose_w.to(device=self.device, dtype=torch.float32)
        if not torch.allclose(
            pose[:, 3:].norm(dim=-1),
            torch.ones(len(ids), device=self.device),
            atol=1e-5,
        ):
            raise ValueError("pose_w requires unit xyzw quaternions.")
        rotation = matrix_from_quat(pose[:, 3:])
        points = (
            torch.einsum("eij,nj->eni", rotation, self._positions) + pose[:, None, :3]
        )
        velocity = torch.as_tensor(
            self.cfg.spawn.initial_velocity, device=self.device, dtype=torch.float32
        )
        velocities = (rotation @ velocity)[..., None, :].expand_as(points)
        self.write_particle_state_to_sim(torch.cat((points, velocities), dim=-1), ids)
