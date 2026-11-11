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

"""Replicated cable facade using only DexSim's public Scene interfaces."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from numbers import Integral
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from dexsim.scene import Scene, SpawnedCable

from embodichain.lab.sim.common import BatchEntity

from .data import CableObjectData

if TYPE_CHECKING:
    from embodichain.lab.sim.cfg import CableObjectCfg

__all__ = ["CableObject"]


class CableObject(BatchEntity):
    """Declared cable, bound in place by ``SimulationManager.prepare()``.

    Cable state and resources remain owned by DexSim. There is no single
    deformable root pose. Segment writes are teleports; complete episode reset
    belongs to ``SimulationManager.reset_objects_state()`` and its owning Scene.

    Args:
        cfg: Typed cable authoring configuration.
        device: Newton compute device selected by the manager.
    """

    def __init__(
        self, cfg: CableObjectCfg, device: torch.device = torch.device("cpu")
    ) -> None:
        self.cfg = deepcopy(cfg)
        self.uid = self.cfg.uid
        self.device = torch.device(device)
        self._entities: list[SpawnedCable] = []
        self._declared_num_instances: int | None = None
        self._spawn_result: Scene | None = None
        self._world = None
        self._data: CableObjectData | None = None

    def _initialize_spawn_declaration(self, num_instances: int) -> None:
        if num_instances <= 0:
            raise ValueError("A declared CableObject requires num_instances > 0.")
        if self._declared_num_instances not in (None, num_instances):
            raise RuntimeError("CableObject was already declared for another count.")
        self._declared_num_instances = num_instances

    @property
    def is_declared(self) -> bool:
        """Whether this facade is waiting for its finalized Scene binding."""
        return self._spawn_result is None

    @property
    def is_spawn_bound(self) -> bool:
        """Whether this facade has completed Spawn binding."""
        return self._spawn_result is not None

    @property
    def num_instances(self) -> int:
        """Materialized or declared environment count."""
        if self._declared_num_instances is None:
            raise RuntimeError("Register CableObject through SimulationManager first.")
        return self._declared_num_instances

    @property
    def data(self) -> CableObjectData | None:
        """Read-only segment data after successful ``prepare()``."""
        return self._data

    def attach_spawn_handles(self, entities: Sequence[SpawnedCable]) -> None:
        """Attach replicated cable handles before final data binding.

        Args:
            entities: One public cable handle per declared environment.
        """
        handles = list(entities)
        if len(handles) != self.num_instances:
            raise ValueError(
                f"CableObject expected {self.num_instances} handles, got {len(handles)}."
            )
        if not all(isinstance(handle, SpawnedCable) for handle in handles):
            raise TypeError("CableObject requires SpawnedCable handles.")
        self._entities = handles

    def bind_spawn(self, result: Scene) -> None:
        """Create the public cable batch after finalization, without stepping.

        Args:
            result: Finalized Newton Scene owning the attached handles.
        """
        if self.is_spawn_bound:
            raise RuntimeError(f"CableObject {self.uid!r} is already Spawn-bound.")
        if not isinstance(result, Scene):
            raise TypeError("CableObject requires a finalized DexSim Scene.")
        if result.backend != "newton":
            raise NotImplementedError("CableObject requires the Newton backend.")
        if len(self._entities) != self.num_instances:
            raise ValueError("Attach all cable handles before binding.")
        # Construct first: a failed batch/default-state read remains retryable.
        world = result.world
        data = CableObjectData(self._entities, result, self.device)
        self._data = data
        self._spawn_result = result
        self._world = world

    def __str__(self) -> str:
        return (
            f"CableObject | uid: {self.uid} | instances: "
            f"{self._declared_num_instances} | bound: {self.is_spawn_bound}"
        )

    def _resolve_env_ids(self, env_ids: Sequence[int] | None) -> list[int]:
        if env_ids is None:
            return list(range(self.num_instances))
        values = (
            env_ids.detach().cpu().reshape(-1).tolist()
            if isinstance(env_ids, torch.Tensor)
            else list(env_ids)
        )
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            for value in values
        ):
            raise TypeError("env_ids must contain integer environment indices.")
        ids = [int(value) for value in values]
        if len(set(ids)) != len(ids):
            raise ValueError("env_ids must not contain duplicates.")
        if any(value < 0 or value >= self.num_instances for value in ids):
            raise IndexError("env_ids are outside this CableObject batch.")
        return ids

    def _require_bound(self) -> None:
        if self._data is None or self._spawn_result is None:
            raise RuntimeError(
                "Call SimulationManager.prepare() before cable state use."
            )
        if not self._data.batch.is_valid:
            raise RuntimeError("CableObject Scene was cleared or released.")

    def _apply_load(
        self,
        operation: str,
        value: Any,
        segment_index: int,
        env_ids: Sequence[int] | None,
    ) -> None:
        self._require_bound()
        ids = self._resolve_env_ids(env_ids)
        assert self._data is not None
        count = self._data.num_segments
        if (
            isinstance(segment_index, (bool, np.bool_))
            or not isinstance(segment_index, Integral)
            or not -count <= segment_index < count
        ):
            raise IndexError("segment_index is outside this cable.")
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        values = np.asarray(value)
        if values.dtype.kind not in "iuf":
            raise ValueError("Cable loads must be finite real XYZ vectors.")
        if values.shape == (3,):
            values = np.broadcast_to(values, (len(ids), 3))
        if values.shape != (len(ids), 3) or not np.isfinite(values).all():
            raise ValueError(f"Cable loads require shape (3,) or ({len(ids)}, 3).")
        values = values.astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Cable loads must be representable as finite float32.")
        handles = [self._entities[index] for index in ids]
        if any(not handle.is_valid for handle in handles):
            raise RuntimeError("CableObject contains an invalid cable handle.")
        for handle, vector in zip(handles, values, strict=True):
            getattr(handle, operation)(vector, segment_index=int(segment_index))

    def set_force(
        self,
        force: torch.Tensor | np.ndarray | Sequence[float],
        *,
        segment_index: int = -1,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set world COM force for the next outer update, then auto-clear.

        Args:
            force: XYZ newtons broadcast to selected environments, or ``(E,3)``.
            segment_index: Segment to load; negative indices count from the tip.
            env_ids: Selected environments in force-row order; None selects all.
        """
        self._apply_load("set_force", force, segment_index, env_ids)

    def set_torque(
        self,
        torque: torch.Tensor | np.ndarray | Sequence[float],
        *,
        segment_index: int = -1,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set world torque for the next outer update, then auto-clear.

        Args:
            torque: XYZ N m broadcast to selected environments, or ``(E,3)``.
            segment_index: Segment to load; negative indices count from the tip.
            env_ids: Selected environments in torque-row order; None selects all.
        """
        self._apply_load("set_torque", torque, segment_index, env_ids)

    def clear_wrenches(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear pending external loads in selected environments.

        Args:
            env_ids: Selected environments; None selects all.
        """
        self._require_bound()
        ids = self._resolve_env_ids(env_ids)
        handles = [self._entities[index] for index in ids]
        if any(not handle.is_valid for handle in handles):
            raise RuntimeError("CableObject contains an invalid cable handle.")
        for handle in handles:
            handle.clear_wrenches()

    def set_local_pose(
        self, pose: torch.Tensor, env_ids: Sequence[int] | None = None
    ) -> None:
        """Reject a single rigid root pose; use segment state writes instead.

        Args:
            pose: Requested root pose.
            env_ids: Requested environment selection.
        """
        raise NotImplementedError("A cable has no single rigid root pose.")

    def write_segment_state_to_sim(
        self, state: torch.Tensor, env_ids: Sequence[int] | None = None
    ) -> None:
        """Teleport selected cables using world segment states.

        Writes both engine states through the public CableBatch. Solver history
        for each owning world (including other cables) is cleared. Controls and
        pending loads are preserved. This is suitable for initialization after
        manager reset, not for resuming a solver-history checkpoint. Call between
        updates, outside caller CUDA Graph capture and gradient mode. Existing
        contact queries require another physics update before fetching.

        Args:
            state: Contiguous float32 ``(E,S,13)`` on this object's device, in
                ``[xyz, xyzw, world COM linear velocity, angular velocity]``
                order. Quaternions must be unit length; values must be finite.
            env_ids: Selected environments in state-row order; None selects all.
        """
        self._require_bound()
        ids = self._resolve_env_ids(env_ids)
        assert self._data is not None
        if not isinstance(state, torch.Tensor):
            raise TypeError("Cable state must be a torch.Tensor.")
        expected = (len(ids), self._data.num_segments, 13)
        if tuple(state.shape) != expected:
            raise ValueError(f"Cable state requires shape {expected}.")
        if state.dtype != torch.float32 or state.device != torch.device(
            self._data.batch.device
        ):
            raise ValueError("Cable state must be float32 on the cable device.")
        if state.requires_grad or not state.is_contiguous():
            raise ValueError("Cable state must be contiguous without gradients.")
        if not ids:
            return
        pose = torch.cat((state[..., 3:7], state[..., :3]), -1)
        velocity = state[..., 7:].contiguous()
        self._data.batch.select(ids).apply_state(pose=pose, velocity=velocity)

    def get_local_pose(self, to_matrix: bool = False) -> torch.Tensor:
        """Reject a single root pose; use ``data.segment_pose_w``.

        Args:
            to_matrix: Requested representation.
        """
        raise NotImplementedError("A cable has no single rigid root pose.")

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Restore selected cable creation states and clear their pending loads.

        This clears solver history in their owning worlds while preserving other
        bodies and drive targets. Use manager reset for a whole episode, including
        attached robots, other cables and runtime controls.

        Args:
            env_ids: Selected environments; None selects all.
        """
        self._require_bound()
        ids = self._resolve_env_ids(env_ids)
        assert self._data is not None
        state = self._data.default_segment_state_w[ids].contiguous()
        self.write_segment_state_to_sim(state, ids)
        self.clear_wrenches(ids)

    def destroy(self) -> None:
        """Reject individual removal; the manager releases the owning Scene."""
        raise NotImplementedError("DexSim does not expose individual cable removal.")
