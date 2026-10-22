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

"""Internal adapters for manager-owned Newton runtime controls."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from dexsim.engine.newton_physics.runtime_control import RuntimeControl

__all__: list[str] = []


class _KinematicNodalTrajectoryControl(RuntimeControl):
    """Drive selected particles along offsets from their initialized positions."""

    def __init__(
        self,
        target: str,
        node_indices: np.ndarray,
        position_offsets: np.ndarray,
        *,
        fps: float | None,
        rebuild_self_contact_bvh: bool,
    ) -> None:
        self.target = target
        self.node_indices = np.asarray(node_indices, dtype=np.int32).reshape(-1).copy()
        self.position_offsets = (
            np.asarray(position_offsets, dtype=np.float32)
            .reshape(-1, len(self.node_indices), 3)
            .copy()
        )
        self.fps = fps
        self.rebuild_self_contact_bvh = rebuild_self_contact_bvh
        self._particle_set: Any | None = None
        self._initial_positions: np.ndarray | None = None
        self._sample_index = 0
        self._elapsed_time = 0.0

    def initialize(self, context: Any) -> None:
        """Resolve the preconfigured inactive particles before the first substep."""
        particle_set = context.result.get_particle_set(self.target)
        if np.any(self.node_indices >= particle_set.particle_count):
            raise ValueError(
                f"Kinematic node index exceeds particle count for {self.target!r}: "
                f"max index {int(self.node_indices.max())}, particle count "
                f"{particle_set.particle_count}."
            )

        positions = np.asarray(
            particle_set.get_particle_positions().numpy(),
            dtype=np.float32,
        ).reshape(particle_set.particle_count, 3)
        self._initial_positions = positions[self.node_indices].copy()
        self._particle_set = particle_set
        self._sample_index = 0
        self._elapsed_time = 0.0

    def exclusive_resource_claims(self) -> tuple[object, ...]:
        """Prevent multiple controls from writing the same particle set."""
        return (("kinematic_nodal_trajectory", self.target),)

    def __call__(
        self,
        context: Any,
        substep_index: int,
        substep_count: int,
        substep_dt: float,
    ) -> None:
        """Apply the interpolated target before one Newton substep."""
        del substep_count
        if self._particle_set is None or self._initial_positions is None:
            raise RuntimeError("Kinematic nodal trajectory was not initialized.")

        if self.rebuild_self_contact_bvh and substep_index == 0:
            rebuild_bvh = getattr(context.solver, "rebuild_bvh", None)
            if callable(rebuild_bvh):
                rebuild_bvh(context.current_state)

        offsets = self._current_offsets()
        positions = np.asarray(
            self._particle_set.get_particle_positions().numpy(),
            dtype=np.float32,
        ).reshape(self._particle_set.particle_count, 3)
        positions[self.node_indices] = self._initial_positions + offsets
        self._particle_set.set_particle_positions(positions)

        self._sample_index += 1
        self._elapsed_time += float(substep_dt)

    def _current_offsets(self) -> np.ndarray:
        """Return the current sample or its time-interpolated value."""
        if self.fps is None:
            sample = min(self._sample_index, len(self.position_offsets) - 1)
            return self.position_offsets[sample]

        sample_position = self._elapsed_time * self.fps
        lower = min(int(np.floor(sample_position)), len(self.position_offsets) - 1)
        upper = min(lower + 1, len(self.position_offsets) - 1)
        alpha = np.float32(sample_position - lower if upper != lower else 0.0)
        return (np.float32(1.0) - alpha) * self.position_offsets[
            lower
        ] + alpha * self.position_offsets[upper]

    def close(self) -> None:
        """Release runtime particle references retained after initialization."""
        self._particle_set = None
        self._initial_positions = None


class _InitialJointState(RuntimeControl):
    """Seed creation defaults through engine public initialization hooks only.

    Unlike a realtime controller this never writes targets during stepping.
    Scene.reset_arenas restores the engine-owned immutable defaults thereafter.
    """

    graph_safe = True

    @staticmethod
    def normalize_positions(positions: Any) -> np.ndarray:
        """Copy a finite source-order vector at the one-time host boundary."""
        if isinstance(positions, torch.Tensor):
            if positions.is_complex() or positions.dtype == torch.bool:
                raise ValueError("init_qpos must be a finite real vector.")
            positions = positions.detach().to(device="cpu", dtype=torch.float32).numpy()
        values = np.asarray(positions)
        if values.ndim != 1 or values.dtype.kind not in "iuf":
            raise ValueError("init_qpos must be a finite real one-dimensional vector.")
        with np.errstate(over="ignore", invalid="ignore"):
            values = values.astype(np.float32, copy=True)
        if not np.isfinite(values).all():
            raise ValueError("init_qpos must contain finite float32 values.")
        return values

    def __init__(self, target: str, positions: Any) -> None:
        self.target = target
        self.positions = self.normalize_positions(positions)[None, :]
        self._initializer: Any | None = None
        self._public_positions: np.ndarray | None = None

    def initialize_model(self, model: Any, result: Any) -> None:
        from dexsim.engine.newton_physics import KinematicJointTrajectoryControl
        from embodichain.lab.sim.objects.backends.articulation_state import (
            map_source_qpos_to_state_order,
        )

        handle = result.get_articulation(self.target)
        self._public_positions = map_source_qpos_to_state_order(
            handle, torch.as_tensor(self.positions), is_spawn_bound=True
        ).numpy()
        self._initializer = KinematicJointTrajectoryControl(
            self.target, self._public_positions, initialize_model_pose=True
        )
        self._initializer.initialize_model(model, result)

    def initialize(self, context: Any) -> None:
        # The backend creates both states and runs FK from the seeded model
        # before this hook. Seed drive targets separately through the public
        # batch API; no MuJoCo-specific realtime controller is required.
        if self._public_positions is None:
            raise RuntimeError("Initial joint model state has not been seeded.")
        articulation = context.result.get_articulation(self.target)
        batch = context.result.create_articulation_batch([articulation])
        batch.apply_state(joint_target_position=self._public_positions)

    def __call__(
        self, context: Any, substep_index: int, substep_count: int, substep_dt: float
    ) -> None:
        pass

    def prepare_episode_reset(
        self, arena_indices: tuple[int, ...]
    ) -> Callable[[], None]:
        return lambda: None

    def close(self) -> None:
        if self._initializer is not None:
            self._initializer.close()
