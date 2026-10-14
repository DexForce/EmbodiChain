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

from typing import Any

import numpy as np

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


class _KinematicRigidTrajectoryControl(RuntimeControl):
    """Sample world pose tracks on the native solver substep clock."""

    def __init__(self, target: str, poses: np.ndarray, fps: float) -> None:
        from scipy.spatial.transform import Rotation, Slerp

        self.target = target
        self.poses = poses.copy()
        self.fps = fps
        self._times = np.arange(len(poses)) / fps
        self._rotations = Rotation.from_quat(poses[:, 3:])
        self._slerp = Slerp(self._times, self._rotations) if len(poses) > 1 else None
        self._actor = None
        self.reset()

    def initialize(self, context: Any) -> None:
        self._actor = context.result.get_actor(self.target)
        self.reset()

    def exclusive_resource_claims(self) -> tuple[object, ...]:
        return (("kinematic_rigid_trajectory", self.target),)

    def reset(self) -> None:
        self._elapsed = 0.0
        self._previous = None

    def __call__(
        self, context: Any, substep_index: int, substep_count: int, substep_dt: float
    ) -> None:
        from scipy.spatial.transform import Rotation

        del context, substep_index, substep_count
        if self._actor is None:
            raise RuntimeError("Kinematic rigid trajectory is not initialized.")
        t = min(self._elapsed, self._times[-1])
        position = np.array(
            [np.interp(t, self._times, self.poses[:, axis]) for axis in range(3)]
        )
        rotation = self._rotations[0] if self._slerp is None else self._slerp(t)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rotation.as_matrix()
        pose[:3, 3] = position
        previous = pose if self._previous is None else self._previous
        self._actor.set_world_pose(pose)
        self._actor.set_linear_velocity((pose[:3, 3] - previous[:3, 3]) / substep_dt)
        self._actor.set_angular_velocity(
            Rotation.from_matrix(pose[:3, :3] @ previous[:3, :3].T).as_rotvec()
            / substep_dt
        )
        self._previous = pose
        self._elapsed += substep_dt

    def close(self) -> None:
        self._actor = None
