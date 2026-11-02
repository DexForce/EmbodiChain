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
"""Select DexUni's rigid articulation branch and translate its reset state."""

from __future__ import annotations

from typing import Any

import torch
import warp as wp
from newton import Model
from dexsim.spawn import SceneDesc
from dexsim.types import ActorType

__all__ = ["DexUniStateAdapter"]


def _mujoco_articulation_ids(scene: SceneDesc, num_envs: int) -> tuple[int, ...]:
    """Select every rigid articulation in the replicated manipulation scene."""
    # Each non-static standalone body creates its own free or mocap articulation.
    standalone = sum(
        item.physics is not None
        and item.physics.actor_type != ActorType.STATIC
        and (
            bool(item.collisions) or bool(getattr(item.physics.newton, "mocap", False))
        )
        for item in scene.mesh_objects
    )
    return tuple(range(num_envs * (len(scene.articulations) + standalone)))


class DexUniStateAdapter:
    """Scatter private MuJoCo state into the owning DexSim state buffers.

    The compact-to-scene maps come from NewtonBackend's public facade. Physical
    stepping remains owned by ``backend.step`` and the outer ``SolverDexUni``.
    """

    def __init__(self, backend: Any) -> None:
        outer = backend.solver
        if backend.solver_type != "dexuni" or type(outer).__name__ != "SolverDexUni":
            raise TypeError("Requested DexUni did not create SolverDexUni")
        if outer.features.backend != "pure_mujoco":
            raise ValueError(
                f"Expected DexUni pure_mujoco dispatch, got {outer.features.backend}"
            )
        if outer.features.vbd_solve_enabled or not outer.features.mujoco_solve_enabled:
            raise ValueError("Unexpected active DexUni solver branches")
        self.backend = backend
        self.solver = backend.mujoco_solver
        if self.solver is None or not hasattr(self.solver, "mjw_data"):
            raise TypeError("DexUni has no active GPU MuJoCo data")
        if ".mjvbd_v2.mujoco." not in type(self.solver).__module__:
            raise TypeError("DexUni did not select its private MuJoCo implementation")
        self.local_state = self.solver.model.state()
        parent = backend.model
        joint_world = wp.to_torch(parent.joint_world).long()
        coord_world = torch.repeat_interleave(
            joint_world, torch.diff(wp.to_torch(parent.joint_q_start)).long()
        )
        dof_world = torch.repeat_interleave(
            joint_world, torch.diff(wp.to_torch(parent.joint_qd_start)).long()
        )
        body_world = wp.to_torch(parent.body_world).long()
        frequencies = (
            ("joint_q", Model.AttributeFrequency.JOINT_COORD, coord_world),
            ("joint_qd", Model.AttributeFrequency.JOINT_DOF, dof_world),
            ("body_q", Model.AttributeFrequency.BODY, body_world),
            ("body_qd", Model.AttributeFrequency.BODY, body_world),
        )
        self.fields = []
        for name, frequency, worlds in frequencies:
            mapping = backend.get_mujoco_index_map(frequency)
            if mapping is None:
                raise ValueError(f"Missing DexUni {name} compact-to-scene map")
            indices = wp.to_torch(mapping).long()
            local = wp.to_torch(getattr(self.local_state, name))
            if local.shape[0] != indices.numel():
                raise ValueError(f"DexUni {name} state and index map disagree")
            self.fields.append((name, indices, worlds[indices]))

    def publish_reset(self, data: Any, env_ids: torch.Tensor) -> None:
        """Publish selected MuJoCo coordinate resets into both scene buffers."""
        selected = [
            (name, indices, torch.isin(worlds, env_ids))
            for name, indices, worlds in self.fields
        ]
        for parent_state in (self.backend.state_0, self.backend.state_1):
            for name, indices, _ in selected:
                wp.to_torch(getattr(self.local_state, name)).copy_(
                    wp.to_torch(getattr(parent_state, name))[indices]
                )
            self.solver._update_newton_state(
                self.solver.model,
                self.local_state,
                data,
                state_prev=self.local_state,
            )
            for name, indices, mask in selected:
                wp.to_torch(getattr(parent_state, name))[indices[mask]] = wp.to_torch(
                    getattr(self.local_state, name)
                )[mask]
