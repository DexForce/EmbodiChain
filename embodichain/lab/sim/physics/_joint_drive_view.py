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
"""Private static-topology batched Newton PD-gain writes for native RL resets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
from dexsim.engine.newton_physics.backend_registry import get_newton_backend
from newton import ModelFlags

from embodichain.utils.device_utils import current_warp_stream

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager

__all__: list[str] = []


class _JointDriveGainView:
    """Write reset gains once per selected batch, retaining source reset defaults.

    This runtime-only view does not rewrite source descriptors or initialization
    snapshots. It rejects topology changes and must close before manager teardown.
    """

    def __init__(
        self, manager: SimulationManager, joint_names: tuple[str, ...]
    ) -> None:
        result = manager.spawn_result
        if result is None or not manager.is_newton_backend:
            raise RuntimeError("PD gain views require prepared Newton physics")
        backend = get_newton_backend(result.world)
        if backend is None or backend.solver_type != "mujoco_warp":
            raise ValueError("PD gain views currently qualify MuJoCo-Warp only")
        if not joint_names or len(set(joint_names)) != len(joint_names):
            raise ValueError("Joint names must be nonempty and distinct")
        self._manager = manager
        self._result = result
        self._revision = result.topology_revision
        self._runtime = backend._runtime
        model = backend.model
        self.device = torch.device(str(model.device))
        starts = model.joint_qd_start.numpy()
        indices = []
        arenas = tuple(n for n in result.arenas.names if n != "default")
        if len(arenas) != manager.num_envs:
            raise ValueError("Expected one prepared Arena per environment")
        matches_by_name: dict[tuple[str, str], list[int]] = {}
        selected_names = set(joint_names)
        for joint, label in enumerate(model.joint_label):
            arena, separator, rest = label.partition("/")
            if not separator:
                continue
            parts = rest.split("/")
            for offset in range(len(parts)):
                name = "/".join(parts[offset:])
                if name in selected_names:
                    matches_by_name.setdefault((arena, name), []).append(joint)
        for arena in arenas:
            row = []
            for name in joint_names:
                matches = matches_by_name.get((arena, name), [])
                if len(matches) != 1:
                    raise ValueError(f"Expected one exact source joint {arena}/{name}")
                joint = matches[0]
                if starts[joint + 1] - starts[joint] != 1:
                    raise ValueError("PD gain views require scalar hinge joints")
                row.append(int(starts[joint]))
            indices.append(row)
        self._indices = torch.tensor(indices, device=self.device, dtype=torch.long)
        self._kp = wp.to_torch(model.joint_target_ke)
        self._kd = wp.to_torch(model.joint_target_kd)

    @torch.no_grad()
    def apply(
        self, kp: torch.Tensor, kd: torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> None:
        """Set nonnegative float32 gains for selected Arena rows in one update."""
        if self._manager is None:
            raise RuntimeError("PD gain view is closed")
        if (
            self._manager.spawn_result is not self._result
            or self._result.topology_revision != self._revision
        ):
            raise RuntimeError("PD gain view is stale after topology change")
        ids = (
            torch.arange(len(self._indices), device=self.device)
            if env_ids is None
            else env_ids
        )
        if ids.device != self.device or ids.dtype != torch.long or ids.ndim != 1:
            raise ValueError("Environment IDs must be int64 on the scene device")
        expected = (len(ids), self._indices.shape[1])
        for values in (kp, kd):
            if (
                values.device != self.device
                or values.dtype != torch.float32
                or values.shape != expected
            ):
                raise ValueError(
                    f"Expected float32 gain shape {expected} on the scene device"
                )
            if not bool(torch.isfinite(values).all() & (values >= 0).all()):
                raise ValueError("PD gains must be finite and nonnegative")
        if ids.numel() == 0:
            return
        if (
            bool(((ids < 0) | (ids >= len(self._indices))).any())
            or ids.unique().numel() != ids.numel()
        ):
            raise ValueError("Environment IDs must be distinct and in range")
        with wp.ScopedStream(current_warp_stream(self.device)):
            columns = self._indices[ids]
            self._kp[columns] = kp
            self._kd[columns] = kd
            self._runtime.notify_model_changed(
                ModelFlags.JOINT_DOF_PROPERTIES, reason="batched native reset PD gains"
            )

    def close(self) -> None:
        """Release borrowed gain buffers before native destruction."""
        self._kp = self._kd = None
        self._runtime = self._manager = self._result = None
