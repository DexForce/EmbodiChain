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
"""Private reset-time Newton contact scaling for native task experiments.

The physics adapter owns finalized-model access. Tasks only submit one scalar
per environment; they never write solver arrays. This view is bound to one
prepared topology and must be closed before its manager is destroyed.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import warp as wp
from dexsim.engine.newton_physics.backend_registry import get_newton_backend
from dexsim.engine.newton_physics.runtime_control.contact_material import (
    resolve_contact_shape_ids,
)
from newton import ModelFlags

from embodichain.utils.device_utils import current_warp_stream

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager

__all__: list[str] = []


class _ContactMaterialScaleView:
    """Scale source sliding/torsional/rolling coefficients without compounding."""

    _FIELDS = (
        "shape_material_mu",
        "shape_material_mu_torsional",
        "shape_material_mu_rolling",
    )

    def __init__(self, manager: SimulationManager, uids: tuple[str, ...]) -> None:
        result = manager.spawn_result
        if result is None or not manager.is_newton_backend:
            raise RuntimeError("Contact scaling requires a prepared Newton scene")
        backend = get_newton_backend(result.world)
        if backend is None or backend.model is None:
            raise RuntimeError("Newton model is unavailable")
        if backend.solver_type != "mujoco_warp":
            raise ValueError(
                "This experimental view currently qualifies MuJoCo-Warp only"
            )
        self._manager = manager
        self._result = result
        self._revision = result.topology_revision
        self._runtime = backend._runtime
        self._model = backend.model
        self._device = torch.device(str(backend.model.device))
        indices, rows = [], []
        arenas = tuple(name for name in result.arenas.names if name != "default")
        if len(arenas) != manager.num_envs:
            raise ValueError(
                "Prepared scene Arena count differs from the environment count"
            )
        for row, arena in enumerate(arenas):
            for uid in uids:
                shape_ids = resolve_contact_shape_ids(result, f"{arena}/{uid}")
                indices.extend(map(int, shape_ids))
                rows.extend([row] * len(shape_ids))
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("Expected nonempty, distinct owned contact shapes")
        self._indices = torch.tensor(indices, device=self._device, dtype=torch.long)
        self._rows = torch.tensor(rows, device=self._device, dtype=torch.long)
        self._arrays = tuple(
            wp.to_torch(getattr(backend.model, name)) for name in self._FIELDS
        )
        self._source = tuple(array[self._indices].clone() for array in self._arrays)
        self._scales = torch.ones(manager.num_envs, device=self._device)

    @torch.no_grad()
    def apply(self, scales: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Apply positive float32 row scales at reset, preserving source ratios."""
        if self._manager is None:
            raise RuntimeError("Contact-material view is closed")
        if (
            self._manager.spawn_result is not self._result
            or self._result.topology_revision != self._revision
        ):
            raise RuntimeError("Contact-material view is stale after a topology change")
        rows = (
            torch.arange(len(self._scales), device=self._device)
            if env_ids is None
            else env_ids
        )
        if rows.device != self._device or rows.dtype != torch.long or rows.ndim != 1:
            raise ValueError(
                "Environment IDs must be an int64 vector on the scene device"
            )
        if (
            scales.device != self._device
            or scales.dtype != torch.float32
            or scales.shape != rows.shape
        ):
            raise ValueError(
                "Contact scales must be float32, one per selected environment on the scene device"
            )
        if not bool(torch.isfinite(scales).all() & (scales > 0).all()):
            raise ValueError("Contact scales must be finite and positive")
        if rows.numel() == 0:
            return
        if (
            bool(((rows < 0) | (rows >= len(self._scales))).any())
            or rows.unique().numel() != rows.numel()
        ):
            raise ValueError("Environment IDs must be distinct and in range")
        self._scales[rows] = scales
        # All shapes are bounded by the static scene. Rewriting them avoids a
        # reset-time dynamic nonzero/gather allocation and never compounds a
        # previously scaled value. This is O(number of owned shapes).
        with wp.ScopedStream(current_warp_stream(self._device)):
            values = self._scales[self._rows]
            for target, source in zip(self._arrays, self._source, strict=True):
                target[self._indices] = source * values
            self._runtime.notify_model_changed(
                ModelFlags.SHAPE_PROPERTIES, reason="native reset contact scaling"
            )

    @torch.no_grad()
    def set_impedance(self, solimp: tuple[float, float, float, float, float]) -> None:
        """Set one authored MuJoCo impedance profile on the owned shapes."""
        if self._manager is None:
            raise RuntimeError("Contact-material view is closed")
        if (
            self._manager.spawn_result is not self._result
            or self._result.topology_revision != self._revision
        ):
            raise RuntimeError("Contact-material view is stale after a topology change")
        if (
            len(solimp) != 5
            or not all(math.isfinite(v) for v in solimp)
            or not (
                0 <= solimp[0] <= solimp[1] <= 1
                and solimp[2] > 0
                and 0 <= solimp[3] <= 1
                and solimp[4] >= 1
            )
        ):
            raise ValueError("Invalid MuJoCo contact impedance")
        import numpy as np

        vec5 = wp.types.vector(length=5, dtype=wp.float32)
        current = self._model.mujoco.geom_solimp
        with wp.ScopedStream(current_warp_stream(self._device)):
            if current is None:
                # These are MuJoCo's defaults for other scene shapes. The view
                # writes only owned shapes and explicitly invalidates capture.
                current = wp.array(
                    np.tile([0.9, 0.95, 0.001, 0.5, 2.0], (self._model.shape_count, 1)),
                    dtype=vec5,
                    device=str(self._device),
                )
                self._model.mujoco.geom_solimp = current
            values = wp.to_torch(current)
            values[self._indices] = torch.tensor(solimp, device=self._device)
            self._runtime.notify_model_changed(
                ModelFlags.SHAPE_PROPERTIES, reason="native authored contact impedance"
            )

    def close(self) -> None:
        """Release solver buffers before native scene destruction."""
        self._arrays = self._source = ()
        self._runtime = self._manager = self._result = self._model = None
