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

"""Live physical drawer/payload observation for explicitly selected placement modes."""

from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Callable
from copy import copy
import numpy as np
import torch
from embodichain.compute.geometry.drawer import drawer_containment_report
from embodichain.compute.geometry.drawer_path import drawer_entry_poses
from .articulation_inputs import PrismaticBinding

if TYPE_CHECKING:
    from embodichain.lab.sim.sim_manager import SimulationManager

__all__ = ["DrawerPayloadObservation"]


class DrawerPayloadObservation:
    """Borrow native entities and refresh their actual poses on each query.

    This adapter owns no task identifiers, invocation routing or policy inference.
    Callers explicitly select upright proposals and other physical payload readers.
    Planning-only containment does not become execution evidence automatically.
    """

    def __init__(
        self,
        simulation: SimulationManager,
        binding: PrismaticBinding,
        object_id: str,
        lower: tuple[float, float, float],
        upper: tuple[float, float, float],
    ) -> None:
        self._read_guard: Callable[[], None] = lambda: None
        self.binding, self.object_id = binding, object_id
        self.lower, self.upper = tuple(lower), tuple(upper)
        bounds = np.asarray([self.lower, self.upper], dtype=float)
        if (
            bounds.shape != (2, 3)
            or not np.isfinite(bounds).all()
            or (bounds[0] > bounds[1]).any()
        ):
            raise ValueError("Drawer region requires finite ordered 3D bounds.")
        self._art = simulation.get_articulation(binding.object_id)
        self._obj = simulation.get_rigid_object(object_id)
        if self._art is None or self._obj is None:
            raise ValueError("Drawer observation requires both native entities.")
        self.vertices = (
            self._obj.get_vertices(scale=True)[0].detach().cpu().numpy().copy()
        )

    def guarded(self, read_guard: Callable[[], None]) -> DrawerPayloadObservation:
        """Return a local read view retaining geometry under an explicit source guard."""
        if not callable(read_guard):
            raise TypeError("Drawer read guard must be callable.")
        result = copy(self)
        result._read_guard = read_guard
        return result

    def payload_pose(self) -> torch.Tensor:
        """Read only the current payload pose for a lazy carried-body query."""
        self._read_guard()
        pose = self._obj.get_local_pose(to_matrix=True)
        if pose.shape != (1, 4, 4):
            raise ValueError("Drawer observations support batch_size=1 only.")
        self._read_guard()
        return pose

    def poses(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Read current payload and selected link poses in one environment."""
        self._read_guard()
        obj = self._obj.get_local_pose(to_matrix=True)
        link = self._art.get_link_pose(self.binding.link, to_matrix=True)
        if obj.shape != (1, 4, 4) or link.shape != (1, 4, 4):
            raise ValueError("Drawer observations support batch_size=1 only.")
        self._read_guard()
        return obj, link

    def entry_pose_candidates(
        self,
        *,
        preserve_vertical: bool = False,
        release_lift: float = 0.12,
        other_payloads: tuple[DrawerPayloadObservation, ...] = (),
    ) -> tuple[tuple[torch.Tensor, ...], ...]:
        """Refresh finite outside/rim/release targets without simulation steps."""
        result = []
        for target in self.target_poses(
            preserve_vertical=preserve_vertical, other_payloads=other_payloads
        ):
            _, link = self.poses()
            result.append(
                drawer_entry_poses(
                    target,
                    link,
                    self.vertices,
                    self.lower,
                    self.upper,
                    self.binding.axis,
                    self.binding.axis_sign,
                    preserve_vertical=preserve_vertical,
                    release_lift=release_lift,
                )
            )
        return tuple(result)

    def target_poses(
        self,
        *,
        preserve_vertical: bool = False,
        other_payloads: tuple[DrawerPayloadObservation, ...] = (),
    ) -> tuple[torch.Tensor, ...]:
        """Build bounded payload targets from live drawer and object poses."""
        obj, link = self.poses()
        targets: list[torch.Tensor] = []
        index = list(self._art.joint_names).index(self.binding.joint)
        opening = (
            (self._art.get_qpos()[:, index] - self.binding.target("closed"))
            .abs()
            .to(link)
        )
        self._read_guard()
        from embodichain.compute.geometry.drawer_path import (
            drawer_exposed_floor_targets,
        )

        raw_rows = drawer_exposed_floor_targets(
            self.vertices,
            obj,
            link,
            opening,
            self.lower,
            self.upper,
            self.binding.axis,
            self.binding.axis_sign,
            preserve_vertical=preserve_vertical,
        )
        for row, poses in enumerate(raw_rows):
            row_targets: list[torch.Tensor] = []
            for target in poses:
                if preserve_vertical:
                    for side in (0, -1, 1):
                        candidate = self._unoccupied_target(
                            target, link[row], side=side, other_payloads=other_payloads
                        )
                        if not any(
                            torch.allclose(candidate, previous)
                            for previous in row_targets
                        ):
                            row_targets.append(candidate)
                else:
                    row_targets.append(target)
            if not row_targets:
                raise ValueError(
                    f"Drawer payload {self.object_id!r} produced no "
                    "finite placement candidate."
                )
            targets.append(torch.stack(row_targets))
        width = max(item.shape[0] for item in targets)
        padded = obj.new_zeros((len(targets), width, 4, 4))
        padded[:] = torch.eye(4, dtype=obj.dtype, device=obj.device)
        for row, values in enumerate(targets):
            padded[row, : values.shape[0]] = values
            if values.shape[0] < width:
                padded[row, values.shape[0] :] = values[-1]
        return tuple(padded[:, index] for index in range(width))

    def _unoccupied_target(
        self,
        target: torch.Tensor,
        link: torch.Tensor,
        *,
        side: int = 0,
        other_payloads: tuple[DrawerPayloadObservation, ...] = (),
    ) -> torch.Tensor:
        from embodichain.compute.geometry.drawer_path import drawer_unoccupied_target

        occupied = tuple(
            (other.vertices, other.poses()[0][0])
            for other in other_payloads
            if other.containment_report()["accepted"]
        )
        return drawer_unoccupied_target(
            target,
            link,
            self.vertices,
            self.lower,
            self.upper,
            self.binding.axis,
            occupied,
            side=side,
        )

    def containment_report(
        self, *, lateral_margin: float | None = None
    ) -> dict[str, object]:
        """Read a live link-local mesh containment report for the payload."""
        obj, link = self.poses()
        report = drawer_containment_report(
            self.vertices,
            obj[0].detach().cpu().numpy(),
            link[0].detach().cpu().numpy(),
            np.asarray(self.lower),
            np.asarray(self.upper),
            lateral_margin=lateral_margin,
        )
        return report
