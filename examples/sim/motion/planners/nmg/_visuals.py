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

"""Render the NMG demo's target sequence and TCP paths with marker groups."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import TYPE_CHECKING

import numpy as np

from embodichain.lab.visualization.markers import MarkerGroupCfg, MarkerPrototypeCfg
from embodichain.utils.math import quat_from_matrix

if TYPE_CHECKING:
    import torch

    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.objects import Robot


_PENDING = (0.73, 0.81, 0.90, 1.0)
_ACTIVE = (1.0, 0.57, 0.06, 1.0)
_COMPLETED = (0.15, 0.84, 0.40, 1.0)
_PLANNED = (0.19, 0.36, 0.55, 1.0)
_MEASURED = (0.04, 0.85, 0.91, 1.0)
_PERTURBATION = (0.99, 0.16, 0.47, 1.0)


def _path_samples(points: np.ndarray) -> np.ndarray:
    """Sample the reference at regular spatial intervals."""
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    distance = np.concatenate(([0.0], np.cumsum(lengths, dtype=np.float64)))
    distinct = np.concatenate(([True], lengths > 0))
    samples = np.append(np.arange(0, distance[-1], 0.015), distance[-1])
    return np.stack(
        [np.interp(samples, distance[distinct], axis[distinct]) for axis in points.T],
        axis=-1,
    )


def _segment_orientation(delta: np.ndarray) -> tuple[float, float, float, float]:
    """Align a marker's local +X with a displacement, in xyzw order."""
    x, y, z = delta
    yaw = 0.5 * math.atan2(y, x)
    pitch = -0.5 * math.atan2(z, math.hypot(x, y))
    sy, cy = math.sin(yaw), math.cos(yaw)
    sp, cp = math.sin(pitch), math.cos(pitch)
    return -sy * sp, cy * sp, sy * cp, cy * cp


class TrajectoryMarkers:
    """Show planned TCP samples, the measured TCP trail, and goal progress."""

    def __init__(
        self,
        sim: SimulationManager,
        robot: Robot,
        goal_poses: torch.Tensor,
        path_poses: torch.Tensor,
        waypoint_indices: Sequence[int],
        modalities: Sequence[str],
    ) -> None:
        self._sim = sim
        self._robot = robot
        path = path_poses[:, :3, 3].detach().cpu().numpy()
        positions = goal_poses[:, :3, 3].detach().cpu().numpy().copy()
        # An orientation-only command leaves translation free. Anchor its frame
        # at the planner's arrival position so the rotation is easy to read.
        for slot, (kind, index) in enumerate(
            zip(modalities, waypoint_indices, strict=True)
        ):
            if kind == "orientation":
                positions[slot] = path[index]

        plan = sim.add_marker_group(
            MarkerGroupCfg(
                name="nmg_planned_tcp",
                prototypes={
                    "sample": MarkerPrototypeCfg(
                        shape="sphere", scale=(0.010,) * 3, color=_PLANNED
                    )
                },
            )
        )
        plan.update(translations=_path_samples(path))

        self._diameters = np.repeat(
            np.array(
                [0.014 if kind == "orientation" else 0.025 for kind in modalities]
            )[:, None],
            3,
            axis=1,
        )
        self._colors = np.tile(_PENDING, (len(modalities), 1))
        self._targets = sim.add_marker_group(
            MarkerGroupCfg(
                name="nmg_targets",
                prototypes={
                    "tcp": MarkerPrototypeCfg(shape="sphere", color=_PENDING),
                    "joint": MarkerPrototypeCfg(shape="box", color=_PENDING),
                },
            )
        )
        self._targets.update(
            translations=positions,
            scales=self._diameters,
            prototype_indices=[int(kind == "joint") for kind in modalities],
        )

        self._frame_slots = np.flatnonzero(np.isin(modalities, ["orientation", "pose"]))
        self._frames = None
        if len(self._frame_slots):
            self._frames = sim.add_marker_group(
                MarkerGroupCfg(
                    name="nmg_target_orientations",
                    prototypes={
                        "frame": MarkerPrototypeCfg(shape="frame", scale=(0.077,) * 3)
                    },
                )
            )
            quaternions = quat_from_matrix(goal_poses[:, :3, :3]).detach().cpu().numpy()
            self._frames.update(
                translations=positions[self._frame_slots],
                orientations_xyzw=quaternions[self._frame_slots],
            )

        self._trail = sim.add_marker_group(
            MarkerGroupCfg(
                name="nmg_measured_tcp",
                prototypes={
                    "segment": MarkerPrototypeCfg(
                        shape="box", scale=(1.0, 0.008, 0.008), color=_MEASURED
                    )
                },
            )
        )
        self._tip = self._read_tip()
        self._centers: list[np.ndarray] = []
        self._orientations: list[tuple[float, float, float, float]] = []
        self._scales: list[tuple[float, float, float]] = []

    def show_targets(self, completed: int, active: int | None = None) -> None:
        """Show the active command and the phases whose playback finished."""
        self._colors[:] = _PENDING
        self._colors[:completed] = _COMPLETED
        scales = self._diameters.copy()
        if active is not None:
            self._colors[active] = _ACTIVE
            scales[active] *= 1.30
        self._targets.update(colors=self._colors, scales=scales)
        if self._frames is not None:
            frame_slot = active if active is not None else completed - 1
            self._frames.update(visible=self._frame_slots == frame_slot)

    def _read_tip(self, qpos: torch.Tensor | None = None) -> np.ndarray:
        if qpos is None:
            qpos = self._robot.get_qpos(name="arm")
        return (
            self._robot.compute_fk(qpos=qpos, name="arm")[0, :3]
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

    def observe(self, _command: int, _count: int) -> None:
        """Sample measured TCP position after a playback command interval."""
        self.record_tip()

    def record_tip(self, qpos: torch.Tensor | None = None) -> None:
        """Read the measured TCP from the robot and extend its trail."""
        self._append_tip(self._read_tip(qpos))

    def record_tcp(self, position: torch.Tensor) -> None:
        """Extend the trail using a TCP pose already computed by the controller."""
        self._append_tip(position.detach().cpu().numpy().copy())

    def _append_tip(self, tip: np.ndarray) -> None:
        delta = tip - self._tip
        length = float(np.linalg.norm(delta))
        if length < 0.010:
            return

        self._centers.append((self._tip + tip) * 0.5)
        self._orientations.append(_segment_orientation(delta))
        self._scales.append((length + 0.002, 1.0, 1.0))
        self._trail.update(
            translations=np.asarray(self._centers),
            orientations_xyzw=self._orientations,
            scales=self._scales,
        )
        self._tip = tip

    def show_perturbation(self, before: torch.Tensor, after: torch.Tensor) -> None:
        """Locate the injected jump and restart the continuous TCP trace."""
        start = before.detach().cpu().numpy()
        end = after.detach().cpu().numpy().copy()
        delta = end - start
        length = float(np.linalg.norm(delta))
        self._append_tip(start.copy())
        self._tip = end
        if length == 0:
            return
        group = self._sim.add_marker_group(
            MarkerGroupCfg(
                name="nmg_joint_perturbation",
                prototypes={
                    "displacement": MarkerPrototypeCfg(
                        shape="arrow", color=_PERTURBATION
                    ),
                    "displaced_tcp": MarkerPrototypeCfg(
                        shape="sphere", scale=(0.022,) * 3, color=_PERTURBATION
                    ),
                },
            )
        )
        group.update(
            translations=np.stack((start, end)),
            orientations_xyzw=[_segment_orientation(delta), (0.0, 0.0, 0.0, 1.0)],
            scales=[(length,) * 3, (1.0, 1.0, 1.0)],
            prototype_indices=[0, 1],
        )
