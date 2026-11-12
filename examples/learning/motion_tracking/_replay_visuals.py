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

"""World-space floor traces for recorded motion-tracking playback."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.visualization.markers import VisualizationMarkers

__all__ = ["FloorTrajectory"]


class FloorTrajectory:
    """Draw one clip's recent motion and future reference on the ground.

    Args:
        sim: Prepared simulation owning the render-only markers.
        reference_xy: Recorded reference root positions shaped ``(T, 2)``.
        actual_xy: Recorded policy root positions shaped ``(T, 2)``.
        dt: Time between recorded frames, in seconds.

    Traces use the original world coordinates. Reference-robot display alignment
    and ghost offsets are independent of these positions. Each update publishes
    159 marker instances; the caller synchronizes rendering before capture.
    """

    def __init__(
        self,
        sim: SimulationManager,
        reference_xy: ArrayLike,
        actual_xy: ArrayLike,
        *,
        dt: float = 0.02,
    ) -> None:
        from embodichain.lab.visualization.markers import (
            MarkerPrototypeCfg,
            VisualizationMarkers,
            VisualizationMarkersCfg,
        )

        self._reference = np.array(reference_xy, dtype=np.float64, copy=True)
        self._actual = np.array(actual_xy, dtype=np.float64, copy=True)
        if (
            self._reference.ndim != 2
            or self._reference.shape[1:] != (2,)
            or len(self._reference) == 0
            or self._actual.shape != self._reference.shape
        ):
            raise ValueError("Reference and actual XY must have matching shape (T, 2).")
        if (
            not np.isfinite(self._reference).all()
            or not np.isfinite(self._actual).all()
        ):
            raise ValueError("Recorded XY positions must be finite.")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("The recorded frame interval must be finite and positive.")

        self._past_offsets = np.rint(np.linspace(-round(3 / dt), 0, 51)).astype(
            np.int64
        )
        self._future_offsets = np.rint(np.linspace(0, round(2 / dt), 51)).astype(
            np.int64
        )
        theta = np.arange(48) * (2 * np.pi / 48)
        vertices = np.concatenate(
            [
                np.column_stack([r * np.cos(theta), r * np.sin(theta), np.zeros(48)])
                for r in (0.085, 0.06)
            ]
        )
        faces = []
        for i in range(48):
            following = (i + 1) % 48
            faces.extend([(i, following, 48 + i), (following, 48 + following, 48 + i)])
        colors = (
            (0.86, 0.12, 0.10, 1),
            (0.02, 0.37, 0.16, 1),
            (0.57, 0.76, 0.61, 1),
            (0.25, 0.29, 0.28, 1),
        )
        prototypes = {
            f"line_{i}": MarkerPrototypeCfg(shape="box", color=color)
            for i, color in enumerate(colors)
        }
        prototypes["actual"] = MarkerPrototypeCfg(shape="cylinder", color=colors[0])
        prototypes["reference"] = MarkerPrototypeCfg(
            shape="mesh", vertices=vertices, faces=np.array(faces), color=colors[1]
        )
        prototypes["future"] = MarkerPrototypeCfg(
            shape="mesh",
            vertices=np.array([[0, 0, 0], [-0.16, 0.08, 0], [-0.16, -0.08, 0]]),
            faces=np.array([[0, 1, 2]]),
            color=colors[1],
        )
        self._markers: VisualizationMarkers | None = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="replay_floor_trajectory", markers=prototypes
            ),
            sim,
        )
        self._positions = np.zeros((159, 3), dtype=np.float64)
        self._rotations = np.tile([0.0, 0.0, 0.0, 1.0], (159, 1))
        self._scales = np.ones((159, 3), dtype=np.float64)
        self._kinds = np.concatenate(
            [np.repeat([2, 1, 0], 50), np.full(6, 3), [4, 5, 6]]
        ).astype(np.int64)

    @staticmethod
    def _trim_arrow(
        points: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        """End the shaft at the arrowhead base, preserving its original tip."""
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        distance = np.r_[0.0, np.cumsum(lengths)]
        if distance[-1] < 0.12:
            return points, np.zeros(2), 0.0
        cut = distance[-1] - min(0.16, distance[-1] * 0.5)
        index = min(
            int(np.searchsorted(distance, cut, side="right")) - 1, len(points) - 2
        )
        fraction = (cut - distance[index]) / lengths[index]
        base = points[index] + fraction * (points[index + 1] - points[index])
        direction = points[-1] - base
        head_length = float(np.linalg.norm(direction))
        if head_length < 0.04:
            return points, np.zeros(2), 0.0
        shaft = points.copy()
        shaft[index + 1 :] = base
        return shaft, direction, head_length

    def _write_lines(
        self,
        start: int,
        points: NDArray[np.float64],
        width: float,
        z: float,
    ) -> None:
        """Write a polyline into the existing instance arrays."""
        delta = np.diff(points, axis=0)
        length = np.linalg.norm(delta, axis=1)
        angle = np.arctan2(delta[:, 1], delta[:, 0])
        rows = slice(start, start + len(delta))
        self._positions[rows, :2] = (points[:-1] + points[1:]) / 2
        self._positions[rows, 2] = z
        self._rotations[rows, 2] = np.sin(angle / 2)
        self._rotations[rows, 3] = np.cos(angle / 2)
        self._scales[rows, 0] = np.maximum(length, 1e-5)
        self._scales[rows, 1] = np.where(length > 1e-6, width, 1e-5)
        self._scales[rows, 2] = 0.001

    def update(self, frame: int) -> None:
        """Publish three seconds of history and two seconds of reference.

        Args:
            frame: Clip-relative frame index, clamped to the recorded interval.
        """
        if self._markers is None:
            raise RuntimeError("Floor trajectory markers have been closed.")
        frame = min(max(0, frame), len(self._actual) - 1)
        past = np.clip(frame + self._past_offsets, 0, len(self._actual) - 1)
        future = np.clip(frame + self._future_offsets, 0, len(self._reference) - 1)
        self._write_lines(0, self._reference[past], 0.042, 0.001)
        shaft, direction, head_length = self._trim_arrow(self._reference[future])
        self._write_lines(50, shaft, 0.032, 0.003)
        self._write_lines(100, self._actual[past], 0.028, 0.005)

        current, goal = self._actual[frame], self._reference[frame]
        for i, offset in enumerate(range(0, 12, 2)):
            self._write_lines(
                150 + i,
                np.array(
                    [
                        goal + (current - goal) * offset / 12,
                        goal + (current - goal) * (offset + 1) / 12,
                    ]
                ),
                0.018,
                0.007,
            )
        end = self._reference[future[-1]]
        angle = np.arctan2(direction[1], direction[0])
        self._positions[156:] = [[*current, 0.011], [*goal, 0.009], [*end, 0.0035]]
        self._rotations[158] = [0, 0, np.sin(angle / 2), np.cos(angle / 2)]
        arrow_length = max(head_length / 0.16, 1e-4)
        arrow_width = (
            max(max(0.035, head_length / 2) / 0.08, 1e-4) if head_length else 1e-4
        )
        self._scales[156:] = [
            [0.11, 0.11, 0.002],
            [1, 1, 1],
            [arrow_length, arrow_width, 1],
        ]
        self._markers.visualize(
            translations=self._positions,
            orientations=self._rotations,
            scales=self._scales,
            marker_indices=self._kinds,
        )

    def close(self) -> None:
        """Release this clip's markers before starting the next clip."""
        if self._markers is not None:
            self._markers.remove()
            self._markers = None
