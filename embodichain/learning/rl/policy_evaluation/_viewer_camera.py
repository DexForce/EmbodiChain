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

"""Native window camera lifecycle for task policy evaluation."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from .camera import PolicyViewerCameraCfg


class _ViewerCamera:
    """Track a task target through public window camera operations."""

    def __init__(
        self,
        window: Any,
        cfg: PolicyViewerCameraCfg,
        target_pose: Callable[[], np.ndarray],
    ) -> None:
        self._window = window
        self._target_pose = target_pose
        self._eye_offset = np.asarray(cfg.eye_offset, dtype=np.float64)
        self._target_height = float(cfg.target_height)
        if (
            self._eye_offset.shape != (3,)
            or not np.isfinite(self._eye_offset).all()
            or not math.isfinite(self._target_height)
            or np.linalg.norm(self._eye_offset[:2]) == 0.0
        ):
            raise ValueError(
                "Policy Viewer camera requires a finite eye_offset with a "
                "nonzero horizontal offset and a finite target_height"
            )
        self._last_xy: np.ndarray | None = None
        self._enabled = False

    def reset(self) -> None:
        """Restore the preset using the current target position and yaw."""
        pose = np.asarray(self._target_pose(), dtype=np.float64)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError("Policy Viewer target pose must contain finite XYZ + XYZW")
        norm = np.linalg.norm(pose[3:])
        if norm == 0.0:
            raise ValueError("Policy Viewer target quaternion must be nonzero")
        x, y, z, w = pose[3:] / norm
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        dx, dy, dz = self._eye_offset
        offset = np.array(
            [cos_yaw * dx - sin_yaw * dy, sin_yaw * dx + cos_yaw * dy, dz]
        )
        target = np.array([pose[0], pose[1], self._target_height])
        self._window.set_look_at(target + offset, target, np.array([0.0, 0.0, 1.0]))
        self._window.set_camera_orbit_pan_enabled(False)
        self._last_xy = pose[:2].copy()
        self._enabled = True

    def update(self) -> None:
        """Translate the existing orbit without following height or rotation."""
        if not self._enabled:
            return
        xy = np.asarray(self._target_pose(), dtype=np.float64)[:2]
        delta = np.zeros(3)
        delta[:2] = xy - self._last_xy
        self._window.translate_camera_orbit(delta)
        self._last_xy = xy.copy()

    def toggle(self) -> None:
        """Switch between free pan and the task's default tracking view."""
        if self._enabled:
            self._enabled = False
            self._window.set_camera_orbit_pan_enabled(True)
        else:
            self.reset()
