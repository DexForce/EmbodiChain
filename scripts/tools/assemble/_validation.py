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

"""Generic rigid placement checks with VISACD acceleration and solid verification."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.tools.mug_rack_pose._collision import (
    ExactCollision,
    HullCollision,
    decompose,
)
from scripts.tools.mug_rack_pose._geometry import load_mesh
from scripts.tools.mug_rack_pose._json_io import pose_matrix


class PlacementValidator:
    """Validate generated meshes and gravity-supported relative transforms."""

    def __init__(self, geometry: dict, settings: dict, cache: Path) -> None:
        self.settings = settings
        self.base = load_mesh(Path(geometry["assets"]["base"]["path"]))
        self.assemble = load_mesh(Path(geometry["assets"]["assemble"]["path"]))
        self.exact = ExactCollision(self.base, self.assemble)
        self.volume_tolerance = max(
            1e-15, min(self.base.volume, self.assemble.volume) * 1e-8
        )
        self.hulls = None
        self.hull_counts = None
        if settings["collision"] == "visacd":
            parts = [
                decompose(mesh, cache, settings["concavity"], settings["max_parts"])
                for mesh in (self.base, self.assemble)
            ]
            self.hulls = HullCollision(*parts)
            self.hull_counts = list(map(len, parts))

    def _intersects(self, pose: np.ndarray, accelerated: bool = False) -> bool:
        if accelerated and self.hulls is not None and not self.hulls.collides(pose):
            return False
        return (
            self.exact.surface_collision(pose)
            or self.exact.intersection_volume(pose) > self.volume_tolerance
        )

    def validate(self, value: object) -> dict:
        """Check an SE(3) transform against the original solid meshes.

        Args:
            value: Matrix mapping assemble-local meters to base-local meters.

        Returns:
            Numerical observations and explicit acceptance or rejection reasons.
        """
        pose = pose_matrix(value)
        v = self.settings
        surface = self.exact.surface_collision(pose)
        volume = self.exact.intersection_volume(pose)
        distance = self.exact.distance(pose)
        down, up = pose.copy(), pose.copy()
        down[2, 3] -= v["probe"]
        up[2, 3] += v["probe"]
        down_volume = self.exact.intersection_volume(down)
        up_clear = not self._intersects(up)
        reasons = []
        if surface or volume > self.volume_tolerance:
            reasons.append("Original meshes intersect or one solid contains the other")
        if not 0 < distance <= v["contact_tolerance"]:
            reasons.append(
                "Pose needs a positive near-contact gap within contact_tolerance"
            )
        if down_volume <= self.volume_tolerance:
            reasons.append(
                "No downward support: the gravity probe does not enter the base"
            )
        if not up_clear:
            reasons.append("The upward release probe intersects the base")
        axis_error = None
        if v["assemble_axis"] is not None:
            dot = (pose[:3, :3] @ v["assemble_axis"]) @ np.asarray(v["target_axis"])
            axis_error = float(np.degrees(np.arccos(np.clip(dot, -1, 1))))
            if axis_error > v["axis_tolerance_degrees"] + 1e-6:
                reasons.append("Configured assembly-axis orientation is not satisfied")
        assembled_vertices = self.assemble.vertices @ pose[:3, :3].T + pose[:3, 3]
        if (
            assembled_vertices[:, 2].min()
            < self.base.bounds[0, 2] - v["contact_tolerance"]
        ):
            reasons.append("Assemble object extends below the base bottom plane")
        return {
            "accepted": not reasons,
            "reasons": reasons,
            "surface_collision": surface,
            "intersection_volume_m3": volume,
            "volume_tolerance_m3": float(self.volume_tolerance),
            "distance_m": distance,
            "down_probe_intersection_m3": down_volume,
            "up_probe_clear": up_clear,
            "axis_error_degrees": axis_error,
            "visacd_hull_collision": self.hulls.collides(pose) if self.hulls else None,
            "visacd_hull_counts": self.hull_counts,
            "checks": "Static geometry, near contact, downward support probe, upward release, optional axis constraint; no dynamics or continuous insertion-path proof",
        }

    def evaluate(self, value: object) -> dict:
        """Settle a collision-free model proposal downward, then verify the result.

        Args:
            value: Proposed relative 4x4 pose.

        Returns:
            Proposed/final matrices, downward adjustment, and original-mesh checks.
        """
        initial = pose_matrix(value)
        pose = initial.copy()
        v = self.settings
        if not self._intersects(initial) and v["settle_distance"] > 0:
            previous = 0.0
            steps = int(np.ceil(v["settle_distance"] / v["settle_step"]))
            for i in range(1, steps + 1):
                offset = min(i * v["settle_step"], v["settle_distance"])
                probe = initial.copy()
                probe[2, 3] -= offset
                if self._intersects(probe, accelerated=True):
                    low, high = previous, offset
                    for _ in range(24):
                        mid = (low + high) / 2
                        probe[2, 3] = initial[2, 3] - mid
                        if self._intersects(probe):
                            high = mid
                        else:
                            low = mid
                    # Offset from the first contact boundary back into free space.
                    pose[2, 3] -= max(0.0, low - v["contact_gap"])
                    break
                previous = offset
        return {
            "proposed_T_base_assemble": initial.tolist(),
            "T_base_assemble": pose.tolist(),
            "settled_down_m": float(initial[2, 3] - pose[2, 3]),
            "validation": self.validate(pose),
        }
