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

"""Shared policy for physical grasp labeling and cache identity."""

from __future__ import annotations

import math

import numpy as np

from embodichain.utils import configclass

__all__ = ["GraspSamplingCfg"]

DEFAULT_NUM_ENVS = 2400
VALIDATION_REVISION = 6


@configclass
class GraspSamplingCfg:
    """Placement, approach, surface sampling and pose NMS parameters."""

    stable_poses: int = 5
    directions: int = 64
    candidates_per_direction: int = 100
    sample_count: int = 20_000
    max_drops: int = 100
    drop_height_m: float = 0.05
    min_rotation_deg: float = 20.0
    support_up_axis: list[float] | None = None
    approach_tolerance_deg: float = 5.0
    nms_distance_m: float = 0.003
    nms_angle_deg: float = 5.0
    seed: int = 0

    def __post_init__(self) -> None:
        for key in (
            "stable_poses",
            "directions",
            "candidates_per_direction",
            "sample_count",
            "max_drops",
        ):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.support_up_axis is not None:
            axis = np.asarray(self.support_up_axis, dtype=float)
            if (
                axis.shape != (3,)
                or not np.isfinite(axis).all()
                or np.linalg.norm(axis) < 1e-8
            ):
                raise ValueError(
                    "support_up_axis must be a finite nonzero object-frame vector"
                )
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer")
        for key in (
            "drop_height_m",
            "min_rotation_deg",
            "approach_tolerance_deg",
            "nms_distance_m",
            "nms_angle_deg",
        ):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
            if key.endswith("_deg") and value > 180:
                raise ValueError(f"{key} must be at most 180 degrees")
