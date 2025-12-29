# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class DimensionConstraint:
    """Configuration for dimensional constraints in workspace analysis."""

    min_bounds: np.ndarray | None = None
    """Minimum bounds for workspace [x_min, y_min, z_min] in meters."""

    max_bounds: np.ndarray | None = None
    """Maximum bounds for workspace [x_max, y_max, z_max] in meters."""

    joint_limits_scale: float = 1.0
    """Scale factor for joint limits (1.0 = use full range, 0.8 = use 80% of range)."""

    exclude_zones: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    """List of excluded zones as [(min_bounds, max_bounds), ...]. Robot end-effector should avoid these regions."""

    ground_height: float = 0.0
    """Ground plane height in meters. Points below this will be filtered out."""

    enforce_collision_free: bool = False
    """Whether to enforce collision-free constraints during analysis."""

    self_collision_check: bool = False
    """Whether to check for self-collision when analyzing workspace."""
