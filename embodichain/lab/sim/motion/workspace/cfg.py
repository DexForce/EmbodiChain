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

"""Runtime workspace configuration."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from embodichain.utils import configclass

__all__ = ["RobotWorkspaceCfg"]


@configclass
class RobotWorkspaceCfg:
    """Runtime configuration for a control-part workspace cache."""

    cache_path: str = MISSING
    """Path to a workspace cache entry directory or ``results.npz`` file."""

    strategy: Literal["point_uniform", "voxel_uniform"] = "voxel_uniform"
    """Default runtime sampling strategy."""

    voxel_size: float = 0.03
    """Cartesian voxel edge length in meters for voxel-uniform sampling."""

    min_score: float | None = None
    """Optional minimum cached reachability score accepted for sampling."""
