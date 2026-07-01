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

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "GravityDropRequest",
    "GravityDropResult",
]


@dataclass(frozen=True)
class GravityDropRequest:
    """Request to drop a GLB asset under gravity simulation."""

    glb_path: Path
    max_convex_hull_num: int = 32
    convex_decomposition_method: str = "vhacd"
    initial_height: float | None = None


@dataclass(frozen=True)
class GravityDropResult:
    """Result of dropping a GLB asset under gravity."""

    final_pose: Any
