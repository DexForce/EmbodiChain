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

from dataclasses import MISSING
from typing import Literal

from embodichain.utils import configclass

__all__ = ["MarkerPrototypeCfg", "MarkerGroupCfg"]


@configclass
class MarkerPrototypeCfg:
    """Reusable render-only geometry and its default RGBA appearance.

    Box, sphere, cylinder and cone have unit bounding dimensions. Capsule has
    unit diameter and total height two. Arrow points along +X from zero to one;
    frame contains three such arrows, colored RGB. Frame RGB stays fixed while
    its alpha follows ``color``. Mesh vertices use caller-defined local units.
    ``scale`` multiplies instance scale before rotation. All dimensions use meters.
    """

    shape: Literal[
        "box", "sphere", "cylinder", "capsule", "cone", "arrow", "frame", "mesh"
    ] = "box"
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    color: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0)
    vertices: object | None = None
    faces: object | None = None


@configclass
class MarkerGroupCfg:
    """Named prototypes shared by isolated environment batches.

    ``scope="env"`` uses all manager environments and environment-local poses.
    ``scope="world"`` uses one world-space batch. Dictionary order defines
    prototype indices. A standalone group defaults to one environment.
    """

    name: str = MISSING
    prototypes: dict[str, MarkerPrototypeCfg] = MISSING
    scope: Literal["env", "world"] = "env"
