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

"""Frozen logical recipes sharing a vectorized native collection World."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .configured import ExpansionRunCfg
    from embodichain.lab.sim.motion.expansion.combined import CombinedExpansionProfile
    from embodichain.lab.sim.scene.feasibility import JointSceneCandidate

__all__: list[str] = []


class CollectionDeadline(TimeoutError):
    """The shared collection wall-clock budget has expired."""


@dataclass(frozen=True)
class CohortPoint:
    """Point-owned values and provenance prepared before the first World step."""

    index: int
    cfg: ExpansionRunCfg
    revision: str
    motion: CombinedExpansionProfile | None
    scene: JointSceneCandidate | None
    artifacts: dict[str, dict[str, str]]


@dataclass(frozen=True)
class CohortSchedule:
    """Map journal ordinals to pinned parameter points and their local recipes."""

    rows: tuple[tuple[CohortPoint, int], ...]
    deadline: float
    viewer_gpu: int | None = None
