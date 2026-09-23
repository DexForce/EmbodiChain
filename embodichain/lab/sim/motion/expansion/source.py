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

"""Source-neutral adapter contracts for expert trajectory generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from .contracts import SceneCase, TrajectoryTemplate

__all__ = ["SourceContext", "SourceAdapter"]

SourceT = TypeVar("SourceT")


@dataclass(frozen=True)
class SourceContext:
    """Stable source and scene identity supplied by a generation coordinator."""

    source_id: str
    source_revision: str
    unit_id: str
    scene_case: SceneCase
    control_dt: float

    def __post_init__(self) -> None:
        for name in ("source_id", "source_revision", "unit_id"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if type(self.control_dt) not in (int, float) or self.control_dt <= 0:
            raise ValueError("control_dt must be positive")


class SourceAdapter(Protocol, Generic[SourceT]):
    """Convert one expert source unit into a reusable trajectory template.

    Implementations may be handwritten, MotionGenerator, Atomic Action, or
    Task Program adapters. They must not own environment slots, reset/step a
    simulator, allocate candidate identities, or persist episodes.
    """

    kind: str

    def export_template(
        self,
        source: SourceT,
        *,
        context: SourceContext,
    ) -> TrajectoryTemplate:
        """Export one complete, phase-annotated qpos template."""
