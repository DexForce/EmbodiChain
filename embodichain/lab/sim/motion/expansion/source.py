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

from dataclasses import dataclass, replace
import math
from typing import ClassVar, Generic, Protocol, TypeVar

import torch

from embodichain.lab.sim.motion.planners import PlanResult

from .contracts import SceneCase, TrajectoryPhase, TrajectoryTemplate

__all__ = [
    "SourceContext",
    "SourceAdapter",
    "TemplateSourceAdapter",
    "PlanResultSourceAdapter",
]

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
        if (
            type(self.control_dt) not in (int, float)
            or not math.isfinite(float(self.control_dt))
            or self.control_dt <= 0
        ):
            raise ValueError("control_dt must be finite and positive")


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


@dataclass(frozen=True)
class TemplateSourceAdapter:
    """Adapt an already annotated handwritten qpos template."""

    kind: ClassVar[str] = "handwritten"

    def export_template(
        self,
        source: TrajectoryTemplate,
        *,
        context: SourceContext,
    ) -> TrajectoryTemplate:
        """Return an owned template with coordinator-owned source identity."""
        if not isinstance(source, TrajectoryTemplate):
            raise TypeError("source must be a TrajectoryTemplate")
        if source.positions.shape[0] < 2:
            raise ValueError("a source template needs at least two samples")
        return replace(
            source,
            source_id=context.source_id,
            source_revision=context.source_revision,
            template_id=context.unit_id,
        )


@dataclass(frozen=True)
class PlanResultSourceAdapter:
    """Adapt one single-row MotionGenerator result to a qpos template.

    Phase annotations and editable operators are supplied by the caller because
    a generic PlanResult does not own semantic contact or hold boundaries.
    """

    joint_names: tuple[str, ...]
    phases: tuple[TrajectoryPhase, ...] = ()
    validator_id: str = "default"
    kind: ClassVar[str] = "motion_generator"

    def export_template(
        self,
        source: PlanResult,
        *,
        context: SourceContext,
    ) -> TrajectoryTemplate:
        """Convert one successful single-row result with explicit timing."""
        if not isinstance(source, PlanResult):
            raise TypeError("source must be a PlanResult")
        if source.positions is None or source.dt is None:
            raise ValueError("PlanResult must contain positions and dt")
        if source.positions.ndim != 3 or source.positions.shape[0] != 1:
            raise ValueError("the initial source adapter supports one plan row")
        if source.dt.shape != source.positions.shape[:2]:
            raise ValueError("PlanResult dt must match positions batch and horizon")
        success = source.success
        if isinstance(success, torch.Tensor):
            if success.numel() != 1 or not bool(success.reshape(-1)[0].item()):
                raise ValueError("cannot export an unsuccessful PlanResult")
        elif success is not True:
            raise ValueError("cannot export an unsuccessful PlanResult")
        if len(self.joint_names) != source.positions.shape[-1]:
            raise ValueError("joint_names must match PlanResult joint width")
        return TrajectoryTemplate(
            source_id=context.source_id,
            source_revision=context.source_revision,
            template_id=context.unit_id,
            joint_names=tuple(self.joint_names),
            positions=source.positions[0],
            dt=source.dt[0],
            phases=self.phases,
            validator_id=self.validator_id,
            controlled_joint_indices=tuple(range(len(self.joint_names))),
        )
