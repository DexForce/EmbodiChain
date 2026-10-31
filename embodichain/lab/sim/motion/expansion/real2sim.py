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

"""Adapt normalized fixed-scene source units to expansion templates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch

from embodichain.data_pipeline.real2sim import NormalizedReal2SimUnit

from .contracts import TrajectoryPhase, TrajectoryTemplate
from .source import SourceContext

__all__ = ["Real2SimSourceAdapter"]


@dataclass(frozen=True)
class Real2SimSourceAdapter:
    """Export a normalized qpos unit under explicit controlled-joint ownership.

    This adapter does not move a scene, validate the initial state, or certify
    a grasp. The host must ensure that source geometry and initial state match;
    changing object poses requires a separate Cartesian replan.

    Args:
        controlled_joint_indices: Full-layout joints authorized for expansion.
        validator_id: Host-owned physical validation contract identifier.
    """

    controlled_joint_indices: tuple[int, ...]
    validator_id: str = "default"
    kind: ClassVar[str] = "real2sim"

    def export_template(
        self,
        source: NormalizedReal2SimUnit,
        *,
        context: SourceContext,
    ) -> TrajectoryTemplate:
        """Return an owned, explicitly phased full-joint trajectory template."""
        if not isinstance(source, NormalizedReal2SimUnit):
            raise TypeError("source must be a NormalizedReal2SimUnit")
        if not isinstance(context, SourceContext):
            raise TypeError("context must be a SourceContext")
        if (
            context.source_id != source.source.source_id
            or context.source_revision != source.source.source_revision
            or context.unit_id != source.source.unit_id
        ):
            raise ValueError("source context must match normalized source identity")
        intervals = np.diff(source.timestamps)
        if not np.allclose(intervals, context.control_dt, atol=1e-9, rtol=0):
            raise ValueError("source control grid does not match the host context")
        controlled = tuple(self.controlled_joint_indices)
        if (
            not controlled
            or len(set(controlled)) != len(controlled)
            or any(
                type(index) is not int or not 0 <= index < len(source.joint_names)
                for index in controlled
            )
        ):
            raise ValueError("controlled joints must be explicit unique joint indices")
        phases = tuple(
            TrajectoryPhase(
                phase.name, phase.start, phase.stop, phase.kind, phase.allowed_operators
            )
            for phase in source.phases
        )
        operators = tuple(
            dict.fromkeys(
                operator for phase in phases for operator in phase.allowed_operators
            )
        )
        dt = np.concatenate(([0.0], intervals))
        return TrajectoryTemplate(
            source_id=context.source_id,
            source_revision=context.source_revision,
            template_id=context.unit_id,
            joint_names=source.joint_names,
            positions=torch.from_numpy(np.array(source.qpos, copy=True)),
            dt=torch.from_numpy(dt),
            phases=phases,
            allowed_operators=operators,
            controlled_joint_indices=controlled,
            validator_id=self.validator_id,
        )
