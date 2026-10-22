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

"""Real qpos sources adapted to the existing source-neutral candidate contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar
import math

import torch

from embodichain.data_pipeline.real2sim.normalization import NormalizedReal2SimUnit
from .contracts import TrajectoryPhase, TrajectoryTemplate
from .source import SourceContext

__all__ = ["Real2SimSourceAdapter"]


@dataclass(frozen=True)
class Real2SimSourceAdapter:
    """Export a reviewed normalized unit without owning execution or persistence.

    Editable free windows are inset by one sample at both ends. This leaves the
    source phase endpoints and adjacent central-difference stencils untouched.
    Full semantic phases remain in the normalized unit, not these narrower
    augmentation windows. V1 permits only same-grid joint_residual expansion.

    Unverified inputs may be normalized for offline diagnostics, but cannot
    enter this generation adapter. Physical success still requires host checks.
    """

    kind: ClassVar[str] = "real2sim_qpos"

    def export_template(
        self,
        source: NormalizedReal2SimUnit,
        *,
        context: SourceContext,
    ) -> TrajectoryTemplate:
        """Export an owned full-joint template with conservative permissions.

        Args:
            source: Audited qpos-authoritative unit on the host control clock.
            context: Coordinator-owned source identity and restored scene case.

        Returns:
            A template ready for the existing CandidateCoordinator. This method
            does not assign candidate IDs, restore scenes or write datasets.
        """
        if not isinstance(source, NormalizedReal2SimUnit) or not isinstance(
            context, SourceContext
        ):
            raise TypeError(
                "Real2SimSourceAdapter requires normalized source and SourceContext values"
            )
        if source.report["authority"] != "qpos":
            raise ValueError("Real2SimSourceAdapter requires qpos authority")
        expected = (source.source.source_id, source.revision, source.source.unit_id)
        actual = (context.source_id, context.source_revision, context.unit_id)
        if actual != expected:
            raise ValueError("SourceContext identity must match the normalized unit")
        if not source.eligible:
            raise ValueError(
                "Unverified real2sim input cannot enter the qualified generation path"
            )
        if not math.isclose(
            context.control_dt,
            float(source.report["control_dt"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Source and host control clocks differ; normalize again explicitly"
            )
        # Existing joint operators do not project arbitrary coupling graphs.
        # Keep coupled drivers locked rather than producing inconsistent full states.
        coupled_parents = {m["parent"] for m in source.report["mimics"]}
        expanded = {source.joint_names[i] for i in source.expandable_joint_indices}
        if expanded & coupled_parents:
            raise ValueError("V1 expansion must keep mimic-driver groups locked")
        phases = []
        for phase in source.phases:
            start, stop = phase.start, phase.stop
            operators = phase.allowed_operators
            if operators:
                if set(operators) != {"joint_residual"}:
                    raise ValueError(
                        "V1 real2sim permits only same-grid joint_residual expansion"
                    )
                if not source.expandable_joint_indices:
                    raise ValueError(
                        "Expansion requires explicitly authorized independent joints"
                    )
                start += 1
                stop -= 1
                if stop - start < 3:
                    raise ValueError(
                        "Free phase is too short for derivative-protected expansion"
                    )
            phases.append(
                TrajectoryPhase(phase.name, start, stop, phase.kind, operators)
            )
        allowed = tuple(sorted({op for p in phases for op in p.allowed_operators}))
        return TrajectoryTemplate(
            source_id=context.source_id,
            source_revision=context.source_revision,
            template_id=context.unit_id,
            joint_names=source.joint_names,
            positions=torch.from_numpy(source.positions.copy()),
            dt=torch.from_numpy(source.dt.copy()),
            phases=tuple(phases),
            allowed_operators=allowed,
            controlled_joint_indices=source.expandable_joint_indices,
        )
