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

"""Host-independent candidate generation and bounded logical queuing."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from .cfg import TrajectoryGenerationJobCfg
from .contracts import CandidateSpec, SceneCase, TrajectoryTemplate
from .session import GenerationSession
from .source import SourceAdapter, SourceContext
from .variants import expand_trajectory_variants

__all__ = ["CandidateCoordinator", "CandidateWorkItem"]


@dataclass(frozen=True)
class CandidateWorkItem:
    """One logical candidate ready for planning validation or slot assignment."""

    spec: CandidateSpec
    template: TrajectoryTemplate


class CandidateCoordinator:
    """Generate source-neutral candidates without owning physical execution.

    The coordinator owns only template expansion and a bounded logical queue.
    Initial-state restoration, physical slots, measured validation, and
    persistence remain host responsibilities. A caller must register the source
    case with the supplied :class:`GenerationSession` before proposing.
    """

    def __init__(
        self,
        cfg: TrajectoryGenerationJobCfg,
        *,
        session: GenerationSession,
        source_adapter: SourceAdapter[object],
        source_context: SourceContext,
        joint_limits: torch.Tensor,
        velocity_limits: torch.Tensor | None = None,
        acceleration_limits: torch.Tensor | None = None,
        task_jacobians: torch.Tensor | None = None,
    ) -> None:
        if not isinstance(cfg, TrajectoryGenerationJobCfg):
            raise TypeError("cfg must be a TrajectoryGenerationJobCfg")
        if not isinstance(session, GenerationSession):
            raise TypeError("session must be a GenerationSession")
        if not callable(getattr(source_adapter, "export_template", None)):
            raise TypeError("source_adapter must implement export_template")
        if not isinstance(source_context, SourceContext):
            raise TypeError("source_context must be a SourceContext")
        if not isinstance(joint_limits, torch.Tensor):
            raise TypeError("joint_limits must be a tensor")
        if (velocity_limits is None) != (acceleration_limits is None):
            raise ValueError(
                "velocity_limits and acceleration_limits must be supplied together"
            )
        self._cfg = TrajectoryGenerationJobCfg.from_mapping(cfg.to_dict())
        self._session = session
        self._source_adapter = source_adapter
        self._source_context = source_context
        self._joint_limits = joint_limits.detach().clone()
        self._velocity_limits = (
            None if velocity_limits is None else velocity_limits.detach().clone()
        )
        self._acceleration_limits = (
            None
            if acceleration_limits is None
            else acceleration_limits.detach().clone()
        )
        self._task_jacobians = (
            None if task_jacobians is None else task_jacobians.detach().clone()
        )
        self._pending: deque[CandidateWorkItem] = deque()

    @property
    def pending_count(self) -> int:
        """Return the number of logical candidates waiting for host execution."""
        return len(self._pending)

    @property
    def source_case(self) -> SceneCase:
        """Return the scene case shared by the current source context."""
        return self._source_context.scene_case

    def enqueue_source(
        self,
        source: object,
        *,
        count: int,
        affordance_selection: Mapping[str, object] | None = None,
        compatibility_key: str | None = None,
    ) -> tuple[CandidateWorkItem, ...]:
        """Export one source unit, expand it, and enqueue bounded candidates.

        Args:
            source: Source value consumed by the registered SourceAdapter.
            count: Maximum accepted trajectory variants for this source unit.
            affordance_selection: Optional lineage for the Affordance proposal
                that produced the source template.
            compatibility_key: Optional host compatibility bucket. When omitted,
                a stable case/control/validator key is derived.

        Returns:
            The newly enqueued work items in deterministic variant order.

        Raises:
            BufferError: If the logical ready queue cannot hold the result.
            ValueError: If source export or variant expansion fails.
        """
        if type(count) is not int or count < 1:
            raise ValueError("count must be a positive integer")
        budget = self._cfg.scheduling.candidate_budget
        if len(self._pending) + count > budget:
            raise BufferError("candidate queue budget is full")
        template = self._source_adapter.export_template(
            source,
            context=self._source_context,
        )
        if not isinstance(template, TrajectoryTemplate):
            raise TypeError(
                "SourceAdapter.export_template must return TrajectoryTemplate"
            )
        variants = expand_trajectory_variants(
            template,
            case=self._source_context.scene_case,
            count=count,
            cfg=self._cfg.augmentation,
            joint_limits=self._joint_limits,
            control_dt=self._source_context.control_dt,
            velocity_limits=self._velocity_limits,
            acceleration_limits=self._acceleration_limits,
            task_jacobians=self._task_jacobians,
        )
        key = compatibility_key or (
            f"{self._source_context.scene_case.scene_case_id}:"
            f"{self._source_context.scene_case.initial_state_id}:"
            f"dt={self._source_context.control_dt:g}:"
            f"validator={template.validator_id}"
        )
        affordance = {} if affordance_selection is None else dict(affordance_selection)
        created: list[CandidateWorkItem] = []
        for index, variant in enumerate(variants.variants):
            provisional = variants.candidates.identities[index]
            identity, _ = self._session.propose(
                self._source_context.scene_case.scene_case_id,
                self._source_context.scene_case.initial_state_id,
                source_id=self._source_context.source_id,
                source_revision=self._source_context.source_revision,
                template_id=self._source_context.unit_id,
                operator_id=variant.spatial_operator,
                geometry_family_id=provisional.geometry_family_id,
            )
            row = variants.candidates.row(index)
            length = int(row.valid_length[0].item())
            row_template = TrajectoryTemplate(
                source_id=self._source_context.source_id,
                source_revision=self._source_context.source_revision,
                template_id=self._source_context.unit_id,
                joint_names=row.joint_names,
                positions=row.positions[0, :length],
                dt=row.dt[0, :length],
                phases=row.phases[0],
                allowed_operators=template.allowed_operators,
                validator_id=template.validator_id,
                controlled_joint_indices=template.controlled_joint_indices,
            )
            spec = CandidateSpec(
                identity=identity,
                affordance_selection=affordance,
                trajectory_variant=variant.factors,
                compatibility_key=key,
                estimated_cost=float(row.dt[0, :length].sum().item()),
            )
            item = CandidateWorkItem(spec=spec, template=row_template)
            self._pending.append(item)
            created.append(item)
        return tuple(created)

    def take_next(self) -> CandidateWorkItem | None:
        """Remove and return the next candidate in deterministic FIFO order."""
        return None if not self._pending else self._pending.popleft()
