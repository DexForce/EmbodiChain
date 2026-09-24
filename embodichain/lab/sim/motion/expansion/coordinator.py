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
import hashlib
import json

import torch

from .cfg import TrajectoryGenerationJobCfg
from .contracts import (
    CandidateSpec,
    CandidateTrajectoryBatch,
    ProposalRequest,
    SceneCase,
    TrajectoryTemplate,
    ValidationResult,
    _json,
)
from .session import GenerationSession
from .source import SourceAdapter, SourceContext
from .variants import TrajectoryVariant, expand_trajectory_variants

__all__ = ["CandidateCoordinator", "CandidateWorkItem"]


def _plain_json(value: object) -> object:
    """Convert validated immutable JSON values to ordinary containers."""
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


def _trajectory_fingerprint(template: TrajectoryTemplate) -> str:
    """Return an exact content digest for one unpadded candidate template."""
    metadata = {
        "joint_names": template.joint_names,
        "positions_shape": tuple(template.positions.shape),
        "positions_dtype": str(template.positions.dtype),
        "dt_shape": tuple(template.dt.shape),
        "dt_dtype": str(template.dt.dtype),
        "phases": tuple(
            (
                phase.phase_id,
                phase.start_index,
                phase.stop_index,
                phase.kind,
                phase.allowed_operators,
            )
            for phase in template.phases
        ),
        "allowed_operators": template.allowed_operators,
        "controlled_joint_indices": template.controlled_joint_indices,
    }
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    )
    digest.update(template.positions.detach().cpu().contiguous().numpy().tobytes())
    digest.update(template.dt.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _compatibility_key(
    template: TrajectoryTemplate,
    *,
    source_context: SourceContext,
    backend_id: str,
    observation_profiles: tuple[str, ...],
) -> str:
    """Derive one strict host compatibility bucket from complete layout data."""
    payload = (
        source_context.scene_case.scene_case_id,
        source_context.scene_case.initial_state_id,
        template.joint_names,
        tuple(
            (
                phase.phase_id,
                phase.start_index,
                phase.stop_index,
                phase.kind,
            )
            for phase in template.phases
        ),
        source_context.control_dt,
        backend_id,
        template.validator_id,
        observation_profiles,
    )
    digest = hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode()
    ).hexdigest()
    return "strict:" + digest


@dataclass(frozen=True)
class CandidateWorkItem:
    """One logical candidate ready for planning validation or slot assignment."""

    spec: CandidateSpec
    template: TrajectoryTemplate

    def to_batch(self) -> CandidateTrajectoryBatch:
        """Return the single-row logical batch consumed by GenerationSession."""
        return CandidateTrajectoryBatch(
            positions=self.template.positions.unsqueeze(0),
            dt=self.template.dt.unsqueeze(0),
            valid_length=torch.tensor(
                [self.template.positions.shape[0]], dtype=torch.int64
            ),
            identities=(self.spec.identity,),
            joint_names=self.template.joint_names,
            phases=(self.template.phases,),
            factors=(self.spec.trajectory_variant,),
        )


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
        backend_id: str = "unspecified",
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
        if type(backend_id) is not str or not backend_id.strip():
            raise ValueError("backend_id must be a nonempty string")
        if (velocity_limits is None) != (acceleration_limits is None):
            raise ValueError(
                "velocity_limits and acceleration_limits must be supplied together"
            )
        self._cfg = TrajectoryGenerationJobCfg.from_mapping(cfg.to_dict())
        self._session = session
        self._source_adapter = source_adapter
        self._source_context = source_context
        self._joint_limits = joint_limits.detach().clone()
        self._backend_id = backend_id
        self._observation_profiles = (
            self._cfg.observation.profiles if self._cfg.observation.enabled else ()
        )
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
        self._seen_fingerprints: set[str] = set()

    @property
    def pending_count(self) -> int:
        """Return the number of logical candidates waiting for host execution."""
        return len(self._pending)

    @property
    def session(self) -> GenerationSession:
        """Return the session that owns candidate lifecycle and receipts."""
        return self._session

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
        template = self._source_adapter.export_template(
            source,
            context=self._source_context,
        )
        if not isinstance(template, TrajectoryTemplate):
            raise TypeError(
                "SourceAdapter.export_template must return TrajectoryTemplate"
            )
        normalized_affordance = _json(
            {} if affordance_selection is None else dict(affordance_selection)
        )
        affordance = _plain_json(normalized_affordance)
        assert isinstance(affordance, dict)
        affordance_payload = json.dumps(
            affordance,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        operation_id = (
            "trajectory_expansion:"
            + hashlib.sha256(affordance_payload.encode()).hexdigest()
        )
        generator = self._session.generation_generator(
            self._source_context.scene_case.scene_case_id,
            self._source_context.scene_case.initial_state_id,
            source_id=self._source_context.source_id,
            source_revision=self._source_context.source_revision,
            template_id=self._source_context.unit_id,
            operation_id=operation_id,
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
            generator=generator,
        )
        if compatibility_key is not None and (
            type(compatibility_key) is not str or not compatibility_key.strip()
        ):
            raise ValueError("compatibility_key must be a nonempty string or None")
        key = compatibility_key or _compatibility_key(
            template,
            source_context=self._source_context,
            backend_id=self._backend_id,
            observation_profiles=self._observation_profiles,
        )
        prepared: list[tuple[TrajectoryVariant, TrajectoryTemplate, str]] = []
        batch_fingerprints: set[str] = set()
        for index, variant in enumerate(variants.variants):
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
            fingerprint = _trajectory_fingerprint(row_template)
            if (
                fingerprint in self._seen_fingerprints
                or fingerprint in batch_fingerprints
            ):
                continue
            batch_fingerprints.add(fingerprint)
            prepared.append((variant, row_template, fingerprint))
        if not prepared:
            return ()
        budget = self._cfg.scheduling.candidate_budget
        if len(self._pending) + len(prepared) > budget:
            raise BufferError("candidate queue budget is full")
        allocations = self._session.propose_many(
            tuple(
                ProposalRequest(
                    self._source_context.scene_case.scene_case_id,
                    self._source_context.scene_case.initial_state_id,
                    self._source_context.source_id,
                    self._source_context.source_revision,
                    self._source_context.unit_id,
                    variant.spatial_operator,
                )
                for variant, _, _ in prepared
            )
        )
        created: list[CandidateWorkItem] = []
        for (variant, row_template, fingerprint), (identity, _) in zip(
            prepared,
            allocations,
            strict=True,
        ):
            spec = CandidateSpec(
                identity=identity,
                affordance_selection=affordance,
                trajectory_variant=variant.factors,
                compatibility_key=key,
                estimated_cost=float(row_template.dt.sum().item()),
                observation_profiles=self._observation_profiles,
            )
            item = CandidateWorkItem(spec=spec, template=row_template)
            created.append(item)
        self._pending.extend(created)
        self._seen_fingerprints.update(fingerprint for _, _, fingerprint in prepared)
        return tuple(created)

    def take_next(self) -> CandidateWorkItem | None:
        """Remove and return the next candidate in deterministic FIFO order."""
        return None if not self._pending else self._pending.popleft()

    def admit_planned(
        self,
        item: CandidateWorkItem,
        validation: ValidationResult,
    ) -> None:
        """Admit one host-validated candidate to GenerationSession's ready pool.

        The host owns collision/path validation; this method only transfers the
        immutable candidate and its validation result into the session state.
        It never executes or persists the candidate.
        """
        if not isinstance(item, CandidateWorkItem):
            raise TypeError("item must be a CandidateWorkItem")
        if not isinstance(validation, ValidationResult):
            raise TypeError("validation must be a ValidationResult")
        self._session.add_planned(item.to_batch(), validation)
