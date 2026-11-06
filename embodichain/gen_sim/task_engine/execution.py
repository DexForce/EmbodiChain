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

"""Execute a sealed bundle through the standard demo and evidence lifecycles."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from .bundle_deployment import LoadedTaskProgramBundle
from .runtime_evidence import TaskEngineEvidenceReport, TaskEngineEvidenceSession

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
    from embodichain.lab.gym.envs.demo import DemoEpisodeResult, DemoExecutionCfg
    from embodichain.lab.task_program.integrations.simulation.recorded_contacts import (
        SimulationRecordedContactCfg,
    )

__all__ = ["TaskProgramBundleEpisode", "execute_task_program_bundle_episode"]


@dataclass(frozen=True, slots=True)
class TaskProgramBundleEpisode:
    """Original executor result and independently qualified physical report."""

    execution: DemoEpisodeResult
    evidence: TaskEngineEvidenceReport
    measurements: Mapping[str, Any] | None = None
    """The executor supplies detached raw packets after final acquisition."""
    binding_declarations: tuple[Mapping[str, Any], ...] | None = None
    """Complete effective declarations; None retains legacy caller compatibility."""

    def __post_init__(self) -> None:
        if self.binding_declarations is not None:
            object.__setattr__(
                self,
                "binding_declarations",
                tuple(deepcopy(dict(value)) for value in self.binding_declarations),
            )

    def to_raw_evidence(self) -> dict[str, Any]:
        """Export owned replay inputs without collecting facts or changing verdicts.

        Returns:
            Versioned bundle/run/episode-linked declarations, measurements,
            original execution metadata and acquisition diagnostics. Missing
            packets and unreached events retain their original representations.

        Raises:
            ValueError: A legacy caller did not supply effective declarations.
        """
        if self.binding_declarations is None:
            raise ValueError("Raw evidence requires effective binding declarations.")
        result = dict(
            schema_version="task_engine.raw_evidence/v2",
            bundle_digest=self.evidence.bundle_digest,
            run_id=self.evidence.run_id,
            episode_ids=list(self.evidence.episode_ids),
            env_ids=list(range(len(self.evidence.episode_ids))),
            bindings=deepcopy(list(self.binding_declarations)),
            measurements=deepcopy(self.measurements),
            runtime=deepcopy(self.execution.to_metadata()),
            acquisition_errors=deepcopy(dict(self.evidence.acquisition_errors)),
        )
        if self.evidence.cargo_observation is not None:
            result["cargo_observation"] = deepcopy(
                dict(self.evidence.cargo_observation)
            )
        return result


def execute_task_program_bundle_episode(
    bundle: LoadedTaskProgramBundle,
    environment: EmbodiedEnv,
    *,
    prepare_evidence: Callable[
        [TaskEngineEvidenceSession], Sequence[Mapping[str, Any]]
    ],
    final_evidence: Callable[
        [TaskEngineEvidenceSession, DemoEpisodeResult], Sequence[Mapping[str, Any]]
    ],
    timeline_cfg: SimulationRecordedContactCfg | None = None,
    control_operation_ids: Sequence[str] = (),
    episode_index: int = 0,
    attempt_id: int = 0,
    instruction: str | None = None,
    execution_cfg: DemoExecutionCfg | None = None,
    should_stop: Callable[[], bool] | None = None,
    initial_plan_observer: Callable[[dict[str, object]], None] | None = None,
) -> TaskProgramBundleEpisode:
    """Run one already-initialized bundle episode without replacing its executor.

    Args:
        bundle: Publicly loaded and statically preflighted sealed bundle.
        environment: Initialized environment configured from this bundle; its
            owner retains reset, persistence and final environment cleanup.
        prepare_evidence: Explicit declaration projector called after collectors
            bind at step zero. Returns serialized recorded/known final bindings.
        final_evidence: Explicit projector called after execution using actual
            session runtime metadata and final boundary. Returns fresh final
            declarations; historical predicates cannot be replaced.
        timeline_cfg: Existing-sensor declaration for recorded contact evidence.
        control_operation_ids: Explicit operations to collect from the selected bridge.
        episode_index: Standard executor's logical episode index.
        attempt_id: Standard executor's attempt identity.
        instruction: Optional explicit episode instruction.
        execution_cfg: Standard demo output/persistence configuration.
        should_stop: Optional standard cancellation predicate.
        initial_plan_observer: Optional data-only callback for the actual first
            canonical complete execution plan. This performs no extra planning.

    Returns:
        Original program result and separate physical/strict evidence report.
        No historical certificate or prior run identity is reused.

    The projector callbacks are trusted in-process application code, not bundle
    payloads. They must not step or reset. This entry selects the configured
    program once, captures each completed boundary through the public callback,
    and closes only owned evidence observers and its selected segment iterator.
    It never resets/closes the borrowed environment or relaxes success criteria.
    Current-loader inputs are rechecked before segment selection, after evidence
    preparation, and before final acquisition/reporting.
    """
    if type(bundle) is not LoadedTaskProgramBundle:
        raise TypeError("Bundle execution requires a publicly loaded sealed bundle.")
    if not callable(prepare_evidence) or not callable(final_evidence):
        raise TypeError("Evidence projectors must be explicit callables.")
    if initial_plan_observer is not None and not callable(initial_plan_observer):
        raise TypeError("initial_plan_observer must be callable or None.")
    bundle.assert_current_inputs()
    from embodichain.lab.gym.envs.demo import execute_demo_episode

    segments = (
        environment.create_demo_segments()
        if initial_plan_observer is None
        else environment.create_demo_segments(
            initial_plan_observer=initial_plan_observer
        )
    )
    if segments is None:
        raise ValueError("Configured bundle did not select Task Program segments.")
    try:
        bridge = environment.active_task_program_bridge
        with TaskEngineEvidenceSession(
            bundle,
            environment,
            bridge,
            timeline_cfg=timeline_cfg,
            control_operation_ids=control_operation_ids,
        ) as session:
            session.bind(prepare_evidence(session))
            bundle.assert_current_inputs()
            result = execute_demo_episode(
                environment,
                episode_index=episode_index,
                execution_cfg=execution_cfg,
                attempt_id=attempt_id,
                instruction=instruction,
                should_stop=should_stop,
                segments=segments,
                on_segment_completed=session.capture_boundary,
            )
            bundle.assert_current_inputs()
            final = final_evidence(session, result)
            report = session.finish(result, final_bindings=final)
            bundle.assert_current_inputs()
            return TaskProgramBundleEpisode(
                result, report, session.snapshots, session.binding_declarations
            )
    finally:
        close = getattr(segments, "close", None)
        if close is not None:
            close()
