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

"""Connect scene candidates to the existing Task Program demo executor."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import inspect
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch

from embodichain.lab.gym.envs._json import json_safe_copy
from embodichain.lab.gym.envs.demo import (
    DemoEpisodeResult,
    DemoExecutionCfg,
    execute_demo_episode,
)
from embodichain.lab.sim.motion.expansion import ValidationResult
from embodichain.lab.sim.scene.expansion import SceneVariant
from embodichain.lab.task_program import CompiledTaskProgram, TaskProgramCfg

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
    from embodichain.lab.sim.motion.expansion import CombinedExpansionProfile
    from embodichain.lab.task_program.runtime.expansion import (
        TaskProgramNominalSourceFactory,
        TaskProgramPlanTransformFactory,
    )

__all__ = [
    "ScenePreparationResult",
    "SceneVariantEpisodeResult",
    "execute_scene_variants",
]


def _validation_metadata(
    validation: ValidationResult | None,
) -> list[dict[str, object]]:
    if validation is None:
        return []
    return [
        {
            "check_id": check.check_id,
            "status": check.status,
            "detail": check.detail,
            "metrics": dict(check.metrics),
        }
        for check in validation.checks
    ]


def _freeze_metadata(value: object) -> object:
    """Recursively freeze already validated and owned JSON-compatible data."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_metadata(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_metadata(item) for item in value)
    return value


@dataclass(frozen=True)
class ScenePreparationResult:
    """Host-owned settled state identity and explicit initial-state checks.

    Args:
        initial_state_id: Identity of the actual state saved by the host after
            preparation and settling. This is distinct from the proposal ID.
        validation: Required initial-state checks based on the actual scene.
            Empty or unavailable checks reject the candidate.
        metadata: JSON-compatible provenance such as scene/integration versions
            and the identity of the host's saved initial-state artifact.
    """

    initial_state_id: str
    validation: ValidationResult
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            type(self.initial_state_id) is not str
            or not self.initial_state_id
            or self.initial_state_id.strip() != self.initial_state_id
        ):
            raise ValueError("initial_state_id must be a nonempty stripped string")
        if not isinstance(self.validation, ValidationResult):
            raise TypeError("validation must be a ValidationResult")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        owned = json_safe_copy(self.metadata, field_name="preparation.metadata")
        object.__setattr__(self, "metadata", _freeze_metadata(owned))


@dataclass(frozen=True)
class SceneVariantEpisodeResult:
    """Independent scene preparation, program execution, and measured outcome.

    Args:
        env_id: Physical environment row represented by this result.
        variant: Exact proposed scene evaluated by this attempt.
        preparation: Actual settled initial-state identity and checks.
        execution: Existing demo executor result, absent for an invalid initial
            scene. Program completion alone does not imply acceptance.
        measured_validation: Required measured task checks after execution,
            absent when initial-state validation prevented execution.
        program_completed: Whether the selected canonical bridge completed all
            segment lifecycles normally. Generic Gym success is insufficient.
        program_completion_mask: Owned single-row bridge acceptance snapshot,
            available only after normal program completion.
    """

    variant: SceneVariant
    preparation: ScenePreparationResult
    execution: DemoEpisodeResult | None
    measured_validation: ValidationResult | None
    program_completed: bool = False
    program_completion_mask: tuple[bool, ...] = ()
    env_id: int = field(kw_only=True)

    def __post_init__(self) -> None:
        if type(self.env_id) is not int or self.env_id < 0:
            raise ValueError("env_id must be a nonnegative integer")
        if not isinstance(self.variant, SceneVariant):
            raise TypeError("variant must be a SceneVariant")
        if not isinstance(self.preparation, ScenePreparationResult):
            raise TypeError("preparation must be a ScenePreparationResult")
        if self.execution is not None and not isinstance(
            self.execution, DemoEpisodeResult
        ):
            raise TypeError("execution must be a DemoEpisodeResult or None")
        if self.measured_validation is not None and not isinstance(
            self.measured_validation, ValidationResult
        ):
            raise TypeError("measured_validation must be a ValidationResult or None")
        if self.execution is not None and (
            self.env_id >= len(self.execution.success)
            or len(self.execution.completed_by_env) != len(self.execution.success)
        ):
            raise ValueError("execution must include completion evidence for this row")
        if type(self.program_completed) is not bool:
            raise TypeError("program_completed must be a bool")
        try:
            completion_mask = tuple(self.program_completion_mask)
        except TypeError as exc:
            raise TypeError("program_completion_mask must contain bool values") from exc
        if any(type(value) is not bool for value in completion_mask):
            raise TypeError("program_completion_mask must contain bool values")
        if self.program_completed and len(completion_mask) != 1:
            raise ValueError("program_completion_mask must contain one environment row")
        object.__setattr__(self, "program_completion_mask", completion_mask)

    @property
    def program_success(self) -> bool:
        """Return whether the canonical program and demo execution succeeded.

        Returns:
            Success only with normal bridge completion and its acceptance mask.
        """
        return (
            self.program_completed
            and self.program_completion_mask == (True,)
            and self.execution is not None
            and self.execution.completed_by_env[self.env_id]
            and self.execution.success[self.env_id]
        )

    @property
    def accepted(self) -> bool:
        """Return whether initial checks, execution, and measured checks passed.

        Returns:
            Acceptance for this scene and attempt only; persistence is owned
            by the existing caller and is not implied by this result.
        """
        return (
            self.preparation.validation.accepted
            and self.program_success
            and self.measured_validation is not None
            and self.measured_validation.accepted
        )

    def to_metadata(self) -> dict[str, object]:
        """Export JSON-compatible proposal, initial-state, and outcome evidence.

        Returns:
            Provenance suitable for the existing episode recording path.
        """
        return {
            "schema_version": 2,
            "env_id": self.env_id,
            "variant": _variant_metadata(self.variant),
            "initial_state_id": self.preparation.initial_state_id,
            "preparation": json_safe_copy(
                self.preparation.metadata, field_name="preparation.metadata"
            ),
            "initial_validation": _validation_metadata(self.preparation.validation),
            "program_completed": self.program_completed,
            "program_completion_mask": list(self.program_completion_mask),
            "program_success": self.program_success,
            "measured_validation": _validation_metadata(self.measured_validation),
            "accepted": self.accepted,
        }


def _variant_metadata(variant: SceneVariant) -> dict[str, object]:
    return {
        "variant_id": variant.variant_id,
        "parent_scene_id": variant.parent_scene_id,
        "seed": variant.seed,
        "ordinal": variant.ordinal,
        "pose_changes": [
            {
                "entity_uid": change.entity_uid,
                "arena_pose": [list(row) for row in change.arena_pose],
            }
            for change in variant.pose_changes
        ],
        "metadata": dict(variant.metadata),
    }


def execute_scene_variants(
    env: EmbodiedEnv,
    variants: Mapping[int, SceneVariant],
    *,
    prepare_scene: Callable[
        [Mapping[int, SceneVariant]], Mapping[int, ScenePreparationResult]
    ],
    measured_validator: Callable[[int], ValidationResult],
    task_program: TaskProgramCfg | CompiledTaskProgram | None = None,
    episode_index: int = 0,
    attempt_id: int = 0,
    execution_cfg: DemoExecutionCfg | None = None,
    nominal_source_factory: TaskProgramNominalSourceFactory | None = None,
    expansion_profile: CombinedExpansionProfile | None = None,
    expansion_candidate_index: int = 0,
    plan_transform_factory: TaskProgramPlanTransformFactory | None = None,
    trace_actions: bool = False,
) -> Mapping[int, SceneVariantEpisodeResult]:
    """Prepare selected scenes and execute their valid rows on one World clock.

    The canonical Task Program bridge is created after preparation. Its segments
    use the ordinary batched demo executor with an explicit active-row mask.
    Initial failures prevent execution of those rows; program completion and
    measured acceptance are checked independently for every remaining row.

    This entry performs no persistence. The caller resets to commit accepted
    rows or discard rejected buffers. Checks block saving during preparation,
    execution and measurement; rejection stays attached to its row until reset.
    Physics advances the entire World, including unassigned rows.

    Args:
        env: Initialized EmbodiedEnv or wrapper owning the physical World.
        variants: Nonempty ordered mapping from physical row to scene proposal.
        prepare_scene: Batch callback returning exactly one preparation result
            per requested row, after state writes and any whole-World settling.
        measured_validator: Required post-execution checks for one physical row.
        task_program: Shared config/compiled program, or the configured program.
        episode_index: Logical index used by the existing demo recording path.
        attempt_id: Collection attempt index.
        execution_cfg: Existing batched demo persistence layout settings.
        nominal_source_factory: Optional episode-scoped external motion binding.
        expansion_profile: Optional episode-owned combined motion profile.
        expansion_candidate_index: Shared combined recipe index.
        plan_transform_factory: Optional trusted plan transform; mutually
            exclusive with an expansion profile.
        trace_actions: Enable source tracing, qualified only for a B=1 World.

    Returns:
        Immutable row mapping with separate initial, program and measured
        outcomes. A rejected row does not invalidate a successful peer.

    Raises:
        ValueError: For malformed selections/configuration before preparation.
        TypeError: For invalid callbacks, candidates or result contracts.
        RuntimeError: For another active scene attempt on the same World.
        Exception: Runtime failures reject all requested buffers before propagating.
    """
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv

    target = getattr(env, "unwrapped", env)
    if not isinstance(target, EmbodiedEnv):
        raise TypeError("env must be an EmbodiedEnv or a wrapper around one")
    if getattr(target, "_scene_expansion_active", False):
        raise RuntimeError("A scene expansion attempt already owns this World")
    if (
        not isinstance(variants, Mapping)
        or not variants
        or any(
            type(row) is not int or not 0 <= row < target.num_envs for row in variants
        )
    ):
        raise ValueError("variants must map nonempty valid environment rows")
    variants = MappingProxyType(dict(variants))
    rows = tuple(variants)
    if any(not isinstance(variant, SceneVariant) for variant in variants.values()):
        raise TypeError("variants must contain SceneVariant objects")
    for callback, argument, name in (
        (prepare_scene, variants, "prepare_scene"),
        (measured_validator, rows[0], "measured_validator"),
    ):
        if not callable(callback):
            raise TypeError(f"{name} must be callable")
        try:
            inspect.signature(callback).bind(argument)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} must accept its row contract") from error
    selected_program = task_program
    if selected_program is None:
        selected_program = getattr(target.cfg, "task_program", None)
    if type(selected_program) not in (TaskProgramCfg, CompiledTaskProgram):
        raise TypeError(
            "scene expansion requires a Task Program config or compiled program"
        )
    for name, value in (("episode_index", episode_index), ("attempt_id", attempt_id)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if execution_cfg is not None and not isinstance(execution_cfg, DemoExecutionCfg):
        raise TypeError("execution_cfg must be a DemoExecutionCfg or None")
    if plan_transform_factory is not None and expansion_profile is not None:
        raise ValueError("one episode may have only one expansion owner")
    if expansion_profile is None and expansion_candidate_index != 0:
        raise ValueError("expansion_candidate_index requires an expansion profile")
    if type(trace_actions) is not bool:
        raise TypeError("trace_actions must be a boolean")
    if trace_actions and target.num_envs != 1:
        raise ValueError("source execution tracing requires a B=1 World")
    previous_no_auto_reset = bool(getattr(target, "_demo_no_auto_reset", False))
    pending = {
        row: {
            "schema_version": 2,
            "env_id": row,
            "variant": _variant_metadata(variant),
            "accepted": False,
            "status": "preparing",
        }
        for row, variant in variants.items()
    }
    with target.episode_context.check("scene_expansion/task_program", rows) as finish:
        target._scene_expansion_active = True
        target._demo_no_auto_reset = True
        try:
            preparation = prepare_scene(variants)
            if (
                not isinstance(preparation, Mapping)
                or any(type(row) is not int for row in preparation)
                or set(preparation) != set(rows)
                or any(
                    not isinstance(value, ScenePreparationResult)
                    for value in preparation.values()
                )
            ):
                raise TypeError(
                    "prepare_scene must return exactly the requested row results"
                )
            preparation = dict(preparation)
            results = {
                row: SceneVariantEpisodeResult(
                    variants[row], preparation[row], None, None, env_id=row
                )
                for row in rows
                if not preparation[row].validation.accepted
            }
            active_rows = tuple(row for row in rows if row not in results)
            if active_rows:
                for row in active_rows:
                    pending[row].update(
                        initial_state_id=preparation[row].initial_state_id,
                        status="executing",
                    )
                plan_options = {}
                if nominal_source_factory is not None:
                    plan_options["nominal_source_factory"] = nominal_source_factory
                if plan_transform_factory is not None:
                    plan_options["plan_transform_factory"] = plan_transform_factory
                if expansion_profile is not None:
                    plan_options["expansion_profile"] = expansion_profile
                    plan_options["expansion_candidate_index"] = (
                        expansion_candidate_index
                    )
                segments = EmbodiedEnv.create_demo_segments(
                    target, task_program=selected_program, **plan_options
                )
                bridge = target._active_task_program_bridge
                if bridge is None:
                    raise RuntimeError("canonical Task Program bridge was not created")
                if trace_actions:
                    bridge.enable_execution_trace()
                execution = execute_demo_episode(
                    env,
                    episode_index=episode_index,
                    attempt_id=attempt_id,
                    execution_cfg=execution_cfg,
                    segments=() if segments is None else segments,
                    task_program=selected_program,
                    active_env_ids=active_rows,
                )
                completed = bridge.program_completed
                if type(completed) is not bool:
                    raise TypeError(
                        "Task Program bridge program_completed must be a bool"
                    )
                completion = ()
                if completed:
                    mask = bridge.completion_mask
                    if not isinstance(mask, torch.Tensor):
                        raise TypeError("Task Program completion mask must be a tensor")
                    if mask.dtype != torch.bool or mask.shape != (target.num_envs,):
                        raise ValueError(
                            "Task Program completion mask must match the World rows"
                        )
                    completion = tuple(mask.detach().cpu().tolist())
                for row in active_rows:
                    measured = measured_validator(row)
                    if not isinstance(measured, ValidationResult):
                        raise TypeError(
                            "measured_validator must return a ValidationResult"
                        )
                    results[row] = SceneVariantEpisodeResult(
                        variants[row],
                        preparation[row],
                        execution,
                        measured,
                        program_completed=completed,
                        program_completion_mask=(completion[row],) if completed else (),
                        env_id=row,
                    )
            ordered = {row: results[row] for row in rows}
            finish(
                accepted=[ordered[row].accepted for row in rows],
                metadata={row: ordered[row].to_metadata() for row in rows},
            )
            return MappingProxyType(ordered)
        except BaseException as error:
            for metadata in pending.values():
                metadata.update(status="error", error_type=type(error).__name__)
            finish(accepted=False, metadata=pending)
            raise
        finally:
            target._demo_no_auto_reset = previous_no_auto_reset
            target._scene_expansion_active = False
