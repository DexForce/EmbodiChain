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
from types import MappingProxyType
from typing import TYPE_CHECKING

from embodichain.lab.gym.envs._json import json_safe_copy
from embodichain.lab.gym.envs.demo import (
    DemoEpisodeResult,
    DemoExecutionCfg,
    execute_demo_episode,
)
from embodichain.lab.sim.motion.expansion import ValidationResult
from embodichain.lab.sim.scene_expansion import SceneVariant
from embodichain.lab.task_program import CompiledTaskProgram, TaskProgramCfg

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv

__all__ = [
    "ScenePreparationResult",
    "SceneVariantEpisodeResult",
    "execute_scene_variant",
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
        variant: Exact proposed scene evaluated by this attempt.
        preparation: Actual settled initial-state identity and checks.
        execution: Existing demo executor result, absent for an invalid initial
            scene. Program completion alone does not imply acceptance.
        measured_validation: Required measured task checks after execution,
            absent when initial-state validation prevented execution.
    """

    variant: SceneVariant
    preparation: ScenePreparationResult
    execution: DemoEpisodeResult | None
    measured_validation: ValidationResult | None

    def __post_init__(self) -> None:
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

    @property
    def accepted(self) -> bool:
        """Return whether initial checks, execution, and measured checks passed.

        Returns:
            Acceptance for this scene and attempt only; persistence is owned
            by the existing caller and is not implied by this result.
        """
        return (
            self.preparation.validation.accepted
            and self.execution is not None
            and self.execution.completed
            and self.execution.all_success
            and self.measured_validation is not None
            and self.measured_validation.accepted
        )

    def to_metadata(self) -> dict[str, object]:
        """Export JSON-compatible proposal, initial-state, and outcome evidence.

        Returns:
            Provenance suitable for the existing episode recording path.
        """
        return {
            "schema_version": 1,
            "variant": _variant_metadata(self.variant),
            "initial_state_id": self.preparation.initial_state_id,
            "preparation": json_safe_copy(
                self.preparation.metadata, field_name="preparation.metadata"
            ),
            "initial_validation": _validation_metadata(self.preparation.validation),
            "program_success": (
                self.execution is not None
                and self.execution.completed
                and self.execution.all_success
            ),
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


def execute_scene_variant(
    env: EmbodiedEnv,
    variant: SceneVariant,
    *,
    prepare_scene: Callable[[SceneVariant], ScenePreparationResult],
    measured_validator: Callable[[], ValidationResult],
    task_program: TaskProgramCfg | CompiledTaskProgram | None = None,
    episode_index: int = 0,
    attempt_id: int = 0,
    execution_cfg: DemoExecutionCfg | None = None,
) -> SceneVariantEpisodeResult:
    """Prepare and evaluate one scene through the ordinary demo execution path.

    The host prepares/restores the candidate, settles physics, validates the
    actual initial state, and refreshes recording observations and bindings.
    Only then does the canonical
    ``EmbodiedEnv.create_demo_segments(task_program=...)`` create the fresh
    runtime bridge. Its segments flow through :func:`execute_demo_episode`,
    including for subclasses with a handwritten segment-planner override.

    This function performs one attempt and no persistence. The caller retains
    the existing transaction boundary: reset to commit an accepted episode or
    reset with ``save_data=False`` to discard it. Failed or unverified scene
    acceptance also blocks a later ordinary reset from saving this recording.

    Args:
        env: Initialized single-row EmbodiedEnv or a wrapper around one.
        variant: Existing-entity scene candidate to evaluate.
        prepare_scene: Narrow host callback that applies/restores the variant,
            settles and refreshes observations, and returns explicit checks.
        measured_validator: Callback reading physical state and required
            execution evidence. It must return ValidationResult, independently
            of Task Program completion; arbitrary truthy values are rejected.
        task_program: Explicit config/compiled program, or the environment's
            configured Task Program. Handwritten action-list execution is not
            accepted by this entry point.
        episode_index: Existing demo episode metadata index.
        attempt_id: Existing demo execution attempt index.
        execution_cfg: Existing demo persistence layout settings.

    Returns:
        Separate initial-state, program, and measured validation results with
        an acceptance decision. Invalid initial checks prevent execution.

    Raises:
        TypeError: For invalid contracts, callbacks, or program types.
        ValueError: For unsupported batching or invalid attempt identifiers.
        Exception: Host, planning, execution, and measurement errors propagate
            after the episode is marked ineligible for persistence.
    """
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv

    target = getattr(env, "unwrapped", env)
    if not isinstance(target, EmbodiedEnv):
        raise TypeError("env must be an EmbodiedEnv or a wrapper around one")
    if target.num_envs != 1:
        raise ValueError("scene expansion currently requires B=1")
    if not isinstance(variant, SceneVariant):
        raise TypeError("variant must be a SceneVariant")
    if not callable(prepare_scene) or not callable(measured_validator):
        raise TypeError("prepare_scene and measured_validator must be callable")
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

    pending: dict[str, object] = {
        "schema_version": 1,
        "variant": _variant_metadata(variant),
        "accepted": False,
        "status": "preparing",
    }

    def publish(metadata: Mapping[str, object]) -> None:
        target._demo_episode_metadata[0]["scene_expansion"] = json_safe_copy(
            metadata, field_name="scene_expansion"
        )

    publish(pending)
    try:
        preparation = prepare_scene(variant)
        if not isinstance(preparation, ScenePreparationResult):
            raise TypeError("prepare_scene must return a ScenePreparationResult")
        if not preparation.validation.accepted:
            result = SceneVariantEpisodeResult(variant, preparation, None, None)
            publish(result.to_metadata())
            return result

        pending["initial_state_id"] = preparation.initial_state_id
        pending["status"] = "executing"
        publish(pending)
        segments = EmbodiedEnv.create_demo_segments(
            target, task_program=selected_program
        )
        execution = execute_demo_episode(
            env,
            episode_index=episode_index,
            attempt_id=attempt_id,
            execution_cfg=execution_cfg,
            segments=() if segments is None else segments,
        )
        measured = measured_validator()
        if not isinstance(measured, ValidationResult):
            raise TypeError("measured_validator must return a ValidationResult")
        result = SceneVariantEpisodeResult(variant, preparation, execution, measured)
        publish(result.to_metadata())
        return result
    except BaseException as error:
        pending["status"] = "error"
        pending["error_type"] = type(error).__name__
        publish(pending)
        raise
