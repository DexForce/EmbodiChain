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

"""Source-independent inputs and Skill semantics for joint-plan assembly."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import torch

from .effects import StateDelta
from .plans import (
    EffectVerificationRequirement,
    PlannerDiagnostics,
    TimedTrajectory,
    TrajectorySegment,
    normalize_success_mask,
)

__all__ = [
    "SkillTrajectoryGeometry",
    "SkillTrajectoryContract",
    "SkillTrajectoryInput",
    "SkillPlanSemantics",
]


def _name(value: str) -> None:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError("Trajectory schema names must be nonempty stripped strings")


def _names(values: tuple[str, ...]) -> tuple[str, ...]:
    result = tuple(values)
    for value in result:
        _name(value)
    if len(set(result)) != len(result):
        raise ValueError("Trajectory schema names must be unique")
    return result


@runtime_checkable
class SkillTrajectoryGeometry(Protocol):
    """Skill-owned geometric facts with an independently owned snapshot."""

    def snapshot(self) -> SkillTrajectoryGeometry:
        """Return an independently owned geometric-facts value.

        Returns:
            A distinct value of the same type with owned tensor storage.
        """


@dataclass(frozen=True, slots=True)
class SkillTrajectoryContract:
    """Versioned, declarative opt-in to trajectory-backed plan assembly.

    Args:
        schema_id: Stable Skill-specific trajectory schema identity.
        revision: Exact schema version required by an input.
        segment_names: Ordered semantic stages; empty uses the framework's
            default Skill-named motion stage.
        landmark_names: Exact required event IDs on the execution grid.
        landmark_stages: Optional event-to-stage-end constraints. ``None``
            refers to the terminal sample of the full path.
        geometry_type: Optional concrete geometric-facts type accepted by the Skill.
            Omitting geometry asks the Skill to derive facts from the trajectory
            and current context; it never authorizes arbitrary metadata.
        supports_batch: Whether rows sharing one stage layout are supported.

    This contract does not grant expansion, source admission or recovery
    capabilities. Changing execution grids requires an explicitly mapped new
    input; the existing same-grid rebuild API retains its timing restriction.
    """

    schema_id: str
    revision: str
    segment_names: tuple[str, ...] = ()
    landmark_names: tuple[str, ...] = ()
    landmark_stages: Mapping[str, str | None] = field(default_factory=dict)
    geometry_type: type[SkillTrajectoryGeometry] | None = None
    supports_batch: bool = True

    def __post_init__(self) -> None:
        _name(self.schema_id)
        _name(self.revision)
        object.__setattr__(self, "segment_names", _names(self.segment_names))
        object.__setattr__(self, "landmark_names", _names(self.landmark_names))
        stages = dict(self.landmark_stages)
        if set(stages) - set(self.landmark_names) or any(
            value is not None and value not in self.segment_names
            for value in stages.values()
        ):
            raise ValueError(
                "Landmark stage constraints must name declared events and stages"
            )
        object.__setattr__(self, "landmark_stages", MappingProxyType(stages))
        if self.geometry_type is not None and (
            not isinstance(self.geometry_type, type)
            or not issubclass(self.geometry_type, SkillTrajectoryGeometry)
        ):
            raise TypeError("geometry_type must implement SkillTrajectoryGeometry")
        if type(self.supports_batch) is not bool:
            raise TypeError("supports_batch must be a bool")

    def validate_input(self, motion: SkillTrajectoryInput) -> None:
        """Check an owned input against this exact semantic schema.

        Args:
            motion: Trajectory and semantic annotations to admit.
        """
        if type(motion) is not SkillTrajectoryInput:
            raise TypeError("motion must be exactly SkillTrajectoryInput")
        if (motion.schema_id, motion.schema_revision) != (
            self.schema_id,
            self.revision,
        ):
            raise ValueError(
                "Trajectory schema identity or revision does not match Skill"
            )
        if tuple(s.name for s in motion.segments) != self.segment_names:
            raise ValueError("Trajectory semantic stages do not match Skill contract")
        if set(motion.landmarks) != set(self.landmark_names):
            raise ValueError("Trajectory event landmarks do not match Skill contract")
        if any(
            motion.landmarks[name] != index
            for name, index in self.stage_landmarks(
                motion.segments, motion.trajectory.waypoint_count
            ).items()
        ):
            raise ValueError(
                "Trajectory event landmark violates its declared stage boundary"
            )
        if (
            motion.geometry is not None
            and type(motion.geometry) is not self.geometry_type
        ):
            raise TypeError("Trajectory geometric facts do not match Skill contract")
        if motion.trajectory.batch_size > 1 and not self.supports_batch:
            raise ValueError("Skill trajectory contract requires B=1")

    def stage_landmarks(
        self,
        segments: tuple[TrajectorySegment, ...],
        waypoint_count: int,
    ) -> Mapping[str, int]:
        """Resolve declared stage-end events on a supplied execution grid.

        Args:
            segments: Semantic ranges in this version of the trajectory.
            waypoint_count: Number of valid trajectory samples.

        Returns:
            Bound event indices; interior events still require explicit annotation.
        """
        if type(waypoint_count) is not int or waypoint_count < 1:
            raise ValueError("Event resolution requires a positive sample count")
        by_name = {segment.name: segment for segment in segments}
        if any(
            stage is not None and stage not in by_name
            for stage in self.landmark_stages.values()
        ):
            raise ValueError("Event resolution is missing a declared semantic stage")
        return MappingProxyType(
            {
                name: waypoint_count - 1 if stage is None else by_name[stage].stop - 1
                for name, stage in self.landmark_stages.items()
            }
        )


@dataclass(frozen=True, slots=True, eq=False)
class SkillTrajectoryInput:
    """Owned, source-neutral motion supplied to a Skill's semantic assembler.

    Args:
        schema_id: Selected Skill trajectory contract identity.
        schema_revision: Exact selected contract version.
        trajectory: Full-joint motion on the host's execution clock.
        segments: Contiguous half-open semantic ranges covering the path, or
            empty for the framework's default Skill-named motion stage.
        landmarks: Stable event IDs mapped to actual samples, never exclusive
            end boundaries. A stage's last sample is ``stop - 1``.
        planning_success: Per-row planning eligibility, not physical success.
        geometry: Optional typed Skill facts; data decoding belongs to the source.
        diagnostics: Source/planner provenance, without executable callbacks.
        replannable: Whether the supplying planning path can regenerate this
            input. Registered source admission still constrains recovery policy.

    Semantic stages do not grant motion-expansion permissions. Existing
    ``TrajectoryPhase`` values remain owned by the augmentation boundary.
    """

    schema_id: str
    schema_revision: str
    trajectory: TimedTrajectory
    segments: tuple[TrajectorySegment, ...] = ()
    landmarks: Mapping[str, int] = field(default_factory=dict)
    planning_success: bool | torch.Tensor = True
    geometry: SkillTrajectoryGeometry | None = None
    diagnostics: PlannerDiagnostics | None = None
    replannable: bool = False

    def __post_init__(self) -> None:
        _name(self.schema_id)
        _name(self.schema_revision)
        if not isinstance(self.trajectory, TimedTrajectory):
            raise TypeError("trajectory must be a TimedTrajectory")
        trajectory = self.trajectory.snapshot()
        if not trajectory.waypoint_count:
            raise ValueError("Trajectory assembly requires at least one sample")
        segments = tuple(self.segments)
        offset = 0
        for segment in segments:
            if type(segment) is not TrajectorySegment:
                raise TypeError("segments must contain TrajectorySegment values")
            if segment.start != offset:
                raise ValueError("Semantic stages must be contiguous and start at zero")
            offset = segment.stop
        _names(tuple(s.name for s in segments))
        if segments and offset != trajectory.waypoint_count:
            raise ValueError("Semantic stages must cover the complete trajectory")
        landmarks = dict(self.landmarks)
        for name, index in landmarks.items():
            _name(name)
            if type(index) is not int or not 0 <= index < trajectory.waypoint_count:
                raise ValueError("Event landmarks must index actual trajectory samples")
        mask = normalize_success_mask(
            self.planning_success,
            num_envs=trajectory.batch_size,
            device=trajectory.positions.device,
            name="Trajectory planning success",
        )
        geometry = self.geometry
        if geometry is not None:
            if not isinstance(geometry, SkillTrajectoryGeometry):
                raise TypeError("geometry must implement SkillTrajectoryGeometry")
            copied = geometry.snapshot()
            if type(copied) is not type(geometry) or copied is geometry:
                raise TypeError(
                    "Geometric snapshots must be distinct values of the same type"
                )
            geometry = copied
        diagnostics = self.diagnostics
        if diagnostics is not None:
            if type(diagnostics) is not PlannerDiagnostics:
                raise TypeError("diagnostics must be PlannerDiagnostics or None")
            diagnostics = PlannerDiagnostics(
                diagnostics.backend,
                diagnostics.messages,
                diagnostics.metadata,
                diagnostics.failure,
            )
        if type(self.replannable) is not bool:
            raise TypeError("replannable must be a bool")
        object.__setattr__(self, "trajectory", trajectory)
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "landmarks", MappingProxyType(landmarks))
        object.__setattr__(self, "planning_success", mask.clone())
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "diagnostics", diagnostics)

    def snapshot(self) -> SkillTrajectoryInput:
        """Return an independently owned input for a planning boundary.

        Returns:
            Revalidated trajectory, annotations and geometric facts.
        """
        return SkillTrajectoryInput(
            schema_id=self.schema_id,
            schema_revision=self.schema_revision,
            trajectory=self.trajectory,
            segments=self.segments,
            landmarks=self.landmarks,
            planning_success=self.planning_success,
            geometry=self.geometry,
            diagnostics=self.diagnostics,
            replannable=self.replannable,
        )


@dataclass(frozen=True, slots=True, eq=False)
class SkillPlanSemantics:
    """Uncommitted Skill effects and scene-monitor boundaries for one motion.

    Args:
        expected_effects: Terminal symbolic effects handled by the existing runtime.
        effect_candidates: Intermediate attachment baselines for gates and guards.
        effect_verification: Optional explicit physical-effect requirement.
        scene_dependency_monitor_until: Exclusive execution-index monitor bounds,
            computed from this input's current events and stages.
        scene_dependency_end_segment: Optional named last monitored stage.

    Constructing this value never changes task state or establishes measured
    task success. Effect assurance remains owned by the existing invocation.
    """

    expected_effects: StateDelta = field(default_factory=StateDelta)
    effect_candidates: StateDelta = field(default_factory=StateDelta)
    effect_verification: EffectVerificationRequirement | None = None
    scene_dependency_monitor_until: Mapping[str, int] = field(default_factory=dict)
    scene_dependency_end_segment: str | None = None

    def __post_init__(self) -> None:
        for name in ("expected_effects", "effect_candidates"):
            value = getattr(self, name)
            if type(value) is not StateDelta:
                raise TypeError(f"{name} must be a StateDelta")
            object.__setattr__(self, name, value.snapshot())
        if self.effect_verification is not None:
            if type(self.effect_verification) is not EffectVerificationRequirement:
                raise TypeError(
                    "effect_verification must be EffectVerificationRequirement"
                )
            object.__setattr__(
                self, "effect_verification", self.effect_verification.snapshot()
            )
        bounds = dict(self.scene_dependency_monitor_until)
        for entity, bound in bounds.items():
            _name(entity)
            if type(bound) is not int or bound < 0:
                raise ValueError(
                    "Scene monitor bounds must be nonnegative integer indices"
                )
        if self.scene_dependency_end_segment is not None:
            _name(self.scene_dependency_end_segment)
        object.__setattr__(
            self, "scene_dependency_monitor_until", MappingProxyType(bounds)
        )
