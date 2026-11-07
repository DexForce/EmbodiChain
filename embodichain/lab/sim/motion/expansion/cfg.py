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

"""Strict configuration for the first fixed-scene expansion implementation.

These are pure job contracts, not a runnable host integration. A runner must
still resolve registered sources, restoration profiles, physical validators,
and a persistence sink before execution. Capability checks use explicit trusted
registries; configuration strings are never imported or evaluated. The initial
subset rejects modes that require unimplemented scheduling or retry policies.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
import math
from typing import Any, get_type_hints, TYPE_CHECKING

from embodichain.compute.kinematics import DifferentiableIKCfg
from embodichain.utils import configclass

from embodichain.lab.expansion.session_cfg import (
    ExpansionExecutionCfg as _ExecutionCfg,
    ExpansionValidationCfg as _ValidationCfg,
    ExpansionCollectionCfg as _CollectionCfg,
    ExpansionPersistenceCfg as _PersistenceCfg,
    _boolean,
    _count,
    _decode,
    _fixed,
    _id,
    _positive,
    _validate_nested,
)

from .operators import TIMING_PROFILES

if TYPE_CHECKING:
    from embodichain.lab.expansion.session_cfg import ExpansionSessionCfg
    from embodichain.lab.expansion.coverage import ExpansionCoverage

__all__ = [
    "SPATIAL_METHODS",
    "TrajectoryExpansionCfg",
    "TrajectoryAugmentationCfg",
    "TrajectoryExpansionJobCfg",
]

SPATIAL_METHODS = ("joint_residual", "via_points")
"""Joint-path methods the spatial factor can enumerate."""

_MAX_VIA_POINTS = 8
"""Allocation bound on interior knots per augmented phase."""

_SPATIAL_JACOBIAN_ROWS = 6
"""Row count of a spatial Jacobian: linear velocity first, then angular."""


@configclass
class _ProvidedStartCfg:
    mode: str = "provided"

    def __post_init__(self) -> None:
        _fixed(self.mode, "provided", "start_state.mode")


@configclass
class _DisabledFactorCfg:
    enabled: bool = False

    def __post_init__(self) -> None:
        _boolean(self.enabled, "factor.enabled")
        if self.enabled:
            raise ValueError("this augmentation factor is not implemented")


@configclass
class _SpatialCfg:
    enabled: bool = False
    method: tuple[str, ...] = ("joint_residual",)
    joint_offset_scale: float = 0.05
    via_count: int = 1

    def __post_init__(self) -> None:
        _boolean(self.enabled, "spatial.enabled")
        methods = (self.method,) if isinstance(self.method, str) else tuple(self.method)
        if not methods:
            raise ValueError("spatial.method must name at least one method")
        for value in methods:
            if value not in SPATIAL_METHODS:
                raise ValueError(
                    f"spatial.method entries must be in {list(SPATIAL_METHODS)}"
                )
        if len(set(methods)) != len(methods):
            raise ValueError("spatial.method entries must be unique")
        self.method = methods
        _positive(
            self.joint_offset_scale, "spatial.joint_offset_scale", allow_zero=True
        )
        if self.joint_offset_scale > 1:
            raise ValueError(
                "spatial.joint_offset_scale is normalized and must be <= 1"
            )
        _count(self.via_count, "spatial.via_count")
        if self.via_count > _MAX_VIA_POINTS:
            raise ValueError(f"spatial.via_count must be at most {_MAX_VIA_POINTS}")


_SpatialCfg.__tuple_shorthand__ = frozenset({"method"})
"""Sequence fields that also accept one bare name, for backward compatibility."""


@configclass
class _IkCfg:
    enabled: bool = False
    method: str = "nullspace_residual"
    normalized_scale: float = 0.05
    task_rows: tuple[int, ...] = (0, 1, 2, 3, 4)

    def __post_init__(self) -> None:
        _boolean(self.enabled, "ik.enabled")
        _fixed(self.method, "nullspace_residual", "ik.method")
        _positive(self.normalized_scale, "ik.normalized_scale", allow_zero=True)
        if self.normalized_scale > 1:
            raise ValueError("ik.normalized_scale is normalized and must be <= 1")
        rows = tuple(self.task_rows)
        if not rows or len(set(rows)) != len(rows):
            raise ValueError("ik.task_rows must be a nonempty sequence of unique rows")
        for row in rows:
            if type(row) is not int or not 0 <= row < _SPATIAL_JACOBIAN_ROWS:
                raise ValueError(
                    "ik.task_rows must index a spatial Jacobian row in "
                    f"[0, {_SPATIAL_JACOBIAN_ROWS})"
                )
        self.task_rows = rows


@configclass
class _TaskSpaceCfg:
    """Root-frame deformation bounds and fixed-step trajectory IK options."""

    enabled: bool = False
    frame_id: str = "tool"
    translation_axes: tuple[bool, bool, bool] = (True, True, True)
    rotation_axes: tuple[bool, bool, bool] = (True, True, True)
    max_translation_m: float = 0.02
    max_rotation_rad: float = 0.05
    ik: DifferentiableIKCfg = DifferentiableIKCfg()

    def __post_init__(self) -> None:
        _boolean(self.enabled, "task_space.enabled")
        _id(self.frame_id, "task_space.frame_id")
        for name in ("translation_axes", "rotation_axes"):
            axes = tuple(getattr(self, name))
            if len(axes) != 3 or any(type(value) is not bool for value in axes):
                raise ValueError(f"task_space.{name} must contain three booleans")
            setattr(self, name, axes)
        for name in ("max_translation_m", "max_rotation_rad"):
            _positive(getattr(self, name), f"task_space.{name}", allow_zero=True)
        if self.enabled and not (
            any(self.translation_axes)
            and self.max_translation_m > 0
            or any(self.rotation_axes)
            and self.max_rotation_rad > 0
        ):
            raise ValueError(
                "enabled task_space deformation must permit a non-zero change"
            )
        if not isinstance(self.ik, DifferentiableIKCfg):
            raise ValueError("task_space.ik must be a DifferentiableIKCfg")
        _decode(DifferentiableIKCfg, self.ik.to_dict(), "task_space.ik")


@configclass
class _ApproachCfg:
    enabled: bool = False
    cone_half_angle_rad: float = 0.0
    directions: int = 1
    align_tool: bool = True

    def __post_init__(self) -> None:
        _boolean(self.enabled, "approach.enabled")
        _boolean(self.align_tool, "approach.align_tool")
        _positive(
            self.cone_half_angle_rad, "approach.cone_half_angle_rad", allow_zero=True
        )
        if self.cone_half_angle_rad >= math.pi / 2:
            raise ValueError(
                "approach.cone_half_angle_rad must be smaller than a right angle"
            )
        _count(self.directions, "approach.directions")
        if self.enabled and self.cone_half_angle_rad == 0:
            raise ValueError(
                "enabled approach variation requires a positive cone half angle"
            )


@configclass
class _TimingCfg:
    enabled: bool = False
    duration_scales: tuple[float, ...] = (1.0,)
    profiles: tuple[str, ...] = ("uniform",)

    def __post_init__(self) -> None:
        _boolean(self.enabled, "timing.enabled")
        if (
            not isinstance(self.duration_scales, (list, tuple))
            or not self.duration_scales
        ):
            raise ValueError("timing.duration_scales must be a nonempty sequence")
        for value in self.duration_scales:
            _positive(value, "timing.duration_scales")
        if len(set(self.duration_scales)) != len(self.duration_scales):
            raise ValueError("timing.duration_scales must be unique")
        self.duration_scales = tuple(self.duration_scales)
        if not isinstance(self.profiles, (list, tuple)) or not self.profiles:
            raise ValueError("timing.profiles must be a nonempty sequence")
        for value in self.profiles:
            if value not in TIMING_PROFILES:
                raise ValueError(
                    f"timing.profiles must contain only {list(TIMING_PROFILES)}"
                )
        if len(set(self.profiles)) != len(self.profiles):
            raise ValueError("timing.profiles must be unique")
        self.profiles = tuple(self.profiles)
        if not self.enabled and (
            self.duration_scales != (1.0,) or self.profiles != ("uniform",)
        ):
            raise ValueError("disabled timing must retain the reference time law")


@configclass
class _ManipulabilityCfg:
    """Manipulability-guided proposal steering and banded coverage quotas.

    Bands are ratios against the reference manipulability registered for one
    initial state, so one set of edges transfers across robots and across the
    initial states of a case. A per-band quota, shared by scene case, spreads
    accepted episodes over well- and poorly-conditioned postures instead of
    concentrating them near the reference posture.
    """

    enabled: bool = False
    jacobian_rows: str = "all"
    band_edges: tuple[float, ...] = (0.5, 0.9)
    target_per_band: int = 1
    guided_proposals: int = 1

    def __post_init__(self) -> None:
        _boolean(self.enabled, "manipulability.enabled")
        if self.jacobian_rows not in ("all", "translational", "rotational"):
            raise ValueError(
                "manipulability.jacobian_rows must be all, translational, or rotational"
            )
        if not isinstance(self.band_edges, (list, tuple)) or not self.band_edges:
            raise ValueError("manipulability.band_edges must be a nonempty sequence")
        for value in self.band_edges:
            _positive(value, "manipulability.band_edges")
        if any(low >= high for low, high in zip(self.band_edges, self.band_edges[1:])):
            raise ValueError("manipulability.band_edges must increase strictly")
        self.band_edges = tuple(float(value) for value in self.band_edges)
        _count(self.target_per_band, "manipulability.target_per_band")
        _count(self.guided_proposals, "manipulability.guided_proposals")


@configclass
class _FactorsCfg:
    contact: _DisabledFactorCfg = _DisabledFactorCfg()
    ik: _IkCfg = _IkCfg()
    approach: _ApproachCfg = _ApproachCfg()
    spatial: _SpatialCfg = _SpatialCfg()
    task_space: _TaskSpaceCfg = _TaskSpaceCfg()
    timing: _TimingCfg = _TimingCfg()
    manipulability: _ManipulabilityCfg = _ManipulabilityCfg()
    contact_timing: _DisabledFactorCfg = _DisabledFactorCfg()
    recovery: _DisabledFactorCfg = _DisabledFactorCfg()


@configclass
class _CoverageCfg:
    geometry_samples_per_phase: int = 32
    joint_dedup_normalized_tol: float = 0.01
    target_per_cell: int = 1

    def __post_init__(self) -> None:
        _count(self.geometry_samples_per_phase, "geometry_samples_per_phase", 2)
        _positive(self.joint_dedup_normalized_tol, "joint_dedup_normalized_tol")
        if self.joint_dedup_normalized_tol > 1:
            raise ValueError(
                "joint_dedup_normalized_tol is normalized and must be <= 1"
            )
        _count(self.target_per_cell, "target_per_cell")


@configclass
class TrajectoryAugmentationCfg:
    """Local random seed, explicitly enabled factors, and geometry coverage limits."""

    seed: int = 0
    max_variants_per_reference: int = 1
    start_state: _ProvidedStartCfg = _ProvidedStartCfg()
    factors: _FactorsCfg = _FactorsCfg()
    coverage: _CoverageCfg = _CoverageCfg()

    def __post_init__(self) -> None:
        _count(self.seed, "seed", 0)
        if self.seed >= 2**63:
            raise ValueError("seed must be less than 2**63")
        _count(
            self.max_variants_per_reference,
            "max_variants_per_reference",
        )
        self.validate_semantics()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TrajectoryAugmentationCfg:
        """Decode closed nested fields without executing configuration values.

        Args:
            data: Nested augmentation settings using the declared configuration fields.

        Returns:
            A decoded configuration with semantic constraints validated.
        """
        return _decode(cls, data)

    def validate_semantics(self) -> None:
        """Recheck typed nested values, including mutations made after decoding."""
        for name, expected in (
            ("start_state", _ProvidedStartCfg),
            ("factors", _FactorsCfg),
            ("coverage", _CoverageCfg),
        ):
            _validate_nested(getattr(self, name), expected, name)
        _count(self.seed, "seed", 0)
        if self.seed >= 2**63:
            raise ValueError("seed must be less than 2**63")
        _count(
            self.max_variants_per_reference,
            "max_variants_per_reference",
        )
        spatial = self.factors.spatial
        guided_residual = (
            self.factors.manipulability.enabled
            and self.factors.manipulability.guided_proposals > 1
            and not self.factors.task_space.enabled
            and not self.factors.ik.enabled
        )
        # A spatial offset at or below the deduplication tolerance measures as a
        # duplicate of the reference, so the reference survives alone and the
        # enabled factor silently contributes nothing. Only a disabled factor
        # may hold a scale that low.
        if (
            spatial.enabled or guided_residual
        ) and spatial.joint_offset_scale <= self.coverage.joint_dedup_normalized_tol:
            raise ValueError(
                "factors.spatial.joint_offset_scale must exceed "
                "coverage.joint_dedup_normalized_tol; a combined expansion "
                "profile sets that tolerance through trajectory.spatial."
                "dedup_tolerance. At or below the tolerance every spatial "
                "proposal measures as a duplicate of the reference trajectory, so "
                "only the nominal variant survives"
            )


@configclass
class _SourceCfg:
    kind: str = "handwritten"
    source_id: str = "handwritten_qpos"
    source_revision: str = "unversioned"
    unit_scope: str = "action"
    template_id: str = "reference_0"
    phase_permissions: dict[str, tuple[str, ...]] = {}
    phase_kinds: dict[str, str] = {}

    def __post_init__(self) -> None:
        if self.kind not in (
            "handwritten",
            "motion_generator",
            "atomic_action",
            "task_program",
        ):
            raise ValueError(
                "source.kind must be handwritten, motion_generator, "
                "atomic_action, or task_program"
            )
        for name in ("source_id", "source_revision", "template_id"):
            _id(getattr(self, name), name)
        _fixed(self.unit_scope, "action", "source.unit_scope")
        permissions = {
            phase_id: tuple(operators)
            for phase_id, operators in self.phase_permissions.items()
        }
        kinds = dict(self.phase_kinds)
        if set(permissions) != set(kinds):
            raise ValueError(
                "source.phase_permissions and source.phase_kinds must name "
                "the same phases"
            )
        for phase_id, operators in permissions.items():
            _id(phase_id, "source phase")
            if len(set(operators)) != len(operators):
                raise ValueError(f"source.phase_permissions.{phase_id} must be unique")
            if any(operator not in SPATIAL_METHODS for operator in operators):
                raise ValueError(
                    f"source.phase_permissions.{phase_id} contains an unsupported operator"
                )
        for phase_id, kind in kinds.items():
            _id(phase_id, "source phase")
            if kind not in ("free", "contact", "hold"):
                raise ValueError(
                    f"source.phase_kinds.{phase_id} must be free, contact, or hold"
                )
        self.phase_permissions = permissions
        self.phase_kinds = kinds


@configclass
class _AffordanceCfg:
    """Source-neutral Affordance proposal policy."""

    enabled: bool = False
    branches_per_family: int = 1
    max_proposals: int = 128

    def __post_init__(self) -> None:
        _boolean(self.enabled, "affordance.enabled")
        _count(self.branches_per_family, "affordance.branches_per_family")
        _count(self.max_proposals, "affordance.max_proposals")


@configclass
class _ObservationCfg:
    """Post-rollout observation profile declarations without fan-out runtime."""

    enabled: bool = False
    profiles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _boolean(self.enabled, "observation.enabled")
        for profile in self.profiles:
            _id(profile, "observation profile")
        if len(set(self.profiles)) != len(self.profiles):
            raise ValueError("observation.profiles must be unique")
        self.profiles = tuple(self.profiles)
        if self.enabled and not self.profiles:
            raise ValueError("enabled observation requires at least one profile")
        if not self.enabled and self.profiles:
            raise ValueError("disabled observation cannot declare profiles")


@configclass
class _SchedulingCfg:
    """Candidate selection policy and bounded logical expansion budget."""

    policy: str = "fifo"
    candidate_budget: int = 128
    reference_family_budget: int = 1
    exploration_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.policy == "coverage_per_cost":
            raise ValueError("scheduling.policy coverage_per_cost is not implemented")
        if self.policy != "fifo":
            raise ValueError("scheduling.policy currently supports only fifo")
        _count(self.candidate_budget, "scheduling.candidate_budget")
        _count(
            self.reference_family_budget,
            "scheduling.reference_family_budget",
        )
        _positive(
            self.exploration_fraction,
            "scheduling.exploration_fraction",
            allow_zero=True,
        )
        if self.exploration_fraction > 1:
            raise ValueError("scheduling.exploration_fraction must be <= 1")


@configclass
class _PlanningCfg:
    batch_mode: str = "env_rows"

    def __post_init__(self) -> None:
        _fixed(self.batch_mode, "env_rows", "planning.batch_mode")


@configclass
class _ResetCfg:
    outer_mode: str = "provided"
    inner_mode: str = "restore_initial"
    prepare_profile_id: str = "fixed_scene_initial_state"
    initial_state_tolerances_profile_id: str = "fixed_scene_tolerances"
    on_initial_state_mismatch: str = "error"

    def __post_init__(self) -> None:
        _fixed(self.outer_mode, "provided", "reset.outer_mode")
        _fixed(self.inner_mode, "restore_initial", "reset.inner_mode")
        _fixed(self.on_initial_state_mismatch, "error", "on_initial_state_mismatch")
        _id(self.prepare_profile_id, "prepare_profile_id")
        _id(
            self.initial_state_tolerances_profile_id,
            "initial_state_tolerances_profile_id",
        )


@configclass
class TrajectoryExpansionCfg:
    """Source-neutral expansion settings shared by every expert source.

    This contract deliberately excludes source identity, Task Program
    selection, environment ownership, and reset policy. Handwritten,
    MotionGenerator, Atomic Action, and Task Program adapters can all consume
    the same expansion settings.
    """

    augmentation: TrajectoryAugmentationCfg = TrajectoryAugmentationCfg()
    affordance: _AffordanceCfg = _AffordanceCfg()
    observation: _ObservationCfg = _ObservationCfg()
    scheduling: _SchedulingCfg = _SchedulingCfg()
    execution: _ExecutionCfg = _ExecutionCfg()
    validation: _ValidationCfg = _ValidationCfg()
    collection: _CollectionCfg = _CollectionCfg()
    persistence: _PersistenceCfg = _PersistenceCfg()

    def to_session_cfg(self) -> ExpansionSessionCfg:
        """Project legacy settings into the canonical source-independent session config."""
        from embodichain.lab.expansion.session_cfg import ExpansionSessionCfg

        self.__post_init__()
        values = self.to_dict()
        return ExpansionSessionCfg.from_mapping(
            {
                "seed": self.augmentation.seed,
                **{
                    name: values[name]
                    for name in ("execution", "validation", "collection", "persistence")
                },
            }
        )

    def create_session_coverage(self) -> ExpansionCoverage:
        """Create the trajectory-specific measured coverage adapter for this session."""
        from .session_coverage import TrajectorySessionCoverage

        return TrajectorySessionCoverage(self.augmentation)

    def __post_init__(self) -> None:
        for name, expected in get_type_hints(type(self)).items():
            _validate_nested(getattr(self, name), expected, name)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TrajectoryExpansionCfg":
        """Decode source-neutral expansion settings with closed fields."""
        return _decode(cls, data)

    @classmethod
    def from_expansion_job(
        cls,
        job: "TrajectoryExpansionJobCfg",
    ) -> "TrajectoryExpansionCfg":
        """Project a legacy job into the source-neutral expansion contract."""
        if not isinstance(job, TrajectoryExpansionJobCfg):
            raise TypeError("job must be a TrajectoryExpansionJobCfg")
        payload = job.to_dict()
        fields_to_keep = (
            "augmentation",
            "affordance",
            "observation",
            "scheduling",
            "execution",
            "validation",
            "collection",
            "persistence",
        )
        return cls.from_mapping({name: payload[name] for name in fields_to_keep})


@configclass
class TrajectoryExpansionJobCfg:
    """Standalone expansion job configuration; control periods remain host-owned."""

    source: _SourceCfg = _SourceCfg()
    augmentation: TrajectoryAugmentationCfg = TrajectoryAugmentationCfg()
    affordance: _AffordanceCfg = _AffordanceCfg()
    observation: _ObservationCfg = _ObservationCfg()
    scheduling: _SchedulingCfg = _SchedulingCfg()
    planning: _PlanningCfg = _PlanningCfg()
    execution: _ExecutionCfg = _ExecutionCfg()
    reset: _ResetCfg = _ResetCfg()
    validation: _ValidationCfg = _ValidationCfg()
    collection: _CollectionCfg = _CollectionCfg()
    persistence: _PersistenceCfg = _PersistenceCfg()

    def to_session_cfg(self) -> ExpansionSessionCfg:
        """Project legacy settings into the canonical source-independent session config."""
        from embodichain.lab.expansion.session_cfg import ExpansionSessionCfg

        self.__post_init__()
        values = self.to_dict()
        return ExpansionSessionCfg.from_mapping(
            {
                "seed": self.augmentation.seed,
                **{
                    name: values[name]
                    for name in ("execution", "validation", "collection", "persistence")
                },
            }
        )

    def create_session_coverage(self) -> ExpansionCoverage:
        """Create the trajectory-specific measured coverage adapter for this session."""
        from .session_coverage import TrajectorySessionCoverage

        return TrajectorySessionCoverage(self.augmentation)

    def __post_init__(self) -> None:
        self.validate_semantics()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TrajectoryExpansionJobCfg:
        """Decode the supported nested YAML schema and reject unknown fields.

        Args:
            data: Nested job settings for source, augmentation, execution, and collection.

        Returns:
            A decoded job configuration with semantic constraints validated.
        """
        return _decode(cls, data)

    def validate_semantics(self) -> None:
        """Validate configuration without loading hosts, profiles, or sources."""
        for name, expected in get_type_hints(type(self)).items():
            _validate_nested(getattr(self, name), expected, name)
        if (
            self.scheduling.candidate_budget
            < self.augmentation.max_variants_per_reference
        ):
            raise ValueError(
                "scheduling.candidate_budget cannot be smaller than "
                "augmentation.max_variants_per_reference"
            )

    def validate_capabilities(
        self,
        *,
        source_ids: Collection[str],
        validator_ids: Collection[str],
        profile_ids: Collection[str],
        sink_ids: Collection[str],
        operators: Collection[str] = (),
    ) -> None:
        """Require IDs and enabled operators in explicitly supplied registries.

        This is a preflight step in addition to semantic decoding. Registry
        membership asserts that the caller has already resolved trusted
        implementations; it does not itself certify physical validation.

        Args:
            source_ids: Available trusted trajectory source identifiers.
            validator_ids: Available trusted validator identifiers.
            profile_ids: Available preparation, state, validation, and motion-limit profiles.
            sink_ids: Available episode sink identifiers.
            operators: Implemented spatial and timing operator names.
        """
        self.validate_semantics()
        references = (
            (self.source.source_id, source_ids, "source"),
            (self.validation.validator_id, validator_ids, "validator"),
            (self.persistence.sink, sink_ids, "sink"),
            (self.reset.prepare_profile_id, profile_ids, "prepare profile"),
            (
                self.reset.initial_state_tolerances_profile_id,
                profile_ids,
                "initial-state profile",
            ),
            (self.validation.profile_id, profile_ids, "validation profile"),
            (
                self.validation.motion_limits_profile_id,
                profile_ids,
                "motion-limits profile",
            ),
        )
        for value, registry, name in references:
            if value not in registry:
                raise ValueError(f"unregistered {name}: {value!r}")
        factors = self.augmentation.factors
        spatial = factors.spatial
        if spatial.enabled:
            for method in spatial.method:
                if method not in operators:
                    raise ValueError(
                        f"spatial operator capability unavailable: {method!r}"
                    )
        if factors.ik.enabled and factors.ik.method not in operators:
            raise ValueError(
                f"ik operator capability unavailable: {factors.ik.method!r}"
            )
        if factors.task_space.enabled and "task_space_residual" not in operators:
            raise ValueError("task-space operator capability unavailable")
        if factors.approach.enabled and "perturb_approach_direction" not in operators:
            raise ValueError("approach operator capability unavailable")
        if factors.timing.enabled and "retime" not in operators:
            raise ValueError("retime operator capability unavailable")
        manipulability = self.augmentation.factors.manipulability
        if (
            manipulability.enabled
            and manipulability.guided_proposals > 1
            and "manipulability_guided_residual" not in operators
        ):
            raise ValueError("manipulability guided residual capability unavailable")
