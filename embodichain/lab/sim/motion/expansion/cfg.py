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

"""Strict configuration for the first fixed-scene generation implementation.

These are pure job contracts, not a runnable host integration. A runner must
still resolve registered sources, restoration profiles, physical validators,
and a persistence sink before execution. Capability checks use explicit trusted
registries; configuration strings are never imported or evaluated. The initial
subset rejects modes that require unimplemented scheduling or retry policies.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import fields
import math
from typing import Any, get_args, get_origin, get_type_hints

from embodichain.utils import configclass

from .operators import TIMING_PROFILES

__all__ = [
    "SPATIAL_METHODS",
    "TrajectoryAugmentationCfg",
    "TrajectoryGenerationJobCfg",
]

SPATIAL_METHODS = ("joint_residual", "via_points")
"""Joint-path methods the spatial factor can enumerate."""

_MAX_VIA_POINTS = 8
"""Allocation bound on interior knots per augmented phase."""

_SPATIAL_JACOBIAN_ROWS = 6
"""Row count of a spatial Jacobian: linear velocity first, then angular."""


def _positive(value: float, name: str, *, allow_zero: bool = False) -> None:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or (value < 0 if allow_zero else value <= 0)
    ):
        raise ValueError(
            f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}"
        )


def _count(value: int, name: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _boolean(value: bool, name: str) -> None:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")


def _id(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty registered ID")


def _fixed(value: str, supported: str, name: str) -> None:
    if value != supported:
        raise ValueError(f"{name} currently supports only {supported!r}")


def _decode(cls: type, data: Mapping[str, Any], path: str = "") -> Any:
    if not isinstance(data, Mapping):
        raise ValueError(f"{path or cls.__name__} must be a mapping")
    hints = get_type_hints(cls)
    allowed = {field.name for field in fields(cls)}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(
            f"unknown fields in {path or cls.__name__}: {sorted(unknown, key=str)}"
        )
    decoded = {}
    for key, value in data.items():
        expected = hints[key]
        name = f"{path}.{key}" if path else key
        if hasattr(expected, "__dataclass_fields__"):
            decoded[key] = _decode(expected, value, name)
        elif get_origin(expected) is tuple:
            member = get_args(expected)[0]
            # Accept one value where a sequence of names is expected, so a
            # single-choice spelling keeps working after a field grows.
            if member is str and isinstance(value, str):
                value = (value,)
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"{name} must be a sequence")
            if any(
                type(item) not in ((float, int) if member is float else (member,))
                for item in value
            ):
                raise ValueError(f"{name} contains an invalid value type")
            decoded[key] = tuple(value)
        elif expected is float and type(value) in (float, int):
            decoded[key] = float(value)
        elif type(value) is expected:
            decoded[key] = value
        else:
            raise ValueError(f"{name} has an invalid type")
    return cls(**decoded)


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
    start_state: _ProvidedStartCfg = _ProvidedStartCfg()
    factors: _FactorsCfg = _FactorsCfg()
    coverage: _CoverageCfg = _CoverageCfg()

    def __post_init__(self) -> None:
        _count(self.seed, "seed", 0)
        if self.seed >= 2**63:
            raise ValueError("seed must be less than 2**63")
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


@configclass
class _SourceCfg:
    kind: str = "handwritten"
    source_id: str = "handwritten_qpos"
    template_id: str = "reference_0"

    def __post_init__(self) -> None:
        if self.kind not in ("handwritten", "atomic"):
            raise ValueError("source.kind must be handwritten or atomic")
        _id(self.source_id, "source_id")
        _id(self.template_id, "template_id")


@configclass
class _PlanningCfg:
    batch_mode: str = "env_rows"

    def __post_init__(self) -> None:
        _fixed(self.batch_mode, "env_rows", "planning.batch_mode")


@configclass
class _ExecutionCfg:
    pool_mode: str = "per_env_case"
    scheduler: str = "full_batch"
    ready_low_watermark: int = 1
    ready_high_watermark: int = 16
    ready_max_bytes: int = 268435456
    overlap_planning_and_physics: bool = False

    def __post_init__(self) -> None:
        _fixed(self.pool_mode, "per_env_case", "execution.pool_mode")
        _fixed(self.scheduler, "full_batch", "execution.scheduler")
        _count(self.ready_low_watermark, "ready_low_watermark", 0)
        _count(self.ready_high_watermark, "ready_high_watermark")
        _count(self.ready_max_bytes, "ready_max_bytes")
        if self.ready_low_watermark >= self.ready_high_watermark:
            raise ValueError(
                "ready_low_watermark must be less than ready_high_watermark"
            )
        _boolean(self.overlap_planning_and_physics, "overlap_planning_and_physics")
        if self.overlap_planning_and_physics:
            raise ValueError("overlapping planning and physics is not implemented")


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
class _ValidationCfg:
    validator_id: str = "task_success"
    profile_id: str = "verified_motion"
    motion_limits_profile_id: str = "robot_execution_limits"
    require_path_collision: bool = True
    require_task_success: bool = True
    path_length_ratio_max: float = 1.5
    duration_ratio_max: float = 1.5

    def __post_init__(self) -> None:
        for name in ("validator_id", "profile_id", "motion_limits_profile_id"):
            _id(getattr(self, name), name)
        for name in ("require_path_collision", "require_task_success"):
            _boolean(getattr(self, name), name)
            if not getattr(self, name):
                raise ValueError(f"expert collection requires {name}")
        _positive(self.path_length_ratio_max, "path_length_ratio_max")
        _positive(self.duration_ratio_max, "duration_ratio_max")


@configclass
class _CollectionCfg:
    target_committed_episodes: int = 1
    max_proposals: int = 100
    max_rollout_attempts: int = 100
    max_attempts_per_candidate: int = 1
    max_wall_time_s: float = 60.0

    def __post_init__(self) -> None:
        for name in (
            "target_committed_episodes",
            "max_proposals",
            "max_rollout_attempts",
            "max_attempts_per_candidate",
        ):
            _count(getattr(self, name), name)
        _positive(self.max_wall_time_s, "max_wall_time_s")
        if self.max_attempts_per_candidate != 1:
            raise ValueError(
                "rollout retries are not implemented; max_attempts_per_candidate must be 1"
            )
        if self.target_committed_episodes > min(
            self.max_proposals, self.max_rollout_attempts
        ):
            raise ValueError(
                "collection budgets cannot be smaller than the committed target"
            )


@configclass
class _PersistenceCfg:
    sink: str = "lerobot"
    accepted_only: bool = True
    async_write: bool = False
    pending_episode_limit: int = 8
    pending_max_bytes: int = 536870912
    save_audit: bool = True
    split_unit: str = "trajectory_family"

    def __post_init__(self) -> None:
        _id(self.sink, "sink")
        for name in ("accepted_only", "async_write", "save_audit"):
            _boolean(getattr(self, name), name)
        if not self.accepted_only or not self.save_audit:
            raise ValueError("expert collection requires accepted_only and save_audit")
        if self.async_write:
            raise ValueError("asynchronous persistence is not implemented")
        _count(self.pending_episode_limit, "pending_episode_limit")
        _count(self.pending_max_bytes, "pending_max_bytes")
        _fixed(self.split_unit, "trajectory_family", "split_unit")


def _validate_nested(value: Any, expected: type, name: str) -> None:
    if not isinstance(value, expected):
        raise ValueError(
            f"{name} must use its typed configuration; use from_mapping for dictionaries"
        )
    # Re-decode a plain mapping to reject mutated nested types and unsupported values.
    _decode(expected, value.to_dict(), name)


@configclass
class TrajectoryGenerationJobCfg:
    """Standalone generation job configuration; control periods remain host-owned."""

    source: _SourceCfg = _SourceCfg()
    augmentation: TrajectoryAugmentationCfg = TrajectoryAugmentationCfg()
    planning: _PlanningCfg = _PlanningCfg()
    execution: _ExecutionCfg = _ExecutionCfg()
    reset: _ResetCfg = _ResetCfg()
    validation: _ValidationCfg = _ValidationCfg()
    collection: _CollectionCfg = _CollectionCfg()
    persistence: _PersistenceCfg = _PersistenceCfg()

    def __post_init__(self) -> None:
        self.validate_semantics()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TrajectoryGenerationJobCfg:
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
