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

"""Configuration loading and hashing for motion-generation benchmark suites."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import yaml

from embodichain.utils import configclass

from .contracts import EvaluationDomain
from .models import AlgorithmRole

__all__ = [
    "AtomicPoseRandomizationCfg",
    "DomainCfg",
    "EmbodimentSpecCfg",
    "BENCHMARK_ROOT",
    "FreeSpaceTrackCfg",
    "PlannerSpecCfg",
    "ProtocolCfg",
    "RobotSpecCfg",
    "SuiteCfg",
    "TrackCfg",
    "resolve_atomic_batch_sizes",
    "resolve_atomic_pose_randomization",
    "load_suite",
    "stable_hash",
    "suite_to_dict",
]

BENCHMARK_ROOT = Path(__file__).resolve().parent
_FREE_SPACE_TRACK_ID = "free-space-common"
_FREE_SPACE_SCENARIO = "free_space"
_SUPPORTED_TARGET_TYPES = {"eef", "joint"}
_SUPPORTED_PATH_SHAPES = {
    "direct",
    "l_turn",
    "s_curve",
    "orientation_only",
    "combined",
}
_SUPPORTED_START_STATE_BINS = {
    "nominal",
    "random_reachable",
    "near_limit",
    "near_singularity",
}


@dataclass(frozen=True)
class AtomicPoseRandomizationCfg:
    """Normalized random-pose settings for an ``atomic_skill`` track.

    The benchmark represents each random pose as one simulator environment
    row.  Consequently ``pose_batch_size`` is both the number of independently
    perturbed poses sent to a batched Atomic Action goal and the default
    simulator batch size when a track does not declare ``batch_sizes``.

    The nested ``randomization`` mapping is intentionally small and limited
    to strict pose changes.  Skill-level legacy keys such as
    ``object_position_jitter_m`` remain supported by the scenario provider;
    this object only normalizes track-level controls shared by all skills.
    """

    enabled: bool = False
    pose_batch_size: int = 1
    mode: str = "translation"
    sampling_pattern: str = "uniform"
    object_translation_jitter_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    object_yaw_jitter_rad: float = 0.0
    object_roll_pitch_jitter_rad: tuple[float, float] = (0.0, 0.0)
    target_translation_jitter_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_yaw_jitter_rad: float = 0.0
    target_roll_pitch_jitter_rad: tuple[float, float] = (0.0, 0.0)
    support_object_translation_jitter_m: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )
    support_object_yaw_jitter_rad: float = 0.0
    support_object_roll_pitch_jitter_rad: tuple[float, float] = (0.0, 0.0)
    task_frame_translation_jitter_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    task_frame_yaw_jitter_rad: float = 0.0
    task_frame_roll_pitch_jitter_rad: tuple[float, float] = (0.0, 0.0)
    seed_stride: int = 1


@configclass
class PlannerSpecCfg:
    """Configuration for one registered planner adapter."""

    id: str = ""
    adapter: str = ""
    role: str = AlgorithmRole.DIAGNOSTIC_BASELINE.value
    enabled: bool = False
    config: dict[str, Any] = {}


@configclass
class ProtocolCfg:
    """Common timing and external-validation protocol."""

    warmup_trials: int = 1
    measured_trials: int = 3
    sample_interval: int = 40
    validation_samples: int = 128
    position_threshold_m: float = 0.01
    rotation_threshold_rad: float = 0.1
    joint_limit_tolerance_rad: float = 1.0e-5
    confidence_level: float = 0.95
    report_schema_version: str = "3"
    seeds: list[int] = []
    joint_velocity_limit_rad_s: float | list[float] | None = None
    """Optional per-joint or scalar speed budget, capped by effective robot limits.

    Planning and validation use the same resolved limit. The explicit budget
    also supplies a usable limit when a backend reports an unavailable asset
    limit; it cannot increase a valid robot limit.
    """
    joint_acceleration_limit_rad_s2: float | list[float] | None = None
    """Per-joint or scalar acceleration limit, or ``None`` to report N/A.

    URDF carries no acceleration field and :class:`Robot` exposes no
    ``qacc_limits``, so this limit has no asset source and must be stated by the
    suite as an evaluation budget.
    """
    joint_jerk_limit_rad_s3: float | list[float] | None = None
    """Per-joint or scalar jerk limit, or ``None`` to report N/A.

    This is a suite-owned evaluation budget with no asset fallback.
    """
    dynamic_limit_tolerance: float = 1.0e-3
    """Relative slack on dynamic-limit utilization before a violation is flagged."""


@configclass
class RobotSpecCfg:
    """Robot provider selected for every track in one suite run."""

    id: str = "franka_panda"
    provider: str = "franka_panda"
    config: dict[str, Any] = {}


@configclass
class EmbodimentSpecCfg:
    """Official component supplying the runtime robot and endpoint bindings."""

    component: str = ""
    resource_id: str | None = None
    overrides: dict[str, Any] = {}
    initial_qpos: list[float] = []
    endpoint_bindings: dict[str, Any] = {}
    runtime_services: dict[str, Any] = {}


@configclass
class FreeSpaceTrackCfg:
    """Case matrix for the ``free-space-common`` track."""

    batch_sizes: list[int] = [1]
    waypoint_counts: list[int] = [1, 3, 5]
    path_shapes: list[str] = ["direct", "l_turn", "s_curve"]
    start_state_bins: list[str] = ["nominal"]
    seeds: list[int] = [11]
    target_types: list[str] = ["eef"]
    """Target modalities; omission preserves the legacy Cartesian population."""
    joint_threshold_rad: float = 0.02
    """Maximum per-joint error when matching ordered joint targets."""


@configclass
class DomainCfg:
    """Explicit provenance for one track's existing case population.

    This declaration labels cases; it does not generate perturbations or
    establish that a population was held out during model development.
    """

    id: str = ""
    version: str = ""
    kind: str = ""
    overrides: dict[str, Any] = {}
    required_capabilities: list[str] = []
    applicability: list[str] = []
    objects: list[str] = []
    scene_layout: str | None = None
    case_generator: str | None = None
    frozen_cases: list[dict[str, Any]] = []
    perturbations: dict[str, Any] = {}
    metadata: dict[str, Any] = {}

    def to_identity(self) -> EvaluationDomain:
        """Validate and snapshot domain provenance for a frozen case.

        Returns:
            Immutable domain identity, independent of planner configuration.
        """
        return EvaluationDomain(
            id=self.id,
            version=self.version,
            kind=self.kind,
            required_capabilities=frozenset(self.required_capabilities),
        )


@configclass
class TrackCfg:
    """One enabled benchmark track and its scenario provider."""

    id: str = ""
    scenario: str = ""
    enabled: bool = True
    config: dict[str, Any] = {}
    domain: DomainCfg | None = None
    domains: list[DomainCfg] = []
    group_id: str | None = None

    def evaluation_domains(self) -> list[DomainCfg]:
        """Return explicit domains while preserving the legacy singular field."""
        if self.domain is not None and self.domains:
            raise ValueError("A track cannot define both domain and domains.")
        if self.domain is not None:
            return [self.domain]
        return list(self.domains)


@configclass
class SuiteCfg:
    """Resolved benchmark suite configuration."""

    schema_version: int = 1
    name: str = "free_space_common"
    suite_version: str = "free_space_common_v2"
    profile: str = "smoke"
    planners: list[PlannerSpecCfg] = []
    robot: RobotSpecCfg = RobotSpecCfg()
    embodiment: EmbodimentSpecCfg | None = None
    protocol: ProtocolCfg = ProtocolCfg()
    tracks: list[TrackCfg] = []
    free_space: FreeSpaceTrackCfg = FreeSpaceTrackCfg()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SuiteCfg":
        """Build and validate a suite from a YAML-compatible mapping."""
        planners = [PlannerSpecCfg(**item) for item in data.get("planners", [])]
        tracks, free_space = _resolve_tracks_and_free_space(data)
        protocol = ProtocolCfg(**data.get("protocol", {}))
        raw_embodiment = data.get("embodiment")
        if raw_embodiment is not None:
            if "robot" in data:
                raise ValueError(
                    "embodiment.component owns robot; remove inline robot."
                )
            if not isinstance(raw_embodiment, Mapping):
                raise TypeError("embodiment must be a mapping.")
            embodiment = EmbodimentSpecCfg(**dict(raw_embodiment))
        else:
            embodiment = None
        if protocol.seeds:
            free_space.seeds = list(protocol.seeds)
            for track in tracks:
                if track.scenario == "atomic_skill" and "seeds" not in track.config:
                    track.config["seeds"] = list(protocol.seeds)
        suite = cls(
            schema_version=int(data.get("schema_version", 1)),
            name=str(data.get("name", "free_space_common")),
            suite_version=str(data.get("suite_version", "free_space_common_v2")),
            profile=str(data.get("profile", "smoke")),
            planners=planners,
            robot=RobotSpecCfg(**data.get("robot", {})),
            embodiment=embodiment,
            protocol=protocol,
            tracks=tracks,
            free_space=free_space,
        )
        suite.validate_benchmark()
        return suite

    def enabled_tracks(self) -> list[TrackCfg]:
        """Return enabled tracks in suite order."""
        return [track for track in self.tracks if track.enabled]

    @property
    def resolved_embodiment_id(self) -> str:
        """Return the stable embodiment identity used in manifests."""
        if self.embodiment is not None and self.embodiment.component:
            return self.embodiment.component
        return self.robot.id

    def evaluation_tracks(self) -> list[TrackCfg]:
        """Expand multi-domain tracks into deterministic evaluation tracks.

        Domain expansion happens at the runner boundary.  Existing scenario
        providers therefore keep their stable ``TrackCfg`` input while each
        domain receives a distinct case population and track identity.
        """
        expanded: list[TrackCfg] = []
        for track in self.enabled_tracks():
            domains = track.evaluation_domains()
            if not domains:
                expanded.append(track)
                continue
            group_id = track.group_id or track.id
            for domain in domains:
                domain.to_identity()
                effective_id = track.id
                if len(domains) > 1:
                    effective_id = f"{track.id}@{domain.id}"
                config = _merge_mappings(track.config, domain.overrides)
                if track.scenario == _FREE_SPACE_SCENARIO:
                    _validate_free_space(FreeSpaceTrackCfg(**config))
                elif track.scenario == "atomic_skill":
                    _validate_atomic_skill_track(
                        replace(track, id=effective_id, config=config)
                    )
                expanded.append(
                    replace(
                        track,
                        id=effective_id,
                        config=config,
                        domain=domain,
                        domains=[],
                        group_id=group_id,
                    )
                )
        expanded_ids = [track.id for track in expanded]
        if len(expanded_ids) != len(set(expanded_ids)):
            raise ValueError(
                "Expanded benchmark track ids must be unique; got duplicates: "
                f"{sorted({track_id for track_id in expanded_ids if expanded_ids.count(track_id) > 1})}."
            )
        return expanded

    def sync_track_configs(self) -> None:
        """Copy typed free-space settings into the matching track config."""
        free_space_tracks = [
            track
            for track in self.tracks
            if track.scenario == _FREE_SPACE_SCENARIO and track.enabled
        ]
        for track in self.tracks:
            if track.scenario == _FREE_SPACE_SCENARIO and track.enabled:
                track.id = track.id or _FREE_SPACE_TRACK_ID
                if len(free_space_tracks) == 1:
                    track.config = asdict(self.free_space)
                elif track.config:
                    _validate_free_space(FreeSpaceTrackCfg(**track.config))
                else:
                    track.config = asdict(self.free_space)

    def validate_benchmark(self) -> None:
        """Validate values that affect benchmark correctness."""
        self.sync_track_configs()
        missing_fields = self.validate()
        if missing_fields:
            raise ValueError(
                "Benchmark suite has missing required fields: "
                + ", ".join(missing_fields)
            )
        if self.schema_version != 1:
            raise ValueError(
                f"Unsupported suite schema_version={self.schema_version}; expected 1."
            )
        if not self.planners:
            raise ValueError("The benchmark suite must declare at least one planner.")
        planner_ids = [spec.id for spec in self.planners]
        if len(planner_ids) != len(set(planner_ids)):
            raise ValueError("Planner ids must be unique within a suite.")
        for spec in self.planners:
            if not spec.id or not spec.adapter:
                raise ValueError(
                    "Every planner must define a non-empty id and adapter."
                )
            AlgorithmRole(spec.role)
        if not self.robot.id or not self.robot.provider:
            raise ValueError("robot must define non-empty id and provider values.")
        if self.embodiment is not None:
            if not self.embodiment.component.strip():
                raise ValueError("embodiment.component must be non-empty.")
            if not isinstance(self.embodiment.overrides, Mapping):
                raise TypeError("embodiment.overrides must be a mapping.")
            if not isinstance(self.embodiment.endpoint_bindings, Mapping):
                raise TypeError("embodiment.endpoint_bindings must be a mapping.")
            if not isinstance(self.embodiment.runtime_services, Mapping):
                raise TypeError("embodiment.runtime_services must be a mapping.")
        if not self.tracks:
            raise ValueError("The benchmark suite must declare at least one track.")
        track_ids = [track.id for track in self.tracks]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("Track ids must be unique within a suite.")
        if not self.enabled_tracks():
            raise ValueError("At least one track must be enabled.")
        for track in self.tracks:
            domains = track.evaluation_domains()
            domain_ids = []
            for domain in domains:
                domain.to_identity()
                if not isinstance(domain.overrides, Mapping):
                    raise TypeError("domain.overrides must be a mapping.")
                if not isinstance(domain.perturbations, Mapping):
                    raise TypeError("domain.perturbations must be a mapping.")
                if not isinstance(domain.metadata, Mapping):
                    raise TypeError("domain.metadata must be a mapping.")
                if any(
                    not isinstance(capability, str) or not capability.strip()
                    for capability in domain.required_capabilities
                ):
                    raise TypeError(
                        "domain.required_capabilities must contain strings."
                    )
                if any(
                    not isinstance(skill_id, str) or not skill_id.strip()
                    for skill_id in (*domain.applicability, *domain.objects)
                ):
                    raise TypeError(
                        "domain applicability and objects must contain strings."
                    )
                domain_ids.append(domain.id)
            if len(domain_ids) != len(set(domain_ids)):
                raise ValueError(f"Track {track.id!r} must not repeat domain ids.")
            if not track.id or not track.scenario:
                raise ValueError("Every track must define a non-empty id and scenario.")
        if self.protocol.warmup_trials < 0:
            raise ValueError("warmup_trials must be >= 0.")
        if self.protocol.measured_trials < 1:
            raise ValueError("measured_trials must be >= 1.")
        if self.protocol.sample_interval < 2:
            raise ValueError("sample_interval must be >= 2.")
        if self.protocol.validation_samples < 2:
            raise ValueError("validation_samples must be >= 2.")
        if self.protocol.position_threshold_m <= 0.0:
            raise ValueError("position_threshold_m must be > 0.")
        if self.protocol.rotation_threshold_rad <= 0.0:
            raise ValueError("rotation_threshold_rad must be > 0.")
        if self.protocol.joint_limit_tolerance_rad < 0.0:
            raise ValueError("joint_limit_tolerance_rad must be >= 0.")
        if not 0.0 < self.protocol.confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0 and 1.")
        if self.protocol.report_schema_version != "3":
            raise ValueError(
                "This runner writes report_schema_version '3'; migrate the suite "
                "protocol to '3' before generating new reports. Existing artifacts "
                "retain their original schema."
            )
        if any(type(seed) is not int for seed in self.protocol.seeds):
            raise TypeError("protocol.seeds must contain integers.")
        if len(self.protocol.seeds) != len(set(self.protocol.seeds)):
            raise ValueError("protocol.seeds must not contain duplicates.")
        for field_name in (
            "joint_velocity_limit_rad_s",
            "joint_acceleration_limit_rad_s2",
            "joint_jerk_limit_rad_s3",
        ):
            limit = getattr(self.protocol, field_name)
            if limit is None:
                continue
            values = limit if isinstance(limit, (list, tuple)) else [limit]
            if not values or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
                for value in values
            ):
                raise ValueError(f"{field_name} entries must be finite and > 0.")
        # NaN fails every comparison, so `tolerance < 0` would let it through
        # and `utilization > 1 + nan` would then never report a violation.
        tolerance = self.protocol.dynamic_limit_tolerance
        if (
            not isinstance(tolerance, (int, float))
            or isinstance(tolerance, bool)
            or not math.isfinite(float(tolerance))
            or float(tolerance) < 0.0
        ):
            raise ValueError("dynamic_limit_tolerance must be finite and >= 0.")
        if any(track.scenario == _FREE_SPACE_SCENARIO for track in self.tracks):
            _validate_free_space(self.free_space)
        for track in self.tracks:
            if track.scenario == "atomic_skill":
                _validate_atomic_skill_track(track)
        for nmg in (spec for spec in self.planners if spec.adapter == "nmg_onnx"):
            for field, default in (
                ("num_waypoints", 5),
                ("steps_per_waypoint", 30),
                ("max_steps", 150),
            ):
                _positive_int(
                    nmg.config.get(field),
                    name=f"NMG {field}",
                    default=default,
                )
            if float(nmg.config.get("pos_eps", 0.01)) <= 0.0:
                raise ValueError("NMG pos_eps must be > 0.")
            if float(nmg.config.get("rot_eps", 0.1)) <= 0.0:
                raise ValueError("NMG rot_eps must be > 0.")


def _resolve_tracks_and_free_space(
    data: dict[str, Any],
) -> tuple[list[TrackCfg], FreeSpaceTrackCfg]:
    """Accept either ``tracks`` or legacy top-level ``free_space``."""
    tracks_data = data.get("tracks")
    free_space_data = dict(data.get("free_space", {}) or {})
    if tracks_data is None:
        tracks = [
            TrackCfg(
                id=_FREE_SPACE_TRACK_ID,
                scenario=_FREE_SPACE_SCENARIO,
                enabled=True,
                config=dict(free_space_data),
            )
        ]
    else:
        if not isinstance(tracks_data, list):
            raise TypeError("tracks must be a list of track mappings.")
        tracks = []
        for item in tracks_data:
            values = dict(item)
            domain = values.get("domain")
            if domain is not None:
                if not isinstance(domain, Mapping):
                    raise TypeError("track.domain must be a mapping.")
                values["domain"] = DomainCfg(**domain)
            domains = values.get("domains")
            if domains is not None:
                if not isinstance(domains, Sequence) or isinstance(
                    domains, (str, bytes)
                ):
                    raise TypeError("track.domains must be a sequence of mappings.")
                resolved_domains = []
                for item_domain in domains:
                    if not isinstance(item_domain, Mapping):
                        raise TypeError("Every track domain must be a mapping.")
                    resolved_domains.append(DomainCfg(**dict(item_domain)))
                values["domains"] = resolved_domains
            tracks.append(TrackCfg(**values))
        free_space_tracks = [
            track
            for track in tracks
            if track.scenario == _FREE_SPACE_SCENARIO and track.enabled
        ]
        if len(free_space_tracks) == 1 and free_space_tracks[0].config:
            free_space_data = {**free_space_data, **dict(free_space_tracks[0].config)}
        elif len(free_space_tracks) > 1 and free_space_data:
            for track in free_space_tracks:
                if not track.config:
                    track.config = deepcopy(free_space_data)
    return tracks, FreeSpaceTrackCfg(**free_space_data)


def _merge_mappings(
    base: Mapping[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    """Recursively merge domain overrides without mutating suite input."""
    merged = {str(key): deepcopy(value) for key, value in base.items()}
    for key, value in override.items():
        key = str(key)
        if isinstance(merged.get(key), Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_mappings(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _validate_free_space(free_space: FreeSpaceTrackCfg) -> None:
    """Validate free-space case-matrix fields."""
    if not isinstance(free_space.target_types, (list, tuple)) or not all(
        isinstance(value, str) for value in free_space.target_types
    ):
        raise TypeError("target_types must be a sequence of strings.")
    if not free_space.target_types:
        raise ValueError("target_types must not be empty.")
    unknown_targets = set(free_space.target_types) - _SUPPORTED_TARGET_TYPES
    if unknown_targets:
        raise ValueError(f"Unsupported target_types: {sorted(unknown_targets)}.")
    threshold = free_space.joint_threshold_rad
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not math.isfinite(float(threshold))
        or threshold <= 0.0
    ):
        raise ValueError("joint_threshold_rad must be finite and > 0.")
    if not free_space.batch_sizes or any(value < 1 for value in free_space.batch_sizes):
        raise ValueError("batch_sizes must contain positive integers.")
    if not free_space.waypoint_counts or any(
        value < 1 for value in free_space.waypoint_counts
    ):
        raise ValueError("waypoint_counts must contain positive integers.")
    if not free_space.seeds:
        raise ValueError("seeds must not be empty.")
    if not free_space.path_shapes:
        raise ValueError("path_shapes must not be empty.")
    unknown_shapes = set(free_space.path_shapes) - _SUPPORTED_PATH_SHAPES
    if unknown_shapes:
        raise ValueError(f"Unsupported path_shapes: {sorted(unknown_shapes)}.")
    if not free_space.start_state_bins:
        raise ValueError("start_state_bins must not be empty.")
    unknown_bins = set(free_space.start_state_bins) - _SUPPORTED_START_STATE_BINS
    if unknown_bins:
        raise ValueError(f"Unsupported start_state_bins: {sorted(unknown_bins)}.")
    for name, values in (
        ("batch_sizes", free_space.batch_sizes),
        ("waypoint_counts", free_space.waypoint_counts),
        ("path_shapes", free_space.path_shapes),
        ("start_state_bins", free_space.start_state_bins),
        ("seeds", free_space.seeds),
        ("target_types", free_space.target_types),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"{name} must not contain duplicate values.")


def _finite_nonnegative_vector(
    value: object,
    *,
    name: str,
    default: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Normalize one finite non-negative XYZ perturbation vector."""
    resolved = default if value is None else value
    if not isinstance(resolved, Sequence) or isinstance(resolved, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of three numbers.")
    values = tuple(float(item) for item in resolved)
    if len(values) != 3 or not all(math.isfinite(item) for item in values):
        raise ValueError(f"{name} must contain three finite values.")
    if any(item < 0.0 for item in values):
        raise ValueError(f"{name} values must be non-negative.")
    return values


def _finite_nonnegative_scalar(value: object, *, name: str, default: float) -> float:
    """Normalize one finite non-negative scalar perturbation bound."""
    resolved = default if value is None else float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return resolved


def _finite_nonnegative_pair(
    value: object,
    *,
    name: str,
    default: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float]:
    """Normalize one finite non-negative roll/pitch perturbation pair."""
    resolved = default if value is None else value
    if not isinstance(resolved, Sequence) or isinstance(resolved, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of two numbers.")
    values = tuple(float(item) for item in resolved)
    if len(values) != 2 or not all(math.isfinite(item) for item in values):
        raise ValueError(f"{name} must contain two finite values.")
    if any(item < 0.0 for item in values):
        raise ValueError(f"{name} values must be non-negative.")
    return values


def _positive_int(value: object, *, name: str, default: int) -> int:
    """Normalize one strict positive integer configuration value."""
    resolved = default if value is None else value
    if type(resolved) is not int:
        raise TypeError(f"{name} must be a positive integer.")
    if resolved < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return resolved


def _integer(value: object, *, name: str) -> int:
    """Normalize an integer that may be negative (for example a random seed)."""
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer.")
    return value


def resolve_atomic_pose_randomization(
    track: TrackCfg,
) -> AtomicPoseRandomizationCfg:
    """Resolve shared random-pose controls from an Atomic Skill track.

    Accepted spellings are deliberately compatible with the existing
    skill-level randomization keys.  The canonical nested form is:

    .. code-block:: yaml

       randomization:
         enabled: true
         mode: spatial_pose
         sampling_pattern: stratified
         pose_batch_size: 8
         object_translation_jitter_m: [0.03, 0.03, 0.0]
         object_yaw_jitter_rad: 0.5236
         object_roll_pitch_jitter_rad: [0.1745, 0.1745]
         target_translation_jitter_m: [0.03, 0.03, 0.02]
         target_yaw_jitter_rad: 0.5236
         target_roll_pitch_jitter_rad: [0.1745, 0.1745]
         support_object_translation_jitter_m: [0.03, 0.03, 0.0]
         support_object_yaw_jitter_rad: 0.5236
         support_object_roll_pitch_jitter_rad: [0.1745, 0.1745]
         task_frame_translation_jitter_m: [0.03, 0.03, 0.0]
         task_frame_yaw_jitter_rad: 0.1745
         task_frame_roll_pitch_jitter_rad: [0.0873, 0.0873]
         seed_stride: 1

    ``random_pose_count`` is accepted as a readable alias for
    ``pose_batch_size``.  The helper does not mutate the track mapping, so the
    resolved suite and case manifest remain faithful to the source YAML.
    ``workspace_shell`` assigns ordered normalized XY offset radii to rows;
    three-dimensional translation vectors use those radii while scalar and
    rotation ranges retain deterministic stratification.
    """
    if track.scenario != "atomic_skill":
        raise ValueError(
            "resolve_atomic_pose_randomization expects an atomic_skill track."
        )
    config = track.config
    if not isinstance(config, Mapping):
        raise TypeError("atomic-skill track config must be a mapping.")
    raw_randomization = config.get("randomization", {})
    if raw_randomization is None:
        raw_randomization = {}
    if not isinstance(raw_randomization, Mapping):
        raise TypeError("atomic-skill randomization must be a mapping.")
    randomization = dict(raw_randomization)

    batch_value = randomization.get("pose_batch_size")
    if batch_value is None:
        batch_value = randomization.get("random_pose_count")
    if batch_value is None:
        batch_value = config.get("pose_batch_size")
    if batch_value is None:
        batch_value = config.get("random_pose_count")
    pose_batch_size = _positive_int(
        batch_value,
        name="atomic-skill randomization.pose_batch_size",
        default=1,
    )

    mode = str(
        randomization.get("mode", config.get("randomization_mode", "translation"))
    )
    if mode not in {"translation", "planar_pose", "spatial_pose", "none"}:
        raise ValueError(
            "atomic-skill randomization.mode must be 'translation', "
            "'planar_pose', 'spatial_pose', or 'none'."
        )

    sampling_pattern = str(
        randomization.get(
            "sampling_pattern",
            config.get("pose_sampling_pattern", "uniform"),
        )
    )
    if sampling_pattern not in {"uniform", "stratified", "workspace_shell"}:
        raise ValueError(
            "atomic-skill randomization.sampling_pattern must be "
            "'uniform', 'stratified', or 'workspace_shell'."
        )

    object_jitter_value = randomization.get("object_translation_jitter_m")
    if object_jitter_value is None:
        object_jitter_value = randomization.get("object_position_jitter_m")
    if object_jitter_value is None:
        object_jitter_value = config.get("object_translation_jitter_m")
    if object_jitter_value is None:
        object_jitter_value = config.get("object_position_jitter_m")
    object_jitter = _finite_nonnegative_vector(
        object_jitter_value,
        name="atomic-skill randomization.object_translation_jitter_m",
        default=(0.0, 0.0, 0.0),
    )

    object_yaw_jitter = _finite_nonnegative_scalar(
        randomization.get(
            "object_yaw_jitter_rad",
            config.get("object_yaw_jitter_rad"),
        ),
        name="atomic-skill randomization.object_yaw_jitter_rad",
        default=0.0,
    )
    object_roll_pitch_jitter = _finite_nonnegative_pair(
        randomization.get(
            "object_roll_pitch_jitter_rad",
            config.get("object_roll_pitch_jitter_rad"),
        ),
        name="atomic-skill randomization.object_roll_pitch_jitter_rad",
    )

    target_jitter_value = randomization.get("target_translation_jitter_m")
    if target_jitter_value is None:
        target_jitter_value = randomization.get("target_offset_jitter_m")
    if target_jitter_value is None:
        target_jitter_value = config.get("target_translation_jitter_m")
    if target_jitter_value is None:
        target_jitter_value = config.get("target_offset_jitter_m")
    target_jitter = _finite_nonnegative_vector(
        target_jitter_value,
        name="atomic-skill randomization.target_translation_jitter_m",
        default=(0.0, 0.0, 0.0),
    )

    target_yaw_jitter = _finite_nonnegative_scalar(
        randomization.get(
            "target_yaw_jitter_rad",
            config.get("target_yaw_jitter_rad"),
        ),
        name="atomic-skill randomization.target_yaw_jitter_rad",
        default=0.0,
    )
    target_roll_pitch_jitter = _finite_nonnegative_pair(
        randomization.get(
            "target_roll_pitch_jitter_rad",
            config.get("target_roll_pitch_jitter_rad"),
        ),
        name="atomic-skill randomization.target_roll_pitch_jitter_rad",
    )

    support_jitter_value = randomization.get("support_object_translation_jitter_m")
    if support_jitter_value is None:
        support_jitter_value = randomization.get("support_object_position_jitter_m")
    if support_jitter_value is None:
        support_jitter_value = config.get("support_object_translation_jitter_m")
    if support_jitter_value is None:
        support_jitter_value = config.get("support_object_position_jitter_m")
    support_jitter = _finite_nonnegative_vector(
        support_jitter_value,
        name="atomic-skill randomization.support_object_translation_jitter_m",
        default=(0.0, 0.0, 0.0),
    )

    support_yaw_jitter = _finite_nonnegative_scalar(
        randomization.get(
            "support_object_yaw_jitter_rad",
            config.get("support_object_yaw_jitter_rad"),
        ),
        name="atomic-skill randomization.support_object_yaw_jitter_rad",
        default=0.0,
    )
    support_roll_pitch_jitter = _finite_nonnegative_pair(
        randomization.get(
            "support_object_roll_pitch_jitter_rad",
            config.get("support_object_roll_pitch_jitter_rad"),
        ),
        name="atomic-skill randomization.support_object_roll_pitch_jitter_rad",
    )

    task_frame_jitter_value = randomization.get("task_frame_translation_jitter_m")
    if task_frame_jitter_value is None:
        task_frame_jitter_value = config.get("task_frame_translation_jitter_m")
    task_frame_jitter = _finite_nonnegative_vector(
        task_frame_jitter_value,
        name="atomic-skill randomization.task_frame_translation_jitter_m",
        default=(0.0, 0.0, 0.0),
    )

    task_frame_yaw_jitter = _finite_nonnegative_scalar(
        randomization.get(
            "task_frame_yaw_jitter_rad",
            config.get("task_frame_yaw_jitter_rad"),
        ),
        name="atomic-skill randomization.task_frame_yaw_jitter_rad",
        default=0.0,
    )
    task_frame_roll_pitch_jitter = _finite_nonnegative_pair(
        randomization.get(
            "task_frame_roll_pitch_jitter_rad",
            config.get("task_frame_roll_pitch_jitter_rad"),
        ),
        name="atomic-skill randomization.task_frame_roll_pitch_jitter_rad",
    )

    if sampling_pattern == "workspace_shell":
        translation_jitters = {
            "object_translation_jitter_m": object_jitter,
            "target_translation_jitter_m": target_jitter,
            "support_object_translation_jitter_m": support_jitter,
            "task_frame_translation_jitter_m": task_frame_jitter,
        }
        single_axis = [
            name
            for name, values in translation_jitters.items()
            if (values[0] > 0.0) != (values[1] > 0.0)
        ]
        if single_axis:
            raise ValueError(
                "atomic-skill workspace_shell requires both X and Y for "
                f"{', '.join(single_axis)}."
            )
        if not any(
            values[0] > 0.0 and values[1] > 0.0
            for values in translation_jitters.values()
        ):
            raise ValueError(
                "atomic-skill workspace_shell requires at least one translation "
                "jitter with positive X and Y ranges."
            )

    stride = _positive_int(
        randomization.get("seed_stride", config.get("randomization_seed_stride")),
        name="atomic-skill randomization.seed_stride",
        default=1,
    )
    enabled_value = randomization.get("enabled")
    if enabled_value is None:
        enabled = pose_batch_size > 1 or any(
            value > 0.0
            for value in (
                *object_jitter,
                object_yaw_jitter,
                *object_roll_pitch_jitter,
                *target_jitter,
                target_yaw_jitter,
                *target_roll_pitch_jitter,
                *support_jitter,
                support_yaw_jitter,
                *support_roll_pitch_jitter,
                *task_frame_jitter,
                task_frame_yaw_jitter,
                *task_frame_roll_pitch_jitter,
            )
        )
    elif isinstance(enabled_value, bool):
        enabled = enabled_value
    else:
        raise TypeError("atomic-skill randomization.enabled must be a boolean.")
    if (
        mode == "none"
        and enabled
        and (
            any(value > 0.0 for value in object_jitter)
            or object_yaw_jitter > 0.0
            or any(value > 0.0 for value in object_roll_pitch_jitter)
            or any(value > 0.0 for value in target_jitter)
            or target_yaw_jitter > 0.0
            or any(value > 0.0 for value in target_roll_pitch_jitter)
            or any(value > 0.0 for value in support_jitter)
            or support_yaw_jitter > 0.0
            or any(value > 0.0 for value in support_roll_pitch_jitter)
            or any(value > 0.0 for value in task_frame_jitter)
            or task_frame_yaw_jitter > 0.0
            or any(value > 0.0 for value in task_frame_roll_pitch_jitter)
        )
    ):
        raise ValueError(
            "atomic-skill randomization.mode='none' cannot specify non-zero jitter."
        )
    yaw_jitters = {
        "object_yaw_jitter_rad": object_yaw_jitter,
        "target_yaw_jitter_rad": target_yaw_jitter,
        "support_object_yaw_jitter_rad": support_yaw_jitter,
        "task_frame_yaw_jitter_rad": task_frame_yaw_jitter,
    }
    nonzero_yaw = [name for name, value in yaw_jitters.items() if value > 0.0]
    roll_pitch_jitters = {
        "object_roll_pitch_jitter_rad": object_roll_pitch_jitter,
        "target_roll_pitch_jitter_rad": target_roll_pitch_jitter,
        "support_object_roll_pitch_jitter_rad": support_roll_pitch_jitter,
        "task_frame_roll_pitch_jitter_rad": task_frame_roll_pitch_jitter,
    }
    nonzero_roll_pitch = [
        name for name, values in roll_pitch_jitters.items() if any(values)
    ]
    nonzero_rotation = [*nonzero_yaw, *nonzero_roll_pitch]
    if mode == "translation" and enabled and nonzero_rotation:
        raise ValueError(
            "atomic-skill randomization.mode='translation' cannot specify "
            f"{', '.join(nonzero_rotation)}."
        )
    if mode == "planar_pose" and enabled and nonzero_roll_pitch:
        raise ValueError(
            "atomic-skill randomization.mode='planar_pose' cannot specify "
            f"{', '.join(nonzero_roll_pitch)}."
        )
    return AtomicPoseRandomizationCfg(
        enabled=enabled,
        pose_batch_size=pose_batch_size,
        mode=mode,
        sampling_pattern=sampling_pattern,
        object_translation_jitter_m=object_jitter,
        object_yaw_jitter_rad=object_yaw_jitter,
        object_roll_pitch_jitter_rad=object_roll_pitch_jitter,
        target_translation_jitter_m=target_jitter,
        target_yaw_jitter_rad=target_yaw_jitter,
        target_roll_pitch_jitter_rad=target_roll_pitch_jitter,
        support_object_translation_jitter_m=support_jitter,
        support_object_yaw_jitter_rad=support_yaw_jitter,
        support_object_roll_pitch_jitter_rad=support_roll_pitch_jitter,
        task_frame_translation_jitter_m=task_frame_jitter,
        task_frame_yaw_jitter_rad=task_frame_yaw_jitter,
        task_frame_roll_pitch_jitter_rad=task_frame_roll_pitch_jitter,
        seed_stride=stride,
    )


def resolve_atomic_batch_sizes(track: TrackCfg) -> list[int]:
    """Return simulator batch sizes for an Atomic Skill track.

    Explicit ``batch_sizes`` remains authoritative.  When omitted, an enabled
    random-pose configuration uses its ``pose_batch_size`` as the single
    batch, allowing one planner call to evaluate all sampled poses.
    """
    if track.scenario != "atomic_skill":
        raise ValueError("resolve_atomic_batch_sizes expects an atomic_skill track.")
    randomization = resolve_atomic_pose_randomization(track)
    raw_values = track.config.get("batch_sizes")
    if raw_values is None:
        return [randomization.pose_batch_size if randomization.enabled else 1]
    if not isinstance(raw_values, Sequence) or isinstance(raw_values, (str, bytes)):
        raise TypeError("atomic-skill batch_sizes must be a sequence of integers.")
    values = [
        _positive_int(value, name="atomic-skill batch_sizes", default=1)
        for value in raw_values
    ]
    if not values:
        raise ValueError("atomic-skill batch_sizes must not be empty.")
    if len(values) != len(set(values)):
        raise ValueError("atomic-skill batch_sizes must not contain duplicates.")
    if randomization.enabled and values != [randomization.pose_batch_size]:
        # One randomized pose is assigned to each environment row, so its
        # declared pose batch and simulator batch must remain identical.
        raise ValueError(
            "atomic-skill randomization requires batch_sizes to contain exactly "
            f"[{randomization.pose_batch_size}] (one pose batch per case)."
        )
    return values


def _validate_atomic_skill_track(track: TrackCfg) -> None:
    """Validate shared atomic-track matrix controls without touching skills."""
    resolve_atomic_pose_randomization(track)
    resolve_atomic_batch_sizes(track)
    values = track.config.get("seeds", [11])
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("atomic-skill seeds must be a sequence of integers.")
    if not values:
        raise ValueError("atomic-skill seeds must not be empty.")
    seeds = [_integer(value, name="atomic-skill seeds") for value in values]
    if len(seeds) != len(set(seeds)):
        raise ValueError("atomic-skill seeds must not contain duplicate values.")


def load_suite(name_or_path: str | Path = "motion_panda_free_space_smoke") -> SuiteCfg:
    """Load an installed suite or an external YAML file.

    Args:
        name_or_path: Installed suite stem or path to an external YAML file.

    Returns:
        A validated suite with component paths resolved relative to its file.

    Raises:
        FileNotFoundError: No file or installed suite matches the request.
    """
    requested = Path(name_or_path)
    if requested.is_file():
        return _load_suite_path(requested)
    resource = files(__package__).joinpath("suites", f"{name_or_path}.yaml")
    if not resource.is_file():
        raise FileNotFoundError(f"Benchmark suite not found: {name_or_path}")
    with as_file(resource) as path:
        return _load_suite_path(path)


def _load_suite_path(path: Path) -> SuiteCfg:
    """Decode a suite while its resource or external file is available."""
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise TypeError(
            f"Expected a mapping in suite {path}, got {type(data).__name__}."
        )
    suite = SuiteCfg.from_dict(data)
    if suite.embodiment is not None:
        from .embodiments import _resolve_component_path

        suite.embodiment.component = str(
            _resolve_component_path(
                suite.embodiment.component, base_dir=path.resolve().parent
            )
        )
    return suite


def suite_to_dict(suite: SuiteCfg) -> dict[str, Any]:
    """Convert a resolved suite to plain YAML/JSON-compatible values."""
    suite.sync_track_configs()
    data = asdict(suite)
    data.pop("free_space", None)
    if data.get("embodiment") is None:
        data.pop("embodiment", None)
    else:
        # The component is the sole owner; robot is derived runtime metadata.
        data.pop("robot", None)
    for track in data.get("tracks", []):
        domain = track.get("domain")
        if isinstance(domain, dict):
            track["domain"] = _compact_domain_dict(domain)
        domains = track.get("domains")
        if isinstance(domains, list):
            track["domains"] = [
                _compact_domain_dict(item) if isinstance(item, dict) else item
                for item in domains
            ]
    return data


def _compact_domain_dict(domain: dict[str, Any]) -> dict[str, Any]:
    """Drop empty optional domain fields from resolved YAML output."""
    compact = dict(domain)
    for key in (
        "overrides",
        "required_capabilities",
        "applicability",
        "objects",
        "frozen_cases",
        "perturbations",
        "metadata",
    ):
        if compact.get(key) in ({}, [], None):
            compact.pop(key, None)
    for key in ("case_generator", "scene_layout"):
        if compact.get(key) is None:
            compact.pop(key, None)
    return compact


def stable_hash(value: object) -> str:
    """Return a stable SHA256 hash for JSON-compatible configuration data."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
