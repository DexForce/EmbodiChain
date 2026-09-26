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

"""Episode-level combined generation contracts.

The contracts in this module deliberately do not depend on Gym, a simulator,
or a renderer.  They provide the deterministic schedule that a host runner
uses to combine reference families, affordance branches, trajectory variants,
and one visual profile per physical episode.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from embodichain.utils import configclass

__all__ = [
    "CombinedGenerationProfile",
    "ReferenceFamilySpec",
    "CubeInitialPoseProvider",
    "CycleRecipe",
    "CandidateRecipe",
    "PhysicalSlotPool",
    "SlotReservation",
    "enumerate_candidate_recipes",
    "load_visual_profile_registry",
    "schedule_digest",
]


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _keys(data: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown fields in {name}: {sorted(unknown, key=str)}")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a nonempty stripped string")
    return value


def _bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _int(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _strings(value: object, name: str, *, nonempty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a sequence")
    result = tuple(_text(item, name) for item in value)
    if nonempty and not result:
        raise ValueError(f"{name} must be nonempty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique values")
    return result


@configclass
class _CombinedSourceCfg:
    kind: str = "task_program"
    source_id: str = "repeated_cube_pick_place"
    source_revision: str = "config:repeated_pick_place_v1"
    unit_scope: str = "episode"
    template_id: str = "repeated_pick_place_episode"

    def __post_init__(self) -> None:
        if self.kind != "task_program":
            raise ValueError("combined source.kind must be task_program")
        for name in ("source_id", "source_revision", "template_id"):
            _text(getattr(self, name), f"source.{name}")
        if self.unit_scope != "episode":
            raise ValueError("combined source.unit_scope must be episode")


@configclass
class _SceneRandomizationCfg:
    enabled: bool = True
    reference_family_count: int = 4
    profile: str = "cube_initial_pose"
    change_scope: str = "cube_initial_pose_only"
    keep_drop_targets: bool = True

    def __post_init__(self) -> None:
        _bool(self.enabled, "scene_randomization.enabled")
        _int(
            self.reference_family_count, "scene_randomization.reference_family_count", 1
        )
        _text(self.profile, "scene_randomization.profile")
        if self.change_scope != "cube_initial_pose_only":
            raise ValueError(
                "scene_randomization.change_scope must be cube_initial_pose_only"
            )
        if not self.keep_drop_targets:
            raise ValueError("combined generation must keep authored drop targets")


@configclass
class _AffordanceCfg:
    enabled: bool = True
    branches_per_family: int = 4
    max_proposals: int = 128
    fallback: str = "next_surviving_branch"

    def __post_init__(self) -> None:
        _bool(self.enabled, "affordance.enabled")
        _int(self.branches_per_family, "affordance.branches_per_family", 1)
        _int(self.max_proposals, "affordance.max_proposals", 1)
        if self.fallback != "next_surviving_branch":
            raise ValueError("affordance.fallback must be next_surviving_branch")


@configclass
class _SpatialCfg:
    enabled: bool = True
    operators: tuple[str, ...] = ("joint_residual", "via_points")
    joint_offset_scale: float = 0.05
    via_count: int = 1

    def __post_init__(self) -> None:
        self.operators = _strings(self.operators, "trajectory.spatial.operators")
        if any(item not in ("joint_residual", "via_points") for item in self.operators):
            raise ValueError(
                "trajectory.spatial.operators contains an unsupported operator"
            )
        _bool(self.enabled, "trajectory.spatial.enabled")
        self.joint_offset_scale = _finite(
            self.joint_offset_scale, "trajectory.spatial.joint_offset_scale"
        )
        if not 0 <= self.joint_offset_scale <= 1:
            raise ValueError("trajectory.spatial.joint_offset_scale must be in [0, 1]")
        _int(self.via_count, "trajectory.spatial.via_count", 1)


@configclass
class _TimingCfg:
    enabled: bool = False

    def __post_init__(self) -> None:
        _bool(self.enabled, "trajectory.timing.enabled")


@configclass
class _TrajectoryCfg:
    variants_per_family: int = 4
    spatial: _SpatialCfg = _SpatialCfg()
    timing: _TimingCfg = _TimingCfg()

    def __post_init__(self) -> None:
        _int(self.variants_per_family, "trajectory.variants_per_family", 1)


@configclass
class _VisualCfg:
    enabled: bool = True
    profile_file: str = "generation_profiles/rgb_visual.yaml"
    profiles: tuple[str, ...] = (
        "rgb_canonical",
        "rgb_material_01",
        "rgb_camera_01",
        "rgb_light_01",
    )
    apply_per_rollout: bool = True
    balance: str = "round_robin"
    restore_before_next_candidate: bool = True

    def __post_init__(self) -> None:
        self.profile_file = _text(self.profile_file, "visual.profile_file")
        self.profiles = _strings(self.profiles, "visual.profiles")
        _bool(self.enabled, "visual.enabled")
        _bool(self.apply_per_rollout, "visual.apply_per_rollout")
        _bool(
            self.restore_before_next_candidate, "visual.restore_before_next_candidate"
        )
        if self.enabled and not self.apply_per_rollout:
            raise ValueError("enabled visual profiles must apply_per_rollout")
        if self.balance != "round_robin":
            raise ValueError("visual.balance must be round_robin")


@configclass
class _ObservationCfg:
    profiles: tuple[str, ...] = ("rgb_lowres",)
    capture_rgb: bool = True
    capture_video: bool = False

    def __post_init__(self) -> None:
        self.profiles = _strings(self.profiles, "observation.profiles")
        _bool(self.capture_rgb, "observation.capture_rgb")
        _bool(self.capture_video, "observation.capture_video")


@configclass
class _SchedulingCfg:
    policy: str = "round_robin_fifo"
    candidate_budget: int = 64
    reference_family_budget: int = 4
    queue_max_items: int = 64
    queue_max_bytes: int = 536870912

    def __post_init__(self) -> None:
        if self.policy != "round_robin_fifo":
            raise ValueError("combined scheduling.policy must be round_robin_fifo")
        for name in (
            "candidate_budget",
            "reference_family_budget",
            "queue_max_items",
            "queue_max_bytes",
        ):
            _int(getattr(self, name), f"scheduling.{name}", 1)


@configclass
class _ExecutionCfg:
    max_inflight: int = 16
    compatibility: str = "strict"
    restore_initial_state: bool = True
    cancel_policy: str = "safe_finish_and_drain"

    def __post_init__(self) -> None:
        _int(self.max_inflight, "execution.max_inflight", 1)
        if self.compatibility != "strict":
            raise ValueError("execution.compatibility must be strict")
        _bool(self.restore_initial_state, "execution.restore_initial_state")
        if self.cancel_policy != "safe_finish_and_drain":
            raise ValueError("execution.cancel_policy must be safe_finish_and_drain")


@configclass
class _PersistenceCfg:
    sink: str = "local_artifact"
    output_dir: str = "/tmp/repeated-pick-place-generation"
    write_trajectories: bool = True
    write_observations: bool = True
    write_manifest: bool = True

    def __post_init__(self) -> None:
        if self.sink != "local_artifact":
            raise ValueError("combined persistence.sink must be local_artifact")
        self.output_dir = _text(self.output_dir, "persistence.output_dir")
        for name in ("write_trajectories", "write_observations", "write_manifest"):
            _bool(getattr(self, name), f"persistence.{name}")


def _construct(cls: type, data: object, name: str, allowed: set[str]) -> Any:
    mapping = _mapping(data, name)
    _keys(mapping, allowed, name)
    values = dict(mapping)
    return cls(**values)


@configclass
class CombinedGenerationProfile:
    """Strict episode-level profile for the combined showcase."""

    schema_version: int = 1
    source: _CombinedSourceCfg = _CombinedSourceCfg()
    scene_randomization: _SceneRandomizationCfg = _SceneRandomizationCfg()
    affordance: _AffordanceCfg = _AffordanceCfg()
    trajectory: _TrajectoryCfg = _TrajectoryCfg()
    visual: _VisualCfg = _VisualCfg()
    observation: _ObservationCfg = _ObservationCfg()
    scheduling: _SchedulingCfg = _SchedulingCfg()
    execution: _ExecutionCfg = _ExecutionCfg()
    persistence: _PersistenceCfg = _PersistenceCfg()

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("combined schema_version must be 1")
        m = self.scene_randomization.reference_family_count
        a = self.affordance.branches_per_family
        t = self.trajectory.variants_per_family
        if self.scheduling.candidate_budget < m * a * t:
            raise ValueError("scheduling.candidate_budget must cover m × a × t recipes")
        if self.scheduling.reference_family_budget < m:
            raise ValueError(
                "scheduling.reference_family_budget must cover all families"
            )
        if self.visual.enabled and not self.visual.profiles:
            raise ValueError("enabled visual generation requires profiles")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CombinedGenerationProfile":
        """Decode the canonical combined schema without evaluating callables."""
        allowed = {
            "schema_version",
            "source",
            "scene_randomization",
            "affordance",
            "trajectory",
            "visual",
            "observation",
            "scheduling",
            "execution",
            "persistence",
        }
        _keys(data, allowed, "combined profile")
        values = {
            "schema_version": _int(data.get("schema_version", 1), "schema_version", 1),
            "source": _construct(
                _CombinedSourceCfg,
                data.get("source", {}),
                "source",
                {"kind", "source_id", "source_revision", "unit_scope", "template_id"},
            ),
            "scene_randomization": _construct(
                _SceneRandomizationCfg,
                data.get("scene_randomization", {}),
                "scene_randomization",
                {
                    "enabled",
                    "reference_family_count",
                    "profile",
                    "change_scope",
                    "keep_drop_targets",
                },
            ),
            "affordance": _construct(
                _AffordanceCfg,
                data.get("affordance", {}),
                "affordance",
                {"enabled", "branches_per_family", "max_proposals", "fallback"},
            ),
            "visual": _construct(
                _VisualCfg,
                data.get("visual", {}),
                "visual",
                {
                    "enabled",
                    "profile_file",
                    "profiles",
                    "apply_per_rollout",
                    "balance",
                    "restore_before_next_candidate",
                },
            ),
            "observation": _construct(
                _ObservationCfg,
                data.get("observation", {}),
                "observation",
                {"profiles", "capture_rgb", "capture_video"},
            ),
            "scheduling": _construct(
                _SchedulingCfg,
                data.get("scheduling", {}),
                "scheduling",
                {
                    "policy",
                    "candidate_budget",
                    "reference_family_budget",
                    "queue_max_items",
                    "queue_max_bytes",
                },
            ),
            "execution": _construct(
                _ExecutionCfg,
                data.get("execution", {}),
                "execution",
                {
                    "max_inflight",
                    "compatibility",
                    "restore_initial_state",
                    "cancel_policy",
                },
            ),
            "persistence": _construct(
                _PersistenceCfg,
                data.get("persistence", {}),
                "persistence",
                {
                    "sink",
                    "output_dir",
                    "write_trajectories",
                    "write_observations",
                    "write_manifest",
                },
            ),
        }
        trajectory = _mapping(data.get("trajectory", {}), "trajectory")
        _keys(trajectory, {"variants_per_family", "spatial", "timing"}, "trajectory")
        spatial = _construct(
            _SpatialCfg,
            trajectory.get("spatial", {}),
            "trajectory.spatial",
            {"enabled", "operators", "joint_offset_scale", "via_count"},
        )
        timing = _construct(
            _TimingCfg, trajectory.get("timing", {}), "trajectory.timing", {"enabled"}
        )
        values["trajectory"] = _TrajectoryCfg(
            variants_per_family=_int(
                trajectory.get("variants_per_family", 4),
                "trajectory.variants_per_family",
                1,
            ),
            spatial=spatial,
            timing=timing,
        )
        return cls(**values)

    def validate_for_num_envs(self, num_envs: int) -> None:
        """Validate the host-owned slot capacity before simulator construction."""
        _int(num_envs, "num_envs", 1)
        if self.execution.max_inflight > num_envs:
            raise ValueError("execution.max_inflight cannot exceed resolved num_envs")

    def validate_registries(
        self,
        *,
        visual_profile_ids: Sequence[str],
        observation_profile_ids: Sequence[str],
    ) -> None:
        """Validate that host registries contain each requested profile once."""
        visual = _strings(visual_profile_ids, "visual_profile_ids")
        observations = _strings(observation_profile_ids, "observation_profile_ids")
        if self.visual.enabled and set(visual) != set(self.visual.profiles):
            raise ValueError(
                "visual profile registry does not match the combined profile"
            )
        if set(observations) != set(self.observation.profiles):
            raise ValueError(
                "observation profile registry does not match the combined profile"
            )


def load_visual_profile_registry(
    path: str | Path,
    *,
    requested_profile_ids: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Load and validate seedable per-environment visual profile IDs.

    Global renderer mutations are rejected at this boundary because they would
    affect unrelated physical slots in the default compatibility bucket.
    """
    with Path(path).open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    root = _mapping(data, "visual profile registry")
    _keys(root, {"profile_id", "revision", "profiles"}, "visual profile registry")
    profiles = _mapping(root.get("profiles", {}), "visual profile registry.profiles")
    ids = tuple(_text(profile_id, "visual profile ID") for profile_id in profiles)
    if len(set(ids)) != len(ids):
        raise ValueError("visual profile IDs must be unique")
    for profile_id, value in profiles.items():
        profile = _mapping(value, f"visual profile {profile_id}")
        _keys(profile, {"seed_offset", "operations"}, f"visual profile {profile_id}")
        _int(
            profile.get("seed_offset", 0), f"visual profile {profile_id}.seed_offset", 0
        )
        operations = profile.get("operations", [])
        if not isinstance(operations, (list, tuple)) or not operations:
            raise ValueError(f"visual profile {profile_id} must declare operations")
        for operation in operations:
            operation_mapping = _mapping(
                operation, f"visual profile {profile_id}.operation"
            )
            if operation_mapping.get("scope") != "per_environment":
                raise ValueError(
                    f"visual profile {profile_id} requests unsupported global state"
                )
    if requested_profile_ids is not None:
        requested = _strings(requested_profile_ids, "requested_profile_ids")
        if set(requested) != set(ids):
            raise ValueError("requested visual profiles do not match the registry")
    return ids


@dataclass(frozen=True)
class ReferenceFamilySpec:
    """One immutable legal initial cube pose and its authored targets."""

    reference_family_id: str
    cube_position: tuple[float, float, float]
    cube_quaternion_xyzw: tuple[float, float, float, float]
    drop_targets: tuple[tuple[str, tuple[float, float, float]], ...]
    deterministic_seed: int
    profile_revision: str = "cube_initial_pose:v1"
    scene_signature: str = "repeated_pick_place_scene_v1"

    def __post_init__(self) -> None:
        _text(self.reference_family_id, "reference_family_id")
        if len(self.cube_position) != 3 or len(self.cube_quaternion_xyzw) != 4:
            raise ValueError(
                "cube pose must contain position(3) and quaternion_xyzw(4)"
            )
        for value in (*self.cube_position, *self.cube_quaternion_xyzw):
            _finite(value, "cube pose value")
        if type(self.deterministic_seed) is not int or self.deterministic_seed < 0:
            raise ValueError("deterministic_seed must be non-negative")
        _text(self.profile_revision, "profile_revision")
        _text(self.scene_signature, "scene_signature")
        names = [name for name, _ in self.drop_targets]
        if len(names) != len(set(names)):
            raise ValueError("drop target IDs must be unique")


class CubeInitialPoseProvider:
    """Return deterministic authored cube pose families for a generation job."""

    def __init__(
        self,
        poses: Sequence[tuple[float, float, float]] | None = None,
        *,
        drop_targets: Sequence[tuple[str, tuple[float, float, float]]] = (
            ("drop_pose_00", (-0.40, 0.48, 0.10)),
            ("drop_pose_01", (-0.42, -0.08, 0.10)),
        ),
        seed: int = 0,
    ) -> None:
        positions = tuple(
            poses
            or (
                (-0.42, -0.08, 0.025),
                (-0.38, -0.02, 0.025),
                (-0.46, -0.14, 0.025),
                (-0.42, 0.04, 0.025),
            )
        )
        if not positions:
            raise ValueError("at least one cube pose is required")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be non-negative")
        targets = tuple(
            (
                _text(name, "drop target ID"),
                tuple(_finite(value, "drop target coordinate") for value in position),
            )
            for name, position in drop_targets
        )
        if any(len(position) != 3 for _, position in targets):
            raise ValueError("drop target positions must have three coordinates")
        self._positions = positions
        self._targets = targets
        self._seed = seed

    def enumerate(self, count: int | None = None) -> tuple[ReferenceFamilySpec, ...]:
        """Enumerate the first ``count`` legal deterministic families."""
        limit = len(self._positions) if count is None else _int(count, "count", 1)
        if limit > len(self._positions):
            raise ValueError("initial-pose provider does not expose enough legal poses")
        result = []
        for index, position in enumerate(self._positions[:limit]):
            if len(position) != 3:
                raise ValueError("cube positions must contain three coordinates")
            result.append(
                ReferenceFamilySpec(
                    reference_family_id=f"cube_pose_{index:02d}",
                    cube_position=tuple(
                        _finite(value, "cube position") for value in position
                    ),
                    cube_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                    drop_targets=self._targets,
                    deterministic_seed=self._seed + index,
                )
            )
        return tuple(result)


@dataclass(frozen=True)
class CycleRecipe:
    """Requested affordance and trajectory ordinals for one Pick/Place cycle."""

    cycle_index: int
    affordance_requested: int
    trajectory_requested: int


@dataclass(frozen=True)
class CandidateRecipe:
    """Complete episode recipe; identity is independent of simulator row."""

    recipe_index: int
    candidate_id: str
    reference_family_id: str
    physical_geometry_family_id: str
    cycle_schedule: tuple[CycleRecipe, ...]
    visual_profile_id: str
    visual_seed: int
    source_unit_id: str
    source_revision: str
    deterministic_seed_digest: str


@dataclass(frozen=True)
class SlotReservation:
    """Opaque reservation returned by :class:`PhysicalSlotPool`."""

    slot_id: int
    candidate_id: str
    compatibility_key: str
    token: str


class PhysicalSlotPool:
    """Generic compatible-slot reservation with exact release semantics.

    A slot can advertise a strict compatibility key.  Candidates remain
    independent of the numeric slot they receive; releasing a stale or foreign
    reservation raises rather than freeing another candidate's slot.
    """

    def __init__(
        self,
        slot_count: int,
        *,
        compatibility_keys: Mapping[int, str] | None = None,
    ) -> None:
        self._slot_count = _int(slot_count, "slot_count", 1)
        keys = {} if compatibility_keys is None else dict(compatibility_keys)
        if any(
            type(slot) is not int or not 0 <= slot < self._slot_count for slot in keys
        ):
            raise ValueError("compatibility_keys contains an invalid slot")
        self._keys = {
            slot: _text(value, "compatibility key") for slot, value in keys.items()
        }
        self._reservations: dict[int, SlotReservation] = {}

    @property
    def capacity(self) -> int:
        """Return the number of physical slots."""
        return self._slot_count

    @property
    def available_count(self) -> int:
        """Return the number of currently unreserved compatible rows."""
        return self._slot_count - len(self._reservations)

    def reserve(self, candidate_id: str, compatibility_key: str) -> SlotReservation:
        """Reserve the first free row compatible with ``compatibility_key``."""
        _text(candidate_id, "candidate_id")
        _text(compatibility_key, "compatibility_key")
        for slot_id in range(self._slot_count):
            if slot_id in self._reservations:
                continue
            advertised = self._keys.get(slot_id)
            if advertised is not None and advertised != compatibility_key:
                continue
            token = _seed_digest("slot", slot_id, candidate_id, compatibility_key)
            reservation = SlotReservation(
                slot_id, candidate_id, compatibility_key, token
            )
            self._reservations[slot_id] = reservation
            return reservation
        raise BufferError("no compatible physical slot is available")

    def release(self, reservation: SlotReservation) -> None:
        """Release exactly the reservation returned by :meth:`reserve`."""
        if not isinstance(reservation, SlotReservation):
            raise TypeError("reservation must be a SlotReservation")
        current = self._reservations.get(reservation.slot_id)
        if current != reservation:
            raise ValueError("reservation is stale or owned by another candidate")
        del self._reservations[reservation.slot_id]

    def reservation_for(self, slot_id: int) -> SlotReservation | None:
        """Inspect a row without exposing mutable pool state."""
        _int(slot_id, "slot_id", 0)
        if slot_id >= self._slot_count:
            raise ValueError("slot_id is outside the pool")
        return self._reservations.get(slot_id)


def _seed_digest(*values: object) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def enumerate_candidate_recipes(
    profile: CombinedGenerationProfile,
    *,
    job_id: str = "repeated_pick_place_generation",
    families: Sequence[ReferenceFamilySpec] | None = None,
) -> tuple[CandidateRecipe, ...]:
    """Enumerate the deterministic ``m × a × t`` episode recipes."""
    if not isinstance(profile, CombinedGenerationProfile):
        raise TypeError("profile must be a CombinedGenerationProfile")
    _text(job_id, "job_id")
    family_count = profile.scene_randomization.reference_family_count
    family_values = (
        tuple(families)
        if families is not None
        else CubeInitialPoseProvider().enumerate(family_count)
    )
    if len(family_values) < family_count:
        raise ValueError("families does not contain the requested reference count")
    a_count = profile.affordance.branches_per_family
    t_count = profile.trajectory.variants_per_family
    visual_profiles = profile.visual.profiles or ("rgb_canonical",)
    result: list[CandidateRecipe] = []
    for family_index, family in enumerate(family_values[:family_count]):
        for a0 in range(a_count):
            for t0 in range(t_count):
                recipe_index = family_index * (a_count * t_count) + a0 * t_count + t0
                cycles = tuple(
                    CycleRecipe(
                        cycle_index=cycle,
                        affordance_requested=(
                            0
                            if recipe_index == 0
                            else (a0 + cycle + 2 * family_index) % a_count
                        ),
                        trajectory_requested=(
                            0
                            if recipe_index == 0
                            else (t0 + 2 * cycle + family_index) % t_count
                        ),
                    )
                    for cycle in range(3)
                )
                visual_index = (
                    0 if recipe_index == 0 else recipe_index % len(visual_profiles)
                )
                digest = _seed_digest(
                    job_id,
                    profile.source.source_revision,
                    family.reference_family_id,
                    recipe_index,
                    tuple(
                        (c.affordance_requested, c.trajectory_requested) for c in cycles
                    ),
                )
                candidate_payload = {
                    "job_id": job_id,
                    "revision": profile.source.source_revision,
                    "family": family.reference_family_id,
                    "recipe_index": recipe_index,
                    "cycles": [
                        (c.affordance_requested, c.trajectory_requested) for c in cycles
                    ],
                    "visual": visual_profiles[visual_index],
                }
                candidate_id = hashlib.sha256(
                    json.dumps(
                        candidate_payload, sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest()
                result.append(
                    CandidateRecipe(
                        recipe_index=recipe_index,
                        candidate_id=f"episode_{recipe_index:04d}_{candidate_id[:16]}",
                        reference_family_id=family.reference_family_id,
                        physical_geometry_family_id=family.reference_family_id,
                        cycle_schedule=cycles,
                        visual_profile_id=visual_profiles[visual_index],
                        visual_seed=int(digest[:16], 16),
                        source_unit_id=profile.source.source_id,
                        source_revision=profile.source.source_revision,
                        deterministic_seed_digest=digest,
                    )
                )
    return tuple(result)


def schedule_digest(recipes: Sequence[CandidateRecipe]) -> str:
    """Hash the complete schedule in order for deterministic rerun checks."""
    payload = [
        {
            "recipe_index": item.recipe_index,
            "candidate_id": item.candidate_id,
            "family": item.reference_family_id,
            "cycles": [
                [cycle.affordance_requested, cycle.trajectory_requested]
                for cycle in item.cycle_schedule
            ],
            "visual": item.visual_profile_id,
            "seed": item.deterministic_seed_digest,
        }
        for item in recipes
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
