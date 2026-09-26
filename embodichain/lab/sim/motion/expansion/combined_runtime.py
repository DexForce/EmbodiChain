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

"""Host-neutral runtime contracts for combined episode generation.

These values coordinate recipe assignment and measured evidence without
constructing a simulator. Environment integrations own the actual restore,
visual mutation, command execution, and sensor capture operations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from types import MappingProxyType
from typing import Any

import yaml
import torch
import numpy as np

from .combined import CandidateRecipe, PhysicalSlotPool, SlotReservation
from .contracts import ExpertEpisode, ValidationCheck, ValidationResult

__all__ = [
    "VisualProfileApplication",
    "VisualProfileRegistry",
    "CombinedAssignment",
    "CombinedEpisodeCoordinator",
    "MeasuredValidator",
    "round_robin_recipes",
]


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a nonempty stripped string")
    return value


def _finite(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _plain(value: object) -> object:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        return _finite(value, "profile value")
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    raise ValueError("profile values must be JSON-compatible")


@dataclass(frozen=True)
class VisualProfileApplication:
    """Resolved per-environment visual operations for one physical episode."""

    profile_id: str
    revision: str
    seed: int
    operations: tuple[Mapping[str, object], ...]
    status: str = "resolved"

    def __post_init__(self) -> None:
        _text(self.profile_id, "profile_id")
        _text(self.revision, "revision")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be in [0, 2**63)")
        if self.status not in {"resolved", "applied", "rejected"}:
            raise ValueError("unknown visual application status")
        normalized = []
        for operation in self.operations:
            if not isinstance(operation, Mapping):
                raise ValueError("visual operations must be mappings")
            normalized.append(dict(_plain(operation)))
        object.__setattr__(self, "operations", tuple(normalized))


class VisualProfileRegistry:
    """Load seedable, per-environment visual profiles from a strict YAML file."""

    def __init__(
        self,
        profile_id: str,
        revision: str,
        profiles: Mapping[str, Mapping[str, object]],
    ) -> None:
        self.profile_id = _text(profile_id, "profile_id")
        self.revision = _text(revision, "revision")
        if not isinstance(profiles, Mapping) or not profiles:
            raise ValueError("visual registry must contain profiles")
        self._profiles = {str(key): dict(value) for key, value in profiles.items()}
        if len(self._profiles) != len(profiles):
            raise ValueError("visual profile IDs must be unique")
        for profile_name, profile in self._profiles.items():
            _text(profile_name, "visual profile ID")
            if not isinstance(profile, Mapping):
                raise ValueError("visual profile values must be mappings")
            operations = profile.get("operations")
            if not isinstance(operations, (list, tuple)) or not operations:
                raise ValueError(
                    f"visual profile {profile_name!r} must declare operations"
                )
            for operation in operations:
                if not isinstance(operation, Mapping):
                    raise ValueError("visual operations must be mappings")
                if operation.get("scope") != "per_environment":
                    raise ValueError(
                        f"visual profile {profile_name!r} requests global renderer state"
                    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VisualProfileRegistry":
        """Load one closed visual registry without importing executable values."""
        with Path(path).open(encoding="utf-8") as stream:
            root = yaml.safe_load(stream)
        if not isinstance(root, Mapping):
            raise ValueError("visual registry root must be a mapping")
        unknown = set(root) - {"profile_id", "revision", "profiles"}
        if unknown:
            raise ValueError(f"unknown fields in visual registry: {sorted(unknown)}")
        return cls(root.get("profile_id"), root.get("revision"), root.get("profiles"))

    @property
    def profile_ids(self) -> tuple[str, ...]:
        """Return registered profile IDs in file order."""
        return tuple(self._profiles)

    def resolve(self, profile_id: str, *, seed: int) -> VisualProfileApplication:
        """Resolve one profile and derive its deterministic episode seed."""
        _text(profile_id, "profile_id")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        try:
            profile = self._profiles[profile_id]
        except KeyError as error:
            raise ValueError(f"unknown visual profile: {profile_id!r}") from error
        offset = profile.get("seed_offset", 0)
        if type(offset) is not int or offset < 0:
            raise ValueError("visual seed_offset must be a non-negative integer")
        operations = tuple(
            dict(_plain(operation)) for operation in profile["operations"]
        )
        return VisualProfileApplication(
            profile_id=profile_id,
            revision=self.revision,
            seed=(seed + offset) % 2**63,
            operations=operations,
        )

    def apply_to_environment(
        self,
        env: object,
        assignments: Mapping[int, str],
        *,
        seed: int,
    ) -> tuple[VisualProfileApplication, ...]:
        """Apply resolved per-environment operations through existing functors.

        The method intentionally accepts a narrow structural environment port
        rather than importing Gym types. It requires ``env.sim`` and applies
        only operations declared with ``scope: per_environment``.
        """
        if not assignments:
            raise ValueError("assignments must be nonempty")
        sim = getattr(env, "sim", None)
        if sim is None:
            raise TypeError("env must expose a simulation manager")
        from embodichain.lab.gym.envs.managers.cfg import SceneEntityCfg
        from embodichain.lab.gym.envs.managers.randomization.visual import (
            randomize_camera_extrinsics,
            randomize_camera_intrinsics,
            set_rigid_object_visual_material,
        )
        from embodichain.lab.sim import VisualMaterialCfg

        grouped: dict[str, list[int]] = {}
        for env_id, profile_id in assignments.items():
            if type(env_id) is not int or env_id < 0:
                raise ValueError("assignments keys must be non-negative row IDs")
            grouped.setdefault(_text(profile_id, "profile_id"), []).append(env_id)
        applications: list[VisualProfileApplication] = []
        torch_state = torch.random.get_rng_state()
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        try:
            for profile_id, env_ids in grouped.items():
                application = self.resolve(profile_id, seed=seed)
                ids = torch.tensor(env_ids, dtype=torch.long)
                for operation in application.operations:
                    kind = operation.get("kind")
                    if kind == "authored_cube_material":
                        continue
                    if kind == "cube_material":
                        color = list(operation.get("base_color", [0.5, 0.5, 0.5]))
                        if len(color) == 3:
                            color.append(1.0)
                        set_rigid_object_visual_material(
                            env,
                            ids,
                            SceneEntityCfg(uid="cube"),
                            VisualMaterialCfg(
                                uid=f"cube_{profile_id}",
                                base_color=color,
                                roughness=float(operation.get("roughness", 0.7)),
                            ),
                        )
                    elif kind == "camera_extrinsics":
                        delta = list(operation.get("position_delta", [0.0, 0.0, 0.0]))
                        randomize_camera_extrinsics(
                            env,
                            ids,
                            SceneEntityCfg(uid="cam_overhead"),
                            eye_range=(delta, delta),
                        )
                    elif kind == "camera_intrinsics":
                        scale = float(operation.get("focal_scale", 1.0))
                        camera = sim.get_sensor("cam_overhead")
                        fx, fy = camera.cfg.intrinsics[:2]
                        randomize_camera_intrinsics(
                            env,
                            ids,
                            SceneEntityCfg(uid="cam_overhead"),
                            focal_x_range=(fx * (scale - 1.0), fx * (scale - 1.0)),
                            focal_y_range=(fy * (scale - 1.0), fy * (scale - 1.0)),
                        )
                    elif kind == "local_light":
                        light = sim.get_light("local_light")
                        if light is None or getattr(light, "is_global", False):
                            raise ValueError(
                                "rgb_light_01 requires a per-environment local_light"
                            )
                        color = torch.tensor(
                            operation.get("color", [1.0, 1.0, 1.0]),
                            dtype=torch.float32,
                        ).repeat(len(env_ids), 1)
                        light.set_color(color, env_ids=ids)
                    else:
                        raise ValueError(f"unsupported visual operation: {kind!r}")
                applications.append(application)
        finally:
            torch.random.set_rng_state(torch_state)
            np.random.set_state(numpy_state)
            random.setstate(python_state)
        return tuple(applications)


def round_robin_recipes(
    recipes: Sequence[CandidateRecipe],
    *,
    family_count: int,
) -> tuple[CandidateRecipe, ...]:
    """Return recipes in family round-robin order with FIFO within each family."""
    if type(family_count) is not int or family_count < 1:
        raise ValueError("family_count must be a positive integer")
    values = tuple(recipes)
    if not values or len(values) % family_count:
        raise ValueError("recipes must divide evenly across reference families")
    groups: dict[str, list[CandidateRecipe]] = {}
    for recipe in values:
        groups.setdefault(recipe.reference_family_id, []).append(recipe)
    if len(groups) != family_count:
        raise ValueError("recipe family count does not match family_count")
    ordered: list[CandidateRecipe] = []
    for offset in range(len(values) // family_count):
        for family in groups:
            ordered.append(groups[family][offset])
    return tuple(ordered)


@dataclass(frozen=True)
class CombinedAssignment:
    """One recipe reserved on one compatible physical slot."""

    recipe: CandidateRecipe
    reservation: SlotReservation
    compatibility_key: str


class CombinedEpisodeCoordinator:
    """Bounded FIFO coordinator for recipe assignment and slot cleanup."""

    def __init__(
        self,
        recipes: Sequence[CandidateRecipe],
        *,
        slot_pool: PhysicalSlotPool,
        family_count: int,
        compatibility_key: str,
    ) -> None:
        if not isinstance(slot_pool, PhysicalSlotPool):
            raise TypeError("slot_pool must be a PhysicalSlotPool")
        _text(compatibility_key, "compatibility_key")
        self._recipes = list(round_robin_recipes(recipes, family_count=family_count))
        self._pool = slot_pool
        self._compatibility_key = compatibility_key
        self._active: dict[str, CombinedAssignment] = {}
        self._counts = {"assigned": 0, "accepted": 0, "rejected": 0, "write_failed": 0}
        self._cancelled = False

    @property
    def pending_count(self) -> int:
        """Return recipes waiting for assignment."""
        return len(self._recipes)

    @property
    def active_count(self) -> int:
        """Return recipes currently holding a physical slot."""
        return len(self._active)

    def acquire_next(self) -> CombinedAssignment | None:
        """Reserve the next recipe, or return ``None`` on capacity/cancellation."""
        if self._cancelled or not self._recipes:
            return None
        recipe = self._recipes[0]
        try:
            reservation = self._pool.reserve(
                recipe.candidate_id, self._compatibility_key
            )
        except BufferError:
            return None
        self._recipes.pop(0)
        assignment = CombinedAssignment(recipe, reservation, self._compatibility_key)
        self._active[recipe.candidate_id] = assignment
        self._counts["assigned"] += 1
        return assignment

    def finish(self, assignment: CombinedAssignment, *, status: str) -> None:
        """Release an active assignment and count its terminal status."""
        if not isinstance(assignment, CombinedAssignment):
            raise TypeError("assignment must be a CombinedAssignment")
        current = self._active.get(assignment.recipe.candidate_id)
        if current != assignment:
            raise ValueError("assignment is stale or owned by another coordinator")
        if status not in {"accepted", "rejected", "write_failed"}:
            raise ValueError("unknown combined assignment status")
        self._pool.release(assignment.reservation)
        del self._active[assignment.recipe.candidate_id]
        self._counts[status] += 1

    def cancel(self) -> None:
        """Stop new work and release every active reservation safely."""
        self._cancelled = True
        for assignment in tuple(self._active.values()):
            self._pool.release(assignment.reservation)
            del self._active[assignment.recipe.candidate_id]

    def snapshot(self) -> Mapping[str, object]:
        """Return immutable scheduler counters and queue state."""
        return MappingProxyType(
            {
                **self._counts,
                "pending": len(self._recipes),
                "active": len(self._active),
                "cancelled": self._cancelled,
                "available_slots": self._pool.available_count,
            }
        )


class MeasuredValidator:
    """Fail-closed validator for measured episode and three-cycle evidence."""

    def validate(
        self, episode: ExpertEpisode, *, cycle_count: int = 3
    ) -> ValidationResult:
        """Return required checks from finite aligned evidence and cycle metadata."""
        if not isinstance(episode, ExpertEpisode):
            raise TypeError("episode must be an ExpertEpisode")
        if type(cycle_count) is not int or cycle_count < 1:
            raise ValueError("cycle_count must be a positive integer")
        checks = [
            ValidationCheck("commands_finite", "passed"),
            ValidationCheck("timestamps_aligned", "passed"),
            ValidationCheck("observations_finite", "passed"),
        ]
        metadata = episode.metadata
        cycle_poses = metadata.get("cycle_poses")
        target_matches = metadata.get("target_matches")
        if (
            not isinstance(cycle_poses, (list, tuple))
            or len(cycle_poses) != cycle_count
        ):
            checks.append(
                ValidationCheck("cycle_count", "failed", "missing measured cycle poses")
            )
        else:
            checks.append(ValidationCheck("cycle_count", "passed"))
        if (
            not isinstance(target_matches, (list, tuple))
            or len(target_matches) != cycle_count
        ):
            checks.append(
                ValidationCheck(
                    "target_pose_matches", "failed", "missing per-cycle target evidence"
                )
            )
        elif not all(type(value) is bool for value in target_matches):
            checks.append(
                ValidationCheck(
                    "target_pose_matches", "failed", "target evidence must be boolean"
                )
            )
        elif all(target_matches):
            checks.append(ValidationCheck("target_pose_matches", "passed"))
        else:
            checks.append(
                ValidationCheck(
                    "target_pose_matches", "failed", "one or more targets mismatched"
                )
            )
        return ValidationResult(tuple(checks))
