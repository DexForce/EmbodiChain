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

"""Interaction-constrained nominal pose repair before the expansion baseline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from copy import deepcopy
import hashlib
import json
import math
from types import MappingProxyType

import numpy as np
from scipy.spatial.transform import Rotation

from embodichain.compute.trajectory.refinement import (
    PoseRefinementResult,
    _poses,
    refine_pose_trajectory,
)
from embodichain.data_pipeline.real2sim import PhaseAnnotation
from embodichain.utils import configclass

__all__ = [
    "MotionRefinementCfg",
    "InteractionConstraintSet",
    "MotionRefinementResult",
    "KinematicMotionRefiner",
]


@configclass
class MotionRefinementCfg:
    """Explicit permission and finite budgets for Cartesian nominal repair.

    Translation and rotation budgets cover the whole source-to-refined change,
    including anchor alignment. Contact events, gripper commands and control
    time are unchanged. Geometry compatibility, IK and collision acceptance
    remain independent host/skill checks.
    """

    max_translation_m: float = 0.02
    max_rotation_rad: float = math.radians(10)
    smoothing_weight: float = 0.2
    max_candidates: int = 8
    seed: int = 0
    support_margin_m: float = 0.002
    max_release_gap_m: float = 0.04
    max_iterations: int = 4000
    project_pick_lift: bool = False
    """Explicitly repair a Pick lift endpoint before protecting its baseline."""
    project_pick_approach: bool = False
    """Repair the Pick approach endpoint locally while preserving its free entry."""
    project_free_boundaries: bool = False
    """Stop at free-phase boundaries and join adjacent contact anchors exactly."""
    pick_entry_projection_samples: int = 0
    """Explicit free-entry transition width; zero preserves an already paced entry."""

    def __post_init__(self) -> None:
        if any(
            type(value) is not bool
            for value in (
                self.project_pick_lift,
                self.project_pick_approach,
                self.project_free_boundaries,
            )
        ):
            raise ValueError("Pick projection permissions must be boolean")
        if (
            type(self.pick_entry_projection_samples) is not int
            or self.pick_entry_projection_samples not in (0,)
            and not 4 <= self.pick_entry_projection_samples <= 4096
        ):
            raise ValueError(
                "Pick entry projection width must be zero or within [4, 4096]"
            )
        if self.pick_entry_projection_samples and not self.project_pick_approach:
            raise ValueError(
                "Pick entry projection requires approach projection permission"
            )
        for name in (
            "max_translation_m",
            "max_rotation_rad",
            "smoothing_weight",
            "support_margin_m",
            "max_release_gap_m",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"refinement {name} must be finite and nonnegative")
        if self.max_translation_m <= 0 or not 0 < self.max_rotation_rad < math.pi / 2:
            raise ValueError("refinement correction budgets are invalid")
        for name, minimum in (
            ("max_candidates", 1),
            ("seed", 0),
            ("max_iterations", 1),
        ):
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value < 2**31:
                raise ValueError(f"refinement {name} must be a bounded integer")
        if self.max_candidates > 64 or self.max_iterations > 100_000:
            raise ValueError("refinement search budget is too large")

    @property
    def fingerprint(self) -> str:
        """Return the content identity of all permissions and solver settings."""
        payload = asdict(self)
        # Disabled projection has the same semantics/identity as older profiles.
        if not self.project_pick_lift:
            payload.pop("project_pick_lift")
        if not self.project_pick_approach:
            payload.pop("project_pick_approach")
        if not self.project_free_boundaries:
            payload.pop("project_free_boundaries")
        if not self.pick_entry_projection_samples:
            payload.pop("pick_entry_projection_samples")
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()


@dataclass(frozen=True)
class InteractionConstraintSet:
    """One grounded skill's exact anchor and provenance.

    Args:
        skill_id: Pick, held transport or direct Place.
        phases: Complete, contiguous source phase partition.
        anchor_index: Grasp/lift entry, transport endpoint or release sample.
        anchor_pose: World TCP target derived from the task and geometry.
        geometry_id: Geometry/affordance evidence identity, never a category name.
        lift_height_m: Explicit requested world-Z lift for opt-in Pick repair.
    """

    skill_id: str
    phases: tuple[PhaseAnnotation, ...]
    anchor_index: int
    anchor_pose: np.ndarray
    geometry_id: str
    lift_height_m: float | None = None

    def __post_init__(self) -> None:
        if self.skill_id not in {"pick_up", "move_held_object", "place"}:
            raise ValueError("unsupported refinement skill")
        if self.lift_height_m is not None and (
            self.skill_id != "pick_up"
            or type(self.lift_height_m) not in (int, float)
            or not math.isfinite(self.lift_height_m)
            or self.lift_height_m < 0
        ):
            raise ValueError("lift_height_m requires a nonnegative finite Pick goal")
        if not self.geometry_id or not isinstance(self.geometry_id, str):
            raise ValueError("refinement requires geometry evidence identity")
        phases = tuple(self.phases)
        if (
            not phases
            or any(not isinstance(p, PhaseAnnotation) for p in phases)
            or phases[0].start != 0
            or any(a.stop != b.start for a, b in zip(phases, phases[1:]))
        ):
            raise ValueError("refinement phases must be a complete partition")
        if (
            type(self.anchor_index) is not int
            or not 0 <= self.anchor_index < phases[-1].stop
        ):
            raise ValueError("refinement anchor index is outside the trajectory")
        if any(phase.authority == "qpos" for phase in phases):
            raise ValueError("refinement cannot override qpos-authority phases")
        expected = {
            "pick_up": ("approach", "close", "lift"),
            "move_held_object": ("transport",),
            "place": ("approach", "release", "retract"),
        }[self.skill_id]
        if tuple(p.name for p in phases) != expected:
            raise ValueError("refinement phases do not match the skill contract")
        expected_index = (
            phases[2].start
            if self.skill_id == "pick_up"
            else (phases[1].start if self.skill_id == "place" else phases[-1].stop - 1)
        )
        if self.anchor_index != expected_index:
            raise ValueError(
                "refinement anchor must address the skill interaction event"
            )
        pose = _poses(np.asarray(self.anchor_pose)[None], "interaction anchor")[0]
        pose.setflags(write=False)
        object.__setattr__(self, "anchor_pose", pose)
        object.__setattr__(self, "phases", phases)


@dataclass(frozen=True)
class MotionRefinementResult:
    """Owned pose correction evidence for one scene-specific baseline.

    Args:
        original_poses: Retargeted source TCP path before any repair.
        fit: Fixed-grid numerical result.
        report: JSON-compatible grounded constraints and correction diagnostics.
    """

    original_poses: np.ndarray
    fit: PoseRefinementResult
    report: Mapping[str, object]

    def __post_init__(self) -> None:
        poses = _poses(self.original_poses, "original poses")
        if (
            not isinstance(self.fit, PoseRefinementResult)
            or self.fit.poses.shape != poses.shape
        ):
            raise ValueError("refinement result arrays must share one grid")
        poses.setflags(write=False)
        object.__setattr__(self, "original_poses", poses)
        object.__setattr__(
            self, "report", MappingProxyType(deepcopy(dict(self.report)))
        )


class KinematicMotionRefiner:
    """Fit a bounded Cartesian path to skill-owned interaction constraints.

    Args:
        cfg: Explicit repair permissions and numerical budgets.
    """

    def __init__(self, cfg: MotionRefinementCfg) -> None:
        if not isinstance(cfg, MotionRefinementCfg):
            raise TypeError("refiner requires MotionRefinementCfg")
        self.cfg = cfg.copy()
        self.cfg.__post_init__()

    def refine(
        self,
        poses: np.ndarray,
        constraints: InteractionConstraintSet,
        *,
        observation_weights: np.ndarray | None = None,
    ) -> MotionRefinementResult:
        """Repair geometry before freezing protected contact and hold samples.

        A gravity-preserving anchor warp supplies the initial reference. Contact
        phases are exact stationary TCP constraints; hold phases retain the
        warped reference, with world-up displacement preserved. Explicit Pick
        endpoint projection repairs a lift's entry/exit before freezing that
        reference; all resulting changes consume the same correction budget.
        Free interiors
        are fitted by a sparse QP. Two samples at each phase edge remain at the
        corrected reference so later expansion can protect derivative stencils.

        Args:
            poses: Scene-retargeted source TCP path, before anchor repair.
            constraints: Grounded interaction anchor and complete phase contract.
            observation_weights: Optional positive source-confidence weights.

        Returns:
            Refined path and auditable, scene-specific correction report.

        Raises:
            ValueError: Source permissions, timing partition or repair is invalid.
        """
        original = _poses(poses, "source TCP")
        if not isinstance(constraints, InteractionConstraintSet) or constraints.phases[
            -1
        ].stop != len(original):
            raise ValueError("refinement constraints must cover the source TCP path")
        anchor = original[constraints.anchor_index]
        goal = constraints.anchor_pose
        delta = goal @ np.linalg.inv(anchor)
        reference = delta @ original
        # The observed roll/pitch correction rotates the tool, not gravity.
        yaw = math.atan2(delta[1, 0], delta[0, 0])
        rotation = np.array(
            [
                [math.cos(yaw), -math.sin(yaw), 0],
                [math.sin(yaw), math.cos(yaw), 0],
                [0, 0, 1],
            ]
        )
        reference[:, :3, 3] = (
            goal[:3, 3] + (original[:, :3, 3] - anchor[:3, 3]) @ rotation.T
        )
        projected_lift = (
            self.cfg.project_pick_lift and constraints.skill_id == "pick_up"
        )
        projected_approach = (
            self.cfg.project_pick_approach and constraints.skill_id == "pick_up"
        )
        if projected_approach:
            phase = constraints.phases[0]
            count = phase.stop - phase.start
            if count < 4:
                raise ValueError("Pick approach projection needs four boundary samples")
            part = original[phase.start : phase.stop].copy()
            progress = np.clip((np.arange(count) - 1) / (count - 3), 0, 1)
            blend = progress**3 * (10 - 15 * progress + 6 * progress**2)
            part[:, :3, 3] += blend[:, None] * (goal[:3, 3] - part[-1, :3, 3])
            correction = Rotation.from_matrix(
                part[-1, :3, :3].T @ goal[:3, :3]
            ).as_rotvec()
            part[:, :3, :3] = (
                part[:, :3, :3]
                @ Rotation.from_rotvec(blend[:, None] * correction).as_matrix()
            )
            # A newly planned entry connection ends at rest. Blend the first
            # free samples from that rest pose instead of introducing the
            # recording's nonzero entry velocity at one control boundary.
            entry_moving = self.cfg.pick_entry_projection_samples > 0
            entry_count = (
                min(self.cfg.pick_entry_projection_samples, count - 2)
                if entry_moving
                else 0
            )
            entry_progress = np.clip(
                (np.arange(count) - 1) / max(1, entry_count - 2), 0, 1
            )
            entry_blend = (
                entry_progress**3 * (10 - 15 * entry_progress + 6 * entry_progress**2)
                if entry_moving
                else np.ones(count)
            )
            part[:, :3, 3] = part[0, :3, 3] + entry_blend[:, None] * (
                part[:, :3, 3] - part[0, :3, 3]
            )
            entry_rotation = Rotation.from_matrix(
                part[0, :3, :3].T @ part[:, :3, :3]
            ).as_rotvec()
            part[:, :3, :3] = (
                part[0, :3, :3]
                @ Rotation.from_rotvec(
                    entry_blend[:, None] * entry_rotation
                ).as_matrix()
            )
            exit_moving = (
                np.linalg.norm(part[-2, :3, 3] - goal[:3, 3]) > 1e-4
                or Rotation.from_matrix(part[-2, :3, :3].T @ goal[:3, :3]).magnitude()
                > 1e-3
            )
            exit_count = min(12, count - 2) if exit_moving else 0
            exit_progress = np.clip(
                (np.arange(count) - (count - exit_count)) / max(1, exit_count - 3), 0, 1
            )
            exit_blend = (
                exit_progress**3 * (10 - 15 * exit_progress + 6 * exit_progress**2)
                if exit_moving
                else np.zeros(count)
            )
            part[:, :3, 3] += exit_blend[:, None] * (goal[:3, 3] - part[:, :3, 3])
            exit_rotation = Rotation.from_matrix(
                part[:, :3, :3].transpose(0, 2, 1) @ goal[:3, :3]
            ).as_rotvec()
            part[:, :3, :3] = (
                part[:, :3, :3]
                @ Rotation.from_rotvec(exit_blend[:, None] * exit_rotation).as_matrix()
            )
            part[:2] = part[0]
            part[-2:] = goal
            reference[phase.start : phase.stop] = part
        if projected_lift:
            if constraints.lift_height_m is None:
                raise ValueError(
                    "Pick lift projection requires its explicit goal height"
                )
            phase = constraints.phases[2]
            count = phase.stop - phase.start
            if count < 4:
                raise ValueError("Pick lift projection needs four boundary samples")
            target = goal.copy()
            target[2, 3] += constraints.lift_height_m
            progress = np.clip((np.arange(count) - 1) / (count - 3), 0, 1)
            blend = progress**3 * (10 - 15 * progress + 6 * progress**2)
            part = np.repeat(goal[None], count, axis=0)
            part[:, 2, 3] += blend * constraints.lift_height_m
            # The skill's entry/exit stencil is stationary. Source time and
            # gripper events are unchanged; the fitter still checks the full
            # source-to-result translation/rotation correction budgets.
            part[:2] = goal
            part[-2:] = target
            reference[phase.start : phase.stop] = part
        if self.cfg.project_free_boundaries and constraints.skill_id != "pick_up":
            for i, phase in enumerate(constraints.phases):
                if phase.kind != "free":
                    continue
                if phase.stop - phase.start < 4:
                    raise ValueError("free boundary projection needs four samples")
                first = reference[phase.start].copy()
                last = reference[phase.stop - 1].copy()
                if i and constraints.phases[i - 1].kind == "contact":
                    first = goal
                if (
                    i + 1 < len(constraints.phases)
                    and constraints.phases[i + 1].kind == "contact"
                ):
                    last = goal
                if phase.name == "retract" and phase.stop - phase.start < 8:
                    count = phase.stop - phase.start
                    progress = np.linspace(0, 1, count) ** 2
                    part = np.repeat(first[None], count, axis=0)
                    part[:, :3, 3] += progress[:, None] * (last[:3, 3] - first[:3, 3])
                    angular = Rotation.from_matrix(
                        first[:3, :3].T @ last[:3, :3]
                    ).as_rotvec()
                    part[:, :3, :3] = (
                        first[:3, :3]
                        @ Rotation.from_rotvec(progress[:, None] * angular).as_matrix()
                    )
                    reference[phase.start : phase.stop] = part
                boundary_width = 1 if phase.stop - phase.start < 8 else 2
                reference[phase.start : phase.start + boundary_width] = first
                reference[phase.stop - boundary_width : phase.stop] = last
        anchors: dict[int, np.ndarray] = {constraints.anchor_index: goal}
        for phase in constraints.phases:
            if phase.name in {"close", "release"}:
                reference[phase.start : phase.stop] = goal
                indices: Sequence[int] = range(phase.start, phase.stop)
            elif phase.kind in {"hold", "contact"}:
                indices = range(phase.start, phase.stop)
            else:
                indices = (
                    (phase.start, phase.stop - 1)
                    if self.cfg.project_free_boundaries and phase.stop - phase.start < 8
                    else sorted(
                        {
                            phase.start,
                            min(phase.start + 1, phase.stop - 1),
                            max(phase.start, phase.stop - 2),
                            phase.stop - 1,
                        }
                    )
                )
            anchors.update({int(index): reference[index] for index in indices})
        anchors[constraints.anchor_index] = goal
        fit = refine_pose_trajectory(
            original,
            anchors=anchors,
            reference=reference,
            weights=observation_weights,
            max_translation_m=self.cfg.max_translation_m,
            max_rotation_rad=self.cfg.max_rotation_rad,
            smoothing_weight=self.cfg.smoothing_weight,
            max_iterations=self.cfg.max_iterations,
        )
        digest = hashlib.sha256()
        for array in (original, fit.poses):
            digest.update(np.asarray(array, dtype="<f8").tobytes())
        digest.update(self.cfg.fingerprint.encode())
        digest.update(constraints.geometry_id.encode())
        report = {
            "schema": "kinematic-refinement/v1",
            "baseline_id": digest.hexdigest(),
            "profile_sha256": self.cfg.fingerprint,
            "geometry_id": constraints.geometry_id,
            "skill_id": constraints.skill_id,
            "anchor_index": constraints.anchor_index,
            "max_translation_m": fit.max_translation_m,
            "max_rotation_rad": fit.max_rotation_rad,
            "translation_budget_m": self.cfg.max_translation_m,
            "rotation_budget_rad": self.cfg.max_rotation_rad,
            "hard_sample_count": len(anchors),
            "pick_lift_projected": projected_lift,
            "pick_approach_projected": projected_approach,
            "pick_entry_projection_samples": entry_count if projected_approach else 0,
            "free_boundaries_projected": self.cfg.project_free_boundaries,
            "samples": len(original),
            "iterations": fit.iterations,
            "time_grid_changed": False,
            "calibration_estimated": False,
            "collision_qualified": False,
        }
        original.setflags(write=False)
        return MotionRefinementResult(original, fit, report)
