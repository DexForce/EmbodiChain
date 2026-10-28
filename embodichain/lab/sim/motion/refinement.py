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

    def __post_init__(self) -> None:
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
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, allow_nan=False).encode()
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
    """

    skill_id: str
    phases: tuple[PhaseAnnotation, ...]
    anchor_index: int
    anchor_pose: np.ndarray
    geometry_id: str

    def __post_init__(self) -> None:
        if self.skill_id not in {"pick_up", "move_held_object", "place"}:
            raise ValueError("unsupported refinement skill")
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
        warped reference, with world-up displacement preserved. Free interiors
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
        anchors: dict[int, np.ndarray] = {constraints.anchor_index: goal}
        for phase in constraints.phases:
            if phase.name in {"close", "release"}:
                reference[phase.start : phase.stop] = goal
                indices: Sequence[int] = range(phase.start, phase.stop)
            elif phase.kind in {"hold", "contact"}:
                indices = range(phase.start, phase.stop)
            else:
                indices = sorted(
                    {
                        phase.start,
                        min(phase.start + 1, phase.stop - 1),
                        max(phase.start, phase.stop - 2),
                        phase.stop - 1,
                    }
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
            "samples": len(original),
            "iterations": fit.iterations,
            "time_grid_changed": False,
            "calibration_estimated": False,
            "collision_qualified": False,
        }
        original.setflags(write=False)
        return MotionRefinementResult(original, fit, report)
