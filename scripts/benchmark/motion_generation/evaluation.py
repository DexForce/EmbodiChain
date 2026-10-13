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

"""Versioned Atomic Skill criteria applied to measured, row-local evidence.

This module never reads a planner, simulator, or projected symbolic context.
Observation adapters own contact interpretation, geometry and segment sampling.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal
import math

from .contracts import PhysicalEvaluation, StageOutcome

__all__ = [
    "STANDARD_VERSION",
    "TOLERANCES",
    "SKILL_CRITERIA",
    "Criterion",
    "StageCriterion",
    "StageEvidence",
    "evaluate_skill_stage_batch",
    "evaluate_skill_stages",
    "validate_standard_evaluation",
]

STANDARD_VERSION = "atomic-skill-v1"
# Adopt #642's absolute SI thresholds, including its rounded angular values.
TOLERANCES: Mapping[str, float] = MappingProxyType(
    {
        "primitive_position_m": 0.003,
        "primitive_joint_l2_rad": 0.00524,
        "task_position_m": 0.01,
        "task_rotation_rad": 0.08727,
        "pick_min_lift_m": 0.04,
        "press_min_stroke_ratio": 0.80,
        "drop_margin_m": 0.15,
        "handover_xy_m": 0.03,
        "handover_z_lower_m": -0.10,
        "handover_z_upper_m": 0.03,
    }
)


@dataclass(frozen=True)
class Criterion:
    """One finite measured quantity and its absolute acceptance bound."""

    measurement: str
    operator: Literal["le", "ge", "gt", "true"]
    threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.operator not in {"le", "ge", "gt", "true"}:
            raise ValueError("Unknown criterion operator.")

    def accepts(self, value: float | bool) -> bool:
        """Check a measurement without interpreting missing evidence as zero.

        Args:
            value: Measured scalar in the unit encoded in the measurement name.

        Returns:
            Whether the scalar meets this criterion.
        """
        if self.operator == "true":
            return type(value) is bool and value
        if type(value) not in {int, float} or not math.isfinite(value):
            return False
        if not self.measurement.startswith("signed_") and (
            "error_" in self.measurement or "drift_" in self.measurement
        ):
            if value < 0:
                return False
        if self.operator == "le":
            return value <= self.threshold
        if self.operator == "ge":
            return value >= self.threshold
        return value > self.threshold


@dataclass(frozen=True)
class StageCriterion:
    """Required observation window, measurements and physical failure reason."""

    stage_id: str
    window: str
    criteria: tuple[Criterion, ...]
    failure_code: str


@dataclass(frozen=True)
class StageEvidence:
    """Measured evidence for one stage of one environment.

    A proxy must name its frozen measurement method; it cannot be a projected
    held-object state. Sample times and raw traces belong in the case artifacts.
    """

    window: str
    source: Literal["simulation_state", "contact_sensor", "measured_proxy"]
    measurements: Mapping[str, float | bool]
    method: str

    def __post_init__(self) -> None:
        if self.source not in {"simulation_state", "contact_sensor", "measured_proxy"}:
            raise ValueError("Evidence must originate from execution measurements.")
        if not isinstance(self.method, str) or not self.method.strip():
            raise ValueError("Evidence requires a named measurement method.")
        if not isinstance(self.window, str) or not self.window.strip():
            raise ValueError("Evidence requires an observation window.")
        values = dict(self.measurements)
        for name, value in values.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Measurement names must be non-empty strings.")
            if type(value) not in {int, float, bool} or not math.isfinite(value):
                raise ValueError("Measurements must be finite numeric values.")
        object.__setattr__(self, "measurements", MappingProxyType(values))


def _stage(
    stage_id: str, window: str, code: str, *criteria: Criterion
) -> StageCriterion:
    return StageCriterion(stage_id, window, criteria, code)


def _bool(name: str) -> Criterion:
    return Criterion(name, "true")


def _position(name: str = "position_error_m") -> Criterion:
    return Criterion(name, "le", TOLERANCES["task_position_m"])


def _rotation(name: str = "rotation_error_rad") -> Criterion:
    return Criterion(name, "le", TOLERANCES["task_rotation_rad"])


# Height uses the same frozen object reference point as the case support plane.
_NO_DROP = Criterion("min_height_above_support_m", "ge", -TOLERANCES["drop_margin_m"])
_GRASP = _stage(
    "grasped", "grasp_end", "object_not_grasped", _bool("grasp_established")
)
_HELD = _stage(
    "held", "terminal_settle", "object_dropped", _bool("object_retained"), _NO_DROP
)
_RELEASE = _stage(
    "released", "release_end", "release_failure", _bool("release_established")
)
_LIFT = _stage(
    "lifted",
    "lift_end",
    "task_goal_miss",
    Criterion("lift_delta_m", "ge", TOLERANCES["pick_min_lift_m"]),
    _NO_DROP,
)
_STABLE = _stage(
    "stable",
    "settle_interval",
    "task_goal_miss",
    _position("position_drift_m"),
    _rotation("rotation_drift_rad"),
    _NO_DROP,
)

SKILL_CRITERIA: Mapping[str, tuple[StageCriterion, ...]] = MappingProxyType(
    {
        "move_end_effector": (
            _stage(
                "reached",
                "terminal_settle",
                "task_goal_miss",
                Criterion(
                    "tcp_position_error_m", "le", TOLERANCES["primitive_position_m"]
                ),
            ),
        ),
        "move_joints": (
            _stage(
                "reached",
                "terminal_settle",
                "task_goal_miss",
                Criterion(
                    "joint_error_l2_rad", "le", TOLERANCES["primitive_joint_l2_rad"]
                ),
            ),
        ),
        "pick_up": (_GRASP, _LIFT, _HELD),
        "axis_align": (
            _GRASP,
            _stage(
                "aligned",
                "actuation_end",
                "task_goal_miss",
                _rotation("axis_angle_error_rad"),
                _NO_DROP,
            ),
            _HELD,
        ),
        "move_held_object": (
            _stage(
                "transported",
                "actuation_end",
                "task_goal_miss",
                _position(),
                _rotation(),
                _NO_DROP,
            ),
        ),
        "place": (
            _RELEASE,
            _stage(
                "placed",
                "terminal_settle",
                "task_goal_miss",
                _position("xy_error_m"),
                _NO_DROP,
            ),
            _STABLE,
        ),
        "pour": (
            _stage(
                "poured",
                "actuation_interval",
                "task_goal_miss",
                _rotation("peak_signed_axis_angle_error_rad"),
                _NO_DROP,
            ),
        ),
        "push_object": (
            _stage(
                "contacted",
                "contact_end",
                "contact_not_established",
                _bool("contact_established"),
            ),
            _stage(
                "pushed",
                "actuation_end",
                "task_goal_miss",
                Criterion("signed_displacement_m", "gt", 0.0),
                _position("in_plane_error_m"),
            ),
            _stage(
                "on_support_plane",
                "terminal_settle",
                "task_goal_miss",
                _bool("on_support_plane"),
                _rotation("tilt_error_rad"),
            ),
        ),
        "press": (
            _stage(
                "contacted",
                "actuation_interval",
                "contact_not_established",
                _bool("contact_established"),
            ),
            _stage(
                "pressed",
                "actuation_interval",
                "task_goal_miss",
                Criterion(
                    "peak_signed_stroke_ratio",
                    "ge",
                    TOLERANCES["press_min_stroke_ratio"],
                ),
            ),
        ),
        "slide": (
            _GRASP,
            _stage(
                "slid", "actuation_end", "task_goal_miss", _position("travel_error_m")
            ),
            _RELEASE,
        ),
        "open_door": (
            _GRASP,
            _stage(
                "opened",
                "actuation_end",
                "task_goal_miss",
                _rotation("hinge_angle_error_rad"),
            ),
            _RELEASE,
        ),
        "twist": (
            _GRASP,
            _stage(
                "twisted",
                "actuation_end",
                "task_goal_miss",
                _rotation("tcp_axis_angle_error_rad"),
            ),
            _RELEASE,
        ),
        "coordinated_pickment": (
            _stage(
                "dual_grasped",
                "grasp_end",
                "object_not_grasped",
                _bool("left_grasp_established"),
                _bool("right_grasp_established"),
            ),
            _LIFT,
            _stage(
                "moved",
                "actuation_end",
                "task_goal_miss",
                _position(),
                _rotation(),
                _NO_DROP,
            ),
            _HELD,
        ),
        "coordinated_placement": (
            _stage(
                "aligned",
                "actuation_end",
                "task_goal_miss",
                _position("relative_position_error_m"),
                _rotation("relative_rotation_error_rad"),
                _NO_DROP,
            ),
            _stage(
                "released",
                "release_end",
                "release_failure",
                _bool("placing_release_established"),
                _bool("support_retained"),
            ),
            _STABLE,
        ),
        "hand_over": (
            _GRASP,
            _stage(
                "transferred",
                "transfer_end",
                "object_not_grasped",
                _bool("both_grasps_established"),
                _NO_DROP,
            ),
            _stage(
                "handed_over",
                "handover_end",
                "release_failure",
                _bool("giver_released"),
                _bool("receiver_retained"),
                _NO_DROP,
            ),
            _stage(
                "placed",
                "terminal_settle",
                "task_goal_miss",
                Criterion("xy_error_m", "le", TOLERANCES["handover_xy_m"]),
                Criterion("signed_z_error_m", "ge", TOLERANCES["handover_z_lower_m"]),
                Criterion("signed_z_error_m", "le", TOLERANCES["handover_z_upper_m"]),
                _NO_DROP,
            ),
        ),
    }
)


def evaluate_skill_stages(
    skill_id: str,
    evidence: Mapping[str, StageEvidence],
    *,
    env_index: int,
) -> PhysicalEvaluation:
    """Evaluate the declared stage ladder using measured evidence only.

    Args:
        skill_id: Canonical snake-case Atomic Skill identifier.
        evidence: Row-local observations indexed by stage id, sampled by the host.
        env_index: Environment row associated with every supplied observation.

    Returns:
        Versioned results; missing evidence stops the ladder as unavailable.

    Raises:
        ValueError: Skill, stage identity or observation window is incorrect.
    """
    if skill_id not in SKILL_CRITERIA:
        raise ValueError(f"Unknown standard skill: {skill_id!r}.")
    specs = SKILL_CRITERIA[skill_id]
    if set(evidence) - {spec.stage_id for spec in specs}:
        raise ValueError("Evidence contains an undeclared stage.")
    results = []
    stopped = False
    for spec in specs:
        if stopped:
            results.append(StageOutcome(spec.stage_id, "not_reached"))
            continue
        sample = evidence.get(spec.stage_id)
        if sample is not None and sample.window != spec.window:
            raise ValueError(f"{spec.stage_id} requires window {spec.window!r}.")
        measurements = {} if sample is None else dict(sample.measurements)
        source = None if sample is None else f"{sample.source}:{sample.method}"
        if sample is None or any(
            c.measurement not in measurements for c in spec.criteria
        ):
            stage = StageOutcome(
                spec.stage_id,
                "unavailable",
                "evidence_unavailable",
                measurements,
                source,
                spec.window,
            )
        else:
            # A physically dropped object has a more useful attribution than a pose miss.
            dropped = _NO_DROP in spec.criteria and not _NO_DROP.accepts(
                measurements[_NO_DROP.measurement]
            )
            passed = all(c.accepts(measurements[c.measurement]) for c in spec.criteria)
            stage = StageOutcome(
                spec.stage_id,
                "passed" if passed else "failed",
                None if passed else "object_dropped" if dropped else spec.failure_code,
                measurements,
                source,
                spec.window,
            )
        results.append(stage)
        stopped = stage.status != "passed"
    return PhysicalEvaluation(env_index, tuple(results), STANDARD_VERSION)


def evaluate_skill_stage_batch(
    skill_id: str,
    evidence_by_env: Sequence[Mapping[str, StageEvidence]],
) -> tuple[PhysicalEvaluation, ...]:
    """Evaluate row-local stage evidence in stable environment order.

    This is the bridge between batched replay observations and the standard's
    deliberately row-local evaluator. Observation adapters remain responsible
    for sampling physical state and naming the measurement method; this helper
    only assigns environment indices and applies the frozen criteria.

    Args:
        skill_id: Canonical snake-case Atomic Skill identifier.
        evidence_by_env: One stage-evidence mapping per environment row.

    Returns:
        Physical evaluations ordered by environment index.
    """
    return tuple(
        evaluate_skill_stages(skill_id, evidence, env_index=env_index)
        for env_index, evidence in enumerate(evidence_by_env)
    )


def validate_standard_evaluation(skill_id: str, result: PhysicalEvaluation) -> None:
    """Reject a claimed v1 result inconsistent with the executable criteria.

    Args:
        skill_id: Canonical skill identifier for the frozen case.
        result: Evaluator output, possibly from a legacy custom evaluator.

    Raises:
        ValueError: Version, ladder, provenance, measurements or verdict disagree.
    """
    if result.protocol_version is None:
        return  # Explicit legacy/unversioned result; never advertised as v1.
    if result.protocol_version != STANDARD_VERSION:
        raise ValueError(f"Unsupported physical protocol: {result.protocol_version!r}.")
    if skill_id not in SKILL_CRITERIA:
        raise ValueError(f"Unknown standard skill: {skill_id!r}.")
    if tuple(s.stage_id for s in result.stages) != tuple(
        s.stage_id for s in SKILL_CRITERIA[skill_id]
    ):
        raise ValueError("Physical stage ladder does not match the standard skill.")
    evidence = {}
    for stage in result.stages:
        if stage.status == "not_reached":
            continue
        if stage.status == "not_applicable":
            raise ValueError(
                "Required standard stages cannot be declared inapplicable."
            )
        if stage.evidence_source is None:
            if stage.status == "unavailable" and not stage.measurements:
                continue
            raise ValueError("Standard stages require measured evidence provenance.")
        source, separator, method = stage.evidence_source.partition(":")
        if not separator:
            raise ValueError("Evidence provenance must include its measurement method.")
        evidence[stage.stage_id] = StageEvidence(
            stage.observation_window or "", source, stage.measurements, method
        )
    expected = evaluate_skill_stages(skill_id, evidence, env_index=result.env_index)
    if expected != result:
        raise ValueError("Physical verdict does not match the standard measurements.")
