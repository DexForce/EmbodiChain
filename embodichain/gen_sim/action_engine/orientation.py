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

"""Compile task-facing orientation goals into a small runtime contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

__all__ = [
    "AlignAxisConstraint",
    "MatchRotationConstraint",
    "OrientationConstraint",
    "compile_orientation_constraint",
]

_LONG_AXES = frozenset({"long", "long_axis", "longest"})
_SCOPES = frozenset({"terminal"})


@dataclass(frozen=True)
class AlignAxisConstraint:
    """Require one local object axis to align with a target axis."""

    local_axis: str
    target_axis: str = "world_up"
    directed: bool = True
    tolerance: float | None = None
    scope: str = "terminal"


@dataclass(frozen=True)
class MatchRotationConstraint:
    """Require a complete object rotation relative to a captured reference."""

    reference: str
    equivalence: str = "none"
    tolerance: float | None = None
    scope: str = "terminal"


OrientationTerm = AlignAxisConstraint | MatchRotationConstraint


@dataclass(frozen=True)
class OrientationConstraint:
    """Canonical hard constraints plus a separate planning preference."""

    terms: tuple[OrientationTerm, ...]
    planning_preference: str = "minimize_rotation_from_current"

    @property
    def requires_reference(self) -> bool:
        """Return whether execution must capture a step-start rotation."""
        return any(
            isinstance(term, MatchRotationConstraint) and term.reference == "step_start"
            for term in self.terms
        )

    @property
    def allows_yaw_search(self) -> bool:
        """Return whether hard constraints leave world-Z yaw unconstrained."""
        return all(
            isinstance(term, AlignAxisConstraint) and term.target_axis == "world_up"
            for term in self.terms
        )

    @property
    def requires_upright_axis_alignment(self) -> bool:
        """Return whether a hard term requires alignment with world up."""
        return bool(self.terms) and all(
            isinstance(term, AlignAxisConstraint) and term.target_axis == "world_up"
            for term in self.terms
        )


def compile_orientation_constraint(
    goal: Mapping[str, Any],
) -> OrientationConstraint:
    """Compile legacy goal enums or a composable serialized constraint.

    Existing persisted graphs continue to carry explicit ``orientation_goal``
    values. New tasks may omit the field, which intentionally means no hard
    orientation constraint while retaining a minimum-rotation preference.
    """
    serialized = goal.get("orientation_constraint")
    if serialized is not None:
        return _compile_serialized(serialized)

    orientation_goal = str(goal.get("orientation_goal", "none"))
    if orientation_goal == "none":
        terms: tuple[OrientationTerm, ...] = ()
    elif orientation_goal == "preserve":
        terms = (MatchRotationConstraint(reference="step_start"),)
    elif orientation_goal == "upright":
        local_axis = str(goal.get("upright_local_axis", "long_axis"))
        if local_axis == "auto":
            local_axis = "long_axis"
        directed = goal.get(
            "orientation_directed", local_axis.lower() not in _LONG_AXES
        )
        if not isinstance(directed, bool):
            raise ValueError("orientation_directed must be a boolean.")
        terms = (
            AlignAxisConstraint(
                local_axis=local_axis,
                target_axis="world_up",
                directed=directed,
            ),
        )
    elif orientation_goal == "lay_flat":
        terms = (
            AlignAxisConstraint(
                local_axis="short_axis",
                target_axis="world_up",
                directed=False,
            ),
        )
    elif orientation_goal == "axis_align":
        # axis_align explicitly requests a horizontal heading. Its established
        # target-pose contract remains strict until it is replaced by a typed
        # non-world-up axis term.
        terms = (MatchRotationConstraint(reference="target_pose"),)
    else:
        raise ValueError(f"Unsupported orientation_goal {orientation_goal!r}.")
    return OrientationConstraint(terms=terms)


def _compile_serialized(value: Any) -> OrientationConstraint:
    if not isinstance(value, Mapping):
        raise ValueError("orientation_constraint must be a mapping.")
    unknown = set(value) - {"terms", "planning_preference"}
    if unknown:
        raise ValueError(
            "orientation_constraint contains unsupported fields: " f"{sorted(unknown)}."
        )
    raw_terms = value.get("terms", ())
    if not isinstance(raw_terms, Sequence) or isinstance(
        raw_terms, (str, bytes, bytearray)
    ):
        raise ValueError("orientation_constraint.terms must be a list.")
    terms = tuple(_compile_term(item, index) for index, item in enumerate(raw_terms))
    preference = str(value.get("planning_preference", "minimize_rotation_from_current"))
    if preference not in {"minimize_rotation_from_current", "none"}:
        raise ValueError(
            "orientation_constraint.planning_preference must be "
            "'minimize_rotation_from_current' or 'none'."
        )
    return OrientationConstraint(terms=terms, planning_preference=preference)


def _compile_term(value: Any, index: int) -> OrientationTerm:
    context = f"orientation_constraint.terms[{index}]"
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping.")
    kind = str(value.get("type", ""))
    scope = str(value.get("scope", "terminal"))
    if scope not in _SCOPES:
        raise ValueError(
            f"{context}.scope {scope!r} is unsupported by the current runtime."
        )
    if kind == "align_axis":
        unknown = set(value) - {
            "type",
            "local_axis",
            "target_axis",
            "directed",
            "tolerance",
            "scope",
        }
        if unknown:
            raise ValueError(
                f"{context} contains unsupported fields: {sorted(unknown)}."
            )
        local_axis = str(value.get("local_axis", ""))
        if local_axis not in {"x", "y", "z", "long_axis", "short_axis"}:
            raise ValueError(f"{context}.local_axis {local_axis!r} is unsupported.")
        target_axis = str(value.get("target_axis", "world_up"))
        if target_axis != "world_up":
            raise ValueError(f"{context}.target_axis {target_axis!r} is unsupported.")
        directed = value.get("directed", True)
        if not isinstance(directed, bool):
            raise ValueError(f"{context}.directed must be a boolean.")
        return AlignAxisConstraint(
            local_axis=local_axis,
            target_axis=target_axis,
            directed=directed,
            tolerance=_optional_tolerance(value, context),
            scope=scope,
        )
    if kind == "match_rotation":
        unknown = set(value) - {
            "type",
            "reference",
            "equivalence",
            "tolerance",
            "scope",
        }
        if unknown:
            raise ValueError(
                f"{context} contains unsupported fields: {sorted(unknown)}."
            )
        reference = str(value.get("reference", ""))
        if reference not in {"step_start", "target_pose"}:
            raise ValueError(f"{context}.reference {reference!r} is unsupported.")
        equivalence = str(value.get("equivalence", "none"))
        if equivalence != "none":
            raise ValueError(f"{context}.equivalence {equivalence!r} is unsupported.")
        return MatchRotationConstraint(
            reference=reference,
            equivalence=equivalence,
            tolerance=_optional_tolerance(value, context),
            scope=scope,
        )
    raise ValueError(f"{context}.type {kind!r} is unsupported.")


def _optional_tolerance(value: Mapping[str, Any], context: str) -> float | None:
    raw = value.get("tolerance")
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{context}.tolerance must be a finite positive number.")
    tolerance = float(raw)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(f"{context}.tolerance must be a finite positive number.")
    return tolerance
