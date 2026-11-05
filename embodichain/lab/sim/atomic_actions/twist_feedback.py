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

"""Measured Twist chunk validation and bounded physical recovery transitions."""

from __future__ import annotations

from collections.abc import Sequence
import math

from .articulation_evidence import (
    JointContactSample,
    TwistEventTarget,
    evaluate_twist_event,
)
from .primitives.twist_recovery import (
    TwistMeasuredFeedback,
    TwistPlanningOutcome,
    TwistRecoveryCursor,
)

__all__ = ["commit_observed_twist_chunk"]


def commit_observed_twist_chunk(
    cursor: TwistRecoveryCursor,
    outcome: TwistPlanningOutcome,
    target: TwistEventTarget,
    all_samples: Sequence[JointContactSample],
    parent_contacts: Sequence[bool | None],
    *,
    chunk_start: int,
    observed_qpos: float,
    stable_positions: Sequence[float],
    stable_velocities: Sequence[float],
    physics_dt: float,
    feedback_timestamp: float,
    window_available: bool,
) -> dict[str, object]:
    """Validate measured chunk facts and advance the adopted recovery cursor.

    Args:
        cursor: Exact operation cursor that adopted the installed plan.
        outcome: Immutable adopted plan identity.
        target: Declared physical event target and criterion.
        all_samples: Complete operation samples, including released onset.
        parent_contacts: Aligned measured parent/hand contact flags.
        chunk_start: First sample belonging to this execution occurrence.
        observed_qpos: Current physical joint read at the post-policy boundary.
        stable_positions: Joint readings from the half-second control-grid hold.
        stable_velocities: Corresponding measured angular velocities.
        physics_dt: Actual contact sensor physics-substep duration.
        feedback_timestamp: Same execution clock domain as planning observations.
        window_available: Host qualification of source scope, cadence and capacity.

    Returns:
        Source-compatible chunk receipt. Acceptance authorizes continuation or a
        bounded fallback; it is not task completion. Missing evidence never
        qualifies a successful chunk. The cursor verifies exact adoption and time.
    """
    all_samples, parent_contacts = tuple(all_samples), tuple(parent_contacts)
    if (
        type(chunk_start) is not int
        or not 0 <= chunk_start <= len(all_samples)
        or len(parent_contacts) != len(all_samples)
    ):
        raise ValueError(
            "Twist chunk samples require an exact aligned operation prefix."
        )
    if isinstance(physics_dt, bool) or not math.isfinite(physics_dt) or physics_dt <= 0:
        raise ValueError("Twist feedback requires positive physics cadence.")
    if type(window_available) is not bool:
        raise TypeError("Twist window availability must be explicit.")
    state = cursor.snapshot()
    samples = all_samples[chunk_start:]
    positions, velocities = tuple(stable_positions), tuple(stable_velocities)
    if not positions or len(positions) != len(velocities):
        raise ValueError(
            "Twist stabilization requires matching nonempty measured sequences."
        )
    stable = bool(
        positions
        and len(positions) == len(velocities)
        and max(positions) - min(positions) <= math.radians(1.0)
        and all(
            math.isfinite(value) and abs(value) <= math.radians(2.0)
            for value in velocities
        )
    )
    release_count = max(1, math.ceil(0.15 / physics_dt))
    recent = all_samples[-release_count:]
    released = bool(recent) and all(
        sample.valid and sample.target_contact is False for sample in recent
    )
    target_contacts = [sample.target_contact for sample in samples]
    chunk_parent_contacts = [bool(value) for value in parent_contacts[chunk_start:]]
    parent_fraction = (
        sum(chunk_parent_contacts) / len(chunk_parent_contacts)
        if chunk_parent_contacts
        else 1.0
    )
    target_error = abs(observed_qpos - target.target_qpos)
    already_reached = target_error <= target.angle_tolerance
    available = (
        window_available and bool(all_samples) and all_samples[-1].qpos == observed_qpos
    )
    valid_chunk = bool(
        available
        and samples
        and all(sample.valid for sample in samples)
        and all(sample.target_contact is not None for sample in samples)
        and math.isfinite(observed_qpos)
        and target.limits[0] - 0.02 <= observed_qpos <= target.limits[1] + 0.02
    )
    goal_check = evaluate_twist_event(target, all_samples, final_qpos=observed_qpos)
    contact_path = goal_check.get("contact_path", 0.0)
    progress = max(0.0, contact_path - state.contact_path)
    recovery_safe = bool(stable and released and len(recent) == release_count)
    valid_chunk = valid_chunk and goal_check["reason"] not in {
        "invalid_observation",
        "joint_out_of_bounds",
        "unstable_initial_state",
    }
    decision = cursor.commit_measurement(
        TwistMeasuredFeedback(
            state.scope,
            state.cursor_id,
            state.version,
            outcome.invocation_id,
            outcome.invocation_revision,
            outcome.plan_id,
            feedback_timestamp,
            valid_chunk,
            bool(any(target_contacts)),
            already_reached,
            recovery_safe,
            contact_path,
            bool(goal_check["accepted"] and available),
        )
    )
    next_state = decision.snapshot
    good, fallback = decision.accepted, decision.fallback_used
    result = {
        "accepted": good,
        "phase": "bounded_chunk",
        "reason": (
            "candidate_fallback"
            if fallback
            else (
                "chunk_observed"
                if good
                else (
                    "invalid_observation"
                    if not valid_chunk
                    else (
                        "recovery_boundary_not_safe"
                        if not recovery_safe
                        else (
                            "contact_without_progress"
                            if any(target_contacts)
                            else "target_contact_not_observed"
                        )
                    )
                )
            )
        ),
        "sample_count": len(samples),
        "target_contact": bool(any(target_contacts)),
        "parent_contact_fraction": parent_fraction,
        "observed_qpos": observed_qpos,
        "target_error": target_error,
        "already_reached": already_reached,
        "parent_contact_fraction_limit": 0.25,
        "parent_contact_warning": parent_fraction > 0.25,
        "candidate_selected": next_state.selected_index,
        "candidate_bite_depth": next_state.selected_bite_depth,
        "candidate_roll": next_state.selected_roll,
        "candidate_fallback": fallback,
        "candidate_axis_switch": decision.axis_switch,
        "candidate_next": next_state.forced_index if fallback else None,
        "candidate_reserved_indices": list(next_state.reserved_indices),
        "candidate_fallback_count": next_state.fallback_count,
        "candidate_fallback_limit": 2,
        "candidate_order": list(next_state.order),
        "candidate_attempt_count": next_state.fallback_count + 1,
        "candidate_stalled_chunks": next_state.stalled_chunks,
        "candidate_stalled_chunk_limit": 2,
        "chunk_contact_progress": progress,
        "contact_released": released,
        "stable": stable,
        "recovery_safe": recovery_safe,
        "candidate_exhausted": bool(
            valid_chunk
            and not good
            and not next_state.locked
            and next_state.order
            and recovery_safe
        ),
        "coarse_turn_threshold": target.coarse_turn_threshold,
        "cumulative_contact_path": next_state.contact_path,
        "coarse_turn_reached": next_state.goal_complete,
        "early_stop": next_state.goal_complete,
        "goal_complete": next_state.goal_complete,
        "observation_available": available,
    }
    return result
