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

"""Frozen Twist transitions and explicit planning/measurement ownership."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import math

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    ActionPlan,
    EndpointCommand,
    JointPositionPayload,
    JointPositionTarget,
    PlannerDiagnostics,
    PlanningFailure,
    PlanningRecoveryOutcome,
    RecoveryPolicy,
    RuntimeCommandFrame,
    TimedCommandSequence,
    TimedTrajectory,
    TrackingPolicy,
)
from embodichain.lab.sim.atomic_actions.primitives.twist_recovery import (
    TwistMeasuredFeedback,
    TwistPlanningOutcome,
    TwistPlanningRecovery,
    TwistRecoveryCursor,
    TwistRecoveryScope,
    TwistRecoverySnapshot,
    _contact_decision,
    required_twist_occurrences,
)


def _scope(**overrides):
    return TwistRecoveryScope(
        **{
            "program_fingerprint": "program-digest",
            "operation_id": "turn-a",
            "episode_id": "episode-1",
            "env_id": 0,
            **overrides,
        }
    )


def _proposal(cursor, *, invocation="chunk-1", plan_id="plan-1", **changes):
    current = cursor.snapshot()
    return TwistPlanningOutcome(
        "twist",
        invocation,
        0,
        plan_id,
        replace(
            current,
            version=current.version + 1,
            selected_index=0,
            selected_roll=0.0,
            selected_bite_depth=0.005 / 3,
            order=(0, 4),
            candidates=((0, 0.0, -0.005 * 2 / 3), (4, 0.0, -0.005 / 3)),
            **changes,
        ),
    )


def _plan(outcome, *, success=True):
    env_ids = torch.tensor([outcome.next_state.scope.env_id])
    active = torch.tensor([success])
    qpos = torch.zeros(1, 1)
    command = EndpointCommand(
        JointPositionTarget("hand", (0,)), JointPositionPayload(qpos)
    )
    frame = RuntimeCommandFrame((command,), active, env_ids, torch.tensor([0.04]))
    return ActionPlan(
        skill_id="twist",
        plan_success=active,
        commands=TimedCommandSequence((frame,), env_ids),
        recovery_policy=RecoveryPolicy(planning_recovery=TwistPlanningRecovery()),
        tracking_policy=TrackingPolicy.timed(),
        planned_scene_version=0,
        planned_collision_world_revision=(0,),
        diagnostics=PlannerDiagnostics(
            "fixture", failure=None if success else PlanningFailure("no_path")
        ),
        joint_trajectory=TimedTrajectory.from_uniform_step(
            qpos[:, None], env_ids=env_ids, step_dt=0.04
        ),
        invocation_id=outcome.invocation_id,
        invocation_revision=outcome.invocation_revision,
        planning_recovery_outcome=outcome,
    )


def _feedback(cursor, outcome, **changes):
    state = cursor.snapshot()
    return TwistMeasuredFeedback(
        **{
            "scope": state.scope,
            "cursor_id": state.cursor_id,
            "expected_version": state.version,
            "invocation_id": outcome.invocation_id,
            "invocation_revision": outcome.invocation_revision,
            "plan_id": outcome.plan_id,
            "timestamp": 1.0,
            "valid": True,
            "target_contact": False,
            "already_reached": False,
            "recovery_safe": True,
            "contact_path": 0.0,
            "goal_complete": False,
            **changes,
        }
    )


_REFERENCE = json.loads(
    (
        Path(__file__).with_name("data") / "twist_feedback_source_reference.json"
    ).read_text()
)


@pytest.mark.parametrize("case", _REFERENCE["cases"], ids=lambda case: case["name"])
def test_contact_transition_matches_pinned_f1(case):
    state = TwistRecoverySnapshot(_scope(), "fixture-cursor", **case["initial"])
    for step in case["steps"]:
        state, accepted, fallback = _contact_decision(
            state, TwistPlanningRecovery(), **step["input"]
        )
        assert [accepted, fallback] == step["result"]
        actual = {name: asdict(state)[name] for name in step["state"]}
        # JSON normalizes the immutable tuple containers while retaining values.
        assert json.loads(json.dumps(actual)) == step["state"]


@pytest.mark.parametrize(
    "angle,count",
    [(0, 8), (-0.0, 8), (1e-300, 32), (math.pi / 2, 32), (-math.radians(145), 32)],
)
def test_twist_occurrence_budget_preserves_zero_and_source_boundary(angle, count):
    rule = TwistPlanningRecovery()
    assert required_twist_occurrences(angle, rule) == count
    assert (
        rule.occurrence_angle_budget,
        rule.minimum_occurrences,
        rule.maximum_occurrences,
        rule.fallback_occurrences,
    ) == (math.radians(5), 8, 32, 3)
    assert rule.maximum_fallbacks == 2


@pytest.mark.parametrize(
    "angle",
    [
        True,
        "1",
        math.nan,
        math.inf,
        math.radians(145) + 1e-9,
        -math.radians(145) - 1e-9,
    ],
)
def test_twist_occurrence_budget_rejects_invalid_or_excessive_turns(angle):
    with pytest.raises(ValueError):
        required_twist_occurrences(angle, TwistPlanningRecovery())


def test_failed_planning_commits_logical_selection_but_requires_adoption_for_feedback():
    cursor = TwistRecoveryCursor(_scope())
    first = _proposal(cursor)
    failed = _plan(first, success=False)
    assert failed.snapshot().planning_recovery_outcome == first
    assert (
        failed.snapshot().planning_recovery_outcome
        is not failed.planning_recovery_outcome
    )
    state = cursor.commit_planning(failed.planning_recovery_outcome)
    assert state.selected_index == 0 and state.version == 1
    assert state.fallback_count == 0 and state.reserved_indices == ()
    with pytest.raises(ValueError, match="successful"):
        cursor.adopt_plan(failed)
    with pytest.raises(ValueError, match="adopted plan"):
        cursor.commit_measurement(_feedback(cursor, first))

    second = _proposal(cursor, invocation="chunk-2", plan_id="plan-2")
    cursor.commit_planning(second)
    accepted = _plan(second)
    with pytest.raises(ValueError, match="typed joint plan"):
        cursor.adopt_plan(replace(accepted, joint_trajectory=None))
    cursor.adopt_plan(accepted)
    before = cursor.snapshot()
    assert before.fallback_count == 0
    result = cursor.commit_measurement(_feedback(cursor, second))
    assert result.accepted and result.fallback_used and not result.axis_switch
    assert result.snapshot.forced_index == 4 and result.snapshot.fallback_count == 1
    assert result.snapshot.reserved_indices == (0, 4)
    assert not result.snapshot.goal_complete
    with pytest.raises(ValueError, match="stale cursor version"):
        cursor.commit_measurement(
            _feedback(cursor, second, expected_version=before.version)
        )


def test_operation_episode_environment_and_preview_isolation():
    real = TwistRecoveryCursor(_scope())
    outcome = _proposal(real)
    preview = real.fork_preview()
    preview_outcome = _proposal(preview, invocation="preview", plan_id="preview-plan")
    preview.commit_planning(preview_outcome)
    assert real.snapshot().selected_index is None
    for other in (
        real,
        TwistRecoveryCursor(_scope(operation_id="turn-b")),
        TwistRecoveryCursor(_scope(episode_id="episode-2")),
        TwistRecoveryCursor(_scope(env_id=1)),
    ):
        with pytest.raises(ValueError, match="another cursor scope"):
            other.commit_planning(preview_outcome)
    real.commit_planning(outcome)
    with pytest.raises(ValueError, match="stale cursor version"):
        real.commit_planning(outcome)
    real.reset_ready()
    assert real.snapshot().selected_index is None and real.snapshot().version == 2
    real.close()
    with pytest.raises(RuntimeError, match="ended"):
        real.snapshot()


def test_planning_cannot_spend_physical_budget_or_claim_completion():
    cursor = TwistRecoveryCursor(_scope())
    for change in (
        {"fallback_count": 1},
        {"goal_complete": True},
        {"contact_path": 0.2},
    ):
        with pytest.raises(ValueError, match="cannot change measured"):
            cursor.commit_planning(_proposal(cursor, **change))
    assert cursor.snapshot().version == 0


def test_feedback_rejects_other_plan_revision_or_operation_and_retains_freshness():
    cursor = TwistRecoveryCursor(_scope())
    outcome = _proposal(cursor)
    cursor.commit_planning(outcome)
    cursor.adopt_plan(_plan(outcome))
    for change in (
        {"plan_id": "old-plan"},
        {"invocation_revision": 1},
        {"scope": _scope(operation_id="other")},
    ):
        with pytest.raises(ValueError):
            cursor.commit_measurement(_feedback(cursor, outcome, **change))
    result = cursor.commit_measurement(
        _feedback(cursor, outcome, valid=False, contact_path=0.1)
    )
    assert not result.accepted and not result.fallback_used
    assert (
        result.snapshot.contact_path == 0.1 and result.snapshot.reserved_indices == ()
    )
    next_outcome = _proposal(cursor, invocation="chunk-2", plan_id="next-plan")
    cursor.commit_planning(next_outcome)
    cursor.adopt_plan(_plan(next_outcome))
    with pytest.raises(ValueError, match="follow the previous observation"):
        cursor.commit_measurement(_feedback(cursor, next_outcome, timestamp=0.5))


def test_planning_outcomes_reject_mutable_or_live_values_and_wrong_plan_identity():
    @dataclass(frozen=True, slots=True)
    class InvalidOutcome(PlanningRecoveryOutcome):
        value: object

    for value in ([1], {"cursor": 1}, lambda: None):
        with pytest.raises(TypeError, match="immutable"):
            InvalidOutcome("twist", "chunk", 0, "plan", value)
    plan = _plan(_proposal(TwistRecoveryCursor(_scope())))
    with pytest.raises(ValueError, match="does not identify this plan"):
        replace(plan, invocation_id="different")


@pytest.mark.parametrize(
    "case", ["progress", "fallback", "unsafe", "invalid", "calibrated"]
)
def test_measured_chunk_matches_frozen_source_receipt_and_transition(case):
    from embodichain.lab.sim.atomic_actions.articulation_evidence import (
        JointContactSample,
        TwistEventTarget,
    )
    from embodichain.lab.sim.atomic_actions.twist_feedback import (
        commit_observed_twist_chunk,
    )

    reference = json.loads(
        Path(__file__)
        .with_name("data")
        .joinpath("twist_feedback_source.json")
        .read_text()
    )
    expected = next(row for row in reference["cases"] if row["case"] == case)
    cursor = TwistRecoveryCursor(_scope())
    outcome = _proposal(cursor)
    cursor.commit_planning(outcome)
    cursor.adopt_plan(_plan(outcome))
    final = float(torch.tensor(0.3 if case in ("progress", "calibrated") else 0.0))
    contacted = case in ("progress", "calibrated")
    target = TwistEventTarget(
        1.0 if case == "calibrated" else 0.3,
        (-2.0, 2.0),
        success_criterion=(
            "calibrated_setting"
            if case == "calibrated"
            else "contact_backed_coarse_turn"
        ),
    )
    samples = [
        JointContactSample(0, 0, "prepare", False),
        JointContactSample(0.004, 0, "twist", contacted),
        JointContactSample(0.008, final, "twist", contacted),
    ]
    samples += [
        JointContactSample(i * 0.004, final, "twist", False) for i in range(3, 133)
    ]
    if case == "invalid":
        samples[2] = JointContactSample(0.008, final, "twist", None, False)
    if case == "progress":
        with pytest.raises(ValueError, match="nonempty measured"):
            commit_observed_twist_chunk(
                cursor,
                outcome,
                target,
                samples,
                [False] * len(samples),
                chunk_start=1,
                observed_qpos=final,
                stable_positions=[],
                stable_velocities=[],
                physics_dt=0.004,
                feedback_timestamp=0.6,
                window_available=True,
            )
        assert cursor.snapshot().version == outcome.next_state.version
    result = commit_observed_twist_chunk(
        cursor,
        outcome,
        target,
        samples,
        [False] * len(samples),
        chunk_start=1,
        observed_qpos=final,
        stable_positions=[final] * 13,
        stable_velocities=[0.1 if case == "unsafe" else 0.0] * 13,
        physics_dt=0.004,
        feedback_timestamp=0.6,
        window_available=case != "invalid",
    )
    assert {
        key: value for key, value in result.items() if key != "observation_available"
    } == expected["receipt"]
    actual = json.loads(json.dumps(asdict(cursor.snapshot())))
    assert {key: actual[key] for key in expected["cursor"]} == expected["cursor"]
    assert result["observation_available"] is (case != "invalid")


@pytest.mark.parametrize(
    "extra", [{"max_candidates": 13}, {"factory": "module.callback"}]
)
def test_configured_twist_recovery_remains_closed_and_immutable(extra):
    from embodichain.lab.task_program.integrations.configured import (
        _decode_recovery_policy,
        _decode_action_options,
    )

    declaration = {"kind": "twist_bounded", "revision": 1}
    config = {
        "max_replans": 0,
        "max_action_retries": 0,
        "planning_recovery": declaration,
    }
    policy = _decode_recovery_policy(config, path="recovery")
    assert type(policy.planning_recovery) is TwistPlanningRecovery
    options = _decode_action_options(
        {
            "kind": "twist",
            "contact_bite_depth": 0.006,
            "contact_roll": None,
            "rotation_enabled": False,
        },
        path="options",
    )
    assert options.contact_bite_depth == 0.006 and options.contact_roll is None
    assert options.rotation_enabled is False
    with pytest.raises(ValueError, match="unsupported"):
        _decode_recovery_policy(
            {**config, "planning_recovery": {**declaration, **extra}}, path="recovery"
        )
