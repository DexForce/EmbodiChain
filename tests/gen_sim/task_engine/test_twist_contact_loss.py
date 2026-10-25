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

"""Safe contact-loss retries must not manufacture coarse-turn evidence."""

from __future__ import annotations

import math
from types import SimpleNamespace as NS

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import (
    control_evidence,
    twist_runtime,
)
from embodichain.gen_sim.task_engine._task_program.twist_runtime import (
    TWIST_CANDIDATE_FALLBACKS,
    TwistAcceptancePort,
    _TwistCandidateState,
    _candidate_contact_decision,
)


def _productive_grasp():
    state = _TwistCandidateState(
        order=(5, 9),
        selected_index=5,
        selected_roll=math.pi / 2,
        candidates=((5, math.pi / 2, 0.005), (9, math.pi / 2, 0.008)),
        contact_path=math.radians(5),
    )
    assert _candidate_contact_decision(
        state,
        target_contact=True,
        already_reached=False,
        progress=True,
        recovery_safe=True,
    ) == (True, False)
    assert state.locked
    return state


def test_later_contact_loss_can_use_one_unused_same_axis_candidate():
    state = _productive_grasp()
    path = state.contact_path
    assert _candidate_contact_decision(
        state,
        target_contact=False,
        already_reached=False,
        progress=False,
        recovery_safe=True,
    ) == (True, True)
    assert state.forced_index == 9 and state.fallback_count == 1
    assert state.reserved_indices == (5, 9)
    assert state.contact_path == path and not state.goal_complete


@pytest.mark.parametrize(
    "reason", ["unreleased", "budget_exhausted", "no_unused_candidate"]
)
def test_contact_loss_preserves_recovery_safety_and_budget(reason):
    state = _productive_grasp()
    if reason == "budget_exhausted":
        state.fallback_count = TWIST_CANDIDATE_FALLBACKS
    if reason == "no_unused_candidate":
        state.reserved_indices = (5, 9)
    assert _candidate_contact_decision(
        state,
        target_contact=False,
        already_reached=False,
        progress=False,
        recovery_safe=reason != "unreleased",
    ) == (False, False)
    assert not state.goal_complete


def test_continued_contact_retains_the_productive_candidate():
    state = _productive_grasp()
    assert _candidate_contact_decision(
        state,
        target_contact=True,
        already_reached=False,
        progress=True,
        recovery_safe=True,
    ) == (True, False)
    assert state.forced_index == 5 and state.fallback_count == 0


def test_completed_goal_does_not_reserve_a_new_candidate():
    state = _productive_grasp()
    state.goal_complete = True
    assert _candidate_contact_decision(
        state,
        target_contact=False,
        already_reached=True,
        progress=False,
        recovery_safe=True,
    ) == (True, False)
    assert state.fallback_count == 0


@pytest.mark.parametrize("stable", [True, False])
def test_real_post_policy_cursor_is_retained_without_claiming_turn_completion(
    monkeypatch, stable
):
    state = _productive_grasp()
    path = state.contact_path
    route = NS(
        binding=NS(object_id="knob", target_qpos=2.0, limits=(-2.5, 2.5)),
        preset=lambda name: f"test.{name}",
    )
    samples = [NS(valid=True, target_contact=False) for _ in range(20)]
    sensor = NS(
        samples=samples,
        trace=[{"parent_contact": False} for _ in samples],
        art=NS(
            get_qpos=lambda: torch.tensor([[-0.18]]),
            get_qvel=lambda: torch.tensor([[0.0 if stable else 0.1]]),
        ),
        joint_index=0,
        _native_sleeping=lambda: False,
        _sim=NS(sim_config=NS(physics_dt=0.01)),
    )
    port = TwistAcceptancePort.__new__(TwistAcceptancePort)
    port.route, port.sensor, port.dt = route, sensor, 0.04
    port.robot = NS(get_qpos=lambda **_: torch.zeros((1, 7)))
    port._candidate_state, port._chunk_cursor = state, 0
    port.results, port.metadata = {}, {}
    policy = NS(
        cfg=NS(kind="wait_stable", preset=route.preset("chunk")),
        entity=NS(entity_id="knob"),
    )
    monkeypatch.setattr(
        twist_runtime,
        "evaluate_twist",
        lambda *_a, **_k: {
            "accepted": False,
            "reason": "coarse_contact_travel_insufficient",
            "contact_path": path,
        },
    )
    monkeypatch.setattr(
        control_evidence, "stamp_control_result", lambda result, *_: result
    )
    commands = list(
        port.actions(policy, segment=NS(), active_mask=torch.tensor([True]))
    )
    assert len(commands) == math.ceil(0.5 / port.dt)
    assert bool(port.post_policy_result(policy, segment=NS())[0]) is stable
    result = port.post_policy_metadata(policy, segment=NS())
    assert result["candidate_fallback"] is stable
    assert result["goal_complete"] is False and state.contact_path == path
    assert state.forced_index == (9 if stable else 5)


@pytest.mark.parametrize(
    "sleeping,velocity,moving,expected",
    [
        (False, 0.0, False, True),
        (False, 0.037794139, False, False),
        (True, 0.037794139, False, True),
        (True, 0.037794139, True, False),
        (True, float("nan"), False, False),
        (True, float("inf"), False, False),
    ],
)
def test_terminal_twist_keeps_raw_velocity_and_requires_native_rest(
    monkeypatch, sleeping, velocity, moving, expected
):
    route = NS(
        binding=NS(object_id="knob", target_qpos=2.0),
        preset=lambda name: f"test.{name}",
    )
    count = math.ceil(0.5 / 0.04)
    positions = iter([0.1 * i if moving else 0.1 for i in range(count + 1)])
    samples = [NS(valid=True, target_contact=False) for _ in range(20)]
    sensor = NS(
        samples=samples,
        trace=[{"parent_contact": False} for _ in samples],
        art=NS(
            get_qpos=lambda: torch.tensor([[next(positions)]]),
            get_qvel=lambda: torch.tensor([[velocity]]),
        ),
        joint_index=0,
        _native_sleeping=lambda: sleeping,
        _sim=NS(sim_config=NS(physics_dt=0.01)),
        other_qpos=torch.zeros((1, 1)),
    )
    port = TwistAcceptancePort.__new__(TwistAcceptancePort)
    port.route, port.sensor, port.dt = route, sensor, 0.04
    port.robot = NS(get_qpos=lambda **_: torch.zeros((1, 7)))
    port._candidate_state, port._chunk_cursor = _TwistCandidateState(), 0
    port._clearance = lambda: 0.05
    port.results, port.metadata = {}, {}
    policy = NS(
        cfg=NS(kind="wait_stable", preset=route.preset("turned")),
        entity=NS(entity_id="knob"),
    )
    monkeypatch.setattr(
        twist_runtime,
        "evaluate_twist",
        lambda *_a, **_k: {"accepted": True, "contact_path": math.radians(17)},
    )
    monkeypatch.setattr(
        control_evidence, "stamp_control_result", lambda result, *_: result
    )
    commands = list(
        port.actions(policy, segment=NS(calls=()), active_mask=torch.tensor([True]))
    )
    assert len(commands) == count
    result = port.post_policy_metadata(policy, segment=NS())
    assert result["accepted"] is expected and result["stable"] is expected
    assert result["native_sleeping"] == [sleeping] * count
    assert len(result["joint_velocities"]) == count
    if math.isfinite(velocity):
        assert result["joint_velocities"] == pytest.approx([velocity] * count)


@pytest.mark.parametrize("newton", [False, True])
def test_native_sleep_uses_default_actor_without_querying_newton(newton):
    def unsupported():
        pytest.fail("Unsupported Scene sleep query must not be used.")

    native = NS(is_sleeping=unsupported if newton else lambda: True)
    sensor = twist_runtime.TwistContactSensor.__new__(twist_runtime.TwistContactSensor)
    sensor.art = NS(
        body_data=NS(
            is_newton_backend=newton,
            entities=[NS(is_sleeping=unsupported, render_articulation=native)],
        )
    )
    assert sensor._native_sleeping() is (not newton)
