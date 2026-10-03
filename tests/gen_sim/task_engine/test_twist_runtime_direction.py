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
from __future__ import annotations

import math
from dataclasses import make_dataclass
from types import SimpleNamespace

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program.press import PressSample
from embodichain.lab.sim.atomic_actions import TwistOptions
from embodichain.gen_sim.task_engine._task_program.twist_runtime import (
    GenSimTwist,
    TwistAcceptancePort,
    _TwistCandidateState,
    _bounded_twist_qpos,
    _initial_twist_state_stable,
    evaluate_twist,
)

TARGET_ANGLE = math.pi / 2.0
CHUNK_TRAVEL = math.radians(6.0)
CHUNK_COUNT = 5


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_twist_boundary_roundoff_matches_native_joint_dtype(dtype):
    limit = -math.pi / 2
    assert _bounded_twist_qpos(
        torch.tensor(limit, dtype=dtype), (limit, 0.0)
    ) == pytest.approx(limit)
    assert _bounded_twist_qpos(torch.tensor(0.0, dtype=dtype), (limit, 0.0)) == 0.0
    assert (
        _bounded_twist_qpos(torch.tensor(limit - 0.001, dtype=dtype), (limit, 0.0))
        is None
    )
    assert _bounded_twist_qpos(torch.tensor(0.001, dtype=dtype), (limit, 0.0)) is None


def test_twist_reads_latest_lowerer_measurement_before_symbolic_state():
    goal = SimpleNamespace(
        semantics=SimpleNamespace(geometry={"gen_sim_twist_measured_qpos": -1.5})
    )
    request = SimpleNamespace(goal=goal)
    assert GenSimTwist._current_qpos(request, object()) == -1.5


@pytest.mark.parametrize(
    "final,velocity,clearance,released,valid,complete",
    [
        (-TARGET_ANGLE, 0.0, 0.1, True, True, True),
        (-0.5, 0.0, 0.1, True, True, False),
        (-TARGET_ANGLE, 1.0, 0.1, True, True, False),
        (-TARGET_ANGLE, 0.0, 0.01, True, True, False),
        (-TARGET_ANGLE, 0.0, 0.1, False, True, False),
        (-TARGET_ANGLE, 0.0, 0.1, True, False, False),
    ],
)
def test_goal_hold_requires_full_contact_release_acceptance(
    final, velocity, clearance, released, valid, complete
):
    route = _route(-1.0)
    route.binding.object_id = "control"
    route.preset = lambda phase: phase
    samples = _samples([(0.0, False), (0.0, True), (-TARGET_ANGLE, True)])
    samples.extend(
        PressSample(3.0 + index * 0.01, -TARGET_ANGLE, "cleanup", not released, valid)
        for index in range(20)
    )
    sensor = SimpleNamespace(
        route=route,
        art=SimpleNamespace(
            get_qpos=lambda: torch.tensor([[final]]),
            get_qvel=lambda: torch.tensor([[velocity]]),
        ),
        joint_index=0,
        samples=samples,
        trace=[{"parent_contact": False} for _ in samples],
        _sim=SimpleNamespace(sim_config=SimpleNamespace(physics_dt=0.01)),
    )
    port = object.__new__(TwistAcceptancePort)
    port.route, port.sensor, port.dt = route, sensor, 0.04
    port.robot = SimpleNamespace(get_qpos=lambda: torch.zeros(1, 2))
    port._chunk_cursor = 0
    port._candidate_state = _TwistCandidateState()
    port._clearance = lambda: clearance
    port.results, port.metadata = {}, {}
    policy = SimpleNamespace(
        cfg=SimpleNamespace(preset="chunk", kind="wait_stable"),
        entity=SimpleNamespace(entity_id="control"),
    )
    list(
        port.actions(
            policy,
            segment=SimpleNamespace(calls=[]),
            active_mask=torch.tensor([True]),
        )
    )
    assert port._candidate_state.goal_complete is complete
    if complete:
        from embodichain.gen_sim.task_engine._task_program import twist_runtime

        request_type = make_dataclass(
            "HoldRequest", [("goal", object), ("skill_options", object)]
        )
        request = request_type(
            SimpleNamespace(
                semantics=SimpleNamespace(
                    geometry={
                        "gen_sim_twist_angle": 0.0,
                        "gen_sim_twist_measured_qpos": -TARGET_ANGLE,
                        "gen_sim_twist_target_qpos": -TARGET_ANGLE,
                        "gen_sim_twist_axis_sign": -1.0,
                    }
                )
            ),
            TwistOptions(),
        )
        context = SimpleNamespace(
            batch_size=1,
            env_ids=torch.tensor([0]),
            robot=SimpleNamespace(qpos=torch.zeros(1, 2)),
            require_control_dt=lambda: 0.04,
        )
        action = SimpleNamespace(
            device=torch.device("cpu"),
            _current_qpos=GenSimTwist._current_qpos,
            build_plan=lambda request, context, **kwargs: kwargs,
        )
        token = twist_runtime._CANDIDATE_STATE.set(port._candidate_state)
        try:
            plan = GenSimTwist._plan(action, request, context)
        finally:
            twist_runtime._CANDIDATE_STATE.reset(token)
        assert plan["segment_lengths"] == {"approach": 1}
        assert plan["trajectory"].positions.shape == (1, 1, 2)
        assert not plan["expected_effects"].articulation_joint_updates
        assert not plan["expected_effects"].held_object_updates
    if not valid:
        assert sensor.acceptance["accepted"] is False


def test_twist_candidate_score_compares_arena_frames(monkeypatch) -> None:
    from embodichain.gen_sim.task_engine._task_program import twist_runtime

    root = torch.eye(4).unsqueeze(0)
    root[0, :3, 3] = torch.tensor([1.0, 2.0, 0.7])
    pads = torch.eye(4).expand(1, 2, 4, 4).clone()
    pads[0, 0, 0, 3] = -0.02
    pads[0, 1, 0, 3] = 0.02
    names = ["left_inner_finger_pad", "left_right_inner_finger_pad"]
    robot = SimpleNamespace(
        link_names=names,
        joint_names=["joint"],
        compute_fk=lambda **kwargs: pads,
        get_local_pose=lambda **kwargs: root,
        get_link_vert_face=lambda name: (torch.zeros(1, 3), None),
    )
    monkeypatch.setattr(twist_runtime, "resolve_pose_goal", lambda *a, **kw: root)
    monkeypatch.setattr(twist_runtime, "resolve_pose_target", lambda *a, **kw: root)
    geometry = {
        "gen_sim_twist_arm": "left",
        "gen_sim_twist_target_points": [[-0.02, 0, 0], [0.02, 0, 0]],
        "gen_sim_twist_parent_points": [[1.0, 0, 0]],
        "gen_sim_twist_target_to_parent": torch.eye(4).tolist(),
    }
    request = SimpleNamespace(
        goal=SimpleNamespace(
            semantics=SimpleNamespace(geometry=geometry), target_pose=None
        )
    )
    plan = SimpleNamespace(
        segments=[SimpleNamespace(name="close", stop=1)],
        joint_trajectory=SimpleNamespace(positions=torch.zeros(1, 1, 1)),
    )
    score = GenSimTwist._candidate_geometry_score(
        SimpleNamespace(robot=robot),
        plan,
        request,
        SimpleNamespace(robot=SimpleNamespace(qpos=torch.zeros(1, 1))),
    )
    assert score[:3] == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)


@pytest.mark.parametrize(
    "positions,velocities,expected",
    [
        ([0.0, -1e-5], [0.0, 6.64e-5], True),
        ([0.0, 0.0], [0.0, 1000.0], False),
        ([0.0, 0.1], [0.0, 0.0], False),
        ([0.0], [float("nan")], False),
        ([0.0], [], False),
        ([], [], False),
    ],
)
def test_initial_twist_stability_checks_velocity(
    positions: list[float], velocities: list[float], expected: bool
) -> None:
    assert _initial_twist_state_stable(positions, velocities) is expected


def _route(direction: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(
        binding=SimpleNamespace(
            target_qpos=direction * TARGET_ANGLE,
            limits=(-TARGET_ANGLE, TARGET_ANGLE),
        )
    )


def _samples(states: list[tuple[float, bool]]) -> list[PressSample]:
    return [
        PressSample(float(index), angle, "twist" if contact else "prepare", contact)
        for index, (angle, contact) in enumerate(states)
    ]


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_disjoint_contact_chunks_accumulate_net_direction(direction: float) -> None:
    states = [(0.0, False)]
    for chunk in range(CHUNK_COUNT):
        start = direction * chunk * CHUNK_TRAVEL
        end = direction * (chunk + 1) * CHUNK_TRAVEL
        states.extend([(start, True), (end, True), (end, False)])
    states.append((direction * TARGET_ANGLE, False))

    result = evaluate_twist(
        _route(direction), _samples(states), final_qpos=direction * TARGET_ANGLE
    )

    assert result["accepted"]
    assert result["contact_travel"] == pytest.approx(CHUNK_COUNT * CHUNK_TRAVEL)
    assert result["signed_contact_travel"] == pytest.approx(CHUNK_COUNT * CHUNK_TRAVEL)
    assert result["uncontacted_reverse_travel"] == pytest.approx(0.0)


def test_contact_oscillation_does_not_accumulate_direction() -> None:
    states = [(0.0, False), (0.0, True)]
    states.extend([(CHUNK_TRAVEL, True), (0.0, True)] * CHUNK_COUNT)
    states.append((TARGET_ANGLE, False))

    result = evaluate_twist(_route(), _samples(states), final_qpos=TARGET_ANGLE)

    assert not result["accepted"]
    assert result["reason"] == "direction_not_confirmed"
    assert result["contact_travel"] == pytest.approx(0.0)
    assert result["coarse_turn_reached"]


def test_reverse_contact_chunks_cancel_forward_chunks() -> None:
    states = [(0.0, False)]
    for _ in range(CHUNK_COUNT):
        states.extend(
            [
                (0.0, True),
                (CHUNK_TRAVEL, True),
                (CHUNK_TRAVEL, False),
                (CHUNK_TRAVEL, True),
                (0.0, True),
                (0.0, False),
            ]
        )
    states.append((TARGET_ANGLE, False))

    result = evaluate_twist(_route(), _samples(states), final_qpos=TARGET_ANGLE)

    assert not result["accepted"]
    assert result["reason"] == "direction_not_confirmed"
    assert result["signed_contact_travel"] == pytest.approx(0.0)


def test_uncontacted_reversal_cancels_repeated_contact_excursions() -> None:
    states = [(0.0, False)]
    for _ in range(CHUNK_COUNT):
        states.extend(
            [(0.0, True), (CHUNK_TRAVEL, True), (CHUNK_TRAVEL, False), (0.0, False)]
        )
    states.append((TARGET_ANGLE, False))

    result = evaluate_twist(_route(), _samples(states), final_qpos=TARGET_ANGLE)

    assert not result["accepted"]
    assert result["reason"] == "direction_not_confirmed"
    assert result["signed_contact_travel"] == pytest.approx(CHUNK_COUNT * CHUNK_TRAVEL)
    assert result["uncontacted_reverse_travel"] == pytest.approx(
        CHUNK_COUNT * CHUNK_TRAVEL
    )
    assert result["contact_travel"] == pytest.approx(0.0)


def test_movement_between_isolated_contact_samples_earns_no_direction() -> None:
    states = [(0.0, False)]
    for chunk in range(CHUNK_COUNT):
        states.extend(
            [
                (chunk * CHUNK_TRAVEL, True),
                ((chunk + 1) * CHUNK_TRAVEL, False),
            ]
        )
    states.append((TARGET_ANGLE, False))

    result = evaluate_twist(_route(), _samples(states), final_qpos=TARGET_ANGLE)

    assert not result["accepted"]
    assert result["reason"] == "direction_not_confirmed"
    assert result["contact_path"] == pytest.approx(0.0)
    assert result["contact_travel"] == pytest.approx(0.0)


def test_no_contact_target_motion_is_rejected() -> None:
    samples = _samples([(0.0, False), (TARGET_ANGLE, False)])

    result = evaluate_twist(_route(), samples, final_qpos=TARGET_ANGLE)

    assert not result["accepted"]
    assert result["reason"] == "target_contact_missing"


def test_contact_direction_does_not_replace_final_target_convergence() -> None:
    partial_angle = CHUNK_COUNT * CHUNK_TRAVEL
    samples = _samples(
        [(0.0, False), (0.0, True), (partial_angle, True), (partial_angle, False)]
    )

    result = evaluate_twist(_route(), samples, final_qpos=partial_angle)

    assert not result["accepted"]
    assert result["reason"] == "target_qpos_not_reached"
    assert result["contact_travel"] > result["directional_travel_required"]
