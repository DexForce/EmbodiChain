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
from dataclasses import make_dataclass, replace
from types import SimpleNamespace
from unittest.mock import Mock

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


def test_twist_port_delegates_non_e8_policy_without_reading_robot():
    command = torch.tensor([[0.3]])
    delegate = SimpleNamespace(
        validate_policy=Mock(), actions=Mock(return_value=iter([command]))
    )
    port = object.__new__(TwistAcceptancePort)
    port.delegate = delegate
    port.route = SimpleNamespace(preset=lambda phase: f"e8_{phase}")
    port.robot = SimpleNamespace(get_qpos=Mock(side_effect=AssertionError))
    policy = SimpleNamespace(cfg=SimpleNamespace(preset="unrelated_policy"))
    segment, mask = object(), torch.tensor([True])
    result = list(port.actions(policy, segment=segment, active_mask=mask))
    assert result[0] is command
    delegate.validate_policy.assert_called_once_with(policy, segment=segment)
    delegate.actions.assert_called_once_with(policy, segment=segment, active_mask=mask)
    port.robot.get_qpos.assert_not_called()


def test_failed_depth_plans_do_not_fall_back_to_unrequested_base_bite(monkeypatch):
    from embodichain.gen_sim.task_engine._task_program import twist_runtime
    from embodichain.lab.sim.atomic_actions import Twist

    request_type = make_dataclass(
        "DepthRequest", [("goal", object), ("skill_options", object)]
    )
    geometry = {
        "gen_sim_twist_angle": 0.1,
        "gen_sim_twist_measured_qpos": 0.0,
        "gen_sim_twist_target_qpos": 1.0,
        "gen_sim_twist_axis_sign": 1.0,
        "gen_sim_twist_chunk_angle": 0.1,
        "gen_sim_twist_grip_depth": 0.0125,
        "gen_sim_twist_bite_depth": 0.00625,
    }
    request = request_type(
        SimpleNamespace(semantics=SimpleNamespace(geometry=geometry), marker=0.0),
        TwistOptions(),
    )
    calls = []

    def candidate(request, roll, shift):
        return replace(
            request,
            goal=SimpleNamespace(semantics=request.goal.semantics, marker=shift),
        )

    def plan(self, request, context):
        calls.append(request.goal.marker)
        return SimpleNamespace(
            plan_success=torch.tensor([False]),
            diagnostics=None,
            marker=request.goal.marker,
        )

    monkeypatch.setattr(GenSimTwist, "_candidate_request", staticmethod(candidate))
    monkeypatch.setattr(Twist, "_plan", plan)
    action = object.__new__(GenSimTwist)
    token = twist_runtime._CANDIDATE_STATE.set(_TwistCandidateState())
    try:
        result = action._plan(request, SimpleNamespace())
    finally:
        twist_runtime._CANDIDATE_STATE.reset(token)
    assert len(calls) == 12
    assert result.marker == pytest.approx(-2 * 0.00625 / 3)


def test_depth_candidate_tcp_compensation_produces_the_requested_tip_depth():
    from embodichain.lab.sim.atomic_actions import (
        ObjectSemantics,
        SceneEntityPose,
        TwistAffordance,
        TwistGoal,
    )
    from embodichain.gen_sim.task_engine._task_program.twist_runtime import (
        _candidate_table,
    )

    outer, bite, tip_offset = 0.0125, 0.00625, 0.017
    point = (0.0, 0.0, outer - bite + tip_offset)
    goal = TwistGoal(
        ObjectSemantics(
            entity_id="control::knob",
            geometry={
                "gen_sim_twist_grasp_point": point,
                "gen_sim_twist_outer_point": (0.0, 0.0, outer),
                "gen_sim_twist_axis": (0.0, 0.0, -1.0),
            },
            affordance=TwistAffordance(
                grasp_position=point,
                axis_origin=(0.0, 0.0, 0.0),
                twist_axis=torch.tensor([0.0, 0.0, -1.0]),
                joint_name="axis",
                joint_limits=(-1.0, 1.0),
            ),
        ),
        SceneEntityPose("control::knob"),
    )
    request_type = make_dataclass("TipRequest", [("goal", object)])
    request = request_type(goal)
    for index, roll, shift in _candidate_table(bite):
        candidate = GenSimTwist._candidate_request(request, roll, shift)
        tcp_z = candidate.goal.semantics.affordance.grasp_position[2]
        final_tip_depth = outer - (tcp_z - tip_offset)
        assert final_tip_depth == pytest.approx(bite * (index // 4 + 1) / 3)
        assert 0 < final_tip_depth <= bite + 1e-12


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
        (-0.5, 0.0, 0.1, True, True, True),
        (-0.5, 0.0, 0.0, True, True, True),
        (-TARGET_ANGLE, 1.0, 0.1, True, True, True),
        (-TARGET_ANGLE, 0.0, 0.01, True, True, True),
        (-TARGET_ANGLE, 0.0, 0.1, False, True, True),
        (-TARGET_ANGLE, 0.0, 0.1, True, False, False),
    ],
)
def test_coarse_goal_stops_turning_before_park_checks_release(
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
        other_qpos=torch.zeros(1, 1),
        samples=samples,
        trace=[{"parent_contact": False} for _ in samples],
        _sim=SimpleNamespace(sim_config=SimpleNamespace(physics_dt=0.01)),
    )
    port = object.__new__(TwistAcceptancePort)
    port.route, port.sensor, port.dt = route, sensor, 0.04
    commanded = torch.full((1, 2), 0.2)
    port.robot = SimpleNamespace(
        get_qpos=lambda target=False: commanded.clone() if target else torch.zeros(1, 2)
    )
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
    if complete and released:
        samples.extend(
            PressSample(3.2 + index * 0.01, final, "cleanup", False, True)
            for index in range(20)
        )
        sensor.trace.extend({"parent_contact": False} for _ in range(20))
        hold_actions = list(
            port.actions(
                policy,
                segment=SimpleNamespace(calls=[]),
                active_mask=torch.tensor([True]),
            )
        )
        assert sensor.acceptance["accepted"] is True
        assert sensor.acceptance["early_stop"] is True
        assert all(torch.equal(action, commanded) for action in hold_actions)
    park_policy = SimpleNamespace(
        cfg=SimpleNamespace(preset="turned", kind="wait_stable"),
        entity=SimpleNamespace(entity_id="control"),
    )
    list(
        port.actions(
            park_policy,
            segment=SimpleNamespace(calls=[]),
            active_mask=torch.tensor([True]),
        )
    )
    assert sensor.acceptance["accepted"] is bool(
        complete and velocity == 0.0 and released
    )
    list(
        port.actions(
            park_policy,
            segment=SimpleNamespace(
                calls=[
                    SimpleNamespace(
                        call=SimpleNamespace(semantic_id="gen_sim.articulation_park")
                    )
                ]
            ),
            active_mask=torch.tensor([True]),
        )
    )
    assert sensor.acceptance["accepted"] is bool(
        complete and velocity == 0.0 and clearance >= 0.04 and released
    )
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
                        "gen_sim_twist_measured_qpos": final,
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
            robot=port.robot,
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
        assert torch.equal(plan["trajectory"].positions[:, 0], commanded)
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
        "gen_sim_twist_grip_points": [[-0.02, 0, 0], [0.02, 0, 0]],
        "gen_sim_twist_parent_collision_points": [[1.0, 0, 0]],
        "gen_sim_twist_pad_collision_points": {name: [[0, 0, 0]] for name in names},
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


def test_coarse_contact_oscillation_counts_path_not_net_direction() -> None:
    states = [(0.0, False), (0.0, True)]
    states.extend([(CHUNK_TRAVEL, True), (0.0, True)] * CHUNK_COUNT)
    states.append((TARGET_ANGLE, False))

    result = evaluate_twist(_route(), _samples(states), final_qpos=TARGET_ANGLE)

    assert result["accepted"]
    assert result["reason"] == "coarse_turn_reached"
    assert result["contact_travel"] == pytest.approx(0.0)
    assert result["coarse_turn_reached"]


def test_coarse_reverse_contact_chunks_count_absolute_path() -> None:
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

    assert result["accepted"]
    assert result["reason"] == "coarse_turn_reached"
    assert result["signed_contact_travel"] == pytest.approx(0.0)


def test_coarse_uncontacted_reversal_is_diagnostic_not_a_gate() -> None:
    states = [(0.0, False)]
    for _ in range(CHUNK_COUNT):
        states.extend(
            [(0.0, True), (CHUNK_TRAVEL, True), (CHUNK_TRAVEL, False), (0.0, False)]
        )
    states.append((TARGET_ANGLE, False))

    result = evaluate_twist(_route(), _samples(states), final_qpos=TARGET_ANGLE)

    assert result["accepted"]
    assert result["reason"] == "coarse_turn_reached"
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
    assert result["reason"] == "coarse_contact_travel_insufficient"
    assert result["contact_path"] == pytest.approx(0.0)
    assert result["contact_travel"] == pytest.approx(0.0)


def test_no_contact_target_motion_is_rejected() -> None:
    samples = _samples([(0.0, False), (TARGET_ANGLE, False)])

    result = evaluate_twist(_route(), samples, final_qpos=TARGET_ANGLE)

    assert not result["accepted"]
    assert result["reason"] == "target_contact_missing"


def test_coarse_contact_does_not_require_final_target_convergence() -> None:
    partial_angle = CHUNK_COUNT * CHUNK_TRAVEL
    samples = _samples(
        [(0.0, False), (0.0, True), (partial_angle, True), (partial_angle, False)]
    )

    result = evaluate_twist(_route(), samples, final_qpos=partial_angle)

    assert result["accepted"]
    assert result["reason"] == "coarse_turn_reached"
    assert result["contact_travel"] > result["directional_travel_required"]
    assert result["target_qpos_reached"] is False


def test_coarse_contact_noise_cannot_accumulate_fifteen_degrees() -> None:
    states = [(0.0, False), (0.0, True)]
    states.extend([(math.radians(0.1), True), (0.0, True)] * 1000)
    result = evaluate_twist(_route(), _samples(states), final_qpos=0.0)
    assert not result["accepted"]
    assert result["contact_path"] == 0.0
    assert result["raw_contact_path"] > math.radians(15.0)


def test_coarse_contact_small_substeps_accumulate_real_travel() -> None:
    states = [(0.0, False)] + [
        (math.radians(index * 0.1), True) for index in range(161)
    ]
    result = evaluate_twist(_route(), _samples(states), final_qpos=states[-1][0])
    assert result["accepted"]
    assert result["contact_path"] >= math.radians(15.0)


def test_coarse_contact_exact_fifteen_degree_boundary() -> None:
    states = [(0.0, False), (0.0, True), (math.radians(15.0), True)]
    result = evaluate_twist(_route(), _samples(states), final_qpos=states[-1][0])
    assert result["accepted"]
