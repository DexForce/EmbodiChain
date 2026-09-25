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

import pytest
import torch
from types import SimpleNamespace
from unittest.mock import Mock
from dataclasses import dataclass

from embodichain.gen_sim.task_engine._task_program.adaptive_grasp import (
    grasp_strategies,
    opposite_grasp_end,
    select_grasp,
    ADAPTIVE_GRASP,
)


@pytest.mark.parametrize("axis", [(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
def test_horizontal_grasp_tracks_the_arm_not_axis_sign(axis):
    axis = torch.tensor([axis]).repeat(2, 1)
    pose = torch.eye(4).repeat(2, 1, 1)
    tcp = pose.clone()
    tcp[:, :3, 3] = axis * torch.tensor([[-1.0], [1.0]])
    choices = grasp_strategies(axis, pose, tcp, ends=True)
    torch.testing.assert_close(
        choices[0].direction, torch.tensor([[0.0, 0.0, -1.0]]).repeat(2, 1)
    )
    assert choices[0].positive.tolist() == [False, True]
    flipped = grasp_strategies(-axis, pose, tcp, ends=True)
    torch.testing.assert_close(
        axis * (choices[0].positive.float() * 2 - 1)[:, None],
        -axis * (flipped[0].positive.float() * 2 - 1)[:, None],
    )


def test_vertical_grasp_approaches_from_each_arm_and_has_downward_alternative():
    pose = torch.eye(4).repeat(2, 1, 1)
    tcp = pose.clone()
    tcp[:, 1, 3] = torch.tensor([-1.0, 1.0])
    choices = grasp_strategies(
        torch.tensor([[0.0, 0.0, 1.0]]).repeat(2, 1), pose, tcp, ends=False
    )
    assert choices[0].direction[:, 1].sign().tolist() == [1.0, -1.0]
    assert (choices[0].direction[:, 2] < 0).all()
    torch.testing.assert_close(
        choices[1].direction, torch.tensor([[0.0, 0.0, -1.0]]).repeat(2, 1)
    )
    assert choices[0].axis is None


def test_receiver_uses_actual_source_grasp_not_arm_name():
    pose = torch.eye(4).repeat(2, 1, 1)
    source = pose.clone()
    source[:, 0, 3] = torch.tensor([-0.1, 0.1])
    destination = pose.clone()
    destination[:, 0, 3] = torch.tensor([1.0, -1.0])
    axis = torch.tensor([[1.0, 0.0, 0.0]]).repeat(2, 1)
    assert opposite_grasp_end(axis, pose, source, destination).tolist() == [True, False]


def test_vertical_approach_with_no_horizontal_separation_is_finite():
    pose = torch.eye(4)[None]
    choices = grasp_strategies(torch.tensor([[0.0, 0.0, 1.0]]), pose, pose, ends=True)
    for choice in choices:
        assert torch.isfinite(choice.direction).all()
        torch.testing.assert_close(
            torch.linalg.vector_norm(choice.direction, dim=-1), torch.ones(1)
        )


def test_candidate_fallback_keeps_successful_rows_and_reports_selected_direction():
    from embodichain.lab.sim.atomic_actions.affordance import AntipodalAffordance
    from embodichain.lab.sim.atomic_actions.affordance_sampling import (
        AffordancePoseCandidates,
    )

    affordance = AntipodalAffordance()
    poses = torch.eye(4).repeat(2, 1, 1)
    tcp = poses.clone()
    tcp[:, 0, 3] = -1
    choices = grasp_strategies(
        torch.tensor([[0.0, 0.0, 1.0]]).repeat(2, 1), poses, tcp, ends=False
    )
    first = torch.tensor([[True], [False]])
    affordance.get_grasp_candidates = Mock(
        side_effect=[
            AffordancePoseCandidates(poses[:, None], torch.zeros(2, 1), first),
            AffordancePoseCandidates(
                poses[:, None], torch.zeros(2, 1), torch.ones(2, 1, dtype=torch.bool)
            ),
        ]
    )
    context = SimpleNamespace(affordance_sampling=None, env_ids=torch.tensor([0, 1]))
    sample, direction = select_grasp(
        affordance, object(), poses, choices, context, "probe"
    )
    assert sample.success.tolist() == [True, True]
    assert sample.metadata["adaptive_grasp"]["strategy_indices"] == [0, 1]
    torch.testing.assert_close(direction[0], choices[0].direction[0])
    torch.testing.assert_close(direction[1], choices[1].direction[1])


def test_pick_uses_selected_direction_for_execution_and_restores_scope(monkeypatch):
    from embodichain.gen_sim.task_engine._task_program.actions import (
        GenSimPickUp,
        _PICK_SELECTION,
    )
    from embodichain.lab.sim.atomic_actions.primitives.pick_up import PickUp

    direction = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.8, -0.6]])

    def plan(self, request, context):
        _PICK_SELECTION.get().direction = direction
        return self._get_full_pickup_trajectory(*([None] * 11))

    monkeypatch.setattr(PickUp, "_plan", plan)
    parent = Mock(side_effect=lambda *args: args[5])
    monkeypatch.setattr(PickUp, "_get_full_pickup_trajectory", parent)
    result = GenSimPickUp()._plan(object(), object())
    torch.testing.assert_close(result, direction)
    assert _PICK_SELECTION.get() is None
    monkeypatch.setattr(PickUp, "_plan", Mock(side_effect=ValueError("probe")))
    with pytest.raises(ValueError, match="probe"):
        GenSimPickUp()._plan(object(), object())
    assert _PICK_SELECTION.get() is None


@pytest.mark.parametrize(
    "call_id,mode",
    [
        ("gen_sim.pick.handover_source", "ends"),
        ("gen_sim.pick.handover_horizontal_source.cup", "ends"),
    ],
)
def test_adaptive_lowering_uses_same_policy_for_horizontal_and_vertical(call_id, mode):
    from embodichain.gen_sim.task_engine._task_program.services import (
        make_pick_factory,
        _PickRoute,
    )
    from embodichain.lab.sim.atomic_actions import ObjectSemantics
    from embodichain.lab.sim.atomic_actions.affordance import AntipodalAffordance
    from embodichain.lab.sim.atomic_actions.primitives.pick_up import PickUpOptions
    from embodichain.lab.task_program.semantics.calls import RegisteredSemanticCall

    semantics = ObjectSemantics(
        affordance=AntipodalAffordance(), geometry={}, entity_id="cup"
    )
    routes = (_PickRoute(object_id="cup", target_id="grasp"),)
    factory = make_pick_factory(routes, call_id)
    lowerer = factory.lowerer_type(routes, (semantics,))
    result = lowerer.lower(
        RegisteredSemanticCall(
            call_id=call_id, arguments={"object": "cup", "target": "grasp"}
        ),
        context=SimpleNamespace(),
        bound=SimpleNamespace(),
        option_template=PickUpOptions(),
    )
    assert result.goal.semantics.affordance.get_custom_config(ADAPTIVE_GRASP) == mode
    assert semantics.affordance.custom_config == {}


def test_handover_adapter_keeps_descriptor_and_unmarked_receiver_unchanged(monkeypatch):
    from embodichain.gen_sim.task_engine._task_program.actions import GenSimHandOver
    from embodichain.lab.sim.atomic_actions.primitives.hand_over import HandOver

    assert GenSimHandOver.descriptor() == HandOver.descriptor()
    parent = Mock(return_value=object())
    monkeypatch.setattr(HandOver, "_resolve_grasp", parent)
    result = GenSimHandOver()._resolve_grasp(
        None,
        None,
        None,
        "hand",
        obj_longest_axis=None,
        is_positive_part=None,
        context=None,
        sample_key="explicit",
    )
    assert result is parent.return_value
    parent.assert_called_once()


def test_pick_falls_back_after_path_failure_without_replacing_successful_rows(
    monkeypatch,
):
    from embodichain.gen_sim.task_engine._task_program import actions
    from embodichain.lab.sim.atomic_actions.affordance_sampling import AffordanceSample
    from embodichain.lab.sim.atomic_actions.primitives.pick_up import PickUp

    poses = torch.eye(4).repeat(2, 1, 1)
    alternate = poses.clone()
    alternate[:, 0, 3] = 1
    samples = [
        AffordanceSample(torch.ones(2, dtype=torch.bool), p, {"candidate_ids": [0, 0]})
        for p in (poses, alternate)
    ]
    monkeypatch.setattr(
        actions,
        "select_grasp",
        Mock(
            side_effect=[
                (sample, torch.tensor([[0.0, 0.0, -1.0]]).repeat(2, 1))
                for sample in samples
            ]
        ),
    )
    lengths = {"approach": 1, "close": 1, "lift": 1}
    planner = Mock(
        side_effect=[
            (torch.tensor([False, True]), torch.ones(2, 3, 4), lengths),
            (torch.ones(2, dtype=torch.bool), torch.full((2, 3, 4), 2.0), lengths),
        ]
    )
    monkeypatch.setattr(PickUp, "_get_full_pickup_trajectory", planner)
    action = actions.GenSimPickUp()
    action._planning_services = SimpleNamespace(device=torch.device("cpu"))
    grasp = SimpleNamespace(
        require_target=lambda kind: object(),
        joint_positions=lambda *a, **kw: torch.zeros(2, 1),
    )
    selection = actions._PickSelection(
        request=SimpleNamespace(
            binding=SimpleNamespace(endpoint=lambda *a: grasp), motion_policy=object()
        )
    )
    context = SimpleNamespace(
        batch_size=2,
        robot=SimpleNamespace(qpos=torch.zeros(2, 4)),
        last_qpos=torch.zeros(2, 4),
        require_control_dt=lambda: 0.04,
    )
    sample = action._select_adaptive_pick(
        object(),
        object(),
        poses,
        torch.zeros(2, 1),
        object(),
        object(),
        context,
        "probe",
        (object(), object()),
        selection,
    )
    assert sample.success.tolist() == [True, True]
    assert sample.metadata["adaptive_grasp"]["strategy_indices"] == [1, 0]
    torch.testing.assert_close(sample.poses[:, 0, 3], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(
        selection.trajectory[1][:, 0, 0], torch.tensor([2.0, 1.0])
    )
    token = actions._PICK_SELECTION.set(selection)
    try:
        assert (
            action._get_full_pickup_trajectory(*([None] * 11)) is selection.trajectory
        )
    finally:
        actions._PICK_SELECTION.reset(token)
    assert planner.call_count == 2


@pytest.mark.parametrize("explicit", [False, True])
def test_builtin_pick_keeps_downstream_contract_and_explicit_grasp(
    monkeypatch, explicit
):
    from embodichain.gen_sim.task_engine._task_program.actions import GenSimPickUp
    from embodichain.lab.sim.atomic_actions import ObjectSemantics
    from embodichain.lab.sim.atomic_actions.affordance import AntipodalAffordance
    from embodichain.lab.sim.atomic_actions.primitives.pick_up import (
        PickUp,
        PickUpOptions,
        GraspGoal,
    )

    @dataclass(frozen=True)
    class Request:
        goal: object
        skill_options: object

    semantics = ObjectSemantics(
        affordance=AntipodalAffordance(), geometry={}, entity_id="cup"
    )
    options = PickUpOptions(downstream_object_target_poses=(torch.eye(4),))
    request = Request(
        GraspGoal(semantics=semantics, grasp_xpos=torch.eye(4) if explicit else None),
        options,
    )
    monkeypatch.setattr(PickUp, "_plan", lambda self, request, context: request)
    action = GenSimPickUp()
    action.adaptive_unconstrained = True
    forwarded = action._plan(request, object())
    assert forwarded.skill_options is options
    assert forwarded.goal.semantics.affordance.get_custom_config(ADAPTIVE_GRASP) == (
        None if explicit else "free"
    )
    assert semantics.affordance.custom_config == {}
    if explicit:
        assert forwarded is request
