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
        None if explicit else "ordinary"
    )
    assert semantics.affordance.custom_config == {}
    if explicit:
        assert forwarded is request


@pytest.mark.parametrize("axis", [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0)])
@pytest.mark.parametrize("purpose", ["ordinary", "handover_source"])
def test_pick_purpose_prefers_down_before_diagonal(axis, purpose):
    pose = torch.eye(4)[None]
    tcp = pose.clone()
    tcp[:, :3, 3] = torch.tensor([[-1.0, 0.0, 1.0]])
    ends = purpose == "handover_source"
    choices = grasp_strategies(
        torch.tensor([axis]), pose, tcp, ends=ends, purpose=purpose
    )
    width = 2 if ends else 1
    for choice in choices[:width]:
        torch.testing.assert_close(choice.direction, torch.tensor([[0.0, 0.0, -1.0]]))
    for choice in choices[width:]:
        assert choice.direction[0, 2].item() == -0.5
    if ends:
        assert torch.equal(choices[0].positive, choices[2].positive)
        assert torch.equal(choices[1].positive, ~choices[0].positive)


@pytest.mark.parametrize("purpose", ["pour", "handover_receive"])
@pytest.mark.parametrize("vertical", [True, False])
def test_pour_and_receiver_keep_orientation_dependent_order(purpose, vertical):
    pose = torch.eye(4)[None]
    tcp = pose.clone()
    tcp[:, 0, 3] = 1
    choices = grasp_strategies(
        torch.tensor([[0.0, 0.0, 1.0] if vertical else [1.0, 0.0, 0.0]]),
        pose,
        tcp,
        ends=purpose == "handover_receive",
        positive=torch.tensor([False]),
        purpose=purpose,
    )
    assert choices[0].direction[0, 2].item() == (-0.5 if vertical else -1.0)
    assert choices[1].direction[0, 2].item() == (-1.0 if vertical else -0.5)
    assert not choices[0].positive.any() and not choices[1].positive.any()


@pytest.mark.parametrize("purpose", ["ordinary", "pour"])
@pytest.mark.parametrize(
    "protected", ["none", "upright", "stack", "fixed", "explicit_direction"]
)
def test_pick_call_purpose_does_not_override_protected_grasps(
    monkeypatch, purpose, protected
):
    from dataclasses import replace
    from embodichain.gen_sim.task_engine._task_program import actions
    from embodichain.lab.sim.atomic_actions import (
        ObjectSemantics,
        GraspGoal,
        PickUpOptions,
    )
    from embodichain.lab.sim.atomic_actions.primitives.pick_up import PickUp
    from embodichain.lab.sim.atomic_actions.affordance import AntipodalAffordance

    affordance = AntipodalAffordance()
    if protected == "upright":
        affordance.set_custom_config(actions._PICK_GRASP_RULE, ("cup", "target"))
    semantics = ObjectSemantics(affordance=affordance, geometry={}, entity_id="cup")
    options = PickUpOptions()
    if protected == "stack":
        options = replace(options, pick_object_part="top")
    elif protected == "fixed":
        options = replace(options, fixed_object_to_eef=torch.eye(4))
    elif protected == "explicit_direction":
        options = replace(options, approach_direction=torch.tensor([1.0, 0.0, 0.0]))

    @dataclass(frozen=True)
    class Request:
        goal: object
        skill_options: object
        invocation_id: str

    request = Request(GraspGoal(semantics=semantics), options, "call")
    action = actions.GenSimPickUp()
    action.adaptive_unconstrained = True
    action.pick_purposes = {"call": purpose}
    monkeypatch.setattr(PickUp, "_plan", lambda self, request, context: request)
    forwarded = action._plan(request, object())
    assert forwarded.goal.semantics.affordance.get_custom_config(ADAPTIVE_GRASP) == (
        purpose if protected == "none" else None
    )
    assert semantics.affordance.get_custom_config(ADAPTIVE_GRASP) is None
    assert forwarded.skill_options is options


def test_pick_purposes_bind_per_call_not_per_object():
    from embodichain.gen_sim.task_engine._task_program.adaptive_grasp import (
        bind_pick_purposes,
    )
    from embodichain.lab.task_program.semantics import Pick, SceneObjectRef

    segments = [
        SimpleNamespace(
            name=name,
            segment_id=f"segment_{index}",
            calls=(
                SimpleNamespace(
                    call=Pick(object=SceneObjectRef("cup")), segment_call_index=0
                ),
            ),
        )
        for index, name in enumerate(("place_pick", "pour_pick"))
    ]
    program = SimpleNamespace(program_id="test", iter_segments=lambda: iter(segments))
    assert dict(
        bind_pick_purposes({"place_pick": "ordinary", "pour_pick": "pour"}, program)
    ) == {
        "test/segment_0:0": "ordinary",
        "test/segment_1:0": "pour",
    }
    with pytest.raises(ValueError, match="regenerate"):
        bind_pick_purposes({"place_pick": "ordinary"}, program)
    with pytest.raises(ValueError, match="missing or non-Pick"):
        bind_pick_purposes(
            {"place_pick": "ordinary", "pour_pick": "pour", "unknown": "ordinary"},
            program,
        )


def test_generated_pick_purposes_separate_pour_and_ordinary():
    from embodichain.gen_sim.task_engine.task_program_bundle import (
        _task_stability_payload,
    )

    graph = {
        "targets": {},
        "nodes": [
            {
                "id": name,
                "task_instance_id": name,
                "task_type": kind,
                "call": {"kind": "pick", "object": "same_object"},
            }
            for name, kind in (("first_pick", "E1"), ("second_pick", "E3"))
        ],
        "task_groups": [
            {"id": name, "node_ids": [name]} for name in ("first_pick", "second_pick")
        ],
    }
    result = _task_stability_payload(
        graph, SimpleNamespace(planner_objects=()), {"skill_profile": {"resources": []}}
    )
    assert result["pick_purposes"] == {"first_pick": "ordinary", "second_pick": "pour"}


@pytest.mark.parametrize("vertical", [True, False])
@pytest.mark.parametrize("failure", ["candidate", "ik", "path", "both", "none"])
def test_receiver_retries_through_real_shared_handover(monkeypatch, vertical, failure):
    from dataclasses import replace
    from pathlib import Path
    import runpy

    fixtures = SimpleNamespace(
        **runpy.run_path(
            str(Path(__file__).parents[2] / "sim/atomic_actions/test_actions.py")
        )
    )
    from embodichain.gen_sim.task_engine._task_program import actions
    from embodichain.lab.sim.atomic_actions import (
        ActionInvocation,
        HandOverGoal,
        HandOverOptions,
        MotionPolicy,
        TaskState,
    )
    from embodichain.lab.sim.atomic_actions.affordance_sampling import (
        AffordancePoseCandidates,
    )
    from embodichain.lab.sim.atomic_actions.primitives.hand_over import HandOver

    fixtures._torch_interpolation.__wrapped__(monkeypatch)
    generator = fixtures._dual_motion_generator()

    def fk(qpos=None, name=None, to_matrix=True):
        poses = torch.eye(4).repeat(2 if qpos is None else qpos.shape[0], 1, 1)
        if name == "right_arm":
            poses[:, 0, 3] = 0.4
        return poses

    generator.robot.compute_fk.side_effect = fk
    axis = torch.tensor([0.0, 0.0, 1.0] if vertical else [1.0, 0.0, 0.0])
    semantics, affordance = fixtures._handover_semantics(axis)
    affordance.set_custom_config(ADAPTIVE_GRASP, "ends")
    observed_directions, observed_ends = [], []

    def candidates(generator, pose, direction, *, obj_longest_axis, is_positive_part):
        torch.rand(1)
        observed_directions.append(direction.clone())
        observed_ends.append(is_positive_part.clone())
        poses = pose.clone()
        z = direction
        x = torch.tensor([[0.0, 1.0, 0.0]]).expand_as(z)
        y = torch.linalg.cross(z, x)
        poses[:, :3, :3] = torch.stack((x, y, z), dim=-1)
        poses[:, :3, 3] += obj_longest_axis * -0.1
        valid = torch.ones(2, 1, dtype=torch.bool)
        if failure == "candidate" and len(observed_directions) == 1:
            valid[0] = False
        return AffordancePoseCandidates(poses[:, None], torch.zeros(2, 1), valid)

    affordance.get_grasp_candidates = candidates
    held = fixtures._held(semantics, env_mask=torch.ones(2, dtype=torch.bool))
    held.object_to_eef[:, :3, 3] = axis * 0.1
    context = fixtures._handover_context(
        torch.eye(4).repeat(2, 1, 1),
        TaskState(2, "cpu", held_objects={"left_arm": held}),
    )
    first_z = -0.5 if vertical else -1.0
    native_ik = generator.robot.compute_ik.side_effect

    def ik(pose=None, name=None, joint_seed=None, **kwargs):
        ok, qpos = native_ik(pose=pose, name=name, joint_seed=joint_seed, **kwargs)
        if name == "right_arm" and failure in {"ik", "both"}:
            if failure == "both" or torch.isclose(pose[0, 2, 2], torch.tensor(first_z)):
                ok[0] = False
        return ok, qpos

    generator.robot.compute_ik.side_effect = ik
    native_generate = generator.generate

    def generate(targets, options=None):
        result = native_generate(targets, options=options)
        if (
            failure == "path"
            and options.control_part == "right_arm"
            and len(observed_directions) == 1
        ):
            ok = result.success.clone()
            ok[0] = False
            result = replace(result, success=ok)
        return result

    generator.generate = generate
    plans, rng_states = [], []
    shared_plan = HandOver._plan_existing_hold

    def record_shared(self, *args):
        plan = shared_plan(self, *args)
        plans.append(plan)
        rng_states.append(torch.random.get_rng_state().clone())
        return plan

    monkeypatch.setattr(HandOver, "_plan_existing_hold", record_shared)
    action = fixtures._bind_action(generator, actions.GenSimHandOver())
    target = torch.eye(4)
    target[2, 3] = 0.2
    plan = fixtures._plan_action(
        action,
        ActionInvocation(
            skill_id="hand_over",
            goal=HandOverGoal(semantics, target_pose=target),
            binding=fixtures._dual_binding(action, "source", "destination"),
            motion_policy=MotionPolicy(strategy="ik_interp", sample_count=60),
            skill_options=HandOverOptions(release_at_target=False),
        ),
        context,
    )
    assert plan.plan_success.tolist() == [failure != "both", True]
    assert len(plans) == (1 if failure == "none" else 2)
    assert actions._RECEIVE_SELECTION.get() is None
    torch.testing.assert_close(torch.random.get_rng_state(), rng_states[0])
    assert context.task.get_held_object("right_arm") is None
    assert context.task.get_held_object("left_arm") is not None
    assert observed_directions[0][0, 2].item() == first_z
    if len(plans) == 2:
        assert plan.diagnostics.metadata["selected_receive_attempt"] == (
            [-1, 0] if failure == "both" else [1, 0]
        )
        torch.testing.assert_close(observed_ends[0], observed_ends[1])
        assert observed_directions[1][0, 2].item() == (-1.0 if vertical else -0.5)
        torch.testing.assert_close(
            plan.joint_trajectory.positions[1], plans[0].joint_trajectory.positions[1]
        )
        torch.testing.assert_close(
            plan.effect_candidates.held_object_updates["right_arm"].object_to_eef[1],
            plans[0]
            .effect_candidates.held_object_updates["right_arm"]
            .object_to_eef[1],
        )
        if failure != "both":
            torch.testing.assert_close(
                plan.expected_effects.held_object_updates["right_arm"].object_to_eef[0],
                plans[1]
                .expected_effects.held_object_updates["right_arm"]
                .object_to_eef[0],
            )
    if failure == "both":
        torch.testing.assert_close(
            plan.joint_trajectory.positions[0],
            context.robot.qpos[0].expand_as(plan.joint_trajectory.positions[0]),
        )
