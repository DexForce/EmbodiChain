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

"""GenSim-owned service contracts migrated from shared integration tests."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program.align_held import _AlignHeldFactory

from embodichain.gen_sim.task_engine._task_program.configured import (
    decode_task_lowerer,
)
from embodichain.lab.task_program.integrations.configured import _decode_robot_profile
from embodichain.gen_sim.task_engine._task_program.services import (
    _AbsolutePoseTarget,
    _CoordinatedHoldLowerer,
    _CoordinatedTransportRoute,
    _MoveHeldObjectLowerer,
    _MoveHeldObjectRoute,
    _PickLowerer,
    _PickRoute,
)
from embodichain.lab.gym.utils._component_composition import _resolve_gym_components
from embodichain.lab.task_program.integrations._configured_composition import (
    _compose_integration_payload,
    _resolve_task_program_components,
)
from embodichain.lab.sim.atomic_actions import (
    AntipodalAffordance,
    CoordinatedPickmentOptions,
    EntityState,
    GraspGoal,
    HandOverOptions,
    HeldObjectPoseGoal,
    MoveHeldObjectOptions,
    ObjectSemantics,
    PickUpOptions,
    PlanningContext,
    RobotObservation,
    SceneSnapshot,
    TaskState,
)
from embodichain.lab.task_program.semantics import (
    HeldObjectRelation,
    RegisteredSemanticCall,
    SceneObjectRef,
    SemanticEffectKind,
)
from embodichain.utils.utility import load_config

__all__: list[str] = []


@pytest.mark.parametrize("horizontal", [False, True])
def test_handover_source_axis_is_call_local_and_does_not_filter_receiver(horizontal):
    from embodichain.gen_sim.task_engine._task_program.adaptive_grasp import (
        ADAPTIVE_GRASP,
    )
    from embodichain.gen_sim.task_engine._task_program.services import make_pick_factory
    from embodichain.lab.sim.atomic_actions.affordance import AxisAlignAffordance

    semantics = ObjectSemantics(
        affordance=AxisAlignAffordance(
            mesh_vertices=torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            ),
            mesh_triangles=torch.tensor([[0, 1, 2]]),
            internal_axis=torch.tensor([1.0, 0.0, 0.0]),
        ),
        geometry={},
        entity_id="cup",
    )
    route = _PickRoute(object_id="cup", target_id="source")
    call_id = (
        "gen_sim.pick.handover_horizontal_source.cup"
        if horizontal
        else "gen_sim.pick.ordinary"
    )
    factory = make_pick_factory((route,), call_id)
    lowerer = factory.lowerer_type((route,), (semantics,))
    result = lowerer.lower(
        RegisteredSemanticCall(
            call_id=call_id, arguments={"object": "cup", "target": "source"}
        ),
        context=SimpleNamespace(),
        bound=SimpleNamespace(),
        option_template=PickUpOptions(pick_object_part="center"),
    )
    assert semantics.affordance.custom_config == {}
    affordance = result.goal.semantics.affordance
    assert affordance.get_custom_config(ADAPTIVE_GRASP) == (
        "ends" if horizontal else None
    )
    assert factory.revision == ("4" if horizontal else "2")
    if horizontal:
        assert affordance is not semantics.affordance
    # Even if HandOver observes the held snapshot, its center request stays unfiltered.
    generator = SimpleNamespace(
        get_valid_grasp_poses=Mock(return_value=[(torch.eye(4)[None], torch.zeros(1))])
    )
    affordance.get_grasp_candidates(
        generator, torch.eye(4)[None], torch.tensor([0.0, 0.0, -1.0])
    )
    assert generator.get_valid_grasp_poses.call_args.kwargs["obj_longest_axis"] is None


@pytest.mark.parametrize("marked", [False, True])
def test_explicit_top_selection_is_not_reinterpreted_as_automatic(monkeypatch, marked):
    from embodichain.gen_sim.task_engine._task_program.actions import (
        GenSimPickUp,
    )
    from embodichain.gen_sim.task_engine._task_program.adaptive_grasp import (
        ADAPTIVE_GRASP,
    )
    from embodichain.lab.sim.atomic_actions.affordance import AxisAlignAffordance

    affordance = AxisAlignAffordance(internal_axis=torch.tensor([1.0, 0.0, 0.0]))
    if marked:
        affordance.set_custom_config(ADAPTIVE_GRASP, "ends")
    poses = torch.eye(4).repeat(2, 2, 1, 1)
    valid = torch.ones(2, 2, dtype=torch.bool)
    ik_valid = torch.tensor([[True, False], [False, True]])
    candidates = SimpleNamespace(poses=poses, costs=torch.zeros(2, 2), valid=valid)
    get_candidates = Mock(return_value=candidates)
    monkeypatch.setattr(affordance, "get_grasp_candidates", get_candidates)
    sample = Mock(side_effect=lambda candidates, **kw: candidates)
    monkeypatch.setattr(affordance, "sample_candidates", sample)
    action = GenSimPickUp()
    action._planning_services = SimpleNamespace(
        device=torch.device("cpu"), grasp_pose_generator=Mock()
    )
    monkeypatch.setattr(
        action, "_select_feasible_grasp_variants", Mock(return_value=(poses, ik_valid))
    )
    object_pose = torch.eye(4).repeat(2, 1, 1)
    object_pose[0, :3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    object_pose[1, :3, :3] = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    )
    context = SimpleNamespace(affordance_sampling=None, env_ids=torch.tensor([0, 1]))
    result = action._resolve_grasp_pose(
        ObjectSemantics(affordance=affordance, geometry={}, entity_id="cup"),
        object_pose,
        torch.zeros(2, 7),
        object(),
        "hand",
        PickUpOptions(pick_object_part="top"),
        torch.tensor([0.0, 0.0, -1.0]),
        context,
        sample_key="source",
    )
    expected = torch.tensor([0.0, 0.0, 1.0])
    torch.testing.assert_close(
        get_candidates.call_args.kwargs["obj_longest_axis"], expected
    )
    assert torch.equal(result.valid, ik_valid)
    assert sample.call_args.kwargs["key"] == "source"


def test_shared_pick_options_do_not_expose_gensim_axis():
    from dataclasses import fields
    from embodichain.lab.task_program.integrations.configured import (
        _decode_action_options,
    )

    assert "pick_object_local_axis" not in {item.name for item in fields(PickUpOptions)}
    with pytest.raises(ValueError, match="unsupported fields"):
        _decode_action_options(
            {"kind": "pick_up", "pick_object_local_axis": [1.0, 0.0, 0.0]},
            path="options",
        )


def test_unmarked_gensim_pick_delegates_without_changing_inputs(monkeypatch):
    from embodichain.gen_sim.task_engine._task_program.actions import GenSimPickUp
    from embodichain.lab.sim.atomic_actions.primitives.pick_up import PickUp

    semantics = ObjectSemantics(
        affordance=AntipodalAffordance(), geometry={}, entity_id="cup"
    )
    original = Mock(return_value=object())
    monkeypatch.setattr(PickUp, "_resolve_grasp_pose", original)
    args = (
        semantics,
        object(),
        object(),
        object(),
        "hand",
        object(),
        object(),
        object(),
    )
    result = GenSimPickUp()._resolve_grasp_pose(*args, sample_key="unchanged")
    original.assert_called_once_with(*args, sample_key="unchanged")
    assert result is original.return_value


def test_constrained_pick_marks_only_its_goal_and_uses_a_scoped_generator(monkeypatch):
    from embodichain.gen_sim.task_engine._task_program.actions import (
        GenSimPickUp,
        _PICK_GRASP_RULE,
    )
    from embodichain.gen_sim.task_engine._task_program.grasp_filter import (
        TaskGraspPoseGenerator,
    )

    semantics = ObjectSemantics(
        affordance=AntipodalAffordance(), geometry={}, entity_id="cup"
    )
    route = _PickRoute(object_id="cup", target_id="upright", grasp_region="upper_half")
    sampler = Mock(spec=TaskGraspPoseGenerator)
    bound = SimpleNamespace(
        binding=SimpleNamespace(
            resources={
                "primary": SimpleNamespace(
                    endpoints={
                        "grasp": SimpleNamespace(
                            runtime_target=SimpleNamespace(target_id="hand")
                        )
                    }
                )
            }
        )
    )
    result = _PickLowerer((route,), (semantics,), {"hand": sampler}).lower(
        RegisteredSemanticCall(
            call_id="gen_sim.pick", arguments={"object": "cup", "target": "upright"}
        ),
        context=SimpleNamespace(),
        bound=bound,
        option_template=PickUpOptions(),
    )
    assert semantics.affordance.custom_config == {}
    marked = result.goal.semantics
    assert marked.affordance.get_custom_config(_PICK_GRASP_RULE) == ("cup", "upright")
    poses = torch.eye(4)[None, None]
    valid = torch.ones(1, 1, dtype=torch.bool)
    candidates = SimpleNamespace(poses=poses, valid=valid, costs=torch.zeros(1, 1))
    get_candidates = Mock(return_value=candidates)
    monkeypatch.setattr(marked.affordance, "get_grasp_candidates", get_candidates)
    monkeypatch.setattr(marked.affordance, "sample_candidates", Mock())
    action = GenSimPickUp()
    action._planning_services = SimpleNamespace(grasp_pose_generator=lambda _: sampler)
    monkeypatch.setattr(
        action, "_select_feasible_grasp_variants", lambda *a: (poses, valid)
    )
    action._resolve_grasp_pose(
        marked,
        torch.eye(4)[None],
        torch.zeros(1, 7),
        object(),
        "hand",
        PickUpOptions(),
        torch.tensor([0.0, 0.0, -1.0]),
        SimpleNamespace(affordance_sampling=None, env_ids=torch.tensor([0])),
        sample_key="upright",
    )
    sampler.for_pick.assert_called_once()
    assert get_candidates.call_args.args[0] is sampler.for_pick.return_value
    assert get_candidates.call_args.kwargs["obj_longest_axis"] is None


def test_adaptive_source_lowerer_rejects_conflicting_explicit_region():
    from embodichain.gen_sim.task_engine._task_program.services import make_pick_factory

    semantics = ObjectSemantics(
        affordance=AntipodalAffordance(), geometry={}, entity_id="cup"
    )
    routes = (_PickRoute(object_id="cup", target_id="source"),)
    call_id = "gen_sim.pick.handover_horizontal_source.cup"
    lowerer = make_pick_factory(routes, call_id).lowerer_type(routes, (semantics,))
    with pytest.raises(ValueError, match="unrestricted region"):
        lowerer.lower(
            RegisteredSemanticCall(
                call_id=call_id, arguments={"object": "cup", "target": "source"}
            ),
            context=SimpleNamespace(),
            bound=SimpleNamespace(),
            option_template=PickUpOptions(pick_object_part="top"),
        )


def test_motion_velocity_guard_rejects_joint_branch_jumps_per_environment() -> None:
    from embodichain.gen_sim.task_engine._task_program.motion import _velocity_validity

    positions = torch.tensor([[[0.0], [0.02], [0.04]], [[0.0], [3.14], [3.15]]])
    dt = torch.tensor([[0.0, 0.04, 0.04], [0.0, 0.04, 0.04]])
    limits = torch.ones(2, 1)
    assert _velocity_validity(positions, dt, limits).tolist() == [True, False]
    dt[0, 1] = 0
    assert _velocity_validity(positions, dt, limits).tolist() == [False, False]
    with pytest.raises(ValueError, match="matching batched"):
        _velocity_validity(positions, dt, torch.ones(1, 1))


def test_final_plan_velocity_rejects_resampled_and_hand_jumps(monkeypatch) -> None:
    from embodichain.gen_sim.task_engine._task_program import assembly

    monkeypatch.setattr(assembly, "_joint_velocity_limits", lambda *a: torch.ones(1, 2))
    for joint in (0, 1):
        positions = torch.zeros(1, 2, 2)
        positions[0, 1, joint] = 0.2
        plan = SimpleNamespace(
            plan_success=torch.tensor([True]),
            joint_trajectory=SimpleNamespace(
                positions=positions, dt=torch.tensor([[0.0, 0.04]])
            ),
        )
        result = assembly._validate_final_plan_velocity(plan, object())
        assert result["valid_mask"].tolist() == [False]
        assert result["diagnostics"][0]["control_joint_index"] == joint
        assert plan.plan_success.tolist() == [True]


def test_failed_plan_can_expose_grasp_proposals_without_claiming_attachment() -> None:
    from embodichain.gen_sim.task_engine._task_program.assembly import (
        _planned_grasp_candidates,
    )

    transform = torch.eye(4).unsqueeze(0)
    held = SimpleNamespace(
        semantics=SimpleNamespace(entity_id="block"),
        object_to_eef=transform,
        grasp_xpos=transform,
    )
    updates = {"left": held, "right": None}
    plan = SimpleNamespace(
        plan_success=torch.tensor([False]),
        expected_effects=SimpleNamespace(held_object_updates=updates),
    )
    proposals = _planned_grasp_candidates(plan)
    assert len(proposals) == 1
    assert proposals[0]["evidence_scope"] == "proposal_only_not_observed_attachment"
    assert proposals[0]["object_id"] == "block"
    assert proposals[0]["resource"] == "left"
    proposals[0]["object_to_eef"][0][0][0] = 0.0
    assert transform[0, 0, 0] == 1.0
    assert plan.plan_success.tolist() == [False]
    assert updates["left"] is held


def test_initial_success_requires_final_trajectory_evidence() -> None:
    from embodichain.gen_sim.task_engine._task_program.assembly import (
        _validate_final_plan_velocity,
    )

    plan = SimpleNamespace(plan_success=torch.tensor([True]), joint_trajectory=None)
    with pytest.raises(ValueError, match="final joint trajectory"):
        _validate_final_plan_velocity(plan, object())


def test_velocity_diagnostics_locate_peak_and_serialize_zero_duration() -> None:
    import json
    from embodichain.gen_sim.task_engine._task_program.motion import (
        _velocity_diagnostics,
    )

    positions = torch.tensor([[[0.0, 0.0], [0.01, 0.4], [0.02, 0.4]]])
    dt = torch.tensor([[0.0, 0.04, 0.04]])
    limits = torch.tensor([[1.0, 2.0]])
    record = _velocity_diagnostics(positions, dt, limits)[0]
    assert record["frame"] == 1
    assert record["control_joint_index"] == 1
    assert record["speed_ratio"] == pytest.approx(5.0)
    dt[0, 1] = 0
    records = _velocity_diagnostics(positions, dt, limits)
    assert records[0]["speed_ratio"] is None
    json.dumps(records, allow_nan=False)


def test_motion_wrapper_masks_unsafe_rows_without_clipping_commands(
    monkeypatch,
) -> None:
    from embodichain.gen_sim.task_engine._task_program import motion

    generator = object.__new__(motion.ApproachMotionGenerator)
    generator.robot = SimpleNamespace(get_qvel_limits=lambda **kw: torch.ones(2, 1))
    monkeypatch.setattr(motion, "_joint_velocity_limits", lambda *a: torch.ones(2, 1))
    positions = torch.tensor([[[0.0], [0.02]], [[0.0], [3.14]]])
    original = motion.PlanResult(
        success=torch.tensor([True, True]),
        positions=positions,
        dt=torch.tensor([[0.0, 0.04], [0.0, 0.04]]),
    )
    monkeypatch.setattr(motion.MotionGenerator, "generate", lambda *a, **kw: original)
    result = generator.generate(
        [motion.PlanState(move_type=motion.MoveType.EEF_MOVE, xpos=torch.eye(4))],
        motion.MotionGenOptions(control_part="arm"),
    )
    assert result.success.tolist() == [True, False]
    assert original.success.tolist() == [True, True]
    torch.testing.assert_close(result.positions, positions)


def test_single_pose_transport_samples_cartesian_path(monkeypatch) -> None:
    from embodichain.gen_sim.task_engine._task_program import motion

    start = torch.eye(4).unsqueeze(0)
    start[:, 2, 3] = 1.1
    target = start.clone()
    target[:, :3, 3] = torch.tensor([0.2, 0.1, 1.3])
    target[:, :3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    generator = object.__new__(motion.ApproachMotionGenerator)
    generator.robot = SimpleNamespace(compute_fk=lambda **kw: start.clone())
    monkeypatch.setattr(motion, "_joint_velocity_limits", lambda *a: torch.ones(1, 1))
    result = motion.PlanResult(
        success=torch.tensor([True]),
        positions=torch.zeros(1, 5, 1),
        dt=torch.full((1, 5), 0.04),
    )
    backend = Mock(return_value=result)
    monkeypatch.setattr(motion.MotionGenerator, "generate", backend)
    options = motion.MotionGenOptions(
        strategy="ik_interp",
        control_part="arm",
        start_qpos=torch.zeros(1, 1),
        sample_count=5,
    )

    generator.generate(
        [motion.PlanState(move_type=motion.MoveType.EEF_MOVE, xpos=target)], options
    )

    samples = backend.call_args.args[0]
    forwarded = backend.call_args.kwargs["options"]
    assert len(samples) == 4
    assert forwarded.preserve_cartesian_samples is True
    assert forwarded.is_linear is True
    assert forwarded.sample_count == 5
    for index, sample in enumerate(samples, start=1):
        expected = start[:, :3, 3].lerp(target[:, :3, 3], index / 4)
        torch.testing.assert_close(sample.xpos[:, :3, 3], expected)
        assert float(sample.xpos[0, 2, 3]) >= 1.1
    torch.testing.assert_close(samples[-1].xpos, target)
    assert options.preserve_cartesian_samples is False


def test_velocity_invalid_e6_retries_with_denser_samples(monkeypatch) -> None:
    from embodichain.gen_sim.task_engine._task_program import motion

    generator = object.__new__(motion.ApproachMotionGenerator)
    start = torch.eye(4).unsqueeze(0)
    start[:, 2, 3] = 1.0
    generator.robot = SimpleNamespace(compute_fk=lambda **kw: start.clone())
    monkeypatch.setattr(motion, "_joint_velocity_limits", lambda *a: torch.ones(1, 1))
    calls = []

    def backend(_self, targets, *, options):
        calls.append((len(targets), options.sample_count))
        success = torch.tensor([len(calls) > 1])
        return motion.PlanResult(
            success=success,
            positions=torch.zeros(1, options.sample_count, 1),
            dt=torch.full((1, options.sample_count), 0.04),
        )

    monkeypatch.setattr(motion.MotionGenerator, "generate", backend)
    options = motion.MotionGenOptions(
        strategy="ik_interp",
        control_part="arm",
        start_qpos=torch.zeros(1, 1),
        sample_count=140,
    )
    result = generator.generate(
        [motion.PlanState(move_type=motion.MoveType.EEF_MOVE, xpos=start)], options
    )
    assert result.success.tolist() == [True]
    assert [sample_count for _, sample_count in calls] == [140, 260]
    assert options.sample_count == 140


@pytest.mark.parametrize(
    "recoverable,primary_safe", [(True, False), (True, True), (False, False)]
)
def test_local_place_ik_preserves_targets_limits_and_rng(
    monkeypatch, recoverable, primary_safe
):
    from embodichain.gen_sim.task_engine._task_program import motion

    seed = torch.tensor([[-0.99, 0.0]])
    pose = torch.eye(4).unsqueeze(0)
    calls = []

    def solve(_generator, targets, *, options):
        torch.rand(1)
        calls.append(targets[0].xpos.clone())
        qpos = (
            torch.tensor([[-0.97, 0.0]])
            if (recoverable and options.start_qpos[0, 0] > -0.9)
            else torch.tensor([[-0.999 if primary_safe else 0.5, 0.0]])
        )
        return motion.PlanResult(
            success=torch.tensor([True]),
            positions=torch.stack((options.start_qpos, qpos), dim=1),
            dt=torch.tensor([[0.0, 0.04]]),
        )

    robot = SimpleNamespace(
        get_qpos_limits=lambda **kw: torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]]),
    )
    monkeypatch.setattr(motion.MotionGenerator, "generate", solve)
    monkeypatch.setattr(motion, "_joint_velocity_limits", lambda *a: torch.ones(1, 2))
    options = motion.MotionGenOptions(
        strategy="ik_interp", control_part="arm", start_qpos=seed, interpolation_dt=0.04
    )
    rng = torch.get_rng_state().clone()
    result = motion._local_place_ik(
        SimpleNamespace(robot=robot),
        [motion.PlanState(move_type=motion.MoveType.EEF_MOVE, xpos=pose)],
        options,
    )
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, pose) for value in calls)
    torch.testing.assert_close(seed, torch.tensor([[-0.99, 0.0]]))
    if recoverable:
        assert result.success.tolist() == [True]
        torch.testing.assert_close(
            result.positions, torch.tensor([[[-0.99, 0.0], [-0.97, 0.0]]])
        )
        torch.testing.assert_close(result.dt, torch.tensor([[0.0, 0.04]]))
    else:
        assert result is None


@pytest.mark.parametrize(
    "generator_name", ["CheckedMotionGenerator", "ApproachMotionGenerator"]
)
@pytest.mark.parametrize(
    "scoped,success,batch,recovered",
    [
        (False, False, 1, False),
        (True, True, 1, False),
        (True, False, 1, True),
        (True, False, 2, False),
    ],
)
def test_place_local_ik_only_runs_after_scoped_single_env_failure(
    monkeypatch, generator_name, scoped, success, batch, recovered
):
    from contextlib import nullcontext
    from embodichain.gen_sim.task_engine._task_program import motion

    generator = object.__new__(getattr(motion, generator_name))
    start = torch.eye(4).repeat(batch, 1, 1)
    generator.robot = SimpleNamespace(compute_fk=lambda **kw: start.clone())
    monkeypatch.setattr(
        motion, "_joint_velocity_limits", lambda *a: torch.ones(batch, 1)
    )
    original = motion.PlanResult(
        success=torch.full((batch,), success),
        positions=torch.zeros(batch, 5, 1),
        dt=torch.full((batch, 5), 0.04),
    )
    backend = Mock(return_value=original)
    monkeypatch.setattr(motion.MotionGenerator, "generate", backend)
    local = Mock(
        return_value=motion.PlanResult(
            success=torch.ones(batch, dtype=torch.bool),
            positions=original.positions,
            dt=original.dt,
        )
    )
    monkeypatch.setattr(motion, "_local_place_ik", local)
    options = motion.MotionGenOptions(
        strategy="ik_interp",
        control_part="arm",
        start_qpos=torch.zeros(batch, 1),
        sample_count=5,
        interpolation_dt=0.04,
    )
    target = start.clone()
    target[:, 0, 3] = 0.2
    targets = [motion.PlanState(move_type=motion.MoveType.EEF_MOVE, xpos=target)]
    with motion.place_ik_recovery() if scoped else nullcontext() as scope:
        result = generator.generate(targets, options)
        if scope is not None:
            assert scope.used is recovered
    assert local.called is recovered
    assert motion._PLACE_IK_RECOVERY.get() is None
    if recovered:
        assert result is local.return_value
        local_targets, local_options = local.call_args.args[1:]
        assert len(local_targets) == options.sample_count - 1
        assert local_options.sample_count == options.sample_count
        torch.testing.assert_close(local_targets[-1].xpos, targets[-1].xpos)
    else:
        assert result is original
    if scoped or generator_name == "ApproachMotionGenerator":
        expected_counts = [5] if success else [5, 260, 320]
        assert [
            call.kwargs["options"].sample_count for call in backend.call_args_list
        ] == expected_counts
        for call, sample_count in zip(backend.call_args_list, expected_counts):
            assert len(call.args[0]) == sample_count - 1
            assert call.kwargs["options"].preserve_cartesian_samples is True
            assert call.kwargs["options"].is_linear is True
            torch.testing.assert_close(call.args[0][-1].xpos, target)
    else:
        backend.assert_called_once_with(targets, options=options)
        assert backend.call_args.args[0] is targets
        assert backend.call_args.kwargs["options"] is options
    assert options.sample_count == 5
    assert options.preserve_cartesian_samples is False


@pytest.mark.parametrize(
    "generator_name", ["CheckedMotionGenerator", "ApproachMotionGenerator"]
)
@pytest.mark.parametrize("case", ["presampled", "non_ik", "joint_target"])
def test_place_scope_preserves_noneligible_planning_requests(
    monkeypatch, generator_name, case
):
    from embodichain.gen_sim.task_engine._task_program import motion

    generator = object.__new__(getattr(motion, generator_name))
    generator.robot = SimpleNamespace(compute_fk=Mock())
    monkeypatch.setattr(motion, "_joint_velocity_limits", lambda *a: torch.ones(1, 1))
    original = motion.PlanResult(
        success=torch.tensor([False]),
        positions=torch.zeros(1, 5, 1),
        dt=torch.full((1, 5), 0.04),
    )
    backend = Mock(return_value=original)
    local = Mock()
    monkeypatch.setattr(motion.MotionGenerator, "generate", backend)
    monkeypatch.setattr(motion, "_local_place_ik", local)
    options = motion.MotionGenOptions(
        strategy="motion_gen" if case == "non_ik" else "ik_interp",
        control_part="arm",
        start_qpos=torch.zeros(1, 1),
        sample_count=5,
        preserve_cartesian_samples=case == "presampled",
        interpolation_dt=0.04,
    )
    targets = [
        (
            motion.PlanState(
                move_type=motion.MoveType.JOINT_MOVE, qpos=torch.zeros(1, 1)
            )
            if case == "joint_target"
            else motion.PlanState(move_type=motion.MoveType.EEF_MOVE, xpos=torch.eye(4))
        )
    ]

    with motion.place_ik_recovery() as scope:
        result = generator.generate(targets, options)
        assert scope.used is False

    assert result is original
    backend.assert_called_once_with(targets, options=options)
    assert backend.call_args.args[0] is targets
    assert backend.call_args.kwargs["options"] is options
    local.assert_not_called()
    generator.robot.compute_fk.assert_not_called()
    assert motion._PLACE_IK_RECOVERY.get() is None


def test_place_planning_exception_does_not_leak_recovery_scope(monkeypatch):
    from embodichain.gen_sim.task_engine._task_program import motion
    from embodichain.gen_sim.task_engine._task_program.actions import GenSimPlace
    from embodichain.lab.sim.atomic_actions.primitives.place import Place

    def fail(*args):
        assert motion._PLACE_IK_RECOVERY.get() is not None
        raise RuntimeError("planner failed")

    monkeypatch.setattr(Place, "_plan", fail)
    with pytest.raises(RuntimeError, match="planner failed"):
        GenSimPlace()._plan(object(), object())
    assert motion._PLACE_IK_RECOVERY.get() is None


def test_stack_place_allows_equivalent_tcp_roll_without_moving_release(
    monkeypatch,
) -> None:
    from embodichain.gen_sim.task_engine._task_program import stack_place
    from embodichain.lab.sim.atomic_actions.primitives.place import PlaceGoal
    from embodichain.lab.task_program.compiler.lowering import SemanticLowering
    from embodichain.lab.task_program.semantics.calls import RegisteredSemanticCall

    pose = torch.eye(4)
    pose[2, 3] = 1.20
    monkeypatch.setattr(
        stack_place._RelativePlaceLowerer,
        "lower",
        lambda *a, **kw: SemanticLowering(goal=PlaceGoal(xpos=pose)),
    )
    lowerer = object.__new__(stack_place._StackPlaceLowerer)
    lowerer._approach = 0.02
    lowerer._release = 0.003
    lowerer._nominal = 0.01

    result = lowerer.lower(
        RegisteredSemanticCall(call_id="gen_sim.stack_place", arguments={}),
        context=SimpleNamespace(batch_size=1),
        bound=object(),
        option_template=object(),
    )

    assert result.goal.tcp_symmetry == "z_roll_180"
    torch.testing.assert_close(
        result.goal.xpos[0, :, 2, 3], torch.tensor([1.21, 1.193])
    )


@pytest.mark.parametrize(
    "recovered,unsafe", [(False, True), (True, False), (True, True)]
)
def test_recovered_place_checks_final_resampled_commands(
    monkeypatch, recovered, unsafe
):
    from embodichain.gen_sim.task_engine._task_program import motion
    from embodichain.gen_sim.task_engine._task_program.actions import GenSimPlace
    from embodichain.lab.sim.atomic_actions.primitives.place import Place

    action = GenSimPlace()
    action._planning_services = SimpleNamespace(robot=object())
    failed = object()
    monkeypatch.setattr(action, "failed_plan", Mock(return_value=failed))
    monkeypatch.setattr(motion, "_joint_velocity_limits", lambda *a: torch.ones(1, 1))
    plan = SimpleNamespace(
        plan_success=torch.tensor([True]),
        joint_trajectory=SimpleNamespace(
            positions=torch.tensor([[[0.0], [0.2 if unsafe else 0.02]]]),
            dt=torch.tensor([[0.0, 0.04]]),
        ),
    )

    def generate(*args):
        motion._PLACE_IK_RECOVERY.get().used = recovered
        return plan

    monkeypatch.setattr(Place, "_plan", generate)
    result = action._plan(object(), object())
    assert result is (failed if recovered and unsafe else plan)
    assert motion._PLACE_IK_RECOVERY.get() is None


def test_velocity_limits_use_urdf_names_and_preserve_stricter_runtime_limits(
    tmp_path,
) -> None:
    from embodichain.gen_sim.task_engine._task_program.motion import (
        _joint_velocity_limits,
    )

    path = tmp_path / "robot.urdf"
    path.write_text(
        '<robot><joint name="a"><limit velocity="2"/></joint><joint name="b"><limit velocity="3"/></joint></robot>'
    )
    robot = SimpleNamespace(
        cfg=SimpleNamespace(fpath=path),
        joint_names=["a", "b"],
        get_joint_ids=lambda **kw: [1, 0],
        get_qvel_limits=lambda **kw: torch.tensor([[1e10, 1.0]]),
    )
    torch.testing.assert_close(
        _joint_velocity_limits(robot, "arm"), torch.tensor([[3.0, 1.0]])
    )
    robot.joint_names = ["missing", "b"]
    with pytest.raises(ValueError, match="URDF velocity limit"):
        _joint_velocity_limits(robot, "arm")


def test_curobo_adapter_allocates_only_declared_mesh_cache(monkeypatch) -> None:
    from embodichain.gen_sim.task_engine._task_program import assembly
    from embodichain.lab.sim.motion import motion_generator
    from embodichain.lab.task_program.semantics import SceneCollisionRole

    cfgs = []
    monkeypatch.setattr(
        motion_generator, "MotionGenerator", lambda cfg: cfgs.append(cfg) or object()
    )
    captured = {}

    def factory(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(create_adapter=lambda: "adapter")

    monkeypatch.setattr(assembly, "_TaskFactory", factory)
    monkeypatch.setattr(assembly, "install_grasp_filters", lambda *a: {})
    bindings = [
        SimpleNamespace(
            entity_id=f"object_{i}",
            simulation_uid=f"native_{i}",
            collision_role=SceneCollisionRole.DYNAMIC,
        )
        for i in range(3)
    ]
    registration = SimpleNamespace(
        assert_unchanged=lambda: None,
        robot_profile_binding=SimpleNamespace(
            presets=[
                SimpleNamespace(motion_policy=SimpleNamespace(strategy="motion_gen"))
            ]
        ),
        scene_binding=SimpleNamespace(rigid_objects=bindings),
    )
    objects = {b.simulation_uid: Mock() for b in bindings}
    env = SimpleNamespace(
        robot=SimpleNamespace(uid="robot"),
        sim=SimpleNamespace(get_rigid_object=objects.get),
        step_dt=0.04,
    )
    adapter = assembly.TaskAdapterFactory(registration, "test", (), ())
    assert adapter.create_adapter(env) == "adapter"
    captured["motion_generator_factory"]()
    world = cfgs[0].planner_cfg.world
    assert world.representation == "auto"
    assert set(world.rigid_objects) == {b.entity_id for b in bindings}
    assert world.dynamic_obstacle_names == [b.entity_id for b in bindings]


def test_relative_place_corrects_goal_without_mutating_grasp_state(monkeypatch) -> None:
    from embodichain.gen_sim.task_engine._task_program import services
    from embodichain.lab.sim.atomic_actions.primitives.place import (
        PlaceGoal,
        PlaceOptions,
    )
    from embodichain.lab.task_program.compiler.lowering import SemanticLowering

    object_pose = torch.eye(4).repeat(2, 1, 1)
    object_pose[:, 0, 3] = torch.tensor([0.2, 0.4])
    measured_eef = object_pose.clone()
    measured_eef[:, 2, 3] = torch.tensor([0.1, 0.15])
    nominal = torch.eye(4).repeat(2, 1, 1)
    canonical = SemanticLowering(
        goal=PlaceGoal(
            xpos=services.SceneEntityPose(
                "table",
                relative_pose=nominal,
                world_displacement=torch.tensor([0.0, 0.0, 0.1]),
                world_orientation=object_pose[:, :3, :3].clone(),
            )
        )
    )
    monkeypatch.setattr(
        services._RelativePlaceLowerer, "lower", lambda *a, **kw: canonical
    )
    robot = Mock()
    robot.compute_fk.return_value = measured_eef
    lowerer = services._ObservedRelativePlaceLowerer(
        (
            services._RelativePlaceRoute(
                object_id="cup",
                reference_entity_id="table",
                relation="on",
                world_displacement=(0.0, 0.0, 0.1),
                world_yaw_offset=math.pi / 6.0,
            ),
        ),
        robot,
    )
    context = SimpleNamespace(
        batch_size=2,
        env_ids=torch.tensor([1, 3]),
        robot=SimpleNamespace(qpos=torch.zeros(2, 4)),
        scene=SimpleNamespace(
            entities={"cup": SimpleNamespace(pose=object_pose, confidence=1.0)}
        ),
    )
    bound = SimpleNamespace(
        binding=SimpleNamespace(
            resources={
                "primary": SimpleNamespace(
                    endpoints={
                        "motion": SimpleNamespace(
                            runtime_target=SimpleNamespace(
                                joint_ids=(1, 3), control_part="arm"
                            )
                        )
                    }
                )
            }
        )
    )
    result = lowerer.lower(
        RegisteredSemanticCall(
            call_id="simulation.place_relative",
            arguments={"object": "cup", "reference": "table", "relation": "on"},
        ),
        context=context,
        bound=bound,
        option_template=PlaceOptions(),
    )
    torch.testing.assert_close(
        object_pose @ result.goal.xpos.relative_pose, measured_eef
    )
    angle = torch.tensor(math.pi / 6.0)
    yaw = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    torch.testing.assert_close(
        result.goal.xpos.world_orientation,
        yaw @ object_pose[:, :3, :3],
    )
    torch.testing.assert_close(canonical.goal.xpos.relative_pose, nominal)
    torch.testing.assert_close(
        canonical.goal.xpos.world_orientation, object_pose[:, :3, :3]
    )
    assert result.registered_effect is canonical.registered_effect
    assert robot.compute_fk.call_args.kwargs["env_ids"] == [1, 3]


@pytest.mark.parametrize("success", [True, False])
def test_motion_diagnostics_preserve_the_underlying_result(
    monkeypatch, success
) -> None:
    import json
    from embodichain.gen_sim.task_engine._task_program import motion

    generator = object.__new__(motion.ApproachMotionGenerator)
    result = motion.PlanResult(success=torch.tensor([success]))
    backend = Mock(return_value=result)
    log = Mock()
    monkeypatch.setattr(motion.MotionGenerator, "generate", backend)
    monkeypatch.setattr(motion.logger, "log_info", log)
    targets = [motion.PlanState(move_type=motion.MoveType.EEF_MOVE, xpos=torch.eye(4))]
    options = motion.MotionGenOptions(control_part="left_arm")
    assert generator.generate(targets, options) is result
    backend.assert_called_once_with(targets, options=options)
    record = json.loads(log.call_args.args[0].removeprefix("GenSim motion plan: "))
    assert record == {
        "control_part": "left_arm",
        "input_target_count": 1,
        "planned_target_count": 1,
        "input_poses": [torch.eye(4).tolist()],
        "success": [success],
    }


@pytest.mark.parametrize("verify_retention", [False, True])
def test_current_pose_upright_binding_preserves_each_environments_position(
    verify_retention: bool,
) -> None:
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[:, :3, 3] = torch.tensor([[0.1, -0.2, 1.0], [0.3, 0.2, 1.2]])
    before = poses.clone()
    robot = Mock()
    lowerer = _AlignHeldFactory(
        (("can", "current_object_pose", True, (1.0, 0.0, 0.0), None),),
        verify_retention=verify_retention,
    ).create(
        simulation=None,
        robot=robot,
        scene_registry=Mock(),
        engine=SimpleNamespace(robot=robot),
    )
    context = SimpleNamespace(
        batch_size=2,
        task=SimpleNamespace(
            get_held_object=lambda key: SimpleNamespace(
                semantics=SimpleNamespace(entity_id="can")
            )
        ),
        scene=SimpleNamespace(
            entities={"can": SimpleNamespace(pose=poses, confidence=1.0)}
        ),
    )
    bound = SimpleNamespace(
        binding=SimpleNamespace(
            resources={
                "primary": SimpleNamespace(
                    endpoints={"motion": SimpleNamespace(task_state_key="left")}
                )
            }
        )
    )
    result = lowerer.lower(
        RegisteredSemanticCall(
            call_id="gen_sim.align_held",
            arguments={
                "object": "can",
                "target": "current_object_pose",
                "preserve_yaw": True,
            },
        ),
        context=context,
        bound=bound,
        option_template=MoveHeldObjectOptions(),
    )
    target = result.goal.object_target_pose
    torch.testing.assert_close(target[:, :3, 3], before[:, :3, 3])
    torch.testing.assert_close(
        target[:, :3, 0],
        torch.tensor([[0.0, 0.0, 1.0]]).repeat(2, 1),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(poses, before)
    assert robot.mock_calls == []
    assert lowerer.preserves_symbolic_state is not verify_retention
    if verify_retention:
        assert lowerer.effect_contract_kind is SemanticEffectKind.ATTACH
        assert result.registered_effect.effect_kind is SemanticEffectKind.ATTACH
        effect = result.registered_effect.held_objects[0]
        assert effect.object_id == "can"
        assert effect.relation is HeldObjectRelation.ATTACHED
        assert effect.slot_id == "primary"
    else:
        assert lowerer.effect_contract_kind is None
        assert result.registered_effect is None


@pytest.mark.parametrize("verify_retention", [False, True])
@pytest.mark.parametrize("height", [1.3, 0.8])
def test_upright_alignment_lifts_before_rotating(
    verify_retention: bool, height: float
) -> None:
    pose = torch.eye(4).unsqueeze(0)
    pose[:, :3, :3] = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    pose[:, :3, 3] = torch.tensor([0.1, 0.2, 1.0])
    original = pose.clone()
    robot = SimpleNamespace(
        cfg=SimpleNamespace(solver_cfg={"arm": SimpleNamespace(root_link_name="base")}),
        get_link_pose=lambda **kw: torch.eye(4).unsqueeze(0),
    )
    lowerer = _AlignHeldFactory(
        (("can", "staging", False, (0.0, 0.0, 1.0), (0.1, 0.2, height)),),
        verify_retention=verify_retention,
    ).create(
        simulation=None,
        robot=robot,
        scene_registry=Mock(),
        engine=SimpleNamespace(robot=robot),
    )
    held = SimpleNamespace(
        semantics=SimpleNamespace(entity_id="can"),
        object_to_eef=torch.eye(4).unsqueeze(0),
    )
    context = SimpleNamespace(
        batch_size=1,
        env_ids=torch.tensor([0]),
        task=SimpleNamespace(get_held_object=lambda key: held),
        scene=SimpleNamespace(
            entities={"can": SimpleNamespace(pose=pose, confidence=1.0)}
        ),
    )
    bound = SimpleNamespace(
        binding=SimpleNamespace(
            resources={
                "primary": SimpleNamespace(
                    endpoints={
                        "motion": SimpleNamespace(
                            task_state_key="arm",
                            runtime_target=SimpleNamespace(control_part="arm"),
                        )
                    }
                )
            }
        )
    )
    result = lowerer.lower(
        RegisteredSemanticCall(
            call_id="gen_sim.align_held",
            arguments={
                "object": "can",
                "target": "staging",
                "preserve_yaw": False,
            },
        ),
        context=context,
        bound=bound,
        option_template=MoveHeldObjectOptions(),
    )
    waypoints = result.goal.object_target_pose
    assert waypoints.shape == (1, 2, 4, 4)
    torch.testing.assert_close(waypoints[:, 0, :3, :3], original[:, :3, :3])
    torch.testing.assert_close(
        waypoints[:, :, :3, 3],
        torch.tensor([[[0.1, 0.2, max(1.0, height)], [0.1, 0.2, height]]]),
    )
    assert result.goal.world_yaw_free
    torch.testing.assert_close(
        waypoints[:, 1, :3, 2], torch.tensor([[0.0, 0.0, 1.0]]), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(pose, original)
    assert (result.registered_effect is not None) is verify_retention


def test_coordinated_legacy_reference_tuple_still_normalizes():
    from embodichain.gen_sim.task_engine._task_program.services import (
        _coordinated_transport_route,
    )

    route = _coordinated_transport_route(
        ("tray", "goal", "table", tuple(torch.eye(4).flatten().tolist())), index=0
    )
    assert route.reference_entity_id == "table"
    assert route.relation is None


def test_coordinated_on_follows_observed_support_and_preserves_object_heading():
    from embodichain.gen_sim.task_engine._task_program.services import (
        _CoordinatedTransportLowerer,
    )
    from embodichain.lab.sim.atomic_actions.goals import resolve_pose_goal

    route = {
        "object_id": "small",
        "target_id": "landing",
        "reference_entity_id": "large",
        "relation": "on",
        "world_displacement": [0, 0, 0.06],
    }
    factory = decode_task_lowerer(
        {"kind": "coordinated_transport", "routes": [route]}, path="test"
    )
    semantics = ObjectSemantics(
        affordance=AntipodalAffordance(), geometry={}, label="plate", entity_id="small"
    )
    lowerer = _CoordinatedTransportLowerer(factory.routes, (semantics,))
    obj = torch.eye(4)[None]
    obj[:, :2, :2] = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    ref = torch.eye(4)[None]
    ref[:, :3, 3] = torch.tensor([0.2, -0.3, 0.7])
    context = PlanningContext(
        robot=RobotObservation(
            timestamp=1.0, qpos=torch.zeros(1, 1), qvel=torch.zeros(1, 1)
        ),
        task=TaskState(batch_size=1, device="cpu"),
        scene=SceneSnapshot(
            timestamp=1.0,
            version=1,
            entities={"small": EntityState(obj), "large": EntityState(ref)},
        ),
        env_ids=torch.tensor([0]),
    )
    call = RegisteredSemanticCall(
        call_id="simulation.coordinated_transport",
        arguments={
            "object": "small",
            "target": "landing",
            "reference": "large",
            "relation": "on",
        },
    )
    result = lowerer.lower(
        call,
        context=context,
        bound=None,
        option_template=CoordinatedPickmentOptions(release=True),
    )
    resolved = resolve_pose_goal(
        result.goal.object_target_pose, context=context, name="target"
    )
    torch.testing.assert_close(resolved[:, :3, 3], torch.tensor([[0.2, -0.3, 0.76]]))
    torch.testing.assert_close(resolved[:, :3, :3], obj[:, :3, :3])
    from dataclasses import replace

    shifted = ref.clone()
    shifted[:, 0, 3] += 0.1
    moved_context = replace(
        context,
        scene=SceneSnapshot(
            timestamp=2.0,
            version=2,
            entities={"small": EntityState(obj), "large": EntityState(shifted)},
        ),
    )
    moved = resolve_pose_goal(
        result.goal.object_target_pose, context=moved_context, name="target"
    )
    torch.testing.assert_close(moved[:, :3, 3], torch.tensor([[0.3, -0.3, 0.76]]))
    assert all(
        e.relation is HeldObjectRelation.DETACHED
        for e in result.registered_effect.held_objects
    )
    with pytest.raises(ValueError, match="on placement"):
        decode_task_lowerer(
            {"kind": "coordinated_hold", "routes": [route]}, path="test"
        )
    with pytest.raises(ValueError):
        decode_task_lowerer(
            {
                "kind": "coordinated_transport",
                "routes": [{**route, "relation": "inside"}],
            },
            path="test",
        )


def test_coordinated_hold_lowerer_retains_both_verified_attachments() -> None:
    semantics = ObjectSemantics(
        affordance=AntipodalAffordance(),
        geometry={},
        label="tray",
        entity_id="tray",
    )
    lowerer = _CoordinatedHoldLowerer(
        (
            _CoordinatedTransportRoute(
                object_id="tray",
                target_id="tray_up",
                world_displacement=(0.0, 0.0, 0.14),
            ),
        ),
        (semantics,),
    )

    lowering = lowerer.lower(
        RegisteredSemanticCall(
            call_id="simulation.coordinated_hold",
            arguments={
                "object": "tray",
                "target": "tray_up",
                "world_displacement": [0.0, 0.0, 0.14],
            },
        ),
        context=PlanningContext(
            robot=RobotObservation(
                timestamp=1.0,
                qpos=torch.zeros((1, 1)),
                qvel=torch.zeros((1, 1)),
            ),
            task=TaskState(batch_size=1, device="cpu"),
            scene=SceneSnapshot(
                timestamp=1.0,
                version=1,
                entities={"tray": EntityState(torch.eye(4).unsqueeze(0))},
            ),
            env_ids=torch.tensor([0]),
        ),
        bound=None,  # type: ignore[arg-type]
        option_template=CoordinatedPickmentOptions(release=False),
    )

    assert lowering.registered_effect is not None
    assert lowering.registered_effect.effect_kind is SemanticEffectKind.ATTACH
    assert [effect.relation for effect in lowering.registered_effect.held_objects] == [
        HeldObjectRelation.ATTACHED,
        HeldObjectRelation.ATTACHED,
    ]
    factory = decode_task_lowerer(
        {
            "kind": "coordinated_hold",
            "routes": [
                {
                    "object_id": "tray",
                    "target_id": "tray_up",
                    "world_displacement": [0.0, 0.0, 0.14],
                }
            ],
        },
        path="integration.runtime_services.registered_semantic_lowerers[0]",
    )
    assert factory.call_id == "simulation.coordinated_hold"


def test_configured_handover_uses_baseline_timing_and_rejects_removed_wait_field() -> (
    None
):
    path = (
        Path(__file__).parents[3]
        / "embodichain_tasks/configs/tasks/manipulation/hand_over"
        / "task.dual_ur5_dh_pgi_140_80.yaml"
    )
    physical = _resolve_gym_components(load_config(path), base_dir=path.parent)
    _, task, policy = _resolve_task_program_components(
        physical.config["task_program"], base_dir=path.parent
    )
    payload = _compose_integration_payload(
        task=task,
        policy=policy,
        skill_profile=physical.embodiment_skill_profile,
        scene=task["scene_binding"],
    )["robot_profile"]
    payload["presets"][0]["action_options"]["hand_over"].update(
        retreat_distance=0.12, retreat_steps=28
    )
    before = deepcopy(payload)
    options = (
        _decode_robot_profile(payload).presets[0].action_option_templates["hand_over"]
    )

    assert payload == before
    assert type(options) is HandOverOptions
    assert options.retreat_distance == pytest.approx(0.12)
    assert options.retreat_steps == 28
    assert not hasattr(options, "source_release_settle_steps")
    payload["presets"][0]["action_options"]["hand_over"][
        "source_release_settle_steps"
    ] = 16
    with pytest.raises(ValueError, match="source_release_settle_steps"):
        _decode_robot_profile(payload)


def test_configured_transport_binds_one_baseline_pose_without_planning() -> None:
    pose = _AbsolutePoseTarget((0.1, 0.2, 0.8), (1.0, 0.0, 0.0, 0.0))
    lowerer = _MoveHeldObjectLowerer(
        (_MoveHeldObjectRoute("part", "inspection", pose),)
    )
    forbidden_robot = Mock()
    lowering = lowerer.lower(
        RegisteredSemanticCall(
            call_id="simulation.move_held_object",
            arguments={"object": "part", "target": "inspection"},
        ),
        context=forbidden_robot,
        bound=forbidden_robot,
        option_template=MoveHeldObjectOptions(),
    )
    goal = lowering.goal
    assert lowerer.effect_contract_kind is SemanticEffectKind.ATTACH
    effect = lowering.registered_effect
    assert effect is not None
    assert effect.effect_kind is SemanticEffectKind.ATTACH
    assert effect.held_objects[0].object_id == "part"
    assert effect.held_objects[0].relation is HeldObjectRelation.ATTACHED
    assert effect.held_objects[0].slot_id == "primary"
    forbidden_robot.assert_not_called()
    assert forbidden_robot.mock_calls == []
    assert type(goal) is HeldObjectPoseGoal
    torch.testing.assert_close(goal.object_target_pose, pose.to_matrix())
    assert not hasattr(goal, "alternative_object_target_poses")
    lookahead = lowerer.pick_lookahead_targets(
        RegisteredSemanticCall(
            call_id="simulation.move_held_object", arguments={"target": "inspection"}
        ),
        picked_object=SceneObjectRef("different_part"),
        bound=None,
        previous_target=None,
    )
    assert lookahead is None


def test_transport_decoder_rejects_old_alternative_pose_declarations() -> None:
    pose = {
        "kind": "pose",
        "position": [0.1, 0.2, 0.8],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    with pytest.raises(ValueError, match="alternatives"):
        decode_task_lowerer(
            {
                "kind": "move_held_object",
                "routes": [
                    {
                        "object_id": "part",
                        "target_id": "inspection",
                        "pose": pose,
                        "alternatives": [pose],
                    }
                ],
            },
            path="lowerer",
        )


def test_configured_pick_keeps_baseline_goal_and_preset_ownership() -> None:
    semantics = ObjectSemantics(
        affordance=AntipodalAffordance(), geometry={}, entity_id="part"
    )
    route = _PickRoute("part", "inspection")
    options = PickUpOptions(
        pick_object_part="bottom", approach_direction=torch.tensor([1.0, 0.0, 0.0])
    )
    lowering = _PickLowerer((route,), (semantics,)).lower(
        RegisteredSemanticCall(
            call_id="simulation.pick",
            arguments={"object": "part", "target": "inspection"},
        ),
        context=None,
        bound=None,
        option_template=options,
    )
    assert type(lowering.goal) is GraspGoal
    assert lowering.goal.semantics is semantics
    assert lowering.skill_options is None
    assert options.pick_object_part == "bottom"
    assert options.downstream_object_target_poses == ()
    assert lowering.registered_effect.effect_kind is SemanticEffectKind.ATTACH


def test_pick_decoder_rejects_removed_runtime_option_declarations() -> None:
    with pytest.raises(ValueError, match="required_object_target_poses"):
        decode_task_lowerer(
            {
                "kind": "pick",
                "routes": [
                    {
                        "object_id": "part",
                        "target_id": "inspection",
                        "required_object_target_poses": [],
                    }
                ],
            },
            path="pick",
        )


def test_task_pick_alias_factory_has_stable_immutable_identity() -> None:
    payload = {
        "kind": "pick",
        "call_id": "gen_sim.pick.step_01",
        "routes": [{"object_id": "part", "target_id": "release"}],
    }
    first = decode_task_lowerer(payload, path="pick")
    second = decode_task_lowerer(payload, path="pick")
    assert first.call_id == "gen_sim.pick.step_01"
    assert first.lowerer_type.call_id == first.call_id
    assert type(first).__qualname__ == type(second).__qualname__
    assert first.target_descriptor.skill_id == "pick_up"
    robot = object()
    registry = Mock()
    registry.resolve.return_value = SceneObjectRef("part")
    registry.object_semantics.return_value = ObjectSemantics(
        affordance=AntipodalAffordance(), geometry={}, entity_id="part"
    )
    lowerer = first.create(
        simulation=None,
        robot=robot,
        scene_registry=registry,
        engine=SimpleNamespace(robot=robot, grasp_pose_generators={}),
    )
    assert type(lowerer).call_id == first.call_id
    result = lowerer.lower(
        RegisteredSemanticCall(
            call_id=first.call_id, arguments={"object": "part", "target": "release"}
        ),
        context=None,
        bound=None,
        option_template=PickUpOptions(),
    )
    assert type(result.goal) is GraspGoal
    assert result.skill_options is None
    with pytest.raises(FrozenInstanceError):
        first.routes = ()


@pytest.mark.parametrize(
    "kind",
    [
        "articulation_link_slide",
        "articulation_link_press",
        "articulation_link_twist",
        "release_safe_pick",
        "move_held_object_upright",
    ],
)
def test_task_decoder_rejects_out_of_scope_services(kind: str) -> None:
    with pytest.raises(ValueError, match="unsupported Task Engine service"):
        decode_task_lowerer({"kind": kind}, path="lowerer")


@pytest.mark.parametrize("articulated", [False, True])
def test_relative_place_observes_either_native_reference_root(articulated):
    from embodichain.lab.task_program.semantics import (
        SceneRegistry,
        SceneArticulationRef,
        SceneObjectRef,
    )
    from embodichain.lab.sim.atomic_actions import PlaceOptions
    from embodichain.lab.sim.atomic_actions.goals import resolve_pose_goal

    cup_pose = torch.eye(4).repeat(2, 1, 1)
    reference_pose = cup_pose.clone()
    cup = SimpleNamespace(get_local_pose=lambda **kw: cup_pose.clone())
    reference = SimpleNamespace(
        get_local_pose=lambda **kw: reference_pose.clone(),
        joint_names=("press",),
        get_qpos=lambda **kw: torch.zeros(2, 1),
    )
    rigid = {"cup": cup, **({} if articulated else {"anchor": reference})}
    simulation = SimpleNamespace(
        get_rigid_object=rigid.get, get_articulation=lambda uid: reference
    )
    registry = SceneRegistry.from_simulation(
        simulation,
        rigid_objects={uid: uid for uid in rigid},
        articulations={"anchor": "anchor"} if articulated else {},
    )
    assert type(registry.resolve("anchor")) is (
        SceneArticulationRef if articulated else SceneObjectRef
    )
    robot = SimpleNamespace(compute_fk=lambda **kw: cup_pose.clone())
    factory = decode_task_lowerer(
        {
            "kind": "place_relative",
            "routes": [
                {
                    "object_id": "cup",
                    "reference_entity_id": "anchor",
                    "relation": "behind",
                    "world_displacement": [0.2, 0.0, 0.0],
                }
            ],
        },
        path="relative",
    )
    lowerer = factory.create(
        simulation=simulation,
        robot=robot,
        scene_registry=registry,
        engine=SimpleNamespace(robot=robot),
    )
    provider = registry.make_scene_provider(batch_size=2)
    env_ids = torch.tensor([0, 1])
    held = SimpleNamespace(
        semantics=SimpleNamespace(entity_id="cup"),
        object_to_eef=torch.eye(4).repeat(2, 1, 1),
    )
    context = SimpleNamespace(
        scene=provider.snapshot(timestamp=0.0, env_ids=env_ids),
        task=SimpleNamespace(get_held_object=lambda key: held),
        robot=SimpleNamespace(qpos=torch.zeros(2, 1)),
        batch_size=2,
        env_ids=env_ids,
    )
    bound = SimpleNamespace(
        binding=SimpleNamespace(
            resources={
                "primary": SimpleNamespace(
                    endpoints={
                        "motion": SimpleNamespace(
                            task_state_key="arm",
                            runtime_target=SimpleNamespace(
                                joint_ids=(0,), control_part="arm"
                            ),
                        )
                    }
                )
            }
        )
    )
    result = lowerer.lower(
        RegisteredSemanticCall(
            call_id="simulation.place_relative",
            arguments={"object": "cup", "reference": "anchor", "relation": "behind"},
        ),
        context=context,
        bound=bound,
        option_template=PlaceOptions(preserve_current_object_orientation=True),
    )
    assert result.goal.xpos.entity_id == "anchor"
    before = resolve_pose_goal(result.goal.xpos, context, name="release")
    reference_pose[1, 0, 3] = 0.1
    context.scene = provider.snapshot(timestamp=1.0, env_ids=env_ids)
    after = resolve_pose_goal(result.goal.xpos, context, name="release")
    torch.testing.assert_close(
        after[:, 0, 3] - before[:, 0, 3], torch.tensor([0.0, 0.1])
    )
    if articulated:
        config = {
            "kind": "place_relative",
            "routes": [
                {
                    "object_id": "cup",
                    "reference_entity_id": "anchor",
                    "relation": "on",
                    "world_displacement": [0.0, 0.0, 0.1],
                }
            ],
        }
        with pytest.raises(ValueError, match="qualified link"):
            decode_task_lowerer(config, path="support").create(
                simulation=simulation,
                robot=robot,
                scene_registry=registry,
                engine=SimpleNamespace(robot=robot),
            )
