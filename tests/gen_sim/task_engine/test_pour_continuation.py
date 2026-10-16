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

"""CPU contracts for typed Pour recipes and shared candidate/actual geometry."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from embodichain.gen_sim.task_engine._task_program import actions
from embodichain.gen_sim.task_engine._task_program.pour_continuation import (
    CandidatePreview,
    PourContinuation,
    bind_calls,
    observed_pose,
    pad_line_world,
    prepare_preview,
    supports_ordinary_pour,
)
from embodichain.gen_sim.task_engine._task_program.pour_geometry import (
    pour_frames,
    staging_pose,
)
from embodichain.gen_sim.task_engine._task_program.pour_services import (
    PourStageRoute,
    PourTransportFactory,
)
from embodichain.gen_sim.task_engine._task_program.services import (
    _AbsolutePoseTarget,
    _MoveHeldObjectRoute,
    _SceneEntityTarget,
)
from embodichain.gen_sim.task_engine.semantic_planner import _initial_quaternion_xyzw
from embodichain.lab.sim.atomic_actions import JointPositionTarget, PickUpOptions
from embodichain.lab.sim.atomic_actions.plans import PlannerDiagnostics
from embodichain.lab.sim.atomic_actions.primitives.pick_up import PickUp
from embodichain.lab.sim.atomic_actions.primitives.pour import PourOptions
from embodichain.lab.task_program.compiler.program import (
    CompiledTaskProgramCall,
    CompiledTaskProgramSegment,
)
from embodichain.lab.task_program.semantics import (
    Pick,
    Place,
    RegisteredSemanticCall,
    SceneObjectRef,
    SemanticPose,
)

_GRAVITY = (0.0, 0.0, -9.81)
_ROLES = ("staging", "peak", "restore", "return_transport", "return_place")
_SEGMENT_IDS = ("pick", "stage", "pour", "return", "place")
_PAD_NAMES = {
    "left": ("left_inner_finger_pad", "left_right_inner_finger_pad"),
    "right": ("right_left_inner_finger_pad", "right_inner_finger_pad"),
}


def _program(calls):
    segments = tuple(
        CompiledTaskProgramSegment(
            segment_index=index,
            segment_id=name,
            name=name,
            calls=(CompiledTaskProgramCall(index, 0, call, ("steps", index)),),
            source_path=("steps", index),
        )
        for index, (name, call) in enumerate(zip(_SEGMENT_IDS, calls, strict=True))
    )
    return SimpleNamespace(program_id="pour_test", iter_segments=lambda: iter(segments))


def _binding_inputs():
    resources = {"primary": "left"}
    returned = SemanticPose((0.1, 0.2, 0.3), (0, 0, 0, 1))
    calls = [
        Pick(SceneObjectRef("source"), resources=resources),
        RegisteredSemanticCall(
            call_id="simulation.move_held_object",
            arguments={"object": "source", "target": "stage", "reference": "receiver"},
            resources=resources,
        ),
        RegisteredSemanticCall(
            call_id="simulation.pour",
            arguments={"object": "source"},
            resources=resources,
        ),
        RegisteredSemanticCall(
            call_id="simulation.move_held_object",
            arguments={"object": "source", "target": "return"},
            resources=resources,
        ),
        Place(SceneObjectRef("source"), at=returned, resources=resources),
    ]
    factory = PourTransportFactory(
        routes=(
            _MoveHeldObjectRoute("source", "stage", _SceneEntityTarget("receiver")),
            _MoveHeldObjectRoute(
                "source",
                "return",
                _AbsolutePoseTarget((0.1, 0.2, 0.3), (0, 0, 0, 1)),
            ),
        ),
        staging_routes=(PourStageRoute("source", "stage", "receiver", 0.025),),
    )
    preset = SimpleNamespace(
        action_option_template=Mock(return_value=PourOptions(rotate_angle=1.0))
    )
    return calls, factory, preset, (("pour_test/pour:0", 11),)


def _bind(calls, factory, preset, samples):
    return bind_calls(_program(calls), preset, factory, samples)


def _vertices():
    return np.array(list(product((-0.02, 0.03), (-0.015, 0.025), (-0.04, 0.08))))


def _native_source_pose():
    # Actual f32 native rigid pose: retain its roundoff, never project R onto SO(3).
    return np.array(
        [
            [0.00506109, -0.99995047, -0.0085645225, 0.1801715],
            [0.9999847, 0.0050800443, -0.0021989949, -0.0802346],
            [0.002242394, -0.008553262, 0.9999609, 0.67571634],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def _geometry():
    source = _vertices()
    receiver = source * np.array([3.0, 3.0, 0.5])
    receiver_pose = np.eye(4)
    receiver_pose[:3, 3] = [0.4, -0.1, 0.5]
    stage = staging_pose(
        source,
        receiver,
        _native_source_pose(),
        receiver_pose,
        clearance_m=0.025,
        gravity_world=_GRAVITY,
    )
    returned = torch.eye(4, dtype=torch.float64)
    returned[:3, 3] = torch.tensor([0.1, 0.2, 0.3])
    route = PourContinuation(
        "pick", "stage", "pour", "source", "receiver", "left", returned, 0.025, 1.0, 11
    )
    return source, receiver, receiver_pose, stage, route


def _transform():
    transform = torch.eye(4, dtype=torch.float64)
    transform[:3, :3] = torch.as_tensor(
        Rotation.from_euler("xyz", [0.2, -0.1, 0.3]).as_matrix()
    )
    transform[:3, 3] = torch.tensor([0.01, -0.03, 0.12])
    return transform


def _robot(prefix, tcp, jaw_world):
    names = _PAD_NAMES[prefix]
    poses = {name: tcp.clone() for name in names}
    poses[names[0]][:, :3, 3] -= jaw_world / 2
    poses[names[1]][:, :3, 3] += jaw_world / 2
    return SimpleNamespace(
        link_names=names,
        get_link_pose=Mock(side_effect=lambda name, **kwargs: poses[name]),
        compute_fk=Mock(return_value=tcp.clone()),
    )


def _context(source_pose, receiver_pose):
    return SimpleNamespace(
        batch_size=1,
        robot=SimpleNamespace(qpos=torch.zeros(1, 7, dtype=torch.float64)),
        scene=SimpleNamespace(
            entities={
                "source": SimpleNamespace(pose=torch.as_tensor(source_pose)[None]),
                "receiver": SimpleNamespace(pose=torch.as_tensor(receiver_pose)),
            }
        ),
    )


def test_exact_compiled_five_role_recipe_binds_original_ids_and_budget():
    calls, factory, preset, samples = _binding_inputs()
    routes = _bind(calls, factory, preset, samples)
    assert len(routes) == 1
    route = routes[0]
    assert (route.pick_id, route.stage_id, route.pour_id) == (
        "pour_test/pick:0",
        "pour_test/stage:0",
        "pour_test/pour:0",
    )
    assert (route.object_id, route.receiver_id, route.resource) == (
        "source",
        "receiver",
        "left",
    )
    assert (route.clearance_m, route.angle_rad, route.sample_count) == (0.025, 1.0, 11)
    assert torch.equal(route.return_pose, calls[-1].at.to_matrix())
    preset.action_option_template.assert_called_once_with("simulation.pour")


def test_owned_stage_without_complete_recipe_is_not_silently_legacy():
    calls, factory, preset, samples = _binding_inputs()
    complete = tuple(_program(calls).iter_segments())
    program = SimpleNamespace(
        program_id="pour_test", iter_segments=lambda: iter(complete[:-1])
    )
    with pytest.raises(ValueError, match="lacks a bound Pick continuation"):
        bind_calls(program, preset, factory, samples)


@pytest.mark.parametrize("angles", [(87, 29, 87 / 7), (179, -83, 41), (23, 61, -117)])
@pytest.mark.parametrize("quaternion_sign", [1, -1])
def test_same_declared_return_rotation_survives_float32_pose_snapshots(
    angles, quaternion_sign
):
    calls, factory, preset, samples = _binding_inputs()
    quaternion = tuple(_initial_quaternion_xyzw(angles))
    returned = SemanticPose((0.1, 0.2, 0.3), quaternion)
    for _ in range(3):
        returned = returned.snapshot()
    calls[-1] = replace(calls[-1], at=returned)
    target = _AbsolutePoseTarget(
        (0.1, 0.2, 0.3), tuple(quaternion_sign * value for value in quaternion)
    )
    factory = replace(
        factory, routes=(factory.routes[0], replace(factory.routes[1], pose=target))
    )
    route = _bind(calls, factory, preset, samples)[0]
    torch.testing.assert_close(
        route.return_pose, calls[-1].at.to_matrix(), rtol=0, atol=1e-6
    )


@pytest.mark.parametrize("difference", ["position", "rotation"])
def test_different_declared_return_pose_is_not_float_roundoff(difference):
    calls, factory, preset, samples = _binding_inputs()
    position = (0.1, 0.2, 0.3001) if difference == "position" else (0.1, 0.2, 0.3)
    quaternion = (
        tuple(Rotation.from_rotvec([1e-4, 0, 0]).as_quat())
        if difference == "rotation"
        else (0, 0, 0, 1)
    )
    calls[-1] = replace(calls[-1], at=SemanticPose(position, quaternion))
    with pytest.raises(ValueError, match="return poses disagree"):
        _bind(calls, factory, preset, samples)


@pytest.mark.parametrize("role", range(5))
def test_recipe_rejects_foreign_subject_at_every_role(role):
    calls, factory, preset, samples = _binding_inputs()
    call = calls[role]
    calls[role] = (
        replace(call, object=SceneObjectRef("foreign"))
        if role in (0, 4)
        else replace(call, arguments={**call.arguments, "object": "foreign"})
    )
    with pytest.raises(ValueError, match="same-object"):
        _bind(calls, factory, preset, samples)


@pytest.mark.parametrize("role", range(1, 5))
def test_recipe_rejects_foreign_resource_at_every_continuation(role):
    calls, factory, preset, samples = _binding_inputs()
    calls[role] = replace(calls[role], resources={"primary": "right"})
    with pytest.raises(ValueError, match="same-object"):
        _bind(calls, factory, preset, samples)


@pytest.mark.parametrize(
    "change",
    [
        "receiver",
        "extra_stage_argument",
        "extra_pour_argument",
        "return_target",
        "return_pose",
    ],
)
def test_recipe_rejects_mismatched_receiver_suffix_or_return(change):
    calls, factory, preset, samples = _binding_inputs()
    if change in ("receiver", "extra_stage_argument"):
        extra = (
            {"reference": "foreign"} if change == "receiver" else {"extra": "foreign"}
        )
        calls[1] = replace(calls[1], arguments={**calls[1].arguments, **extra})
    elif change == "extra_pour_argument":
        calls[2] = replace(
            calls[2], arguments={**calls[2].arguments, "receiver": "foreign"}
        )
    elif change == "return_target":
        calls[3] = replace(
            calls[3], arguments={**calls[3].arguments, "target": "foreign"}
        )
    else:
        calls[4] = replace(calls[4], at=SemanticPose((0.1, 0.2, 0.31), (0, 0, 0, 1)))
    with pytest.raises(ValueError):
        _bind(calls, factory, preset, samples)


@pytest.mark.parametrize("prefix", ["left", "right"])
def test_native_pad_line_uses_declared_order_for_each_arm(prefix):
    tcp = torch.eye(4, dtype=torch.float64)[None]
    jaw = torch.tensor([[0.01, -0.09, 0.02]], dtype=torch.float64)
    robot = _robot(prefix, tcp, jaw)
    actual = pad_line_world(
        robot, JointPositionTarget(f"{prefix}_arm", tuple(range(7)))
    )
    torch.testing.assert_close(actual, jaw, rtol=0, atol=1e-16)
    assert [call.args[0] for call in robot.get_link_pose.call_args_list] == list(
        _PAD_NAMES[prefix]
    )


@pytest.mark.parametrize("prefix", ["left", "right"])
def test_preview_matches_all_five_actual_kernel_targets_with_one_grasp_composition(
    prefix,
):
    source, receiver, receiver_pose, stage, route = _geometry()
    transform = _transform()
    jaw_tcp = torch.tensor(
        [0.02, 0.09, 0.01] if prefix == "left" else [-0.02, -0.09, -0.01],
        dtype=torch.float64,
    )
    preview = CandidatePreview(
        route, source, receiver, stage, receiver_pose, jaw_tcp, _GRAVITY
    )
    transforms = transform[None, None, None]
    targets, valid = preview.targets(transforms)
    jaw_object = (transform[:3, :3] @ jaw_tcp).numpy()
    frames = pour_frames(
        source,
        receiver,
        stage,
        receiver_pose,
        jaw_object,
        route.angle_rad,
        route.return_pose.numpy(),
        sample_count=route.sample_count,
        gravity_world=_GRAVITY,
    )
    expected_objects = torch.as_tensor(frames["object_targets"])
    expected_eef = expected_objects @ transform
    assert valid.tolist() == [[[True]]]
    assert frames["target_roles"] == _ROLES
    assert torch.equal(targets[:, 0, 0, 0], expected_eef)
    assert not torch.allclose(targets[:, 0, 0, 0], expected_eef @ transform)
    np.testing.assert_array_equal(stage[:3, :3], _native_source_pose()[:3, :3])
    assert frames["physical_evidence"] is False


def test_two_rolls_mask_only_degenerate_axis_and_keep_all_five_finite_targets():
    source, receiver, receiver_pose, _, route = _geometry()
    stage = np.eye(4)
    first = np.eye(4)
    first[:3, :3] = Rotation.from_euler("y", -np.pi / 4).as_matrix()
    second = first.copy()
    second[:3, :3] = first[:3, :3] @ Rotation.from_euler("z", np.pi).as_matrix()
    transforms = torch.as_tensor(np.stack((first, second)))[None, None]
    preview = CandidatePreview(
        route,
        source,
        receiver,
        stage,
        receiver_pose,
        torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64),
        _GRAVITY,
    )
    targets, valid = preview.targets(transforms)
    assert valid.tolist() == [[[False, True]]]
    assert targets.shape == (5, 1, 1, 2, 4, 4)
    assert torch.isfinite(targets).all()
    selection = actions._PickSelection(
        object_pose=torch.eye(4, dtype=torch.float64)[None], pour_preview=preview
    )
    token = actions._PICK_SELECTION.set(selection)
    try:
        action = actions.GenSimPickUp()
        options = PickUpOptions(approach_alignment_max_angle=None)
        aligned = action._approach_alignment_mask(
            transforms, options, torch.tensor([0.0, 0.0, 1.0])
        )
        assert aligned.tolist() == [[[False, True]]]
        assert len(tuple(action._iter_downstream_eef_targets(transforms, options))) == 5
    finally:
        actions._PICK_SELECTION.reset(token)
    assert actions._PICK_SELECTION.get() is None


@pytest.mark.parametrize("with_selection", [False, True])
def test_unowned_pick_delegates_legacy_count_and_exact_targets(with_selection):
    transforms = _transform().float()[None, None, None].repeat(2, 3, 2, 1, 1)
    first, second = torch.eye(4), torch.eye(4).repeat(2, 1, 1)
    first[0, 3], second[:, 1, 3] = 0.2, torch.tensor([-0.2, 0.4])
    options = PickUpOptions(downstream_object_target_poses=(first, second))
    action = actions.GenSimPickUp()
    action._planning_services = SimpleNamespace(device=torch.device("cpu"))
    token = actions._PICK_SELECTION.set(
        actions._PickSelection() if with_selection else None
    )
    try:
        actual = tuple(action._iter_downstream_eef_targets(transforms, options))
        expected = tuple(
            PickUp._iter_downstream_eef_targets(action, transforms, options)
        )
    finally:
        actions._PICK_SELECTION.reset(token)
    assert len(actual) == len(expected) == 2
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))


@pytest.mark.parametrize("retain", [False, True])
def test_failed_pour_keeps_canonical_inputs_without_more_ik_or_changing_legacy(retain):
    @dataclass(frozen=True)
    class Failed:
        diagnostics: PlannerDiagnostics

    original = Failed(PlannerDiagnostics(backend="mock", messages=("IK failed",)))
    generate = Mock(
        return_value=SimpleNamespace(
            positions=None, dt=None, success=torch.tensor([False])
        )
    )
    action = actions.GenSimPour()
    action._planning_services = SimpleNamespace(
        device=torch.device("cpu"), motion_generator=SimpleNamespace(generate=generate)
    )
    action.failed_plan = Mock(return_value=original)
    request = SimpleNamespace(
        motion_policy=SimpleNamespace(
            to_motion_gen_options=lambda **kwargs: SimpleNamespace(
                **kwargs, strategy="ik_interp"
            )
        )
    )
    context = SimpleNamespace(require_control_dt=lambda: 0.04)
    poses = torch.eye(4)[None, None].repeat(1, 3, 1, 1)
    poses[0, :, 0, 3] = torch.tensor([0.1, 0.2, 0.3])
    transform = torch.eye(4)[None]
    transform[0, 1, 3] = 0.05
    qpos = torch.zeros(1, 7)
    before_rng = torch.get_rng_state().clone()
    result = action._build_tilt_plan(
        request,
        context,
        None,
        None,
        qpos,
        JointPositionTarget("left_arm", tuple(range(7))),
        None,
        transform,
        poses,
        {"geometry": "same_request"},
        retain=retain,
    )
    generate.assert_called_once()
    action.failed_plan.assert_called_once()
    assert torch.equal(torch.get_rng_state(), before_rng)
    if not retain:
        assert result is original
    else:
        assert result.diagnostics.messages == original.diagnostics.messages
        metadata = result.diagnostics.metadata
        assert metadata["geometry"] == "same_request"
        inputs = metadata["gen_sim_pour_failed_kinematics"]
        assert inputs["entry_arm_qpos"] == qpos.tolist()
        assert inputs["torch_cpu_rng_state_before_generate"] == before_rng.tolist()
        assert inputs["object_to_eef"] == transform.tolist()
        assert (
            inputs["required_eef_targets"]
            == (poses @ transform[:, None])[:, 1:].tolist()
        )
        assert inputs["per_waypoint_status"] == "not_exposed_by_shared_result"


@pytest.mark.parametrize("retain", [False, True])
def test_existing_seed_frame_owns_pre_pour_gate_without_adding_commands(retain):
    from embodichain.lab.sim.atomic_actions.execution import ExecutionSession
    from embodichain.lab.sim.atomic_actions.invocation import PhaseEffectGateRequirement
    from embodichain.lab.sim.atomic_actions.trajectory_ops import (
        to_full_robot_trajectory,
    )
    from embodichain.lab.sim.motion.planners.utils import PlanResult

    count = 140
    arm_values = torch.zeros(1, count, 7)
    arm_values[0, :, 0] = torch.arange(count) * 0.0001
    dt = torch.full((1, count), 0.04)
    dt[:, 0] = 0
    generate = Mock(
        return_value=PlanResult(
            success=torch.tensor([True]), positions=arm_values, dt=dt
        )
    )
    action = actions.GenSimPour()
    action._planning_services = SimpleNamespace(
        device=torch.device("cpu"),
        motion_generator=SimpleNamespace(generate=generate),
        planner_name="mock",
    )
    action.build_plan = Mock(return_value="plan")
    request = SimpleNamespace(
        motion_policy=SimpleNamespace(
            to_motion_gen_options=lambda **kwargs: SimpleNamespace(
                **kwargs, strategy="ik_interp"
            )
        )
    )
    context = SimpleNamespace(
        robot=SimpleNamespace(qpos=torch.zeros(1, 8)),
        last_qpos=torch.zeros(1, 8),
        env_ids=torch.tensor([0]),
        batch_size=1,
        require_control_dt=lambda: 0.04,
        task=SimpleNamespace(
            exclusive_held_object_mask=lambda resource: torch.tensor([True])
        ),
    )
    hand = SimpleNamespace(
        require_target=lambda cls: JointPositionTarget("left_eef", (7,)),
        joint_positions=lambda *args, **kwargs: torch.zeros(1, 1),
    )
    handoff = SimpleNamespace(resource="left", retain=Mock(return_value="retained"))
    result = action._build_tilt_plan(
        request,
        context,
        handoff,
        None,
        torch.zeros(1, 7),
        JointPositionTarget("left_arm", tuple(range(7))),
        hand,
        torch.eye(4)[None],
        torch.eye(4)[None, None].repeat(1, count, 1, 1),
        {},
        retain=retain,
    )
    assert result == ("retained" if retain else "plan")
    generate.assert_called_once()
    values = action.build_plan.call_args.kwargs
    trajectory = values["trajectory"]
    assert trajectory.waypoint_count == count
    _, expected = to_full_robot_trajectory(
        generate.return_value,
        base_qpos=context.last_qpos,
        joint_ids=list(range(7)),
        env_ids=context.env_ids,
        control_dt=context.require_control_dt(),
    )
    torch.testing.assert_close(trajectory.positions, expected.positions, rtol=0, atol=0)
    torch.testing.assert_close(trajectory.dt, expected.dt, rtol=0, atol=0)
    lengths = values["segment_lengths"]
    assert lengths == (
        {"entry_hold": 1, "pour": count - 1} if retain else {"pour": count}
    )
    start = sum(length for name, length in lengths.items() if name != "pour")
    plan = SimpleNamespace(
        plan_success=torch.tensor([True]),
        segment=lambda name: SimpleNamespace(start=start),
    )
    session = SimpleNamespace(
        _invocation_index=0,
        _requests=[
            SimpleNamespace(
                phase_effect_gates=(
                    PhaseEffectGateRequirement("primary_acquired", "pour"),
                )
            )
        ],
    )
    if retain:
        ExecutionSession._validate_phase_effect_gates(session, plan)
        handoff.retain.assert_called_once_with("plan", None)
    else:
        with pytest.raises(ValueError, match="first trajectory segment"):
            ExecutionSession._validate_phase_effect_gates(session, plan)
        handoff.retain.assert_not_called()


@pytest.mark.parametrize("prefix", ["left", "right"])
def test_prepare_preview_uses_native_current_R_and_current_ordered_pads(prefix):
    source, receiver, receiver_pose, stage, route = _geometry()
    tcp = torch.as_tensor(stage)[None] @ _transform()[None]
    jaw = torch.tensor([[0.02, 0.08, 0.01]], dtype=torch.float64)
    robot = _robot(prefix, tcp, jaw)
    action = SimpleNamespace(
        robot=robot,
        pour_scene_vertices={
            "source": torch.as_tensor(source)[None],
            "receiver": torch.as_tensor(receiver)[None],
        },
        pour_gravity_world=_GRAVITY,
    )
    motion = JointPositionTarget(f"{prefix}_arm", tuple(range(7)))
    request = SimpleNamespace(
        goal=SimpleNamespace(semantics=SimpleNamespace(entity_id="source")),
        skill_options=PickUpOptions(
            downstream_object_target_poses=(torch.eye(4), torch.eye(4))
        ),
        binding=SimpleNamespace(
            endpoint=lambda *args: SimpleNamespace(require_target=lambda cls: motion)
        ),
    )
    context = _context(_native_source_pose(), receiver_pose)
    preview = prepare_preview(action, request, context, route)
    np.testing.assert_array_equal(preview.staging, stage)
    measured_jaw = pad_line_world(robot, motion)
    expected_jaw_tcp = (tcp[:, :3, :3].transpose(1, 2) @ measured_jaw[..., None])[
        0, :, 0
    ]
    assert torch.equal(preview.pad_line_tcp, expected_jaw_tcp)
    robot.compute_fk.assert_called_once()
    assert torch.equal(
        context.scene.entities["source"].pose[0], torch.as_tensor(_native_source_pose())
    )


@pytest.mark.parametrize(
    "problem", ["foreign_subject", "missing_suffix", "extra_suffix"]
)
def test_prepare_preview_rejects_unrecognized_pick_identity_or_suffix(problem):
    *_, route = _geometry()
    request = SimpleNamespace(
        goal=SimpleNamespace(
            semantics=SimpleNamespace(
                entity_id="foreign" if problem == "foreign_subject" else "source"
            )
        ),
        skill_options=SimpleNamespace(
            downstream_object_target_poses=(
                () if problem == "missing_suffix" else (None,) * 3
            )
        ),
    )
    with pytest.raises(ValueError):
        prepare_preview(object(), request, object(), route)


@pytest.mark.parametrize("problem", ["missing", "no_pose", "batch"])
def test_observed_pose_requires_exact_observed_B1_entity(problem):
    context = _context(np.eye(4), np.eye(4))
    if problem == "missing":
        context.scene.entities.pop("source")
    elif problem == "no_pose":
        context.scene.entities["source"].pose = None
    else:
        context.batch_size = 2
    with pytest.raises(ValueError):
        observed_pose(context, "source")


@pytest.mark.parametrize("prefix", ["left", "right"])
@pytest.mark.parametrize("wrong_resource", [False, True])
def test_actual_canonical_branch_records_predicted_entry_peak_restore_not_physical_evidence(
    monkeypatch, prefix, wrong_resource
):
    source, receiver, receiver_pose, stage, route = _geometry()
    resource = ("right" if prefix == "left" else "left") if wrong_resource else prefix
    route = replace(route, resource=resource)
    transform = _transform().float()[None]
    entry = torch.as_tensor(stage).float()[None]
    tcp = entry @ transform
    jaw = torch.tensor([[0.03, 0.08, 0.01]])
    robot = _robot(prefix, tcp, jaw)
    action = actions.GenSimPour({"pour": "receiver"})
    action._planning_services = SimpleNamespace(robot=robot, device=torch.device("cpu"))
    action.continuations = {"pour": route}
    action.scene_vertices = {
        "source": torch.as_tensor(source)[None],
        "receiver": torch.as_tensor(receiver)[None],
    }
    action.gravity_world = _GRAVITY
    held = SimpleNamespace(
        object_to_eef=transform.float(), semantics=SimpleNamespace(entity_id="source")
    )
    handoff = SimpleNamespace(resource=prefix, require_held=lambda: held)
    monkeypatch.setattr(
        actions.GenSimStateHandoff, "from_request", Mock(return_value=handoff)
    )
    motion = JointPositionTarget(f"{prefix}_arm", tuple(range(7)))
    request = SimpleNamespace(
        invocation_id="pour",
        binding=SimpleNamespace(
            endpoint=lambda *args: SimpleNamespace(require_target=lambda cls: motion)
        ),
        motion_policy=SimpleNamespace(sample_count=route.sample_count),
        skill_options=PourOptions(rotate_angle=route.angle_rad),
    )
    context = _context(_native_source_pose(), receiver_pose)
    context.robot.qpos = context.robot.qpos.float()
    # Existing handoff G stays authoritative for command continuity. Native
    # source observations are separate observer inputs, never planning evidence.
    captured = Mock(return_value="captured_without_motion_or_IK")
    monkeypatch.setattr(action, "_build_tilt_plan", captured)
    if wrong_resource:
        with pytest.raises(ValueError, match="resource differs"):
            action._plan(request, context)
        captured.assert_not_called()
        return
    assert action._plan(request, context) == "captured_without_motion_or_IK"
    args, kwargs = captured.call_args
    arc, metadata = args[-2:]
    predicted = tcp @ actions.pose_inv(transform)
    measured_jaw = pad_line_world(robot, motion)
    local_jaw = (predicted[:, :3, :3].transpose(1, 2) @ measured_jaw[..., None])[
        0, :, 0
    ]
    frames = pour_frames(
        source,
        receiver,
        predicted[0].numpy(),
        receiver_pose,
        local_jaw.numpy(),
        route.angle_rad,
        route.return_pose.numpy(),
        sample_count=route.sample_count,
        gravity_world=_GRAVITY,
    )
    assert kwargs == {"retain": True}
    np.testing.assert_array_equal(
        arc[0].numpy(), frames["object_arc"].astype(np.float32)
    )
    evidence = metadata["gen_sim_pour_geometry"]
    assert evidence["physical_evidence"] is False
    assert evidence["fluid_transfer"] == "unmodeled"
    assert (
        evidence["source_entity_id"] == "source"
        and evidence["receiver_entity_id"] == "receiver"
    )
    assert tuple(evidence["target_roles"]) == _ROLES
    for key, index in (
        ("entry_pose", 0),
        ("peak_pose", frames["peak_index"]),
        ("restore_pose", -1),
    ):
        np.testing.assert_array_equal(evidence[key], frames["object_arc"][index])
    assert not np.allclose(
        evidence["entry_pose"], context.scene.entities["source"].pose[0].numpy()
    )


@pytest.mark.parametrize("problem", ["subject", "receiver", "budget", "angle", "batch"])
def test_actual_canonical_arc_rejects_bound_recipe_mismatch(problem):
    source, receiver, receiver_pose, stage, route = _geometry()
    action = actions.GenSimPour(
        {"pour": "foreign" if problem == "receiver" else "receiver"}
    )
    held = SimpleNamespace(
        semantics=SimpleNamespace(
            entity_id="foreign" if problem == "subject" else "source"
        )
    )
    context = _context(stage, receiver_pose)
    context.batch_size = 2 if problem == "batch" else 1
    request = SimpleNamespace(
        invocation_id="pour",
        motion_policy=SimpleNamespace(
            sample_count=route.sample_count + (problem == "budget")
        ),
        skill_options=PourOptions(
            rotate_angle=route.angle_rad + (0.1 if problem == "angle" else 0)
        ),
    )
    with pytest.raises(ValueError, match="bound recipe"):
        action._canonical_arc(
            request,
            context,
            route,
            held,
            torch.as_tensor(stage)[None],
            JointPositionTarget("left_arm", tuple(range(7))),
        )


def _capability_inputs(cleanup_count=1):
    resources = {"primary": "left"}
    calls = [
        {"kind": "pick", "object": "source"},
        {
            "kind": "registered",
            "call_id": "simulation.move_held_object",
            "arguments": {
                "object": "source",
                "target": "stage",
                "reference": "receiver",
            },
        },
        {
            "kind": "registered",
            "call_id": "simulation.pour",
            "arguments": {"object": "source"},
        },
        {
            "kind": "registered",
            "call_id": "simulation.move_held_object",
            "arguments": {"object": "source", "target": "return"},
        },
        {
            "kind": "place",
            "object": "source",
            "at": {"kind": "target_ref", "target": "return"},
        },
        *(
            {"kind": "registered", "call_id": "simulation.park", "arguments": {}}
            for _ in range(cleanup_count)
        ),
    ]
    nodes = [
        {
            "id": f"pour_{index}",
            "task_type": "E3",
            "task_instance_id": "pour",
            "call": {**call, "resources": dict(resources)},
        }
        for index, call in enumerate(calls)
    ]
    graph = {
        "nodes": nodes,
        "task_groups": [
            {
                "id": "pour",
                "task_type": "E3",
                "node_ids": [node["id"] for node in nodes],
            }
        ],
    }
    candidate = {
        "draft": {
            "steps": [
                {
                    "id": "pour",
                    "task_type": "E3",
                    "orientation_goal": "none",
                    "axis": "none",
                    "direction": "none",
                }
            ]
        }
    }
    return graph, candidate


@pytest.mark.parametrize("cleanup_count", [0, 1, 2])
def test_ordinary_pour_capability_owns_five_calls_and_trailing_cleanup_without_mutation(
    cleanup_count,
):
    from copy import deepcopy

    graph, candidate = _capability_inputs(cleanup_count)
    before = deepcopy((graph, candidate))
    assert supports_ordinary_pour(graph, candidate)
    assert (graph, candidate) == before


@pytest.mark.parametrize(
    "field,value",
    [("orientation_goal", "upright"), ("axis", "x"), ("direction", "left")],
)
def test_ordinary_pour_capability_leaves_explicit_constraints_unowned(field, value):
    graph, candidate = _capability_inputs()
    candidate["draft"]["steps"][0][field] = value
    assert not supports_ordinary_pour(graph, candidate)


@pytest.mark.parametrize("missing", ["candidate", "graph", "both"])
def test_ordinary_pour_capability_requires_an_E3_candidate_and_graph_group(missing):
    graph, candidate = _capability_inputs()
    if missing in ("candidate", "both"):
        candidate["draft"]["steps"][0]["task_type"] = "E1"
    if missing in ("graph", "both"):
        graph["task_groups"][0]["task_type"] = "E1"
    assert not supports_ordinary_pour(graph, candidate)


def test_held_prefix_E3_without_its_own_Pick_remains_legacy():
    graph, candidate = _capability_inputs(2)
    graph["nodes"][0]["task_type"] = "E4"
    graph["nodes"][0]["call"] = {
        "kind": "hand_over",
        "object": "source",
        "resources": {"source": "right", "destination": "left"},
    }
    graph["task_groups"][0]["node_ids"].pop(0)
    graph["task_groups"].insert(
        0, {"id": "held_prefix", "task_type": "E4", "node_ids": ["pour_0"]}
    )
    assert not supports_ordinary_pour(graph, candidate)


@pytest.mark.parametrize("role", range(5))
def test_ordinary_pour_capability_requires_every_primary_call_in_order(role):
    graph, candidate = _capability_inputs()
    graph["nodes"][role]["call"] = {
        "kind": "registered",
        "call_id": "simulation.park",
        "arguments": {},
        "resources": {"primary": "left"},
    }
    assert not supports_ordinary_pour(graph, candidate)


@pytest.mark.parametrize("role", range(1, 5))
def test_ordinary_pour_capability_requires_one_subject(role):
    graph, candidate = _capability_inputs()
    call = graph["nodes"][role]["call"]
    (call if role == 4 else call["arguments"])["object"] = "foreign"
    assert not supports_ordinary_pour(graph, candidate)


@pytest.mark.parametrize("role", range(5))
def test_ordinary_pour_capability_requires_same_primary_resource(role):
    graph, candidate = _capability_inputs()
    graph["nodes"][role]["call"]["resources"] = {"primary": "right"}
    assert not supports_ordinary_pour(graph, candidate)


def test_ordinary_pour_capability_rejects_non_primary_binding():
    graph, candidate = _capability_inputs()
    for node in graph["nodes"]:
        node["call"]["resources"] = {"other": "left"}
    assert not supports_ordinary_pour(graph, candidate)


@pytest.mark.parametrize(
    "problem",
    ["no_stage_reference", "relative_return", "no_place_pose", "interleaved_call"],
)
def test_ordinary_pour_capability_requires_independent_contiguous_stage_and_return(
    problem,
):
    graph, candidate = _capability_inputs()
    if problem == "no_stage_reference":
        graph["nodes"][1]["call"]["arguments"].pop("reference")
    elif problem == "relative_return":
        graph["nodes"][3]["call"]["arguments"]["reference"] = "receiver"
    elif problem == "no_place_pose":
        graph["nodes"][4]["call"].pop("at")
    else:
        graph["nodes"].insert(
            2,
            {
                "id": "other",
                "task_type": "E1",
                "call": {"kind": "pick", "object": "foreign"},
            },
        )
    assert not supports_ordinary_pour(graph, candidate)


@pytest.mark.parametrize("second_supported", [False, True])
def test_ordinary_pour_capability_requires_all_E3_groups_to_be_owned(second_supported):
    from copy import deepcopy

    graph, candidate = _capability_inputs()
    second = deepcopy(graph)
    for node in second["nodes"]:
        node["id"] += "_second"
    group = second["task_groups"][0]
    group["id"] = "second"
    group["node_ids"] = [node["id"] for node in second["nodes"]]
    if not second_supported:
        group["node_ids"].pop(0)
    graph["nodes"].extend(second["nodes"])
    graph["task_groups"].append(group)
    candidate["draft"]["steps"].append(
        {**candidate["draft"]["steps"][0], "id": "second"}
    )
    assert supports_ordinary_pour(graph, candidate) is second_supported
