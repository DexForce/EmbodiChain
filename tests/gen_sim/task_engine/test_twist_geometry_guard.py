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

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import numpy as np
import torch

from embodichain.gen_sim.task_engine._task_program import motion, twist_runtime
from embodichain.gen_sim.task_engine._task_program.twist_runtime import GenSimTwist
from embodichain.lab.sim.atomic_actions import TimedTrajectory, Twist, TwistOptions


@dataclass(frozen=True)
class _Diagnostics:
    metadata: dict


@dataclass(frozen=True)
class _Request:
    goal: object
    skill_options: object


@dataclass(frozen=True)
class _Plan:
    plan_success: torch.Tensor
    diagnostics: _Diagnostics
    joint_trajectory: TimedTrajectory
    segments: tuple
    scene_dependencies: tuple = ("control::knob",)
    scene_dependency_monitor_until: object = None
    scene_dependency_end_segment: str | None = None
    commands: object = None
    recovery_policy: object = None
    tracking_policy: object = None


@pytest.fixture
def candidate_case(monkeypatch: pytest.MonkeyPatch) -> NS:
    geometry = {
        "gen_sim_twist_angle": 0.1,
        "gen_sim_twist_measured_qpos": 0.0,
        "gen_sim_twist_target_qpos": 1.0,
        "gen_sim_twist_axis_sign": 1.0,
        "gen_sim_twist_chunk_angle": 0.1,
        "gen_sim_twist_bite_depth": 0.006,
        "gen_sim_twist_settle_steps": 0,
    }
    request = _Request(NS(semantics=NS(geometry=geometry)), TwistOptions())
    context = NS(robot=NS(qpos=torch.zeros(1, 2)), batch_size=1)
    trajectory = TimedTrajectory.from_uniform_step(
        torch.zeros(1, 3, 2), env_ids=torch.tensor([0]), step_dt=0.04
    )
    plan = _Plan(
        torch.tensor([True]),
        _Diagnostics({}),
        trajectory,
        (NS(name="approach", start=0, stop=1), NS(name="reach", start=1, stop=3)),
        scene_dependency_monitor_until={"control::knob": 2},
    )
    generated, scored = [], []

    def public_plan(action: object, candidate: object, context: object) -> _Plan:
        generated.append(candidate)
        return plan

    def score(action: object, candidate: _Plan, request: object, context: object):
        index = candidate.diagnostics.metadata["gen_sim_twist_roll_candidate"]
        scored.append(index)
        return 0.0, 0.0, 0.0, float(index)

    monkeypatch.setattr(Twist, "_plan", public_plan)
    monkeypatch.setattr(
        GenSimTwist, "_candidate_request", staticmethod(lambda r, roll, shift: r)
    )
    monkeypatch.setattr(GenSimTwist, "_candidate_geometry_score", score)
    monkeypatch.setattr(motion, "_joint_velocity_limits", lambda *args: torch.ones(2))
    validity = Mock(return_value=torch.tensor([True]))
    monkeypatch.setattr(motion, "_velocity_validity", validity)
    token = twist_runtime._CANDIDATE_STATE.set(twist_runtime._TwistCandidateState())
    yield NS(
        request=request,
        context=context,
        plan=plan,
        generated=generated,
        scored=scored,
        validity=validity,
    )
    twist_runtime._CANDIDATE_STATE.reset(token)


def _bound_action(guard: object | None = None) -> GenSimTwist:
    action = GenSimTwist(geometry_guard=guard)
    action._planning_services = NS(robot=NS(device=torch.device("cpu")))
    return action


def test_unsafe_candidates_never_enter_ranking(candidate_case: NS) -> None:
    case = candidate_case
    filtered, aliased = [], []

    def filter_candidate(action, plan, request, context):
        index = plan.diagnostics.metadata["gen_sim_twist_roll_candidate"]
        filtered.append(index)
        return plan if index == 3 else replace(plan, plan_success=torch.tensor([False]))

    def root_dependency(plan, request, context):
        assert case.validity.call_count == 1
        assert plan.scene_dependency_end_segment == "reach"
        aliased.append(plan)
        return replace(plan, scene_dependencies=("control",)), {"qualified": True}

    guard = NS(filter_candidate=filter_candidate, root_dependency=root_dependency)
    result = _bound_action(guard)._plan(case.request, case.context)
    assert filtered == list(range(12))
    assert case.scored == [3]
    assert result.diagnostics.metadata["gen_sim_twist_roll_candidate"] == 3
    assert result.scene_dependencies == ("control",)
    assert result.joint_trajectory is case.plan.joint_trajectory
    assert len(aliased) == 1


def test_all_rejected_candidates_are_a_failed_plan_not_a_bad_best_score(
    candidate_case: NS,
) -> None:
    guard = NS(
        filter_candidate=lambda action, plan, request, context: replace(
            plan, plan_success=torch.tensor([False])
        ),
        root_dependency=Mock(side_effect=AssertionError("Failed plans cannot alias.")),
    )
    case = candidate_case
    result = _bound_action(guard)._plan(case.request, case.context)
    assert not result.plan_success.any()
    assert len(case.generated) == 12 and case.scored == []
    case.validity.assert_not_called()
    guard.root_dependency.assert_not_called()


def test_velocity_rejection_cannot_be_overridden_by_dependency_alias(
    candidate_case: NS,
) -> None:
    case = candidate_case
    case.validity.return_value = torch.tensor([False])
    guard = NS(
        filter_candidate=lambda action, plan, request, context: plan,
        root_dependency=Mock(
            side_effect=AssertionError("Velocity guard precedes alias.")
        ),
    )
    action = _bound_action(guard)
    failure = replace(case.plan, plan_success=torch.tensor([False]))
    action.failed_plan = Mock(return_value=failure)
    result = action._plan(case.request, case.context)
    assert result is failure
    guard.root_dependency.assert_not_called()


def test_default_twist_wrapper_keeps_candidate_behavior_without_guard(
    candidate_case: NS,
) -> None:
    case = candidate_case
    result = _bound_action()._plan(case.request, case.context)
    assert case.scored == list(range(12))
    assert result.diagnostics.metadata["gen_sim_twist_roll_candidate"] == 0
    assert result.scene_dependencies == ("control::knob",)
    assert result.joint_trajectory is case.plan.joint_trajectory
    assert GenSimTwist.descriptor() == Twist.descriptor()


def test_factory_borrows_existing_e8_sensor_without_changing_other_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from embodichain.gen_sim.task_engine._task_program import (
        assembly,
        twist_geometry_guard,
    )
    from embodichain.lab.task_program.semantics import RobotSkillProfile

    engine = Mock()
    monkeypatch.setattr(assembly, "GenSimActionEngine", Mock(return_value=engine))
    guard, sensor, simulation, route = object(), object(), object(), object()
    constructor = Mock(return_value=guard)
    monkeypatch.setattr(twist_geometry_guard, "E8GeometryGuard", constructor)
    factory = object.__new__(assembly._TaskFactory)
    factory._drawers = factory._press_routes = ()
    factory._twist_routes = (route,)
    factory._pour_receivers = {}
    factory._adaptive_pick = factory._coordinated_motion = False
    factory._pick_purposes = factory._motion_samples = factory._cartesian_calls = ()
    factory._articulation_calls = ()
    factory._robot = object()
    factory._simulation = simulation
    factory._task_post_port = NS(sensor=sensor)
    factory._robot_profile_binding = NS(profile_id="profile")
    factory._grasp_pose_generators = {}
    factory._registration = NS(validate_engine=Mock())
    factory._create_motion_generator = Mock(return_value=NS(robot=factory._robot))
    profile = Mock(spec=RobotSkillProfile, profile_id="profile")
    profile.action_control_profiles.return_value = {}
    assert factory.create_atomic_action_engine(profile) is engine
    constructor.assert_called_once_with(route, simulation, factory._robot, sensor)
    assert engine.register.call_count == 6
    action = engine.register.call_args_list[-1].args[0]
    assert isinstance(action, GenSimTwist) and action._geometry_guard is guard


def test_continuation_suffix_is_screened_in_full_without_inventing_approach(
    candidate_case: NS,
) -> None:
    from embodichain.gen_sim.task_engine._task_program.twist_geometry_guard import (
        E8GeometryGuard,
    )

    guard = E8GeometryGuard.__new__(E8GeometryGuard)
    guard._filter_table = Mock(return_value=candidate_case.plan)
    action = NS(failed_plan=Mock(return_value="rejected"))
    names = ("reach", "close", "twist", "open", "retract")
    plan = replace(candidate_case.plan, segments=tuple(NS(name=name) for name in names))
    request = NS(
        goal=NS(semantics=NS(geometry={"gen_sim_twist_resume_phase": "close"}))
    )
    assert (
        guard.filter_continuation(action, plan, request, candidate_case.context)
        is candidate_case.plan
    )
    guard._filter_table.assert_called_once_with(
        action, plan, request, candidate_case.context, names
    )
    action.failed_plan.assert_not_called()
    incomplete = replace(plan, segments=plan.segments[:-1])
    assert (
        guard.filter_continuation(action, incomplete, request, candidate_case.context)
        == "rejected"
    )


def test_feedback_factory_returns_exact_registered_state_and_rejects_foreign_robot():
    from embodichain.gen_sim.task_engine._task_program.twist_feedback import (
        TwistFeedbackState,
    )
    from embodichain.gen_sim.task_engine._task_program.twist_feedback_factory import (
        TwistFeedbackFactory,
    )

    route, robot, simulation = object(), object(), object()
    state = TwistFeedbackState.__new__(TwistFeedbackState)
    state.route, state.robot, state.simulation = route, robot, simulation
    action = GenSimTwist(feedback_state=state)
    engine = NS(robot=robot, actions={"twist": action})
    factory = TwistFeedbackFactory(route)
    assert (
        factory.create(
            simulation=simulation, robot=robot, scene_registry=object(), engine=engine
        )
        is state
    )
    assert (state.controller_id, state.revision, state.supported_skill_ids) == (
        factory.controller_id,
        factory.revision,
        factory.supported_skill_ids,
    )
    with pytest.raises(ValueError, match="exact registered"):
        factory.create(
            simulation=simulation,
            robot=object(),
            scene_registry=object(),
            engine=engine,
        )


@pytest.mark.parametrize(
    "points, expected",
    (
        ([[3.0, 4.0]], 5.0),
        ([[-2.0, 1.0], [2.0, 1.0]], 1.0),
        ([[-2.0, 1.0], [0.0, 1.0], [2.0, 1.0]], 1.0),
        ([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], 0.0),
        ([[1.0, 1.0], [3.0, 1.0], [3.0, 2.0], [1.0, 2.0]], 2**0.5),
    ),
)
def test_projection_bounds_the_complete_polygon_not_only_vertices(
    points: list, expected: float
) -> None:
    from embodichain.gen_sim.task_engine._task_program.twist_geometry_guard import (
        _radial_distance,
    )

    distance = _radial_distance(np.asarray(points))
    assert 0 <= distance <= expected
    assert distance == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("points", ([], [[float("nan"), 0.0]], [[1, 2, 3]]))
def test_invalid_axis_projection_cannot_certify_separation(points: list) -> None:
    from embodichain.gen_sim.task_engine._task_program.twist_geometry_guard import (
        _radial_distance,
    )

    with pytest.raises(ValueError):
        _radial_distance(np.asarray(points))


def _native_table_case() -> NS:
    root = torch.eye(4).unsqueeze(0)
    root[0, :3, 3] = torch.tensor([1.0, -2.0, 3.0])
    shapes = []
    for x in (-2.0, 2.0):
        pose = torch.eye(4)
        pose[0, 3] = x
        shapes.append(
            NS(
                local_pose=pose,
                vertices=None,
                triangles=None,
                half_extents=torch.tensor([0.2, 0.1, 0.3]),
            )
        )
    entity = NS(
        native=lambda: NS(get_physical_body=lambda: NS(get_shape_count=lambda: 2)),
        get_physical_attr=lambda: NS(contact_offset=0.002, rest_offset=0.0),
    )
    return NS(
        num_instances=1,
        body_type="static",
        cfg=NS(body_scale=[3.0] * 3),
        body_data=NS(entities=[entity]),
        get_local_pose=lambda **kw: root,
        get_collision_shapes=lambda **kw: shapes,
        shapes=shapes,
    )


def test_native_table_uses_all_shapes_and_does_not_scale_vertices_twice() -> None:
    from embodichain.gen_sim.task_engine._task_program.twist_geometry_guard import (
        _native_table,
    )

    table = _native_table_case()
    bounds, contact = _native_table(table)
    np.testing.assert_allclose(bounds, [[-1.2, -2.1, 2.7], [3.2, -1.9, 3.3]])
    assert contact == 0.002


@pytest.mark.parametrize(
    "invalid", ("dynamic", "missing_shape", "scale_pose", "offset")
)
def test_unqualified_native_table_fails_closed(invalid: str) -> None:
    from embodichain.gen_sim.task_engine._task_program.twist_geometry_guard import (
        _native_table,
    )

    table = _native_table_case()
    if invalid == "dynamic":
        table.body_type = "dynamic"
    elif invalid == "missing_shape":
        table.shapes.pop()
    elif invalid == "scale_pose":
        table.shapes[0].local_pose[0, 0] = 2.0
    else:
        table.body_data.entities[0].get_physical_attr = lambda: NS(
            contact_offset=0.001, rest_offset=0.002
        )
    with pytest.raises(ValueError):
        _native_table(table)


def _shell_case() -> tuple[NS, NS]:
    import trimesh

    angle = np.arange(32) * (2 * np.pi / 32)
    ring = np.column_stack((np.cos(angle) * 0.03, np.sin(angle) * 0.03, np.zeros(32)))
    mesh = trimesh.convex.convex_hull(np.r_[ring + [0, 0, 0.01], ring + [0, 0, 0.03]])
    grip = NS(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), path="/grip")
    return NS(source_sha256="same", grip=grip, target_collisions=(grip,)), NS(
        source_sha256="same", origin=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0)
    )


def test_circular_shell_qualification_is_geometric_not_task_specific() -> None:
    from embodichain.gen_sim.task_engine._task_program.twist_geometry_guard import (
        _cylinder_shell,
    )

    geometry, binding = _shell_case()
    assert _cylinder_shell(geometry, binding)["radius_m"] == pytest.approx(0.03)


@pytest.mark.parametrize(
    "invalid", ("oval", "off_axis", "missing_faces", "extra", "hash")
)
def test_non_axisymmetric_or_unbound_shell_cannot_alias(invalid: str) -> None:
    from embodichain.gen_sim.task_engine._task_program.twist_geometry_guard import (
        _cylinder_shell,
    )

    geometry, binding = _shell_case()
    if invalid == "oval":
        geometry.grip.vertices[:, 0] *= 1.2
    elif invalid == "off_axis":
        geometry.grip.vertices[:, 0] += 0.001
    elif invalid == "missing_faces":
        geometry.grip.faces = geometry.grip.faces[:-2]
    elif invalid == "extra":
        geometry.target_collisions += (
            NS(path="/extra", vertices=np.array([[0.04, 0, 0]])),
        )
    else:
        binding.source_sha256 = "foreign"
    with pytest.raises(ValueError):
        _cylinder_shell(geometry, binding)


def _resolved_guard_case(tmp_path: Path, case: NS) -> NS:
    from embodichain.gen_sim.task_engine._task_program.twist_geometry_guard import (
        E8GeometryGuard,
    )
    from embodichain.lab.sim.atomic_actions import (
        ActionBinding,
        ActionInvocation,
        AtomicActionEngine,
        ObjectSemantics,
        SceneEntityPose,
        TwistAffordance,
        TwistGoal,
    )

    asset, urdf = tmp_path / "source.usda", tmp_path / "robot.urdf"
    asset.write_text("immutable source fixture")
    urdf.write_text("immutable robot fixture")
    binding = NS(
        object_id="control",
        link="knob",
        joint="turn",
        parent="base",
        target_qpos=1.0,
        source_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
        axis_sign=1.0,
        scale=0.5,
        limits=(-np.pi, np.pi),
    )
    route = NS(binding=binding, link_id="control::knob", arm="right")
    joint = NS(
        origin_pose=np.eye(4), target_pose=np.eye(4), axis=np.array([0.0, 0.0, 1.0])
    )
    entity = NS(
        get_root_link_name=lambda: "base",
        render_articulation=NS(get_joint_info=lambda name: joint),
    )
    art = NS(
        cfg=NS(fpath=str(asset), body_scale=[0.5] * 3),
        joint_names=["turn"],
        body_data=NS(entities=[entity]),
        get_qpos=lambda: torch.zeros(1, 1),
        get_link_pose=lambda *args, **kwargs: torch.eye(4)[None],
        get_link_physical_attr=lambda *args, **kwargs: [
            NS(contact_offset=0.002, rest_offset=0.0)
        ],
    )
    robot = NS(cfg=NS(fpath=str(urdf), body_scale=[1.0] * 3), num_instances=1)
    guard = E8GeometryGuard.__new__(E8GeometryGuard)
    guard._setup_error = None
    guard.route, guard.binding, guard.art, guard.robot = route, binding, art, robot
    guard.sensor = NS(art=art)
    guard.simulation = NS(is_default_backend=True, get_articulation=lambda name: art)
    guard._asset_path, guard._robot_path, guard._robot_scale = asset, urdf, (1.0,) * 3
    guard._sources = {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (asset, urdf)
    }
    guard._shell, guard.geometry, guard._names = (
        {},
        NS(target_collisions=()),
        ("right_pad",),
    )
    guard._offsets = lambda names: {name: 0.002 for name in names}
    guard._clouds = lambda *args: iter(())
    values = {
        **case.request.goal.semantics.geometry,
        "gen_sim_twist_articulation_id": "control",
        "gen_sim_twist_joint": "turn",
        "gen_sim_twist_arm": "right",
    }
    goal = TwistGoal(
        semantics=ObjectSemantics(
            entity_id="control::knob",
            geometry=values,
            affordance=TwistAffordance(
                grasp_position=(0.0, 0.0, 0.05),
                axis_origin=(0.0, 0.0, 0.0),
                twist_axis=torch.tensor([0.0, 0.0, 1.0]),
                joint_name="turn",
                joint_limits=(-np.pi, np.pi),
            ),
        ),
        target_pose=SceneEntityPose("control::knob"),
    )
    action = GenSimTwist()
    action._planning_services = NS(
        validate_binding=lambda *args: None,
        apply_command_overrides=lambda binding, overrides: binding,
    )
    engine = AtomicActionEngine.__new__(AtomicActionEngine)
    engine._actions = {"twist": action}
    request = engine.resolve(
        ActionInvocation(
            skill_id="twist",
            goal=goal,
            binding=ActionBinding(owner_id="test"),
            invocation_id="test",
        )
    ).snapshot()
    assert request.goal is not goal
    context = NS(
        batch_size=1, scene=NS(entities={"control": NS(pose=torch.eye(4)[None])})
    )
    return NS(guard=guard, request=request, context=context, joint=joint, asset=asset)


def test_real_engine_snapshot_qualifies_without_lowerer_identity_or_token(
    tmp_path: Path, candidate_case: NS
) -> None:
    case = _resolved_guard_case(tmp_path, candidate_case)
    before = case.joint.axis.copy()
    plan, evidence = case.guard.root_dependency(
        candidate_case.plan, case.request, case.context
    )
    assert evidence["qualified"] and evidence["changed"]
    assert plan.scene_dependencies == ("control",)
    assert plan.joint_trajectory is candidate_case.plan.joint_trajectory
    assert plan.scene_dependency_monitor_until == {"control": 2}
    np.testing.assert_array_equal(case.joint.axis, before)


def test_readonly_native_axis_is_normalized_without_mutating_getter_buffer(
    tmp_path: Path, candidate_case: NS
) -> None:
    case = _resolved_guard_case(tmp_path, candidate_case)
    case.joint.axis = np.array([0.0, 0.0, 2.0])
    case.joint.axis.flags.writeable = False
    _, evidence = case.guard.root_dependency(
        candidate_case.plan, case.request, case.context
    )
    assert evidence["qualified"]
    np.testing.assert_array_equal(case.joint.axis, [0.0, 0.0, 2.0])


@pytest.mark.parametrize(
    "invalid", ("source", "scale", "qpos", "root", "hinge", "foreign")
)
def test_invalid_qualification_retains_original_dependency_and_trajectory(
    tmp_path: Path, candidate_case: NS, invalid: str
) -> None:
    case = _resolved_guard_case(tmp_path, candidate_case)
    if invalid == "source":
        case.asset.write_text("source changed")
    elif invalid == "scale":
        case.guard.art.cfg.body_scale = [0.6] * 3
    elif invalid == "qpos":
        case.guard.art.get_qpos = lambda: torch.tensor([[0.1]])
    elif invalid == "root":
        case.context.scene.entities["control"].pose[0, 0, 3] += 0.03
    elif invalid == "hinge":
        case.joint.target_pose[0, 3] += 0.01
    else:
        case.request = replace(
            case.request,
            goal=replace(
                case.request.goal,
                semantics=replace(case.request.goal.semantics, entity_id="other::knob"),
            ),
        )
    plan, evidence = case.guard.root_dependency(
        candidate_case.plan, case.request, case.context
    )
    assert plan is candidate_case.plan
    assert not evidence["qualified"] and not evidence["changed"]


@pytest.mark.parametrize(
    "child, parent, parent_present, expected",
    (
        (None, None, False, None),
        (0, None, False, 0),
        (10, None, False, 10),
        (None, 3, True, None),
        (10, None, True, None),
        (None, None, True, None),
        (0, 3, True, 3),
        (10, 3, True, 10),
        (3, 10, True, 10),
    ),
)
def test_dependency_alias_never_shortens_existing_monitor_window(
    candidate_case: NS,
    child: int | None,
    parent: int | None,
    parent_present: bool,
    expected: int | None,
) -> None:
    from embodichain.gen_sim.task_engine._task_program.twist_geometry_guard import (
        _remap_dependency,
    )

    dependencies = ("control::knob", "obstacle") + (
        ("control",) if parent_present else ()
    )
    cutoff = {"obstacle": 7}
    if child is not None:
        cutoff["control::knob"] = child
    if parent_present and parent is not None:
        cutoff["control"] = parent
    plan = replace(
        candidate_case.plan,
        scene_dependencies=dependencies,
        scene_dependency_monitor_until=cutoff,
        scene_dependency_end_segment="reach",
        commands=object(),
        recovery_policy=object(),
        tracking_policy=object(),
    )
    updated = _remap_dependency(plan, "control::knob", "control")
    assert updated.scene_dependencies == ("control", "obstacle")
    assert updated.scene_dependency_monitor_until == (
        {"obstacle": 7} if expected is None else {"obstacle": 7, "control": expected}
    )
    assert updated.scene_dependency_end_segment == "reach"
    assert updated.joint_trajectory is plan.joint_trajectory
    assert updated.commands is plan.commands
    assert updated.recovery_policy is plan.recovery_policy
    assert updated.tracking_policy is plan.tracking_policy
    assert plan.scene_dependency_monitor_until is cutoff


def test_no_child_dependency_is_an_identity_operation(candidate_case: NS) -> None:
    from embodichain.gen_sim.task_engine._task_program.twist_geometry_guard import (
        _remap_dependency,
    )

    plan = replace(
        candidate_case.plan,
        scene_dependencies=("obstacle",),
        scene_dependency_monitor_until={"obstacle": 7},
    )
    assert _remap_dependency(plan, "control::knob", "control") is plan


@pytest.mark.parametrize(
    "clouds", ("valid", "empty", "nonfinite", "missing", "duplicate")
)
def test_table_acceptance_requires_complete_finite_command_frame_clouds(
    candidate_case: NS, monkeypatch: pytest.MonkeyPatch, clouds: str
) -> None:
    from embodichain.gen_sim.task_engine._task_program import (
        twist_geometry_guard as module,
    )

    plan = replace(
        candidate_case.plan,
        segments=tuple(
            NS(name=name, start=index, stop=index + 1)
            for index, name in enumerate(("approach", "reach", "retract"))
        ),
    )
    table = NS(cfg=NS(shape=NS(fpath=None)))
    mesh = NS(path="pad/collision/0")
    guard = module.E8GeometryGuard.__new__(module.E8GeometryGuard)
    guard._fresh = lambda *args: None
    guard.table, guard._table_path = table, None
    guard.simulation = NS(get_rigid_object=lambda name: table)
    guard._names, guard._meshes = ("right_pad",), {"right_pad": (mesh,)}
    guard.sensor = NS(finger_names=("right_pad",))
    guard._offsets = lambda names: {"right_pad": 0.002}
    native = NS(get_physical_body=lambda name: NS(get_shape_count=lambda: 1))
    guard.robot = NS(
        joint_names=["arm"],
        get_parent_joint_chain=lambda name: [NS(name="arm")],
        body_data=NS(entities=[NS(render_articulation=native)]),
    )
    request = NS(
        binding=NS(
            endpoint=lambda *args: NS(require_target=lambda cls: NS(joint_ids=(0,)))
        )
    )
    monkeypatch.setattr(
        module,
        "_native_table",
        lambda table: (np.array([[0, -1, -1], [1, 1, 1]]), 0.002),
    )
    frames = (
        ()
        if clouds == "empty"
        else (
            (0, 1)
            if clouds == "missing"
            else (0, 0, 2) if clouds == "duplicate" else (0, 1, 2)
        )
    )
    point = float("nan") if clouds == "nonfinite" else 2.0
    guard._clouds = lambda *args: iter(
        (index, "right_pad", mesh, torch.tensor([[point, 0, 0]])) for index in frames
    )
    action = NS(
        failed_plan=Mock(return_value=replace(plan, plan_success=torch.tensor([False])))
    )
    result = guard.filter_candidate(action, plan, request, candidate_case.context)
    assert bool(result.plan_success.any()) is (clouds == "valid")
    evidence = result.diagnostics.metadata["gen_sim_twist_table_clearance"]
    assert evidence["accepted"] is (clouds == "valid")
    assert evidence["supported"] is (clouds == "valid")
