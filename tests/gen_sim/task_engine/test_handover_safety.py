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

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import trimesh
from dexsim.engine import BoxGeometry, PlaneGeometry, SphereGeometry

from embodichain.gen_sim.task_engine._task_program import handover_safety as safety
from embodichain.gen_sim.task_engine._task_program import motion
from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
from embodichain.lab.sim.atomic_actions.runtime_commands import (
    EndpointCommand,
    JointPositionPayload,
)
from embodichain.lab.sim.motion.motion_generator import MotionGenOptions


def pose(x=0.0, y=0.0, z=0.0):
    result = np.eye(4)
    result[:3, 3] = [x, y, z]
    return result


def body(enabled=True, extent=0.02):
    geometry = BoxGeometry()
    geometry.half_extents = np.full(3, extent / 2)
    geometry.local_pose = np.eye(4)
    shapes = [geometry]
    return SimpleNamespace(
        is_collision_enable=Mock(return_value=enabled),
        meshes=(trimesh.creation.box(extents=[extent] * 3),),
        shapes=shapes,
        get_shape_count=Mock(side_effect=lambda: len(shapes)),
        get_shape_geometry=Mock(side_effect=lambda index: shapes[index]),
        get_contact_offset=Mock(return_value=0.003),
        get_rest_offset=Mock(return_value=0.001),
    )


@pytest.fixture
def case(monkeypatch, tmp_path):
    urdf = tmp_path / "hand.urdf"
    urdf.write_text("""<robot name="fixture">
<link name="base"/><link name="wrist"/><link name="palm"><collision/></link>
<link name="finger"><collision/></link><link name="pad"><collision/></link><link name="tip"/>
<joint name="arm" type="revolute"><parent link="base"/><child link="wrist"/></joint>
<joint name="mount" type="fixed"><parent link="wrist"/><child link="palm"/></joint>
<joint name="drive" type="revolute"><parent link="palm"/><child link="finger"/></joint>
<joint name="mimic" type="revolute"><parent link="palm"/><child link="pad"/><mimic joint="drive" multiplier="-1"/></joint>
<joint name="tip_fixed" type="fixed"><parent link="finger"/><child link="tip"/></joint>
</robot>""")
    root_world = pose(x=3.0)
    qpos = torch.zeros(1, 3)

    def fk(*, qpos, link_names=None, qpos_joint_names=None, name=None, to_matrix=False):
        if name is not None:
            assert name == "arm" and to_matrix is True
            result = torch.eye(4).repeat(len(qpos), 1, 1)
            result[:, 0, 3] = qpos[:, 0]
            return result
        assert tuple(qpos_joint_names) == ("arm", "drive", "mimic")
        result = torch.eye(4).repeat(len(qpos), len(link_names), 1, 1)
        for column, link in enumerate(link_names):
            result[:, column, 0, 3] = qpos[:, 0]
            if link in {"finger", "tip"}:
                result[:, column, 1, 3] = qpos[:, 1]
            if link == "pad":
                result[:, column, 2, 3] = qpos[:, 2]
        return result

    robot = SimpleNamespace(
        uid="robot",
        joint_names=["arm", "drive", "mimic"],
        cfg=SimpleNamespace(
            fpath=str(urdf),
            body_scale=[1.0] * 3,
            solver_cfg={
                "arm": SimpleNamespace(
                    end_link_name="wrist",
                    class_type="PytorchSolver",
                    pos_eps=0.005,
                    rot_eps=0.0005,
                    is_only_position_constraint=False,
                )
            },
        ),
        get_qpos=Mock(return_value=qpos),
        compute_fk=Mock(side_effect=fk),
        compute_ik=Mock(
            side_effect=AssertionError("candidate screening cannot call IK")
        ),
    )
    hand_bodies = {link: body() for link in ("palm", "finger", "pad")}
    native_robot = SimpleNamespace(
        path="/env/robot",
        arena_name="env",
        get_name=lambda: "robot",
        get_world_pose=lambda: root_world,
        get_link_names=lambda: ["wrist", "palm", "finger", "pad", "tip"],
        get_link_pose=lambda link: root_world,
        render_articulation=SimpleNamespace(
            get_physical_body=lambda link: hand_bodies.get(link)
        ),
    )
    housing = body()
    housing_pose = root_world @ pose(x=0.2)
    coffee = SimpleNamespace(
        path="/env/coffee",
        arena_name="env",
        get_name=lambda: "coffee",
        get_link_names=lambda: ["housing"],
        get_link_pose=Mock(side_effect=lambda link: housing_pose),
        render_articulation=SimpleNamespace(get_physical_body=lambda link: housing),
        get_link_desc=lambda link: SimpleNamespace(collisions=[object()]),
        desc=SimpleNamespace(body_scale=[1.0] * 3),
        cfg=SimpleNamespace(body_scale=[0.442324817181] * 3),
    )

    def rigid(uid, at, physical):
        return SimpleNamespace(
            path="/default/" + uid,
            arena_name="default",
            get_name=lambda: uid,
            get_world_pose=lambda: root_world @ at,
            native=lambda: SimpleNamespace(get_physical_body=lambda: physical),
        )

    cup = rigid("native_cup", pose(), body())
    table_body = body()
    table = rigid("table", pose(y=-0.3), table_body)
    handles = [native_robot, cup, coffee, table]
    scene = SimpleNamespace(
        get_paths=lambda: list(range(len(handles))),
        get=lambda index: handles[index],
        get_arena=lambda name: SimpleNamespace(
            get_root_node=lambda: SimpleNamespace(get_world_pose=lambda: root_world)
        ),
    )
    simulation = SimpleNamespace(
        num_envs=1, is_default_backend=True, spawn_result=scene
    )
    bindings = SimpleNamespace(
        rigid_objects=[SimpleNamespace(entity_id="cup", simulation_uid="native_cup")]
    )
    arm, hand = JointPositionTarget("arm", (0,)), JointPositionTarget("hand", (1, 2))
    targets = {"motion": arm, "grasp": hand}
    request = SimpleNamespace(
        invocation_id="handover",
        revision=0,
        goal=SimpleNamespace(semantics=SimpleNamespace(entity_id="cup")),
        binding=SimpleNamespace(
            endpoint=lambda role, kind: SimpleNamespace(
                require_target=lambda target_type: targets[kind]
            )
        ),
    )
    context = SimpleNamespace(
        batch_size=1,
        env_ids=torch.tensor([0]),
        scene=SimpleNamespace(timestamp=1.0, version=7),
        robot=SimpleNamespace(timestamp=1.0, qpos=qpos, root_pose=torch.eye(4)[None]),
    )
    resources = SimpleNamespace(
        second=SimpleNamespace(arm=arm, hand=hand, hand_open_qpos=torch.zeros(1, 2))
    )
    return SimpleNamespace(
        robot=robot,
        simulation=simulation,
        bindings=bindings,
        request=request,
        context=context,
        resources=resources,
        targets=targets,
        hand_bodies=hand_bodies,
        coffee=coffee,
        housing=housing,
        table=table,
        table_body=table_body,
        handles=handles,
        root_world=root_world,
        fk=fk,
    )


def build(case):
    return safety.build_handover_environment_screen(
        case.robot,
        case.simulation,
        case.bindings,
        case.request,
        case.context,
        case.resources,
    )


def plan(case, values):
    values = torch.tensor(values, dtype=torch.float32)[None]
    frames = tuple(
        SimpleNamespace(
            commands=tuple(
                EndpointCommand(
                    target=target,
                    payload=JointPositionPayload(
                        values[:, frame, list(target.joint_ids)]
                    ),
                )
                for target in case.targets.values()
            )
        )
        for frame in range(values.shape[1])
    )
    return SimpleNamespace(
        invocation_id="handover",
        invocation_revision=0,
        planned_scene_version=7,
        joint_trajectory=SimpleNamespace(positions=values),
        commands=SimpleNamespace(
            frame_count=values.shape[1], frames=frames, env_ids=torch.tensor([0])
        ),
    )


def test_interior_non_target_collision_rejects_endpoints_that_are_clear(case):
    screen = build(case)
    candidate = plan(case, [[0, 0, 0], [0.2, 0, 0], [0.4, 0, 0]])
    result = screen.inspect(candidate)
    assert result["valid"] is False and result["frame"] == 1
    assert result["obstacle_owner"] == "coffee"
    assert result["scope"] == "receiver_hand_non_target_scene_discrete_native_shapes"


def test_safe_inspection_does_not_mutate_plan_or_consume_rng(case):
    candidate = plan(case, [[0, 0, 0], [-0.1, 0, 0]])
    before = candidate.joint_trajectory.positions.clone()
    rng = torch.random.get_rng_state().clone()
    result = build(case).inspect(candidate)
    assert result["valid"] is True
    assert result["excluded_target_object"] == "cup"
    assert "held_target_contact" in result["not_evaluated"]
    assert torch.equal(candidate.joint_trajectory.positions, before)
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_observed_and_command_mimic_positions_are_not_recomputed(case):
    case.hand_bodies["palm"].is_collision_enable.return_value = False
    case.hand_bodies["finger"].is_collision_enable.return_value = False
    candidate = plan(case, [[0, 0, 0], [0.2, 0, 0.1]])
    screen = build(case)
    assert screen.inspect(candidate)["valid"] is True
    command_fk = case.robot.compute_fk.call_args.kwargs
    assert command_fk["qpos_joint_names"] == ("arm", "drive", "mimic")
    assert command_fk["qpos"][1, 2] == pytest.approx(0.1)
    assert command_fk["qpos"][1, 1] == 0


def test_pad_has_no_exception_for_non_target_environment(case):
    case.hand_bodies["palm"].is_collision_enable.return_value = False
    case.hand_bodies["finger"].is_collision_enable.return_value = False
    result = build(case).inspect(plan(case, [[0, 0, 0], [0.2, 0, 0]]))
    assert result["valid"] is False and result["receiver_link"] == "pad"


def test_existing_contact_envelope_rejects_positive_surface_gap(case):
    result = build(case).inspect(plan(case, [[0.177, 0, 0]]))
    assert result["valid"] is False
    assert result["minimum_distance_m"] == pytest.approx(0.003, abs=1e-7)
    assert result["native_pair_contact_offset_m"] == pytest.approx(0.006)


def test_native_disabled_body_is_explicitly_omitted(case):
    case.housing.is_collision_enable.return_value = False
    screen = build(case)
    assert screen.inspect(plan(case, [[0.2, 0, 0]]))["valid"] is True
    assert any(
        item["owner"] == "coffee" and item["status"] == "native_collision_disabled"
        for item in screen.inventory
    )


def test_native_shape_on_authored_collisionless_hand_link_is_not_omitted(case):
    for part in case.hand_bodies.values():
        part.is_collision_enable.return_value = False
    case.hand_bodies["tip"] = body()
    result = build(case).inspect(plan(case, [[0.2, 0, 0]]))
    assert result["valid"] is False and result["receiver_link"] == "tip"


def test_missing_bound_physical_body_cannot_certify_partial_inventory(case):
    case.bindings.rigid_objects.append(
        SimpleNamespace(entity_id="missing", simulation_uid="missing")
    )
    with pytest.raises(ValueError, match="physical_inventory_unavailable"):
        build(case)


def test_unknown_missing_body_is_not_treated_as_disabled(case):
    case.coffee.render_articulation.get_physical_body = lambda link: None
    with pytest.raises(ValueError, match="native_body_unavailable"):
        build(case)


def test_retained_descriptor_effective_scale_is_not_multiplied_by_asset_cfg(case):
    screen = build(case)
    obstacle = next(item for item in screen.obstacles if item.owner == "coffee")
    assert np.ptp(obstacle.meshes[0].vertices[:, 0]) == pytest.approx(0.02)
    entry = next(item for item in screen.inventory if item["owner"] == "coffee")
    assert entry["retained_body_scale"] == [1.0] * 3
    assert screen.inspect(plan(case, [[0.2, 0, 0]]))["valid"] is False


def test_global_static_body_is_not_dropped_by_arena_name(case):
    case.table.get_world_pose = lambda: case.root_world @ pose(x=0.1)
    result = build(case).inspect(plan(case, [[0.1, 0, 0]]))
    assert result["valid"] is False and result["obstacle_owner"] == "table"


@pytest.mark.parametrize(
    "case_name",
    ["clock", "root", "qpos", "fk", "hand_joints", "offset", "empty_shapes"],
)
def test_missing_or_inconsistent_native_geometry_fails_closed(case, case_name):
    if case_name == "clock":
        case.context.scene.timestamp = 0.9
    elif case_name == "root":
        case.context.robot.root_pose[0, 0, 3] = 0.1
    elif case_name == "qpos":
        case.robot.get_qpos.return_value = torch.ones(1, 3)
    elif case_name == "fk":
        case.robot.compute_fk.side_effect = lambda **kw: case.fk(**kw) + 0.01
    elif case_name == "hand_joints":
        case.resources.second.hand = JointPositionTarget("hand", (1,))
        case.targets["grasp"] = case.resources.second.hand
    elif case_name == "offset":
        case.housing.get_contact_offset.return_value = float("nan")
    elif case_name == "empty_shapes":
        case.housing.shapes.clear()
    with pytest.raises(ValueError):
        build(case)


@pytest.mark.parametrize("damage", ["invocation", "revision", "scene", "command"])
def test_command_certificate_requires_exact_current_payload_and_identity(case, damage):
    screen = build(case)
    candidate = plan(case, [[0, 0, 0]])
    if damage == "invocation":
        candidate.invocation_id = "other"
    elif damage == "revision":
        candidate.invocation_revision = 1
    elif damage == "scene":
        candidate.planned_scene_version = 8
    else:
        candidate.commands.frames[0].commands[0].payload.positions.fill_(0.2)
    with pytest.raises(ValueError):
        screen.inspect(candidate)


def add_plane(case, *, normal=(0.0, 0.0, 1.0), distance=0.0, local=None, at=None):
    geometry = PlaneGeometry()
    geometry.normal = np.array(normal)
    geometry.distance = distance
    geometry.local_pose = np.eye(4) if local is None else local
    physical = body()
    physical.shapes[:] = [geometry]
    handle = SimpleNamespace(
        path="/default/ground",
        arena_name="default",
        get_name=lambda: "ground",
        get_world_pose=lambda: case.root_world @ (np.eye(4) if at is None else at),
        native=lambda: SimpleNamespace(get_physical_body=lambda: physical),
    )
    case.handles.append(handle)
    return physical


def test_global_native_plane_is_infinite_not_a_finite_render_box(case):
    add_plane(case)
    result = build(case).inspect(plan(case, [[10000.0, 0, 0]]))
    assert result["valid"] is False
    assert result["obstacle_owner"] == "ground"
    assert result["reason"] == "receiver_hand_non_target_plane_contact_envelope"
    assert result["minimum_signed_distance_m"] < 0
    assert result["revision"] == 4


def test_plane_clearance_checks_original_native_offsets_on_free_side(case):
    add_plane(case, at=pose(z=-0.1))
    result = build(case).inspect(plan(case, [[0.0, 0, 0]]))
    assert result["valid"] is True
    assert result["plane_queries"] > 0
    assert result["minimum_plane_signed_distance_m"] == pytest.approx(0.09, abs=1e-7)


def test_positive_plane_gap_inside_original_contact_envelope_is_rejected(case):
    add_plane(case, at=pose(z=-0.013))
    result = build(case).inspect(plan(case, [[0, 0, 0]]))
    assert result["valid"] is False
    assert result["minimum_signed_distance_m"] == pytest.approx(0.003, abs=1e-7)
    assert result["native_pair_contact_offset_m"] == pytest.approx(0.006)


def test_native_plane_world_pose_is_converted_to_robot_arena(case):
    case.root_world[:3, :3] = np.array([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    case.coffee.get_link_pose.side_effect = lambda link: case.root_world @ pose(x=0.2)
    add_plane(case, at=pose(z=-0.013))
    result = build(case).inspect(plan(case, [[0, 0, 0]]))
    assert result["valid"] is False
    assert result["obstacle_owner"] == "ground"
    assert result["plane_normal_arena"] == pytest.approx([0.0, 0, 1])
    assert result["plane_point_arena"] == pytest.approx([0.0, 0, -0.013])


def test_plane_normal_distance_and_shape_pose_are_applied_once(case):
    local = pose(x=0.2)
    local[:3, :3] = np.array([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    add_plane(case, local=local, distance=0.05)
    screen = build(case)
    assert screen.inspect(plan(case, [[0.3, 0, 0]]))["valid"] is True
    result = screen.inspect(plan(case, [[0.255, 0, 0]]))
    assert result["valid"] is False
    assert result["plane_normal_arena"] == pytest.approx([1.0, 0, 0])
    assert result["plane_point_arena"] == pytest.approx([0.25, 0, 0])


def test_compound_body_keeps_plane_and_finite_native_shapes(case):
    physical = add_plane(case, at=pose(z=-0.1))
    cube = BoxGeometry()
    cube.half_extents = np.full(3, 0.01)
    cube.local_pose = pose(x=-0.1, z=0.1)
    physical.shapes.append(cube)
    screen = build(case)
    assert screen.inspect(plan(case, [[0, 0, 0]]))["valid"] is True
    result = screen.inspect(plan(case, [[-0.1, 0, 0]]))
    assert result["valid"] is False and result["obstacle_owner"] == "ground"
    assert result["reason"] == "receiver_hand_non_target_contact_envelope"
    entry = next(item for item in screen.inventory if item["owner"] == "ground")
    assert entry["shapes"] == 2 and entry["plane_shapes"] == 1


def test_unknown_native_type_names_owner_shape_and_type(case):
    sphere = SphereGeometry()
    sphere.radius = 0.01
    sphere.local_pose = np.eye(4)
    case.housing.shapes[:] = [sphere]
    with pytest.raises(
        ValueError,
        match=r"unsupported:/env/coffee/housing:shape=0:type=.*SphereGeometry",
    ):
        build(case)


@pytest.mark.parametrize(
    "normal,distance",
    [((0.0, 0, 0), 0.0), ((float("nan"), 0, 1), 0.0), ((0.0, 0, 1), float("inf"))],
)
def test_invalid_native_plane_fails_with_named_identity(case, normal, distance):
    add_plane(case, normal=normal, distance=distance)
    with pytest.raises(
        ValueError, match=r"native_plane_unavailable:/default/ground:shape=0"
    ):
        build(case)


def pool(*positions):
    return torch.tensor(
        np.array([[pose(*position) for position in positions]]), dtype=torch.float32
    )


def screen_pool(screen, case, poses, *, eligible=None, samples=5, distance=0.0):
    return screen.candidate_mask(
        poses,
        torch.ones(poses.shape[:2], dtype=torch.bool) if eligible is None else eligible,
        resources=case.resources,
        approach_samples=samples,
        pre_grasp_distance=distance,
    )


def test_candidate_path_rejects_interior_collision_and_keeps_original_indices(case):
    poses = pool((0.4, 0, 0), (-0.4, 0, 0), (0.1, 0, 0))
    eligible = torch.tensor([[True, True, False]])
    before, validity = poses.clone(), eligible.clone()
    mask, report = screen_pool(build(case), case, poses, eligible=eligible)
    assert mask.tolist() == [[False, True, False]]
    assert torch.equal(poses, before) and torch.equal(eligible, validity)
    assert report["eligible_count"] == 2
    assert [item["pool_index"] for item in report["candidates"]] == [0, 1]
    failure = report["candidates"][0]
    assert failure["frame"] == 1 and failure["obstacle_owner"] == "coffee"
    assert report["requires_complete_command_guard"] is True


def test_candidate_mask_does_not_mutate_robot_resources_or_rng(case):
    case.resources.second.hand_open_qpos[:] = torch.tensor([[0.04, -0.03]])
    qpos = case.context.robot.qpos.clone()
    opened = case.resources.second.hand_open_qpos.clone()
    rng = torch.random.get_rng_state().clone()
    screen = build(case)
    mask, _ = screen_pool(screen, case, pool((-0.4, 0, 0)))
    assert mask.tolist() == [[True]]
    assert torch.equal(case.context.robot.qpos, qpos)
    assert torch.equal(case.robot.get_qpos(), qpos)
    assert torch.equal(case.resources.second.hand_open_qpos, opened)
    assert torch.equal(torch.random.get_rng_state(), rng)
    case.robot.compute_ik.assert_not_called()
    named = case.robot.compute_fk.call_args.kwargs
    assert named["qpos_joint_names"] == ("arm", "drive", "mimic")
    assert torch.equal(named["qpos"][:, 1:], opened)


def test_candidate_open_hand_uses_bound_actual_mimic_not_drive_formula(case):
    case.hand_bodies["palm"].is_collision_enable.return_value = False
    case.hand_bodies["finger"].is_collision_enable.return_value = False
    case.resources.second.hand_open_qpos[:] = torch.tensor([[0.0, 0.1]])
    mask, report = screen_pool(build(case), case, pool((0.4, 0, 0)))
    assert mask.tolist() == [[True]]
    np.testing.assert_allclose(report["open_hand_joint_positions"], [[0.0, 0.1]])
    case.resources.second.hand_open_qpos.zero_()
    assert screen_pool(build(case), case, pool((0.4, 0, 0)))[0].tolist() == [[False]]


def test_candidate_mask_reuses_exact_original_92_sample_cartesian_path(case):
    screen = build(case)
    screen._inspect_hand_poses = Mock(return_value={"valid": True})
    poses = pool((-0.4, 0.2, 0.1))
    screen_pool(screen, case, poses, samples=92, distance=0.1)
    start = case.robot.compute_fk(qpos=torch.zeros(1, 1), name="arm", to_matrix=True)
    grasp = poses[:, 0]
    pre_grasp = safety.translate_pose_world(grasp, -grasp[:, :3, 2] * 0.1)
    targets = safety.build_pose_plan_states(torch.stack((pre_grasp, grasp), dim=1))
    generator = SimpleNamespace(
        robot=case.robot,
        _generate_checked=Mock(
            return_value=SimpleNamespace(success=torch.tensor([True]))
        ),
    )
    motion._generate_cartesian_approach(
        generator,
        targets,
        MotionGenOptions(
            strategy="ik_interp",
            start_qpos=torch.zeros(1, 1),
            control_part="arm",
            sample_count=92,
        ),
    )
    states = generator._generate_checked.call_args.args[0]
    expected = torch.stack([start, *(state.xpos for state in states)], dim=1)[0]
    inspected = screen._inspect_hand_poses.call_args.args[0]
    assert inspected.shape == (92, len(screen.parts), 4, 4)
    np.testing.assert_array_equal(inspected[:, 0], expected.numpy())
    assert torch.equal(expected[45], pre_grasp[0])
    assert torch.equal(expected[91], grasp[0])


def test_candidate_geometry_uses_arena_tcp_and_root_local_full_fk_once(case):
    case.root_world[:3, :3] = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    root = torch.tensor(pose(y=0.4), dtype=torch.float32)
    root[:3, :3] = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    case.context.robot.root_pose[:] = root
    case.handles[0].get_world_pose = lambda: case.root_world @ root.numpy()
    case.handles[0].get_link_pose = lambda link: case.root_world @ root.numpy()
    case.coffee.get_link_pose.side_effect = lambda link: case.root_world @ pose(x=0.2)

    def fk(**kwargs):
        result = case.fk(**kwargs)
        if kwargs.get("name") is not None:
            result = root @ result
        return result

    case.robot.compute_fk.side_effect = fk
    screen = build(case)
    screen._inspect_hand_poses = Mock(return_value={"valid": True})
    screen_pool(screen, case, pool((-0.4, 0.4, 0)))
    inspected = screen._inspect_hand_poses.call_args.args[0]
    np.testing.assert_allclose(inspected[0, 0], root.numpy(), atol=1e-7)
    np.testing.assert_allclose(inspected[-1, 0], pose(x=-0.4, y=0.4), atol=1e-7)


def test_candidate_mask_uses_same_kernel_for_mixed_plane_and_mesh_scene(case):
    add_plane(case, at=pose(z=-0.1))
    mask, report = screen_pool(
        build(case), case, pool((-0.4, 0, 0), (-0.4, 0, -0.2), (0.4, 0, 0))
    )
    assert mask.tolist() == [[True, False, False]]
    assert report["candidates"][0]["plane_queries"] > 0
    assert report["candidates"][1]["obstacle_owner"] == "ground"
    assert report["candidates"][2]["obstacle_owner"] == "coffee"


def test_safe_nominal_candidate_does_not_replace_complete_command_guard(case):
    screen = build(case)
    assert screen_pool(screen, case, pool((-0.4, 0, 0)))[0].tolist() == [[True]]
    result = screen.inspect(plan(case, [[0, 0, 0], [0.2, 0, 0]]))
    assert result["valid"] is False and result["obstacle_owner"] == "coffee"
    assert result["scope"] == "receiver_hand_non_target_scene_discrete_native_shapes"


def test_empty_original_pool_never_invokes_kinematics_or_checks_padding(case):
    screen = build(case)
    case.robot.compute_fk.reset_mock()
    poses = pool((0, 0, 0))
    poses.fill_(float("nan"))
    mask, report = screen_pool(screen, case, poses, eligible=torch.tensor([[False]]))
    assert mask.tolist() == [[False]] and report["candidates"] == []
    case.robot.compute_fk.assert_not_called()


@pytest.mark.parametrize(
    "damage", ["mask", "pose", "steps", "distance", "open", "resource", "clock", "qpos"]
)
def test_candidate_mask_unavailable_inputs_fail_closed(case, damage):
    screen = build(case)
    poses, eligible, samples, distance = (
        pool((-0.4, 0, 0)),
        torch.tensor([[True]]),
        5,
        0.0,
    )
    if damage == "mask":
        eligible = eligible.float()
    elif damage == "pose":
        poses[0, 0, 0, 0] = float("nan")
    elif damage == "steps":
        samples = 2
    elif damage == "distance":
        distance = float("inf")
    elif damage == "open":
        case.resources.second.hand_open_qpos = torch.zeros(1, 1)
    elif damage == "resource":
        case.resources.second.hand = JointPositionTarget("other", (1, 2))
    elif damage == "clock":
        case.context.scene.timestamp = 0.9
    else:
        case.robot.get_qpos.return_value = torch.ones(1, 3)
    with pytest.raises(ValueError):
        screen_pool(
            screen, case, poses, eligible=eligible, samples=samples, distance=distance
        )


def exchange_pool(
    screen, case, poses, *, eligible=None, source=None, destination=None, exchange=None
):
    return screen.exchange_candidates(
        torch.eye(4)[None] if exchange is None else exchange,
        poses,
        torch.ones(poses.shape[:2], dtype=torch.bool) if eligible is None else eligible,
        resources=case.resources,
        approach_samples=92,
        pre_grasp_distance=0.1,
        source_root=(
            torch.tensor(pose(y=-1), dtype=torch.float32)[None]
            if source is None
            else source
        ),
        destination_root=(
            torch.tensor(pose(y=1), dtype=torch.float32)[None]
            if destination is None
            else destination
        ),
    )


def test_exchange_derives_three_bounded_native_points_without_mutation_or_ik(case):
    screen = build(case)
    poses = pool((0.2, 0, 0), (0.25, 0, 0))
    eligible = torch.tensor([[True, False]])
    original, validity = poses.clone(), eligible.clone()
    qpos = case.context.robot.qpos.clone()
    rng = torch.random.get_rng_state().clone()
    proposals, report = exchange_pool(screen, case, poses, eligible=eligible)
    assert len(proposals) == 3
    assert {item["axis"] for item in proposals} == {"+x", "-x", "+z"}
    assert [p["norm_m"] for p in proposals] == sorted(p["norm_m"] for p in proposals)
    assert report["dominant_root_axis"] == 1
    assert report["preserved_midpoint_coordinate"] == 0
    assert report["suffix_indices_inclusive"] == [45, 91]
    assert report["requires_complete_command_guard"] is True
    assert report["near_native_bodies"][0]["name"] == "/env/coffee/housing"
    for proposal in proposals:
        assert proposal["translation"].shape == (1, 3)
        assert proposal["translation"][0, 1] == 0
        assert proposal["mask"].shape == eligible.shape
        assert not proposal["mask"][0, 1]
        assert [row["pool_index"] for row in proposal["report"]["candidates"]] == [0]
    assert any(item["mask"].any() for item in proposals)
    assert torch.equal(poses, original) and torch.equal(eligible, validity)
    assert torch.equal(case.context.robot.qpos, qpos)
    assert torch.equal(torch.random.get_rng_state(), rng)
    case.robot.compute_ik.assert_not_called()


def test_exchange_uncertainty_is_derived_from_original_ik_tolerances(case):
    poses = pool((0.2, 0, 0))
    first, report = exchange_pool(build(case), case, poses)
    uncertainty = report["ik_geometric_uncertainty"]
    expected = 0.005 + 2 * uncertainty["hand_radius_m"] * np.sin(0.0005 / 2)
    assert uncertainty["surface_uncertainty_m"] == pytest.approx(expected)
    assert uncertainty["not_a_physical_acceptance_threshold"] is True
    case.robot.cfg.solver_cfg["arm"].pos_eps += 0.02
    second, _ = exchange_pool(build(case), case, poses)
    before = {p["axis"]: p["norm_m"] for p in first}
    after = {p["axis"]: p["norm_m"] for p in second}
    for axis in before:
        assert after[axis] - before[axis] == pytest.approx(0.02, abs=1e-7)


def test_exchange_native_contact_offsets_determine_separation_not_fixed_padding(case):
    poses = pool((0.2, 0, 0))
    first, _ = exchange_pool(build(case), case, poses)
    case.housing.get_contact_offset.return_value += 0.01
    second, _ = exchange_pool(build(case), case, poses)
    previous = {p["axis"]: p["norm_m"] for p in first}
    for proposal in second:
        assert proposal["norm_m"] - previous[proposal["axis"]] == pytest.approx(
            0.01, abs=1e-7
        )


def test_exchange_root_x_separation_preserves_x_and_only_moves_y_or_up(case):
    proposals, report = exchange_pool(
        build(case),
        case,
        pool((0.2, 0, 0)),
        source=torch.tensor(pose(x=-1), dtype=torch.float32)[None],
        destination=torch.tensor(pose(x=1), dtype=torch.float32)[None],
    )
    assert report["dominant_root_axis"] == 0
    assert {p["axis"] for p in proposals} == {"+y", "-y", "+z"}
    assert all(p["translation"][0, 0] == 0 for p in proposals)


def test_exchange_full_path_never_translates_original_receiver_start(case):
    screen = build(case)
    screen._inspect_hand_poses = Mock(return_value={"valid": True})
    proposals, _ = exchange_pool(screen, case, pool((0.2, 0, 0)))
    assert len(proposals) == 3
    for call in screen._inspect_hand_poses.call_args_list:
        poses = call.args[0]
        assert poses.shape == (92, len(screen.parts), 4, 4)
        np.testing.assert_array_equal(
            poses[0], np.repeat(np.eye(4)[None], len(screen.parts), axis=0)
        )


def test_exchange_derivation_keeps_arena_and_robot_root_frames_separate(case):
    case.root_world[:3, :3] = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    root = torch.tensor(pose(y=0.4), dtype=torch.float32)
    root[:3, :3] = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    case.context.robot.root_pose[:] = root
    case.handles[0].get_world_pose = lambda: case.root_world @ root.numpy()
    case.handles[0].get_link_pose = lambda link: case.root_world @ root.numpy()
    case.coffee.get_link_pose.side_effect = lambda link: case.root_world @ pose(
        x=0.2, y=0.4
    )

    def fk(**kwargs):
        result = case.fk(**kwargs)
        return root @ result if kwargs.get("name") is not None else result

    case.robot.compute_fk.side_effect = fk
    proposals, report = exchange_pool(
        build(case),
        case,
        pool((0.2, 0.4, 0)),
        source=torch.tensor(pose(y=-0.6), dtype=torch.float32)[None],
        destination=torch.tensor(pose(y=1.4), dtype=torch.float32)[None],
        exchange=root[None],
    )
    assert proposals and report["preserved_midpoint_coordinate"] == pytest.approx(0.4)
    assert all(p["translation"][0, 1] == 0 for p in proposals)


def test_exchange_rechecks_all_scene_bodies_not_only_original_near_set(case):
    case.table.get_world_pose = lambda: case.root_world @ pose(x=0.2, z=0.13)
    proposals, report = exchange_pool(build(case), case, pool((0.2, 0, 0)))
    assert all(
        item["name"] != "/default/table" for item in report["near_native_bodies"]
    )
    lifted = next(p for p in proposals if p["axis"] == "+z")
    assert lifted["mask"].tolist() == [[False]]
    assert lifted["report"]["candidates"][0]["obstacle_owner"] == "table"


def test_exchange_keeps_global_planes_in_fixed_start_recheck(case):
    add_plane(case)
    proposals, report = exchange_pool(build(case), case, pool((0.2, 0, 0)))
    assert len(report["near_native_planes"]) == 1
    assert [p["axis"] for p in proposals] == ["+z"]
    assert proposals[0]["mask"].tolist() == [[False]]
    rejection = proposals[0]["report"]["candidates"][0]
    assert rejection["frame"] == 0 and rejection["obstacle_owner"] == "ground"


def test_exchange_with_no_near_suffix_geometry_does_not_add_points(case):
    screen = build(case)
    screen.candidate_mask = Mock(
        side_effect=AssertionError("No derived point expected")
    )
    proposals, report = exchange_pool(screen, case, pool((-0.4, 0, 0)))
    assert proposals == [] and report["near_native_bodies"] == []
    screen.candidate_mask.assert_not_called()


def test_exchange_empty_original_pool_does_not_invoke_fk(case):
    screen = build(case)
    case.robot.compute_fk.reset_mock()
    proposals, report = exchange_pool(
        screen, case, pool((0.2, 0, 0)), eligible=torch.tensor([[False]])
    )
    assert proposals == [] and report["eligible_count"] == 0
    case.robot.compute_fk.assert_not_called()


@pytest.mark.parametrize(
    "damage",
    [
        "vertical",
        "same_roots",
        "midpoint",
        "solver",
        "position",
        "rotation",
        "position_only",
        "boolean",
    ],
)
def test_exchange_unknown_geometry_or_policy_never_invents_a_default(case, damage):
    source, destination, exchange = pose(y=-1), pose(y=1), pose()
    solver = case.robot.cfg.solver_cfg["arm"]
    if damage == "vertical":
        source, destination = pose(z=-1), pose(z=1)
    elif damage == "same_roots":
        source, destination = pose(), pose()
    elif damage == "midpoint":
        exchange = pose(y=0.1)
    elif damage == "solver":
        solver.class_type = "unknown"
    elif damage == "position":
        solver.pos_eps = float("nan")
    elif damage == "rotation":
        del solver.rot_eps
    elif damage == "position_only":
        solver.is_only_position_constraint = True
    else:
        solver.pos_eps = True
    with pytest.raises(ValueError):
        exchange_pool(
            build(case),
            case,
            pool((0.2, 0, 0)),
            source=torch.tensor(source)[None],
            destination=torch.tensor(destination)[None],
            exchange=torch.tensor(exchange)[None],
        )
