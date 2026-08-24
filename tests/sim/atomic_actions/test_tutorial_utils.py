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

"""Tests for atomic-action tutorial helpers."""

from __future__ import annotations

import importlib
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import torch

from embodichain.lab.sim.atomic_actions import TimedTrajectory
from scripts.tutorials.atomic_action.dynamic_obstacle_recovery import (
    _animate_obstacle_to_pose,
    _blocking_obstacle_pose,
    _maximum_path_deviation,
    _minimum_cuboid_clearance,
)
from scripts.tutorials.atomic_action.coordinated_pickment import (
    compute_left_to_right_arm_direction,
)
from scripts.tutorials.atomic_action.scenario_utils import (
    create_dual_tutorial_robot_cfg,
)
from scripts.tutorials.atomic_action.tutorial_utils import (
    TUTORIAL_ROBOTS,
    broadcast_pose_batch,
    broadcast_waypoint_pose_batch,
    clone_local_pose_from_first_env,
    create_antipodal_semantics,
    create_curobo_motion_generator,
    create_franka_panda_robot_cfg,
    create_tutorial_argument_parser,
    create_tutorial_robot_cfg,
    create_ur5_gripper_robot_cfg,
    create_parallel_jaw_grasp_pose_generator,
    get_hand_open_close_qpos,
    replay_trajectory,
    should_open_tutorial_window,
    should_wait_for_tutorial_input,
)

PHYSICS_DT = 0.1
MOVE_DURATION = 0.25
Y_OFFSET = 0.18
EXPECTED_STEP_COUNT = 3
CUBOID_SIZE = (0.2, 0.2, 0.2)
FRANKA_TUTORIAL_BASE_ROTATION = (0.0, 0.0, 180.0)
DUAL_FRANKA_MOUNT_X_AXIS = torch.tensor([0.0, -1.0, 0.0])
PGI_TUTORIAL_TCP = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.17],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
ATOMIC_ACTION_TUTORIAL_MODULES = (
    "assemble",
    "coordinated_pickment",
    "coordinated_placement",
    "dynamic_obstacle_recovery",
    "hand_over",
    "move_end_effector",
    "move_held_object",
    "move_joints",
    "moving_target_recovery",
    "pickup",
    "place",
    "press",
)


def _run_obstacle_animation(*, pace_wall_time: bool) -> tuple[MagicMock, MagicMock]:
    obstacle = MagicMock()
    adapter = MagicMock()
    adapter.physics_dt = PHYSICS_DT
    start_pose = torch.eye(4).unsqueeze(0)
    start_pose[:, 1, 3] = -0.2
    target_pose = start_pose.clone()
    target_pose[:, 1, 3] += Y_OFFSET

    result = _animate_obstacle_to_pose(
        obstacle,
        adapter,
        start_pose,
        target_pose=target_pose,
        duration=MOVE_DURATION,
        pace_wall_time=pace_wall_time,
    )

    assert torch.equal(result, target_pose)
    assert result.data_ptr() != target_pose.data_ptr()
    return obstacle, adapter


def test_should_wait_for_tutorial_input_is_disabled_for_headless_modes() -> None:
    assert (
        should_wait_for_tutorial_input(
            Namespace(
                auto_play=False,
                headless=True,
                diagnose_plan=False,
                headless_play=False,
            )
        )
        is False
    )
    assert (
        should_wait_for_tutorial_input(
            Namespace(
                auto_play=False,
                headless=False,
                viser=True,
                diagnose_plan=False,
                headless_play=False,
            )
        )
        is False
    )
    assert (
        should_wait_for_tutorial_input(
            Namespace(
                auto_play=False,
                headless=False,
                diagnose_plan=True,
                headless_play=False,
            )
        )
        is False
    )
    assert (
        should_wait_for_tutorial_input(
            Namespace(
                auto_play=False,
                headless=False,
                diagnose_plan=False,
                headless_play=True,
            )
        )
        is False
    )


def test_viser_does_not_open_native_tutorial_window() -> None:
    args = Namespace(
        headless=False,
        viser=True,
        diagnose_plan=False,
        headless_play=False,
    )

    assert not should_open_tutorial_window(args)


def test_broadcast_pose_batch_repeats_single_pose_for_each_env() -> None:
    pose = torch.eye(4, dtype=torch.float32)

    batched = broadcast_pose_batch(pose, num_envs=3)

    assert batched.shape == (3, 4, 4)
    assert torch.allclose(batched[0], pose)
    assert torch.allclose(batched[1], pose)
    assert torch.allclose(batched[2], pose)


def test_broadcast_waypoint_pose_batch_repeats_waypoints_for_each_env() -> None:
    waypoints = torch.stack(
        [torch.eye(4, dtype=torch.float32), 2.0 * torch.eye(4, dtype=torch.float32)],
        dim=0,
    )

    batched = broadcast_waypoint_pose_batch(waypoints, num_envs=2)

    assert batched.shape == (2, 2, 4, 4)
    assert torch.allclose(batched[0], waypoints)
    assert torch.allclose(batched[1], waypoints)


def test_clone_local_pose_from_first_env_sets_shared_pose() -> None:
    first_pose = torch.eye(4, dtype=torch.float32)
    first_pose[0, 3] = 0.2
    poses = torch.stack(
        [
            first_pose,
            2.0 * torch.eye(4, dtype=torch.float32),
            3.0 * torch.eye(4, dtype=torch.float32),
        ],
        dim=0,
    )
    entity = MagicMock()
    entity.get_local_pose.return_value = poses

    shared = clone_local_pose_from_first_env(entity)

    expected = first_pose.unsqueeze(0).repeat(3, 1, 1)
    assert torch.allclose(shared, expected)
    entity.set_local_pose.assert_called_once()
    assert torch.allclose(entity.set_local_pose.call_args.args[0], expected)


def test_create_antipodal_semantics_keeps_mesh_data_on_affordance() -> None:
    vertices = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    triangles = torch.tensor([[0, 1, 1]])
    obj = MagicMock()
    obj.get_vertices.return_value = vertices.unsqueeze(0)
    obj.get_triangles.return_value = triangles.unsqueeze(0)

    semantics = create_antipodal_semantics(obj, label="cube")

    assert semantics.entity is obj
    assert semantics.label == "cube"
    assert semantics.geometry == {}
    assert torch.equal(semantics.affordance.mesh_vertices, vertices)
    assert torch.equal(semantics.affordance.mesh_triangles, triangles)
    assert not hasattr(semantics.affordance, "generator_cfg")

    generator = create_parallel_jaw_grasp_pose_generator(
        n_sample=64,
        force_refresh=True,
    )
    assert generator.algorithm_cfg.sample_count == 64
    assert generator.annotation_cfg.force_refresh is True
    assert generator.gripper_model.model_id == "dh_pgi_140_80"


def test_franka_tutorial_config_uses_ur5_gripper_component() -> None:
    ur5_cfg = create_ur5_gripper_robot_cfg()
    franka_cfg = create_franka_panda_robot_cfg()

    assert franka_cfg.urdf_cfg.components["hand"]["urdf_path"] == (
        ur5_cfg.urdf_cfg.components["hand"]["urdf_path"]
    )
    assert franka_cfg.urdf_cfg.components["arm"]["urdf_path"].endswith(
        "/Franka/Panda/Panda.urdf"
    )
    assert franka_cfg.init_qpos[-2:] == [0.0, 0.0]
    assert franka_cfg.init_rot == FRANKA_TUTORIAL_BASE_ROTATION
    for property_name in ("stiffness", "damping", "max_effort"):
        ur5_values = getattr(ur5_cfg.drive_pros, property_name)
        franka_values = getattr(franka_cfg.drive_pros, property_name)
        assert franka_values["gripper_finger1_joint_1"] == (
            ur5_values["gripper_finger1_joint_1"]
        )
        assert "fr3_finger_joint[1-2]" not in franka_values


def test_ur5_and_franka_configs_share_place_binding_contract() -> None:
    configs = (
        create_ur5_gripper_robot_cfg(),
        create_franka_panda_robot_cfg(),
    )

    assert all(set(cfg.control_parts) == {"arm", "hand"} for cfg in configs)
    assert all(set(cfg.solver_cfg) == {"arm"} for cfg in configs)
    assert configs[0].control_parts["hand"] == configs[1].control_parts["hand"]
    assert all(
        torch.allclose(torch.as_tensor(cfg.solver_cfg["arm"].tcp), PGI_TUTORIAL_TCP)
        for cfg in configs
    )


@pytest.mark.parametrize(
    ("robot_type", "arm_dof", "solver_name"),
    (("ur5", 6, "URSolverCfg"), ("franka", 7, "PytorchSolverCfg")),
)
def test_dual_tutorial_configs_share_pgi_binding_contract(
    robot_type: str,
    arm_dof: int,
    solver_name: str,
) -> None:
    single_cfg = create_tutorial_robot_cfg(robot_type)
    dual_cfg = create_dual_tutorial_robot_cfg(
        robot_type=robot_type,
        uid=f"test_{robot_type}",
        urdf_name=f"test_dual_{robot_type}",
        tcp_z=0.121,
    )
    expected_tcp = PGI_TUTORIAL_TCP.clone()
    expected_tcp[2, 3] = 0.121

    assert tuple(dual_cfg.urdf_cfg.components) == (
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
    )
    expected_arm_home = list(single_cfg.init_qpos[:arm_dof])
    assert dual_cfg.init_qpos[: 2 * arm_dof : 2] == expected_arm_home
    assert dual_cfg.init_qpos[1 : 2 * arm_dof : 2] == expected_arm_home
    for side in ("left", "right"):
        assert len(dual_cfg.control_parts[f"{side}_arm"]) == arm_dof
        assert dual_cfg.control_parts[f"{side}_hand"] == [
            f"{side}_gripper_finger1_joint_1"
        ]
        assert dual_cfg.urdf_cfg.components[f"{side}_hand"]["urdf_path"] == (
            single_cfg.urdf_cfg.components["hand"]["urdf_path"]
        )
        solver = dual_cfg.solver_cfg[f"{side}_arm"]
        assert type(solver).__name__ == solver_name
        assert torch.allclose(torch.as_tensor(solver.tcp), expected_tcp)


def test_dual_franka_mount_preserves_single_arm_facing_direction() -> None:
    cfg = create_dual_tutorial_robot_cfg(
        robot_type="franka",
        uid="test_franka_orientation",
        urdf_name="test_dual_franka_orientation",
        tcp_z=0.121,
    )

    for side in ("left", "right"):
        mount = torch.as_tensor(
            cfg.urdf_cfg.components[f"{side}_arm"]["transform"],
            dtype=torch.float32,
        )
        assert torch.allclose(
            mount[:3, 0],
            DUAL_FRANKA_MOUNT_X_AXIS,
            atol=1e-6,
        )


def test_hand_commands_use_pgi_open_limit() -> None:
    robot = MagicMock()
    robot.device = torch.device("cpu")
    robot.get_qpos_limits.return_value = torch.tensor([[[0.0, 0.04]]])

    hand_open, hand_close = get_hand_open_close_qpos(robot)

    assert torch.allclose(hand_open, torch.tensor([0.0]))
    assert torch.allclose(hand_close, torch.tensor([0.024]))


def test_curobo_motion_generator_factory_selects_curobo_backend() -> None:
    robot = MagicMock(uid="tutorial_robot")

    with patch(
        "scripts.tutorials.atomic_action.tutorial_utils.MotionGenerator"
    ) as motion_generator_cls:
        result = create_curobo_motion_generator(robot)

    cfg = motion_generator_cls.call_args.kwargs["cfg"]
    assert result is motion_generator_cls.return_value
    assert cfg.planner_cfg.planner_type == "curobo"
    assert cfg.planner_cfg.robot_uid == "tutorial_robot"


def test_shared_robot_selection_keeps_ur5_default_and_accepts_franka() -> None:
    parser = create_tutorial_argument_parser("test parser")
    default_args = parser.parse_args([])
    franka_args = parser.parse_args(["--robot", "franka"])

    assert TUTORIAL_ROBOTS == ("ur5", "franka")
    assert default_args.robot == "ur5"
    assert franka_args.robot == "franka"


def test_arm_direction_uses_selected_robot_solver_roots() -> None:
    robot = MagicMock()
    robot.cfg.solver_cfg = {
        "left_arm": SimpleNamespace(root_link_name="left_franka_root"),
        "right_arm": SimpleNamespace(root_link_name="right_franka_root"),
    }
    left_pose = torch.eye(4).unsqueeze(0)
    right_pose = torch.eye(4).unsqueeze(0)
    right_pose[0, 1, 3] = 2.0
    robot.get_link_pose.side_effect = (left_pose, right_pose)

    direction = compute_left_to_right_arm_direction(robot, "cpu")

    assert torch.allclose(direction, torch.tensor([0.0, 1.0, 0.0]))
    assert robot.get_link_pose.call_args_list == [
        call(link_name="left_franka_root", env_ids=[0], to_matrix=True),
        call(link_name="right_franka_root", env_ids=[0], to_matrix=True),
    ]


@pytest.mark.parametrize("module_name", ATOMIC_ACTION_TUTORIAL_MODULES)
def test_all_atomic_action_tutorials_accept_both_robot_choices(
    module_name: str,
) -> None:
    module = importlib.import_module(f"scripts.tutorials.atomic_action.{module_name}")

    with patch("sys.argv", [f"{module_name}.py"]):
        default_args = module.parse_arguments()
    with patch("sys.argv", [f"{module_name}.py", "--robot", "franka"]):
        franka_args = module.parse_arguments()

    assert default_args.robot == "ur5"
    assert franka_args.robot == "franka"


def test_replay_timed_trajectory_uses_arrival_intervals() -> None:
    sim = MagicMock()
    sim.sim_config.physics_dt = 0.1
    robot = MagicMock()
    trajectory = TimedTrajectory.from_positions(
        torch.zeros(1, 3, 2),
        env_ids=torch.tensor([0], dtype=torch.long),
        dt=torch.tensor([[0.0, 0.2, 0.25]]),
    )

    with patch("scripts.tutorials.atomic_action.tutorial_utils.time.sleep"):
        replay_trajectory(
            sim,
            robot,
            trajectory,
            Namespace(auto_play=False),
            video_prefix="unused",
            hold_steps=0,
        )

    assert sim.update.call_args_list == [call(step=2), call(step=3), call(step=3)]
    assert robot.set_qpos.call_count == 3


def test_broadcast_pose_batch_rejects_wrong_env_count() -> None:
    poses = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)

    with pytest.raises(ValueError, match="num_envs"):
        broadcast_pose_batch(poses, num_envs=3)


def test_obstacle_animation_interpolates_and_reaches_target_pose() -> None:
    obstacle, adapter = _run_obstacle_animation(pace_wall_time=False)

    poses = [entry.args[0] for entry in obstacle.set_local_pose.call_args_list]
    expected_y = torch.tensor([-0.14, -0.08, -0.02])
    actual_y = torch.stack([pose[0, 1, 3] for pose in poses])
    assert actual_y.tolist() == pytest.approx(expected_y.tolist())
    assert adapter.sleep.call_args_list == [call(PHYSICS_DT)] * EXPECTED_STEP_COUNT


def test_obstacle_animation_paces_live_viewer_in_wall_time() -> None:
    with patch(
        "scripts.tutorials.atomic_action.dynamic_obstacle_recovery.time.sleep"
    ) as sleep:
        _run_obstacle_animation(pace_wall_time=True)

    assert sleep.call_args_list == [call(PHYSICS_DT)] * EXPECTED_STEP_COUNT


def test_blocking_pose_targets_the_selected_initial_path_waypoint() -> None:
    start_pose = torch.eye(4).unsqueeze(0)
    path = torch.tensor([[[0.4, -0.1, 0.3], [0.5, 0.0, 0.4], [0.6, 0.1, 0.5]]])

    target_pose, waypoint_index = _blocking_obstacle_pose(
        start_pose,
        path,
        path_fraction=0.5,
    )

    assert waypoint_index == 1
    assert torch.equal(target_pose[:, :3, 3], path[:, waypoint_index])
    assert torch.equal(start_pose, torch.eye(4).unsqueeze(0))


def test_maximum_path_deviation_measures_detour_from_reference_polyline() -> None:
    reference_path = torch.tensor([[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    detour_path = torch.tensor([[[0.0, 0.0, 0.0], [0.5, 0.2, 0.0], [1.0, 0.0, 0.0]]])

    deviation = _maximum_path_deviation(detour_path, reference_path)

    assert deviation.tolist() == pytest.approx([0.2])


def test_minimum_cuboid_clearance_is_positive_outside_cuboid() -> None:
    path = torch.tensor([[[0.25, 0.0, 0.0], [0.15, 0.0, 0.0]]])
    cuboid_pose = torch.eye(4).unsqueeze(0)

    clearance = _minimum_cuboid_clearance(
        path,
        cuboid_pose,
        size=CUBOID_SIZE,
    )

    assert clearance.tolist() == pytest.approx([0.05])


def test_minimum_cuboid_clearance_is_negative_inside_cuboid() -> None:
    path = torch.tensor([[[0.05, 0.0, 0.0]]])
    cuboid_pose = torch.eye(4).unsqueeze(0)

    clearance = _minimum_cuboid_clearance(
        path,
        cuboid_pose,
        size=CUBOID_SIZE,
    )

    assert clearance.tolist() == pytest.approx([-0.05])


def test_minimum_cuboid_clearance_uses_cuboid_orientation() -> None:
    path = torch.tensor([[[0.25, 0.0, 0.0]]])
    cuboid_pose = torch.eye(4).unsqueeze(0)
    cuboid_pose[0, :3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )

    clearance = _minimum_cuboid_clearance(
        path,
        cuboid_pose,
        size=(0.2, 0.4, 0.2),
    )

    assert clearance.tolist() == pytest.approx([0.05])
