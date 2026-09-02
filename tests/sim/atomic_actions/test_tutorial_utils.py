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
import inspect
import math
import re
import xml.etree.ElementTree as ET
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import torch

from embodichain.lab.sim.atomic_actions import TimedTrajectory
from embodichain.lab.sim.solvers import BaseSolver
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
    ROBOTIQ_2F_140_TCP,
    ROBOTIQ_HAND_JOINT_PATTERN,
    TUTORIAL_ROBOTS,
    broadcast_pose_batch,
    broadcast_waypoint_pose_batch,
    clone_local_pose_from_first_env,
    create_antipodal_semantics,
    create_curobo_motion_generator,
    create_franka_panda_robot_cfg,
    create_tutorial_argument_parser,
    create_tutorial_robot_cfg,
    create_ur10_robotiq_robot_cfg,
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
STRICT_RECOVERY_TRACKING_ERROR = 0.1
STRICT_RECOVERY_SPHERE_DENSITY = 0.3
STRICT_RECOVERY_MINIMUM_CLEARANCE = 0.01
FRANKA_TUTORIAL_BASE_ROTATION = (0.0, 0.0, 180.0)
DUAL_FRANKA_MOUNT_X_AXIS = torch.tensor([0.0, -1.0, 0.0])
UR_RUNTIME_QPOS_LIMITS = torch.tensor([[-2.0 * math.pi, 2.0 * math.pi]] * 6)
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
    "axis_align",
    "control_dt",
    "coordinated_pickment",
    "coordinated_placement",
    "dynamic_obstacle_recovery",
    "hand_over",
    "move_end_effector",
    "move_held_object",
    "move_joints",
    "moving_target_recovery",
    "open_door",
    "pickup",
    "place",
    "pour",
    "press",
    "slide",
    "twist",
)
RIGID_SCENE_TUTORIAL_MODULES = (
    "assemble",
    "axis_align",
    "coordinated_placement",
    "hand_over",
    "move_held_object",
    "pickup",
    "place",
    "pour",
)
STATIC_POSE_TUTORIAL_MODULES = (
    "coordinated_pickment",
    "press",
    "slide",
    "twist",
)
EXPLICIT_SCENE_LIFECYCLE_TUTORIAL_MODULES = (
    "dynamic_obstacle_recovery",
    "moving_target_recovery",
    "open_door",
)
SCENE_FREE_TUTORIAL_MODULES = (
    "control_dt",
    "move_end_effector",
    "move_joints",
)
TUTORIAL_PRISMATIC_JOINT_AXIS = torch.tensor([2.0, 2.0, 1.0])
TUTORIAL_REVOLUTE_JOINT_AXIS = torch.tensor([-1.0, 2.0, 2.0])
TUTORIAL_REVOLUTE_AXIS_ORIGIN = torch.tensor([0.75, -0.5, 0.25])


def _tutorial_axis_geometry(
    neighbor_offset: tuple[float, float, float],
) -> dict[str, torch.Tensor]:
    """Build non-origin target-local geometry for tutorial semantics tests."""
    center = torch.tensor([2.0, -3.0, 4.0])
    target_points = center + torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    return {
        "target_link_point_cloud": target_points,
        "articulation_point_cloud": torch.cat(
            (
                target_points,
                (center + torch.tensor(neighbor_offset)).unsqueeze(0),
            )
        ),
        "target_link_prismatic_joint_axis": TUTORIAL_PRISMATIC_JOINT_AXIS.clone(),
        "target_link_revolute_joint_axis": TUTORIAL_REVOLUTE_JOINT_AXIS.clone(),
        "target_link_revolute_axis_origin": TUTORIAL_REVOLUTE_AXIS_ORIGIN.clone(),
    }


class _TutorialArticulation:
    """Minimal articulation surface used by automatic-axis tutorial tests."""

    def __init__(self, geometry: dict[str, torch.Tensor]) -> None:
        self.device = torch.device("cpu")
        self.geometry = geometry
        self.sampled_link_name: str | None = None

    def get_link_vert_face(self, link_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        del link_name
        return (
            self.geometry["target_link_point_cloud"],
            torch.tensor(
                [
                    [0, 2, 4],
                    [2, 1, 4],
                    [1, 3, 4],
                    [3, 0, 4],
                    [2, 0, 5],
                    [1, 2, 5],
                    [3, 1, 5],
                    [0, 3, 5],
                ]
            ),
        )

    def get_link_pose(self, link_name: str, *, to_matrix: bool) -> torch.Tensor:
        del link_name
        assert to_matrix is True
        return torch.eye(4).unsqueeze(0)

    def sample_initial_point_clouds(
        self,
        target_link_name: str,
    ) -> dict[str, torch.Tensor]:
        self.sampled_link_name = target_link_name
        return self.geometry


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


@pytest.mark.parametrize(
    (
        "module_name",
        "factory_name",
        "link_name",
        "axis_field",
        "neighbor_offset",
        "expected_axis",
    ),
    (
        (
            "slide",
            "create_drawer_semantics",
            "large_handle_bar",
            "translation_axis",
            (1.0, 1.0, 0.5),
            (2.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0),
        ),
        (
            "press",
            "create_button_semantics",
            "button_cap",
            "press_axis",
            (-1.0, -1.0, -0.5),
            (-2.0 / 3.0, -2.0 / 3.0, -1.0 / 3.0),
        ),
        (
            "twist",
            "create_knob_semantics",
            "cap_1",
            "twist_axis",
            (0.5, -1.0, -1.0),
            (1.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0),
        ),
    ),
)
def test_articulation_tutorial_semantics_resolve_signed_parent_joint_axis(
    module_name: str,
    factory_name: str,
    link_name: str,
    axis_field: str,
    neighbor_offset: tuple[float, float, float],
    expected_axis: tuple[float, float, float],
) -> None:
    module = importlib.import_module(f"scripts.tutorials.atomic_action.{module_name}")
    geometry = _tutorial_axis_geometry(neighbor_offset)
    articulation = _TutorialArticulation(geometry)

    with patch.object(module, "Articulation", _TutorialArticulation):
        result = getattr(module, factory_name)(articulation)

    semantics = result[0] if isinstance(result, tuple) else result
    assert articulation.sampled_link_name == link_name
    assert semantics.geometry is geometry
    assert torch.allclose(
        getattr(semantics.affordance, axis_field),
        torch.tensor(expected_axis),
        atol=1.0e-6,
    )
    if module_name == "press":
        assert semantics.affordance.press_position == pytest.approx((2.5, -2.5, 4.0))
    elif module_name == "twist":
        assert semantics.affordance.axis_origin == pytest.approx(
            tuple(float(value) for value in TUTORIAL_REVOLUTE_AXIS_ORIGIN)
        )


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
    obj.uid = "cube"
    obj.get_vertices.return_value = vertices.unsqueeze(0)
    obj.get_triangles.return_value = triangles.unsqueeze(0)

    semantics = create_antipodal_semantics(obj, label="cube")

    assert semantics.entity_id == "cube"
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


def test_ur10_robotiq_config_matches_six_active_hand_joints_and_tcp() -> None:
    cfg = create_ur10_robotiq_robot_cfg()
    hand_urdf = cfg.urdf_cfg.components["hand"]["urdf_path"]
    active_hand_joints = [
        joint.attrib["name"]
        for joint in ET.parse(hand_urdf).getroot().findall("joint")
        if joint.attrib.get("type") != "fixed"
    ]

    assert cfg.control_parts["hand"] == [ROBOTIQ_HAND_JOINT_PATTERN]
    assert len(active_hand_joints) == 6
    assert all(
        re.fullmatch(ROBOTIQ_HAND_JOINT_PATTERN, joint_name)
        for joint_name in active_hand_joints
    )
    assert len(cfg.init_qpos) == 12
    assert cfg.init_qpos[-6:] == [0.0] * 6
    assert torch.allclose(
        torch.as_tensor(cfg.solver_cfg["arm"].tcp),
        torch.as_tensor(ROBOTIQ_2F_140_TCP),
    )


@pytest.mark.parametrize(
    "factory",
    (create_ur5_gripper_robot_cfg, create_ur10_robotiq_robot_cfg),
)
def test_ur_tutorial_solvers_use_runtime_joint_limits(factory) -> None:
    cfg = factory()

    assert torch.allclose(
        torch.as_tensor(cfg.solver_cfg["arm"].user_qpos_limits),
        UR_RUNTIME_QPOS_LIMITS,
    )


@pytest.mark.parametrize(
    "factory",
    (create_ur5_gripper_robot_cfg, create_ur10_robotiq_robot_cfg),
)
def test_ur_tutorial_solver_limits_skip_noop_hard_limit_warning(factory) -> None:
    cfg = factory()
    solver_limits = torch.as_tensor(cfg.solver_cfg["arm"].user_qpos_limits)
    solver = SimpleNamespace(
        lower_qpos_limits=solver_limits[:, 0].clone(),
        upper_qpos_limits=solver_limits[:, 1].clone(),
    )

    with patch("embodichain.lab.sim.solvers.base_solver.logger.log_warning") as warning:
        BaseSolver.update_with_robot_limit(solver, UR_RUNTIME_QPOS_LIMITS)

    assert not warning.called


@pytest.mark.parametrize(
    ("robot_type", "arm_dof", "solver_name", "hand_pattern", "expected_tcp"),
    (
        ("ur5", 6, "URSolverCfg", "gripper_finger1_joint_1", None),
        ("franka", 7, "PytorchSolverCfg", "gripper_finger1_joint_1", None),
        (
            "ur10",
            6,
            "URSolverCfg",
            ROBOTIQ_HAND_JOINT_PATTERN,
            ROBOTIQ_2F_140_TCP,
        ),
    ),
)
def test_dual_tutorial_configs_share_hand_binding_contract(
    robot_type: str,
    arm_dof: int,
    solver_name: str,
    hand_pattern: str,
    expected_tcp: tuple[tuple[float, ...], ...] | None,
) -> None:
    single_cfg = create_tutorial_robot_cfg(robot_type)
    dual_cfg = create_dual_tutorial_robot_cfg(
        robot_type=robot_type,
        uid=f"test_{robot_type}",
        urdf_name=f"test_dual_{robot_type}",
        tcp_z=0.121,
    )
    if expected_tcp is None:
        expected_tcp_tensor = PGI_TUTORIAL_TCP.clone()
        expected_tcp_tensor[2, 3] = 0.121
    else:
        expected_tcp_tensor = torch.as_tensor(expected_tcp)

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
        assert dual_cfg.control_parts[f"{side}_hand"] == [f"{side}_{hand_pattern}"]
        assert dual_cfg.urdf_cfg.components[f"{side}_hand"]["urdf_path"] == (
            single_cfg.urdf_cfg.components["hand"]["urdf_path"]
        )
        solver = dual_cfg.solver_cfg[f"{side}_arm"]
        assert type(solver).__name__ == solver_name
        assert torch.allclose(torch.as_tensor(solver.tcp), expected_tcp_tensor)
        assert solver.user_qpos_limits == single_cfg.solver_cfg["arm"].user_qpos_limits

    if robot_type == "ur10":
        assert len(dual_cfg.init_qpos) == 24
        assert dual_cfg.init_qpos[-12:] == [0.0] * 12


@pytest.mark.parametrize("robot_type", ("ur5", "ur10"))
def test_dual_pytorch_ur_tutorial_solvers_preserve_runtime_joint_limits(
    robot_type: str,
) -> None:
    cfg = create_dual_tutorial_robot_cfg(
        robot_type=robot_type,
        uid=f"test_{robot_type}_pytorch",
        urdf_name=f"test_dual_{robot_type}_pytorch",
        tcp_z=0.121,
        solver="pytorch",
    )

    for side in ("left", "right"):
        assert torch.allclose(
            torch.as_tensor(cfg.solver_cfg[f"{side}_arm"].user_qpos_limits),
            UR_RUNTIME_QPOS_LIMITS,
        )


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


def test_hand_commands_cover_all_six_robotiq_joints_with_mimic_directions() -> None:
    robot = MagicMock()
    robot.device = torch.device("cpu")
    robot.cfg.control_parts = {
        "left_hand": [
            "left_finger_joint",
            "left_left_inner_knuckle_joint",
            "left_left_inner_finger_joint",
            "left_right_outer_knuckle_joint",
            "left_right_inner_knuckle_joint",
            "left_right_inner_finger_joint",
        ]
    }
    robot.get_qpos_limits.return_value = torch.tensor(
        [
            [
                [0.0, 0.7],
                [-0.8757, 0.8757],
                [-0.8757, 0.8757],
                [-0.725, 0.725],
                [-0.8757, 0.8757],
                [-0.8757, 0.8757],
            ]
        ]
    )

    hand_open, hand_close = get_hand_open_close_qpos(
        robot,
        hand_control_part="left_hand",
        close_qpos=0.4,
    )

    assert torch.allclose(hand_open, torch.zeros(6))
    assert torch.allclose(
        hand_close,
        torch.tensor([0.4, -0.4, 0.4, -0.4, -0.4, 0.4]),
    )


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


def test_shared_robot_selection_keeps_ur5_default_and_accepts_all_variants() -> None:
    parser = create_tutorial_argument_parser("test parser")
    default_args = parser.parse_args([])
    franka_args = parser.parse_args(["--robot", "franka"])
    ur10_args = parser.parse_args(["--robot", "ur10"])

    assert TUTORIAL_ROBOTS == ("ur5", "franka", "ur10")
    assert default_args.robot == "ur5"
    assert franka_args.robot == "franka"
    assert ur10_args.robot == "ur10"


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


def test_place_tutorial_registers_pick_object_with_simulation_engine_factory() -> None:
    module = importlib.import_module("scripts.tutorials.atomic_action.place")
    args = Namespace(
        auto_play=True,
        force_reannotate=False,
        n_sample=1,
        no_vis_eef_axis=True,
        robot="ur5",
    )
    sim = MagicMock()
    sim.device = torch.device("cpu")
    sim.sim_config.physics_dt = PHYSICS_DT
    robot = MagicMock()
    robot.get_qpos.return_value = torch.zeros(1, 8)
    obj = MagicMock()
    obj.uid = "cube"
    obj.get_vertices.return_value = [torch.zeros(1, 3)]
    obj.get_triangles.return_value = [torch.zeros(1, 3, dtype=torch.long)]
    engine = MagicMock()
    engine.compile.return_value = SimpleNamespace(plan_success=torch.tensor([False]))

    with (
        patch.object(module, "parse_arguments", return_value=args),
        patch.object(module, "create_tutorial_simulation", return_value=sim),
        patch.object(module, "add_tutorial_robot", return_value=robot),
        patch.object(module, "create_pick_object", return_value=obj),
        patch.object(module, "create_curobo_motion_generator"),
        patch.object(
            module,
            "get_hand_open_close_qpos",
            return_value=(
                torch.zeros(1),
                torch.ones(1),
            ),
        ),
        patch.object(module, "initialize_pre_pick_robot_pose"),
        patch.object(
            module,
            "create_simulation_atomic_action_engine",
            return_value=engine,
        ) as engine_factory,
        patch.object(module, "create_parallel_jaw_grasp_pose_generator"),
        patch.object(module, "prepare_tutorial_scene", return_value=False),
    ):
        module.main()

    assert engine_factory.call_args.kwargs["scene_entities"] == (obj,)
    engine.initial_context.assert_called_once_with(control_dt=PHYSICS_DT)


def test_atomic_action_tutorial_scene_strategies_cover_every_entry_point() -> None:
    classified = (
        set(RIGID_SCENE_TUTORIAL_MODULES)
        | set(STATIC_POSE_TUTORIAL_MODULES)
        | set(EXPLICIT_SCENE_LIFECYCLE_TUTORIAL_MODULES)
        | set(SCENE_FREE_TUTORIAL_MODULES)
    )

    assert classified == set(ATOMIC_ACTION_TUTORIAL_MODULES)


@pytest.mark.parametrize("module_name", RIGID_SCENE_TUTORIAL_MODULES)
def test_rigid_scene_tutorials_use_simulation_engine_factory(
    module_name: str,
) -> None:
    module = importlib.import_module(f"scripts.tutorials.atomic_action.{module_name}")
    source = inspect.getsource(module)

    assert "create_simulation_atomic_action_engine(" in source
    assert "RigidObjectSceneProvider" not in source
    assert "SceneSnapshot" not in source


@pytest.mark.parametrize("module_name", STATIC_POSE_TUTORIAL_MODULES)
def test_static_pose_tutorials_do_not_construct_scene_snapshots(
    module_name: str,
) -> None:
    module = importlib.import_module(f"scripts.tutorials.atomic_action.{module_name}")
    source = inspect.getsource(module)

    assert "SceneSnapshot" not in source
    assert "SceneEntityPose" not in source


def test_axis_align_tutorial_exposes_upright_and_horizontal_modes() -> None:
    module = importlib.import_module("scripts.tutorials.atomic_action.axis_align")

    with patch("sys.argv", ["axis_align.py"]):
        upright_args = module.parse_arguments()
    with patch(
        "sys.argv",
        ["axis_align.py", "--alignment", "horizontal_align"],
    ):
        horizontal_args = module.parse_arguments()

    assert upright_args.alignment == "upright"
    assert horizontal_args.alignment == "horizontal_align"
    assert module.ALIGNMENT_AXES["upright"] == (
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    assert module.ALIGNMENT_AXES["horizontal_align"] == (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )


def test_pour_tutorial_uses_configured_pickup_and_local_rotation_axis() -> None:
    module = importlib.import_module("scripts.tutorials.atomic_action.pour")

    with patch("sys.argv", ["pour.py"]):
        default_args = module.parse_arguments()
    with patch("sys.argv", ["pour.py", "--rotate_angle", "-1.25"]):
        configured_args = module.parse_arguments()

    assert default_args.rotate_angle == pytest.approx(math.pi / 4.0)
    assert configured_args.rotate_angle == pytest.approx(-1.25)
    assert module.APPROACH_DIRECTION == pytest.approx((-0.707, 0.0, -0.707))
    assert module.POUR_INTERNAL_AXIS == (1.0, 0.0, 0.0)
    pick_policy = module._create_pick_motion_policy()
    assert pick_policy.sample_count == module.PICK_SAMPLE_INTERVAL
    assert pick_policy.plan_opts.sample_method is module.TrajectorySampleMethod.QUANTITY
    assert pick_policy.plan_opts.sample_interval == module.PICK_MOTION_SAMPLE_COUNT


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


def test_dynamic_obstacle_recovery_keeps_strict_collision_contract() -> None:
    module = importlib.import_module(
        "scripts.tutorials.atomic_action.dynamic_obstacle_recovery"
    )
    main_source = inspect.getsource(module.main)

    assert module.TRACKING_ERROR_THRESHOLD == pytest.approx(
        STRICT_RECOVERY_TRACKING_ERROR
    )
    assert module.COLLISION_SPHERE_FIT_TYPE == "morphit"
    assert module.COLLISION_SPHERE_FIT_DENSITY == pytest.approx(
        STRICT_RECOVERY_SPHERE_DENSITY
    )
    assert module.MINIMUM_REPLAN_CLEARANCE == pytest.approx(
        STRICT_RECOVERY_MINIMUM_CLEARANCE
    )
    assert "blocked_path_clearance > MAXIMUM_BLOCKED_PATH_CLEARANCE" in main_source
    assert "replan_clearance < MINIMUM_REPLAN_CLEARANCE" in main_source


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
