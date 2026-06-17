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

"""
This script demonstrates the creation and simulation of a robot that uprights a fallen
bottle in a simulated environment using the SimulationManager and atomic actions.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import Robot, RigidObject
from embodichain.lab.sim.shapes import CubeCfg, MeshCfg
from embodichain.lab.sim.solvers import PytorchSolverCfg
from embodichain.lab.sim.planners import MotionGenerator, MotionGenCfg, ToppraPlannerCfg
from embodichain.data import get_data_path
from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.utils import logger
from embodichain.lab.sim.cfg import (
    RenderCfg,
    JointDrivePropertiesCfg,
    RobotCfg,
    LightCfg,
    RigidBodyAttributesCfg,
    RigidObjectCfg,
    URDFCfg,
)
from embodichain.lab.sim.atomic_actions import (
    AntipodalAffordance,
    ObjectSemantics,
    UprightAction,
    UprightActionCfg,
)
from embodichain.toolkits.graspkit.pg_grasp.antipodal_generator import (
    GraspGenerator,
    GraspGeneratorCfg,
    AntipodalSamplerCfg,
)
from embodichain.toolkits.graspkit.pg_grasp.gripper_collision_checker import (
    GripperCollisionCfg,
)

GRIPPER_URDF_PATH = "DH_PGI_140_80/DH_PGI_140_80.urdf"
GRIPPER_HAND_JOINT_PATTERN = "GRIPPER_FINGER1_JOINT_1"
GRIPPER_MAX_OPEN_WIDTH = 0.080
GRIPPER_FINGER_LENGTH = 0.088
GRIPPER_ROOT_Z_WIDTH = 0.096
GRIPPER_Y_THICKNESS = 0.040
GRIPPER_TCP_Z = 0.15
PGI_SAMPLE_INTERVAL = 120
PGI_HAND_CLOSE_STEPS = 12
PGI_GRASP_HOLD_STEPS = 20
BOTTLE_LABEL = "bottle"
BOTTLE_APPROACH_DIRECTION = (0.0, 0.0, -1.0)
BOTTLE_GRASP_SQUEEZE_WIDTH = 0.020
BOTTLE_MAX_GRASP_OPEN_LENGTH = 0.060
BOTTLE_MAX_GRASP_AXIS_APPROACH_DOT = 0.080
BOTTLE_MAX_GRASP_AXIS_UPRIGHT_AXIS_DOT = 0.35
BOTTLE_MIN_DYNAMIC_HAND_CLOSE_QPOS = 0.024
BOTTLE_GRASP_COLLISION_THRESHOLD = -0.004


def parse_arguments():
    """
    Parse command-line arguments to configure the simulation.

    Returns:
        argparse.Namespace: Parsed arguments including simulation and rendering
        options.
    """
    parser = argparse.ArgumentParser(
        description="Create and simulate a robot in SimulationManager"
    )
    add_env_launcher_args_to_parser(parser)
    parser.add_argument(
        "--n_sample",
        type=int,
        default=10000,
        help="Number of surface samples for antipodal grasp generation.",
    )
    parser.add_argument(
        "--force_reannotate",
        action="store_true",
        help=(
            "Force grasp region re-annotation instead of reusing cached antipodal "
            "pairs."
        ),
    )
    parser.add_argument(
        "--debug_hand_state",
        action="store_true",
        help="Log planned hand targets and simulated hand qpos during execution.",
    )
    parser.add_argument(
        "--diagnose_grasp",
        action="store_true",
        help="Plan once and print grasp/TCP diagnostics without opening the viewer.",
    )
    parser.add_argument(
        "--auto_play",
        action="store_true",
        help="Run the viewer demo without waiting for keyboard input.",
    )
    return parser.parse_args()


def initialize_simulation(args) -> SimulationManager:
    """
    Initialize the simulation environment based on the provided arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        SimulationManager: Configured simulation manager instance.
    """
    config = SimulationManagerCfg(
        headless=True,
        sim_device=args.device,
        render_cfg=RenderCfg(renderer=args.renderer),
        physics_dt=1.0 / 100.0,
        arena_space=2.5,
    )
    sim = SimulationManager(config)

    sim.add_light(
        cfg=LightCfg(
            uid="main_light",
            color=(0.6, 0.6, 0.6),
            intensity=30.0,
            init_pos=(1.0, 0, 3.0),
        )
    )

    return sim


def create_robot(sim: SimulationManager, position=[0.0, 0.0, 0.0]) -> Robot:
    """
    Create and configure a robot with an arm and a dexterous hand in the simulation.

    Args:
        sim (SimulationManager): The simulation manager instance.

    Returns:
        Robot: The configured robot instance added to the simulation.
    """
    # Retrieve URDF paths for the robot arm and hand
    ur10_urdf_path = get_data_path("UniversalRobots/UR10/UR10.urdf")
    gripper_urdf_path = get_data_path(GRIPPER_URDF_PATH)
    # Configure the robot with its components and control properties
    cfg = RobotCfg(
        uid="UR10",
        urdf_cfg=URDFCfg(
            components=[
                {"component_type": "arm", "urdf_path": ur10_urdf_path},
                {"component_type": "hand", "urdf_path": gripper_urdf_path},
            ]
        ),
        drive_pros=JointDrivePropertiesCfg(
            stiffness={"JOINT[0-9]": 1e4, GRIPPER_HAND_JOINT_PATTERN: 1e3},
            damping={"JOINT[0-9]": 1e3, GRIPPER_HAND_JOINT_PATTERN: 1e2},
            max_effort={"JOINT[0-9]": 1e5, GRIPPER_HAND_JOINT_PATTERN: 1e4},
            drive_type="force",
        ),
        control_parts={
            "arm": ["JOINT[0-9]"],
            "hand": [GRIPPER_HAND_JOINT_PATTERN],
        },
        solver_cfg={
            "arm": PytorchSolverCfg(
                end_link_name="ee_link",
                root_link_name="base_link",
                tcp=[
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, GRIPPER_TCP_Z],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            )
        },
        init_qpos=[0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.0, 0.0],
        init_pos=position,
    )
    return sim.add_robot(cfg=cfg)


def create_fallen_bottle(sim: SimulationManager) -> RigidObject:
    # Use a slightly smaller and closer bottle for the UR10 gripper demo.
    bottle_scale = 0.0008
    bottle_cfg = RigidObjectCfg(
        uid="bottle",
        shape=MeshCfg(
            fpath=get_data_path("ScannedBottle/yibao.ply"),
        ),
        attrs=RigidBodyAttributesCfg(
            mass=0.02,
            dynamic_friction=0.97,
            static_friction=0.99,
        ),
        max_convex_hull_num=16,
        init_pos=[-0.4294, -0.0825, -0.0997],
        init_rot=[90.0, 135.0, 0.0],
        body_scale=(bottle_scale, bottle_scale, bottle_scale),
    )
    return sim.add_rigid_object(cfg=bottle_cfg)


def settle_object(sim: SimulationManager, obj: RigidObject, step: int = 5) -> None:
    """Settle an object through the same explicit sequence on CPU and CUDA."""
    if sim.device.type == "cuda":
        sim.init_gpu_physics()

    obj.reset()
    sim.update(step=step)
    obj.clear_dynamics()


def release_object_after_grasp(obj: RigidObject) -> None:
    """Clear residual motion after the gripper has closed on the object."""
    obj.clear_dynamics()


def build_grasp_generator_cfg(args: argparse.Namespace) -> GraspGeneratorCfg:
    return GraspGeneratorCfg(
        viser_port=11801,
        antipodal_sampler_cfg=AntipodalSamplerCfg(
            n_sample=args.n_sample,
            max_length=GRIPPER_MAX_OPEN_WIDTH,
            min_length=0.003,
        ),
        is_partial_annotate=False,
        is_filter_ground_collision=False,
    )


def build_gripper_collision_cfg() -> GripperCollisionCfg:
    return GripperCollisionCfg(
        max_open_length=GRIPPER_MAX_OPEN_WIDTH,
        finger_length=GRIPPER_FINGER_LENGTH,
        y_thickness=GRIPPER_Y_THICKNESS,
        root_z_width=GRIPPER_ROOT_Z_WIDTH,
        open_check_margin=0.002,
        point_sample_dense=0.012,
    )


def get_hand_open_close_qpos(
    robot: Robot, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    hand_limits = robot.get_qpos_limits(name="hand")[0].to(
        device=device, dtype=torch.float32
    )
    return hand_limits[:, 0], hand_limits[:, 1]


def format_tensor(tensor: torch.Tensor) -> str:
    rounded = (tensor.detach().cpu() * 10000.0).round() / 10000.0
    return str(rounded.tolist())


def log_hand_setup(robot: Robot, hand_open: torch.Tensor, hand_close: torch.Tensor):
    hand_joint_ids = robot.get_joint_ids(name="hand")
    hand_joint_names = [robot.joint_names[joint_id] for joint_id in hand_joint_ids]
    logger.log_info(f"Hand joint ids: {hand_joint_ids}")
    logger.log_info(f"Hand joint names: {hand_joint_names}")
    logger.log_info(f"Robot mimic ids: {robot.mimic_ids}")
    logger.log_info(f"Robot mimic parents: {robot.mimic_parents}")
    logger.log_info(f"Robot mimic multipliers: {robot.mimic_multipliers}")
    logger.log_info(f"Robot mimic offsets: {robot.mimic_offsets}")
    logger.log_info(
        f"Hand qpos limits: {format_tensor(robot.get_qpos_limits(name='hand')[0])}"
    )
    logger.log_info(f"Hand open qpos: {format_tensor(hand_open)}")
    logger.log_info(f"Hand close qpos: {format_tensor(hand_close)}")


def log_hand_execution_state(
    robot: Robot,
    step_idx: int,
    total_steps: int,
    action_hand_target: torch.Tensor,
) -> None:
    sim_hand_target = robot.get_qpos(name="hand", target=True)
    actual_hand_qpos = robot.get_qpos(name="hand")
    tracking_error = actual_hand_qpos - sim_hand_target
    logger.log_info(
        "Hand state "
        f"step={step_idx}/{total_steps - 1}, "
        f"action_target={format_tensor(action_hand_target[0])}, "
        f"sim_target={format_tensor(sim_hand_target[0])}, "
        f"actual={format_tensor(actual_hand_qpos[0])}, "
        f"actual_minus_target={format_tensor(tracking_error[0])}"
    )


def log_object_execution_state(
    obj: RigidObject,
    step_idx: int,
    total_steps: int,
) -> None:
    obj_pose = obj.get_local_pose(to_matrix=True)
    logger.log_info(
        "Object state "
        f"step={step_idx}/{total_steps - 1}, "
        f"pos={format_tensor(obj_pose[0, :3, 3])}, "
        f"z_axis={format_tensor(obj_pose[0, :3, 2])}"
    )


def get_upright_segment_lengths(action: UprightAction) -> dict[str, int]:
    n_close = action.cfg.hand_interp_steps
    n_final = max(2, action.cfg.final_approach_steps)
    n_hold = max(0, action.cfg.grasp_hold_steps)
    n_press = max(2, action.cfg.place_press_steps)
    n_upright_hold = max(0, action.cfg.upright_hold_steps)
    n_place_hold = max(0, action.cfg.place_hold_steps)
    n_open = max(2, action.cfg.release_interp_steps)
    motion_waypoints = (
        action.cfg.sample_interval
        - n_close
        - n_final
        - n_hold
        - n_upright_hold
        - n_press
        - n_place_hold
        - n_open
    )
    n_pre = max(2, int(round(motion_waypoints * 0.25)))
    n_upright = max(2, int(round(motion_waypoints * 0.60)))
    n_retreat = (
        action.cfg.sample_interval
        - n_close
        - n_final
        - n_hold
        - n_upright_hold
        - n_press
        - n_place_hold
        - n_open
        - n_pre
        - n_upright
    )
    if n_retreat < 2:
        retreat_deficit = 2 - n_retreat
        n_retreat = 2
        n_upright = max(2, n_upright - retreat_deficit)
    return {
        "pre": n_pre,
        "final": n_final,
        "close": n_close,
        "grasp_hold": n_hold,
        "upright": n_upright,
        "upright_hold": n_upright_hold,
        "press": n_press,
        "place_hold": n_place_hold,
        "open": n_open,
        "retreat": n_retreat,
    }


def log_tcp_alignment(
    robot: Robot,
    traj: torch.Tensor,
    grasp_xpos: torch.Tensor,
    arm_dof: int,
    index: int,
    label: str,
) -> None:
    arm_qpos = traj[:, index, :arm_dof]
    tcp_xpos = robot.compute_fk(qpos=arm_qpos, name="arm", to_matrix=True)
    pos_error = torch.norm(tcp_xpos[:, :3, 3] - grasp_xpos[:, :3, 3], dim=1)
    rot_delta = torch.bmm(tcp_xpos[:, :3, :3].transpose(1, 2), grasp_xpos[:, :3, :3])
    trace = rot_delta[:, 0, 0] + rot_delta[:, 1, 1] + rot_delta[:, 2, 2]
    rot_error = torch.acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0))
    logger.log_info(
        f"{label}: index={index}, "
        f"tcp_pos={format_tensor(tcp_xpos[0, :3, 3])}, "
        f"pos_error={format_tensor(pos_error)}, "
        f"rot_error_rad={format_tensor(rot_error)}, "
        f"tcp_rot={format_tensor(tcp_xpos[0, :3, :3])}, "
        f"target_rot={format_tensor(grasp_xpos[0, :3, :3])}, "
        f"hand_target={format_tensor(traj[0, index, arm_dof:])}"
    )


def log_selected_gripper_clearance(
    semantics: ObjectSemantics,
    obj_pose: torch.Tensor,
    grasp_xpos: torch.Tensor,
    grasp_open_length: torch.Tensor,
) -> None:
    generator = semantics.affordance.generator
    if generator is None:
        return

    collision_checker = getattr(generator, "_collision_checker", None)
    if collision_checker is None:
        return

    gripper_pc_world = collision_checker._get_gripper_pc(grasp_xpos, grasp_open_length)
    ground_height = collision_checker.get_ground_height(obj_pose[0])
    min_z = gripper_pc_world[:, :, 2].min(dim=1).values
    max_z = gripper_pc_world[:, :, 2].max(dim=1).values
    is_colliding, min_distance = collision_checker.query(
        obj_pose=obj_pose[0],
        grasp_poses=grasp_xpos,
        open_lengths=grasp_open_length,
        is_filter_ground_collision=True,
        collision_threshold=BOTTLE_GRASP_COLLISION_THRESHOLD,
    )
    logger.log_info(f"Selected gripper pc min z: {format_tensor(min_z)}")
    logger.log_info(f"Selected gripper pc max z: {format_tensor(max_z)}")
    logger.log_info(f"Selected grasp ground height: {ground_height:.4f}")
    logger.log_info(
        f"Selected grasp min collision distance: {format_tensor(min_distance)}"
    )
    logger.log_info(
        f"Selected grasp collision threshold: {BOTTLE_GRASP_COLLISION_THRESHOLD:.4f}"
    )
    logger.log_info(
        f"Selected grasp collision flag: {is_colliding.detach().cpu().tolist()}"
    )


def log_grasp_direction_probe(semantics: ObjectSemantics) -> None:
    generator = semantics.affordance.generator
    if generator is None:
        return

    obj_pose = semantics.entity.get_local_pose(to_matrix=True)[0]
    hit_point_pairs = getattr(generator, "_hit_point_pairs", None)
    if hit_point_pairs is None or hit_point_pairs.numel() == 0:
        logger.log_info("Probe grasp direction: no antipodal pairs available")
        return

    origin_points = hit_point_pairs[:, 0, :]
    hit_points = hit_point_pairs[:, 1, :]
    origin_points_world = generator._apply_transform(origin_points, obj_pose)
    hit_points_world = generator._apply_transform(hit_points, obj_pose)
    centers = (origin_points_world + hit_points_world) * 0.5
    grasp_x = torch.nn.functional.normalize(
        hit_points_world - origin_points_world, dim=-1
    )
    open_lengths = torch.norm(origin_points_world - hit_points_world, dim=-1)

    probe_directions = {
        "top_down": [0.0, 0.0, -1.0],
        "from_robot_x": [-1.0, 0.0, 0.0],
    }
    for label, direction in probe_directions.items():
        approach_direction = torch.tensor(
            direction, dtype=torch.float32, device=generator.device
        )
        cos_angle = torch.clamp((grasp_x * approach_direction).sum(dim=-1), -1.0, 1.0)
        positive_angle = torch.abs(torch.acos(cos_angle))
        angle_mask = (
            positive_angle - torch.pi / 2
        ).abs() <= generator.cfg.max_deviation_angle
        width_mask = open_lengths <= GRIPPER_MAX_OPEN_WIDTH
        candidate_mask = angle_mask & width_mask
        logger.log_info(
            f"Probe grasp direction {label}: "
            f"angle_count={int(angle_mask.sum().item())}, "
            f"width_angle_count={int(candidate_mask.sum().item())}"
        )
        if torch.any(candidate_mask):
            candidate_grasp_poses = GraspGenerator._grasp_pose_from_approach_direction(
                grasp_x[candidate_mask],
                approach_direction,
                centers[candidate_mask],
            )
            candidate_open_lengths = open_lengths[candidate_mask]
            is_colliding, min_distance = generator._collision_checker.query(
                obj_pose,
                candidate_grasp_poses,
                candidate_open_lengths,
                is_filter_ground_collision=True,
                collision_threshold=BOTTLE_GRASP_COLLISION_THRESHOLD,
            )
            collision_free_count = int((~is_colliding).sum().item())
            logger.log_info(
                f"Probe grasp direction {label}: "
                f"collision_free_count={collision_free_count}, "
                f"collision_threshold={BOTTLE_GRASP_COLLISION_THRESHOLD:.4f}, "
                f"min_distance={format_tensor(min_distance.min().unsqueeze(0))}, "
                f"max_distance={format_tensor(min_distance.max().unsqueeze(0))}"
            )
            if collision_free_count > 0:
                grasp_xpos = candidate_grasp_poses[~is_colliding][0]
                open_length = candidate_open_lengths[~is_colliding][0]
                gripper_pc_world = generator._collision_checker._get_gripper_pc(
                    grasp_xpos.unsqueeze(0),
                    open_length.unsqueeze(0),
                )
                ground_height = generator._collision_checker.get_ground_height(obj_pose)
                min_z = gripper_pc_world[:, :, 2].min(dim=1).values
                logger.log_info(
                    f"Probe grasp direction {label}: "
                    f"pos={format_tensor(grasp_xpos[:3, 3])}, "
                    f"open_length={open_length.item():.4f}, "
                    f"min_z={format_tensor(min_z)}, "
                    f"ground_height={ground_height:.4f}"
                )


def diagnose_upright_plan(
    robot: Robot,
    action: UprightAction,
    semantics: ObjectSemantics,
) -> None:
    is_success, grasp_xpos, grasp_open_length = action._resolve_grasp_pose(semantics)
    if not torch.all(is_success).item():
        obj_pose = semantics.entity.get_local_pose(to_matrix=True)
        logger.log_info(f"Object pos: {format_tensor(obj_pose[0, :3, 3])}")
        log_grasp_direction_probe(semantics)
        logger.log_warning("Failed to resolve grasp pose during diagnostics.")
        return

    hand_close_qpos = action._compute_dynamic_hand_close_qpos(grasp_open_length)
    final_approach_qpos = action._compute_final_approach_hand_qpos(
        grasp_open_length, hand_close_qpos
    )
    obj_pose = semantics.entity.get_local_pose(to_matrix=True)
    approach_direction = action.approach_direction / action.approach_direction.norm()
    grasp_axis_dot = torch.abs((grasp_xpos[:, :3, 0] * approach_direction).sum(dim=1))
    bottle_axis = torch.nn.functional.normalize(obj_pose[:, :3, 2], dim=1)
    grasp_axis_bottle_dot = torch.abs((grasp_xpos[:, :3, 0] * bottle_axis).sum(dim=1))

    logger.log_info(f"Object pos: {format_tensor(obj_pose[0, :3, 3])}")
    logger.log_info(f"Grasp pos: {format_tensor(grasp_xpos[0, :3, 3])}")
    logger.log_info(f"Grasp rotation columns: {format_tensor(grasp_xpos[0, :3, :3])}")
    logger.log_info(f"Grasp open length: {format_tensor(grasp_open_length)}")
    logger.log_info(f"Grasp axis approach dot: {format_tensor(grasp_axis_dot)}")
    logger.log_info(f"Grasp axis bottle dot: {format_tensor(grasp_axis_bottle_dot)}")
    log_selected_gripper_clearance(semantics, obj_pose, grasp_xpos, grasp_open_length)
    logger.log_info(
        f"Final approach hand qpos: {format_tensor(final_approach_qpos[0])}"
    )
    logger.log_info(f"Close hand qpos: {format_tensor(hand_close_qpos[0])}")

    is_success, traj, joint_ids = action.execute(semantics)
    if not is_success:
        logger.log_warning("Failed to plan upright trajectory during diagnostics.")
        return

    arm_dof = len(robot.get_joint_ids(name="arm"))
    segments = get_upright_segment_lengths(action)
    logger.log_info(f"Action joint ids: {joint_ids}")
    logger.log_info(f"Upright trajectory segments: {segments}")
    logger.log_info(f"Trajectory shape: {tuple(traj.shape)}")

    grasp_idx = segments["pre"] + segments["final"] - 1
    close_end_idx = grasp_idx + segments["close"]
    hold_end_idx = close_end_idx + segments["grasp_hold"]
    log_tcp_alignment(robot, traj, grasp_xpos, arm_dof, grasp_idx, "grasp")
    log_tcp_alignment(robot, traj, grasp_xpos, arm_dof, close_end_idx, "close_end")
    log_tcp_alignment(robot, traj, grasp_xpos, arm_dof, hold_end_idx, "hold_end")


def create_object_semantics(
    obj: RigidObject, args: argparse.Namespace
) -> ObjectSemantics:
    return ObjectSemantics(
        label=BOTTLE_LABEL,
        geometry={
            "mesh_vertices": obj.get_vertices(env_ids=[0], scale=True)[0],
            "mesh_triangles": obj.get_triangles(env_ids=[0])[0],
        },
        affordance=AntipodalAffordance(
            object_label=BOTTLE_LABEL,
            force_reannotate=args.force_reannotate,
            custom_config={
                "gripper_collision_cfg": build_gripper_collision_cfg(),
                "generator_cfg": build_grasp_generator_cfg(args),
            },
        ),
        entity=obj,
    )


def run_upright_demo(
    args: argparse.Namespace, sim: SimulationManager, robot: Robot
) -> None:

    sim.open_window()

    obj = create_fallen_bottle(sim)
    settle_object(sim, obj, step=5)
    semantics = create_object_semantics(obj, args)
    motion_gen = MotionGenerator(
        cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=robot.uid))
    )
    hand_open, hand_close = get_hand_open_close_qpos(robot, sim.device)
    if args.debug_hand_state:
        log_hand_setup(robot, hand_open, hand_close)

    upright_action = UprightAction(
        motion_generator=motion_gen,
        cfg=UprightActionCfg(
            control_part="arm",
            hand_control_part="hand",
            hand_open_qpos=hand_open,
            hand_close_qpos=hand_close,
            approach_direction=torch.tensor(
                BOTTLE_APPROACH_DIRECTION, dtype=torch.float32, device=sim.device
            ),
            pre_grasp_distance=0.15,
            lift_height=0.15,
            sample_interval=PGI_SAMPLE_INTERVAL,
            hand_interp_steps=PGI_HAND_CLOSE_STEPS,
            upright_axis_sign=-1.0,
            place_press_depth=0.0,
            place_press_steps=4,
            upright_hold_steps=3,
            place_hold_steps=8,
            release_interp_steps=12,
            release_retreat_distance=0.08,
            release_retreat_lift=0.01,
            final_approach_steps=12,
            final_approach_preclose_width_margin=0.010,
            grasp_hold_steps=PGI_GRASP_HOLD_STEPS,
            use_grasp_width_qpos=True,
            gripper_max_open_width=GRIPPER_MAX_OPEN_WIDTH,
            max_grasp_open_length=BOTTLE_MAX_GRASP_OPEN_LENGTH,
            max_grasp_axis_approach_dot=BOTTLE_MAX_GRASP_AXIS_APPROACH_DOT,
            max_grasp_axis_upright_axis_dot=BOTTLE_MAX_GRASP_AXIS_UPRIGHT_AXIS_DOT,
            grasp_squeeze_width=BOTTLE_GRASP_SQUEEZE_WIDTH,
            min_dynamic_hand_close_qpos=torch.full_like(
                hand_close, BOTTLE_MIN_DYNAMIC_HAND_CLOSE_QPOS
            ),
        ),
    )

    if args.diagnose_grasp:
        diagnose_upright_plan(robot, upright_action, semantics)
        return

    if not args.auto_play:
        input("Inspect the fallen bottle, then press Enter to plan upright...")

    start_time = time.time()
    is_success, traj, joint_ids = upright_action.execute(semantics)
    cost_time = time.time() - start_time
    logger.log_info(f"Plan upright trajectory cost time: {cost_time:.2f} seconds")
    if not is_success:
        logger.log_warning("Failed to plan upright trajectory.")
        return

    arm_dof = len(robot.get_joint_ids(name="arm"))
    total_steps = traj.shape[1]
    segments = get_upright_segment_lengths(upright_action)
    post_grasp_clear_step = segments["pre"] + segments["final"] + segments["close"]
    should_clear_object_dynamics = True
    if args.debug_hand_state:
        joint_names = [robot.joint_names[joint_id] for joint_id in joint_ids]
        hand_traj = traj[:, :, arm_dof:]
        logger.log_info(f"Action joint ids: {joint_ids}")
        logger.log_info(f"Action joint names: {joint_names}")
        logger.log_info(
            f"Post-grasp object dynamics clear step: {post_grasp_clear_step}"
        )
        logger.log_info(
            f"Planned hand qpos min: {format_tensor(hand_traj.min(dim=1).values[0])}"
        )
        logger.log_info(
            f"Planned hand qpos max: {format_tensor(hand_traj.max(dim=1).values[0])}"
        )

    if not args.auto_play:
        input("Press Enter to start the upright demo...")
    last_logged_hand_target: torch.Tensor | None = None
    log_stride = max(1, total_steps // 10)
    for i in range(traj.shape[1]):
        robot.set_qpos(traj[:, i, :], joint_ids=joint_ids)
        sim.update(step=4)
        if should_clear_object_dynamics and i + 1 >= post_grasp_clear_step:
            release_object_after_grasp(obj)
            should_clear_object_dynamics = False
            if args.debug_hand_state:
                logger.log_info(
                    f"Object dynamics cleared at step={i}/{total_steps - 1}"
                )
        if args.debug_hand_state:
            action_hand_target = traj[:, i, arm_dof:]
            target_changed = last_logged_hand_target is None or not torch.allclose(
                action_hand_target, last_logged_hand_target, atol=1e-4
            )
            should_log = target_changed or i % log_stride == 0 or i == total_steps - 1
            if should_log:
                log_hand_execution_state(
                    robot,
                    step_idx=i,
                    total_steps=total_steps,
                    action_hand_target=action_hand_target,
                )
                last_logged_hand_target = action_hand_target.detach().clone()
            if i % log_stride == 0 or i == total_steps - 1:
                log_object_execution_state(
                    obj,
                    step_idx=i,
                    total_steps=total_steps,
                )
        time.sleep(1e-2)
    if not args.auto_play:
        input("Press Enter to exit the simulation...")


def main() -> None:
    args = parse_arguments()
    sim = initialize_simulation(args)
    robot = create_robot(sim, position=[0.0, 0.0, 0.0])
    run_upright_demo(args, sim, robot)


if __name__ == "__main__":
    main()
