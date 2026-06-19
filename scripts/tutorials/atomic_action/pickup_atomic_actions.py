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

"""Demonstrate PickUpAction on an upright object with configurable approach."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from embodichain.data import get_data_path
from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.atomic_actions import (
    AntipodalAffordance,
    AtomicActionEngine,
    MoveActionCfg,
    ObjectSemantics,
    PickUpActionCfg,
)
from embodichain.lab.sim.cfg import (
    JointDrivePropertiesCfg,
    LightCfg,
    RenderCfg,
    RigidBodyAttributesCfg,
    RigidObjectCfg,
    RobotCfg,
    URDFCfg,
)
from embodichain.lab.sim.objects import RigidObject, Robot
from embodichain.lab.sim.planners import MotionGenerator, MotionGenCfg, ToppraPlannerCfg
from embodichain.lab.sim.shapes import CubeCfg, MeshCfg
from embodichain.lab.sim.solvers import PytorchSolverCfg
from embodichain.toolkits.graspkit.pg_grasp.antipodal_generator import (
    AntipodalSamplerCfg,
    GraspGeneratorCfg,
)
from embodichain.toolkits.graspkit.pg_grasp.gripper_collision_checker import (
    GripperCollisionCfg,
)
from embodichain.utils import logger
from embodichain.utils.math import matrix_from_euler

GRIPPER_URDF_PATH = "DH_PGI_140_80/DH_PGI_140_80.urdf"
GRIPPER_HAND_JOINT_PATTERN = "GRIPPER_FINGER1_JOINT_1"
GRIPPER_MAX_OPEN_WIDTH = 0.080
GRIPPER_FINGER_LENGTH = 0.088
GRIPPER_ROOT_Z_WIDTH = 0.096
GRIPPER_Y_THICKNESS = 0.040
GRIPPER_TCP_Z = 0.15

OBJECT_MIN_HAND_CLOSE_QPOS = 0.024
OBJECT_XY = (-0.42, -0.08)
OBJECT_CLEARANCE = 0.0

OBJECT_PRESETS = {
    "paper_cup": {
        "label": "paper_cup",
        "mesh_path": "PaperCup/paper_cup.ply",
        "init_rot": (0.0, 0.0, 0.0),
        "body_scale": (1.0, 1.0, 1.0),
        "mass": 0.01,
    },
    "coffee_cup": {
        "label": "coffee_cup",
        "mesh_path": "CoffeeCup/cup.ply",
        "init_rot": (0.0, 0.0, -90.0),
        "body_scale": (1.0, 1.0, 1.0),
        "mass": 0.01,
    },
    "bottle": {
        "label": "bottle",
        "mesh_path": "ScannedBottle/yibao.ply",
        "init_rot": (180.0, 0.0, 0.0),
        "body_scale": (0.0008, 0.0008, 0.0008),
        "mass": 0.02,
    },
}

MOVE_SAMPLE_INTERVAL = 60
PICK_SAMPLE_INTERVAL = 120
HAND_INTERP_STEPS = 12
POST_TRAJECTORY_STEPS = 240
TABLE_SIZE = [1.0, 1.4, 0.05]
TABLE_TOP_Z = -0.045

APPROACH_DIRECTIONS = {
    "top": (0.0, 0.0, -1.0),
    "side": (0.0, 1.0, 0.0),
    "side_y": (0.0, -1.0, 0.0),
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demonstrate PickUpAction on an upright object."
    )
    add_env_launcher_args_to_parser(parser)
    parser.add_argument(
        "--object",
        choices=sorted(OBJECT_PRESETS.keys()),
        default="paper_cup",
        help="Object preset to pick.",
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=10000,
        help="Number of samples for antipodal grasp generation.",
    )
    parser.add_argument(
        "--force_reannotate",
        action="store_true",
        help="Force grasp region re-annotation instead of using cached data.",
    )
    parser.add_argument(
        "--auto_play",
        action="store_true",
        help="Run the viewer demo without waiting for keyboard input.",
    )
    parser.add_argument(
        "--debug_state",
        "--debug",
        action="store_true",
        help="Log object pose during replay.",
    )
    parser.add_argument(
        "--approach",
        choices=["top", "side", "side_y", "custom"],
        default="side",
        help="Pick approach direction preset.",
    )
    parser.add_argument(
        "--custom_approach_direction",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="World-frame approach direction used when --approach custom.",
    )
    return parser.parse_args()


def initialize_simulation(args: argparse.Namespace) -> SimulationManager:
    cfg = SimulationManagerCfg(
        headless=True,
        sim_device=args.device,
        render_cfg=RenderCfg(renderer=args.renderer),
        physics_dt=1.0 / 100.0,
        arena_space=2.5,
    )
    sim = SimulationManager(cfg)
    sim.add_light(
        cfg=LightCfg(
            uid="main_light",
            color=(0.6, 0.6, 0.6),
            intensity=30.0,
            init_pos=(1.0, 0.0, 3.0),
        )
    )
    return sim


def create_robot(sim: SimulationManager, position=(0.0, 0.0, 0.0)) -> Robot:
    ur5_urdf_path = get_data_path("UniversalRobots/UR5/UR5.urdf")
    gripper_urdf_path = get_data_path(GRIPPER_URDF_PATH)

    cfg = RobotCfg(
        uid="UR5",
        urdf_cfg=URDFCfg(
            components=[
                {"component_type": "arm", "urdf_path": ur5_urdf_path},
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
        init_qpos=[0.0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.0, 0.0],
        init_pos=position,
    )
    return sim.add_robot(cfg=cfg)


def create_table(sim: SimulationManager) -> RigidObject:
    cfg = RigidObjectCfg(
        uid="table",
        shape=CubeCfg(size=TABLE_SIZE),
        body_type="static",
        attrs=RigidBodyAttributesCfg(
            dynamic_friction=0.8,
            static_friction=0.9,
        ),
        init_pos=[-0.30, 0.10, TABLE_TOP_Z - 0.5 * TABLE_SIZE[2]],
    )
    return sim.add_rigid_object(cfg=cfg)


def create_pick_object(sim: SimulationManager, object_name: str) -> RigidObject:
    preset = OBJECT_PRESETS[object_name]
    cfg = RigidObjectCfg(
        uid=preset["label"],
        shape=MeshCfg(fpath=get_data_path(preset["mesh_path"])),
        attrs=RigidBodyAttributesCfg(
            mass=preset["mass"],
            dynamic_friction=0.97,
            static_friction=0.99,
        ),
        max_convex_hull_num=16,
        init_pos=[OBJECT_XY[0], OBJECT_XY[1], 0.0],
        init_rot=preset["init_rot"],
        body_scale=preset["body_scale"],
    )
    obj = sim.add_rigid_object(cfg=cfg)
    obj.cfg.init_pos = _compute_tabletop_init_pos(obj, cfg.init_rot)
    obj.reset()
    return obj


def _compute_tabletop_init_pos(
    obj: RigidObject, init_rot: tuple[float, float, float]
) -> tuple[float, float, float]:
    vertices = obj.get_vertices(env_ids=[0], scale=True)[0]
    rot = torch.as_tensor(init_rot, dtype=torch.float32, device=vertices.device)
    rot = rot.unsqueeze(0) * torch.pi / 180.0
    upright_rot = matrix_from_euler(rot, "XYZ")[0]
    rotated_vertices = vertices @ upright_rot.T
    bottom_z = rotated_vertices[:, 2].min().item()
    z = TABLE_TOP_Z + OBJECT_CLEARANCE - bottom_z
    return (OBJECT_XY[0], OBJECT_XY[1], z)


def settle_object(sim: SimulationManager, obj: RigidObject, step: int = 5) -> None:
    if sim.device.type == "cuda":
        sim.init_gpu_physics()
    obj.reset()
    sim.update(step=step)
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


def create_object_semantics(
    obj: RigidObject, args: argparse.Namespace
) -> ObjectSemantics:
    label = OBJECT_PRESETS[args.object]["label"]
    return ObjectSemantics(
        label=label,
        geometry={
            "mesh_vertices": obj.get_vertices(env_ids=[0], scale=True)[0],
            "mesh_triangles": obj.get_triangles(env_ids=[0])[0],
        },
        affordance=AntipodalAffordance(
            object_label=label,
            force_reannotate=args.force_reannotate,
            custom_config={
                "gripper_collision_cfg": build_gripper_collision_cfg(),
                "generator_cfg": build_grasp_generator_cfg(args),
            },
        ),
        entity=obj,
    )


def get_hand_open_close_qpos(
    robot: Robot, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    hand_limits = robot.get_qpos_limits(name="hand")[0].to(
        device=device, dtype=torch.float32
    )
    hand_open = hand_limits[:, 0]
    hand_close_limit = hand_limits[:, 1]
    hand_close = torch.minimum(
        hand_close_limit,
        torch.full_like(hand_close_limit, OBJECT_MIN_HAND_CLOSE_QPOS),
    )
    return hand_open, hand_close


def resolve_approach_direction(
    args: argparse.Namespace, device: torch.device
) -> torch.Tensor:
    if args.approach == "custom":
        if args.custom_approach_direction is None:
            raise ValueError(
                "--custom_approach_direction is required when --approach custom."
            )
        direction = args.custom_approach_direction
    else:
        direction = APPROACH_DIRECTIONS[args.approach]

    approach_direction = torch.tensor(direction, dtype=torch.float32, device=device)
    norm = torch.linalg.norm(approach_direction)
    if norm < 1e-6:
        raise ValueError("approach_direction must be non-zero.")
    return approach_direction / norm


def make_top_down_eef_pose(position: torch.Tensor) -> torch.Tensor:
    pose = torch.eye(4, dtype=torch.float32, device=position.device)
    pose[:3, :3] = torch.tensor(
        [
            [-0.0539, -0.9985, -0.0022],
            [-0.9977, 0.0540, -0.0401],
            [0.0401, 0.0000, -0.9992],
        ],
        dtype=torch.float32,
        device=position.device,
    )
    pose[:3, 3] = position
    return pose


def compute_pick_close_end_step() -> int:
    motion_waypoints = PICK_SAMPLE_INTERVAL - HAND_INTERP_STEPS
    n_approach = int(round(motion_waypoints) * 0.6)
    return MOVE_SAMPLE_INTERVAL + n_approach + HAND_INTERP_STEPS


def format_tensor(tensor: torch.Tensor) -> str:
    rounded = (tensor.detach().cpu() * 10000.0).round() / 10000.0
    return str(rounded.tolist())


def log_object_state(obj: RigidObject, label: str) -> None:
    obj_pose = obj.get_local_pose(to_matrix=True)
    logger.log_info(
        f"{label}: pos={format_tensor(obj_pose[0, :3, 3])}, "
        f"z_axis={format_tensor(obj_pose[0, :3, 2])}"
    )


def build_action_sequence(
    hand_open: torch.Tensor,
    hand_close: torch.Tensor,
    approach_direction: torch.Tensor,
) -> list:
    move_cfg = MoveActionCfg(
        control_part="arm",
        sample_interval=MOVE_SAMPLE_INTERVAL,
    )
    pickup_cfg = PickUpActionCfg(
        control_part="arm",
        hand_control_part="hand",
        hand_open_qpos=hand_open,
        hand_close_qpos=hand_close,
        approach_direction=approach_direction,
        pre_grasp_distance=0.15,
        lift_height=0.16,
        sample_interval=PICK_SAMPLE_INTERVAL,
        hand_interp_steps=HAND_INTERP_STEPS,
    )
    return [move_cfg, pickup_cfg]


def run_pickup_demo(args: argparse.Namespace) -> None:
    sim = initialize_simulation(args)
    robot = create_robot(sim)
    create_table(sim)
    obj = create_pick_object(sim, args.object)

    settle_object(sim, obj, step=5)
    semantics = create_object_semantics(obj, args)
    motion_gen = MotionGenerator(
        cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=robot.uid))
    )
    hand_open, hand_close = get_hand_open_close_qpos(robot, sim.device)
    approach_direction = resolve_approach_direction(args, sim.device)
    action_cfgs = build_action_sequence(hand_open, hand_close, approach_direction)
    atomic_engine = AtomicActionEngine(
        motion_generator=motion_gen,
        actions_cfg_list=action_cfgs,
    )

    sim.open_window()
    if not args.auto_play:
        input(f"Inspect the upright {args.object}, then press Enter to plan...")

    obj_pose = obj.get_local_pose(to_matrix=True)
    move_position = obj_pose[0, :3, 3].clone()
    move_position[2] = 0.36
    move_target = make_top_down_eef_pose(move_position)

    logger.log_info(
        f"Planning move -> pick_up for {args.object} with "
        f"approach_direction={format_tensor(approach_direction)}"
    )
    start_time = time.time()
    is_success, traj = atomic_engine.execute_static(
        target_list=[move_target, semantics]
    )
    cost_time = time.time() - start_time
    logger.log_info(f"Plan trajectory cost time: {cost_time:.2f} seconds")
    if not is_success:
        logger.log_warning("Failed to plan pickup demo trajectory.")
        return

    if not args.auto_play:
        input("Press Enter to replay the pickup demo...")

    post_grasp_clear_step = compute_pick_close_end_step()
    should_clear_object_dynamics = True
    log_stride = max(1, traj.shape[1] // 10)
    for i in range(traj.shape[1]):
        robot.set_qpos(traj[:, i, :])
        sim.update(step=4)
        if should_clear_object_dynamics and i + 1 >= post_grasp_clear_step:
            obj.clear_dynamics()
            should_clear_object_dynamics = False
            logger.log_info(f"Object dynamics cleared after grasp at step={i}")
        if args.debug_state and (i % log_stride == 0 or i == traj.shape[1] - 1):
            log_object_state(obj, f"replay step {i}/{traj.shape[1] - 1}")
        time.sleep(1e-2)

    logger.log_info(
        f"PickUpAction keeps the upright {args.object} suspended in the gripper."
    )

    final_qpos = traj[:, -1, :]
    for i in range(POST_TRAJECTORY_STEPS):
        robot.set_qpos(final_qpos)
        sim.update(step=2)
        if args.debug_state and i % max(1, POST_TRAJECTORY_STEPS // 5) == 0:
            log_object_state(obj, f"post step {i}/{POST_TRAJECTORY_STEPS - 1}")
        time.sleep(1e-2)

    if not args.auto_play:
        input("Press Enter to exit the simulation...")


def main() -> None:
    args = parse_arguments()
    run_pickup_demo(args)


if __name__ == "__main__":
    main()
