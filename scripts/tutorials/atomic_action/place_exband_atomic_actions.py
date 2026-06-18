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

"""Demonstrate object-centric airborne place with optional release.

The sequence uses three atomic actions:

1. Move above a fallen bottle.
2. Pick up the bottle.
3. Place the held bottle at an upright airborne target pose.

Use ``--mode hold`` to keep the gripper closed at the final pose, or
``--mode release`` to open the gripper and let the bottle fall naturally.
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

from embodichain.data import get_data_path
from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.atomic_actions import (
    AntipodalAffordance,
    AtomicActionEngine,
    MoveActionCfg,
    ObjectSemantics,
    PickUpActionCfg,
    PlaceActionCfg,
    PlaceTarget,
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

GRIPPER_URDF_PATH = "DH_PGI_140_80/DH_PGI_140_80.urdf"
GRIPPER_HAND_JOINT_PATTERN = "GRIPPER_FINGER1_JOINT_1"
GRIPPER_MAX_OPEN_WIDTH = 0.080
GRIPPER_FINGER_LENGTH = 0.088
GRIPPER_ROOT_Z_WIDTH = 0.096
GRIPPER_Y_THICKNESS = 0.040
GRIPPER_TCP_Z = 0.15

BOTTLE_LABEL = "bottle"
BOTTLE_APPROACH_DIRECTION = (0.0, 0.0, -1.0)
BOTTLE_MIN_HAND_CLOSE_QPOS = 0.024

MOVE_SAMPLE_INTERVAL = 60
PICK_SAMPLE_INTERVAL = 120
PLACE_SAMPLE_INTERVAL = 120
HAND_INTERP_STEPS = 12
PLACE_HEIGHT_OFFSET = 0.1
POST_TRAJECTORY_STEPS = 240
TABLE_SIZE = [1.0, 1.4, 0.05]
TABLE_TOP_Z = -0.045


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demonstrate expanded PlaceAction with optional release."
    )
    add_env_launcher_args_to_parser(parser)
    parser.add_argument(
        "--mode",
        choices=("hold", "release"),
        default="hold",
        help="hold keeps the bottle in the gripper; release opens the gripper.",
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
        action="store_true",
        help="Log bottle pose during replay.",
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


def create_fallen_bottle(sim: SimulationManager) -> RigidObject:
    bottle_scale = 0.0008
    cfg = RigidObjectCfg(
        uid="bottle",
        shape=MeshCfg(fpath=get_data_path("ScannedBottle/yibao.ply")),
        attrs=RigidBodyAttributesCfg(
            mass=0.02,
            dynamic_friction=0.97,
            static_friction=0.99,
        ),
        max_convex_hull_num=16,
        init_pos=[-0.4294, -0.0825, -0.0997],
        init_rot=[90.0, 45.0, 0.0],
        body_scale=(bottle_scale, bottle_scale, bottle_scale),
    )
    return sim.add_rigid_object(cfg=cfg)


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
        torch.full_like(hand_close_limit, BOTTLE_MIN_HAND_CLOSE_QPOS),
    )
    return hand_open, hand_close


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


def make_upright_object_pose(device: torch.device) -> torch.Tensor:
    pose = torch.eye(4, dtype=torch.float32, device=device)
    pose[:3, :3] = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    pose[:3, 3] = torch.tensor([0.28, -0.2, 0.12], dtype=torch.float32, device=device)
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


def run_place_demo(args: argparse.Namespace) -> None:
    sim = initialize_simulation(args)
    robot = create_robot(sim)
    create_table(sim)
    obj = create_fallen_bottle(sim)

    settle_object(sim, obj, step=5)
    semantics = create_object_semantics(obj, args)
    motion_gen = MotionGenerator(
        cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=robot.uid))
    )
    hand_open, hand_close = get_hand_open_close_qpos(robot, sim.device)

    move_cfg = MoveActionCfg(
        control_part="arm",
        sample_interval=MOVE_SAMPLE_INTERVAL,
    )
    pickup_cfg = PickUpActionCfg(
        control_part="arm",
        hand_control_part="hand",
        hand_open_qpos=hand_open,
        hand_close_qpos=hand_close,
        approach_direction=torch.tensor(
            BOTTLE_APPROACH_DIRECTION, dtype=torch.float32, device=sim.device
        ),
        pre_grasp_distance=0.15,
        lift_height=0.16,
        sample_interval=PICK_SAMPLE_INTERVAL,
        hand_interp_steps=HAND_INTERP_STEPS,
    )
    place_cfg = PlaceActionCfg(
        control_part="arm",
        hand_control_part="hand",
        hand_open_qpos=hand_open,
        hand_close_qpos=hand_close,
        release=args.mode == "release",
        place_height_offset=PLACE_HEIGHT_OFFSET,
        lift_height=0.08,
        sample_interval=PLACE_SAMPLE_INTERVAL,
        hand_interp_steps=HAND_INTERP_STEPS,
    )
    atomic_engine = AtomicActionEngine(
        motion_generator=motion_gen,
        actions_cfg_list=[move_cfg, pickup_cfg, place_cfg],
    )

    sim.open_window()
    if not args.auto_play:
        input("Inspect the fallen bottle, then press Enter to plan...")

    obj_pose = obj.get_local_pose(to_matrix=True)
    move_position = obj_pose[0, :3, 3].clone()
    move_position[2] = 0.36
    move_target = make_top_down_eef_pose(move_position)
    place_target = PlaceTarget(
        object_target_pose=make_upright_object_pose(sim.device),
        release=args.mode == "release",
        height_offset=PLACE_HEIGHT_OFFSET,
    )

    logger.log_info(
        "Planning move -> pick_up -> expanded place "
        f"(mode={args.mode}, release={args.mode == 'release'})"
    )
    start_time = time.time()
    is_success, traj = atomic_engine.execute_static(
        target_list=[move_target, semantics, place_target]
    )
    cost_time = time.time() - start_time
    logger.log_info(f"Plan trajectory cost time: {cost_time:.2f} seconds")
    if not is_success:
        logger.log_warning("Failed to plan expanded place demo trajectory.")
        return

    if not args.auto_play:
        input("Press Enter to replay the expanded place demo...")

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

    if args.mode == "release":
        logger.log_info("Release mode: simulating after gripper opens.")
    else:
        logger.log_info("Hold mode: keeping the bottle suspended in the gripper.")

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
    run_place_demo(args)


if __name__ == "__main__":
    main()
