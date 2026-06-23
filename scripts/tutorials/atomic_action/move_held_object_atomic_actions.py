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

"""Demonstrate moving a held object to an object-centric target pose."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
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
    GraspTarget,
    HeldObjectPoseTarget,
    MoveEndEffector,
    MoveEndEffectorCfg,
    MoveHeldObject,
    MoveHeldObjectCfg,
    ObjectSemantics,
    PickUp,
    PickUpCfg,
    EndEffectorPoseTarget,
)
from embodichain.lab.sim.cfg import (
    JointDrivePropertiesCfg,
    LightCfg,
    MarkerCfg,
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
MOVE_HELD_OBJECT_SAMPLE_INTERVAL = 120
HAND_INTERP_STEPS = 12
POST_TRAJECTORY_STEPS = 240
DEFAULT_AUTO_PLAY_LOOK_AT = (
    (-1.6, -1.5, 1.2),
    (0.0, 0.0, 0.25),
    (0.0, 0.0, 1.0),
)
RECORD_WIDTH = 640
RECORD_HEIGHT = 480
VIEWER_WIDTH = 1600
VIEWER_HEIGHT = 900
AUTO_PLAY_RECORD_FPS = 20
AUTO_PLAY_RECORD_MAX_MEMORY = 2048
EEF_AXIS_LEN = 0.06
EEF_AXIS_SIZE = 0.003
TABLE_SIZE = [1.0, 1.4, 0.05]
TABLE_TOP_Z = -0.045


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demonstrate MoveHeldObject holding a bottle in the gripper."
    )
    add_env_launcher_args_to_parser(parser)
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
    parser.add_argument(
        "--no_vis_eef_axis",
        action="store_true",
        help="Do not draw the current end-effector/TCP coordinate frame before planning.",
    )
    return parser.parse_args()


def get_tutorial_window_size(args: argparse.Namespace) -> tuple[int, int]:
    """Return the viewer window size used by this tutorial."""
    return VIEWER_WIDTH, VIEWER_HEIGHT


def start_auto_play_recording(
    sim: SimulationManager,
    args: argparse.Namespace,
    video_prefix: str,
    look_at: tuple[
        Sequence[float],
        Sequence[float],
        Sequence[float],
    ] = DEFAULT_AUTO_PLAY_LOOK_AT,
) -> bool:
    """Start recording for ``--auto_play`` tutorial runs."""
    if not getattr(args, "auto_play", False):
        return False

    original_width = sim.sim_config.width
    original_height = sim.sim_config.height
    try:
        sim.sim_config.width = RECORD_WIDTH
        sim.sim_config.height = RECORD_HEIGHT
        if not sim.start_window_record(
            fps=AUTO_PLAY_RECORD_FPS,
            max_memory=AUTO_PLAY_RECORD_MAX_MEMORY,
            video_prefix=video_prefix,
            look_at=look_at,
            use_sim_time=True,
        ):
            raise RuntimeError("Failed to start auto_play recording.")
    finally:
        sim.sim_config.width = original_width
        sim.sim_config.height = original_height
    return True


def stop_auto_play_recording(
    sim: SimulationManager,
    recording_started: bool,
) -> None:
    """Stop recording and wait until the mp4 has been written."""
    if not recording_started:
        return

    if sim.is_window_recording():
        sim.stop_window_record()
    sim.wait_window_record_saves()


def initialize_simulation(args: argparse.Namespace) -> SimulationManager:
    width, height = get_tutorial_window_size(args)
    cfg = SimulationManagerCfg(
        width=width,
        height=height,
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
            mesh_vertices=obj.get_vertices(env_ids=[0], scale=True)[0],
            mesh_triangles=obj.get_triangles(env_ids=[0])[0],
            gripper_collision_cfg=build_gripper_collision_cfg(),
            generator_cfg=build_grasp_generator_cfg(args),
            force_reannotate=args.force_reannotate,
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


def make_pre_pick_eef_pose(robot: Robot, position: torch.Tensor) -> torch.Tensor:
    pose = robot.compute_fk(
        qpos=robot.get_qpos(name="arm"),
        name="arm",
        to_matrix=True,
    )[0].clone()
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
    pose[:3, 3] = torch.tensor([0.28, -0.2, 0.22], dtype=torch.float32, device=device)
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


def draw_current_eef_axis(sim: SimulationManager, robot: Robot) -> None:
    eef_pose = robot.compute_fk(
        qpos=robot.get_qpos(name="arm"),
        name="arm",
        to_matrix=True,
    )
    sim.draw_marker(
        cfg=MarkerCfg(
            name="current_eef_axis",
            marker_type="axis",
            axis_xpos=eef_pose,
            axis_size=EEF_AXIS_SIZE,
            axis_len=EEF_AXIS_LEN,
        )
    )


def build_action_sequence(
    hand_open: torch.Tensor,
    hand_close: torch.Tensor,
    device: torch.device,
) -> list:
    move_cfg = MoveEndEffectorCfg(
        control_part="arm",
        sample_interval=MOVE_SAMPLE_INTERVAL,
    )
    pickup_cfg = PickUpCfg(
        control_part="arm",
        hand_control_part="hand",
        hand_open_qpos=hand_open,
        hand_close_qpos=hand_close,
        approach_direction=torch.tensor(
            BOTTLE_APPROACH_DIRECTION, dtype=torch.float32, device=device
        ),
        pre_grasp_distance=0.15,
        lift_height=0.16,
        sample_interval=PICK_SAMPLE_INTERVAL,
        hand_interp_steps=HAND_INTERP_STEPS,
    )
    move_held_object_cfg = MoveHeldObjectCfg(
        control_part="arm",
        hand_control_part="hand",
        hand_close_qpos=hand_close,
        sample_interval=MOVE_HELD_OBJECT_SAMPLE_INTERVAL,
    )
    return [move_cfg, pickup_cfg, move_held_object_cfg]


def run_move_held_object_demo(args: argparse.Namespace) -> None:
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
    action_cfgs = build_action_sequence(hand_open, hand_close, sim.device)
    atomic_engine = AtomicActionEngine(motion_generator=motion_gen)
    _action_classes = {
        "move_end_effector": MoveEndEffector,
        "pick_up": PickUp,
        "move_held_object": MoveHeldObject,
    }
    for cfg in action_cfgs:
        atomic_engine.register(_action_classes[cfg.name](motion_gen, cfg=cfg))

    if not args.headless:
        sim.open_window()
    if not args.no_vis_eef_axis:
        draw_current_eef_axis(sim, robot)
    if not args.auto_play:
        input("Inspect the fallen bottle, then press Enter to plan...")

    obj_pose = obj.get_local_pose(to_matrix=True)
    move_position = obj_pose[0, :3, 3].clone()
    move_position[2] = 0.36
    move_target = make_pre_pick_eef_pose(robot, move_position)
    move_held_object_target = HeldObjectPoseTarget(
        object_target_pose=make_upright_object_pose(sim.device)
    )

    logger.log_info("Planning move_end_effector -> pick_up -> move_held_object")
    start_time = time.time()
    is_success, traj, _ = atomic_engine.run(
        steps=[
            ("move_end_effector", EndEffectorPoseTarget(xpos=move_target)),
            ("pick_up", GraspTarget(semantics=semantics)),
            ("move_held_object", move_held_object_target),
        ]
    )
    cost_time = time.time() - start_time
    logger.log_info(f"Plan trajectory cost time: {cost_time:.2f} seconds")
    if not is_success:
        logger.log_warning("Failed to plan move_held_object demo trajectory.")
        return

    if not args.auto_play:
        input("Press Enter to replay the move_held_object demo...")

    recording_started = start_auto_play_recording(
        sim, args, video_prefix="move_held_object_auto_play"
    )
    try:
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

        logger.log_info("MoveHeldObject keeps the bottle suspended in the gripper.")

        final_qpos = traj[:, -1, :]
        for i in range(POST_TRAJECTORY_STEPS):
            robot.set_qpos(final_qpos)
            sim.update(step=2)
            if args.debug_state and i % max(1, POST_TRAJECTORY_STEPS // 5) == 0:
                log_object_state(obj, f"post step {i}/{POST_TRAJECTORY_STEPS - 1}")
            time.sleep(1e-2)
    finally:
        stop_auto_play_recording(sim, recording_started)

    if not args.auto_play:
        input("Press Enter to exit the simulation...")


def main() -> None:
    args = parse_arguments()
    run_move_held_object_demo(args)


if __name__ == "__main__":
    main()
