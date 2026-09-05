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

"""Demonstrate the unified dual-arm PickUp-to-HandOver atomic action."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.atomic_actions import (
    ControlPartCommandProfile,
    create_simulation_atomic_action_engine,
    HandOverGoal,
    HandOverOptions,
    MotionPolicy,
)
from embodichain.lab.sim.cfg import RigidBodyAttributesCfg, RigidObjectCfg
from embodichain.data import get_data_path
from embodichain.lab.sim.objects import RigidObject, Robot
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.utils import logger
from scripts.tutorials.atomic_action.scenario_utils import (
    add_dual_tutorial_robot,
    add_support_surface,
    settle_object,
)
from scripts.tutorials.atomic_action.tutorial_utils import (
    TutorialRobot,
    create_antipodal_semantics,
    create_parallel_jaw_grasp_pose_generator,
    create_toppra_motion_generator,
    create_tutorial_argument_parser,
    create_tutorial_simulation,
    get_hand_open_close_qpos,
    clone_local_pose_from_first_env,
    prepare_tutorial_scene,
    publish_tutorial_scene,
    replay_trajectory,
    run_tutorial,
    serve_tutorial_scene,
)

VERTICAL_OBJECT_MESH_PATH = get_data_path("SodaCan/simple_cola_can.obj")
HORIZONTAL_OBJECT_MESH_PATH = get_data_path(
    "CoordinatedPlacementAndPickment/pencil.glb"
)
VERTICAL_OBJECT_SCALE = (0.56, 0.56, 0.56)
HORIZONTAL_OBJECT_SCALE = (2.0, 2.0, 2.0)
GRIPPER_TCP_Z = 0.16
SUPPORT_SURFACE_Z = 0.50
SUPPORT_SURFACE_SIZE = (0.8, 1.2, 0.02)
SUPPORT_SURFACE_CENTER = (
    0.0,
    0.0,
    SUPPORT_SURFACE_Z - 0.5 * SUPPORT_SURFACE_SIZE[2],
)

# --- Adjustable scene placeholders -----------------------------------------
# The object starts on one side and is delivered to the other. HandOver chooses
# the nearer arm for pickup and computes the middle handover position itself.
OBJECT_INIT_XY = (0.0, 0.02)
OBJECT_ROT_VERTICAL = (90.0, 0.0, 0.0)
OBJECT_ROT_HORIZONTAL = (90.0, 0.0, 0.0)
FINAL_OBJECT_XYZ = (0.0, -0.2, 0.6)
# ---------------------------------------------------------------------------

HAND_CLOSE_QPOS = 0.04
HANDOVER_SAMPLE_INTERVAL = 220
HANDOVER_HAND_INTERP_STEPS = 10
HANDOVER_PRE_GRASP_DISTANCE = 0.08
HANDOVER_LIFT_HEIGHT = 0.15
TRAJECTORY_SIM_STEPS = 4
HANDOVER_RECORD_LOOK_AT = (
    (-1.0, 0.2, 1.8),
    (-0.4, 0.0, 0.7),
    (0.0, 0.0, 1.0),
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the demo."""
    parser = create_tutorial_argument_parser(
        "Dual-arm handover demo",
        features=("diagnose_plan", "headless_play"),
        default_device="cpu",
        default_renderer="hybrid",
    )
    parser.add_argument(
        "--is_horizontal",
        action="store_true",
        help="Use the horizontal WaterBasin object instead of the vertical soda can.",
    )
    return parser.parse_args()


def create_dual_robot(
    sim: SimulationManager,
    robot_type: TutorialRobot,
) -> Robot:
    """Create the selected dual-arm robot with its matching grippers."""
    return add_dual_tutorial_robot(
        sim,
        robot_type=robot_type,
        uid=f"Dual{robot_type.title()}HandOver",
        urdf_name=f"dual_{robot_type}_hand_over",
        tcp_z=GRIPPER_TCP_Z,
        ur_ik_nearest_weight=(1.0, 4.0, 1.0, 1.0, 1.0, 1.0),
        hand_stiffness=1e3,
        hand_damping=1e2,
        hand_max_effort=1e4,
    )


def create_support_surface(sim: SimulationManager) -> RigidObject:
    """Create a compact support slab under the staged object."""
    return add_support_surface(
        sim,
        size=SUPPORT_SURFACE_SIZE,
        center=SUPPORT_SURFACE_CENTER,
    )


def create_handover_object(sim: SimulationManager, args) -> RigidObject:
    """Create the mode-specific mesh object on the support surface."""
    mesh_path = (
        HORIZONTAL_OBJECT_MESH_PATH if args.is_horizontal else VERTICAL_OBJECT_MESH_PATH
    )
    body_scale = (
        HORIZONTAL_OBJECT_SCALE if args.is_horizontal else VERTICAL_OBJECT_SCALE
    )
    return sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="handover_object",
            shape=MeshCfg(fpath=mesh_path, compute_uv=False),
            attrs=RigidBodyAttributesCfg(
                mass=0.01,
                dynamic_friction=0.97,
                static_friction=0.99,
                angular_damping=1.0,
                linear_damping=0.5,
                contact_offset=0.001,
                rest_offset=0.0,
                restitution=0.01,
                min_position_iters=32,
                min_velocity_iters=8,
                max_depenetration_velocity=2.0,
            ),
            max_convex_hull_num=16,
            init_pos=[OBJECT_INIT_XY[0], OBJECT_INIT_XY[1], SUPPORT_SURFACE_Z + 0.12],
            init_rot=(
                OBJECT_ROT_VERTICAL if not args.is_horizontal else OBJECT_ROT_HORIZONTAL
            ),
            body_scale=body_scale,
        )
    )


def run_handover_demo(
    args: argparse.Namespace,
    sim: SimulationManager,
    robot: Robot,
) -> None:
    """Plan and optionally execute one unified pick-up and handover."""
    create_support_surface(sim)
    obj = create_handover_object(sim, args)
    settle_object(sim, obj, step=0)
    clone_local_pose_from_first_env(obj)
    obj.clear_dynamics()
    publish_tutorial_scene(sim, args)
    object_semantics = create_antipodal_semantics(obj, label="handover")
    motion_gen = create_toppra_motion_generator(robot)

    left_open, left_close = get_hand_open_close_qpos(
        robot, hand_control_part="left_hand", close_qpos=HAND_CLOSE_QPOS
    )
    right_open, right_close = get_hand_open_close_qpos(
        robot, hand_control_part="right_hand", close_qpos=HAND_CLOSE_QPOS
    )

    final_pose = torch.eye(4, dtype=torch.float32)
    final_pose[:3, 3] = torch.as_tensor(FINAL_OBJECT_XYZ)

    handover_options = HandOverOptions(
        pre_grasp_distance=HANDOVER_PRE_GRASP_DISTANCE,
        lift_height=HANDOVER_LIFT_HEIGHT,
        hand_interp_steps=HANDOVER_HAND_INTERP_STEPS,
    )
    grasp_pose_generator = create_parallel_jaw_grasp_pose_generator(
        n_sample=10_000,
        force_refresh=False,
    )
    engine = create_simulation_atomic_action_engine(
        motion_generator=motion_gen,
        scene_entities=(obj,),
        control_profiles={
            "left_hand": ControlPartCommandProfile.joint_positions(
                open=left_open,
                grasp=left_close,
            ),
            "right_hand": ControlPartCommandProfile.joint_positions(
                open=right_open,
                grasp=right_close,
            ),
        },
        grasp_pose_generators={
            "left_hand": grasp_pose_generator,
            "right_hand": grasp_pose_generator,
        },
    )
    wait_for_user = prepare_tutorial_scene(
        sim, args, "Inspect the scene, then press Enter to plan the handover..."
    )
    # wait for object to drop
    for _ in range(50):
        sim.update(step=10)
    compiled = engine.compile(
        (
            engine.make_invocation(
                "hand_over",
                HandOverGoal(object_semantics, target_pose=final_pose),
                control_parts={
                    "source": {
                        "motion": "left_arm",
                        "grasp": "left_hand",
                    },
                    "destination": {
                        "motion": "right_arm",
                        "grasp": "right_hand",
                    },
                },
                motion_policy=MotionPolicy(
                    strategy="motion_gen",
                    sample_count=HANDOVER_SAMPLE_INTERVAL,
                ),
                skill_options=handover_options,
            ),
        ),
        engine.initial_context(control_dt=sim.sim_config.physics_dt),
    )
    success = compiled.plan_success
    traj = compiled.trajectory.positions

    if not success.all():
        logger.log_warning("Failed to plan the unified HandOver trajectory.")
        return

    if args.diagnose_plan:
        logger.log_info(f"Planned full trajectory with {traj.shape[1]} waypoints.")
        return

    if wait_for_user:
        input("Press Enter to execute the handover...")

    replay_trajectory(
        sim,
        robot,
        compiled.trajectory,
        args,
        video_prefix="handover_auto_play",
        hold_steps=0,
        trajectory_sim_steps=TRAJECTORY_SIM_STEPS,
        look_at=HANDOVER_RECORD_LOOK_AT,
    )
    if wait_for_user:
        input("Press Enter to exit the simulation...")


def main() -> None:
    """Run the dual-arm handover demo."""
    args = parse_arguments()
    sim = create_tutorial_simulation(
        args,
        arena_space=3.0,
        light_pos=(0.0, -0.4, 3.0),
    )
    robot = create_dual_robot(sim, args.robot)
    run_handover_demo(args, sim, robot)
    serve_tutorial_scene(sim, args)


if __name__ == "__main__":
    run_tutorial(main)
