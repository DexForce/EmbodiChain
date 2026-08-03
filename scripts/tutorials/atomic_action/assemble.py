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

"""Demonstrate object assembly with a dual-arm UR5.

The left arm picks up a soda can (object A) and places it directly above a cube
(object B). The relative pose of the can with respect to the cube is declared on
an :class:`~embodichain.lab.sim.atomic_actions.AssembleAffordance` and consumed
by the :class:`~embodichain.lab.sim.atomic_actions.Place` action through an
:class:`~embodichain.lab.sim.atomic_actions.AssembleGoal`.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.atomic_actions import (
    ActionBinding,
    ActionInvocation,
    AssembleAffordance,
    AssembleGoal,
    AtomicActionEngine,
    ControlPartCommandProfile,
    GraspGoal,
    PickUpOptions,
    PlaceOptions,
    MotionPolicy,
)
from embodichain.lab.sim.cfg import RigidBodyAttributesCfg, RigidObjectCfg
from embodichain.data import get_data_path
from embodichain.lab.sim.objects import RigidObject, Robot
from embodichain.lab.sim.shapes import CubeCfg, MeshCfg
from embodichain.utils import logger
from scripts.tutorials.atomic_action.scenario_utils import (
    add_dual_ur5_robot,
    add_support_surface,
    make_dual_ur5_solver_cfg,
    settle_object,
)
from scripts.tutorials.atomic_action.tutorial_utils import (
    broadcast_pose_batch,
    clone_local_pose_from_first_env,
    create_antipodal_semantics,
    create_toppra_motion_generator,
    create_tutorial_argument_parser,
    create_tutorial_simulation,
    draw_axis_marker,
    get_hand_open_close_qpos,
    prepare_tutorial_scene,
    publish_tutorial_scene,
    replay_trajectory,
    serve_tutorial_scene,
)

OBJECT_MESH_PATH = get_data_path("SodaCan/simple_cola_can.obj")
GRIPPER_TCP_Z = 0.155
SUPPORT_SURFACE_Z = 0.50
SUPPORT_SURFACE_SIZE = (0.8, 1.2, 0.02)
SUPPORT_SURFACE_CENTER = (
    0.0,
    0.0,
    SUPPORT_SURFACE_Z - 0.5 * SUPPORT_SURFACE_SIZE[2],
)

# --- Adjustable scene placeholders -----------------------------------------
# Object A (soda can) is staged for the left arm to pick up; object B (cube) is
# the assembly base the can is placed onto. Tweak these to match the dual-UR5
# reach and the soda-can mesh geometry.
OBJECT_A_XY = (0.0, 0.02)
OBJECT_B_XY = (0.0, 0.20)
CUBE_SIZE = 0.08
CAN_INIT_ROT = (90.0, 0.0, 0.0)
"""Soda-can initial rotation (xyz Euler, degrees); lays the can on its side."""
CAN_INIT_Z_OFFSET = 0.12
"""Hover height above the support surface so the side grasp can approach."""
ASSEMBLE_MARGIN = 0.01
"""Small gap between the can and the cube so the placement settles cleanly."""
# ---------------------------------------------------------------------------

# Rot_x(90 deg), matching CAN_INIT_ROT (xyz Euler, degrees). The can is held and
# placed in this orientation, so the assembly pose needs no reorientation.
_CAN_INIT_ROTATION = torch.tensor(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=torch.float32
)

HAND_CLOSE_QPOS = 0.026
PICKUP_SAMPLE_INTERVAL = 80
PICKUP_HAND_INTERP_STEPS = 5
PICKUP_PRE_GRASP_DISTANCE = 0.08
PICKUP_LIFT_HEIGHT = 0.1
PLACE_SAMPLE_INTERVAL = 120
PLACE_HAND_INTERP_STEPS = 8
PLACE_LIFT_HEIGHT = 0.1
TRAJECTORY_SIM_STEPS = 4
ASSEMBLE_RECORD_LOOK_AT = (
    (-0.25, 0.02, 2.5),
    (0.0, 0.10, 0.75),
    (0.0, 0.0, 1.0),
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the assembly demo."""
    parser = create_tutorial_argument_parser(
        "Dual-arm object assembly demo",
        features=(
            "diagnose_plan",
            "grasp_sampling",
            "headless_play",
            "visualize_axes",
        ),
        default_device="cpu",
        default_renderer="hybrid",
    )
    return parser.parse_args()


def create_dual_ur5_robot(sim: SimulationManager) -> Robot:
    """Create a dual-UR5 robot with one PGI gripper on each arm."""
    return add_dual_ur5_robot(
        sim,
        uid="DualUR5Assemble",
        urdf_name="dual_ur5_assemble",
        solver_cfg=make_dual_ur5_solver_cfg(
            GRIPPER_TCP_Z,
            ur_ik_nearest_weight=(1.0, 4.0, 1.0, 1.0, 1.0, 1.0),
        ),
        hand_stiffness=1e2,
        hand_damping=1e1,
        hand_max_effort=1e3,
    )


def create_support_surface(sim: SimulationManager) -> RigidObject:
    """Create a compact support slab under the staged objects."""
    return add_support_surface(
        sim,
        size=SUPPORT_SURFACE_SIZE,
        center=SUPPORT_SURFACE_CENTER,
    )


def create_assemble_object(sim: SimulationManager) -> RigidObject:
    """Create the soda can (object A) staged for the left arm to pick up."""
    return sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="assemble_object",
            shape=MeshCfg(fpath=OBJECT_MESH_PATH, compute_uv=False),
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
            max_convex_hull_num=1,
            init_pos=[
                OBJECT_A_XY[0],
                OBJECT_A_XY[1],
                SUPPORT_SURFACE_Z + CAN_INIT_Z_OFFSET,
            ],
            init_rot=list(CAN_INIT_ROT),
            body_scale=(0.56, 0.56, 0.56),
        )
    )


def create_base_object(sim: SimulationManager) -> RigidObject:
    """Create the cube (object B) the soda can is assembled onto."""
    return sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="base_object",
            shape=CubeCfg(size=[CUBE_SIZE, CUBE_SIZE, CUBE_SIZE]),
            attrs=RigidBodyAttributesCfg(
                mass=1.0,
                dynamic_friction=0.9,
                static_friction=0.95,
                restitution=0.01,
            ),
            body_type="static",
            init_pos=[
                OBJECT_B_XY[0],
                OBJECT_B_XY[1],
                SUPPORT_SURFACE_Z + 0.5 * CUBE_SIZE,
            ],
            init_rot=[0.0, 0.0, 0.0],
        )
    )


def compute_can_half_height(can: RigidObject) -> float:
    """Return half the soda-can extent along world Z when laid on its side."""
    vertices = can.get_vertices(env_ids=[0], scale=True)[0].to(torch.float32)
    rotated = vertices @ _CAN_INIT_ROTATION.T
    extent_z = float(rotated[:, 2].max().item() - rotated[:, 2].min().item())
    return 0.5 * extent_z


def make_assemble_to_base_pose(dz: float) -> torch.Tensor:
    """Build the can pose relative to the cube: above it, same orientation."""
    pose = torch.eye(4, dtype=torch.float32)
    pose[:3, :3] = _CAN_INIT_ROTATION
    pose[2, 3] = dz
    return pose


def run_assemble_demo(
    args: argparse.Namespace,
    sim: SimulationManager,
    robot: Robot,
) -> None:
    """Plan and optionally execute a pick-up followed by an assembly place."""
    create_support_surface(sim)
    can = create_assemble_object(sim)
    cube = create_base_object(sim)

    settle_object(sim, can, step=0)
    clone_local_pose_from_first_env(can)
    can.clear_dynamics()
    sim.update(step=10)
    clone_local_pose_from_first_env(cube)
    cube.clear_dynamics()
    publish_tutorial_scene(sim, args)

    can_semantics = create_antipodal_semantics(
        can,
        label="soda_can",
        n_sample=args.n_sample,
        force_reannotate=args.force_reannotate,
    )
    motion_gen = create_toppra_motion_generator(robot)
    left_open, left_close = get_hand_open_close_qpos(
        robot, hand_control_part="left_hand", close_qpos=HAND_CLOSE_QPOS
    )

    can_half_z = compute_can_half_height(can)
    assemble_to_base = make_assemble_to_base_pose(
        0.5 * CUBE_SIZE + can_half_z + ASSEMBLE_MARGIN
    )
    cube_pose = cube.get_local_pose(to_matrix=True)
    assemble_object_target_pose = cube_pose[0] @ assemble_to_base

    n_envs = robot.get_qpos().shape[0]
    if not args.no_vis_eef_axis:
        draw_axis_marker(
            sim,
            "assemble_target_axis",
            broadcast_pose_batch(assemble_object_target_pose, n_envs),
        )

    # Step 1 - the left arm picks the soda can up by its top part.
    pick_up_options = PickUpOptions(
        pick_object_part="top",
        pre_grasp_distance=PICKUP_PRE_GRASP_DISTANCE,
        lift_height=PICKUP_LIFT_HEIGHT,
        hand_interp_steps=PICKUP_HAND_INTERP_STEPS,
        approach_direction=torch.as_tensor(
            [0.0, -math.sqrt(0.5), -math.sqrt(0.5)], dtype=torch.float32
        ),
        downstream_object_target_poses=(assemble_object_target_pose,),
    )
    # Step 2 - the left arm places the can directly above the cube.
    place_options = PlaceOptions(
        lift_height=PLACE_LIFT_HEIGHT,
        hand_interp_steps=PLACE_HAND_INTERP_STEPS,
    )
    engine = AtomicActionEngine(
        motion_generator=motion_gen,
        control_profiles={
            "left_hand": ControlPartCommandProfile.joint_positions(
                open=left_open,
                grasp=left_close,
            )
        },
    )
    wait_for_user = prepare_tutorial_scene(
        sim, args, "Inspect the scene, then press Enter to plan PickUp -> Place..."
    )
    # Let the staged can settle into a stable pose before planning.
    for _ in range(20):
        sim.update(step=10)

    assemble_affordance = AssembleAffordance(
        base_object_label="cube",
        base_object_entity=cube,
        assemble_object_label="soda_can",
        assemble_object_entity=can,
        assemble_to_base_pose=assemble_to_base,
    )
    binding = ActionBinding(
        manipulators={"primary": "left_arm"},
        end_effectors={"primary": "left_hand"},
    )
    compiled = engine.compile(
        (
            ActionInvocation(
                "pick_up",
                GraspGoal(can_semantics),
                binding,
                MotionPolicy(sample_count=PICKUP_SAMPLE_INTERVAL),
                skill_options=pick_up_options,
            ),
            ActionInvocation(
                "place",
                AssembleGoal(affordance=assemble_affordance),
                binding,
                MotionPolicy(sample_count=PLACE_SAMPLE_INTERVAL),
                skill_options=place_options,
            ),
        )
    )
    success = compiled.plan_success
    traj = compiled.trajectory.positions

    if not success.all():
        logger.log_warning("Failed to plan the assemble demo trajectory.")
        return

    if args.diagnose_plan:
        logger.log_info(f"Planned full trajectory with {traj.shape[1]} waypoints.")
        return

    if wait_for_user:
        input("Press Enter to execute the assembly...")

    replay_trajectory(
        sim,
        robot,
        traj,
        args,
        video_prefix="assemble_auto_play",
        hold_steps=0,
        trajectory_sim_steps=TRAJECTORY_SIM_STEPS,
        look_at=ASSEMBLE_RECORD_LOOK_AT,
    )
    if wait_for_user:
        input("Press Enter to exit the simulation...")


def main() -> None:
    """Run the dual-arm object assembly demo."""
    args = parse_arguments()
    sim = create_tutorial_simulation(
        args,
        arena_space=3.0,
        light_pos=(0.0, -0.4, 3.0),
    )
    robot = create_dual_ur5_robot(sim)
    try:
        run_assemble_demo(args, sim, robot)
        serve_tutorial_scene(sim, args)
    finally:
        sim.destroy()


if __name__ == "__main__":
    main()
