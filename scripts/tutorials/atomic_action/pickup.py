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

"""Demonstrate PickUp, optionally with one distinct affordance grasp per env.

Run in ``embodichain2`` to plan and simulate four independent grasps::

    python scripts/tutorials/atomic_action/pickup.py --headless \
        --n_affordance_multi_gen 4 --affordance_output /tmp/pickup-affordances

The requested count sets the physical environment count. Unsolvable candidates
are filtered and replaced from the remaining sampled grasps when possible.
Only successful plans are exported; unused physical rows hold their initial
state. Simulation replay is a demonstration, not expert-data qualification.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from embodichain.lab.sim.atomic_actions import (
    ControlPartCommandProfile,
    create_simulation_atomic_action_engine,
    GraspGoal,
    PickUpOptions,
    MotionPolicy,
)
from embodichain.lab.sim.cfg import RigidBodyAttributesCfg, RigidObjectCfg
from embodichain.lab.sim.objects import RigidObject
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.utils import logger
from scripts.tutorials.atomic_action.tutorial_utils import (
    add_tutorial_robot,
    clone_local_pose_from_first_env,
    create_antipodal_semantics,
    create_curobo_motion_generator,
    create_toppra_motion_generator,
    create_parallel_jaw_grasp_pose_generator,
    create_tutorial_argument_parser,
    create_tutorial_simulation,
    draw_axis_marker,
    get_hand_open_close_qpos,
    initialize_pre_pick_robot_pose,
    make_clear_dynamics_callback,
    prepare_tutorial_scene,
    replay_trajectory,
    run_tutorial,
)

OBJECT_SIZE = (0.05, 0.05, 0.05)
OBJECT_XY = (-0.42, -0.08)
PICK_SAMPLE_INTERVAL = 120
HAND_INTERP_STEPS = 12
POST_TRAJECTORY_STEPS = 240
APPROACH_DIRECTIONS = {
    "top": (0.0, 0.0, -1.0),
    "side": (0.0, 1.0, 0.0),
    "side_y": (0.0, -1.0, 0.0),
}


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the PickUp tutorial."""
    parser = create_tutorial_argument_parser(
        "Demonstrate PickUp on a cube.",
        features=("grasp_sampling", "visualize_axes", "affordance_multi_gen"),
    )
    parser.add_argument(
        "--approach", choices=[*APPROACH_DIRECTIONS, "custom"], default="top"
    )
    parser.add_argument("--custom_approach_direction", type=float, nargs=3)
    return parser.parse_args()


def create_pick_object(sim) -> RigidObject:
    """Create a settled cube for antipodal grasp planning."""
    obj = sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="cube",
            shape=CubeCfg(size=list(OBJECT_SIZE)),
            attrs=RigidBodyAttributesCfg(
                mass=0.05,
                dynamic_friction=0.97,
                static_friction=0.99,
            ),
            max_convex_hull_num=16,
            init_pos=[*OBJECT_XY, OBJECT_SIZE[2]],
        )
    )
    sim.update(step=10)
    clone_local_pose_from_first_env(obj)
    obj.clear_dynamics()
    return obj


def resolve_approach_direction(
    args: argparse.Namespace, device: torch.device
) -> torch.Tensor:
    """Resolve and validate a normalized approach direction."""
    direction = (
        args.custom_approach_direction
        if args.approach == "custom"
        else APPROACH_DIRECTIONS[args.approach]
    )
    if direction is None:
        raise ValueError(
            "--custom_approach_direction is required for --approach custom."
        )
    approach = torch.tensor(direction, dtype=torch.float32, device=device)
    if torch.linalg.norm(approach) < 1e-6:
        raise ValueError("approach_direction must be non-zero.")
    return torch.nn.functional.normalize(approach, dim=0)


def main() -> None:
    """Plan and replay a sampled antipodal PickUp trajectory."""
    args = parse_arguments()
    sim = create_tutorial_simulation(args)
    robot = add_tutorial_robot(sim, args.robot, tcp_z=0.15)
    obj = create_pick_object(sim)
    hand_open, hand_close = get_hand_open_close_qpos(robot)
    initialize_pre_pick_robot_pose(robot, obj, hand_open)
    multi_count = getattr(args, "n_affordance_multi_gen", None)
    motion_gen = (
        create_toppra_motion_generator(robot)
        if multi_count is not None
        else create_curobo_motion_generator(robot)
    )
    grasp_generator = create_parallel_jaw_grasp_pose_generator(
        n_sample=args.n_sample,
        force_refresh=args.force_reannotate,
        max_candidates=max(32, 4 * multi_count) if multi_count is not None else None,
    )

    engine = create_simulation_atomic_action_engine(
        motion_generator=motion_gen,
        scene_entities=(obj,),
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=hand_open,
                grasp=hand_close,
            )
        },
        grasp_pose_generators={"hand": grasp_generator},
    )
    semantics = create_antipodal_semantics(
        obj,
        label="cube",
    )
    if not args.no_vis_eef_axis:
        draw_axis_marker(sim, "pickup_object_axis", obj.get_local_pose(to_matrix=True))
    wait_for_user = prepare_tutorial_scene(
        sim, args, "Inspect the cube, then press Enter to plan PickUp..."
    )

    if multi_count is not None:
        try:
            _run_affordance_pickup(
                sim, robot, obj, engine, grasp_generator, semantics, args, wait_for_user
            )
        finally:
            motion_gen.planner.close()
        return

    compiled = engine.compile(
        (
            engine.make_invocation(
                "pick_up",
                GraspGoal(semantics),
                control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
                motion_policy=MotionPolicy(
                    strategy="motion_gen",
                    sample_count=PICK_SAMPLE_INTERVAL,
                ),
                skill_options=PickUpOptions(
                    approach_direction=resolve_approach_direction(args, sim.device),
                    pre_grasp_distance=0.15,
                    lift_height=0.16,
                    hand_interp_steps=HAND_INTERP_STEPS,
                ),
            ),
        ),
        engine.initial_context(control_dt=sim.sim_config.physics_dt),
    )
    if not compiled.plan_success.all():
        logger.log_warning("Failed to plan PickUp demo trajectory.")
        return

    if wait_for_user:
        input("Press Enter to replay the PickUp demo...")
    clear_after_step = compiled.segment(0, "lift").start
    replay_trajectory(
        sim,
        robot,
        compiled.trajectory,
        args,
        video_prefix="pickup_cube_auto_play",
        hold_steps=POST_TRAJECTORY_STEPS,
        on_trajectory_step=make_clear_dynamics_callback(obj, clear_after_step),
    )
    if wait_for_user:
        input("Press Enter to exit the simulation...")


def _run_affordance_pickup(
    sim, robot, obj, engine, grasp_generator, semantics, args, wait_for_user
) -> None:
    """Select distinct raw grasps, export the compact batch and replay real rows."""
    from embodichain.lab.trajectory_generation.integrations.atomic_affordance import (
        plan_affordance_batch,
    )
    from scripts.tutorials.atomic_action.affordance_utils import (
        sample_affordance_grasps,
        save_affordance_result,
    )

    direction = resolve_approach_direction(args, sim.device)
    grasps = sample_affordance_grasps(
        grasp_generator,
        semantics.affordance,
        object_poses=obj.get_local_pose(to_matrix=True),
        approach_direction=direction,
        seed=args.affordance_seed,
    )
    invocation = engine.make_invocation(
        "pick_up",
        GraspGoal(semantics),
        invocation_id="affordance_pickup",
        control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
        motion_policy=MotionPolicy(strategy="ik_interp", sample_count=240),
        skill_options=PickUpOptions(
            approach_direction=direction,
            pre_grasp_distance=0.15,
            lift_height=0.16,
            hand_interp_steps=HAND_INTERP_STEPS,
            grasp_settle_steps=20,
        ),
    )
    result = plan_affordance_batch(
        engine,
        invocation,
        engine.initial_context(control_dt=sim.sim_config.physics_dt),
        grasps,
    )
    save_affordance_result(
        result,
        output_dir=args.affordance_output,
        name="pickup",
        physical_envs=robot.num_instances,
        metadata={"seed": args.affordance_seed, "approach": args.approach},
    )
    if not bool(result.success_mask.any()):
        logger.log_warning(
            "No feasible affordance PickUp trajectories; replay skipped."
        )
        return
    if wait_for_user:
        input("Press Enter to replay the affordance PickUp trajectories...")
    replay_trajectory(
        sim,
        robot,
        result.trajectory,
        args,
        video_prefix="pickup_affordance_multi_gen",
        hold_steps=POST_TRAJECTORY_STEPS,
    )
    if wait_for_user:
        input("Press Enter to exit the simulation...")


if __name__ == "__main__":
    run_tutorial(main)
