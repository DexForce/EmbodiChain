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

"""Pick up a cube from the side, then rotate it with the Pour action."""

from __future__ import annotations

import argparse
import math

import torch

from embodichain.lab.sim.atomic_actions import (
    ControlPartCommandProfile,
    create_simulation_atomic_action_engine,
    GraspGoal,
    MotionPolicy,
    PickUpOptions,
    PourGoal,
    PourOptions,
)
from embodichain.lab.sim.motion.planners import (
    ToppraPlanOptions,
    TrajectorySampleMethod,
)
from embodichain.utils import logger
from embodichain.lab.scripts._atomic_action.axis_align import (
    OBJECT_SIZE,
    create_align_object,
    create_axis_align_semantics,
)
from embodichain.lab.scripts._atomic_action.support import (
    DEFAULT_GRIPPER_CLOSE_QPOS,
    DEXUNI_CUBE_CLOSE_QPOS,
    create_affordance_sampling_context,
    add_tutorial_robot,
    create_parallel_jaw_grasp_pose_generator,
    create_toppra_motion_generator,
    create_tutorial_argument_parser,
    create_tutorial_simulation,
    draw_axis_marker,
    get_hand_open_close_qpos,
    initialize_pre_pick_robot_pose,
    make_clear_dynamics_callback,
    prepare_tutorial_scene,
    replay_trajectory,
    run_tutorial,
    TutorialPlanner,
)

POUR_INTERNAL_AXIS = (1.0, 0.0, 0.0)
APPROACH_DIRECTION = (-0.707, 0, -0.707)
PICK_SAMPLE_INTERVAL = 120
POUR_SAMPLE_INTERVAL = 80
HAND_INTERP_STEPS = 12
POST_TRAJECTORY_STEPS = 240
PICK_MOTION_SAMPLE_COUNT = PICK_SAMPLE_INTERVAL - HAND_INTERP_STEPS
OBJ_POSITION = (-0.5, 0.0, 0.5 * OBJECT_SIZE[2])
RECORD_LOOK_AT = (
    (-1.5, 0.2, 1.2),
    (-0.4, 0.0, 0.4),
    (0.0, 0.0, 1.0),
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the Pour tutorial."""
    parser = create_tutorial_argument_parser(
        "Pick up a cube horizontally, then pour it about a local axis.",
        features=("grasp_sampling", "visualize_axes"),
    )
    parser.add_argument(
        "--rotate_angle",
        type=float,
        default=math.pi / 4.0,
        help="Signed pouring rotation in radians.",
    )
    return parser.parse_args()


def _create_pick_motion_policy(
    planner: TutorialPlanner = "trapezoidal",
    *,
    fixed_sample_count: bool = True,
) -> MotionPolicy:
    """Build a PickUp policy with optional legacy quantity sampling.

    The configured host lets the planner retain its required waypoints and
    relies on Atomic control-grid retiming instead of imposing a smaller count.
    """
    if planner == "toppra":
        plan_opts = ToppraPlanOptions(
            sample_method=TrajectorySampleMethod.QUANTITY,
            sample_interval=PICK_MOTION_SAMPLE_COUNT,
        )
    elif planner in ("trapezoidal", "curobo"):
        # Retain native waypoints before PickUp resamples its motion segments.
        plan_opts = None
    else:
        raise ValueError(f"Unsupported tutorial planner {planner!r}.")
    return MotionPolicy(
        strategy="motion_gen",
        sample_count=PICK_SAMPLE_INTERVAL,
        plan_opts=plan_opts if fixed_sample_count else None,
    )


def run(args: argparse.Namespace) -> None:
    """Plan and replay PickUp followed by Pour.

    Args:
        args: Resolved standalone or configured host arguments.
    """
    sim = create_tutorial_simulation(
        args,
        newton_solver_options=(
            {
                "vbd_options": {"rigid_contact_history": True},
                "collision_options": {"contact_matching": "latest"},
            }
            if args.physics == "newton"
            else None
        ),
    )
    use_vbd_grasp = sim.physics.solver_type == "dexuni"
    robot = add_tutorial_robot(
        sim,
        "ur5",
        tcp_point="grasp",
        newton_gripper_friction=4.0 if use_vbd_grasp else None,
    )
    obj_position = list(OBJ_POSITION)
    if use_vbd_grasp:
        obj_position[2] += 0.001
    obj = create_align_object(sim, obj_position=obj_position)
    hand_open, hand_close = get_hand_open_close_qpos(
        robot,
        close_qpos=(
            DEXUNI_CUBE_CLOSE_QPOS if use_vbd_grasp else DEFAULT_GRIPPER_CLOSE_QPOS
        ),
    )
    initialize_pre_pick_robot_pose(robot, obj, hand_open)
    planner = getattr(args, "planner", "trapezoidal")
    motion_gen = create_toppra_motion_generator(robot, planner=planner)

    engine = create_simulation_atomic_action_engine(
        motion_generator=motion_gen,
        scene_entities=(obj,),
        control_profiles={
            "eef": ControlPartCommandProfile.joint_positions(
                open=hand_open,
                grasp=hand_close,
            )
        },
        grasp_pose_generators={
            "eef": create_parallel_jaw_grasp_pose_generator(
                n_sample=args.n_sample,
                force_refresh=args.force_reannotate,
            )
        },
    )
    semantics = create_axis_align_semantics(obj, POUR_INTERNAL_AXIS)
    if not args.no_vis_eef_axis:
        draw_axis_marker(sim, "pour_object_axis", obj.get_local_pose(to_matrix=True))
    wait_for_user = prepare_tutorial_scene(
        sim,
        args,
        "Inspect the cube, then press Enter to plan PickUp followed by Pour...",
    )

    control_parts = {"primary": {"motion": "arm", "grasp": "eef"}}
    compiled = engine.compile(
        (
            engine.make_invocation(
                "pick_up",
                GraspGoal(semantics),
                control_parts=control_parts,
                motion_policy=_create_pick_motion_policy(
                    planner,
                    fixed_sample_count=not getattr(args, "_configured_run", False),
                ),
                skill_options=PickUpOptions(
                    approach_direction=torch.tensor(
                        APPROACH_DIRECTION,
                        dtype=torch.float32,
                        device=sim.device,
                    ),
                    pre_grasp_distance=0.15,
                    lift_height=0.16,
                    hand_interp_steps=HAND_INTERP_STEPS,
                ),
            ),
            engine.make_invocation(
                "pour",
                PourGoal(),
                control_parts=control_parts,
                motion_policy=MotionPolicy(
                    strategy="motion_gen",
                    sample_count=POUR_SAMPLE_INTERVAL,
                ),
                skill_options=PourOptions(rotate_angle=args.rotate_angle),
            ),
        ),
        engine.initial_context(
            control_dt=sim.sim_config.physics_dt,
            **(
                {"affordance_sampling": create_affordance_sampling_context(args)}
                if getattr(args, "_configured_run", False)
                else {}
            ),
        ),
    )
    if not compiled.plan_success.all():
        if getattr(args, "_configured_run", False):
            raise RuntimeError(
                f"Atomic planning failed: {compiled.plan_success.tolist()}"
            )
        failed = [
            f"{plan.skill_id}: {plan.diagnostics.messages or ('planning failed',)}"
            for plan in compiled.action_plans
            if not plan.plan_success.all()
        ]
        logger.log_warning(
            "Failed to plan PickUp followed by Pour: " + "; ".join(failed)
        )
        return

    if wait_for_user:
        input("Press Enter to replay the PickUp + Pour trajectory...")
    clear_after_step = compiled.segment(0, "lift").start
    replay_trajectory(
        sim,
        robot,
        compiled.trajectory,
        args,
        video_prefix="pour_cube_auto_play",
        hold_steps=POST_TRAJECTORY_STEPS,
        on_trajectory_step=make_clear_dynamics_callback(obj, clear_after_step),
        look_at=RECORD_LOOK_AT,
        compiled=compiled,
    )
    if wait_for_user:
        input("Press Enter to exit the simulation...")


def main() -> None:
    """Run the standalone example with its original arguments."""
    run(parse_arguments())


if __name__ == "__main__":
    run_tutorial(main)


__all__: list[str] = []
