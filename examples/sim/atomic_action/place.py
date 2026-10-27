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

"""Demonstrate Place after a PickUp precondition has created held-object state."""

from __future__ import annotations

import argparse
import math

import torch

from embodichain.lab.sim.atomic_actions import (
    ControlPartCommandProfile,
    create_simulation_atomic_action_engine,
    GraspGoal,
    PickUpOptions,
    PlaceGoal,
    PlaceOptions,
    MotionPolicy,
)
from embodichain.lab.sim.cfg import RigidObjectCfg
from embodichain.lab.sim.objects import RigidObject
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.utils import logger
from examples.sim.atomic_action.support import (
    DEFAULT_GRIPPER_CLOSE_QPOS,
    DEXUNI_CUBE_CLOSE_QPOS,
    create_affordance_sampling_context,
    add_tutorial_robot,
    broadcast_pose_batch,
    broadcast_waypoint_pose_batch,
    clone_local_pose_from_first_env,
    create_antipodal_semantics,
    create_curobo_motion_generator,
    create_parallel_jaw_grasp_pose_generator,
    create_tutorial_argument_parser,
    create_tutorial_rigid_body_physics,
    create_tutorial_simulation,
    draw_axis_marker,
    get_hand_open_close_qpos,
    initialize_pre_pick_robot_pose,
    make_clear_dynamics_callback,
    prepare_tutorial_scene,
    expand_tutorial_task_space_variants,
    expand_tutorial_trajectory_variants,
    log_trajectory_variant_diagnostics,
    parse_trajectory_variant_arguments,
    replay_trajectory,
    _compile_demo,
    run_tutorial,
    save_tool_path_view,
    save_variant_joint_plot,
    trajectory_variant_rows,
)
from examples.sim.atomic_action.support import (
    compute_pick_close_end_step,
    initialize_benchmark_simulation,
)

OBJECT_SIZE = (0.05, 0.05, 0.05)
OBJECT_XY = (-0.42, -0.08)
PICK_SAMPLE_INTERVAL = 120
PLACE_SAMPLE_INTERVAL = 120
HAND_INTERP_STEPS = 12
POST_TRAJECTORY_STEPS = 240
PLACE_LIFT_HEIGHT = 0.14


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the Place tutorial."""
    parser = create_tutorial_argument_parser(
        "Pick up a cube and place it at a target pose.",
        features=("grasp_sampling", "trajectory_variants", "visualize_axes"),
    )
    parser.add_argument(
        "--trajectory_strategy",
        choices=("motion_gen", "ik_interp"),
        default="motion_gen",
        help=(
            "Generate the PickUp/Place reference with the selected planner or "
            "bounded IK interpolation (default: motion_gen)."
        ),
    )
    parser.add_argument(
        "--trajectory_control_dt",
        type=float,
        default=None,
        help=(
            "Reference-trajectory command period in seconds; defaults to the "
            "physics period. It must be an integer multiple of physics_dt."
        ),
    )
    args = parse_trajectory_variant_arguments(parser)
    if args.trajectory_control_dt is not None and (
        not math.isfinite(args.trajectory_control_dt) or args.trajectory_control_dt <= 0
    ):
        parser.error("--trajectory_control_dt must be finite and positive.")
    return args


def create_pick_object(sim) -> RigidObject:
    """Create a settled cube for the PickUp and Place sequence."""
    ground_clearance = 0.001 if sim.physics.solver_type == "dexuni" else 0.0
    obj = sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="cube",
            shape=CubeCfg(size=list(OBJECT_SIZE)),
            attrs=create_tutorial_rigid_body_physics(
                mass=0.05,
                dynamic_friction=0.97,
                static_friction=0.99,
                linear_damping=0.2,
                angular_damping=0.2,
                newton_contact=sim.is_newton_backend,
            ),
            init_pos=[*OBJECT_XY, 0.5 * OBJECT_SIZE[2] + ground_clearance],
        )
    )
    sim.prepare()
    sim.update(step=10)
    clone_local_pose_from_first_env(obj)
    obj.clear_dynamics()
    return obj


def make_place_eef_poses(device: torch.device) -> torch.Tensor:
    """Build hover and release waypoints for the multi-waypoint Place target."""
    rotation = torch.tensor(
        [
            [0.0539, 0.9985, -0.0022],
            [0.9977, -0.0540, -0.0401],
            [-0.0401, 0.0, -0.9992],
        ],
        dtype=torch.float32,
        device=device,
    )
    poses = []
    for position in ((-0.40, 0.48, 0.20), (-0.40, 0.48, 0.10)):
        pose = torch.eye(4, dtype=torch.float32, device=device)
        pose[:3, :3], pose[:3, 3] = rotation, torch.tensor(position, device=device)
        poses.append(pose)
    return torch.stack(poses)


def run(args: argparse.Namespace) -> None:
    """Plan and replay PickUp followed by a multi-waypoint Place.

    Args:
        args: Resolved standalone or configured host arguments.
    """
    sim = create_tutorial_simulation(
        args,
        newton_solver_options=(
            {
                "vbd_options": {
                    "rigid_contact_history": True,
                    "rigid_avbd_contact_alpha": 0.5,
                },
                "collision_options": {"contact_matching": "latest"},
            }
            if args.physics == "newton" and args.robot == "ur5"
            else None
        ),
    )
    use_vbd_grasp = args.robot == "ur5" and sim.physics.solver_type == "dexuni"
    robot = add_tutorial_robot(
        sim,
        args.robot,
        tcp_point="grasp",
        newton_gripper_friction=4.0 if use_vbd_grasp else None,
        arm_damping=1000.0 if use_vbd_grasp else None,
    )
    obj = create_pick_object(sim)
    sim.prepare()
    trajectory_strategy = getattr(args, "trajectory_strategy", "motion_gen")
    trajectory_control_dt = getattr(args, "trajectory_control_dt", None)
    physics_dt = float(sim.sim_config.physics_dt)
    if trajectory_control_dt is None:
        trajectory_control_dt = physics_dt
    control_ratio = trajectory_control_dt / physics_dt
    if control_ratio < 1.0 or not math.isclose(
        control_ratio,
        round(control_ratio),
        rel_tol=1e-6,
        abs_tol=1e-8,
    ):
        raise ValueError(
            "trajectory_control_dt must be an integer multiple of physics_dt "
            f"({physics_dt:g}); got {trajectory_control_dt:g}."
        )
    motion_gen = create_curobo_motion_generator(
        robot,
        planner=getattr(args, "planner", "trapezoidal"),
    )
    hand_open, hand_close = get_hand_open_close_qpos(
        robot,
        close_qpos=(
            DEXUNI_CUBE_CLOSE_QPOS if use_vbd_grasp else DEFAULT_GRIPPER_CLOSE_QPOS
        ),
    )
    initialize_pre_pick_robot_pose(robot, obj, hand_open)

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
                approach_direction_samples=getattr(args, "grasp_approach_samples", 4),
            )
        },
    )
    semantics = create_antipodal_semantics(
        obj,
        label="cube",
    )
    place_poses = make_place_eef_poses(sim.device)
    if not args.no_vis_eef_axis:
        draw_axis_marker(
            sim,
            "place_target_axis",
            broadcast_pose_batch(place_poses[-1], robot.get_qpos().shape[0]),
        )
    wait_for_user = prepare_tutorial_scene(
        sim, args, "Inspect the cube, then press Enter to plan PickUp -> Place..."
    )

    endpoint_mapping = {"primary": {"motion": "arm", "grasp": "eef"}}
    compiled = _compile_demo(
        engine,
        args,
        (
            engine.make_invocation(
                "pick_up",
                GraspGoal(semantics),
                control_parts=endpoint_mapping,
                motion_policy=MotionPolicy(
                    strategy=trajectory_strategy,
                    sample_count=PICK_SAMPLE_INTERVAL,
                ),
                skill_options=PickUpOptions(
                    pre_grasp_distance=0.15,
                    lift_height=0.16,
                    hand_interp_steps=HAND_INTERP_STEPS,
                ),
            ),
            engine.make_invocation(
                "place",
                PlaceGoal(
                    broadcast_waypoint_pose_batch(
                        place_poses, robot.get_qpos().shape[0]
                    )
                ),
                control_parts=endpoint_mapping,
                motion_policy=MotionPolicy(
                    strategy=trajectory_strategy,
                    sample_count=PLACE_SAMPLE_INTERVAL,
                ),
                skill_options=PlaceOptions(
                    lift_height=PLACE_LIFT_HEIGHT,
                    hand_interp_steps=HAND_INTERP_STEPS,
                ),
            ),
        ),
        engine.initial_context(
            control_dt=trajectory_control_dt,
            **(
                {"affordance_sampling": create_affordance_sampling_context(args)}
                if getattr(args, "run", None) is not None
                else {}
            ),
        ),
    )
    if not compiled.plan_success.all():
        if getattr(args, "run", None) is not None:
            raise RuntimeError(
                f"Atomic planning failed: {compiled.plan_success.tolist()}"
            )
        logger.log_warning("Failed to plan Place demo trajectory.")
        return

    if getattr(args, "run", None) is None:
        # Each row replays a different way of executing the same plan. Only the
        # free motion changes: the grasp and the release keep their planned
        # waypoints, so every row picks and places the cube identically.
        phase_kinds = (
            {"approach": "free", "close": "contact", "lift": "free"},
            {"approach": "free", "release": "contact", "retract": "free"},
        )
        if args.trajectory_variant_mode == "task_space":
            variants = expand_tutorial_task_space_variants(
                compiled,
                robot,
                args,
                phase_kinds=phase_kinds,
                # Keep the pre-grasp approach nominal. The visible variants bend
                # the held-object lift/transport and post-release retract while
                # retaining the grasp and placement endpoints exactly.
                editable=(("lift",), ("approach", "retract")),
            )
        else:
            variants = expand_tutorial_trajectory_variants(
                compiled,
                robot,
                args,
                phase_kinds=phase_kinds,
                # Retiming anything before the lift would move the step index at
                # which the executor clears the cube's dynamics.
                retimable=(("lift",), ("approach", "retract")),
            )
        log_trajectory_variant_diagnostics(variants, args.trajectory_variants)

        if args.variant_plot_dir:
            # Capture before replay so the still shows the starting scene and the
            # paths the variants are about to take.
            save_variant_joint_plot(
                variants, robot, f"{args.variant_plot_dir}/place_variant_joints.png"
            )
            save_tool_path_view(
                robot,
                variants,
                f"{args.variant_plot_dir}/place_variant_tool_paths.png",
                target_object=obj,
                object_size=OBJECT_SIZE,
            )

    if wait_for_user:
        input("Press Enter to replay the Place demo...")
    clear_after_step = compiled.segment(0, "lift").start
    replay_trajectory(
        sim,
        robot,
        (
            compiled.trajectory
            if getattr(args, "run", None) is not None
            else trajectory_variant_rows(variants, args.num_envs)
        ),
        args,
        video_prefix="place_auto_play",
        hold_steps=POST_TRAJECTORY_STEPS,
        on_trajectory_step=make_clear_dynamics_callback(obj, clear_after_step),
        compiled=compiled,
    )
    if wait_for_user:
        input("Press Enter to exit the simulation...")


def main() -> None:
    """Run the standalone example with its original arguments."""
    run(parse_arguments())


if __name__ == "__main__":
    run_tutorial(main)


def initialize_simulation(args) -> "SimulationManager":
    """Create the tutorial simulation for interactive or benchmark runs."""
    return initialize_benchmark_simulation(args)


def create_robot(sim: "SimulationManager") -> "Robot":
    """Add the default Place tutorial robot."""
    robot = add_tutorial_robot(sim, "ur5", tcp_point="grasp")
    # DexSim binds body_data (and therefore get_qpos) only after prepare().
    sim.prepare()
    return robot


__all__: list[str] = []
