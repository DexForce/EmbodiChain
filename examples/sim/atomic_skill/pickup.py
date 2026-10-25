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

"""Demonstrate PickUp on a cube with a configurable approach direction."""

from __future__ import annotations

import argparse

import torch

from embodichain.lab.sim.atomic_skills import (
    ControlPartCommandProfile,
    create_simulation_atomic_skill_engine,
    GraspGoal,
    PickUpOptions,
    MotionPolicy,
)
from embodichain.lab.sim.cfg import RigidObjectCfg
from embodichain.lab.sim.objects import RigidObject
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.utils import logger
from examples.sim.atomic_skill.support import (
    add_tutorial_robot,
    clone_local_pose_from_first_env,
    create_affordance_sampling_context,
    create_antipodal_semantics,
    create_curobo_motion_generator,
    create_parallel_jaw_grasp_pose_generator,
    create_tutorial_argument_parser,
    create_tutorial_rigid_body_physics,
    create_tutorial_simulation,
    draw_axis_marker,
    get_hand_open_close_qpos,
    initialize_pre_pick_robot_pose,
    log_affordance_branch_diagnostics,
    make_clear_dynamics_callback,
    parse_affordance_sampling_arguments,
    prepare_tutorial_scene,
    replay_trajectory,
    _compile_demo,
    run_tutorial,
)
from examples.sim.atomic_skill.support import (
    compute_pick_close_end_step,
    initialize_benchmark_simulation,
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
        features=("affordance_sampling", "grasp_sampling", "visualize_axes"),
    )
    parser.add_argument(
        "--approach", choices=[*APPROACH_DIRECTIONS, "custom"], default="top"
    )
    parser.add_argument("--custom_approach_direction", type=float, nargs=3)
    return parse_affordance_sampling_arguments(parser)


def create_pick_object(sim) -> RigidObject:
    """Create a settled cube for antipodal grasp planning."""
    obj = sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="cube",
            shape=CubeCfg(size=list(OBJECT_SIZE)),
            attrs=create_tutorial_rigid_body_physics(
                mass=0.05,
                dynamic_friction=0.97,
                static_friction=0.99,
                newton_contact=sim.is_newton_backend,
            ),
            init_pos=[*OBJECT_XY, OBJECT_SIZE[2]],
        )
    )
    sim.prepare()
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


def run(args: argparse.Namespace) -> None:
    """Plan and replay a sampled antipodal PickUp trajectory.

    Args:
        args: Resolved standalone or configured host arguments.
    """
    sampling = create_affordance_sampling_context(args)
    sim = create_tutorial_simulation(args)
    robot = add_tutorial_robot(sim, args.robot, tcp_point="grasp")
    obj = create_pick_object(sim)
    sim.prepare()
    hand_open, hand_close = get_hand_open_close_qpos(robot)
    initialize_pre_pick_robot_pose(robot, obj, hand_open)
    motion_gen = create_curobo_motion_generator(
        robot,
        use_cuda_graph=args.physics != "newton",
        planner=getattr(args, "planner", "trapezoidal"),
    )
    engine = create_simulation_atomic_skill_engine(
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
    if not args.no_vis_eef_axis:
        draw_axis_marker(sim, "pickup_object_axis", obj.get_local_pose(to_matrix=True))
    wait_for_user = prepare_tutorial_scene(
        sim, args, "Inspect the cube, then press Enter to plan PickUp..."
    )

    compiled = _compile_demo(
        engine,
        args,
        (
            engine.make_invocation(
                "pick_up",
                GraspGoal(semantics),
                control_parts={"primary": {"motion": "arm", "grasp": "eef"}},
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
        engine.initial_context(
            control_dt=sim.sim_config.physics_dt,
            affordance_sampling=sampling,
        ),
    )
    log_affordance_branch_diagnostics(compiled.skill_plans[0])
    if not compiled.plan_success.all():
        if getattr(args, "run", None) is not None:
            raise RuntimeError(
                f"Atomic planning failed: {compiled.plan_success.tolist()}"
            )
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
    """Add the default PickUp tutorial robot."""
    robot = add_tutorial_robot(sim, "ur5", tcp_point="grasp")
    # DexSim binds body_data (and therefore get_qpos) only after prepare().
    sim.prepare()
    return robot


__all__: list[str] = []
