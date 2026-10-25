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

"""Reuse the same Atomic Skills with six robots and interchangeable grippers.

Change --robot and --gripper; neither task recipe branches on hardware.
Physical calibration is owned by examples/sim/atomic_action/embodiments.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
for _source in (_REPO_ROOT, _REPO_ROOT / "extensions"):
    if str(_source) not in sys.path:
        sys.path.insert(0, str(_source))

import torch
from embodichain.utils import logger, set_seed

from embodichain.lab.sim.atomic_actions import (
    ActionInvocation,
    AssembleAffordance,
    AssembleGoal,
    AtomicActionEngine,
    GraspGoal,
    HandOverGoal,
    HandOverOptions,
    MotionPolicy,
    ObjectSemantics,
    PickUpOptions,
    PlaceOptions,
    SceneEntityPose,
    create_simulation_atomic_action_engine,
)
from examples.sim.atomic_action.embodiments import (
    EmbodimentCfg,
    MeasuredTaskCheck,
    create_embodiment,
    create_scene,
    parse_arguments,
)
from examples.sim.atomic_action.support import (
    create_antipodal_semantics,
    create_parallel_jaw_grasp_pose_generator,
    create_tutorial_motion_generator,
    create_tutorial_simulation,
    prepare_tutorial_scene,
    publish_tutorial_scene,
    replay_trajectory,
    run_tutorial,
    serve_tutorial_scene,
)


def pick_place(
    engine: AtomicActionEngine,
    semantics: ObjectSemantics,
    cfg: EmbodimentCfg,
    target: torch.Tensor,
    place_goal: AssembleGoal,
) -> tuple[ActionInvocation, ...]:
    """Describe the object task using one logical manipulation participant."""
    policy = MotionPolicy(strategy="motion_gen", sample_count=120)
    return (
        engine.make_invocation(
            "pick_up",
            GraspGoal(semantics),
            control_parts=cfg.binding,
            motion_policy=policy,
            skill_options=PickUpOptions(
                pre_grasp_distance=0.10,
                lift_height=0.12,
                hand_interp_steps=60,
                grasp_settle_steps=30,
                approach_alignment_max_angle=0.175,
                downstream_object_target_poses=(target,),
            ),
        ),
        engine.make_invocation(
            "place",
            place_goal,
            control_parts=cfg.binding,
            motion_policy=policy,
            skill_options=PlaceOptions(
                lift_height=0.10, hand_interp_steps=60, release_settle_steps=30
            ),
        ),
    )


def handover(
    engine: AtomicActionEngine,
    semantics: ObjectSemantics,
    cfg: EmbodimentCfg,
    target: torch.Tensor,
    place_goal: AssembleGoal,
) -> tuple[ActionInvocation, ...]:
    """Transfer between explicit roles, then place with the receiving resource."""
    policy = MotionPolicy(strategy="motion_gen", sample_count=480)
    return (
        engine.make_invocation(
            "hand_over",
            HandOverGoal(semantics, target_pose=target),
            control_parts=cfg.binding,
            motion_policy=policy,
            skill_options=HandOverOptions(
                arm_selection="bound",
                release_at_target=False,
                pre_grasp_distance=0.08,
                lift_height=cfg.handover_lift_height,
                retreat_distance=0.03,
                hand_interp_steps=60,
                hold_steps=20,
                retreat_steps=30,
            ),
        ),
        engine.make_invocation(
            "place",
            place_goal,
            control_parts={"primary": cfg.binding["destination"]},
            motion_policy=policy,
            skill_options=PlaceOptions(
                lift_height=0.10, hand_interp_steps=60, release_settle_steps=30
            ),
        ),
    )


def main() -> None:
    """Resolve hardware, bind resources, compile goals, and observe playback."""
    args = parse_arguments()
    set_seed(0)
    logger.log_info(
        f"Selected task={args.task}, robot={args.robot}, gripper={args.gripper}"
    )
    cfg = create_embodiment(args)
    sim = create_tutorial_simulation(args, arena_space=3.0)
    robot, obj, pad = create_scene(sim, cfg)
    publish_tutorial_scene(sim, args)

    # 1. Hardware bindings and named jaw commands form the skill interface.
    engine = create_simulation_atomic_action_engine(
        motion_generator=create_tutorial_motion_generator(robot, args.planner),
        scene_entities=(obj, pad),
        control_profiles=cfg.command_profiles(robot.device),
        grasp_pose_generators={
            part: create_parallel_jaw_grasp_pose_generator(
                n_sample=args.n_sample,
                force_refresh=args.force_reannotate,
                gripper_model=model,
                approach_direction_samples=1,
            )
            for part, model in cfg.models.items()
        },
    )

    # 2. The destination describes an OBJECT pose relative to its support pad.
    # Place resolves the grasp's object_to_eef internally, for every tool TCP.
    relative = torch.eye(4, device=sim.device)
    relative[2, 3] = cfg.destination_half_height + cfg.object_size[2] * 0.5 + 0.001
    place_goal = AssembleGoal(
        affordance=AssembleAffordance(assemble_to_base_pose=relative),
        base_pose=SceneEntityPose(pad.uid),
    )
    target = pad.get_local_pose(to_matrix=True)[0] @ relative
    semantics = create_antipodal_semantics(obj, label="transfer_object")
    recipe = pick_place if args.task == "pick_place" else handover
    invocations = recipe(engine, semantics, cfg, target, place_goal)

    # 3. Compilation predicts attachments for subsequent goals, without stepping
    # physics. Command completion alone does not establish a successful grasp.
    wait = prepare_tutorial_scene(
        sim, args, "Inspect the embodiment, then press Enter..."
    )
    context = engine.initial_context(control_dt=sim.sim_config.physics_dt)
    compiled = engine.compile(invocations, context)
    if not bool(compiled.plan_success.all()):
        logger.log_info(
            'Measured result: {"planning_success": false, "playback_completed": false, "task_success": null}'
        )
        raise RuntimeError(
            f"Planning failed for {args.robot}/{args.gripper}: "
            f"{[p.plan_success.tolist() for p in compiled.action_plans]}"
        )

    # 4. The host executes the commands and reads actual physical observations.
    measured = MeasuredTaskCheck(cfg, robot, obj, compiled)
    x, y, z = cfg.task_origin
    replay_trajectory(
        sim,
        robot,
        compiled.trajectory,
        args,
        video_prefix=f"{args.task}_{args.robot}_{args.gripper}",
        hold_steps=100,
        on_trajectory_step=measured.observe,
        look_at=cfg.preview_look_at
        or ((x - 1.1, y - 1.0, z + 0.9), (x, y, z + 0.1), (0, 0, 1)),
    )
    measured.finish(sim, target)
    if wait:
        input("Press Enter to finish...")
    serve_tutorial_scene(sim, args)


if __name__ == "__main__":
    run_tutorial(main)
