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

"""Use semantic skills to pick a registered cube and place it at an object pose.

Unlike the direct atomic-action tutorial, this example never names ``arm`` or
``hand`` in the workflow. The scene registry owns object identity, the robot
profile owns embodiment-specific resources, and :class:`SkillRuntime`
lowers each call from fresh observations, executes it, and commits only
verified effects.
"""

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
    EffectVerificationRequest,
    MotionPolicy,
    PlanningContext,
    RecoveryPolicy,
    TrackingPolicy,
)
from embodichain.lab.sim.objects import RigidObject, Robot
from embodichain.lab.sim.skills import (
    Pick,
    Place,
    ResourceBinding,
    RobotSkillProfile,
    SceneObjectRef,
    SemanticEffectVerifier,
    SemanticPose,
    SkillRuntime,
    SkillPolicyPreset,
)
from embodichain.utils import logger
from scripts.tutorials.atomic_action.place import create_pick_object
from scripts.tutorials.atomic_action.tutorial_utils import (
    add_tutorial_robot,
    broadcast_pose_batch,
    create_antipodal_semantics,
    create_curobo_motion_generator,
    create_parallel_jaw_grasp_pose_generator,
    create_tutorial_argument_parser,
    create_tutorial_simulation,
    draw_axis_marker,
    get_hand_open_close_qpos,
    initialize_pre_pick_robot_pose,
    prepare_tutorial_scene,
    run_tutorial,
    serve_tutorial_scene,
    start_auto_play_recording,
    stop_auto_play_recording,
)
from scripts.tutorials.semantic_skill.tutorial_utils import (
    compile_semantic_workflow_for_diagnostics,
    create_graspable_object_registry,
    create_manipulator_resource,
    create_runtime_step_observer,
    joint_target_error,
    object_to_eef_translation_error,
)

OBJECT_ID = "workpiece"
OBJECT_SIMULATION_UID = "cube"
TARGET_OBJECT_POSITION = (-0.40, 0.48, 0.025)
TARGET_OBJECT_QUATERNION_WXYZ = (1.0, 0.0, 0.0, 0.0)
PICK_SAMPLE_COUNT = 120
PLACE_SAMPLE_COUNT = 120
TRAJECTORY_SIM_STEPS = 4
TRACKING_ERROR_THRESHOLD = 0.25
MINIMUM_PICK_LIFT = 0.08
MAXIMUM_HELD_RELATION_ERROR = 0.05
MAXIMUM_PLACE_POSITION_ERROR = 0.05
MAXIMUM_OPEN_HAND_ERROR = 0.02
POST_EXECUTION_UPDATES = 240


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the semantic Place tutorial."""
    parser = create_tutorial_argument_parser(
        "Execute and verify semantic Pick -> Place through the skill runtime.",
        features=("diagnose_plan", "grasp_sampling", "visualize_axes"),
    )
    return parser.parse_args()


def create_robot_profile(
    hand_open: torch.Tensor,
    hand_grasp: torch.Tensor,
) -> RobotSkillProfile:
    """Declare how semantic manipulation maps onto the tutorial robot.

    Args:
        hand_open: Joint positions for the semantic ``open`` command.
        hand_grasp: Joint positions for the semantic ``grasp`` command.

    Returns:
        A profile with one manipulation resource and per-skill policies.
    """
    manipulator_id = "primary_manipulator"
    return RobotSkillProfile(
        profile_id="tutorial.single_arm",
        resources={
            manipulator_id: create_manipulator_resource(
                manipulator_id,
                motion_control_part="arm",
                grasp_control_part="hand",
            )
        },
        command_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=hand_open,
                grasp=hand_grasp,
            )
        },
        defaults={
            "pick_up": ResourceBinding({"primary": manipulator_id}),
            "place": ResourceBinding({"primary": manipulator_id}),
        },
        presets={
            "pick": SkillPolicyPreset(
                "pick",
                motion_policy=MotionPolicy(
                    strategy="motion_gen",
                    sample_count=PICK_SAMPLE_COUNT,
                ),
                # The PGI gripper closes in five interpolated commands. Its
                # position controller can legitimately trail one command by
                # more than the generic 0.05-rad threshold.
                tracking_policy=TrackingPolicy.joint_position(
                    in_flight_max_abs_error=TRACKING_ERROR_THRESHOLD,
                    terminal_max_abs_error=TRACKING_ERROR_THRESHOLD,
                ),
            ),
            "place": SkillPolicyPreset(
                "place",
                motion_policy=MotionPolicy(
                    strategy="motion_gen",
                    sample_count=PLACE_SAMPLE_COUNT,
                ),
                # A failed release is not safely repeatable without first
                # reconciling the physical object state.
                tracking_policy=TrackingPolicy.joint_position(
                    in_flight_max_abs_error=TRACKING_ERROR_THRESHOLD,
                    terminal_max_abs_error=TRACKING_ERROR_THRESHOLD,
                ),
                recovery_policy=RecoveryPolicy(max_action_retries=0),
            ),
        },
        default_preset="pick",
        skill_presets={"pick_up": "pick", "place": "place"},
    )


def create_place_task() -> tuple[Pick, Place]:
    """Declare the robot-independent calls submitted at the application entry."""
    object_ref = SceneObjectRef(OBJECT_ID)
    return (
        Pick(object=object_ref),
        Place(
            object=object_ref,
            at=SemanticPose(
                TARGET_OBJECT_POSITION,
                TARGET_OBJECT_QUATERNION_WXYZ,
            ),
        ),
    )


def create_place_effect_verifier(
    obj: RigidObject,
    robot: Robot,
    hand_open: torch.Tensor,
) -> SemanticEffectVerifier:
    """Create physical Pick and Place verification for the live tutorial scene.

    Args:
        obj: Cube manipulated by the workflow.
        robot: Robot executing the semantic calls.
        hand_open: Joint target representing a released object.

    Returns:
        Runtime callback producing a boolean result per environment.
    """
    initial_pose = obj.get_local_pose(to_matrix=True)
    if not isinstance(initial_pose, torch.Tensor) or initial_pose.dim() != 3:
        raise ValueError("The tutorial object pose must have shape (B, 4, 4).")
    initial_height = initial_pose[:, 2, 3].clone()
    target_position = torch.tensor(TARGET_OBJECT_POSITION, dtype=torch.float32)

    def verify(
        call: object,
        request: EffectVerificationRequest,
        context: PlanningContext,
    ) -> torch.Tensor:
        object_position = obj.get_local_pose(to_matrix=True)[:, :3, 3]
        if type(call) is Pick and request.skill_id == "pick_up":
            lift = object_position[:, 2] - initial_height.to(object_position.device)
            held = request.expected_effects.held_object_updates.get("arm")
            if held is None:
                raise RuntimeError("Pick verification requires an arm attachment.")
            held_error = object_to_eef_translation_error(
                obj,
                robot,
                motion_control_part="arm",
                expected_object_to_eef=held.object_to_eef,
            )
            success = (lift >= MINIMUM_PICK_LIFT) & (
                held_error <= MAXIMUM_HELD_RELATION_ERROR
            )
            logger.log_info(
                "Semantic Pick verification: "
                f"lift={lift.detach().cpu().tolist()} m, "
                "object-to-EEF translation error="
                f"{held_error.detach().cpu().tolist()} m, "
                f"success={success.detach().cpu().tolist()}."
            )
        elif type(call) is Place and request.skill_id == "place":
            position_error = torch.linalg.vector_norm(
                object_position
                - target_position.to(
                    device=object_position.device,
                    dtype=object_position.dtype,
                ),
                dim=1,
            )
            hand_error = joint_target_error(
                robot,
                control_part="hand",
                target=hand_open,
            )
            success = (position_error <= MAXIMUM_PLACE_POSITION_ERROR) & (
                hand_error <= MAXIMUM_OPEN_HAND_ERROR
            )
            logger.log_info(
                "Semantic Place verification: "
                f"position_error={position_error.detach().cpu().tolist()} m, "
                f"open_hand_error={hand_error.detach().cpu().tolist()} rad, "
                f"success={success.detach().cpu().tolist()}."
            )
        else:
            raise TypeError(
                f"Unexpected effect request {request.skill_id!r} for "
                f"{type(call).__name__}."
            )
        return success.to(context.robot.qpos.device)

    return verify


def create_place_application(
    simulation: SimulationManager,
    robot: Robot,
    obj: RigidObject,
    *,
    hand_open: torch.Tensor,
    hand_grasp: torch.Tensor,
    n_sample: int,
    force_reannotate: bool,
) -> SkillRuntime:
    """Assemble the application-facing runtime for the Place tutorial.

    The returned runtime owns the scene/profile/compiler binding and the
    default physical-effect verifier. Task code only needs to submit semantic
    calls through :meth:`SkillRuntime.run`.

    Args:
        simulation: Simulation containing the robot and workpiece.
        robot: Robot executing the semantic calls.
        obj: Workpiece registered under :data:`OBJECT_ID`.
        hand_open: Joint target for the semantic ``open`` command.
        hand_grasp: Joint target for the semantic ``grasp`` command.
        n_sample: Number of grasp candidates generated during annotation.
        force_reannotate: Whether to regenerate cached grasp annotations.

    Returns:
        A fully bound semantic runtime with a default effect verifier.
    """
    object_semantics = create_antipodal_semantics(
        obj,
        label="cube",
    )
    registry, _ = create_graspable_object_registry(
        simulation,
        object_id=OBJECT_ID,
        simulation_uid=OBJECT_SIMULATION_UID,
        semantic_type="cube",
        affordance=object_semantics.affordance,
    )
    return SkillRuntime.from_simulation(
        simulation=simulation,
        robot=robot,
        motion_generator=create_curobo_motion_generator(robot),
        scene_registry=registry,
        robot_profile=create_robot_profile(hand_open, hand_grasp),
        grasp_pose_generators={
            "hand": create_parallel_jaw_grasp_pose_generator(
                n_sample=n_sample,
                force_refresh=force_reannotate,
            )
        },
        effect_verifier=create_place_effect_verifier(obj, robot, hand_open),
        control_dt=TRAJECTORY_SIM_STEPS * simulation.sim_config.physics_dt,
    )


def main() -> None:
    """Execute and physically verify the semantic Pick-to-Place workflow."""
    args = parse_arguments()
    sim = create_tutorial_simulation(args)
    robot = add_tutorial_robot(sim, args.robot)
    obj = create_pick_object(sim)
    hand_open, hand_grasp = get_hand_open_close_qpos(robot)
    initialize_pre_pick_robot_pose(robot, obj, hand_open)
    app = create_place_application(
        sim,
        robot,
        obj,
        hand_open=hand_open,
        hand_grasp=hand_grasp,
        n_sample=args.n_sample,
        force_reannotate=args.force_reannotate,
    )
    calls = create_place_task()

    target_pose = calls[1].at
    assert target_pose is not None
    if not args.no_vis_eef_axis:
        draw_axis_marker(
            sim,
            "semantic_place_object_target",
            broadcast_pose_batch(
                target_pose.to_matrix().to(sim.device),
                robot.get_qpos().shape[0],
            ),
        )
    wait_for_user = prepare_tutorial_scene(
        sim,
        args,
        "Inspect the scene, then press Enter to execute Pick -> Place...",
    )
    if args.diagnose_plan:
        try:
            trajectory, skill_ids = compile_semantic_workflow_for_diagnostics(
                app,
                calls,
                workflow_id="tutorial.semantic_pick_place",
            )
        except RuntimeError as exc:
            logger.log_warning(str(exc))
            return
        logger.log_info(
            f"Diagnostic compile lowered {' -> '.join(skill_ids)} with "
            f"{trajectory.waypoint_count} waypoints."
        )
        return

    recording_started = start_auto_play_recording(
        sim,
        args,
        video_prefix="semantic_place_auto_play",
    )
    try:
        result = app.run(
            calls,
            workflow_id="tutorial.semantic_pick_place",
            on_step=create_runtime_step_observer(
                obj,
                robot,
                grasp_control_part="hand",
                grasp_target=hand_grasp,
            ),
        )
        result.require_all_succeeded()
        for _ in range(POST_EXECUTION_UPDATES):
            app.clock.sleep(sim.sim_config.physics_dt)
    finally:
        stop_auto_play_recording(sim, recording_started)
    logger.log_info(
        "Closed-loop semantic Pick -> Place completed with "
        f"{sum(call.command_count for call in result.segments[0].calls)} "
        "accepted commands.",
        color="green",
    )
    if wait_for_user:
        input("Press Enter to exit the simulation...")
    serve_tutorial_scene(sim, args)


if __name__ == "__main__":
    run_tutorial(main)
