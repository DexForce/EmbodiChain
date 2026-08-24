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

"""Use semantic skills to pick with one arm and hand over to the other.

The workflow contains an object-centric ``Pick`` followed by a registered
dual-arm transfer call. The robot profile chooses the left and right resources;
an explicit lowerer supplies the atomic HandOver goal and embodiment-specific
receive behavior at grounding time. :class:`SkillRuntime` executes each
call from fresh observations and commits transfer state only after physical
verification.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import sys
from pathlib import Path
from typing import ClassVar, TYPE_CHECKING

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.atomic_actions import (
    ControlPartCommandProfile,
    EffectVerificationRequest,
    GraspGoal,
    HandOver as AtomicHandOver,
    HandOverOptions,
    MotionPolicy,
    PlanningContext,
    RecoveryPolicy,
)
from embodichain.lab.sim.objects import RigidObject, Robot
from embodichain.lab.sim.skills import (
    GRASP_AFFORDANCE_CAPABILITY,
    Pick,
    RegisteredSemanticCall,
    RegisteredSemanticLowerer,
    ResourceBinding,
    RobotSkillProfile,
    SceneObjectRef,
    SceneRegistry,
    SemanticCallDescriptor,
    SemanticEffectVerifier,
    SemanticLowering,
    SemanticPose,
    SkillRuntime,
    SkillPolicyPreset,
    builtin_semantic_call_catalog,
)
from embodichain.utils import logger
from scripts.tutorials.atomic_action.hand_over import (
    HANDOVER_RECORD_LOOK_AT,
    HAND_CLOSE_QPOS,
    TRAJECTORY_SIM_STEPS,
    create_dual_robot,
    create_handover_object,
    create_support_surface,
)
from scripts.tutorials.atomic_action.scenario_utils import settle_object
from scripts.tutorials.atomic_action.tutorial_utils import (
    clone_local_pose_from_first_env,
    create_antipodal_semantics,
    create_parallel_jaw_grasp_pose_generator,
    create_toppra_motion_generator,
    create_tutorial_argument_parser,
    create_tutorial_simulation,
    get_hand_open_close_qpos,
    prepare_tutorial_scene,
    publish_tutorial_scene,
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

if TYPE_CHECKING:
    from embodichain.lab.sim.skills.integration import BoundSemanticCall

OBJECT_ID = "workpiece"
OBJECT_SIMULATION_UID = "handover_object"
PICK_SAMPLE_COUNT = 80
HANDOVER_SAMPLE_COUNT = 140
MIDDLE_OBJECT_POSITION = (0.0, 0.0, 0.70)
FINAL_OBJECT_POSITION = (0.0, -0.20, 0.70)
OBJECT_QUATERNION_WXYZ = (0.70710678, 0.70710678, 0.0, 0.0)
HANDOVER_CALL_ID = "tutorial.hand_over"
HANDOVER_PRE_GRASP_DISTANCE = 0.08
HANDOVER_LIFT_HEIGHT = 0.08
HANDOVER_HAND_INTERP_STEPS = 10
HANDOVER_HOLD_STEPS = 4
HANDOVER_RETREAT_STEPS = 28
HANDOVER_RECEIVE_APPROACH_DIRECTION = (0.0, 0.70710678, -0.70710678)
TRACKING_ERROR_THRESHOLD = 1.0
MINIMUM_PICK_LIFT = 0.05
MAXIMUM_HELD_RELATION_ERROR = 0.06
MAXIMUM_FINAL_POSITION_ERROR = 0.10
MAXIMUM_HAND_ERROR = 0.03
POST_EXECUTION_UPDATES = 120


class TutorialHandOverLowerer(RegisteredSemanticLowerer):
    """Lower the tutorial's registered transfer call to atomic HandOver."""

    call_id: ClassVar[str] = HANDOVER_CALL_ID
    schema_version: ClassVar[int] = 1

    def __init__(self, registry: SceneRegistry) -> None:
        if not isinstance(registry, SceneRegistry):
            raise TypeError("registry must be a SceneRegistry.")
        self._registry = registry

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
    ) -> SemanticLowering:
        """Build tuned HandOver options from the latest planning device."""
        del bound
        if not isinstance(call.arguments, Mapping):
            raise TypeError("tutorial.hand_over arguments must be a mapping.")
        object_ref = call.arguments.get("object")
        if type(object_ref) is not SceneObjectRef:
            raise TypeError("tutorial.hand_over requires a SceneObjectRef object.")
        grasp_ref = self._registry.resolve_affordance(
            object_ref,
            capability=GRASP_AFFORDANCE_CAPABILITY,
        )
        semantics = self._registry.object_semantics(
            object_ref,
            affordance=grasp_ref,
        )
        device = context.robot.qpos.device
        return SemanticLowering(
            goal=GraspGoal(semantics),
            skill_options=HandOverOptions(
                receive_pick_object_part="bottom",
                middle_object_pose=SemanticPose(
                    MIDDLE_OBJECT_POSITION,
                    OBJECT_QUATERNION_WXYZ,
                )
                .to_matrix()
                .to(device),
                final_object_pose=SemanticPose(
                    FINAL_OBJECT_POSITION,
                    OBJECT_QUATERNION_WXYZ,
                )
                .to_matrix()
                .to(device),
                pre_grasp_distance=HANDOVER_PRE_GRASP_DISTANCE,
                lift_height=HANDOVER_LIFT_HEIGHT,
                hand_interp_steps=HANDOVER_HAND_INTERP_STEPS,
                hold_steps=HANDOVER_HOLD_STEPS,
                retreat_steps=HANDOVER_RETREAT_STEPS,
                receive_approach_direction=torch.tensor(
                    HANDOVER_RECEIVE_APPROACH_DIRECTION,
                    dtype=torch.float32,
                    device=device,
                ),
            ),
        )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the semantic HandOver tutorial."""
    parser = create_tutorial_argument_parser(
        "Execute and verify semantic Pick -> HandOver through the skill runtime.",
        features=("diagnose_plan", "grasp_sampling", "headless_play"),
        default_device="cpu",
        default_renderer="hybrid",
    )
    return parser.parse_args()


def create_robot_profile(
    left_open: torch.Tensor,
    left_grasp: torch.Tensor,
    right_open: torch.Tensor,
    right_grasp: torch.Tensor,
) -> RobotSkillProfile:
    """Declare two disjoint manipulators and their semantic skill defaults.

    Args:
        left_open: Left-hand joint positions for ``open``.
        left_grasp: Left-hand joint positions for ``grasp``.
        right_open: Right-hand joint positions for ``open``.
        right_grasp: Right-hand joint positions for ``grasp``.

    Returns:
        A dual-arm profile with deterministic Pick and HandOver assignments.
    """
    return RobotSkillProfile(
        profile_id="tutorial.dual_arm",
        resources={
            "left": create_manipulator_resource(
                "left",
                motion_control_part="left_arm",
                grasp_control_part="left_hand",
            ),
            "right": create_manipulator_resource(
                "right",
                motion_control_part="right_arm",
                grasp_control_part="right_hand",
            ),
        },
        command_profiles={
            "left_hand": ControlPartCommandProfile.joint_positions(
                open=left_open,
                grasp=left_grasp,
            ),
            "right_hand": ControlPartCommandProfile.joint_positions(
                open=right_open,
                grasp=right_grasp,
            ),
        },
        defaults={
            "pick_up": ResourceBinding({"primary": "left"}),
            "hand_over": ResourceBinding({"source": "left", "destination": "right"}),
        },
        presets={
            "pick": SkillPolicyPreset(
                "pick",
                motion_policy=MotionPolicy(
                    strategy="motion_gen",
                    sample_count=PICK_SAMPLE_COUNT,
                ),
                recovery_policy=RecoveryPolicy(
                    tracking_error_threshold=TRACKING_ERROR_THRESHOLD,
                ),
            ),
            "hand_over": SkillPolicyPreset(
                "hand_over",
                motion_policy=MotionPolicy(
                    strategy="motion_gen",
                    sample_count=HANDOVER_SAMPLE_COUNT,
                ),
                # Retrying after either gripper has changed ownership is not
                # safe without reconciling the physical attachment first.
                recovery_policy=RecoveryPolicy(
                    max_action_retries=0,
                    tracking_error_threshold=TRACKING_ERROR_THRESHOLD,
                ),
            ),
        },
        default_preset="pick",
        skill_presets={"pick_up": "pick", "hand_over": "hand_over"},
    )


def create_handover_task() -> tuple[Pick, RegisteredSemanticCall]:
    """Declare the robot-independent calls submitted at the application entry."""
    object_ref = SceneObjectRef(OBJECT_ID)
    return (
        Pick(object=object_ref),
        RegisteredSemanticCall(
            call_id=HANDOVER_CALL_ID,
            arguments={"object": object_ref},
        ),
    )


def create_handover_effect_verifier(
    obj: RigidObject,
    robot: Robot,
    *,
    left_open: torch.Tensor,
    right_grasp: torch.Tensor,
) -> SemanticEffectVerifier:
    """Create physical Pick and HandOver verification for the tutorial scene.

    Args:
        obj: Object transferred between manipulators.
        robot: Dual-arm robot executing the workflow.
        left_open: Source-hand release target.
        right_grasp: Destination-hand grasp target.

    Returns:
        Runtime callback producing a boolean result per environment.
    """
    initial_pose = obj.get_local_pose(to_matrix=True)
    if not isinstance(initial_pose, torch.Tensor) or initial_pose.dim() != 3:
        raise ValueError("The tutorial object pose must have shape (B, 4, 4).")
    initial_height = initial_pose[:, 2, 3].clone()
    final_position = torch.tensor(FINAL_OBJECT_POSITION, dtype=torch.float32)

    def verify(
        call: object,
        request: EffectVerificationRequest,
        context: PlanningContext,
    ) -> torch.Tensor:
        object_position = obj.get_local_pose(to_matrix=True)[:, :3, 3]
        if type(call) is Pick and request.skill_id == "pick_up":
            lift = object_position[:, 2] - initial_height.to(object_position.device)
            held = request.expected_effects.held_object_updates.get("left_arm")
            if held is None:
                raise RuntimeError("Pick verification requires a left-arm attachment.")
            held_error = object_to_eef_translation_error(
                obj,
                robot,
                motion_control_part="left_arm",
                expected_object_to_eef=held.object_to_eef,
            )
            success = (lift >= MINIMUM_PICK_LIFT) & (
                held_error <= MAXIMUM_HELD_RELATION_ERROR
            )
            logger.log_info(
                "Semantic Pick verification: "
                f"lift={lift.detach().cpu().tolist()} m, "
                "object-to-left-EEF translation error="
                f"{held_error.detach().cpu().tolist()} m, "
                f"success={success.detach().cpu().tolist()}."
            )
        elif (
            type(call) is RegisteredSemanticCall
            and call.call_id == HANDOVER_CALL_ID
            and request.skill_id == "hand_over"
        ):
            final_error = torch.linalg.vector_norm(
                object_position
                - final_position.to(
                    device=object_position.device,
                    dtype=object_position.dtype,
                ),
                dim=1,
            )
            held = request.expected_effects.held_object_updates.get("right_arm")
            if held is None:
                raise RuntimeError(
                    "HandOver verification requires a right-arm attachment."
                )
            receiver_error = object_to_eef_translation_error(
                obj,
                robot,
                motion_control_part="right_arm",
                expected_object_to_eef=held.object_to_eef,
            )
            source_error = joint_target_error(
                robot,
                control_part="left_hand",
                target=left_open,
            )
            receiver_hand_error = joint_target_error(
                robot,
                control_part="right_hand",
                target=right_grasp,
            )
            success = (
                (final_error <= MAXIMUM_FINAL_POSITION_ERROR)
                & (receiver_error <= MAXIMUM_HELD_RELATION_ERROR)
                & (source_error <= MAXIMUM_HAND_ERROR)
                & (receiver_hand_error <= MAXIMUM_HAND_ERROR)
            )
            logger.log_info(
                "Semantic HandOver verification: "
                f"final_error={final_error.detach().cpu().tolist()} m, "
                "object-to-right-EEF translation error="
                f"{receiver_error.detach().cpu().tolist()} m, "
                f"source_open_error={source_error.detach().cpu().tolist()} rad, "
                "receiver_grasp_error="
                f"{receiver_hand_error.detach().cpu().tolist()} rad, "
                f"success={success.detach().cpu().tolist()}."
            )
        else:
            raise TypeError(
                f"Unexpected effect request {request.skill_id!r} for "
                f"{type(call).__name__}."
            )
        return success.to(context.robot.qpos.device)

    return verify


def create_handover_application(
    simulation: SimulationManager,
    robot: Robot,
    obj: RigidObject,
    *,
    left_open: torch.Tensor,
    left_grasp: torch.Tensor,
    right_open: torch.Tensor,
    right_grasp: torch.Tensor,
    n_sample: int,
    force_reannotate: bool,
) -> SkillRuntime:
    """Assemble the application-facing runtime for the HandOver tutorial.

    The returned runtime owns the registered call extension, robot binding,
    scene catalog, and default physical-effect verifier. Task code only needs
    to submit semantic calls through :meth:`SkillRuntime.run`.

    Args:
        simulation: Simulation containing the robot and workpiece.
        robot: Dual-arm robot executing the semantic calls.
        obj: Workpiece registered under :data:`OBJECT_ID`.
        left_open: Left-hand target for the semantic ``open`` command.
        left_grasp: Left-hand target for the semantic ``grasp`` command.
        right_open: Right-hand target for the semantic ``open`` command.
        right_grasp: Right-hand target for the semantic ``grasp`` command.
        n_sample: Number of grasp candidates generated during annotation.
        force_reannotate: Whether to regenerate cached grasp annotations.

    Returns:
        A fully bound semantic runtime with a default effect verifier.
    """
    object_semantics = create_antipodal_semantics(
        obj,
        label="handover object",
    )
    registry, _ = create_graspable_object_registry(
        simulation,
        object_id=OBJECT_ID,
        simulation_uid=OBJECT_SIMULATION_UID,
        semantic_type="handover object",
        affordance=object_semantics.affordance,
    )
    profile = create_robot_profile(
        left_open,
        left_grasp,
        right_open,
        right_grasp,
    )
    call_catalog = builtin_semantic_call_catalog().with_descriptor(
        SemanticCallDescriptor(
            call_id=HANDOVER_CALL_ID,
            spec_type=RegisteredSemanticCall,
            target_descriptor=AtomicHandOver.descriptor(),
        )
    )
    grasp_pose_generator = create_parallel_jaw_grasp_pose_generator(
        n_sample=n_sample,
        force_refresh=force_reannotate,
    )
    return SkillRuntime.from_simulation(
        simulation=simulation,
        robot=robot,
        motion_generator=create_toppra_motion_generator(robot),
        scene_registry=registry,
        robot_profile=profile,
        grasp_pose_generators={
            "left_hand": grasp_pose_generator,
            "right_hand": grasp_pose_generator,
        },
        call_catalog=call_catalog,
        effect_verifier=create_handover_effect_verifier(
            obj,
            robot,
            left_open=left_open,
            right_grasp=right_grasp,
        ),
        registered_lowerers=(TutorialHandOverLowerer(registry),),
        control_dt=TRAJECTORY_SIM_STEPS * simulation.sim_config.physics_dt,
    )


def main() -> None:
    """Execute and physically verify the semantic dual-arm workflow."""
    args = parse_arguments()
    sim = create_tutorial_simulation(
        args,
        arena_space=3.0,
        light_pos=(0.0, -0.4, 3.0),
    )
    robot = create_dual_robot(sim, args.robot)
    create_support_surface(sim)
    obj = create_handover_object(sim)
    settle_object(sim, obj, step=0)
    clone_local_pose_from_first_env(obj)
    obj.clear_dynamics()
    publish_tutorial_scene(sim, args)
    left_open, left_grasp = get_hand_open_close_qpos(
        robot,
        hand_control_part="left_hand",
        close_qpos=HAND_CLOSE_QPOS,
    )
    right_open, right_grasp = get_hand_open_close_qpos(
        robot,
        hand_control_part="right_hand",
        close_qpos=HAND_CLOSE_QPOS,
    )
    app = create_handover_application(
        sim,
        robot=robot,
        obj=obj,
        left_open=left_open,
        left_grasp=left_grasp,
        right_open=right_open,
        right_grasp=right_grasp,
        n_sample=args.n_sample,
        force_reannotate=args.force_reannotate,
    )
    calls = create_handover_task()

    wait_for_user = prepare_tutorial_scene(
        sim,
        args,
        "Inspect the scene, then press Enter to execute Pick -> HandOver...",
    )
    for _ in range(20):
        sim.update(step=10)

    if args.diagnose_plan:
        try:
            trajectory, skill_ids = compile_semantic_workflow_for_diagnostics(
                app,
                calls,
                workflow_id="tutorial.semantic_pick_handover",
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
        video_prefix="semantic_handover_auto_play",
        look_at=HANDOVER_RECORD_LOOK_AT,
    )
    try:
        result = app.run(
            calls,
            workflow_id="tutorial.semantic_pick_handover",
            on_step=create_runtime_step_observer(
                obj,
                robot,
                grasp_control_part="left_hand",
                grasp_target=left_grasp,
            ),
        )
        result.require_all_succeeded()
        for _ in range(POST_EXECUTION_UPDATES):
            app.clock.sleep(sim.sim_config.physics_dt)
    finally:
        stop_auto_play_recording(sim, recording_started)
    logger.log_info(
        "Closed-loop semantic Pick -> HandOver completed with "
        f"{sum(call.command_count for call in result.segments[0].calls)} "
        "accepted commands.",
        color="green",
    )
    if wait_for_user:
        input("Press Enter to exit the simulation...")
    serve_tutorial_scene(sim, args)


if __name__ == "__main__":
    run_tutorial(main)
