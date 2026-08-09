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

"""Replan after a visible target move, then grasp the relocated cube."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from embodichain.lab.sim import SimulationManager, VisualMaterialCfg
from embodichain.lab.sim.atomic_actions import (
    ActionBinding,
    ActionInvocation,
    Affordance,
    AtomicActionEngine,
    ControlPartCommandProfile,
    EndEffectorPoseGoal,
    EntityState,
    ExecutionEventKind,
    ExecutionRunner,
    ExecutionRunnerCfg,
    ExecutionTick,
    GraspGoal,
    MotionPolicy,
    ObjectSemantics,
    PickUpOptions,
    PlanningContext,
    RecoveryPolicy,
    RunnerStatus,
    RunnerStep,
    SceneEntityPose,
    SceneSnapshot,
    SimulationExecutionAdapter,
    TaskState,
)
from embodichain.lab.sim.cfg import RigidBodyAttributesCfg, RigidObjectCfg
from embodichain.lab.sim.objects import RigidObject
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.utils import logger
from scripts.tutorials.atomic_action.tutorial_utils import (
    add_ur5_gripper_robot,
    create_toppra_motion_generator,
    create_tutorial_argument_parser,
    create_tutorial_simulation,
    draw_axis_marker,
    get_hand_open_close_qpos,
    make_top_down_eef_pose,
    prepare_tutorial_scene,
    run_tutorial,
    serve_tutorial_scene,
    start_auto_play_recording,
    stop_auto_play_recording,
)

TARGET_ENTITY_ID = "moving_target"
TARGET_SIZE = (0.05, 0.05, 0.05)
INITIAL_TARGET_POSITION = (-0.42, -0.18, 0.5 * TARGET_SIZE[2])
MOVED_TARGET_POSITION = (-0.42, 0.12, 0.5 * TARGET_SIZE[2])
TARGET_TO_EEF_HEIGHT = 0.30
MOVE_SAMPLE_COUNT = 80
PICK_SAMPLE_COUNT = 120
HAND_INTERP_STEPS = 12
PICK_LIFT_HEIGHT = 0.16
MINIMUM_LIFT_HEIGHT = 0.08
MAXIMUM_HELD_DISTANCE = 0.10
MOVE_AFTER_COMMAND = 20
TARGET_MOVE_DURATION = 0.6
GOAL_TRANSLATION_THRESHOLD = 0.04
TRACKING_ERROR_THRESHOLD = 0.25
POST_EXECUTION_UPDATES = 120


class _MovingTargetScene:
    """Publish a versioned target pose and move it exactly once."""

    def __init__(
        self,
        target: RigidObject,
        destination: tuple[float, float, float],
    ) -> None:
        self.target = target
        self.destination = torch.tensor(
            destination,
            dtype=torch.float32,
            device=target.device,
        )
        self.version = 0
        self.moved = False

    def snapshot(self, timestamp: float) -> SceneSnapshot:
        """Return the target pose used to ground the late-bound goal.

        Args:
            timestamp: Current elapsed simulation time.

        Returns:
            Versioned scene snapshot containing the target pose.
        """
        return SceneSnapshot(
            timestamp=timestamp,
            version=self.version,
            entities={
                TARGET_ENTITY_ID: EntityState(
                    self.target.get_local_pose(to_matrix=True)
                )
            },
        )

    def move(
        self,
        clock: SimulationExecutionAdapter,
        *,
        duration: float,
    ) -> torch.Tensor:
        """Animate the visible target and advance the scene version.

        Args:
            clock: Simulation adapter used to advance physics between poses.
            duration: Requested target-motion duration in seconds.

        Returns:
            Updated batched target pose.
        """
        if self.moved:
            return self.target.get_local_pose(to_matrix=True)
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("duration must be finite and greater than zero.")
        start_pose = self.target.get_local_pose(to_matrix=True).clone()
        step_count = max(1, math.ceil(duration / clock.physics_dt))
        pose = start_pose.clone()
        for step_index in range(1, step_count + 1):
            alpha = step_index / step_count
            pose[:, :3, 3] = torch.lerp(
                start_pose[:, :3, 3],
                self.destination,
                alpha,
            )
            self.target.set_local_pose(pose)
            clock.sleep(clock.physics_dt)
        self.version += 1
        self.moved = True
        return pose


def _create_moving_target(sim: SimulationManager) -> RigidObject:
    """Create the bright cube, initially kinematic for scripted relocation."""
    return sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid=TARGET_ENTITY_ID,
            shape=CubeCfg(
                size=list(TARGET_SIZE),
                visual_material=VisualMaterialCfg(
                    uid="moving_target_blue",
                    base_color=[0.05, 0.30, 1.0, 1.0],
                    metallic=0.15,
                    roughness=0.3,
                ),
            ),
            attrs=RigidBodyAttributesCfg(
                mass=0.05,
                dynamic_friction=0.97,
                static_friction=0.99,
                enable_ccd=True,
            ),
            body_type="kinematic",
            max_convex_hull_num=16,
            init_pos=INITIAL_TARGET_POSITION,
        )
    )


def _compose_goal_pose(
    target_pose: torch.Tensor,
    relative_pose: torch.Tensor,
) -> torch.Tensor:
    """Compose batched target poses with the target-to-EEF transform."""
    relative_batch = relative_pose.unsqueeze(0).expand(target_pose.shape[0], -1, -1)
    return torch.bmm(target_pose, relative_batch)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the moving-target tutorial."""
    parser = create_tutorial_argument_parser(
        "Demonstrate ExecutionRunner replanning after a visible target move."
    )
    parser.add_argument(
        "--no_target_motion",
        action="store_true",
        help="Keep the target fixed to run the no-replanning control case.",
    )
    return parser.parse_args()


def main() -> None:
    """Replan toward a relocated cube, close the gripper, and lift it."""
    args = parse_arguments()
    sim = create_tutorial_simulation(args)
    robot = add_ur5_gripper_robot(sim)
    target = _create_moving_target(sim)
    sim.update(step=10)
    target_scene = _MovingTargetScene(target, MOVED_TARGET_POSITION)
    adapter = SimulationExecutionAdapter(
        sim,
        robot,
        scene_supplier=target_scene.snapshot,
    )
    motion_gen = create_toppra_motion_generator(robot)
    hand_open, hand_close = get_hand_open_close_qpos(robot)
    hand_open_batch = hand_open.unsqueeze(0).repeat(robot.get_qpos().shape[0], 1)
    for target_value in (False, True):
        robot.set_qpos(hand_open_batch, name="hand", target=target_value)
    robot.clear_dynamics()

    target_to_eef = make_top_down_eef_pose(
        torch.tensor(
            [0.0, 0.0, TARGET_TO_EEF_HEIGHT],
            dtype=torch.float32,
            device=sim.device,
        )
    )
    initial_target_pose = target.get_local_pose(to_matrix=True)
    draw_axis_marker(
        sim,
        "moving_target_original_goal",
        _compose_goal_pose(initial_target_pose, target_to_eef),
        axis_len=0.10,
    )

    grasp_target_position = (
        INITIAL_TARGET_POSITION if args.no_target_motion else MOVED_TARGET_POSITION
    )
    grasp_pose = make_top_down_eef_pose(
        torch.tensor(
            grasp_target_position,
            dtype=torch.float32,
            device=sim.device,
        )
    )
    semantics = ObjectSemantics(
        affordance=Affordance(),
        geometry={},
        label="cube",
        entity=target,
    )
    binding = ActionBinding(
        manipulators={"primary": "arm"},
        end_effectors={"primary": "hand"},
    )
    engine = AtomicActionEngine(
        motion_generator=motion_gen,
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=hand_open,
                grasp=hand_close,
            )
        },
    )
    move_invocation = ActionInvocation(
        skill_id="move_end_effector",
        goal=EndEffectorPoseGoal(
            SceneEntityPose(
                TARGET_ENTITY_ID,
                relative_pose=target_to_eef,
            )
        ),
        binding=binding,
        motion_policy=MotionPolicy(
            sample_count=MOVE_SAMPLE_COUNT,
            control_dt=2.0 * adapter.physics_dt,
        ),
        recovery_policy=RecoveryPolicy(
            max_replans=2,
            tracking_error_threshold=TRACKING_ERROR_THRESHOLD,
            goal_translation_threshold=GOAL_TRANSLATION_THRESHOLD,
            phase_timeout=20.0,
        ),
    )
    pick_invocation = ActionInvocation(
        skill_id="pick_up",
        goal=GraspGoal(semantics, grasp_xpos=grasp_pose),
        binding=binding,
        motion_policy=MotionPolicy(
            sample_count=PICK_SAMPLE_COUNT,
            control_dt=2.0 * adapter.physics_dt,
        ),
        recovery_policy=RecoveryPolicy(
            max_phase_retries=1,
            tracking_error_threshold=TRACKING_ERROR_THRESHOLD,
            phase_timeout=30.0,
        ),
        skill_options=PickUpOptions(
            pre_grasp_distance=0.15,
            lift_height=PICK_LIFT_HEIGHT,
            hand_interp_steps=HAND_INTERP_STEPS,
        ),
    )
    task_state = TaskState.empty(robot.get_qpos().shape[0], robot.device)
    initial_context = adapter.observe(task_state)
    session = engine.start((move_invocation, pick_invocation), initial_context)
    runner = ExecutionRunner(
        session,
        adapter,
        adapter,
        clock=adapter,
        cfg=ExecutionRunnerCfg(minimum_cycle_time=adapter.physics_dt),
    )

    wait_for_user = prepare_tutorial_scene(
        sim,
        args,
        "Watch the blue cube, then press Enter to replan and pick it up...",
    )
    dynamic_change_observed = False
    replan_observed = False
    pickup_start_command: int | None = None
    pickup_dynamics_cleared = False

    clear_after_pick_command = (
        round((PICK_SAMPLE_COUNT - HAND_INTERP_STEPS) * 0.6) + HAND_INTERP_STEPS
    )

    def on_step(step: RunnerStep) -> None:
        nonlocal dynamic_change_observed, replan_observed
        nonlocal pickup_dynamics_cleared, pickup_start_command
        if (
            not args.no_target_motion
            and not target_scene.moved
            and step.command_count >= MOVE_AFTER_COMMAND
        ):
            logger.log_warning(
                f"Animating the blue target for {TARGET_MOVE_DURATION:.1f} s "
                "while the robot holds its current command."
            )
            moved_pose = target_scene.move(
                adapter,
                duration=TARGET_MOVE_DURATION,
            )
            draw_axis_marker(
                sim,
                "moving_target_replanned_goal",
                _compose_goal_pose(moved_pose, target_to_eef),
                axis_len=0.10,
            )
            displacement = torch.linalg.vector_norm(
                moved_pose[:, :3, 3] - initial_target_pose[:, :3, 3],
                dim=1,
            )
            logger.log_warning(
                "Moved the blue target after "
                f"{step.command_count} accepted commands by "
                f"{displacement.detach().cpu().tolist()} m; the original goal "
                "axis remains visible."
            )
        if step.tick is None:
            return
        for event in step.tick.events:
            if event.kind in {
                ExecutionEventKind.DYNAMIC_GOAL_CHANGED,
                ExecutionEventKind.REPLANNED,
                ExecutionEventKind.RECOVERY_EXHAUSTED,
            }:
                env_ids = event.env_mask.nonzero(as_tuple=False).flatten().tolist()
                logger.log_info(
                    f"Execution event {event.kind.value}: env rows={env_ids}; "
                    f"{event.message}"
                )
            dynamic_change_observed |= (
                event.kind is ExecutionEventKind.DYNAMIC_GOAL_CHANGED
            )
            replan_observed |= event.kind is ExecutionEventKind.REPLANNED
            if (
                event.kind is ExecutionEventKind.ACTION_PLANNED
                and event.invocation_index == 1
                and pickup_start_command is None
            ):
                target.set_body_type("dynamic")
                target.clear_dynamics()
                pickup_start_command = step.command_count
                logger.log_info(
                    "The approach completed; the cube is now dynamic and PickUp "
                    "is starting.",
                    color="green",
                )
        if (
            pickup_start_command is not None
            and not pickup_dynamics_cleared
            and step.command_count - pickup_start_command > clear_after_pick_command
        ):
            target.clear_dynamics()
            pickup_dynamics_cleared = True

    def verify_pickup_effect(
        _context: PlanningContext,
        _: ExecutionTick,
    ) -> torch.Tensor:
        """Verify that the cube rose with, and remains near, the end effector."""
        cube_position = target.get_local_pose(to_matrix=True)[:, :3, 3]
        eef_position = robot.compute_fk(
            qpos=robot.get_qpos(name="arm"),
            name="arm",
            to_matrix=True,
        )[:, :3, 3]
        lift_height = cube_position[:, 2] - 0.5 * TARGET_SIZE[2]
        held_distance = torch.linalg.vector_norm(cube_position - eef_position, dim=1)
        success = (lift_height >= MINIMUM_LIFT_HEIGHT) & (
            held_distance <= MAXIMUM_HELD_DISTANCE
        )
        logger.log_info(
            "PickUp verification: "
            f"lift={lift_height.detach().cpu().tolist()} m, "
            f"cube-to-EEF={held_distance.detach().cpu().tolist()} m, "
            f"success={success.detach().cpu().tolist()}."
        )
        return success

    recording_started = start_auto_play_recording(
        sim,
        args,
        video_prefix="moving_target_recovery_auto_play",
        look_at=(
            (-1.25, -1.15, 0.95),
            (-0.32, -0.02, 0.25),
            (0.0, 0.0, 1.0),
        ),
    )
    try:
        result = runner.run_until_blocked(
            effect_verifier=verify_pickup_effect,
            on_step=on_step,
        )
        for _ in range(POST_EXECUTION_UPDATES):
            adapter.sleep(adapter.physics_dt)
    finally:
        stop_auto_play_recording(sim, recording_started)

    if result.status is not RunnerStatus.COMPLETED:
        raise RuntimeError(f"Closed-loop execution failed: {result.message}")
    if not args.no_target_motion:
        if not target_scene.moved:
            raise RuntimeError("Execution completed before the target could move.")
        if not dynamic_change_observed:
            raise RuntimeError("The target move was not reported as a dynamic change.")
        if not replan_observed:
            raise RuntimeError("The target move did not trigger replanning.")
    if pickup_start_command is None:
        raise RuntimeError("PickUp did not start after the recovered approach.")
    if not pickup_dynamics_cleared:
        raise RuntimeError("The cube was not stabilized after gripper closure.")
    logger.log_info(
        f"Execution completed and lifted the cube after {result.command_count} "
        "accepted commands.",
        color="green",
    )

    serve_tutorial_scene(sim, args)
    if wait_for_user:
        input("Press Enter to exit the simulation...")


if __name__ == "__main__":
    run_tutorial(main)
