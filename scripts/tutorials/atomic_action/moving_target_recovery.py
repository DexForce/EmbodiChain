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

"""Replan one PickUp action after its visible target moves."""

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
    Affordance,
    AtomicActionEngine,
    ControlPartCommandProfile,
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
    add_tutorial_robot,
    create_curobo_motion_generator,
    create_tutorial_argument_parser,
    create_tutorial_simulation,
    draw_axis_marker,
    get_hand_open_close_qpos,
    initialize_pre_pick_robot_pose,
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
PICK_SAMPLE_COUNT = 120
HAND_INTERP_STEPS = 12
PICK_LIFT_HEIGHT = 0.16
MINIMUM_LIFT_HEIGHT = 0.08
MAXIMUM_HELD_DISTANCE = 0.10
MOVE_AFTER_COMMAND = 20
TARGET_MOVE_DURATION = 0.6
TARGET_PUSH_DURATION = 0.12
TARGET_PUSH_FORCE = 1.25
GOAL_TRANSLATION_THRESHOLD = 0.04
TRACKING_ERROR_THRESHOLD = 1.0
POST_EXECUTION_UPDATES = 120


class _MovingTargetScene:
    """Publish a versioned target pose and physically push it exactly once."""

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

    def push(
        self,
        clock: SimulationExecutionAdapter,
        *,
        duration: float,
        force_duration: float,
        force_magnitude: float,
    ) -> torch.Tensor:
        """Push the visible target with a short force pulse.

        Args:
            clock: Simulation adapter used to advance physics.
            duration: Total time allowed for the push and natural deceleration.
            force_duration: Time spent applying the horizontal force.
            force_magnitude: Magnitude of the applied force in newtons.

        Returns:
            Batched target pose after the physical motion.
        """
        if self.moved:
            return self.target.get_local_pose(to_matrix=True)
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("duration must be finite and greater than zero.")
        if (
            not math.isfinite(force_duration)
            or force_duration <= 0.0
            or force_duration > duration
        ):
            raise ValueError(
                "force_duration must be finite, greater than zero, and no "
                "greater than duration."
            )
        if not math.isfinite(force_magnitude) or force_magnitude <= 0.0:
            raise ValueError("force_magnitude must be finite and greater than zero.")

        start_pose = self.target.get_local_pose(to_matrix=True).clone()
        planar_offset = self.destination - start_pose[:, :3, 3]
        planar_offset[:, 2] = 0.0
        planar_distance = torch.linalg.vector_norm(planar_offset, dim=1)
        if torch.any(planar_distance <= torch.finfo(planar_offset.dtype).eps):
            raise ValueError("destination must differ from the current planar pose.")
        force = force_magnitude * planar_offset / planar_distance.unsqueeze(-1)

        self.target.clear_dynamics()
        step_count = max(1, math.ceil(duration / clock.physics_dt))
        force_step_count = min(
            step_count,
            max(1, math.ceil(force_duration / clock.physics_dt)),
        )
        for step_index in range(step_count):
            if step_index < force_step_count:
                self.target.add_force_torque(force=force)
            clock.sleep(clock.physics_dt)
        self.target.clear_dynamics()

        pose = self.target.get_local_pose(to_matrix=True)
        self.version += 1
        self.moved = True
        return pose


def _create_moving_target(sim: SimulationManager) -> RigidObject:
    """Create the bright dynamic cube used for the physical push."""
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
            body_type="dynamic",
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
    """Replan a late-bound PickUp request and lift the relocated cube."""
    args = parse_arguments()
    sim = create_tutorial_simulation(args)
    robot = add_tutorial_robot(sim, args.robot)
    target = _create_moving_target(sim)
    sim.prepare()
    sim.update(step=10)
    target_scene = _MovingTargetScene(target, MOVED_TARGET_POSITION)
    sim_runtime = SimulationExecutionAdapter(
        sim,
        robot,
        control_dt=2.0 * sim.sim_config.physics_dt,
        scene_supplier=target_scene.snapshot,
    )
    motion_gen = create_curobo_motion_generator(robot)
    hand_open, hand_close = get_hand_open_close_qpos(robot)
    initialize_pre_pick_robot_pose(robot, target, hand_open)
    if args.no_target_motion:
        target.clear_dynamics()

    target_to_grasp = make_top_down_eef_pose(
        torch.zeros(3, dtype=torch.float32, device=sim.device)
    )
    initial_target_pose = target.get_local_pose(to_matrix=True)
    draw_axis_marker(
        sim,
        "moving_target_original_goal",
        _compose_goal_pose(initial_target_pose, target_to_grasp),
        axis_len=0.10,
    )

    semantics = ObjectSemantics(
        affordance=Affordance(),
        geometry={},
        label="cube",
        entity=target,
        entity_id=TARGET_ENTITY_ID,
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
    pick_invocation = engine.make_invocation(
        "pick_up",
        GraspGoal(
            semantics,
            grasp_xpos=SceneEntityPose(
                TARGET_ENTITY_ID,
                relative_pose=target_to_grasp,
            ),
        ),
        control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
        motion_policy=MotionPolicy(
            strategy="motion_gen",
            sample_count=PICK_SAMPLE_COUNT,
        ),
        recovery_policy=RecoveryPolicy(
            max_replans=2,
            max_action_retries=1,
            tracking_error_threshold=TRACKING_ERROR_THRESHOLD,
            goal_translation_threshold=GOAL_TRANSLATION_THRESHOLD,
            action_timeout=30.0,
        ),
        skill_options=PickUpOptions(
            pre_grasp_distance=0.15,
            lift_height=PICK_LIFT_HEIGHT,
            hand_interp_steps=HAND_INTERP_STEPS,
        ),
    )
    task_state = TaskState.empty(robot.get_qpos().shape[0], robot.device)
    initial_context = sim_runtime.observe(task_state)
    session = engine.start((pick_invocation,), initial_context)
    runner = ExecutionRunner(
        session=session,
        observation_provider=sim_runtime,
        command_sink=sim_runtime,
        clock=sim_runtime,
        cfg=ExecutionRunnerCfg(minimum_cycle_time=sim_runtime.physics_dt),
    )

    wait_for_user = prepare_tutorial_scene(
        sim,
        args,
        "Watch the blue cube, then press Enter to run recovering PickUp...",
    )
    observed_events: set[ExecutionEventKind] = set()
    plan_start_command = 0
    pickup_dynamics_cleared = False

    clear_after_pick_command = session.trajectory_segment("lift").start

    def on_step(step: RunnerStep) -> None:
        nonlocal clear_after_pick_command, pickup_dynamics_cleared, plan_start_command
        if (
            not args.no_target_motion
            and not target_scene.moved
            and step.command_count >= MOVE_AFTER_COMMAND
        ):
            logger.log_warning(
                f"Applying a {TARGET_PUSH_FORCE:.2f} N force pulse to the blue "
                "target while the robot holds its current command."
            )
            moved_pose = target_scene.push(
                sim_runtime,
                duration=TARGET_MOVE_DURATION,
                force_duration=TARGET_PUSH_DURATION,
                force_magnitude=TARGET_PUSH_FORCE,
            )
            draw_axis_marker(
                sim,
                "moving_target_replanned_goal",
                _compose_goal_pose(moved_pose, target_to_grasp),
                axis_len=0.10,
            )
            displacement = torch.linalg.vector_norm(
                moved_pose[:, :3, 3] - initial_target_pose[:, :3, 3],
                dim=1,
            )
            logger.log_warning(
                "The force pulse moved the blue target after "
                f"{step.command_count} accepted commands by "
                f"{displacement.detach().cpu().tolist()} m; the original goal "
                "axis remains visible."
            )
        if step.tick is None:
            return
        for event in step.tick.events:
            observed_events.add(event.kind)
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
            if event.kind is ExecutionEventKind.REPLANNED:
                dispatched_active_command = step.tick.command is not None and bool(
                    step.tick.command.active_mask.any().item()
                )
                plan_start_command = step.command_count - int(dispatched_active_command)
                clear_after_pick_command = session.trajectory_segment("lift").start
                logger.log_info(
                    "PickUp discarded the stale plan and restarted from the "
                    "latest cube pose.",
                    color="green",
                )
        if (
            (args.no_target_motion or target_scene.moved)
            and not pickup_dynamics_cleared
            and step.command_count - plan_start_command >= clear_after_pick_command
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
            sim_runtime.sleep(sim_runtime.physics_dt)
    finally:
        stop_auto_play_recording(sim, recording_started)

    if result.status is not RunnerStatus.COMPLETED:
        raise RuntimeError(f"Closed-loop execution failed: {result.message}")
    if not args.no_target_motion:
        if not target_scene.moved:
            raise RuntimeError("Execution completed before the target could move.")
        if ExecutionEventKind.DYNAMIC_GOAL_CHANGED not in observed_events:
            raise RuntimeError("The target move was not reported as a dynamic change.")
        if ExecutionEventKind.REPLANNED not in observed_events:
            raise RuntimeError("The target move did not trigger replanning.")
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
