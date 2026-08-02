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

"""Demonstrate late-bound goal recovery when a target entity moves."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.lab.sim.atomic_actions import (
    ActionBinding,
    ActionInvocation,
    AtomicActionEngine,
    EndEffectorPoseGoal,
    ExecutionEventKind,
    ExecutionRunner,
    ExecutionRunnerCfg,
    MotionPolicy,
    MoveEndEffector,
    RecoveryPolicy,
    RigidObjectSceneProvider,
    RunnerStatus,
    RunnerStep,
    SceneEntityPose,
    SimulationExecutionAdapter,
    TaskState,
)
from embodichain.lab.sim.cfg import RigidBodyAttributesCfg
from embodichain.lab.sim.objects import RigidObjectCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.utils import logger
from scripts.tutorials.atomic_action.tutorial_utils import (
    add_ur5_gripper_robot,
    create_toppra_motion_generator,
    create_tutorial_simulation,
    prepare_tutorial_scene,
    run_tutorial,
    serve_tutorial_scene,
    start_auto_play_recording,
    stop_auto_play_recording,
)

TARGET_UID = "moving_target"
CONTROL_PART = "arm"
SAMPLE_COUNT = 80
MOVE_AFTER_COMMAND = 3
TARGET_Y_OFFSET = 0.14
TOOL_HEIGHT = 0.30
POST_EXECUTION_UPDATES = 80


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the moving-target tutorial."""
    parser = argparse.ArgumentParser(
        description="Demonstrate SceneEntityPose late-binding and replanning."
    )
    add_env_launcher_args_to_parser(parser)
    parser.add_argument("--auto_play", action="store_true")
    parser.add_argument(
        "--no_target_motion",
        action="store_true",
        help="Execute without moving the target after planning.",
    )
    return parser.parse_args()


def main() -> None:
    """Move a referenced entity during execution and follow its new pose."""
    args = parse_arguments()
    sim = create_tutorial_simulation(args)
    robot = add_ur5_gripper_robot(sim)
    target = sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid=TARGET_UID,
            shape=CubeCfg(size=[0.06, 0.06, 0.06]),
            attrs=RigidBodyAttributesCfg(enable_collision=False),
            body_type="kinematic",
            init_pos=[0.40, -0.14, 0.03],
            init_rot=[0.0, 0.0, 0.0],
        )
    )
    motion_gen = create_toppra_motion_generator(robot)
    scene_provider = RigidObjectSceneProvider({TARGET_UID: target})
    adapter = SimulationExecutionAdapter(
        sim,
        robot,
        scene_provider=scene_provider,
    )

    current_eef = robot.compute_fk(
        qpos=robot.get_qpos(name=CONTROL_PART),
        name=CONTROL_PART,
        to_matrix=True,
    )
    target_to_tool = torch.eye(
        4,
        dtype=current_eef.dtype,
        device=current_eef.device,
    )
    target_to_tool[:3, :3] = current_eef[0, :3, :3]
    target_to_tool[2, 3] = TOOL_HEIGHT
    engine = AtomicActionEngine(motion_generator=motion_gen)
    engine.register(MoveEndEffector(motion_gen))
    invocation = ActionInvocation(
        skill_id="move_end_effector",
        goal=EndEffectorPoseGoal(
            SceneEntityPose(TARGET_UID, relative_pose=target_to_tool)
        ),
        binding=ActionBinding(manipulators={"primary": CONTROL_PART}),
        motion_policy=MotionPolicy(
            sample_count=SAMPLE_COUNT,
            control_dt=2.0 * adapter.physics_dt,
        ),
        recovery_policy=RecoveryPolicy(
            max_replans=2,
            tracking_error_threshold=0.15,
            goal_translation_threshold=0.02,
            phase_timeout=20.0,
        ),
        invocation_id="moving-target-demo",
    )
    task_state = TaskState.empty(robot.get_qpos().shape[0], robot.device)
    session = engine.start((invocation,), adapter.observe(task_state))
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
        "Inspect the target, then press Enter to start moving-goal recovery...",
    )
    target_moved = False
    recovery_observed = False

    def on_step(step: RunnerStep) -> None:
        nonlocal target_moved, recovery_observed
        if (
            not args.no_target_motion
            and not target_moved
            and step.command_count >= MOVE_AFTER_COMMAND
        ):
            pose = target.get_local_pose(to_matrix=True).clone()
            pose[:, 1, 3] += TARGET_Y_OFFSET
            target.set_local_pose(pose)
            target_moved = True
            logger.log_warning(
                "Moved the referenced target; the next snapshot should rebind "
                "the goal and invalidate the active trajectory."
            )
        if step.tick is None:
            return
        for event in step.tick.events:
            if event.kind in {
                ExecutionEventKind.DYNAMIC_GOAL_CHANGED,
                ExecutionEventKind.REPLANNED,
                ExecutionEventKind.RECOVERY_EXHAUSTED,
            }:
                rows = event.env_mask.nonzero(as_tuple=False).flatten().tolist()
                logger.log_info(
                    f"Execution event {event.kind.value}: env rows={rows}; "
                    f"{event.message}"
                )
            recovery_observed |= event.kind is ExecutionEventKind.DYNAMIC_GOAL_CHANGED

    recording_started = start_auto_play_recording(
        sim,
        args,
        video_prefix="moving_target_recovery_auto_play",
    )
    try:
        result = runner.run_until_blocked(on_step=on_step)
        for _ in range(POST_EXECUTION_UPDATES):
            adapter.sleep(adapter.physics_dt)
    finally:
        stop_auto_play_recording(sim, recording_started)

    if result.status is not RunnerStatus.COMPLETED:
        raise RuntimeError(f"Closed-loop execution failed: {result.message}")
    if not args.no_target_motion and not recovery_observed:
        raise RuntimeError("Target motion did not invalidate the trajectory.")
    logger.log_info(
        f"Execution completed after {result.command_count} accepted commands.",
        color="green",
    )

    serve_tutorial_scene(sim, args)
    if wait_for_user:
        input("Press Enter to exit the simulation...")


if __name__ == "__main__":
    run_tutorial(main)
