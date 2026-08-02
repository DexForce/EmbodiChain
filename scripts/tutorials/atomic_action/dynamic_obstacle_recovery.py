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

"""Demonstrate collision-world invalidation and dynamic obstacle replanning."""

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
    SimulationExecutionAdapter,
    TaskState,
)
from embodichain.lab.sim.cfg import RigidBodyAttributesCfg
from embodichain.lab.sim.objects import RigidObjectCfg
from embodichain.lab.sim.planners import MotionGenCfg, MotionGenerator
from embodichain.lab.sim.planners.curobo.curobo_planner import (
    CuroboPlannerCfg,
    CuroboWorldCfg,
)
from embodichain.lab.sim.robots import FrankaPandaCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.utils import logger
from scripts.tutorials.atomic_action.tutorial_utils import (
    create_tutorial_simulation,
    prepare_tutorial_scene,
    run_tutorial,
    serve_tutorial_scene,
    start_auto_play_recording,
    stop_auto_play_recording,
)

ROBOT_UID = "dynamic_scene_franka"
OBSTACLE_UID = "dynamic_obstacle"
CONTROL_PART = "arm"
SAMPLE_COUNT = 80
MOVE_AFTER_COMMAND = 3
OBSTACLE_Y_OFFSET = 0.18
POST_EXECUTION_UPDATES = 80


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the dynamic-obstacle tutorial."""
    parser = argparse.ArgumentParser(
        description="Demonstrate collision-world revision recovery with cuRobo."
    )
    add_env_launcher_args_to_parser(parser)
    parser.add_argument("--auto_play", action="store_true")
    parser.add_argument(
        "--no_obstacle_motion",
        action="store_true",
        help="Execute without moving the obstacle after planning.",
    )
    return parser.parse_args()


def main() -> None:
    """Move an obstacle during execution and replan from the latest snapshot."""
    args = parse_arguments()
    sim = create_tutorial_simulation(args)
    robot = sim.add_robot(
        cfg=FrankaPandaCfg.from_dict({"uid": ROBOT_UID, "robot_type": "panda"})
    )
    obstacle = sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid=OBSTACLE_UID,
            shape=CubeCfg(size=[0.16, 0.18, 0.30]),
            attrs=RigidBodyAttributesCfg(),
            body_type="kinematic",
            init_pos=[0.45, -0.20, 0.20],
            init_rot=[0.0, 0.0, 0.0],
        )
    )
    motion_gen = MotionGenerator(
        MotionGenCfg(
            planner_cfg=CuroboPlannerCfg(
                robot_uid=ROBOT_UID,
                world=CuroboWorldCfg(
                    rigid_objects=[obstacle],
                    obstacle_representation="cuboid",
                    dynamic_obstacle_names=[OBSTACLE_UID],
                    multi_env=args.num_envs > 1,
                ),
            )
        )
    )
    scene_provider = RigidObjectSceneProvider(
        {OBSTACLE_UID: obstacle},
        collision_entity_ids=(OBSTACLE_UID,),
    )
    adapter = SimulationExecutionAdapter(
        sim,
        robot,
        scene_provider=scene_provider,
    )

    current_pose = robot.compute_fk(
        qpos=robot.get_qpos(name=CONTROL_PART),
        name=CONTROL_PART,
        to_matrix=True,
    )
    target_pose = current_pose.clone()
    target_pose[:, :3, 3] += torch.tensor(
        [0.22, 0.24, 0.12],
        dtype=target_pose.dtype,
        device=target_pose.device,
    )
    engine = AtomicActionEngine(motion_generator=motion_gen)
    engine.register(MoveEndEffector(motion_gen))
    invocation = ActionInvocation(
        skill_id="move_end_effector",
        goal=EndEffectorPoseGoal(target_pose),
        binding=ActionBinding(manipulators={"primary": CONTROL_PART}),
        motion_policy=MotionPolicy(
            motion_source="motion_gen",
            sample_count=SAMPLE_COUNT,
        ),
        recovery_policy=RecoveryPolicy(
            max_replans=2,
            tracking_error_threshold=0.20,
            phase_timeout=30.0,
        ),
        invocation_id="dynamic-obstacle-demo",
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
        "Inspect the scene, then press Enter to start dynamic replanning...",
    )
    obstacle_moved = False
    recovery_observed = False

    def on_step(step: RunnerStep) -> None:
        nonlocal obstacle_moved, recovery_observed
        if (
            not args.no_obstacle_motion
            and not obstacle_moved
            and step.command_count >= MOVE_AFTER_COMMAND
        ):
            pose = obstacle.get_local_pose(to_matrix=True).clone()
            pose[:, 1, 3] += OBSTACLE_Y_OFFSET
            obstacle.set_local_pose(pose)
            obstacle_moved = True
            logger.log_warning(
                "Moved the collision obstacle; the next scene snapshot should "
                "invalidate the active trajectory."
            )
        if step.tick is None:
            return
        for event in step.tick.events:
            if event.kind in {
                ExecutionEventKind.COLLISION_WORLD_CHANGED,
                ExecutionEventKind.REPLANNED,
                ExecutionEventKind.RECOVERY_EXHAUSTED,
            }:
                rows = event.env_mask.nonzero(as_tuple=False).flatten().tolist()
                logger.log_info(
                    f"Execution event {event.kind.value}: env rows={rows}; "
                    f"{event.message}"
                )
            recovery_observed |= (
                event.kind is ExecutionEventKind.COLLISION_WORLD_CHANGED
            )

    recording_started = start_auto_play_recording(
        sim,
        args,
        video_prefix="dynamic_obstacle_recovery_auto_play",
    )
    try:
        result = runner.run_until_blocked(on_step=on_step)
        for _ in range(POST_EXECUTION_UPDATES):
            adapter.sleep(adapter.physics_dt)
    finally:
        stop_auto_play_recording(sim, recording_started)

    if result.status is not RunnerStatus.COMPLETED:
        raise RuntimeError(f"Closed-loop execution failed: {result.message}")
    if not args.no_obstacle_motion and not recovery_observed:
        raise RuntimeError("Obstacle motion did not invalidate the trajectory.")
    logger.log_info(
        f"Execution completed after {result.command_count} accepted commands.",
        color="green",
    )

    serve_tutorial_scene(sim, args)
    if wait_for_user:
        input("Press Enter to exit the simulation...")


if __name__ == "__main__":
    run_tutorial(main)
