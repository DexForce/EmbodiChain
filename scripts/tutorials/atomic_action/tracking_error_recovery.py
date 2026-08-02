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

"""Demonstrate closed-loop recovery from an injected joint tracking error."""

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
    ExecutionEventKind,
    ExecutionRunner,
    ExecutionRunnerCfg,
    JointPositionGoal,
    MotionPolicy,
    PlanningContext,
    RecoveryPolicy,
    RunnerStatus,
    RunnerStep,
    SimulationExecutionAdapter,
    TaskState,
)
from embodichain.lab.sim.objects import Robot
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

SAMPLE_COUNT = 80
INJECTION_AFTER_COMMAND = 3
TRACKING_ERROR_OFFSET = 0.35
TRACKING_ERROR_THRESHOLD = 0.08
POST_EXECUTION_UPDATES = 80


class _OneShotTrackingErrorInjector:
    """Decorate simulation observations with one deterministic disturbance."""

    def __init__(
        self,
        adapter: SimulationExecutionAdapter,
        robot: Robot,
        *,
        joint_id: int,
        offset: float,
    ) -> None:
        self._adapter = adapter
        self._robot = robot
        self._joint_id = joint_id
        self._offset = offset
        self._pending = False
        self.injected = False

    def arm(self) -> None:
        """Request a disturbance immediately before the next observation."""
        if not self.injected:
            self._pending = True

    def observe(self, task_state: TaskState) -> PlanningContext:
        """Inject one physical-state offset, then capture the observation.

        Args:
            task_state: Session-owned verified task state.

        Returns:
            Latest simulation planning context.
        """
        if self._pending:
            qpos = self._robot.get_qpos().clone()
            qpos[:, self._joint_id] += self._offset
            self._robot.set_qpos(qpos, target=False)
            self._robot.set_qvel(torch.zeros_like(qpos), target=False)
            self._pending = False
            self.injected = True
            logger.log_warning(
                "Injected a joint-position disturbance before observation; "
                "the session should detect tracking error and replan."
            )
        return self._adapter.observe(task_state)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the recovery tutorial."""
    parser = argparse.ArgumentParser(
        description="Demonstrate ExecutionRunner tracking-error recovery."
    )
    add_env_launcher_args_to_parser(parser)
    parser.add_argument("--auto_play", action="store_true")
    parser.add_argument(
        "--no_error_injection",
        action="store_true",
        help="Run the closed-loop trajectory without the demonstration disturbance.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute MoveJoints and recover after a one-shot state disturbance."""
    args = parse_arguments()
    sim = create_tutorial_simulation(args)
    robot = add_ur5_gripper_robot(sim)
    motion_gen = create_toppra_motion_generator(robot)
    adapter = SimulationExecutionAdapter(sim, robot)

    target = torch.tensor(
        [0.35, -1.20, 1.30, -1.65, -1.57, 0.20],
        dtype=torch.float32,
        device=sim.device,
    )
    engine = AtomicActionEngine(motion_generator=motion_gen)
    invocation = ActionInvocation(
        skill_id="move_joints",
        goal=JointPositionGoal(target),
        binding=ActionBinding(manipulators={"primary": "arm"}),
        motion_policy=MotionPolicy(
            sample_count=SAMPLE_COUNT,
            control_dt=2.0 * adapter.physics_dt,
        ),
        recovery_policy=RecoveryPolicy(
            max_replans=2,
            tracking_error_threshold=TRACKING_ERROR_THRESHOLD,
            phase_timeout=20.0,
        ),
    )
    task_state = TaskState.empty(robot.get_qpos().shape[0], robot.device)
    initial_context = adapter.observe(task_state)
    session = engine.start((invocation,), initial_context)

    arm_joint_id = robot.get_joint_ids(name="arm")[0]
    observation_provider = _OneShotTrackingErrorInjector(
        adapter,
        robot,
        joint_id=arm_joint_id,
        offset=TRACKING_ERROR_OFFSET,
    )
    runner = ExecutionRunner(
        session,
        observation_provider,
        adapter,
        clock=adapter,
        cfg=ExecutionRunnerCfg(minimum_cycle_time=adapter.physics_dt),
    )

    wait_for_user = prepare_tutorial_scene(
        sim,
        args,
        "Inspect the robot, then press Enter to start closed-loop execution...",
    )
    recovery_observed = False

    def on_step(step: RunnerStep) -> None:
        nonlocal recovery_observed
        if (
            not args.no_error_injection
            and not observation_provider.injected
            and step.command_count >= INJECTION_AFTER_COMMAND
        ):
            observation_provider.arm()
        if step.tick is None:
            return
        for event in step.tick.events:
            if event.kind in {
                ExecutionEventKind.TRACKING_ERROR,
                ExecutionEventKind.REPLANNED,
                ExecutionEventKind.RECOVERY_EXHAUSTED,
            }:
                env_ids = event.env_mask.nonzero(as_tuple=False).flatten().tolist()
                logger.log_info(
                    f"Execution event {event.kind.value}: env rows={env_ids}; "
                    f"{event.message}"
                )
            recovery_observed |= event.kind is ExecutionEventKind.REPLANNED

    recording_started = start_auto_play_recording(
        sim,
        args,
        video_prefix="tracking_error_recovery_auto_play",
    )
    try:
        result = runner.run_until_blocked(on_step=on_step)
        for _ in range(POST_EXECUTION_UPDATES):
            adapter.sleep(adapter.physics_dt)
    finally:
        stop_auto_play_recording(sim, recording_started)

    if result.status is not RunnerStatus.COMPLETED:
        raise RuntimeError(f"Closed-loop execution failed: {result.message}")
    if not args.no_error_injection and not recovery_observed:
        raise RuntimeError("The injected tracking error did not trigger replanning.")
    logger.log_info(
        f"Execution completed after {result.command_count} accepted commands.",
        color="green",
    )

    serve_tutorial_scene(sim, args)
    if wait_for_user:
        input("Press Enter to exit the simulation...")


if __name__ == "__main__":
    run_tutorial(main)
