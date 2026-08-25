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

"""Live physical held-object loss and workflow-recovery regression coverage."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest
import torch

from embodichain.lab.gym.envs import ControllerAction
from embodichain.lab.gym.envs.demo import (
    DemoEpisodeResult,
    execute_demo_episode,
)
from embodichain_tasks.configs import get_config_path
from scripts.benchmark.expert_program.demo_success import (
    DemoSuccessCase,
    _build_parser,
    load_raw_trials,
    run_gym_demo_success_benchmark,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_CUBE_GYM_CONFIG = get_config_path("gym/expert_program/repeated_pick_place.json")
_CUBE_EXPERT_PROGRAM = get_config_path("expert_program/repeated_pick_place.yaml")
_FAULT_SEGMENT_INDEX = 1
_FAULT_CALL_INDEX = 1
_FAULT_OPEN_STEPS = 20
_SUBPROCESS_TIMEOUT_SECONDS = 480
# ``_main`` closes the environment; bypass native ContactSensor interpreter teardown.
_RUN_FAULT_MAIN = (
    "import os, sys; from runpy import run_path; "
    "module = run_path("
    "'tests/benchmark/expert_program/test_cube_physical_recovery_sim.py', "
    "run_name='cube_physical_recovery_helper'); "
    "code = module['_main'](); "
    "sys.stdout.flush(); sys.stderr.flush(); os._exit(code)"
)


class _GripperOpenFaultEnvironment:
    """Replace a bounded command window with a real gripper-open command.

    The wrapper changes only the controller-ready action before the ordinary
    ``env.step()`` call. It never writes an object pose, velocity, attachment,
    constraint, or symbolic task state.
    """

    def __init__(self, environment: Any) -> None:
        self._environment = environment
        target = getattr(environment, "unwrapped", environment)
        self._hand_joint_ids = tuple(target.robot.get_joint_ids(name="hand"))
        if not self._hand_joint_ids:
            raise ValueError("The cube recovery gate requires hand joint IDs.")
        self.injected_open_steps = 0

    @property
    def unwrapped(self) -> Any:
        """Return the original task environment for demo lifecycle hooks."""
        return getattr(self._environment, "unwrapped", self._environment)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._environment, name)

    def step(self, action: object) -> object:
        """Open the physical gripper during the selected Place approach."""
        if isinstance(action, ControllerAction):
            metadata = action.metadata
            should_inject = (
                metadata.get("program_segment_index") == _FAULT_SEGMENT_INDEX
                and metadata.get("runtime_call_index") == _FAULT_CALL_INDEX
                and metadata.get("bridge_action_kind") == "runtime_command"
                and self.injected_open_steps < _FAULT_OPEN_STEPS
            )
            if should_inject:
                if not isinstance(action.value, torch.Tensor):
                    raise TypeError("The cube task must emit a tensor action.")
                value = action.value.clone()
                value[:, self._hand_joint_ids] = 0.0
                action = ControllerAction(
                    value=value,
                    metadata={
                        **metadata,
                        "test_fault": "open_gripper_before_place",
                    },
                )
                self.injected_open_steps += 1
        return self._environment.step(action)


def _execute_fault_episode(
    environment: Any,
    *,
    episode_index: int,
) -> DemoEpisodeResult:
    """Execute one episode through the controller-command fault wrapper."""
    fault_environment = _GripperOpenFaultEnvironment(environment)
    result = execute_demo_episode(
        fault_environment,
        episode_index=episode_index,
    )
    print(f"injected_open_steps={fault_environment.injected_open_steps}")
    return result


def _run_fault_subprocess(raw_path: Path, report_path: Path) -> int:
    """Create and close the native simulator inside the child process."""
    launcher_args = _build_parser().parse_args(
        [
            "--run-simulation",
            "--gym_config",
            str(_CUBE_GYM_CONFIG),
            "--expert-program",
            str(_CUBE_EXPERT_PROGRAM),
            "--case-id",
            "cube_physical_loss_recovery",
            "--seeds",
            "0",
            "--raw-json",
            str(raw_path),
            "--report",
            str(report_path),
            "--headless",
            "--device",
            "cuda",
            "--num_envs",
            "1",
            "--filter_dataset_saving",
        ]
    )
    run_gym_demo_success_benchmark(
        DemoSuccessCase("cube_physical_loss_recovery", (0,)),
        launcher_args=launcher_args,
        expert_program_path=_CUBE_EXPERT_PROGRAM,
        raw_json_path=raw_path,
        report_path=report_path,
        episode_executor=_execute_fault_episode,
    )
    return 0


@pytest.mark.requires_sim
@pytest.mark.subprocess_sim
@pytest.mark.slow
@pytest.mark.gpu
def test_physical_cube_loss_triggers_real_reacquisition(tmp_path: Path) -> None:
    """Observe physical loss, invalidate state, reacquire, and finish the task."""
    raw_path = tmp_path / "fault_raw.json"
    report_path = tmp_path / "fault_report.md"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _RUN_FAULT_MAIN,
            "--run-fault",
            "--raw-json",
            str(raw_path),
            "--report",
            str(report_path),
        ],
        cwd=_REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT_SECONDS,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert f"injected_open_steps={_FAULT_OPEN_STEPS}" in completed.stdout
    trial = load_raw_trials(raw_path)[0]
    assert trial.rows[0].success
    assert trial.rows[0].terminal_reason == "success"

    segments = trial.episode_result["segments"]
    assert isinstance(segments, list)
    assert len(segments) == 3
    recovery_segment = segments[_FAULT_SEGMENT_INDEX]
    runtime = recovery_segment["metadata"]["runtime"]
    assert runtime["status"] == "completed"
    assert runtime["masks"]["success"] == [True]
    assert runtime["task_state"]["held_objects"] == []

    failed_place = runtime["calls"][_FAULT_CALL_INDEX]
    assert failed_place["semantic_id"] == "place"
    assert failed_place["status"] == "failed"
    assert failed_place["masks"]["failed"] == [True]
    assert {event["kind"] for event in failed_place["events"]} >= {
        "held_object_lost",
        "recovery_required",
    }
    physical_failures = [
        effect
        for effect in failed_place["effects"]
        if effect["boundary"]["kind"] == "in_flight_guard"
        and effect["decision"]["failure_mask"] == [True]
    ]
    assert len(physical_failures) == 1
    physical_failure = physical_failures[0]
    assert physical_failure["decision"]["expectations"][0]["contradicted_mask"] == [
        True
    ]
    assert physical_failure["evidence"]["source.constraint"]["values"] == [False]
    assert physical_failure["evidence"]["source.pose"]["valid_mask"] == [True]

    recoveries = runtime["workflow_recoveries"]
    assert [recovery["role"] for recovery in recoveries] == [
        "reacquire",
        "retry_reacquired",
    ]
    assert [recovery["attempt_index"] for recovery in recoveries] == [1, 1]
    assert [recovery["call"]["semantic_id"] for recovery in recoveries] == [
        "pick",
        "place",
    ]
    assert all(recovery["call"]["status"] == "completed" for recovery in recoveries)
    assert recovery_segment["metadata"]["validation"]["accepted_mask"] == [True]


def _main(argv: list[str] | None = None) -> int:
    """Run only the isolated native-simulation helper mode."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-fault", action="store_true", required=True)
    parser.add_argument("--raw-json", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    return _run_fault_subprocess(args.raw_json, args.report)


if __name__ == "__main__":
    raise SystemExit(_main())
