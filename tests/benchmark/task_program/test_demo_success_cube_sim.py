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

"""Live repeated-cube regression coverage for the Task Program benchmark."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from embodichain_tasks.configs import get_config_path
from scripts.benchmark.task_program.demo_success import (
    aggregate_demo_success_trials,
    load_raw_trials,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_CUBE_GYM_CONFIG = get_config_path(
    "tasks/manipulation/repeated_pick_place/env.ur5.yaml"
)
_CUBE_TASK_PROGRAM = get_config_path(
    "tasks/manipulation/repeated_pick_place/task_program/program.yaml"
)
_CASE_ID = "repeated_cube_three_cycle_live"
_SEED = 0
_NUM_ENVS = 1
_SUBPROCESS_TIMEOUT_SECONDS = 300
# ``main`` closes the environment; bypass native simulator interpreter teardown.
_RUN_PUBLIC_MAIN = (
    "import os, sys; "
    "from scripts.benchmark.task_program.demo_success import main; "
    "code = main(); sys.stdout.flush(); sys.stderr.flush(); os._exit(code)"
)


@pytest.mark.requires_sim
@pytest.mark.subprocess_sim
@pytest.mark.slow
@pytest.mark.gpu
def test_live_repeated_cube_completes_three_trajectory_cycles(
    tmp_path: Path,
) -> None:
    """Run all three open-loop trajectory segments through the public entry point."""
    raw_path = tmp_path / "cube_raw.json"
    report_path = tmp_path / "cube_report.md"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _RUN_PUBLIC_MAIN,
            "--run-simulation",
            "--gym_config",
            str(_CUBE_GYM_CONFIG),
            "--task-program",
            str(_CUBE_TASK_PROGRAM),
            "--case-id",
            _CASE_ID,
            "--seeds",
            str(_SEED),
            "--raw-json",
            str(raw_path),
            "--report",
            str(report_path),
            "--headless",
            "--device",
            "cuda",
            "--num_envs",
            str(_NUM_ENVS),
            "--filter_dataset_saving",
        ],
        cwd=_REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT_SECONDS,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    decoded_trials = load_raw_trials(raw_path)
    assert len(decoded_trials) == 1
    trial = decoded_trials[0]
    assert trial.case_id == _CASE_ID
    assert trial.seed == _SEED
    assert trial.rows[0].success
    assert trial.rows[0].terminal_reason == "success"

    episode = trial.episode_result
    assert episode["completed"] is True
    assert episode["success"] == [True]
    assert episode["terminal_reason"] == "success"
    segments = episode["segments"]
    assert isinstance(segments, list)
    assert len(segments) == 3

    for segment_index, segment in enumerate(segments):
        assert isinstance(segment, dict)
        assert segment["segment_id"] == segment_index
        assert segment["name"] == "move_cube"
        metadata = segment["metadata"]
        assert isinstance(metadata, dict)
        runtime = metadata["runtime"]
        assert runtime["status"] == "completed"
        assert runtime["masks"]["success"] == [True]
        calls = runtime["calls"]
        assert [call["semantic_id"] for call in calls] == ["pick", "place"]
        assert all(call["status"] == "completed" for call in calls)
        assert all(call["effects"] == [] for call in calls)

        assert metadata["post_policies"] == []
        validation = metadata["validation"]
        assert validation["accepted_mask"] == [True]
        assert validation["validators"] == []

    aggregates = aggregate_demo_success_trials(decoded_trials)
    metrics = aggregates.success_and_metrics[0]
    assert metrics["attempted"] == 1
    assert metrics["successes"] == 1
    assert metrics["success_rate"] == 1.0
