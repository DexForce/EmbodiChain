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

"""Live OpenDrawer regression coverage for the Expert Program benchmark."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.benchmark.expert_program.demo_success import (
    aggregate_demo_success_trials,
    load_raw_trials,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_TASK_CONFIG_ROOT = _REPOSITORY_ROOT / "embodichain_tasks/configs"
_OPEN_DRAWER_GYM_CONFIG = _TASK_CONFIG_ROOT / "gym/open_drawer/cobot_magic_3cam.json"
_OPEN_DRAWER_EXPERT_PROGRAM = (
    _TASK_CONFIG_ROOT / "expert_program/tableware/open_drawer.json"
)
_CASE_ID = "open_drawer_live"
_SEED = 0
_NUM_ENVS = 1
_SUBPROCESS_TIMEOUT_SECONDS = 180
_RUN_PUBLIC_MAIN = (
    "from scripts.benchmark.expert_program.demo_success import main; "
    "raise SystemExit(main())"
)


def _write_headless_cpu_gym_config(tmp_path: Path) -> Path:
    """Write a camera-free copy of the packaged live-physics configuration."""
    payload = json.loads(_OPEN_DRAWER_GYM_CONFIG.read_text(encoding="utf-8"))
    if type(payload) is not dict:
        raise TypeError("The packaged OpenDrawer Gym config must be a JSON object.")
    env_config = payload.get("env")
    if type(env_config) is not dict:
        raise TypeError("The packaged OpenDrawer env config must be a JSON object.")

    # Cameras and their recording event are orthogonal to drawer physics and make
    # this CPU regression unnecessarily renderer-sensitive.
    payload["sensor"] = []
    env_config["events"] = {}
    env_config["observations"] = {}
    env_config["dataset"] = {}
    payload["expert_program_path"] = str(_OPEN_DRAWER_EXPERT_PROGRAM)

    output = tmp_path / "open_drawer_headless_cpu.json"
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output


@pytest.mark.requires_sim
@pytest.mark.slow
def test_live_open_drawer_benchmark_writes_successful_decodable_artifacts(
    tmp_path: Path,
) -> None:
    """Run one no-retry seed through the public live benchmark entry point."""
    gym_config_path = _write_headless_cpu_gym_config(tmp_path)
    raw_path = tmp_path / "open_drawer_raw.json"
    report_path = tmp_path / "open_drawer_report.md"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _RUN_PUBLIC_MAIN,
            "--run-simulation",
            "--gym_config",
            str(gym_config_path),
            "--expert-program",
            str(_OPEN_DRAWER_EXPERT_PROGRAM),
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
            "cpu",
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

    # main() returns zero only after the live runner's default closer completes;
    # the process boundary also isolates native simulator teardown from pytest.
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert f"Raw JSON saved: {raw_path}" in completed.stdout
    assert f"Markdown report saved: {report_path}" in completed.stdout

    decoded_trials = load_raw_trials(raw_path)
    assert len(decoded_trials) == 1
    trial = decoded_trials[0]
    assert trial.case_id == _CASE_ID
    assert trial.seed == _SEED
    assert len(trial.rows) == _NUM_ENVS
    row = trial.rows[0]
    assert row.success
    assert row.terminal_reason == "success"
    assert row.length > 0

    segments = trial.episode_result["segments"]
    assert isinstance(segments, list)
    assert len(segments) == 1
    segment = segments[0]
    assert isinstance(segment, dict)
    assert segment["name"] == "open_drawer"
    runtime = segment["metadata"]["runtime"]
    assert runtime["kind"] == "skill_result"
    assert runtime["status"] == "completed"
    calls = runtime["calls"]
    assert isinstance(calls, list)
    assert len(calls) == 1
    call = calls[0]
    assert call["semantic_id"] == "embodichain_tasks.open_drawer"
    assert call["skill_id"] == "slide"
    assert call["status"] == "completed"
    assert call["effects"] == []
    validation = segment["metadata"]["validation"]
    assert validation["accepted_mask"] == [True]
    validators = validation["validators"]
    assert len(validators) == 1
    assert validators[0]["kind"] == "articulation_joint_position"
    assert validators[0]["result_mask"] == [True]
    assert validators[0]["result"]["joint"] == "slide_rails"
    assert validators[0]["result"]["minimum_position"] == pytest.approx(0.09)
    assert validators[0]["result"]["accepted_mask"] == [True]

    aggregates = aggregate_demo_success_trials(decoded_trials)
    assert len(aggregates.success_and_metrics) == 1
    metrics = aggregates.success_and_metrics[0]
    assert metrics["attempted"] == 1
    assert metrics["successes"] == 1
    assert metrics["success_rate"] == 1.0

    report = report_path.read_text(encoding="utf-8")
    assert report.count("\n## ") == 3
    assert "## Success & Other Metrics" in report
    assert "## Leaderboard" in report
    assert _CASE_ID in report
