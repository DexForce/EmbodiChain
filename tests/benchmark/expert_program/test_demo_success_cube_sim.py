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

"""Live repeated-cube regression coverage for the Expert Program benchmark."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from embodichain_tasks.configs import get_config_path
from scripts.benchmark.expert_program.demo_success import (
    aggregate_demo_success_trials,
    load_raw_trials,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_CUBE_GYM_CONFIG = get_config_path("gym/multi_segments/cube_pick_place.json")
_CUBE_EXPERT_PROGRAM = get_config_path(
    "expert_program/multi_segments/repeated_cube_pick_place.yaml"
)
_CASE_ID = "repeated_cube_three_cycle_live"
_SEED = 0
_NUM_ENVS = 1
_SUBPROCESS_TIMEOUT_SECONDS = 180
_RUN_PUBLIC_MAIN = (
    "from scripts.benchmark.expert_program.demo_success import main; "
    "raise SystemExit(main())"
)


def _successful_effect_decisions(call: dict[str, object]) -> list[dict[str, object]]:
    """Return successful physical-effect observations from one call trace."""
    effects = call["effects"]
    assert isinstance(effects, list)
    return [
        effect
        for effect in effects
        if isinstance(effect, dict)
        and isinstance(effect.get("decision"), dict)
        and effect["decision"].get("success_mask") == [True]
    ]


@pytest.mark.requires_sim
@pytest.mark.slow
@pytest.mark.gpu
def test_live_repeated_cube_completes_three_physical_cycles(
    tmp_path: Path,
) -> None:
    """Run all three no-retry segments through the public live entry point."""
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
            "--expert-program",
            str(_CUBE_EXPERT_PROGRAM),
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

    for segment_index, (segment, target_index) in enumerate(
        zip(segments, (0, 1, 0), strict=True)
    ):
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
        assert all(_successful_effect_decisions(call) for call in calls)

        post_policies = metadata["post_policies"]
        assert len(post_policies) == 1
        assert post_policies[0]["kind"] == "wait_stable"
        assert post_policies[0]["result"]["status"] == "settled"
        validation = metadata["validation"]
        assert validation["accepted_mask"] == [True]
        validators = validation["validators"]
        assert len(validators) == 1
        assert validators[0]["result"]["target_value_index"] == target_index
        assert validators[0]["result"]["accepted_mask"] == [True]

    aggregates = aggregate_demo_success_trials(decoded_trials)
    metrics = aggregates.success_and_metrics[0]
    assert metrics["attempted"] == 1
    assert metrics["successes"] == 1
    assert metrics["success_rate"] == 1.0
