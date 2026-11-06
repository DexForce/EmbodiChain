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

from __future__ import annotations

from copy import deepcopy
import pytest
from embodichain.gen_sim.task_engine.experiment_policy import program_completion_mask


def runtime():
    return {
        "completed": True,
        "completed_by_env": [True],
        "success": [True],
        "lengths": [20],
        "segments": [
            {
                "success": True,
                "metadata": {
                    "segment_count": 1,
                    "runtime": {"failures": []},
                    "validation": {"accepted_mask": [True]},
                },
            }
        ],
        "task_spec_evaluation": {
            "environments": [
                {"env_id": 0, "status": "unavailable", "certificate_ready": False}
            ]
        },
    }


def test_program_completion_summary_cannot_replace_strict_failure():
    from embodichain.gen_sim.task_engine.experiment_policy import (
        program_completion_summary,
    )

    raw = runtime()
    strict = {"environments": [{"success": False}], "failure": None}
    experiment = {"environments": [{"success": True}], "failure": None}
    summary = program_completion_summary(
        raw, strict, experiment, acceptance_mode="strict"
    )
    assert summary["experimental_program_completed_by_env"] == [True]
    assert summary["strict_execution_success_by_env"] == [False]
    assert summary["task_spec_certificate_ready_by_env"] == [False]
    assert summary["affects_selected_success"] is False
    assert strict["environments"][0]["success"] is False


@pytest.mark.parametrize("failure", [False, True])
def test_program_completion_summary_keeps_execution_failure_separate(failure):
    from embodichain.gen_sim.task_engine.experiment_policy import (
        program_completion_summary,
    )

    raw = runtime()
    raw["task_spec_evaluation"]["environments"][0].update(
        status="satisfied", certificate_ready=True
    )
    report = {
        "environments": [{"env_id": 0, "success": not failure}],
        "failure": {"message": "execution failed"} if failure else None,
    }
    summary = program_completion_summary(raw, report, report, acceptance_mode="strict")
    assert summary["experimental_program_completed_by_env"] == [not failure]
    assert summary["strict_execution_success_by_env"] == [not failure]
    assert summary["task_spec_certificate_ready_by_env"] == [True]


@pytest.mark.parametrize("rejection", ["cargo", "semantic"])
def test_program_completion_summary_preserves_experiment_postconditions(rejection):
    from embodichain.gen_sim.task_engine.experiment_policy import (
        program_completion_summary,
    )

    strict = {"environments": [{"success": False}], "failure": None}
    experiment = {
        "environments": [
            {
                "success": rejection != "cargo",
                "semantic_success": {"step": rejection != "semantic"},
            }
        ],
        "failure": None,
    }
    summary = program_completion_summary(
        runtime(), strict, experiment, acceptance_mode="experiment"
    )
    assert summary["experimental_program_completed_by_env"] == [False]
