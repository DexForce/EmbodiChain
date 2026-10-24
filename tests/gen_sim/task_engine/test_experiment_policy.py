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
from embodichain.gen_sim.task_engine.config import (
    TaskEngineExecutionCfg,
    TaskEnginePlanningCfg,
    load_task_engine_config,
)
from embodichain.gen_sim.task_engine.experiment_policy import program_completion_mask
from embodichain.gen_sim.task_engine._bundle_runner import _apply_task_spec_verdict
from embodichain.gen_sim.task_engine.workflow import _environment_successes


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


def test_experimental_acceptance_is_explicit_and_does_not_rewrite_taskspec():
    raw = runtime()
    saved = deepcopy(raw)
    assert _apply_task_spec_verdict(raw, [True], ["success"]) == (
        [False],
        ["task_spec_unverified"],
    )
    assert _apply_task_spec_verdict(
        raw, [True], ["success"], acceptance_mode="experiment"
    ) == ([True], ["success"])
    assert raw == saved


@pytest.mark.parametrize(
    "damage",
    [
        "empty",
        "partial",
        "no_steps",
        "missing_segment",
        "failed_segment",
        "runtime_failure",
        "validator_failure",
    ],
)
def test_experimental_acceptance_never_waives_execution_failure(damage):
    raw = runtime()
    if damage == "empty":
        raw = {}
    elif damage == "partial":
        raw["completed"] = False
    elif damage == "no_steps":
        raw["lengths"] = [0]
    elif damage == "missing_segment":
        raw["segments"][0]["metadata"]["segment_count"] = 2
    elif damage == "failed_segment":
        raw["segments"][0]["success"] = False
    elif damage == "runtime_failure":
        raw["segments"][0]["metadata"]["runtime"]["failures"] = ["failed"]
    elif damage == "validator_failure":
        raw["segments"][0]["metadata"]["validation"]["accepted_mask"] = [False]
    assert program_completion_mask(raw, 1) == [False]


def test_workflow_experiment_policy_requires_runner_marker_and_semantic_success():
    report = {
        "status": "succeeded",
        "failure": None,
        "runtime_result": runtime(),
        "environments": [{"success": True, "semantic_success": {"step": True}}],
    }
    assert _environment_successes(report, acceptance_mode="experiment") == [False]
    report["runtime_result"]["acceptance_mode"] = "experiment"
    assert _environment_successes(
        report, required_semantic_steps=["step"], acceptance_mode="experiment"
    ) == [True]
    report["environments"][0]["semantic_success"]["step"] = False
    assert _environment_successes(
        report, required_semantic_steps=["step"], acceptance_mode="experiment"
    ) == [False]


def test_profile_and_acceptance_defaults_remain_legacy():
    assert TaskEngineExecutionCfg().acceptance_mode == "strict"
    assert TaskEnginePlanningCfg().runtime_profile == "default"
    with pytest.raises(ValueError):
        TaskEngineExecutionCfg(acceptance_mode="anything")
    with pytest.raises(ValueError):
        TaskEngineExecutionCfg(acceptance_mode="experiment", num_envs=2)
    with pytest.raises(ValueError):
        TaskEnginePlanningCfg(runtime_profile="task_special_case")


def test_packaged_defaults_enable_automatic_qualified_policy():
    workflow, planning, execution = load_task_engine_config()
    assert workflow.max_parallel_workers == 1
    assert workflow.max_action_attempts == 1
    assert planning.runtime_profile == "auto"
    assert planning.experimental_open_release is None
    assert execution.acceptance_mode == "experiment"


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
