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

"""Provider-free guards for the bounded E1 campaign helper."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.benchmark.task_program.gen_sim_e1 import (
    _device_arguments,
    _native_summary,
    _select_member,
)


def test_legacy_plan_does_not_override_runner_device_default():
    assert _device_arguments({}) == []


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_plan_explicitly_forwards_its_compute_device(device):
    assert _device_arguments({"device": device}) == ["--device", device]


@pytest.mark.parametrize("device", [None, True, 0, "cuda:0", " cuda", "mps"])
def test_invalid_device_is_rejected_before_admission(device):
    with pytest.raises(ValueError, match="compute device"):
        _device_arguments({"device": device})


@pytest.mark.parametrize(
    "uid,group",
    [
        ("block_001", "step_01__01"),
        ("block_002", "step_01__02"),
        ("block_004", "step_01"),
    ],
)
def test_selects_one_expanded_uid_without_rewriting_parent(uid, group):
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/q3_e1_candidate.json").read_text()
    )
    original = deepcopy(fixture)
    candidate, bindings, member = _select_member(
        fixture["candidate"], fixture["bindings"], uid, "diagnostic"
    )
    assert fixture == original
    assert member["group_id"] == group
    assert len(candidate["draft"]["steps"]) == 1
    step = candidate["draft"]["steps"][0]
    assert step["depends_on"] == []
    assert step["object"]["quantifier"] == "one"
    assert bindings["reference_bindings"] == {
        f"{group}.object": [uid],
        f"{group}.target": ["baking_dish_001"],
    }


def test_unknown_uid_is_not_substituted():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/q3_e1_candidate.json").read_text()
    )
    with pytest.raises(ValueError, match="exactly one"):
        _select_member(
            fixture["candidate"], fixture["bindings"], "missing", "diagnostic"
        )


@pytest.mark.parametrize(
    "report_status,program_success,spec_status,ready,expected",
    [
        ("failed", True, "unavailable", False, "unverified"),
        ("succeeded", True, "satisfied", True, "qualified"),
        ("succeeded", True, "unavailable", False, "unverified"),
        ("failed", False, "satisfied", True, "failed"),
    ],
)
def test_program_completion_is_not_taskspec_qualification(
    report_status, program_success, spec_status, ready, expected
):
    report = {
        "status": report_status,
        "runtime_result": {
            "completed": True,
            "program_success": [program_success],
            "segments": [],
        },
        "environments": [
            {
                "success": expected == "qualified",
                "terminal_reason": "task_spec_unverified",
            }
        ],
    }
    evaluation = {"environments": [{"status": spec_status, "certificate_ready": ready}]}
    summary = _native_summary(report, evaluation)
    assert summary["status"] == expected
    assert summary["layers"]["task_spec"] == spec_status
    assert summary["layers"]["runtime"] == (
        "program_completed" if program_success else "incomplete"
    )


def test_missing_evaluation_never_qualifies_green_program():
    report = {
        "status": "succeeded",
        "runtime_result": {
            "completed": True,
            "program_success": [True],
            "segments": [],
        },
        "environments": [],
    }
    summary = _native_summary(report, {})
    assert summary["status"] == "unverified"
    assert summary["layers"]["task_spec"] == "missing"
