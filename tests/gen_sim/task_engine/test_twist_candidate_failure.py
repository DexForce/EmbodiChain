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
"""Physical receipt classification, independent of scale-search mechanics."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from embodichain.gen_sim.task_engine._task_program import twist_attempts

SOURCE_SHA = "a" * 64


def evidence() -> dict[str, Any]:
    startup_trace = [
        {
            "timestamp": time,
            "qpos": 0.0,
            "qvel": 0.0,
            "valid": True,
            "dropped_contacts": 0,
            "contact_pairs": [],
            "impulses": [],
            "table_target_contact": False,
            "robot_target_contact": False,
        }
        for time in (0.0, 0.01)
    ]
    return {
        "route": {
            "binding": {
                "object_id": "control",
                "link": "knob",
                "source_sha256": SOURCE_SHA,
            }
        },
        "target_actor_id": 20,
        "parent_actor_id": 21,
        "finger_actor_ids": [10, 11],
        "geometry_evidence": {
            "source_sha256": SOURCE_SHA,
            "source_edited": False,
            "mass_properties": {
                "verified": True,
                "source_sha256": SOURCE_SHA,
                "source_edited": False,
            },
        },
        "startup_evidence": {
            "complete": True,
            "overflow": False,
            "summary": {
                "accepted": True,
                "reason": "startup_stable",
                "sample_count": 2,
            },
            "trace": startup_trace,
        },
        "acceptance": {
            "accepted": True,
            "phase": "initial_stable",
            "reason": "initial_stable",
        },
        "trace": [
            {
                "timestamp": time,
                "qpos": 0.0,
                "phase": "twist",
                "valid": True,
                "dropped_contacts": 0,
                "target_contact": False,
                "parent_contact": False,
                "contact_pairs": [],
                "impulses": [],
            }
            for time in (1.0, 1.01)
        ],
    }


def table_report() -> dict[str, Any]:
    event = {
        "kind": "action_planning_failed",
        "skill_id": "twist",
        "invocation_id": "turn",
        "invocation_revision": 0,
        "failure_code": "e8_table_clearance_rejected",
        "retryable": False,
        "env_mask": [True],
    }
    call = {
        "semantic_id": "gen_sim.twist",
        "skill_id": "twist",
        "invocation_id": "turn",
        "status": "failed",
        "command_count": 0,
        "events": [event],
        "plan_attempts": [
            {
                "plan_success_mask": [False],
                "planner_diagnostics": {
                    "metadata": {
                        "gen_sim_twist_table_clearance": {
                            "supported": True,
                            "accepted": False,
                            "minimum_clearance_after_skin_m": -0.001,
                        }
                    },
                    "failure": {
                        "code": "e8_table_clearance_rejected",
                        "retryable": False,
                    },
                },
            }
        ],
    }
    return {
        "status": "failed",
        "failure": None,
        "runtime_result": {
            "completed": False,
            "terminal_reason": "segment_validation_failed",
            "segments": [
                {
                    "success": False,
                    "failure_reason": "segment_validation_failed",
                    "metadata": {"runtime": {"events": [event], "calls": [call]}},
                }
            ],
        },
    }


def write_receipts(output: Path, item: dict[str, Any], report: dict[str, Any]) -> None:
    (output / "twist_evidence.json").write_text(json.dumps(item), encoding="utf-8")
    (output / "execution_report.json").write_text(json.dumps(report), encoding="utf-8")


def test_supported_table_rejection_requires_terminal_typed_planning_failure(
    tmp_path: Path,
) -> None:
    item, report = evidence(), table_report()
    write_receipts(tmp_path, item, report)
    assert twist_attempts.size_failure(tmp_path)
    assert twist_attempts.size_failure(tmp_path, report)


@pytest.mark.parametrize(
    "mutation",
    (
        "unsupported",
        "accepted",
        "missing_screen",
        "missing_failure",
        "nonterminal",
        "other_semantic",
        "succeeded",
    ),
)
def test_table_diagnostic_alone_is_not_size_failure(
    tmp_path: Path, mutation: str
) -> None:
    item, report = evidence(), table_report()
    call = report["runtime_result"]["segments"][0]["metadata"]["runtime"]["calls"][0]
    diagnostic = call["plan_attempts"][0]["planner_diagnostics"]
    if mutation == "unsupported":
        diagnostic["metadata"]["gen_sim_twist_table_clearance"]["supported"] = False
    elif mutation == "accepted":
        diagnostic["metadata"]["gen_sim_twist_table_clearance"]["accepted"] = True
    elif mutation == "missing_screen":
        diagnostic["metadata"] = {}
    elif mutation == "missing_failure":
        diagnostic.pop("failure")
        call["events"] = []
        report["runtime_result"]["segments"][0]["metadata"]["runtime"]["events"] = []
    elif mutation == "nonterminal":
        call["status"] = "running"
        report["status"] = "running"
    elif mutation == "other_semantic":
        call["semantic_id"] = "gen_sim.press"
    else:
        report["status"] = "succeeded"
        report["runtime_result"]["completed"] = True
    write_receipts(tmp_path, item, report)
    assert not twist_attempts.size_failure(tmp_path)


@pytest.mark.parametrize(
    "mutation",
    (
        "mass_invalid",
        "mass_missing",
        "source_mismatch",
        "mass_source_mismatch",
        "source_edited",
        "startup_failed",
        "startup_incomplete",
        "startup_invalid_sample",
        "startup_dropped",
        "invalid_sample",
        "dropped",
        "missing_evidence",
    ),
)
def test_unqualified_physical_evidence_cannot_trigger_scale(
    tmp_path: Path, mutation: str
) -> None:
    item, report = evidence(), table_report()
    if mutation == "mass_invalid":
        item["geometry_evidence"]["mass_properties"]["verified"] = False
    elif mutation == "mass_missing":
        item["geometry_evidence"].pop("mass_properties")
    elif mutation == "source_mismatch":
        item["geometry_evidence"]["source_sha256"] = "b" * 64
    elif mutation == "mass_source_mismatch":
        item["geometry_evidence"]["mass_properties"]["source_sha256"] = "b" * 64
    elif mutation == "source_edited":
        item["geometry_evidence"]["mass_properties"]["source_edited"] = True
    elif mutation == "startup_failed":
        item["startup_evidence"]["summary"].update(
            accepted=False, reason="startup_motion"
        )
    elif mutation == "startup_incomplete":
        item["startup_evidence"]["complete"] = False
    elif mutation == "startup_invalid_sample":
        item["startup_evidence"]["trace"][0]["valid"] = False
    elif mutation == "startup_dropped":
        item["startup_evidence"]["trace"][0]["dropped_contacts"] = 1
    elif mutation == "invalid_sample":
        item["trace"][0]["valid"] = False
    elif mutation == "dropped":
        item["trace"][0]["dropped_contacts"] = 1
    write_receipts(tmp_path, item, report)
    if mutation == "missing_evidence":
        (tmp_path / "twist_evidence.json").unlink()
    assert not twist_attempts.size_failure(tmp_path)


@pytest.mark.parametrize(
    "code",
    (
        "ik_failed",
        "joint_velocity_limit",
        "e8_velocity_rejected",
        "resource_interrupted",
        "native_assertion",
        "source_hash_mismatch",
        "native_mass_mismatch",
    ),
)
def test_non_size_terminal_failure_wins_over_stale_table_diagnostic(
    tmp_path: Path, code: str
) -> None:
    item, report = evidence(), table_report()
    call = report["runtime_result"]["segments"][0]["metadata"]["runtime"]["calls"][0]
    call["events"].append(
        {
            "kind": "action_planning_failed",
            "failure_code": code,
            "retryable": False,
            "env_mask": [True],
        }
    )
    write_receipts(tmp_path, item, report)
    assert not twist_attempts.size_failure(tmp_path)


def no_contact_receipts() -> tuple[dict[str, Any], dict[str, Any]]:
    item = evidence()
    item["acceptance"] = {
        "accepted": False,
        "phase": "bounded_chunk",
        "reason": "target_contact_not_observed",
        "candidate_fallback_count": 2,
        "candidate_fallback_limit": 3,
        "candidate_exhausted": True,
        "candidate_order": [3, 7, 11],
        "candidate_selected": 11,
        "candidate_attempt_count": 3,
        "target_contact_seen": False,
    }
    report = {
        "status": "failed",
        "runtime_result": {
            "completed": False,
            "terminal_reason": "segment_validation_failed",
            "segments": [],
        },
        "failure": None,
    }
    return item, report


def test_only_exhausted_no_contact_receipt_is_size_retry_eligible(
    tmp_path: Path,
) -> None:
    item, report = no_contact_receipts()
    write_receipts(tmp_path, item, report)
    assert twist_attempts.size_failure(tmp_path)


@pytest.mark.parametrize(
    "mutation",
    (
        "before_exhaustion",
        "no_limit",
        "missing_marker",
        "marker_false",
        "selected_not_last",
        "attempt_count_mismatch",
        "nonterminal",
        "contact_seen",
        "raw_contact_seen",
        "missed_target",
        "invalid_observation",
        "accepted",
    ),
)
def test_no_contact_reason_cannot_replace_exhaustion_and_validity(
    tmp_path: Path, mutation: str
) -> None:
    item, report = no_contact_receipts()
    accepted = item["acceptance"]
    if mutation == "before_exhaustion":
        accepted["candidate_fallback_count"] = 1
    elif mutation == "no_limit":
        accepted.pop("candidate_fallback_limit")
    elif mutation == "missing_marker":
        accepted.pop("candidate_exhausted")
    elif mutation == "marker_false":
        accepted["candidate_exhausted"] = False
    elif mutation == "selected_not_last":
        accepted["candidate_selected"] = 7
    elif mutation == "attempt_count_mismatch":
        accepted["candidate_attempt_count"] = 4
    elif mutation == "nonterminal":
        report["status"] = "running"
    elif mutation == "contact_seen":
        item["trace"][0]["target_contact"] = True
    elif mutation == "raw_contact_seen":
        item["trace"][0].update(contact_pairs=[[10, 20]], impulses=[0.01])
    elif mutation == "missed_target":
        accepted["reason"] = "target_qpos_not_reached"
    elif mutation == "invalid_observation":
        accepted["reason"] = "invalid_observation"
    else:
        accepted["accepted"] = True
    write_receipts(tmp_path, item, report)
    assert not twist_attempts.size_failure(tmp_path)


def test_missing_or_corrupt_receipts_fail_closed(tmp_path: Path) -> None:
    assert not twist_attempts.size_failure(tmp_path)
    (tmp_path / "twist_evidence.json").write_text("{", encoding="utf-8")
    (tmp_path / "execution_report.json").write_text("{", encoding="utf-8")
    assert not twist_attempts.size_failure(tmp_path)


@pytest.mark.parametrize(
    "mutation", ("runtime_null", "report_list", "summary_null", "diagnostic_null")
)
def test_json_valid_but_malformed_receipt_shapes_fail_closed(
    tmp_path: Path, mutation: str
) -> None:
    item, report = evidence(), table_report()
    if mutation == "runtime_null":
        report["runtime_result"] = None
    elif mutation == "report_list":
        report = []
    elif mutation == "summary_null":
        item["startup_evidence"]["summary"] = None
    else:
        report["runtime_result"]["segments"][0]["metadata"]["runtime"]["calls"][0][
            "plan_attempts"
        ][0]["planner_diagnostics"] = None
    write_receipts(tmp_path, item, report)
    assert not twist_attempts.size_failure(tmp_path)


def test_supplied_report_does_not_mutate_receipt_or_input(tmp_path: Path) -> None:
    item, report = evidence(), table_report()
    write_receipts(tmp_path, item, report)
    before = deepcopy(report)
    saved = (tmp_path / "twist_evidence.json").read_bytes()
    assert twist_attempts.size_failure(tmp_path, report)
    assert report == before and (tmp_path / "twist_evidence.json").read_bytes() == saved
