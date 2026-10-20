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

from embodichain.gen_sim.task_engine._task_group_results import task_group_results


def _graph() -> dict:
    return {
        "nodes": [{"id": name, "call": {"kind": "pick"}} for name in ("a", "b", "c")],
        "task_groups": [
            {"id": "first", "task_type": "E1", "node_ids": ["a"], "depends_on": []},
            {
                "id": "second",
                "task_type": "E3",
                "node_ids": ["b"],
                "depends_on": ["first"],
            },
            {
                "id": "third",
                "task_type": "E6",
                "node_ids": ["c"],
                "depends_on": ["second"],
            },
        ],
    }


def _segment(name: str, success: bool) -> dict:
    return {"name": name, "active": [True], "successes": [success]}


def test_failure_and_unentered_successor_are_not_the_same_result() -> None:
    runtime = {"segments": [_segment("a", True), _segment("b", False)]}
    rows = task_group_results(_graph(), runtime, num_envs=1)[0]
    assert rows["first"]["execution_status"] == "passed"
    assert rows["first"]["status"] == "unverified"
    assert rows["second"]["status"] == "failed"
    assert rows["second"]["first_failed_call"] == "b"
    assert rows["third"]["status"] == "blocked"
    assert rows["third"]["blocked_by"] == ["second"]


@pytest.mark.parametrize(
    "verdict,expected",
    [("satisfied", "passed"), ("unavailable", "unverified"), ("violated", "failed")],
)
def test_exact_e_milestone_not_global_final_state_controls_verification(
    verdict, expected
) -> None:
    runtime = {
        "segments": [_segment("a", True)],
        "task_spec_evaluation": {
            "environments": [
                {
                    "env_id": 0,
                    "status": "unavailable",
                    "certificate_ready": False,
                    "milestones": [
                        {"predicate_id": "milestone:first", "status": verdict}
                    ],
                }
            ]
        },
    }
    row = task_group_results(_graph(), runtime, num_envs=1)[0]["first"]
    assert row["status"] == expected
    assert row["execution_status"] == "passed"
    assert row["task_certificate_ready"] is False


def test_missing_trace_is_unverified_not_a_fabricated_failure() -> None:
    rows = task_group_results(_graph(), None, num_envs=1)[0]
    assert all(row["status"] == "unverified" for row in rows.values())
    assert all(row["first_failed_call"] is None for row in rows.values())


def test_group_metadata_order_does_not_change_blocked_descendants() -> None:
    graph = _graph()
    runtime = {"segments": [_segment("a", False)]}
    original = task_group_results(graph, runtime, num_envs=1)
    graph["task_groups"].reverse()
    reordered = task_group_results(graph, runtime, num_envs=1)
    assert reordered == original
    assert reordered[0]["third"]["status"] == "blocked"


def test_row_local_failure_does_not_block_other_environment() -> None:
    runtime = {
        "segments": [
            {"name": "a", "active": [True, True], "successes": [False, True]},
            {"name": "b", "active": [False, True], "successes": [False, True]},
        ]
    }
    rows = task_group_results(_graph(), runtime, num_envs=2)
    assert rows[0]["second"]["status"] == "blocked"
    assert rows[1]["second"]["execution_status"] == "passed"


def test_original_trace_owned_and_exact_failure_phase_retained() -> None:
    runtime = {"segments": [_segment("a", False)]}
    runtime["segments"][0]["metadata"] = {
        "runtime": {
            "failures": [
                {
                    "env_mask": [True],
                    "phase": "planning",
                    "code": "no_path",
                    "message": "rejected",
                }
            ],
            "calls": [
                {
                    "invocation_id": "call-1",
                    "masks": {"entered": [True]},
                    "command_count": 0,
                }
            ],
            "task_state": {"held_objects": []},
        }
    }
    original = deepcopy(runtime)
    row = task_group_results(_graph(), runtime, num_envs=1)[0]["first"]
    assert row["calls"][0]["failure"]["phase"] == "planning"
    assert row["calls"][0]["invocation_id"] == "call-1"
    row["calls"][0]["state_after"]["batch"]["held_objects"].append("not-real")
    assert runtime == original


def test_completed_task_certificate_can_verify_terminal_group() -> None:
    runtime = {
        "segments": [_segment(name, True) for name in ("a", "b", "c")],
        "task_spec_evaluation": {
            "environments": [
                {"env_id": 0, "status": "satisfied", "certificate_ready": True}
            ]
        },
    }
    rows = task_group_results(_graph(), runtime, num_envs=1)[0]
    assert all(row["status"] == "passed" for row in rows.values())


def test_planning_event_explains_generic_semantic_execution_failure() -> None:
    runtime = {"segments": [_segment("a", False)]}
    runtime["segments"][0]["metadata"] = {
        "runtime": {
            "failures": [
                {
                    "env_mask": [True],
                    "phase": "execution",
                    "code": "semantic_call_failed",
                }
            ],
            "events": [
                {
                    "kind": "action_planning_failed",
                    "env_mask": [True],
                    "failure_code": "held_transport_path_unqualified",
                    "segment_name": "transport",
                    "timestamp": 11.32,
                }
            ],
        }
    }
    call = task_group_results(_graph(), runtime, num_envs=1)[0]["first"]["calls"][0]
    assert call["failure"]["phase"] == "execution"
    assert call["cause"]["phase"] == "planning"
    assert call["cause"]["code"] == "held_transport_path_unqualified"
    assert call["cause"]["segment"] == "transport"


def test_null_runtime_metadata_remains_unverified() -> None:
    runtime = {"segments": [_segment("a", True)]}
    runtime["segments"][0]["metadata"] = {"runtime": None}
    row = task_group_results(_graph(), runtime, num_envs=1)[0]["first"]
    assert row["status"] == "unverified"
    assert row["calls"][0]["state_after"] is None


@pytest.mark.parametrize("env_ids", [[1, 0], [1]])
def test_nested_trace_masks_use_runtime_environment_identity(env_ids) -> None:
    masks = [True] + [False] * (len(env_ids) - 1)
    runtime = {
        "segments": [
            {
                "name": "a",
                "active": [True, True],
                "successes": [False, False],
                "metadata": {
                    "runtime": {
                        "env_ids": env_ids,
                        "calls": [
                            {
                                "invocation_id": "env-1-call",
                                "masks": {"entered": masks},
                                "command_count": 0,
                            }
                        ],
                        "failures": [
                            {
                                "env_mask": masks,
                                "phase": "planning",
                                "code": "env-1-only",
                            }
                        ],
                        "task_state": {"batch_size": len(env_ids), "held_objects": []},
                    }
                },
            }
        ]
    }
    rows = task_group_results(_graph(), runtime, num_envs=2)
    first, second = (rows[index]["first"]["calls"][0] for index in (0, 1))
    assert first["invocation_id"] is None and first["failure"] is None
    assert second["invocation_id"] == "env-1-call"
    assert second["failure"]["code"] == "env-1-only"
    assert second["state_after"]["row_index"] == 0
    assert second["state_after"]["env_ids"] == env_ids
