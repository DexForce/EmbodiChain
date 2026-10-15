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
import torch

from embodichain.gen_sim.task_engine._task_program.support_contract import (
    release_window,
    support_window_flags,
)


def _release_case():
    identity = {"invocation_id": "program/place:0", "invocation_revision": 0}
    receipt = {
        "boundary": {"kind": "terminal"},
        "timestamp": 2.0,
        "verification_id": 8,
        "observation_revision": 12,
        "effect_spec": {
            **identity,
            "effect_kind": "release",
            "env_ids": [0],
            "state_expectations": [
                {
                    "object_id": "item",
                    "resource_id": "left",
                    "relation": "detached",
                    "expectation_id": "released",
                }
            ],
            "clauses": [
                {
                    "expectation_id": "released",
                    "clause_id": "constraint",
                    "evidence_kind": "constraint",
                    "expected": False,
                },
                {
                    "expectation_id": "released",
                    "clause_id": "pose",
                    "expectation": "separated",
                },
            ],
        },
        "evidence": {
            key: {
                "env_ids": [0],
                "timestamp": 2.0,
                "observation_revision": 12,
                "valid_mask": [True],
                **value,
            }
            for key, value in {
                "constraint": {"values": [False]},
                "pose": {"object_to_endpoint": torch.eye(4)[None].tolist()},
            }.items()
        },
        "decision": {"success_mask": [True]},
    }
    call = {
        **identity,
        "call_index": 0,
        "status": "completed",
        "active_plan_attempt_generation": 0,
        "plan_attempts": [
            {
                **identity,
                "invocation_index": 0,
                "attempt_generation": 0,
                "planned_at": 1.0,
            }
        ],
        "events": [
            {**identity, "invocation_index": 0, "env_mask": [True], **event}
            for event in (
                {
                    "kind": "trajectory_segment_entered",
                    "segment_name": "release",
                    "timestamp": 1.4,
                },
                {"kind": "action_completed", "timestamp": 2.2},
            )
        ],
        "effects": [receipt],
    }
    node = {
        "id": "place",
        "call": {
            "kind": "place",
            "object": "item",
            "inside": "tray",
            "resources": {"primary": "left"},
        },
    }
    boundary = {
        "metadata": {
            "runtime": {
                "env_ids": [0],
                "task_state": {
                    "batch_size": 1,
                    "held_objects": [],
                    "coordinated_held_objects": [],
                },
            }
        }
    }
    return (
        node,
        {"place": {"metadata": {"runtime": {"env_ids": [0], "calls": [call]}}}},
        boundary,
    )


def _native():
    return {
        "scope": "native_contact_support",
        "support_contract_revision": 1,
        "clock": "environment_step",
        "observed_at": 3.0,
        "observed_step": 75,
        "step_dt": 0.04,
        "entity": "item",
        "reference": "tray",
        "env_ids": [0],
        "episode_generations": [1],
        "actor_pairs": [
            {
                "object_actor": 11,
                "reference_actor": 22,
                "object_uid": "native_item",
                "reference_uid": "native_tray",
                "object_info": {
                    "actor_id": 11,
                    "path": "arena_0/native_item",
                    "link_name": None,
                    "arena_name": "arena_0",
                    "env_id": 0,
                },
                "reference_info": {
                    "actor_id": 22,
                    "path": "arena_0/native_tray",
                    "link_name": None,
                    "arena_name": "arena_0",
                    "env_id": 0,
                },
            }
        ],
        "rows": [
            {
                "retained": True,
                "sample": {
                    "timestamp": 2.4,
                    "control_step": 60,
                    "substep": 4,
                    "serial": 9,
                    "episode_generation": 1,
                    "native_revision": 55,
                    "topology_revision": 0,
                },
            }
        ],
    }


def _flags(native, window):
    return support_window_flags(
        native,
        window,
        1,
        entity="item",
        reference="tray",
        native_entity="native_item",
        native_reference="native_tray",
    )


def test_typed_inside_uses_same_verified_release_and_actual_phase():
    node, segments, boundary = _release_case()
    released, window = release_window(node, segments, boundary, 1)
    assert released == [True]
    assert window["release_started_at"] == 1.4
    assert window["verification_id"] == 8
    assert _flags(_native(), window) == [True]
    fake = deepcopy(segments["place"]["metadata"]["runtime"]["calls"][0]["effects"][0])
    fake["evidence"]["constraint"]["valid_mask"] = []
    fake["verification_id"] = 99
    segments["place"]["metadata"]["runtime"]["calls"][0]["effects"].append(fake)
    assert release_window(node, segments, boundary, 1)[1]["verification_id"] == 8


def test_support_during_release_command_is_not_post_verified_release():
    _, window = release_window(*_release_case(), 1)
    native = _native()
    native["rows"][0]["sample"].update(timestamp=1.6, control_step=40)
    assert window["release_started_at"] < native["rows"][0]["sample"]["timestamp"]
    assert native["rows"][0]["sample"]["timestamp"] < window["timestamp"]
    assert _flags(native, window) == [False]


@pytest.mark.parametrize(
    "timestamp,step,substep", [(2.0, 50, 4), (2.1, 53, 2), (2.4, 60, 4)]
)
def test_support_at_or_after_terminal_release_remains_eligible(
    timestamp, step, substep
):
    _, window = release_window(*_release_case(), 1)
    native = _native()
    native["rows"][0]["sample"].update(
        timestamp=timestamp, control_step=step, substep=substep
    )
    assert _flags(native, window) == [True]


@pytest.mark.parametrize("timestamp", [None, True, "2.0", float("nan"), -0.1, 1.2, 2.4])
def test_support_requires_a_valid_ordered_verified_release_fence(timestamp):
    _, window = release_window(*_release_case(), 1)
    window["timestamp"] = timestamp
    with pytest.raises(ValueError, match="window identity"):
        _flags(_native(), window)


@pytest.mark.parametrize(
    "damage",
    ["generation", "phase", "receipt_time", "duplicate_phase", "foreign_invocation"],
)
def test_release_window_rejects_unbound_or_superseded_execution(damage):
    node, segments, boundary = _release_case()
    call = segments["place"]["metadata"]["runtime"]["calls"][0]
    if damage == "generation":
        call["active_plan_attempt_generation"] = 1
    elif damage == "phase":
        call["events"][0]["segment_name"] = "approach"
    elif damage == "receipt_time":
        call["events"][0]["timestamp"] = 2.1
    elif damage == "duplicate_phase":
        call["events"].append(deepcopy(call["events"][0]))
    else:
        call["plan_attempts"][0]["invocation_id"] = "foreign"
    with pytest.raises(ValueError):
        release_window(node, segments, boundary, 1)


@pytest.mark.parametrize(
    "damage",
    [
        "wrong_reference",
        "actor",
        "episode",
        "restamped",
        "clock",
        "link",
        "unknown_actor",
        "missing_sample",
    ],
)
def test_stale_or_foreign_support_is_not_new_evidence(damage):
    _, window = release_window(*_release_case(), 1)
    native = _native()
    if damage == "wrong_reference":
        native["actor_pairs"][0]["reference_uid"] = "other"
    elif damage == "actor":
        native["actor_pairs"][0]["reference_info"]["actor_id"] = 33
    elif damage == "episode":
        native["episode_generations"][0] = 2
    elif damage == "restamped":
        native["rows"][0]["sample"]["timestamp"] = native["observed_at"]
    elif damage == "clock":
        native["observed_at"] = 3.2
    elif damage == "link":
        native["actor_pairs"][0]["reference_info"]["link_name"] = "drawer"
    elif damage == "unknown_actor":
        native["actor_pairs"][0]["reference_actor"] = -1
    else:
        native["rows"][0]["sample"] = None
        assert _flags(native, window) == [False]
        return
    with pytest.raises(ValueError):
        _flags(native, window)


def test_sleep_retention_keeps_original_sample_time_but_not_prior_release():
    _, window = release_window(*_release_case(), 1)
    native = _native()
    native["observed_at"], native["observed_step"] = 4.0, 100
    assert _flags(native, window) == [True]
    assert native["rows"][0]["sample"]["timestamp"] == 2.4
    window.update(release_started_at=2.4, timestamp=2.8, completed_at=3.0)
    assert _flags(native, window) == [False]
    native["actor_pairs"][0]["reference_info"]["env_id"] = -1
    native["actor_pairs"][0]["reference_info"]["arena_name"] = "default"
    assert _flags(native, {**window, "release_started_at": 1.4, "timestamp": 2.0}) == [
        True
    ]
