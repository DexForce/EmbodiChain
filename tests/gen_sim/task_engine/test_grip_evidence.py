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

from collections import deque
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program.grip_evidence import (
    classify_contacts,
    NativeGripObserver,
)


@pytest.mark.parametrize("swap", [False, True])
def test_wide_grip_uses_same_target_opposed_contacts_not_closing_angle(swap):
    pairs = torch.tensor([[1, 9], [2, 9]])
    normals = torch.tensor([[0.72, 0.69, 0.0], [-0.72, -0.69, 0.0]])
    if swap:
        pairs = pairs.flip(1)
        normals = -normals
    assert classify_contacts(
        pairs, normals, torch.tensor([0.02, 0.03]), target=9, pads=(1, 2)
    ) == (True, True)
    assert classify_contacts(
        pairs, normals, torch.tensor([0.02, 0.03]), target=10, pads=(1, 2)
    ) == (False, True)


@pytest.mark.parametrize(
    "normals,impulses,expected",
    [
        ([[1.0, 0, 0], [1.0, 0, 0]], [0.02, 0.03], (False, False)),
        ([[1.0, 0, 0], [-1.0, 0, 0]], [0.02, 0.0], (False, False)),
        ([[1.0, 0, 0], [-1.0, 0, 0]], [0.0, 0.0], (False, True)),
        ([[float("nan"), 0, 0], [-1.0, 0, 0]], [0.02, 0.03], (False, False)),
        ([[0.0, 0, 0], [-1.0, 0, 0]], [0.02, 0.03], (False, False)),
    ],
)
def test_unqualified_contact_is_never_positive(normals, impulses, expected):
    assert (
        classify_contacts(
            torch.tensor([[1, 9], [2, 9]]),
            torch.tensor(normals),
            torch.tensor(impulses),
            target=9,
            pads=(1, 2),
        )
        == expected
    )


def test_no_contacts_proves_clear_release():
    assert classify_contacts(
        torch.empty((0, 2), dtype=torch.long),
        torch.empty((0, 3)),
        torch.empty(0),
        target=9,
        pads=(1, 2),
    ) == (False, True)


def _observer_fixture():
    data = {
        "is_valid": torch.tensor([[True, True]]),
        "user_ids": torch.tensor([[[1, 9], [2, 9]]]),
        "normal": torch.tensor([[[1.0, 0, 0], [-1.0, 0, 0]]]),
        "impulse": torch.tensor([[0.02, 0.03]]),
    }
    sensor = SimpleNamespace(
        update=Mock(), get_data=Mock(return_value=data), dropped_contacts=0
    )
    observer = NativeGripObserver(
        sensor,
        {"item": torch.tensor([[9]])},
        {"left_eef": torch.tensor([[1, 2]])},
    )
    query = SimpleNamespace(
        expectation=SimpleNamespace(object_id="item"),
        source=SimpleNamespace(address=SimpleNamespace(control_part="left_eef")),
        clause=SimpleNamespace(expected=True),
    )
    context = SimpleNamespace(
        env_ids=torch.tensor([0]), timestamp=0.25, observation_revision=1
    )
    return observer, sensor, query, context


def test_native_trace_snapshot_owns_rows_and_does_not_read_sensor() -> None:
    observer, sensor, query, context = _observer_fixture()
    observation = observer(query, context)
    assert observation.values.tolist() == [True]
    assert observation.valid.tolist() == [True]
    snapshot = observer.snapshot_trace()
    assert snapshot["source"] == "native_contact_sensor"
    assert snapshot["cumulative_query_count"] == 1
    assert snapshot["retained_query_count"] == 1
    assert snapshot["finite_audit"]["valid"] is True
    assert snapshot["actor_bindings"]["object_actor_ids"] == {"item": [[9]]}
    assert snapshot["actor_bindings"]["pad_actor_ids"] == {"left_eef": [[1, 2]]}
    assert snapshot["actor_bindings"]["env_axis"] == "native_env_id"
    assert sensor.update.call_count == sensor.get_data.call_count == 1
    observer.object_actors["item"][0, 0] = 999
    observer.pad_actors["left_eef"][0, 0] = 888
    observer.trace[0]["rows"][0]["pairs"][0][0] = 99
    observer.trace.clear()
    assert snapshot["trace"][0]["rows"][0]["pairs"][0][0] == 1
    assert snapshot["actor_bindings"]["object_actor_ids"] == {"item": [[9]]}
    assert snapshot["actor_bindings"]["pad_actor_ids"] == {"left_eef": [[1, 2]]}
    json.dumps(snapshot, allow_nan=False)


def test_trace_eviction_is_explicit_and_does_not_change_observation() -> None:
    observer, sensor, query, context = _observer_fixture()
    observer.trace = deque(maxlen=2)
    for _ in range(4):
        assert observer(query, context).valid.all()
    snapshot = observer.snapshot_trace()
    assert snapshot["cumulative_query_count"] == snapshot["recorded_query_count"] == 4
    assert snapshot["retained_query_count"] == snapshot["evicted_query_count"] == 2
    assert snapshot["overflowed"] is True
    assert [record["query_index"] for record in snapshot["trace"]] == [3, 4]


@pytest.mark.parametrize(
    "value, label",
    [(float("nan"), "NaN"), (float("inf"), "+Infinity"), (-float("inf"), "-Infinity")],
)
def test_nonfinite_raw_contact_is_preserved_as_invalid_strict_json(
    value, label
) -> None:
    observer, sensor, query, context = _observer_fixture()
    sensor.get_data.return_value["normal"][0, 0, 0] = value
    observation = observer(query, context)
    assert not observation.valid.any()
    snapshot = observer.snapshot_trace()
    assert snapshot["finite_audit"]["valid"] is False
    assert snapshot["finite_audit"]["nonfinite_query_count"] == 1
    assert snapshot["trace"][0]["rows"][0]["raw_contacts"]["normals"][0][0] == {
        "nonfinite": label
    }
    assert snapshot["trace"][0]["rows"][0]["error"] == observation.acquisition_errors[0]
    json.dumps(snapshot, allow_nan=False)


def test_dropped_contact_and_unrecorded_query_counts_are_auditable() -> None:
    observer, sensor, query, context = _observer_fixture()
    sensor.dropped_contacts = 3
    assert not observer(query, context).valid.any()
    sensor.update.side_effect = RuntimeError("native query failed")
    with pytest.raises(RuntimeError, match="native query failed"):
        observer(query, context)
    snapshot = observer.snapshot_trace()
    assert snapshot["cumulative_query_count"] == 2
    assert snapshot["recorded_query_count"] == 1
    assert snapshot["unrecorded_query_count"] == 1
    assert snapshot["dropped_contacts"]["maximum_per_query"] == 3
    assert snapshot["trace"][0]["dropped_contacts"] == 3


def test_non_target_nonfinite_contacts_are_not_hidden_by_target_filtering() -> None:
    observer, sensor, query, context = _observer_fixture()
    sensor.get_data.return_value = {
        "is_valid": torch.tensor([[True, True, True]]),
        "user_ids": torch.tensor([[[1, 9], [2, 9], [1, 10]]]),
        "normal": torch.tensor([[[1.0, 0, 0], [-1.0, 0, 0], [float("nan"), 0, 0]]]),
        "impulse": torch.tensor([[0.02, 0.03, 0.02]]),
    }
    assert not observer(query, context).valid.any()
    snapshot = observer.snapshot_trace()
    assert snapshot["trace"][0]["rows"][0]["raw_contacts"]["normals"][2][0] == {
        "nonfinite": "NaN"
    }
    assert snapshot["finite_audit"]["valid"] is False


def test_eviction_does_not_erase_cumulative_nonfinite_failure() -> None:
    observer, sensor, query, context = _observer_fixture()
    observer.trace = deque(maxlen=1)
    sensor.get_data.return_value["normal"][0, 0, 0] = float("nan")
    assert not observer(query, context).valid.any()
    sensor.get_data.return_value["normal"][0, 0, 0] = 1.0
    assert observer(query, context).valid.all()
    snapshot = observer.snapshot_trace()
    assert snapshot["trace"][0]["rows"][0]["finite"] is True
    assert snapshot["finite_audit"]["valid"] is False
    assert snapshot["finite_audit"]["nonfinite_query_count"] == 1
    assert snapshot["overflowed"] is True
    assert snapshot["trace_complete"] is False
