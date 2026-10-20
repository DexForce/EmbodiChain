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

"""Exercise recording eligibility independently of simulation and storage."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from embodichain.lab.gym.envs.episode_context import EpisodeContext


def ready_context(count=4):
    context = EpisodeContext(count)
    context._begin_episode(list(range(count)))
    context._finish_initialization(list(range(count)))
    return context


def test_initialization_and_partial_row_reset():
    context = EpisodeContext(4)
    assert not len(context.filter_recordable())
    context._begin_episode([3, 1])
    context._finish_initialization([3, 1])
    assert context.filter_recordable([3, 2, 1]).tolist() == [3, 1]
    assert [row["episode"] for row in context.snapshot()] == [-1, 0, -1, 0]


def test_checks_combine_without_overwriting_other_sources():
    context = ready_context()
    with context.check("one", [1, 3]) as one:
        with context.check("two", [1]) as two:
            one(accepted=True, metadata={"variant_id": "a"})
            assert context.filter_recordable().tolist() == [0, 2]
            two(accepted=False, reasons="unverified")
        assert context.filter_recordable().tolist() == [0, 2]
    assert context.filter_recordable().tolist() == [0, 2, 3]
    row = context.snapshot([1])[0]
    assert set(row["sources"]) == {"one", "two"}
    assert row["rejections"]["two"]["reason"] == "unverified"


def test_rejection_belongs_to_buffers_until_consumed():
    context = ready_context()
    with context.check("layout", [1]) as finish:
        finish(accepted=False)
    with context.check("layout", [1]) as finish:
        finish(accepted=True)
    assert not len(context.filter_recordable([1]))
    context._begin_episode([])
    assert not len(context.filter_recordable([1]))
    context._begin_episode([1])
    context._finish_initialization([1])
    assert context.filter_recordable([1]).tolist() == [1]
    assert context.snapshot([1])[0]["sources"] == {}


def test_scope_survives_internal_resets():
    context = ready_context()
    with context.check("attempt", [1, 3]) as finish:
        context._begin_episode([1, 3])
        context._finish_initialization([1, 3])
        assert context.filter_recordable().tolist() == [0, 2]
        finish(accepted=[True, False], metadata={1: {"id": "a"}, 3: {"id": "b"}})
    assert context.filter_recordable().tolist() == [0, 1, 2]
    assert context.snapshot([3])[0]["sources"]["attempt"]["metadata"]["id"] == "b"


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_exception_overrides_successful_report(error):
    context = ready_context()
    with pytest.raises(error):
        with context.check("test", [1]) as finish:
            finish(accepted=True)
            raise error("failed later")
    assert not len(context.filter_recordable([1]))
    assert context.snapshot([1])[0]["pending_sources"] == []


def test_missing_duplicate_late_and_overlapping_reports():
    context = ready_context()
    with pytest.raises(RuntimeError, match="without a report"):
        with context.check("test", [1]) as finish:
            pass
    with pytest.raises(RuntimeError, match="exactly once"):
        finish(accepted=True)
    with context.check("test", [0]) as finish:
        with pytest.raises(RuntimeError, match="already covers"):
            with context.check("test", [0]):
                pass
        finish(accepted=True)
        with pytest.raises(RuntimeError, match="exactly once"):
            finish(accepted=True)
    assert context.filter_recordable().tolist() == [0, 2, 3]


def test_report_and_snapshots_are_owned():
    context = ready_context()
    value = {"ids": ["source"]}
    with context.check("layout", [1]) as finish:
        finish(accepted=torch.tensor([True]), metadata=value)
        value["ids"].append("mutated")
    row = context.snapshot([1])[0]
    row["sources"]["layout"]["metadata"]["ids"].clear()
    assert context.snapshot([1])[0]["sources"]["layout"]["metadata"] == {
        "ids": ["source"]
    }


@pytest.mark.parametrize(
    "metadata", [{"value": float("nan")}, {"value": object()}, {2: {}}, []]
)
def test_invalid_reports_never_publish_success(metadata):
    context = ready_context()
    with pytest.raises((TypeError, ValueError)):
        with context.check("layout", [1]) as finish:
            finish(accepted=True, metadata=metadata)
    assert not len(context.filter_recordable([1]))


@pytest.mark.parametrize(
    "selection", [[1, 1], [-1], [4], [True], [1.0], torch.tensor([[1]])]
)
def test_invalid_rows_fail_before_state_changes(selection):
    context = ready_context()
    before = context.snapshot()
    with pytest.raises(ValueError):
        with context.check("layout", selection):
            pass
    assert context.snapshot() == before


def test_background_workers_cannot_update_context():
    context = ready_context()
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(RuntimeError, match="environment thread"):
            executor.submit(context.snapshot).result()
