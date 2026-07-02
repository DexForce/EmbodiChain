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

import json

import pytest

from embodichain.gen_sim.action_agent_pipeline.utils.timing import (
    TIMING_PATH_ENV,
    build_timing_summary,
    configure_timing_tracking,
    disable_timing_tracking,
    record_timing,
    timing_scope,
)


@pytest.fixture(autouse=True)
def _clear_timing_env():
    disable_timing_tracking()
    yield
    disable_timing_tracking()


def test_record_timing_writes_jsonl_and_summary(tmp_path):
    timing_path = tmp_path / "timing.jsonl"
    configure_timing_tracking(
        timing_path=timing_path,
        run_id="test-run",
        process_name="pytest",
        reset=True,
    )

    record_timing(
        stage="Action Agent Task",
        duration_s=1.25,
        metadata={"trajectory_idx": 0},
    )
    record_timing(stage="Action Agent Task", duration_s=0.75)
    record_timing(stage="Compile Agent", duration_s=0.5, status="error")

    records = [
        json.loads(line)
        for line in timing_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 3
    assert records[0]["stage"] == "action_agent_task"
    assert records[0]["metadata"] == {"trajectory_idx": 0}
    assert records[0]["process"] == "pytest"

    summary = build_timing_summary(timing_path)
    assert summary["total"]["calls"] == 3
    assert summary["total"]["errors"] == 1
    assert summary["total"]["total_s"] == pytest.approx(2.5)
    assert summary["by_stage"]["action_agent_task"]["calls"] == 2
    assert summary["by_stage"]["action_agent_task"]["avg_s"] == pytest.approx(1.0)
    assert summary["by_process"]["pytest"]["max_s"] == pytest.approx(1.25)


def test_timing_scope_records_errors(tmp_path):
    timing_path = tmp_path / "timing.jsonl"
    configure_timing_tracking(
        timing_path=timing_path,
        run_id="test-run",
        process_name="pytest",
        reset=True,
    )

    with pytest.raises(RuntimeError):
        with timing_scope("failing stage", metadata={"item": "cup"}):
            raise RuntimeError("boom")

    record = json.loads(timing_path.read_text(encoding="utf-8").strip())
    assert record["stage"] == "failing_stage"
    assert record["status"] == "error"
    assert record["duration_s"] >= 0.0
    assert record["metadata"] == {"error_type": "RuntimeError", "item": "cup"}


def test_disable_timing_tracking_removes_env(tmp_path):
    configure_timing_tracking(
        timing_path=tmp_path / "timing.jsonl",
        run_id="test-run",
        process_name="pytest",
        reset=True,
    )

    disable_timing_tracking()

    import os

    assert TIMING_PATH_ENV not in os.environ
