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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from embodichain.data_analysis import Catalog, RecordValidationError


def _measurement(value: object, *, source: str = "measured") -> dict[str, object]:
    return {
        "value": value,
        "source": source,
        "unit": "",
        "frame": "world",
        "scope": "episode",
    }


def _record(episode_id: str = "ep-0", status: str = "proposed") -> dict[str, object]:
    return {
        "schema_version": 1,
        "episode_id": episode_id,
        "run_id": "run-0",
        "candidate_id": episode_id.replace("ep", "candidate"),
        "attempt_id": 0,
        "task_id": "pick-place",
        "robot_id": "franka",
        "status": status,
        "reason": None,
        "seed": 7,
        "parents": [],
        "dimensions": {"material": _measurement("wood"), "pose_x": _measurement(0.0)},
        "artifacts": {},
        "segments": [],
        "metrics": {},
        "provenance": {"producer": "test"},
    }


def test_literal_lifecycle_fixture_and_partial_commit_retry(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path / "catalog.sqlite")
    statuses = (
        ["planning_failed"] * 2
        + ["rollout_failed"] * 2
        + ["rejected"]
        + ["committed"] * 6
    )
    for index, status in enumerate(statuses):
        catalog.upsert(_record(f"ep-{index}", status))
    catalog.upsert(_record("ep-11", "partial_commit"))

    summary = catalog.summary()
    assert summary["total"] == 12
    assert summary["statuses"] == {
        "committed": 6,
        "partial_commit": 1,
        "planning_failed": 2,
        "rejected": 1,
        "rollout_failed": 2,
    }

    catalog.upsert(_record("ep-11", "committed"))
    catalog.upsert(_record("ep-11", "committed"))
    repaired = catalog.summary()
    assert repaired["total"] == 12
    assert repaired["statuses"]["committed"] == 7
    assert "partial_commit" not in repaired["statuses"]
    assert repaired["events"] == 13


def test_identity_is_immutable_and_terminal_status_cannot_regress(
    tmp_path: Path,
) -> None:
    catalog = Catalog(tmp_path / "catalog.sqlite")
    catalog.upsert(_record(status="committed"))

    conflict = _record(status="committed")
    conflict["run_id"] = "other-run"
    with pytest.raises(ValueError, match="immutable identity"):
        catalog.upsert(conflict)

    with pytest.raises(ValueError, match="transition"):
        catalog.upsert(_record(status="partial_commit"))
    assert catalog.get("ep-0")["status"] == "committed"


def test_committed_record_checks_each_declared_artifact(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path / "catalog.sqlite")
    missing = _record(status="committed")
    missing["artifacts"] = {"trajectory": "episodes/ep-0/trajectory.npz"}

    with pytest.raises(RecordValidationError, match="trajectory.*does not exist"):
        catalog.upsert(missing)

    artifact = tmp_path / "episodes" / "ep-0" / "trajectory.npz"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"portable")
    catalog.upsert(missing)
    assert (
        catalog.get("ep-0")["artifacts"]["trajectory"] == "episodes/ep-0/trajectory.npz"
    )


def test_validation_rejects_non_finite_json_unknown_versions_and_bad_measurements(
    tmp_path: Path,
) -> None:
    catalog = Catalog(tmp_path / "catalog.sqlite")
    non_finite = _record()
    non_finite["metrics"] = {"loss": float("nan")}
    with pytest.raises(RecordValidationError, match="finite JSON"):
        catalog.upsert(non_finite)

    wrong_version = _record()
    wrong_version["schema_version"] = 2
    with pytest.raises(RecordValidationError, match="schema_version"):
        catalog.upsert(wrong_version)

    unknown_without_reason = _record()
    unknown_without_reason["dimensions"] = {
        "light": _measurement(None, source="unknown")
    }
    with pytest.raises(RecordValidationError, match="missing_reason"):
        catalog.upsert(unknown_without_reason)


def test_records_filters_snapshot_and_jsonl_import_are_stable(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path / "catalog.sqlite")
    first = _record("ep-a", "committed")
    second = _record("ep-b", "committed")
    second["dimensions"] = {
        "material": _measurement("metal"),
        "pose_x": _measurement(0.6),
    }
    catalog.upsert(first)
    catalog.upsert(second)

    assert [
        record["episode_id"] for record in catalog.records({"material": ["wood"]})
    ] == ["ep-a"]
    assert [
        record["episode_id"]
        for record in catalog.records({"pose_x": {"min": 0.5, "max": 1.0}})
    ] == ["ep-b"]
    snapshot_id = catalog.snapshot("baseline")
    catalog.upsert(_record("ep-c", "committed"))
    assert catalog.snapshot_records(snapshot_id)[1]["metrics"] == {}
    assert len(catalog.snapshot_records(snapshot_id)) == 2

    input_path = tmp_path / "records.jsonl"
    input_path.write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n", encoding="utf-8"
    )
    assert catalog.import_jsonl(input_path) == {
        "inserted": 0,
        "updated": 0,
        "unchanged": 2,
    }


def test_same_status_different_content_is_a_conflict(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path / "catalog.sqlite")
    catalog.upsert(_record(status="committed"))
    changed = _record(status="committed")
    changed["metrics"] = {"score": 1.0}

    with pytest.raises(ValueError, match="different content"):
        catalog.upsert(changed)


def test_catalog_can_serve_a_worker_thread_and_reopen(tmp_path: Path) -> None:
    path = tmp_path / "catalog.sqlite"
    catalog = Catalog(path)
    catalog.upsert(_record(status="committed"))

    with ThreadPoolExecutor(max_workers=1) as executor:
        record = executor.submit(catalog.get, "ep-0").result()
    assert record is not None
    assert record["episode_id"] == "ep-0"
    catalog.close()

    with Catalog(path) as reopened:
        assert reopened.summary()["statuses"] == {"committed": 1}
