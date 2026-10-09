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

"""Restart and partial-commit recovery without simulator or SDK mutations."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest

from embodichain.data_pipeline.recording import (
    RecordingJournal,
    inspect_recording,
    recover_recording,
)


def make_recording(tmp_path: Path, *, frames: int = 2) -> tuple[RecordingJournal, dict]:
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(
        json.dumps({"total_episodes": 1 if frames else 0, "features": {}}),
        encoding="utf-8",
    )
    identity = str(uuid4())
    sidecar = {
        "episode_uuid": identity,
        "source_episode_uuid": identity,
        "run_uuid": str(uuid4()),
        "lerobot_episode_index": 0,
        "length": 2,
        "instruction": "Move the cube",
    }
    if frames:
        (root / "data/chunk-000").mkdir(parents=True)
        parquet.write_table(
            arrow.table({"episode_index": [0] * frames}),
            root / "data/chunk-000/file-000.parquet",
        )
        (root / "meta/episodes/chunk-000").mkdir(parents=True)
        parquet.write_table(
            arrow.table({"episode_index": [0], "length": [frames]}),
            root / "meta/episodes/chunk-000/file-000.parquet",
        )
    return RecordingJournal(root), sidecar


def advance_to(journal: RecordingJournal, identity: str, phase: str) -> None:
    for next_phase in (
        "lerobot_committing",
        "lerobot_committed",
        "depth_committed",
        "complete",
    ):
        journal.advance(identity, next_phase)
        if next_phase == phase:
            return


def test_restart_repairs_missing_sidecar_once(tmp_path: Path) -> None:
    journal, sidecar = make_recording(tmp_path)
    identity = sidecar["episode_uuid"]
    journal.prepare(sidecar)
    advance_to(journal, identity, "depth_committed")
    journal.record_error(identity, OSError("sidecar append failed"))

    # A new journal instance has no in-memory knowledge of the old writer.
    restarted = RecordingJournal(journal.root)
    assert restarted.get(identity)["phase"] == "depth_committed"
    before = recover_recording(journal.root)
    assert before["commits"][0]["status"] == "repairable"
    assert not (journal.root / "meta/embodichain_episodes.jsonl").exists()

    repaired = recover_recording(journal.root, repair=True)
    assert repaired["ok"] and repaired["repaired"] == [identity]
    again = recover_recording(journal.root, repair=True)
    assert again["ok"] and again["repaired"] == []
    records = (
        (journal.root / "meta/embodichain_episodes.jsonl").read_text().splitlines()
    )
    assert [json.loads(line) for line in records] == [sidecar]


def test_unknown_sdk_commit_never_replays_or_claims_rollback(tmp_path: Path) -> None:
    journal, sidecar = make_recording(tmp_path, frames=0)
    journal.prepare(sidecar)
    journal.advance(sidecar["episode_uuid"], "lerobot_committing")

    report = recover_recording(journal.root, repair=True)

    assert not report["ok"]
    assert report["repaired"] == []
    assert report["commits"][0]["status"] == "unresolved"
    with pytest.raises(RuntimeError, match="duplicate SDK write"):
        RecordingJournal(journal.root).prepare(sidecar)


def test_prepared_without_frames_is_distinct_from_unknown_commit(
    tmp_path: Path,
) -> None:
    journal, sidecar = make_recording(tmp_path, frames=0)
    journal.prepare(sidecar)
    journal.record_error(sidecar["episode_uuid"], ValueError("frame conversion failed"))

    report = inspect_recording(journal.root)

    assert report["commits"][0]["status"] == "retryable"
    journal.prepare(sidecar)
    assert journal.get(sidecar["episode_uuid"])["phase"] == "prepared"


def test_returned_sdk_commit_without_durable_frames_is_unresolved(
    tmp_path: Path,
) -> None:
    journal, sidecar = make_recording(tmp_path, frames=0)
    journal.prepare(sidecar)
    advance_to(journal, sidecar["episode_uuid"], "complete")

    report = recover_recording(journal.root, repair=True)

    assert not report["ok"]
    assert report["commits"][0]["status"] == "unresolved"
    assert report["repaired"] == []


def test_depth_partial_commit_cannot_be_repaired_by_language_sidecar(
    tmp_path: Path,
) -> None:
    journal, sidecar = make_recording(tmp_path)
    journal.prepare(sidecar, ["wrist"])
    advance_to(journal, sidecar["episode_uuid"], "lerobot_committed")

    report = recover_recording(journal.root, repair=True)

    assert report["commits"][0]["status"] == "unresolved"
    assert any("Depth artifacts" in issue for issue in report["commits"][0]["issues"])
    assert not (journal.root / "meta/embodichain_episodes.jsonl").exists()


def test_depth_path_cannot_escape_dataset_root(tmp_path: Path) -> None:
    journal, sidecar = make_recording(tmp_path)
    journal.prepare(sidecar, ["wrist"])
    advance_to(journal, sidecar["episode_uuid"], "lerobot_committed")
    (journal.root / "depth_meta.json").write_text(
        json.dumps(
            {
                "sensors": {
                    "wrist": {
                        "episodes": {"0": {"file": "../foreign.mp4", "frame_count": 2}}
                    }
                }
            }
        )
    )

    report = recover_recording(journal.root, repair=True)

    assert not report["ok"]
    assert any(
        "Unsafe dataset-relative" in issue for issue in report["commits"][0]["issues"]
    )


def test_torn_jsonl_blocks_automatic_append(tmp_path: Path) -> None:
    journal, sidecar = make_recording(tmp_path)
    journal.prepare(sidecar)
    advance_to(journal, sidecar["episode_uuid"], "depth_committed")
    path = journal.root / "meta/embodichain_episodes.jsonl"
    path.write_text('{"episode_uuid":', encoding="utf-8")

    report = recover_recording(journal.root, repair=True)

    assert report["errors"]
    assert report["repaired"] == []
    assert path.read_text() == '{"episode_uuid":'


def test_conflicting_metadata_is_not_overwritten(tmp_path: Path) -> None:
    journal, sidecar = make_recording(tmp_path)
    journal.prepare(sidecar)
    advance_to(journal, sidecar["episode_uuid"], "depth_committed")
    conflicting = dict(sidecar, instruction="Different task")
    path = journal.root / "meta/embodichain_episodes.jsonl"
    path.write_text(json.dumps(conflicting) + "\n", encoding="utf-8")

    report = recover_recording(journal.root, repair=True)

    assert report["commits"][0]["status"] == "unresolved"
    assert json.loads(path.read_text()) == conflicting


def test_journal_identity_and_phase_cannot_be_used_as_paths(tmp_path: Path) -> None:
    journal, sidecar = make_recording(tmp_path, frames=0)
    journal.prepare(sidecar)
    with pytest.raises(ValueError):
        journal.get("../../foreign")
    with pytest.raises(ValueError, match="transition"):
        journal.advance(sidecar["episode_uuid"], "complete")
    assert not list((journal.root / "meta/embodichain_commits").glob(".*.tmp"))


def test_replay_source_must_match_full_source_episode_for_fragment(
    tmp_path: Path,
) -> None:
    journal, sidecar = make_recording(tmp_path)
    path = tmp_path / "source.pt"
    path.write_bytes(b"Recovery must not deserialize this file")
    source_uuid = str(uuid4())
    sidecar.update(
        {
            "fragment": True,
            "parent_episode_uuid": source_uuid,
            "source_episode_uuid": source_uuid,
            "source_start_step": 12,
            "replay_artifact": {
                "path": str(path),
                "external": True,
                "source_episode_uuid": source_uuid,
                "initial_state_step": 0,
                "state_alignment": "source_episode",
            },
        }
    )
    journal.prepare(sidecar)
    advance_to(journal, sidecar["episode_uuid"], "depth_committed")

    assert recover_recording(journal.root, repair=True)["ok"]
    assert path.read_bytes() == b"Recovery must not deserialize this file"
