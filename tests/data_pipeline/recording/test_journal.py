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
import shutil
from uuid import uuid4

import numpy as np
import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest

from embodichain.data_pipeline.recording import (
    RecordingJournal,
    inspect_recording,
    recover_recording,
)

FPS = 10
VIDEO_SIZE = 16


def write_video(path: Path, count: int, *, depth: bool = False, fps: int = FPS) -> None:
    av = pytest.importorskip("av")
    path.parent.mkdir(parents=True, exist_ok=True)
    if depth:
        # Fixed lossless HEVC gray12le, 16x16, 10 FPS, code value 1024. Decoder
        # tests must not depend on the host's optional 12-bit encoder build.
        shutil.copyfile(
            Path(__file__).with_name("assets") / f"depth_{count}_frames.mp4", path
        )
        return
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = stream.height = VIDEO_SIZE
        stream.pix_fmt = "yuv420p"
        for _ in range(count):
            values = np.zeros((VIDEO_SIZE, VIDEO_SIZE, 3), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(values, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


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
            arrow.table(
                {
                    "episode_index": [0] * frames,
                    "frame_index": list(range(frames)),
                    "index": list(range(frames)),
                }
            ),
            root / "data/chunk-000/file-000.parquet",
        )
        (root / "meta/episodes/chunk-000").mkdir(parents=True)
        parquet.write_table(
            arrow.table(
                {
                    "episode_index": [0],
                    "length": [frames],
                    "dataset_from_index": [0],
                    "dataset_to_index": [frames],
                }
            ),
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


def make_media_recording(
    tmp_path: Path, kind: str
) -> tuple[RecordingJournal, dict, Path]:
    journal, sidecar = make_recording(tmp_path)
    root = journal.root
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["fps"] = FPS
    if kind == "rgb":
        key = "observation.images.camera"
        info["features"][key] = {"dtype": "video", "shape": [VIDEO_SIZE, VIDEO_SIZE, 3]}
        info["video_path"] = "videos/camera.mp4"
        path = root / info["video_path"]
        episode_path = root / "meta/episodes/chunk-000/file-000.parquet"
        episodes = parquet.read_table(episode_path).to_pylist()
        episodes[0].update(
            {
                f"videos/{key}/chunk_index": 0,
                f"videos/{key}/file_index": 0,
                f"videos/{key}/from_timestamp": 0.0,
                f"videos/{key}/to_timestamp": sidecar["length"] / FPS,
            }
        )
        parquet.write_table(arrow.Table.from_pylist(episodes), episode_path)
    else:
        path = root / "depth_videos/camera.mp4"
        (root / "depth_meta.json").write_text(
            json.dumps(
                {
                    "fps": FPS,
                    "sensors": {
                        "camera": {
                            "shape": [VIDEO_SIZE, VIDEO_SIZE, 1],
                            "video.pix_fmt": "gray12le",
                            "video.depth_min": 0.01,
                            "video.depth_max": 10.0,
                            "video.shift": 3.5,
                            "video.use_log": True,
                            "video.input_unit": "auto",
                            "video.output_unit": "m",
                            "episodes": {
                                "0": {
                                    "file": str(path.relative_to(root)),
                                    "frame_count": sidecar["length"],
                                }
                            },
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
    info_path.write_text(json.dumps(info), encoding="utf-8")
    write_video(path, sidecar["length"], depth=kind == "depth")
    journal.prepare(sidecar, ["camera"] if kind == "depth" else [])
    advance_to(journal, sidecar["episode_uuid"], "depth_committed")
    return journal, sidecar, path


@pytest.mark.parametrize("kind", ["rgb", "depth"])
@pytest.mark.parametrize("corruption", ["unreadable", "short"])
def test_recovery_rejects_unusable_video_evidence(
    tmp_path: Path, kind: str, corruption: str
) -> None:
    journal, sidecar, video = make_media_recording(tmp_path, kind)
    assert inspect_recording(journal.root)["commits"][0]["status"] == "repairable"
    if corruption == "unreadable":
        video.write_bytes(b"nonempty but not a usable MP4")
    else:
        write_video(video, sidecar["length"] - 1, depth=kind == "depth")
    before = video.read_bytes()

    report = recover_recording(journal.root, repair=True)

    assert not report["ok"]
    assert report["commits"][0]["status"] == "unresolved"
    assert report["commits"][0]["issues"]
    assert report["repaired"] == []
    assert journal.get(sidecar["episode_uuid"])["phase"] == "depth_committed"
    assert not (journal.root / "meta/embodichain_episodes.jsonl").exists()
    assert video.read_bytes() == before


@pytest.mark.parametrize("kind", ["rgb", "depth"])
def test_complete_journal_still_requires_readable_video(
    tmp_path: Path, kind: str
) -> None:
    journal, _, video = make_media_recording(tmp_path, kind)
    assert recover_recording(journal.root, repair=True)["ok"]
    sidecar_path = journal.root / "meta/embodichain_episodes.jsonl"
    before = sidecar_path.read_bytes()
    video.write_bytes(b"broken after a completed commit")

    report = recover_recording(journal.root, repair=True)

    assert not report["ok"]
    assert report["commits"][0]["status"] == "unresolved"
    assert report["repaired"] == []
    assert sidecar_path.read_bytes() == before


@pytest.mark.parametrize("kind", ["rgb", "depth"])
def test_readable_media_allows_sidecar_recovery(tmp_path: Path, kind: str) -> None:
    journal, sidecar, video = make_media_recording(tmp_path, kind)
    before = video.read_bytes()

    report = recover_recording(journal.root, repair=True)

    assert report["ok"]
    assert report["repaired"] == [sidecar["episode_uuid"]]
    assert video.read_bytes() == before


def test_depth_recovery_rejects_extra_decoded_frames(tmp_path: Path) -> None:
    journal, sidecar, video = make_media_recording(tmp_path, "depth")
    write_video(video, sidecar["length"] + 1, depth=True)

    report = recover_recording(journal.root, repair=True)

    assert not report["ok"]
    assert report["repaired"] == []
    assert not (journal.root / "meta/embodichain_episodes.jsonl").exists()


@pytest.mark.parametrize("corruption", ["fps", "offset"])
def test_rgb_recovery_checks_media_timing(tmp_path: Path, corruption: str) -> None:
    journal, sidecar, video = make_media_recording(tmp_path, "rgb")
    if corruption == "fps":
        write_video(video, sidecar["length"], fps=2 * FPS)
    else:
        episode_path = journal.root / "meta/episodes/chunk-000/file-000.parquet"
        episodes = parquet.read_table(episode_path).to_pylist()
        key = "observation.images.camera"
        # Shift by half a frame while retaining the expected interval length.
        episodes[0][f"videos/{key}/from_timestamp"] = 0.5 / FPS
        episodes[0][f"videos/{key}/to_timestamp"] += 0.5 / FPS
        parquet.write_table(arrow.Table.from_pylist(episodes), episode_path)

    report = recover_recording(journal.root, repair=True)

    assert not report["ok"]
    assert report["repaired"] == []


def test_shared_rgb_shard_is_decoded_once_per_inspection(
    tmp_path: Path, monkeypatch
) -> None:
    av = pytest.importorskip("av")
    journal, sidecar, video = make_media_recording(tmp_path, "rgb")
    info_path = journal.root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["total_episodes"] = 2
    info_path.write_text(json.dumps(info))
    parquet.write_table(
        arrow.table(
            {
                "episode_index": [0, 0, 1, 1],
                "frame_index": [0, 1, 0, 1],
                "index": [0, 1, 2, 3],
            }
        ),
        journal.root / "data/chunk-000/file-000.parquet",
    )
    episode_path = journal.root / "meta/episodes/chunk-000/file-000.parquet"
    first = parquet.read_table(episode_path).to_pylist()[0]
    second = dict(first, episode_index=1, dataset_from_index=2, dataset_to_index=4)
    key = "observation.images.camera"
    second[f"videos/{key}/from_timestamp"] = 2 / FPS
    second[f"videos/{key}/to_timestamp"] = 4 / FPS
    parquet.write_table(arrow.Table.from_pylist([first, second]), episode_path)
    identity = str(uuid4())
    journal.prepare(
        dict(
            sidecar,
            episode_uuid=identity,
            source_episode_uuid=identity,
            lerobot_episode_index=1,
        )
    )
    advance_to(journal, identity, "depth_committed")
    write_video(video, 4)
    opened = []
    original_open = av.open

    def recording_open(*args, **kwargs):
        opened.append(args[0])
        return original_open(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(av, "open", recording_open)
        report = inspect_recording(journal.root)

    assert [item["status"] for item in report["commits"]] == [
        "repairable",
        "repairable",
    ]
    assert len(opened) == 1
    assert recover_recording(journal.root, repair=True)["ok"]


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


@pytest.mark.parametrize("legacy", [False, True])
def test_recovery_rejects_sidecar_index_owned_by_another_identity(
    tmp_path: Path, legacy: bool
) -> None:
    journal, sidecar = make_recording(tmp_path)
    identity = sidecar["episode_uuid"]
    journal.prepare(sidecar)
    advance_to(journal, identity, "depth_committed")
    conflicting = dict(sidecar, episode_uuid=str(uuid4()))
    if legacy:
        conflicting.pop("episode_uuid")
    path = journal.root / "meta/embodichain_episodes.jsonl"
    path.write_text(json.dumps(conflicting) + "\n", encoding="utf-8")
    before = path.read_bytes()

    report = recover_recording(journal.root, repair=True)

    assert not report["ok"]
    assert report["commits"][0]["status"] == "unresolved"
    assert report["repaired"] == []
    assert path.read_bytes() == before
    assert journal.get(identity)["phase"] == "depth_committed"


def test_recovery_rejects_duplicate_sidecar_indices(tmp_path: Path) -> None:
    journal, sidecar = make_recording(tmp_path)
    journal.prepare(sidecar)
    advance_to(journal, sidecar["episode_uuid"], "complete")
    path = journal.root / "meta/embodichain_episodes.jsonl"
    path.write_text(
        json.dumps(sidecar)
        + "\n"
        + json.dumps(dict(sidecar, episode_uuid=str(uuid4())))
        + "\n",
        encoding="utf-8",
    )
    before = path.read_bytes()

    report = recover_recording(journal.root, repair=True)

    assert not report["ok"]
    assert report["errors"]
    assert report["repaired"] == []
    assert path.read_bytes() == before


@pytest.mark.parametrize("column", ["frame_index", "index"])
def test_recovery_rejects_corrupt_frame_indices(tmp_path: Path, column: str) -> None:
    journal, sidecar = make_recording(tmp_path)
    identity = sidecar["episode_uuid"]
    journal.prepare(sidecar)
    advance_to(journal, identity, "depth_committed")
    path = journal.root / "data/chunk-000/file-000.parquet"
    table = parquet.read_table(path)
    table = table.set_column(
        table.schema.get_field_index(column), column, arrow.array([0, 0])
    )
    parquet.write_table(table, path)
    before = path.read_bytes()

    report = recover_recording(journal.root, repair=True)

    assert not report["ok"]
    assert report["commits"][0]["status"] == "unresolved"
    assert report["repaired"] == []
    assert not (journal.root / "meta/embodichain_episodes.jsonl").exists()
    assert journal.get(identity)["phase"] == "depth_committed"
    assert path.read_bytes() == before


def test_recovery_rejects_inconsistent_sdk_frame_range(tmp_path: Path) -> None:
    journal, sidecar = make_recording(tmp_path)
    identity = sidecar["episode_uuid"]
    journal.prepare(sidecar)
    advance_to(journal, identity, "depth_committed")
    path = journal.root / "meta/episodes/chunk-000/file-000.parquet"
    table = parquet.read_table(path)
    table = table.set_column(
        table.schema.get_field_index("dataset_to_index"),
        "dataset_to_index",
        arrow.array([3]),
    )
    parquet.write_table(table, path)

    report = recover_recording(journal.root, repair=True)

    assert not report["ok"]
    assert report["commits"][0]["status"] == "unresolved"
    assert report["repaired"] == []
    assert not (journal.root / "meta/embodichain_episodes.jsonl").exists()
    assert journal.get(identity)["phase"] == "depth_committed"


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
