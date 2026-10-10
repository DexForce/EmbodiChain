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

import hashlib
import json
from pathlib import Path
import uuid

import av
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from embodichain_lerobot_export import export_dataset
from lerobot.configs.recipe import TrainingRecipe
from lerobot.datasets.depth_utils import dequantize_depth, quantize_depth
from lerobot.datasets.language_render import render_sample
from lerobot.datasets.lerobot_dataset import LeRobotDataset

FPS = 10
EPISODE_LENGTH = 4
DEPTH_SHAPE = (16, 16)
OVERALL_TASK = "Move the cube to the target"
SUBTASKS = ["Pick up the cube", "Place the cube at the target"]


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.fixture
def source(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    dataset = LeRobotDataset.create(
        "local/source",
        fps=FPS,
        root=root,
        features={
            "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
            "subtask_index": {"dtype": "int64", "shape": (1,), "names": None},
            "segment_id": {"dtype": "int64", "shape": (1,), "names": None},
        },
        use_videos=False,
    )
    for episode in range(2):
        for frame in range(EPISODE_LENGTH):
            segment = frame // 2
            dataset.add_frame(
                {
                    "action": np.array([frame, episode], dtype=np.float32),
                    "subtask_index": np.array([segment], dtype=np.int64),
                    "segment_id": np.array([segment], dtype=np.int64),
                    # Legacy action_contract recordings use segment text as task.
                    "task": SUBTASKS[segment],
                }
            )
        dataset.save_episode()
    dataset.finalize()
    pd.DataFrame({"subtask_index": [0, 1]}, index=SUBTASKS).to_parquet(
        root / "meta/subtasks.parquet"
    )
    records = [
        {
            "lerobot_episode_index": index,
            "instruction": OVERALL_TASK,
            "episode_uuid": f"episode-{index}",
            "source_episode_uuid": "source-episode",
            "run_uuid": "run-uuid",
            "fragment": index == 1,
            "source_start_step": 12 if index == 1 else 0,
            "source_end_step": 16 if index == 1 else 4,
            "segments": [
                {"start_step": 0, "end_step": 2, "instruction": SUBTASKS[0]},
                {"start_step": 2, "end_step": 4, "instruction": SUBTASKS[1]},
            ],
        }
        for index in range(2)
    ]
    (root / "meta/embodichain_episodes.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return root


def _add_depth(root: Path, *, use_log: bool = True) -> list[np.ndarray]:
    sensor = {
        "shape": [*DEPTH_SHAPE, 1],
        "video.codec": "libx265",
        "video.pix_fmt": "gray12le",
        "video.depth_min": 0.01,
        "video.depth_max": 10.0,
        "video.shift": 3.5,
        "video.use_log": use_log,
        "video.quant_bits": 12,
        "video.qmax": 4095,
        "video.lossless": True,
        "episodes": {},
    }
    expected = []
    for episode in range(2):
        path = root / f"depth_videos/camera/episode_{episode:06d}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        with av.open(str(path), "w") as container:
            stream = container.add_stream("libx265", rate=FPS)
            stream.width, stream.height = DEPTH_SHAPE[1], DEPTH_SHAPE[0]
            stream.pix_fmt = "gray12le"
            stream.options = {
                "x265-params": "lossless=1:pools=1:frame-threads=1:log-level=error",
                "preset": "ultrafast",
            }
            for frame_index in range(EPISODE_LENGTH):
                # Include clipping and both code endpoints, plus intermediate depths.
                depth = np.tile(
                    np.array([-1.0, 0.01, 1.0 + episode, 12.0], dtype=np.float32),
                    (DEPTH_SHAPE[0], DEPTH_SHAPE[1] // 4),
                )
                codes = quantize_depth(depth, use_log=use_log, video_backend=None)
                if frame_index == 0:
                    expected.append(
                        dequantize_depth(
                            codes, use_log=use_log, output_unit="m", output_tensor=False
                        )
                    )
                frame = quantize_depth(depth, use_log=use_log)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        sensor["episodes"][str(episode)] = {
            "file": str(path.relative_to(root)),
            "frame_count": EPISODE_LENGTH,
        }
    _write_json(root / "depth_meta.json", {"fps": FPS, "sensors": {"camera": sensor}})
    return expected


def test_official_reader_and_recipe_preserve_both_instruction_levels(
    source: Path, tmp_path: Path
) -> None:
    before = _hashes(source)
    destination = tmp_path / "converted"
    manifest = export_dataset(source, destination)
    assert _hashes(source) == before
    assert manifest["official_samples_checked"] == 6
    dataset = LeRobotDataset("local/output", root=destination)
    recipe = TrainingRecipe.from_yaml(destination / "meta/task_subtask_recipe.json")
    for index in range(2 * EPISODE_LENGTH):
        sample = dataset[index]
        assert sample["task"] == OVERALL_TASK
        rendered = render_sample(
            recipe=recipe,
            persistent=sample["language_persistent"],
            events=None,
            t=float(sample["timestamp"]),
            sample_idx=index,
            task=sample["task"],
        )
        assert rendered["messages"] == [
            {"role": "user", "content": OVERALL_TASK},
            {"role": "assistant", "content": SUBTASKS[(index % EPISODE_LENGTH) // 2]},
        ]
        np.testing.assert_array_equal(
            sample["action"], [index % EPISODE_LENGTH, index // EPISODE_LENGTH]
        )
    assert (destination / "meta/subtasks.parquet").exists()
    assert "subtask_index" in dataset[0]
    assert (
        float(dataset.meta.stats["task_index"]["mean"][0])
        == dataset[0]["task_index"].item()
    )


def test_long_episode_language_transition_uses_stored_timestamp(tmp_path: Path) -> None:
    # At 30 FPS, frame 1002's float32 timestamp exceeds 1002 / 30 by >1 us.
    fps, length, transition = 30, 1005, 1002
    root = tmp_path / "source"
    source = LeRobotDataset.create(
        "local/source",
        fps=fps,
        root=root,
        features={"action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]}},
        use_videos=False,
    )
    for frame in range(length):
        source.add_frame(
            {"action": np.array([frame, 0], dtype=np.float32), "task": OVERALL_TASK}
        )
    source.save_episode()
    source.finalize()
    record = {
        "lerobot_episode_index": 0,
        "instruction": OVERALL_TASK,
        "length": length,
        "segments": [
            {"start_step": 0, "end_step": transition, "instruction": SUBTASKS[0]},
            {"start_step": transition, "end_step": length, "instruction": SUBTASKS[1]},
        ],
    }
    (root / "meta/embodichain_episodes.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )
    destination = tmp_path / "converted"

    export_dataset(root, destination)

    dataset = LeRobotDataset("local/output", root=destination)
    recipe = TrainingRecipe.from_yaml(destination / "meta/task_subtask_recipe.json")
    for offset, expected in ((transition - 1, SUBTASKS[0]), (transition, SUBTASKS[1])):
        sample = dataset[offset]
        rendered = render_sample(
            recipe=recipe,
            persistent=sample["language_persistent"],
            events=None,
            t=float(sample["timestamp"]),
            sample_idx=offset,
            task=sample["task"],
        )
        assert rendered["messages"][-1]["content"] == expected


@pytest.mark.parametrize("use_log", [True, False])
def test_depth_reads_in_metres_and_millimetres_with_official_stats(
    source: Path, tmp_path: Path, use_log: bool
) -> None:
    expected = _add_depth(source, use_log=use_log)
    destination = tmp_path / "converted"
    export_dataset(source, destination)
    dataset = LeRobotDataset(
        "local/output", root=destination, video_backend="pyav", depth_output_unit="m"
    )
    key = "observation.depth.camera"
    assert dataset.meta.depth_keys == [key]
    for episode in range(2):
        np.testing.assert_allclose(
            dataset[episode * EPISODE_LENGTH][key], expected[episode], atol=1e-6
        )
        entry = dataset.meta.episodes[episode]
        assert entry[f"videos/{key}/from_timestamp"] == 0
        assert entry[f"videos/{key}/to_timestamp"] == EPISODE_LENGTH / FPS
    assert dataset.meta.features[key]["info"]["depth_unit"] == "m"
    assert dataset.meta.features[key]["info"]["video.quant_bits"] == 12
    assert dataset.meta.stats[key]["mean"].shape == (1, 1, 1)
    np.testing.assert_allclose(
        dataset.meta.stats[key]["mean"], np.mean(expected), atol=1e-6
    )
    millimetres = LeRobotDataset("local/output", root=destination, video_backend="pyav")
    np.testing.assert_allclose(millimetres[0][key], np.rint(expected[0] * 1000), atol=1)
    np.testing.assert_allclose(
        millimetres.meta.stats[key]["mean"], dataset.meta.stats[key]["mean"] * 1000
    )
    assert not (destination / "depth_videos").exists()


def test_fragment_preserves_lineage_and_rebases_annotation_time(
    source: Path, tmp_path: Path
) -> None:
    # Simulate a cropped dataset whose stored timestamps retain source time.
    for path in (source / "data").rglob("*.parquet"):
        table = pq.read_table(path)
        values = np.asarray(table["timestamp"]).copy()
        indices = np.asarray(table["episode_index"])
        values[indices == 1] += 1.2
        table = table.set_column(
            table.column_names.index("timestamp"), "timestamp", pa.array(values)
        )
        pq.write_table(table, path)
    destination = tmp_path / "converted"
    export_dataset(source, destination)
    dataset = LeRobotDataset("local/output", root=destination)
    sample = dataset[EPISODE_LENGTH]
    assert float(sample["timestamp"]) == 0
    assert sample["language_persistent"][0]["timestamp"] == 0
    records = [
        json.loads(line)
        for line in (destination / "meta/embodichain_episodes.jsonl")
        .read_text()
        .splitlines()
    ]
    fragment = records[1]
    assert fragment["episode_uuid"] == "episode-1"
    assert fragment["source_episode_uuid"] == "source-episode"
    assert fragment["run_uuid"] == "run-uuid"
    assert fragment["source_start_step"] == 12
    assert fragment["lerobot_export"]["timestamp_origin"] == pytest.approx(1.2)


def test_legacy_missing_ids_get_reproducible_identity(
    source: Path, tmp_path: Path
) -> None:
    sidecar = source / "meta/embodichain_episodes.jsonl"
    records = [json.loads(line) for line in sidecar.read_text().splitlines()]
    for record in records:
        del record["episode_uuid"], record["source_episode_uuid"], record["run_uuid"]
    sidecar.write_text("".join(json.dumps(record) + "\n" for record in records))
    outputs = [tmp_path / "one", tmp_path / "two"]
    for path in outputs:
        export_dataset(source, path)
    assert (outputs[0] / "meta/embodichain_episodes.jsonl").read_text() == (
        outputs[1] / "meta/embodichain_episodes.jsonl"
    ).read_text()
    converted = [
        json.loads(line)
        for line in (outputs[0] / "meta/embodichain_episodes.jsonl")
        .read_text()
        .splitlines()
    ]
    assert all("source_episode_uuid" not in record for record in converted)
    assert all(record["lineage_unknown"] for record in converted)
    assert len({record["source_dataset_fingerprint"] for record in converted}) == 1


def test_failed_depth_conversion_never_publishes_destination(
    source: Path, tmp_path: Path
) -> None:
    _add_depth(source)
    metadata = json.loads((source / "depth_meta.json").read_text())
    metadata["sensors"]["camera"]["episodes"]["1"]["frame_count"] = EPISODE_LENGTH - 1
    _write_json(source / "depth_meta.json", metadata)
    before = _hashes(source)
    destination = tmp_path / "converted"
    with pytest.raises(ValueError, match="incomplete depth"):
        export_dataset(source, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".converted.staging-*"))
    assert _hashes(source) == before


def test_existing_destination_is_never_changed(source: Path, tmp_path: Path) -> None:
    destination = tmp_path / "existing"
    destination.mkdir()
    marker = destination / "user-file"
    marker.write_text("preserve")
    with pytest.raises(FileExistsError):
        export_dataset(source, destination)
    assert marker.read_text() == "preserve"


def test_destination_must_be_outside_source(source: Path) -> None:
    with pytest.raises(ValueError, match="outside"):
        export_dataset(source, source / "converted")


def test_depth_reference_cannot_escape_dataset(source: Path, tmp_path: Path) -> None:
    _add_depth(source)
    metadata = json.loads((source / "depth_meta.json").read_text())
    metadata["sensors"]["camera"]["episodes"]["0"]["file"] = "../outside.mp4"
    _write_json(source / "depth_meta.json", metadata)
    with pytest.raises(ValueError, match="escapes"):
        export_dataset(source, tmp_path / "converted")


def test_invalid_episode_indices_are_rejected_before_publish(
    source: Path, tmp_path: Path
) -> None:
    path = next((source / "meta/episodes").rglob("*.parquet"))
    rows = pq.read_table(path).to_pylist()
    rows[0]["dataset_to_index"] -= 1
    pq.write_table(pa.Table.from_pylist(rows), path)
    destination = tmp_path / "converted"
    with pytest.raises(ValueError, match="global frame indices"):
        export_dataset(source, destination)
    assert not destination.exists()


def test_annotations_are_not_overwritten_on_reexport(
    source: Path, tmp_path: Path
) -> None:
    destination = tmp_path / "converted"
    export_dataset(source, destination)
    with pytest.raises(ValueError, match="already contains"):
        export_dataset(destination, tmp_path / "another")


def _add_rgb(source: Path, *, extra_frames: int = 0) -> tuple[str, Path]:
    key = "observation.images.camera"
    relative_path = f"videos/{key}/chunk-000/file-000.mp4"
    path = source / relative_path
    path.parent.mkdir(parents=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width, stream.height = DEPTH_SHAPE
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "ultrafast", "crf": "0"}
        for index in range(2 * EPISODE_LENGTH + extra_frames):
            frame = av.VideoFrame.from_ndarray(
                np.full((*DEPTH_SHAPE, 3), index * 20, dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    info_path = source / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["video_path"] = (
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    )
    info["features"][key] = {
        "dtype": "video",
        "shape": [*DEPTH_SHAPE, 3],
        "names": ["height", "width", "channels"],
        "info": {"video.codec": "h264", "video.pix_fmt": "yuv420p", "video.fps": FPS},
    }
    _write_json(info_path, info)
    for episode_path in (source / "meta/episodes").rglob("*.parquet"):
        rows = pq.read_table(episode_path).to_pylist()
        for row in rows:
            index = row["episode_index"]
            row.update(
                {
                    f"videos/{key}/chunk_index": 0,
                    f"videos/{key}/file_index": 0,
                    f"videos/{key}/from_timestamp": index * EPISODE_LENGTH / FPS,
                    f"videos/{key}/to_timestamp": (index + 1) * EPISODE_LENGTH / FPS,
                }
            )
        pq.write_table(pa.Table.from_pylist(rows), episode_path)
    return key, path


def test_rgb_video_bytes_and_episode_offsets_are_preserved(
    source: Path, tmp_path: Path
) -> None:
    key, path = _add_rgb(source)
    destination = tmp_path / "converted"
    export_dataset(source, destination)
    assert (destination / path.relative_to(source)).read_bytes() == path.read_bytes()
    original = LeRobotDataset("local/source", root=source, video_backend="pyav")
    converted = LeRobotDataset(
        "local/converted", root=destination, video_backend="pyav"
    )
    for index in range(2 * EPISODE_LENGTH):
        np.testing.assert_array_equal(original[index][key], converted[index][key])


def test_rgb_rebase_updates_both_offsets_and_keeps_source_frames(
    source: Path, tmp_path: Path
) -> None:
    key, _ = _add_rgb(source, extra_frames=2)
    origin = 2 / FPS
    for path in (source / "data").rglob("*.parquet"):
        table = pq.read_table(path)
        values = np.asarray(table["timestamp"]).copy()
        values[np.asarray(table["episode_index"]) == 1] += origin
        table = table.set_column(
            table.column_names.index("timestamp"), "timestamp", pa.array(values)
        )
        pq.write_table(table, path)
    destination = tmp_path / "converted"
    export_dataset(source, destination)
    original = LeRobotDataset("local/source", root=source, video_backend="pyav")
    converted = LeRobotDataset("local/output", root=destination, video_backend="pyav")
    episode = converted.meta.episodes[1]
    assert episode[f"videos/{key}/from_timestamp"] == pytest.approx(0.6)
    assert episode[f"videos/{key}/to_timestamp"] == pytest.approx(1.0)
    for index in range(EPISODE_LENGTH, 2 * EPISODE_LENGTH):
        np.testing.assert_array_equal(original[index][key], converted[index][key])


def test_rgb_interval_past_stream_is_rejected(source: Path, tmp_path: Path) -> None:
    key, _ = _add_rgb(source)
    for path in (source / "meta/episodes").rglob("*.parquet"):
        rows = pq.read_table(path).to_pylist()
        rows[-1][f"videos/{key}/from_timestamp"] += 0.4
        rows[-1][f"videos/{key}/to_timestamp"] += 0.4
        pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match="exceeds its stream"):
        export_dataset(source, tmp_path / "converted")


@pytest.mark.parametrize("empty_segment", [False, True])
def test_segment_overlap_is_rejected_and_empty_segments_are_skipped(
    source: Path, tmp_path: Path, empty_segment: bool
) -> None:
    path = source / "meta/embodichain_episodes.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["segments"].append(
        {"start_step": 1, "end_step": 1 if empty_segment else 3, "instruction": "extra"}
    )
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    destination = tmp_path / "converted"
    if empty_segment:
        export_dataset(source, destination)
        dataset = LeRobotDataset("local/output", root=destination)
        assert len(dataset[0]["language_persistent"]) == 2
    else:
        with pytest.raises(ValueError, match="overlap"):
            export_dataset(source, destination)


def test_missing_sidecar_length_is_filled_and_conflicting_length_is_rejected(
    source: Path, tmp_path: Path
) -> None:
    destination = tmp_path / "converted"
    export_dataset(source, destination)
    assert all(
        json.loads(line)["length"] == EPISODE_LENGTH
        for line in (destination / "meta/embodichain_episodes.jsonl")
        .read_text()
        .splitlines()
    )
    path = source / "meta/embodichain_episodes.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["length"] = EPISODE_LENGTH + 1
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    with pytest.raises(ValueError, match="sidecar length"):
        export_dataset(source, tmp_path / "invalid")


def test_plural_and_parent_lineage_are_preserved_as_known(
    source: Path, tmp_path: Path
) -> None:
    path = source / "meta/embodichain_episodes.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        del record["source_episode_uuid"]
    records[0]["source_episode_uuids"] = ["parent-a", "parent-b"]
    records[1]["lineage"] = {"parent_episode_uuid": "parent-a"}
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    destination = tmp_path / "converted"
    export_dataset(source, destination)
    converted = [
        json.loads(line)
        for line in (destination / "meta/embodichain_episodes.jsonl")
        .read_text()
        .splitlines()
    ]
    assert converted[0]["source_episode_uuids"] == ["parent-a", "parent-b"]
    assert converted[1]["lineage"] == {"parent_episode_uuid": "parent-a"}
    assert all(not record.get("lineage_unknown") for record in converted)


def _add_journals(source: Path) -> list[Path]:
    sidecar_path = source / "meta/embodichain_episodes.jsonl"
    records = [json.loads(line) for line in sidecar_path.read_text().splitlines()]
    paths = []
    for record in records:
        index = record["lerobot_episode_index"]
        identity = str(uuid.uuid5(uuid.NAMESPACE_URL, f"test-episode:{index}"))
        record.update({"episode_uuid": identity, "length": EPISODE_LENGTH})
        path = source / f"meta/embodichain_commits/{identity}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(
            path,
            {
                "schema_version": 1,
                "episode_uuid": identity,
                "phase": "complete",
                "sidecar": record,
                "depth_sensors": ["camera"],
                "updated_at": "2026-10-10T00:00:00+00:00",
            },
        )
        paths.append(path)
    sidecar_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return paths


def test_completed_source_journals_are_archived_with_original_sidecars(
    source: Path, tmp_path: Path
) -> None:
    _add_depth(source)
    journals = _add_journals(source)
    original = {path.name: path.read_bytes() for path in journals}
    destination = tmp_path / "converted"
    manifest = export_dataset(source, destination)
    assert manifest["source_commit_records"] == len(journals)
    assert not (destination / "meta/embodichain_commits").exists()
    archive = (
        destination / "meta/embodichain_source_commits" / manifest["source_fingerprint"]
    )
    assert {path.name: path.read_bytes() for path in archive.glob("*.json")} == original


@pytest.mark.parametrize("fault", ["unfinished", "conflicting"])
def test_unfinished_or_conflicting_source_journals_are_rejected(
    source: Path, tmp_path: Path, fault: str
) -> None:
    _add_depth(source)
    journals = _add_journals(source)
    record = json.loads(journals[0].read_text())
    if fault == "unfinished":
        record["phase"] = "lerobot_committing"
    else:
        record["sidecar"]["instruction"] = "Conflicting task"
    _write_json(journals[0], record)
    destination = tmp_path / "converted"
    with pytest.raises(ValueError, match="journal"):
        export_dataset(source, destination)
    assert not destination.exists()
