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
from pathlib import Path
import subprocess
import sys
from typing import Any
import uuid

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from embodichain.data_pipeline.datasets.inspection import (
    create_split_manifest,
    main,
    validate_dataset,
)

FPS = 10
EPISODE_LENGTH = 3


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _write_sidecars(root: Path, records: list[dict[str, Any]]) -> None:
    (root / "meta/embodichain_episodes.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )


def _dataset(
    root: Path, *, count: int = 2
) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]]]:
    root.mkdir()
    info = {
        "codebase_version": "v3.0",
        "fps": FPS,
        "total_episodes": count,
        "total_frames": count * EPISODE_LENGTH,
        "total_tasks": 1,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "timestamp": {"dtype": "float32", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "subtask_index": {"dtype": "int64", "shape": [1]},
            "action": {
                "dtype": "float32",
                "shape": [3],
                "info": {
                    "embodichain.action_contract": {
                        "representation": "joint_position_parallel_gripper",
                        "action_terms": [
                            {
                                "slice": [0, 2],
                                "term": {
                                    "action_dim": 2,
                                    "representation": "joint_position",
                                },
                            },
                            {
                                "slice": [2, 3],
                                "term": {
                                    "action_dim": 1,
                                    "representation": "parallel_gripper",
                                },
                            },
                        ],
                    },
                },
            },
        },
    }
    _write_json(root / "meta/info.json", info)
    pd.DataFrame({"task_index": [0]}, index=["Move the object"]).to_parquet(
        root / "meta/tasks.parquet"
    )
    pd.DataFrame({"subtask_index": [0, 1]}, index=["Pick", "Place"]).to_parquet(
        root / "meta/subtasks.parquet"
    )
    episodes = []
    sidecars = []
    frames = []
    for index in range(count):
        episodes.append(
            {
                "episode_index": index,
                "length": EPISODE_LENGTH,
                "dataset_from_index": index * EPISODE_LENGTH,
                "dataset_to_index": (index + 1) * EPISODE_LENGTH,
                "data/chunk_index": 0,
                "data/file_index": 0,
            }
        )
        sidecars.append(
            {
                "lerobot_episode_index": index,
                "length": EPISODE_LENGTH,
                "run_uuid": "recording-run",
                "episode_uuid": f"episode-{index}",
                "source_episode_uuid": f"episode-{index}",
                "scene_id": f"scene-{index}",
                "success": True,
                "recovery_count": 0,
                "physical_validation": {"accepted": True},
                "segments": [
                    {
                        "segment_id": 4,
                        "start_step": 0,
                        "end_step": 1,
                        "instruction": "Pick",
                        "success": True,
                    },
                    {
                        "segment_id": 9,
                        "start_step": 1,
                        "end_step": EPISODE_LENGTH,
                        "instruction": "Place",
                        "success": True,
                    },
                ],
            }
        )
        for frame in range(EPISODE_LENGTH):
            frames.append(
                {
                    "timestamp": frame / FPS,
                    "episode_index": index,
                    "frame_index": frame,
                    "index": index * EPISODE_LENGTH + frame,
                    "task_index": 0,
                    "subtask_index": int(frame > 0),
                    "action": [0.1, -0.1, -1.0],
                    "annotation.episode_step": frame,
                    "annotation.segment_id": 4 if frame == 0 else 9,
                    "annotation.segment_step": max(frame - 1, 0),
                    "annotation.segment_start": frame in (0, 1),
                    "annotation.segment_end": frame in (0, EPISODE_LENGTH - 1),
                    "annotation.segment_accepted": True,
                    "annotation.segment_attempt_id": 0,
                    "annotation.continuity_id": 0,
                }
            )
    metadata_path = root / "meta/episodes/chunk-000/file-000.parquet"
    metadata_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(episodes), metadata_path)
    _write_frames(root, frames)
    _write_sidecars(root, sidecars)
    return root, frames, sidecars


def _write_frames(root: Path, frames: list[dict[str, Any]]) -> None:
    path = root / "data/chunk-000/file-000.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(frames), path)


def _codes(root: Path, **kwargs: Any) -> set[str]:
    return {issue.code for issue in validate_dataset(root, **kwargs).issues}


def test_validate_reads_real_parquet_in_small_batches(tmp_path: Path) -> None:
    root, _, _ = _dataset(tmp_path / "dataset")
    report = validate_dataset(root, batch_size=2)
    assert report.to_dict() == {
        "ok": True,
        "root": str(root),
        "episodes": 2,
        "frames": 6,
        "errors": 0,
        "warnings": 0,
        "issues": [],
    }


@pytest.mark.parametrize(
    ("column", "value", "code"),
    [
        ("timestamp", 0.33, "timestamp"),
        ("frame_index", 6, "frame_index"),
        ("index", None, "frame_index"),
        ("episode_index", 20, "episode_reference"),
        ("task_index", 1, "language_reference"),
        ("subtask_index", -1, "language_reference"),
        ("action", [float("nan"), 0.0, -1.0], "nonfinite"),
        ("action", [0.0, 0.0], "feature_shape"),
        ("action", [0.0, 0.0, 1.01], "gripper_bounds"),
        ("annotation.segment_id", 8, "frame_annotation"),
        ("annotation.continuity_id", 4, "frame_annotation"),
    ],
)
def test_validate_identifies_corrupt_frame(
    tmp_path: Path, column: str, value: Any, code: str
) -> None:
    root, frames, _ = _dataset(tmp_path / "dataset")
    frames[1][column] = value
    _write_frames(root, frames)
    report = validate_dataset(root, batch_size=1)
    assert code in {issue.code for issue in report.issues}
    assert not report.ok
    assert any("file-000.parquet:1" in issue.location for issue in report.issues)


def test_validate_checks_spans_counts_duplicates_and_pending_commit(
    tmp_path: Path,
) -> None:
    root, frames, sidecars = _dataset(tmp_path / "dataset")
    frames.pop()
    _write_frames(root, frames)
    sidecars[0]["segments"][0]["end_step"] = EPISODE_LENGTH + 1
    sidecars[1]["episode_uuid"] = sidecars[0]["episode_uuid"]
    _write_sidecars(root, sidecars)
    _write_json(
        root / "meta/embodichain_commits/pending.json", {"phase": "lerobot_committed"}
    )
    assert {
        "segment_span",
        "episode_length",
        "duplicate_identity",
        "pending_commit",
        "dataset_totals",
    } <= _codes(root)


def test_official_dataset_without_sidecar_remains_valid(tmp_path: Path) -> None:
    root, _, _ = _dataset(tmp_path / "dataset")
    (root / "meta/embodichain_episodes.jsonl").unlink()
    assert validate_dataset(root).ok


def test_unordered_nonoverlapping_spans_and_zero_length_failures_remain_valid(
    tmp_path: Path,
) -> None:
    root, _, sidecars = _dataset(tmp_path / "dataset")
    sidecars[0]["segments"].reverse()
    sidecars[0]["segments"].append(
        {"segment_id": 10, "start_step": 2, "end_step": 2, "success": False}
    )
    _write_sidecars(root, sidecars)
    assert validate_dataset(root).ok


def test_replay_link_checks_source_and_file_without_loading_torch(
    tmp_path: Path,
) -> None:
    root, _, sidecars = _dataset(tmp_path / "dataset")
    sidecars[0]["replay_artifact"] = {
        "path": "replays/missing.pt",
        "source_episode_uuid": "wrong-source",
    }
    _write_sidecars(root, sidecars)
    assert {"replay_missing", "replay_identity"} <= _codes(root)


def _depth_metadata(root: Path) -> dict[str, Any]:
    metadata = {
        "fps": FPS,
        "sensors": {
            "camera": {
                "shape": [8, 8, 1],
                "video.input_unit": "auto",
                "video.output_unit": "m",
                "video.depth_min": 0.01,
                "video.depth_max": 10.0,
                "episodes": {
                    str(index): {
                        "frame_count": EPISODE_LENGTH,
                        "file": f"depth_videos/{index}.mp4",
                    }
                    for index in range(2)
                },
            }
        },
    }
    _write_json(root / "depth_meta.json", metadata)
    return metadata


def test_depth_metadata_and_optional_media_presence(tmp_path: Path) -> None:
    root, _, _ = _dataset(tmp_path / "dataset")
    depth = _depth_metadata(root)
    assert validate_dataset(root).ok
    assert "media_missing" in _codes(root, check_media=True)
    depth["sensors"]["camera"]["video.output_unit"] = "cm"
    depth["sensors"]["camera"]["episodes"]["0"]["frame_count"] = 1
    _write_json(root / "depth_meta.json", depth)
    assert {"depth_unit", "depth_length"} <= _codes(root)


@pytest.mark.parametrize(
    ("input_unit", "output_unit", "valid"),
    [("auto", "m", True), ("mm", "m", True), ("m", "auto", False), ("cm", "m", False)],
)
def test_depth_producer_auto_input_is_supported_but_output_units_are_explicit(
    tmp_path: Path, input_unit: str, output_unit: str, valid: bool
) -> None:
    root, _, _ = _dataset(tmp_path / "dataset")
    metadata = _depth_metadata(root)
    sensor = metadata["sensors"]["camera"]
    sensor.update(
        {
            "is_depth_map": True,
            "video.codec": "libx265",
            "video.pix_fmt": "gray12le",
            "video.fps": FPS,
            "video.quant_bits": 12,
            "video.qmax": 4095,
            "video.shift": 3.5,
            "video.use_log": True,
            "video.input_unit": input_unit,
            "video.output_unit": output_unit,
        }
    )
    _write_json(root / "depth_meta.json", metadata)
    report = validate_dataset(root)
    assert report.ok is valid
    assert ("depth_unit" in {issue.code for issue in report.issues}) is not valid


def _video(path: Path, count: int) -> None:
    av = pytest.importorskip("av")
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=FPS)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        for _ in range(count):
            frame = av.VideoFrame.from_ndarray(
                np.zeros((16, 16, 3), dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_media_decode_validates_shared_v3_video_frame_ranges(tmp_path: Path) -> None:
    root, _, _ = _dataset(tmp_path / "dataset")
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    key = "observation.images.camera"
    info["features"][key] = {"dtype": "video", "shape": [16, 16, 3]}
    _write_json(info_path, info)
    episode_path = root / "meta/episodes/chunk-000/file-000.parquet"
    episodes = pq.read_table(episode_path).to_pylist()
    for index, episode in enumerate(episodes):
        episode.update(
            {
                f"videos/{key}/chunk_index": 0,
                f"videos/{key}/file_index": 0,
                f"videos/{key}/from_timestamp": index * EPISODE_LENGTH / FPS,
                f"videos/{key}/to_timestamp": (index + 1) * EPISODE_LENGTH / FPS,
            }
        )
    pq.write_table(pa.Table.from_pylist(episodes), episode_path)
    video = root / f"videos/{key}/chunk-000/file-000.mp4"
    _video(video, 2 * EPISODE_LENGTH)
    assert validate_dataset(root, media_frame_counts=True).ok
    _video(video, 2 * EPISODE_LENGTH - 1)
    assert "video_length" in _codes(root, media_frame_counts=True)


def _split_for(manifest: dict[str, Any], episode: int) -> str:
    return next(
        name for name, indices in manifest["splits"].items() if episode in indices
    )


def test_split_preserves_transitive_lineage_and_is_deterministic(
    tmp_path: Path,
) -> None:
    root, _, sidecars = _dataset(tmp_path / "dataset", count=6)
    sidecars[1]["source_episode_uuid"] = sidecars[0]["episode_uuid"]
    sidecars[2]["source_episode_uuid"] = sidecars[1]["episode_uuid"]
    _write_sidecars(root, sidecars)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    manifest = create_split_manifest(
        root, tmp_path / "split.json", fractions=(0.5, 0.25, 0.25), seed=17
    )
    assert _split_for(manifest, 0) == _split_for(manifest, 1) == _split_for(manifest, 2)
    assert manifest == create_split_manifest(
        root, tmp_path / "split2.json", fractions=(0.5, 0.25, 0.25), seed=17
    )
    assert json.loads((tmp_path / "split.json").read_text()) == manifest
    assert all(path.read_bytes() == content for path, content in before.items())


def test_multi_source_augmented_episode_groups_both_sources_transitively(
    tmp_path: Path,
) -> None:
    root, _, sidecars = _dataset(tmp_path / "dataset", count=6)
    sidecars[2]["provenance"] = {"source_episode_uuids": ["episode-0", "episode-1"]}
    sidecars[3]["lineage"] = {"source_episode_uuids": ["episode-2", "episode-4"]}
    _write_sidecars(root, sidecars)
    manifest = create_split_manifest(
        root, tmp_path / "split.json", fractions=(0.5, 0.25, 0.25)
    )
    assert len({_split_for(manifest, index) for index in range(5)}) == 1
    assert any(
        group["episode_indices"] == list(range(5)) for group in manifest["groups"]
    )


def test_expansion_reference_groups_template_variants_and_separates_sources(
    tmp_path: Path,
) -> None:
    root, _, sidecars = _dataset(tmp_path / "dataset", count=4)
    for index, sidecar in enumerate(sidecars):
        sidecar["expansion"] = [
            {
                "selected_candidate_lineage": {
                    "scene_case_id": "same-scene",
                    "initial_state_id": "same-initial-state",
                    "source_id": "reference-A" if index < 2 else "reference-B",
                    "source_revision": "revision-1",
                    "candidate_id": f"candidate-{index}",
                    "geometry_family_id": f"family-{index}",
                    "template_id": f"generated-template-{index}",
                    "parent_id": None,
                }
            }
        ]
    _write_sidecars(root, sidecars)
    manifest = create_split_manifest(
        root, tmp_path / "split.json", fractions=(0.5, 0.5, 0)
    )
    assert _split_for(manifest, 0) == _split_for(manifest, 1)
    assert _split_for(manifest, 2) == _split_for(manifest, 3)
    assert _split_for(manifest, 0) != _split_for(manifest, 2)


def test_explicit_unknown_legacy_source_fingerprint_limits_conservative_group(
    tmp_path: Path,
) -> None:
    root, _, sidecars = _dataset(tmp_path / "dataset", count=4)
    for index, sidecar in enumerate(sidecars):
        sidecar.pop("source_episode_uuid")
        sidecar.pop("run_uuid")
        sidecar["lineage_unknown"] = True
        sidecar["source_dataset_fingerprint"] = "source-A" if index < 2 else "source-B"
    _write_sidecars(root, sidecars)
    manifest = create_split_manifest(
        root, tmp_path / "split.json", fractions=(0.5, 0.5, 0)
    )
    assert _split_for(manifest, 0) == _split_for(manifest, 1)
    assert _split_for(manifest, 2) == _split_for(manifest, 3)
    assert _split_for(manifest, 0) != _split_for(manifest, 2)


@pytest.mark.parametrize("sources", [[], ["episode-0", None], [""], [7], "episode-0"])
def test_multi_source_identity_rejects_invalid_list_contents(
    tmp_path: Path, sources: Any
) -> None:
    root, _, sidecars = _dataset(tmp_path / "dataset")
    sidecars[0]["source_episode_uuids"] = sources
    _write_sidecars(root, sidecars)
    assert "identity" in _codes(root)
    with pytest.raises(ValueError, match="source_episode_uuids"):
        create_split_manifest(root, tmp_path / "split.json")


def test_split_unknown_provenance_is_conservative_unless_opted_in(
    tmp_path: Path,
) -> None:
    root, _, _ = _dataset(tmp_path / "dataset", count=6)
    (root / "meta/embodichain_episodes.jsonl").unlink()
    manifest = create_split_manifest(
        root, tmp_path / "split.json", fractions=(0.5, 0.25, 0.25)
    )
    assert len(manifest["groups"]) == 1
    assert manifest["assumptions"]
    independent = create_split_manifest(
        root,
        tmp_path / "independent.json",
        fractions=(0.5, 0.25, 0.25),
        allow_legacy_independent=True,
    )
    assert len(independent["groups"]) == 6
    assert independent["allow_legacy_independent"]


def test_scene_groups_cannot_split_lineage_across_scene_ids(tmp_path: Path) -> None:
    root, _, sidecars = _dataset(tmp_path / "dataset", count=5)
    sidecars[1]["source_episode_uuid"] = sidecars[0]["episode_uuid"]
    sidecars[2]["scene_id"] = sidecars[1]["scene_id"]
    _write_sidecars(root, sidecars)
    manifest = create_split_manifest(
        root, tmp_path / "split.json", group_by="scene", fractions=(0.5, 0.25, 0.25)
    )
    assert _split_for(manifest, 0) == _split_for(manifest, 1) == _split_for(manifest, 2)
    del sidecars[3]["scene_id"]
    _write_sidecars(root, sidecars)
    with pytest.raises(ValueError, match="scene_id"):
        create_split_manifest(root, tmp_path / "missing.json", group_by="scene")


def test_quality_filters_fail_closed_and_do_not_split_related_sources(
    tmp_path: Path,
) -> None:
    root, _, sidecars = _dataset(tmp_path / "dataset", count=5)
    sidecars[1]["success"] = False
    sidecars[2].pop("recovery_count")
    sidecars[3]["recovery_count"] = 2
    sidecars[4].pop("physical_validation")
    sidecars[4]["physical_objective"] = {
        "accepted": True
    }  # A goal is not measured proof.
    _write_sidecars(root, sidecars)
    manifest = create_split_manifest(
        root,
        tmp_path / "split.json",
        success_only=True,
        max_recoveries=0,
        require_physical_valid=True,
    )
    assert manifest["splits"]["train"] == [0]
    assert {
        item["episode_index"]: item["reasons"] for item in manifest["excluded"]
    } == {
        1: ["success_not_confirmed"],
        2: ["recovery_count_unknown"],
        3: ["recovery_limit"],
        4: ["physical_validation_not_confirmed"],
    }


def test_split_rejects_dataset_file_overwrite_and_incomplete_commit(
    tmp_path: Path,
) -> None:
    root, _, _ = _dataset(tmp_path / "dataset")
    with pytest.raises(ValueError, match="separate manifest"):
        create_split_manifest(root, root / "meta/info.json")
    _write_json(root / "meta/embodichain_commits/partial.json", {"phase": "prepared"})
    with pytest.raises(ValueError, match="Incomplete recording commit"):
        create_split_manifest(root, tmp_path / "split.json")


@pytest.mark.parametrize("fault", [None, "identity", "index", "snapshot"])
def test_complete_journal_matches_persisted_identity_episode_and_sidecar(
    tmp_path: Path, fault: str | None
) -> None:
    root, _, sidecars = _dataset(tmp_path / "dataset")
    identity = str(uuid.uuid4())
    sidecars[0]["episode_uuid"] = identity
    _write_sidecars(root, sidecars)
    target = json.loads(json.dumps(sidecars[0]))
    record = {
        "schema_version": 1,
        "episode_uuid": identity,
        "phase": "complete",
        "sidecar": target,
        "depth_sensors": [],
    }
    if fault == "identity":
        record["episode_uuid"] = str(uuid.uuid4())
    elif fault == "index":
        target["lerobot_episode_index"] = 5
    elif fault == "snapshot":
        target["success"] = False
    _write_json(root / f"meta/embodichain_commits/{identity}.json", record)
    if fault is None:
        assert validate_dataset(root).ok
        assert create_split_manifest(root, tmp_path / "split.json")
    else:
        expected_code = {
            "identity": "journal_metadata",
            "index": "journal_episode",
            "snapshot": "journal_snapshot",
        }[fault]
        assert expected_code in _codes(root)
        with pytest.raises(ValueError, match="Dataset has"):
            create_split_manifest(root, tmp_path / "split.json")


def test_filters_accept_real_row_local_runtime_and_measured_objective_schema(
    tmp_path: Path,
) -> None:
    root, _, sidecars = _dataset(tmp_path / "dataset", count=3)
    for index, sidecar in enumerate(sidecars):
        sidecar.pop("physical_validation")
        sidecar.pop("recovery_count")
        sidecar["env_id"] = 7
        sidecar["physical_objective"] = {
            "predicate": "ordered_stable_regions",
            "success": True,
            "metrics": {"measurement_valid": index != 2, "stable": True},
        }
        runtime = {
            "schema_version": 2,
            "kind": "skill_result",
            "env_ids": [3, 7],
            "events": [
                {"kind": "action_retry", "env_mask": [False, index == 1]},
                {"kind": "replanned", "env_mask": [False, index == 1]},
                {"kind": "replanned", "env_mask": [True, False]},
            ],
            "workflow_recoveries": [],
        }
        for segment in sidecar["segments"]:
            segment["metadata"] = {"runtime": runtime}
    _write_sidecars(root, sidecars)
    manifest = create_split_manifest(
        root, tmp_path / "split.json", max_recoveries=0, require_physical_valid=True
    )
    assert manifest["splits"]["train"] == [0]
    assert {
        item["episode_index"]: item["reasons"] for item in manifest["excluded"]
    } == {
        1: ["recovery_limit"],
        2: ["physical_validation_not_confirmed"],
    }


def test_frame_language_labels_must_match_sidecar_hierarchy(tmp_path: Path) -> None:
    root, frames, sidecars = _dataset(tmp_path / "dataset")
    sidecars[0]["instruction"] = "Wrong overall task"
    _write_sidecars(root, sidecars)
    frames[1]["subtask_index"] = 0
    _write_frames(root, frames)
    assert {"task_description", "subtask_description"} <= _codes(root)


def test_external_replay_links_require_explicit_marker_and_accept_parent_source(
    tmp_path: Path,
) -> None:
    root, _, sidecars = _dataset(tmp_path / "dataset")
    replay = tmp_path / "replay.pt"
    replay.write_bytes(b"No deserialization is needed")
    sidecars[0]["source_episode_uuid"] = "older-root"
    sidecars[0]["parent_episode_uuid"] = "immediate-parent"
    sidecars[0]["replay_artifact"] = {
        "path": str(replay),
        "source_episode_uuid": "immediate-parent",
        "initial_state_step": 0,
    }
    _write_sidecars(root, sidecars)
    assert "replay_path" in _codes(root)
    sidecars[0]["replay_artifact"]["external"] = True
    _write_sidecars(root, sidecars)
    assert validate_dataset(root).ok


def test_full_augmented_episode_replay_uses_own_initial_state_and_zero_alignment(
    tmp_path: Path,
) -> None:
    root, _, sidecars = _dataset(tmp_path / "dataset")
    replay = root / "replay.pt"
    replay.write_bytes(b"Only existence is inspected")
    sidecars[0]["source_episode_uuid"] = "original-before-augmentation"
    sidecars[0]["replay_artifact"] = {
        "path": "replay.pt",
        "source_episode_uuid": sidecars[0]["episode_uuid"],
        "initial_state_step": 0,
        "state_alignment": "source_episode",
    }
    _write_sidecars(root, sidecars)
    assert validate_dataset(root).ok
    sidecars[0]["replay_artifact"]["initial_state_step"] = 1
    _write_sidecars(root, sidecars)
    assert "replay_state" in _codes(root)


def test_validate_report_cannot_overwrite_source_owned_metadata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, _, _ = _dataset(tmp_path / "dataset")
    info = root / "meta/info.json"
    before = info.read_bytes()
    assert main(["validate", str(root), "--output", str(info)]) == 2
    assert "separate manifest/report" in json.loads(capsys.readouterr().out)["error"]
    assert info.read_bytes() == before


def test_cli_reports_errors_and_runs_without_simulation_imports(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, frames, _ = _dataset(tmp_path / "dataset")
    assert main(["validate", str(root)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"]
    frames[0]["action"][-1] = 2.0
    _write_frames(root, frames)
    assert main(["validate", str(root)]) == 1
    assert not json.loads(capsys.readouterr().out)["ok"]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from embodichain.data_pipeline.datasets import inspection; "
            "assert not any(name.startswith('embodichain.lab.sim') for name in sys.modules)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
