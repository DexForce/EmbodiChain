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

"""Pure-file history proof fixtures; no arrays, runtime or GPU initialization."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from typing import Any, Callable

import pytest

from scripts.benchmark.rendering.frozen_source import (
    artifact,
    read_json,
    sha256,
    stable_hash,
    write_json,
)
from scripts.benchmark.rendering.history_contract import (
    build_history_schedule,
    frame_history,
    validate_history_proof,
)
from scripts.benchmark.rendering.quality_protocol import (
    protocol_for_modalities,
    protocol_requirements,
)


def build_valid_history_fixture(
    root: Path,
    *,
    warmup: int = 2,
    measured: int = 6,
    replay_length: int = 3,
    num_envs: int = 1,
    cameras_per_env: int = 1,
    modalities: tuple[str, ...] = ("rgb",),
    write_arrays: Callable[[Path, int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Build a genuinely validated file-contract fixture for other CPU tests.

    ``write_arrays(path, exposure_index, metadata)`` may save actual NPZ data for
    numerical tests. By default payloads are opaque bytes: this validator does
    not claim image quality or parse arrays. Returned expected_original includes
    authoritative byte hashes because this self-contained fixture is not placed
    inside an actual original experiment directory.
    """
    root.mkdir(parents=True, exist_ok=True)
    names = (
        ["external_front"]
        if cameras_per_env == 1
        else ["external_front", "external_side", "wrist"]
    )
    transform = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    scene = {
        "schema_version": 1,
        "scene_id": "fixture",
        "units": "metres",
        "up_axis": "Z",
        "physics_steps": 0,
        "objects": [
            {
                "uid": "box",
                "instance_id": 1,
                "geometry": {"kind": "box", "size": [1, 1, 1]},
                "world_from_object": transform,
            }
        ],
        "cameras": [
            {"name": name, "eye": [0, 0, -3], "target": [0, 0, 0], "up": [0, -1, 0]}
            for name in names
        ],
        "camera_model": {"horizontal_fov_deg": 90, "near_m": 0.01, "far_m": 8},
    }
    frames = []
    for index in range(replay_length):
        pose = deepcopy(transform)
        pose[0][3] = index / 100
        frames.append(
            {
                "frame_id": index,
                "time_s": index / 30,
                "objects": {"box": pose},
                "cameras": {view["name"]: view for view in scene["cameras"]},
            }
        )
    replay = {
        "schema_version": 1,
        "frame_count": replay_length,
        "fps": 30,
        "frames": frames,
    }
    case = {
        "experiment_id": "R-03",
        "definition_version": "2.0",
        "scene_id": "fixture",
        "width": 4,
        "height": 4,
        "num_envs": num_envs,
        "cameras_per_env": cameras_per_env,
        "modalities": list(modalities),
        "profile": "realtime",
        "temporal_mode": "moving",
        "delivery": "device_ready",
        "diagnostic": False,
        "deformation": False,
        "sequence_frames": replay_length,
        "view_count": 1,
    }
    profile = {"name": "realtime", "version": "file-fixture"}
    config = {
        "case": case,
        "profile": profile,
        "warmup_frames": warmup,
        "measured_frames": measured,
        "scene_path": "original/scene.json",
        "replay_path": "original/replay.json",
        "scene_sha256": stable_hash(scene),
        "view_sha256": "workload-view-fixture",
        "profile_sha256": stable_hash(profile),
    }

    def packet(index: int, submission: int) -> dict[str, Any]:
        state = frames[index % replay_length]
        return {
            "frame_index": index,
            "delivery": "device_ready",
            "render_calls": 1,
            "state_publish_calls": 1,
            "exposure_count": num_envs * cameras_per_env,
            "adapter_evidence": {
                "status": "recorded",
                "render_submission_id": submission,
                "logical_state": {
                    "frame_id": state["frame_id"],
                    "time_s": state["time_s"],
                    "camera_names": names,
                    "num_envs": num_envs,
                    "modalities": list(modalities),
                    "batch_order": "environment_then_camera",
                },
            },
        }

    capture = {
        "warmup_raw": [packet(index, 5 + index) for index in range(warmup)],
        "raw": [packet(index, 5 + warmup + index) for index in range(measured)],
        "metrics": {},
        "counts": {"batch_steps": measured},
        "warmup_counts": {"batch_steps": warmup},
    }
    source_names = {
        "scripts.benchmark.rendering.formal_worker": "scripts/benchmark/rendering/formal_worker.py",
        "scripts.benchmark.rendering.capture": "scripts/benchmark/rendering/capture.py",
        "scripts.benchmark.core.measurement": "scripts/benchmark/core/measurement.py",
        "scripts.benchmark.rendering.backends.scene_embodichain": "scripts/benchmark/rendering/backends/scene_embodichain.py",
    }
    source_files, sources = {}, []
    for name in source_names.values():
        path = root / "original/source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# File-contract fixture; no GPU source executed.\n")
        source_files[name] = sha256(path)
        sources.append(
            artifact(root, path, "frozen_benchmark_source", role="source:" + name)
        )
    renderer = {
        "effective_profile": profile,
        "effective_lighting_controls": {"fixture": True},
        "postprocess_evidence": {"scope": "fixture", "settings": {}},
        "resolved_render_configuration": {"fixture": True},
    }
    software = {
        "benchmark_commit": "a" * 40,
        "benchmark_source_sha256": source_files,
        "packages": {},
        "python": sys.version,
    }
    result = {
        "schema_version": 1,
        "experiment_id": "R-03",
        "run_id": "r00_fixture",
        "run_dir": "r00_fixture",
        "case_id": "fixture-moving",
        "backend": "embodichain",
        "repeat": 0,
        "command": [sys.executable, "worker.py", "--gpu-id", "0"],
        "selected_gpu_id": 0,
        "status": "completed",
        "hardware_id": "GPU-fixture",
        "config": case,
        "config_sha256": stable_hash(case),
        "scene_sha256": config["scene_sha256"],
        "view_sha256": config["view_sha256"],
        "profile_sha256": config["profile_sha256"],
        "metrics": {},
        "counts": capture["counts"],
        "warmup_counts": capture["warmup_counts"],
        "renderer": renderer,
        "software": software,
        "resources": {"gpu": {"driver": "fixture-driver"}},
    }
    original_records = {
        "result": result,
        "worker_config": config,
        "capture": capture,
        "scene": scene,
        "replay": replay,
        "definition": {
            "schema_version": 1,
            "definition_version": "2.0",
            "experiment_id": "R-03",
        },
        "freshness": {"passed": True},
    }
    original_artifacts = []
    paths = {}
    for role, value in original_records.items():
        path = write_json(root / "original" / f"{role}.json", value)
        paths[role] = path
        original_artifacts.append(
            artifact(root, path, "frozen_original_input", role=role)
        )
    paths["raw"] = root / "original/raw.jsonl"
    import json

    paths["raw"].write_text(
        json.dumps({**result, "record_id": result["run_id"]}) + "\n"
    )
    original_artifacts.append(
        artifact(root, paths["raw"], "frozen_original_input", role="raw")
    )
    original = {
        key: result[key]
        for key in (
            "experiment_id",
            "run_id",
            "case_id",
            "backend",
            "repeat",
            "config_sha256",
            "scene_sha256",
            "view_sha256",
            "profile_sha256",
            "hardware_id",
        )
    }
    original.update(
        source_commit=software["benchmark_commit"],
        source_files=source_files,
        runtime_versions=software,
        gpu_driver="fixture-driver",
        artifacts=original_artifacts + sources,
    )
    schedule = build_history_schedule(config, replay, capture)
    policy = protocol_for_modalities(
        modalities, timed_history=True, temporal_dt_s=1 / 30
    )
    request = {
        "original": original,
        "schedule": schedule,
        "schedule_sha256": stable_hash(schedule),
        "history_quality_protocol": policy,
        "history_quality_requirements": protocol_requirements(policy["id"], "sequence"),
    }
    producer = [
        artifact(
            root,
            write_json(root / "request.json", request),
            "history_request",
            role="request",
        )
    ]
    for name in ("history_worker.py", "history_contract.py", "frozen_source.py"):
        path = root / "producer" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("# File-contract producer fixture; no runtime used.\n")
        producer.append(
            artifact(
                root, path, "history_producer_source", role="producer_source:" + name
            )
        )
    envs = sorted({0, num_envs // 2, num_envs - 1})
    frame_records = []
    for exposure in range(-1, measured):
        state_entry = (
            schedule["guard"] if exposure == -1 else schedule["measured"][exposure]
        )
        state = replay["frames"][state_entry["index"] % replay_length]
        for env in envs:
            for camera_index, name in enumerate(names):
                camera = {
                    "name": name,
                    "width": 4,
                    "height": 4,
                    "intrinsics": [[2.0, 0, 2.0], [0, 2.0, 2.0], [0, 0, 1]],
                    "world_from_camera": [
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, -3],
                        [0, 0, 0, 1],
                    ],
                }
                metadata = {
                    "backend": "embodichain",
                    "scene_sha256": original["scene_sha256"],
                    "view_sha256": stable_hash(camera),
                    "workload_view_sha256": original["view_sha256"],
                    "profile_sha256": original["profile_sha256"],
                    "frame_id": state["frame_id"],
                    "frame_index": state["frame_id"],
                    "time_s": state["time_s"],
                    "capture_index": exposure + 1,
                    "env_id": env,
                    "camera_index": camera_index,
                    "camera_name": name,
                    "batch_index": env * cameras_per_env + camera_index,
                    "camera": camera,
                    "world_from_objects": state["objects"],
                    "world_from_instances": {"1": state["objects"]["box"]},
                    "instance_ids": [1],
                    "moving_instance_ids": [1],
                    "rgb_encoding": "srgb_uint8",
                    "history": frame_history(schedule, exposure),
                }
                stem = (
                    f"capture_{exposure+1:04d}_env_{env:04d}_camera_{camera_index:02d}"
                )
                arrays = root / "capture/frames" / (stem + ".npz")
                arrays.parent.mkdir(parents=True, exist_ok=True)
                if write_arrays is None:
                    arrays.write_bytes(
                        b"opaque file-contract fixture; no image quality claimed"
                    )
                else:
                    write_arrays(arrays, exposure, metadata)
                frame_path = write_json(
                    arrays.with_suffix(".json"),
                    {
                        "schema_version": 1,
                        "metadata": metadata,
                        "arrays": {"path": arrays.name, "sha256": sha256(arrays)},
                    },
                )
                frame_records.append(
                    {
                        "path": frame_path.relative_to(root / "capture").as_posix(),
                        "sha256": sha256(frame_path),
                        "arrays_path": arrays.relative_to(root / "capture").as_posix(),
                        "arrays_sha256": sha256(arrays),
                        "frame_id": state["frame_id"],
                        "time_s": state["time_s"],
                        "env_id": env,
                        "camera_name": name,
                    }
                )
    frame_manifest = {
        "frames": frame_records,
        "representative_env_ids": envs,
        "capture_batches": measured + 1,
        "saved_exposures": len(frame_records),
    }
    manifest_path = write_json(root / "capture/frames.json", frame_manifest)
    events = {
        "freshness": [
            {"phase": "freshness", "index": index, "packet": packet(0, index + 1)}
            for index in range(4)
        ],
        "warmup": [
            {"phase": "warmup", "index": index, "packet": value}
            for index, value in enumerate(capture["warmup_raw"])
        ],
        "measured": [
            {"phase": "measured", "index": index, "packet": value}
            for index, value in enumerate(capture["raw"])
        ],
    }
    sampled = root / "capture/freshness_probe.npz"
    sampled.write_bytes(b"opaque freshness fixture; no image quality claimed")
    log = root / "capture/worker.log"
    log.write_bytes(b"file-only completed acquisition fixture\n")
    renderer_validation = {
        "schema_version": 1,
        "status": "no_known_capacity_errors",
        "checkpoint": "before_native_close",
        "log_path": "worker.log",
        "scanned_bytes": log.stat().st_size,
        "scanned_prefix_sha256": sha256(log),
    }
    resource_path = write_json(
        root / "capture/resources.json",
        {
            "schema_version": 1,
            "pid": 321,
            "gpu": {"uuid": "GPU-fixture", "driver": "fixture-driver"},
        },
    )
    validation_path = write_json(
        root / "capture/renderer_validation.json", renderer_validation
    )
    write_json(
        root / "artifact_index.json",
        {
            "schema_version": 1,
            "artifacts": [
                {
                    **artifact(root, path, "history_proof_evidence"),
                    "bytes": path.stat().st_size,
                }
                for path in (sampled, log, resource_path, validation_path)
            ],
        },
    )
    history = {
        "schema_version": 1,
        "kind": "timed_render_history/v1",
        "status": "completed",
        "performance_claim": False,
        **request,
        "producer_artifacts": producer,
        "acquisition": {
            "pid": 321,
            "fresh_process": True,
            "source_commit": software["benchmark_commit"],
            "hardware_id": "GPU-fixture",
            "gpu_driver": "fixture-driver",
            "runtime_versions": {"packages": {}, "python": sys.version},
            "imported_sources": {
                name: {"path": path, "sha256": source_files[path]}
                for name, path in source_names.items()
            },
        },
        "frames_manifest": artifact(root, manifest_path, "history_frame_manifest"),
        "renderer": artifact(
            root,
            write_json(root / "capture/renderer.json", renderer),
            "history_renderer",
        ),
        "freshness": artifact(
            root,
            write_json(
                root / "capture/freshness_probe.json",
                {
                    "passed": True,
                    "sampled_arrays": {"path": sampled.name, "sha256": sha256(sampled)},
                },
            ),
            "history_freshness",
        ),
        "capture_events": artifact(
            root,
            write_json(root / "capture/events.json", events),
            "history_capture_events",
        ),
    }
    history["acquisition_result"] = artifact(
        root,
        write_json(
            root / "capture/result.json",
            {
                "status": "completed",
                "performance_claim": False,
                "metrics": {},
                "pid": 321,
                "evidence": ["resources.json", "renderer_validation.json"],
                "renderer_validation": renderer_validation,
                "original_run_id": result["run_id"],
            },
        ),
        "history_acquisition_result",
    )
    history_path = write_json(root / "history.json", history)
    expected_original = {
        **result,
        "_history_artifact_hashes": {
            role: sha256(path) for role, path in paths.items()
        },
    }
    return {
        "root": root,
        "path": history_path,
        "manifest": history,
        "expected_original": expected_original,
        "config": config,
        "replay": replay,
        "capture": capture,
        "frames_manifest_path": manifest_path,
    }


def test_full_300_measured_history_retains_entry_and_wrap_seams(tmp_path: Path) -> None:
    fixture = build_valid_history_fixture(
        tmp_path, warmup=30, measured=300, replay_length=100
    )
    proof = validate_history_proof(
        tmp_path, fixture["path"], fixture["expected_original"]
    )
    assert proof["valid"], proof["reasons"]
    assert len(proof["frames"]) == 301
    assert proof["schedule"]["entry_seam"] == {
        "from_frame_id": 29,
        "to_frame_id": 0,
        "measured_index": 0,
    }
    assert proof["schedule"]["wrap_measured_indices"] == [100, 200]
    assert [
        proof["frames"][index]["metadata"]["frame_id"]
        for index in (0, 1, 100, 101, 200, 201, 300)
    ] == [29, 0, 99, 0, 99, 0, 99]
    assert proof["frames"][0]["metadata"]["time_s"] == 29 / 30
    assert proof["frames"][0]["metadata"]["history"]["scoring_time_s"] == 0


def test_standalone_acquisition_ledger_cannot_replace_original_binding(
    tmp_path: Path,
) -> None:
    fixture = build_valid_history_fixture(tmp_path)
    (tmp_path / "raw.jsonl").write_text('{"run_id":"capture","status":"completed"}\n')
    proof = validate_history_proof(
        tmp_path, fixture["path"], fixture["expected_original"]
    )
    assert proof["valid"], proof["reasons"]


@pytest.mark.parametrize(
    "changed",
    [
        "old100",
        "missing_guard",
        "producer",
        "runtime",
        "controls",
        "source",
        "expected_result",
        "raw_bytes",
    ],
)
def test_incomplete_or_incompatible_history_never_qualifies(
    tmp_path: Path, changed: str
) -> None:
    fixture = build_valid_history_fixture(tmp_path)
    history = fixture["manifest"]
    if changed == "old100":
        history["kind"] = "post_measurement_sequence_100"
    elif changed == "missing_guard":
        frames = read_json(fixture["frames_manifest_path"])
        frames["frames"] = frames["frames"][1:]
        write_json(fixture["frames_manifest_path"], frames)
        history["frames_manifest"]["sha256"] = sha256(fixture["frames_manifest_path"])
    elif changed == "producer":
        history["producer_artifacts"] = []
    elif changed == "runtime":
        history["acquisition"]["runtime_versions"]["python"] = "different"
    elif changed == "controls":
        path = tmp_path / history["renderer"]["path"]
        renderer = read_json(path)
        renderer["postprocess_evidence"]["settings"]["exposure"] = 99
        write_json(path, renderer)
        history["renderer"]["sha256"] = sha256(path)
    elif changed == "source":
        history["acquisition"]["imported_sources"].pop(
            "scripts.benchmark.rendering.backends.scene_embodichain"
        )
    elif changed == "expected_result":
        fixture["expected_original"]["metrics"] = {"batch_steps_per_s": 99}
    else:
        fixture["expected_original"]["_history_artifact_hashes"]["capture"] = "0" * 64
    write_json(fixture["path"], history)
    assert not validate_history_proof(
        tmp_path, fixture["path"], fixture["expected_original"]
    )["valid"]
