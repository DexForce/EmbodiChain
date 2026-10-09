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

"""CPU evaluation of immutable rendering captures against independent references.

Reference generation belongs to the independent Mitsuba implementation. Tested
backend pixels never become reference inputs. Quality events preserve the
original execution snapshot and raw ledger; only the derived quality view of
``runs.json`` changes. Run in the reference Python environment with the existing
``TORCH_HOME`` cache; no model weights are downloaded by the quality owner.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scripts.benchmark.core.artifacts import (
    ArtifactStore,
    JsonlLedger,
    read_json,
    read_json_object,
    stable_hash,
    write_json,
)
from scripts.benchmark.rendering import reference
from scripts.benchmark.rendering.formal import RenderCase, view_fingerprint
from scripts.benchmark.rendering.quality_protocol import (
    protocol_for_modalities,
    protocol_requirements,
    validate_protocol,
)
from scripts.benchmark.rendering.replay import load_replay
from scripts.benchmark.rendering.scenes import load_scene, scene_fingerprint
from scripts.benchmark.rendering.scene_cases import fixed_views

__all__ = ["evaluate_experiment", "main"]

_BINDINGS = ("config_sha256", "scene_sha256", "view_sha256", "profile_sha256")
_QUALITY_FIELDS = {
    "quality_status",
    "quality_evaluation_id",
    "quality_evidence",
    "quality",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(root: Path, value: str | Path) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Evaluation evidence must remain inside the experiment root")
    return path


def _verify(root: Path, record: dict[str, Any]) -> Path:
    path = _inside(root, record["path"])
    if not path.is_file() or _sha256(path) != record.get("sha256"):
        raise ValueError(f"Evidence checksum mismatch: {path}")
    return path


def _artifact(root: Path, path: Path, kind: str) -> dict[str, Any]:
    path = _inside(root, path)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "kind": kind,
    }


def _register(store: ArtifactStore, path: Path, kind: str) -> None:
    """Reuse an existing evidence classification while enforcing byte identity."""
    index_path = store.root / "artifact_index.json"
    relative = path.relative_to(store.root).as_posix()
    if index_path.exists():
        for record in read_json_object(index_path).get("artifacts", []):
            if record["path"] == relative:
                _verify(store.root, record)
                return
    store.register_artifact(path, kind=kind)


def _immutable_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in _QUALITY_FIELDS and key != "record_id"
    }


def _snapshot(
    root: Path, rows: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_path = root / "raw.jsonl"
    raw = [
        json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()
    ]
    raw_by_id = {row.get("run_id", row.get("record_id")): row for row in raw}
    for row in rows:
        original = raw_by_id.get(row["run_id"])
        if original is None or _immutable_row(original) != _immutable_row(row):
            raise ValueError(
                f"Run snapshot differs from immutable raw execution: {row['run_id']}"
            )
    directory = root / "quality_evaluations"
    directory.mkdir(exist_ok=True)
    baseline_path = directory / "baseline.json"
    if baseline_path.exists():
        baseline = read_json_object(baseline_path)
        for item in baseline["inputs"]:
            _verify(root, item)
        original = read_json(_verify(root, baseline["runs_snapshot"]))
        if {_row["run_id"]: _immutable_row(_row) for _row in original} != {
            row["run_id"]: _immutable_row(row) for row in rows
        }:
            raise ValueError(
                "Derived run timing/configuration changed since the original snapshot"
            )
    else:
        snapshot_path = directory / "original_runs.json"
        if snapshot_path.exists():
            raise ValueError("Unbound existing original run snapshot")
        snapshot_path.write_bytes((root / "runs.json").read_bytes())
        baseline = {
            "schema_version": 1,
            "runs_snapshot": _artifact(
                root, snapshot_path, "original_execution_snapshot"
            ),
            "inputs": [
                _artifact(root, root / name, "frozen_execution_input")
                for name in (
                    "raw.jsonl",
                    "definition.json",
                    "manifest.json",
                    "assets_manifest.json",
                    "selection.json",
                )
                if (root / name).is_file()
            ],
        }
        write_json(baseline_path, baseline)
        ArtifactStore(root).register_artifact(
            snapshot_path, kind="original_execution_snapshot"
        )
        ArtifactStore(root).register_artifact(baseline_path, kind="quality_baseline")
        original = deepcopy(rows)
    # Every prior evaluation stays verifiable, even when a newer event supersedes it.
    ledger = root / "quality_events.jsonl"
    previous_records = {}
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            for evidence in event["artifacts"]:
                _verify(root, evidence)
            frozen_event = read_json_object(_inside(root, event["event"]))
            previous_records.update(
                {record["run_id"]: record for record in frozen_event["quality"]["runs"]}
            )
    for record in read_json_object(root / "quality.json").get("runs", []):
        if (
            record.get("evaluation_id")
            and previous_records.get(record["run_id"]) != record
        ):
            raise ValueError(
                "Derived quality record differs from its immutable evaluation event"
            )
    return baseline, original


def _load_run(root: Path, row: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    run_dir = _inside(root, row.get("run_dir", row["run_id"]))
    result = read_json_object(run_dir / "result.json")
    for name in (
        *_BINDINGS,
        "backend",
        "case_id",
        "config",
        "metrics",
        "status",
        "renderer",
    ):
        if result.get(name) != row.get(name):
            raise ValueError(f"Worker snapshot mismatch for {row['run_id']}: {name}")
    config = read_json_object(run_dir / "worker_config.json")
    config_path = run_dir / "worker_config.json"
    frozen_config_path = _inside(
        root, Path("cases") / row["case_id"] / "worker-config.json"
    )
    if frozen_config_path.exists() and read_json_object(frozen_config_path) != config:
        raise ValueError("Worker configuration differs from the frozen launcher input")
    indexed = read_json_object(root / "artifact_index.json").get("artifacts", [])
    for path in (run_dir / "result.json", config_path, frozen_config_path):
        for evidence in indexed:
            if evidence.get("path") == path.relative_to(root).as_posix():
                _verify(root, evidence)
    case_data = dict(config["case"])
    case_data["modalities"] = tuple(case_data["modalities"])
    case = RenderCase(**case_data)
    if (
        case.case_id != row["case_id"]
        or case.to_dict() != row["config"]
        or stable_hash(case.to_dict()) != row["config_sha256"]
    ):
        raise ValueError("Frozen worker case/config hash mismatch")
    for name in _BINDINGS[1:]:
        if config[name] != row[name]:
            raise ValueError(f"Worker config binding mismatch: {name}")
    scene_path = _inside(root, config["scene_path"])
    scene = load_scene(scene_path)
    if (
        scene_fingerprint(scene) != row["scene_sha256"]
        or stable_hash(config["profile"]) != row["profile_sha256"]
    ):
        raise ValueError("Scene or profile no longer matches the run")
    replay = (
        load_replay(_inside(root, config["replay_path"]), scene_path)
        if config.get("replay_path")
        else None
    )
    if view_fingerprint(case, scene, replay) != row["view_sha256"]:
        raise ValueError("Frozen views or replay changed")
    manifest_path = _verify(
        root,
        {
            "path": run_dir / row["frames_manifest"]["path"],
            "sha256": row["frames_manifest"]["sha256"],
        },
    )
    manifest = read_json_object(manifest_path)
    for name in _BINDINGS[1:]:
        if manifest.get(name) != row[name]:
            raise ValueError(f"Frame manifest binding mismatch: {name}")
    descriptors = []
    seen = set()
    for entry in manifest["frames"]:
        metadata_path = _verify(
            root, {"path": run_dir / entry["path"], "sha256": entry["sha256"]}
        )
        record = read_json_object(metadata_path)
        arrays_path = _verify(
            root,
            {
                "path": metadata_path.parent / record["arrays"]["path"],
                "sha256": record["arrays"]["sha256"],
            },
        )
        if (
            arrays_path != _inside(root, run_dir / entry["arrays_path"])
            or record["arrays"]["sha256"] != entry["arrays_sha256"]
        ):
            raise ValueError("Frame array index disagrees with frame metadata")
        metadata = record["metadata"]
        for field, expected in (
            ("backend", row["backend"]),
            ("scene_sha256", row["scene_sha256"]),
            ("profile_sha256", row["profile_sha256"]),
            ("workload_view_sha256", row["view_sha256"]),
        ):
            if metadata.get(field) != expected:
                raise ValueError(f"Captured frame binding mismatch: {field}")
        for field in (
            "frame_id",
            "time_s",
            "camera_name",
            "camera_index",
            "env_id",
            "batch_index",
        ):
            if entry.get(field) != metadata.get(field):
                raise ValueError(f"Frame manifest and metadata disagree: {field}")
        identity = (metadata["env_id"], metadata["camera_name"], metadata["frame_id"])
        if identity in seen:
            raise ValueError("Duplicate captured frame identity")
        seen.add(identity)
        if metadata["view_sha256"] != stable_hash(metadata["camera"]):
            raise ValueError("Captured camera hash mismatch")
        if set(metadata["world_from_objects"]) != {
            obj["uid"] for obj in scene["objects"]
        }:
            raise ValueError("Captured object identity differs from the frozen scene")
        if replay is not None:
            frame_id = metadata["frame_id"]
            if type(frame_id) is not int or not 0 <= frame_id < replay["frame_count"]:
                raise ValueError("Captured frame is outside the frozen replay")
            state = replay["frames"][frame_id]
        else:
            state = {
                "objects": {
                    obj["uid"]: obj["world_from_object"] for obj in scene["objects"]
                },
                "cameras": {view["name"]: view for view in scene["cameras"]},
                "time_s": 0.0,
            }
        if (
            metadata["world_from_objects"] != state["objects"]
            or metadata["time_s"] != state["time_s"]
        ):
            raise ValueError(
                "Captured object state/timestamp differs from frozen source"
            )
        view = (
            fixed_views(scene)[metadata["view_id"]]
            if case.experiment_id == "R-01"
            else state["cameras"][metadata["camera_name"]]
        )
        canonical_camera = reference.camera_metadata(
            scene, view, case.width, case.height
        )
        if any(
            not np.allclose(
                metadata["camera"][key], canonical_camera[key], atol=1e-8, rtol=0
            )
            for key in ("intrinsics", "world_from_camera")
        ):
            raise ValueError("Captured view differs from frozen source")
        if (
            metadata["camera_name"] != view["name"]
            or metadata["batch_index"]
            != metadata["env_id"] * case.cameras_per_env + metadata["camera_index"]
        ):
            raise ValueError("Captured camera/environment batch identity mismatch")
        if metadata["instance_ids"] != [
            obj["instance_id"] for obj in scene["objects"]
        ] or metadata["world_from_instances"] != {
            str(obj["instance_id"]): state["objects"][obj["uid"]]
            for obj in scene["objects"]
        }:
            raise ValueError(
                "Captured canonical instance state differs from frozen source"
            )
        descriptors.append(
            {
                "metadata": metadata,
                "arrays_path": arrays_path,
                "metadata_path": metadata_path,
                "canonical_camera": canonical_camera,
            }
        )
    envs = sorted({0, case.num_envs // 2, case.num_envs - 1})
    count = (
        20
        if case.experiment_id == "R-01"
        else 100 if case.experiment_id == "R-02" else 1
    )
    expected = len(envs) * case.cameras_per_env * count
    actual = len(descriptors)
    complete = (
        actual == expected
        and manifest["saved_exposures"] == actual
        and manifest["representative_env_ids"] == envs
    )
    if case.experiment_id == "R-01":
        complete = complete and all(
            {
                item["metadata"].get("view_id")
                for item in descriptors
                if item["metadata"]["env_id"] == env
            }
            == set(range(20))
            for env in envs
        )
    else:
        names = (
            ("external_front",)
            if case.cameras_per_env == 1
            else ("external_front", "external_side", "wrist")
        )
        for env in envs:
            for name in names:
                members = [
                    item
                    for item in descriptors
                    if item["metadata"]["env_id"] == env
                    and item["metadata"]["camera_name"] == name
                ]
                complete = complete and len(members) == count
    return {
        "row": row,
        "renderer": deepcopy(result.get("renderer", {})),
        "case": case,
        "config": config,
        "scene_path": scene_path,
        "descriptors": descriptors,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "input_artifacts": [
            path
            for path in (run_dir / "result.json", config_path, frozen_config_path)
            if path.is_file()
        ],
        "coverage": {
            "expected_samples": expected,
            "evaluated_samples": actual,
            "representative_env_ids": envs,
            "camera_names": sorted(
                {item["metadata"]["camera_name"] for item in descriptors}
            ),
            "view_count": case.view_count,
            "sequence_frames": count if case.experiment_id == "R-02" else None,
            "sequence_groups": 0,
            "complete": bool(complete),
        },
    }


def _candidate(descriptor: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    with np.load(descriptor["arrays_path"], allow_pickle=False) as arrays:
        values = {name: arrays[name].copy() for name in arrays.files}
    return {**values, "metadata": deepcopy(descriptor["metadata"])}


def _reference_spec(metadata: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    pose = np.asarray(metadata["camera"]["world_from_camera"], dtype=float)
    if (
        pose.shape != (4, 4)
        or not np.isfinite(pose).all()
        or not np.allclose(pose[3], [0, 0, 0, 1])
        or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-5)
    ):
        raise ValueError("Captured camera must be a finite rigid OpenCV transform")
    eye = pose[:3, 3]
    return {
        "name": "canonical_view",
        "eye": eye.tolist(),
        "target": (eye + pose[:3, 2]).tolist(),
        "up": (-pose[:3, 1]).tolist(),
        "frame_id": metadata["frame_id"],
        "frame_index": metadata["frame_id"],
        "time_s": metadata["time_s"],
        "object_transforms": metadata["world_from_objects"],
    }


def _independent_reference(
    root: Path,
    run: dict[str, Any],
    descriptor: dict[str, Any],
    settings: dict[str, int],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    metadata, case = descriptor["metadata"], run["case"]
    spec = _reference_spec(metadata)
    identity = {
        "schema_version": 1,
        "scene_sha256": metadata["scene_sha256"],
        "camera": {
            key: value
            for key, value in descriptor["canonical_camera"].items()
            if key != "name"
        },
        "object_transforms": spec["object_transforms"],
        "width": case.width,
        "height": case.height,
        "settings": settings,
        "reference_implementation_sha256": _sha256(Path(reference.__file__)),
    }
    key = stable_hash(identity)
    directory = root / "references" / key
    descriptor_path = directory / "cache_binding.json"
    if directory.exists():
        if not descriptor_path.is_file():
            raise ValueError(
                "Incomplete immutable reference directory; refusing replacement"
            )
        binding = read_json_object(descriptor_path)
        if binding.get("identity") != identity:
            raise ValueError("Reference cache identity mismatch")
        for artifact in binding["artifacts"]:
            _verify(root, artifact)
    else:
        directory.parent.mkdir(exist_ok=True)
        try:
            reference.render_reference(
                run["scene_path"],
                [spec],
                case.width,
                case.height,
                directory,
                **settings,
            )
        except (ImportError, RuntimeError) as exc:
            return None, {
                "cache_key": key,
                "status": "missing",
                "reason": f"independent_reference_unavailable: {type(exc).__name__}: {exc}",
            }
        manifest_path = directory / "reference.json"
        manifest = read_json_object(manifest_path)
        frame_path = directory / manifest["frames"][0]["path"]
        frame = read_json_object(frame_path)
        binding = {
            "identity": identity,
            "frame": frame_path.relative_to(root).as_posix(),
            "artifacts": [
                _artifact(root, path, "independent_reference")
                for path in (
                    manifest_path,
                    frame_path,
                    frame_path.parent / frame["arrays"]["path"],
                )
            ],
        }
        write_json(descriptor_path, binding)
        for artifact in binding["artifacts"]:
            _register(ArtifactStore(root), root / artifact["path"], artifact["kind"])
        _register(ArtifactStore(root), descriptor_path, "reference_cache_binding")
    loaded = reference.load_reference_frame(_inside(root, binding["frame"]))
    # The immutable render is keyed by camera and object geometry, not temporal
    # labels. Identical held/static states can reuse pixels at another timestamp.
    # Preserve the binding explicitly; never modify the reference files.
    alias = {
        "cache_key": key,
        "status": "completed",
        "reference_id": loaded["reference_provenance"]["reference_id"],
        "reference_frame": binding["frame"],
        "source_frame_id": loaded["metadata"]["frame_id"],
        "frame_id": metadata["frame_id"],
        "time_s": metadata["time_s"],
        "geometry_identity_sha256": key,
        "binding_artifact": _artifact(root, descriptor_path, "reference_cache_binding"),
    }
    loaded["metadata"]["frame_id"] = loaded["metadata"]["frame_index"] = metadata[
        "frame_id"
    ]
    loaded["metadata"]["time_s"] = metadata["time_s"]
    loaded["artifacts"].append(
        {**alias["binding_artifact"], "path": str(descriptor_path)}
    )
    return loaded, alias


def _calibration(
    root: Path,
    document: dict[str, Any] | None,
    row: dict[str, Any],
    candidate: dict[str, Any],
    ref: dict[str, Any] | None,
    protocol: dict[str, Any],
) -> None:
    if document is None or ref is None:
        return
    reference_id = ref["reference_provenance"]["reference_id"]
    matches = [
        record
        for record in document["records"]
        if record.get("reference_id") == reference_id
        and record.get("backend") == row["backend"]
        and record.get("case_id") == row["case_id"]
    ]
    if not matches:
        return
    if len(matches) != 1:
        raise ValueError("Ambiguous photometric calibration record")
    record = matches[0]
    if any(record.get(key) != row[key] for key in _BINDINGS):
        raise ValueError("Photometric calibration is not bound to this frozen case")
    calibration = deepcopy(record["photometric_calibration"])
    if (
        calibration.get("status") != "qualified"
        or calibration.get("reference_id") != reference_id
        or calibration.get("profile_sha256") != row["profile_sha256"]
        or not calibration.get("evidence")
    ):
        raise ValueError("Calibration record lacks measured, case-bound qualification")
    if (
        len(calibration["evidence"]) != 1
        or calibration["evidence"][0].get("kind") != "photometric_calibration_receipt"
    ):
        raise ValueError(
            "Calibration requires exactly one measured photometric_calibration_receipt"
        )
    for evidence in calibration["evidence"]:
        evidence["path"] = str(_verify(root, evidence))
    from scripts.benchmark.rendering.calibration import verify_calibration

    check = verify_calibration(calibration, candidate["metadata"], ref, protocol)
    if check.get("status") != "passed":
        raise ValueError(
            "Measured calibration proof did not pass: "
            + str(check.get("reason") or check.get("status", "missing verification"))
        )
    sources = check.get("evidence", {}).get("source_evidence")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Measured calibration proof has no verified source evidence")
    receipt_root = Path(calibration["evidence"][0]["path"]).parent
    retained = {item["path"]: item for item in calibration["evidence"]}
    for source in sources:
        evidence = deepcopy(source)
        evidence["path"] = str(receipt_root / evidence["path"])
        evidence["path"] = str(_verify(root, evidence))
        retained[evidence["path"]] = evidence
    calibration["evidence"] = list(retained.values())
    candidate["metadata"]["photometric_calibration"] = calibration


def _all_check(values: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    failed = any(value.get("status") == "failed" for value in values)
    missing = len(values) != expected or any(
        value.get("status") not in {"passed", "failed"} for value in values
    )
    return {
        "status": "failed" if failed else "missing" if missing else "passed",
        "aggregation": "all_required_samples",
        "denominator": expected,
        "observed": len(values),
        "reasons": sorted(
            {str(value.get("reason")) for value in values if value.get("reason")}
        ),
    }


def _aggregate(
    reports: list[dict[str, Any]],
    requirements: dict[str, Any],
    protocol: dict[str, Any],
    expected: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checks = {
        name: _all_check(
            [report["checks"][name] for report in reports if name in report["checks"]],
            expected,
        )
        for name in requirements["required_checks"]
    }
    metrics = {}
    rules = (
        protocol["temporal_thresholds"]
        if requirements["evaluation_kind"] == "sequence"
        else protocol["frame_thresholds"]
    )
    for name in requirements["required_metrics"]:
        comparison, threshold, unit = rules[name]
        members = [
            report["metrics"][name] for report in reports if name in report["metrics"]
        ]
        finite = [
            item["value"]
            for item in members
            if type(item.get("value")) in (int, float) and math.isfinite(item["value"])
        ]
        value = (min(finite) if comparison == "ge" else max(finite)) if finite else None
        status = _all_check(members, expected)
        if len(finite) != expected and status["status"] == "passed":
            status["status"] = "missing"
        if value is not None and not (
            value >= threshold if comparison == "ge" else value <= threshold
        ):
            status["status"] = "failed"
        metrics[name] = {
            **status,
            "required": True,
            "value": value,
            "comparison": comparison,
            "threshold": threshold,
            "unit": unit,
            "aggregation": (
                "minimum_over_required_samples"
                if comparison == "ge"
                else "maximum_over_required_samples"
            ),
            "sample_denominator": expected,
            "valid_samples": len(finite),
            "missing_samples": expected - len(finite),
            "denominator": sum(item.get("denominator", 0) for item in members),
        }
    for name, (comparison, threshold, unit) in rules.items():
        if name not in metrics:
            metrics[name] = {
                "value": None,
                "status": "not_requested",
                "required": False,
                "reason": "modality_not_requested",
                "comparison": comparison,
                "threshold": threshold,
                "unit": unit,
                "denominator": 0,
            }
    return checks, metrics


def _relative_artifacts(root: Path, value: Any) -> Any:
    if isinstance(value, list):
        return [_relative_artifacts(root, item) for item in value]
    if isinstance(value, dict):
        converted = {
            key: _relative_artifacts(root, item) for key, item in value.items()
        }
        if (
            isinstance(value.get("path"), str)
            and value.get("sha256")
            and Path(value["path"]).is_absolute()
        ):
            converted["path"] = (
                _inside(root, value["path"]).relative_to(root).as_posix()
            )
        return converted
    return value


def _evaluate_run(
    root: Path,
    directory: Path,
    run: dict[str, Any],
    settings: dict[str, int],
    calibration: dict[str, Any] | None,
) -> dict[str, Any]:
    from scripts.benchmark.rendering.quality import evaluate_frame, evaluate_sequence

    row, case = run["row"], run["case"]
    temporal = case.temporal_mode == "moving" and set(case.modalities) == {
        "rgb",
        "depth",
        "seg",
    }
    policy = protocol_for_modalities(
        case.modalities, temporal_dt_s=1 / 30 if temporal else None
    )
    contract = protocol_requirements(policy["id"], "sequence" if temporal else "frame")
    declared_policy = run["config"].get("quality_protocol")
    if declared_policy is not None and validate_protocol(declared_policy) != policy:
        raise ValueError("Evaluation policy differs from the frozen worker policy")
    declared = run["config"].get("quality_requirements", {})
    if declared and any(declared.get(key) != value for key, value in contract.items()):
        raise ValueError(
            "Evaluation requirements differ from the frozen worker contract"
        )
    if temporal and case.experiment_id != "R-02":
        run["coverage"]["capture_expected_samples"] = run["coverage"][
            "expected_samples"
        ]
        run["coverage"]["expected_samples"] = (
            len(run["coverage"]["representative_env_ids"]) * case.cameras_per_env * 100
        )
        run["coverage"]["sequence_frames"] = 100
        run["coverage"]["complete"] = False
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "protocol.json", policy)
    reports, bindings = [], []
    reference_ids = set()
    reference_artifacts = {}

    def inputs(item: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        candidate = _candidate(item)
        for key in ("effective_lighting_controls", "postprocess_evidence"):
            candidate["metadata"][key] = deepcopy(run["renderer"].get(key))
        ref, binding = _independent_reference(root, run, item, settings)
        bindings.append(binding)
        if ref is not None:
            reference_ids.add(ref["reference_provenance"]["reference_id"])
            for artifact in ref["artifacts"]:
                path = _verify(root, artifact)
                reference_artifacts[str(path)] = _artifact(
                    root, path, artifact.get("kind", "reference_evidence")
                )
        _calibration(root, calibration, row, candidate, ref, policy)
        return candidate, ref

    if temporal:
        groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for item in run["descriptors"]:
            groups[
                (item["metadata"]["env_id"], item["metadata"]["camera_name"])
            ].append(item)
        for index, (identity, items) in enumerate(sorted(groups.items())):
            ordered = sorted(items, key=lambda item: item["metadata"]["frame_id"])
            pairs = [inputs(item) for item in ordered]
            candidates, refs = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
            report = evaluate_sequence(
                candidates,
                refs if all(ref is not None for ref in refs) else None,
                policy,
                output_dir=directory / f"sequence_{index:03d}",
            )
            report["env_id"], report["camera_name"] = identity
            reports.append(report)
        run["coverage"]["sequence_groups"] = len(groups)
        expected = len(run["coverage"]["representative_env_ids"]) * case.cameras_per_env
    else:
        for index, item in enumerate(run["descriptors"]):
            candidate, ref = inputs(item)
            reports.append(
                evaluate_frame(
                    candidate, ref, policy, output_dir=directory / f"sample_{index:04d}"
                )
            )
        expected = run["coverage"]["expected_samples"]
    checks, metrics = _aggregate(reports, contract, policy, expected)
    checks["sample_population"] = {
        "status": "passed" if run["coverage"]["complete"] else "missing",
        "reason": (
            None
            if run["coverage"]["complete"]
            else "Required representative camera/view/frame population is incomplete"
        ),
        "evidence": deepcopy(run["coverage"]),
    }
    if case.temporal_mode == "moving" and case.experiment_id != "R-02":
        checks["temporal_evidence"] = {
            "status": "missing",
            "reason": "Only final-frame performance samples exist; no linked matching hundred-frame temporal proof",
        }
    if contract["unassessed_modalities"]:
        checks["all_requested_modalities_assessed"] = {
            "status": "missing",
            "reason": "No quality policy assesses: "
            + ", ".join(contract["unassessed_modalities"]),
        }
    statuses = {**checks, **metrics}
    failed = [name for name, item in statuses.items() if item["status"] == "failed"]
    missing = [name for name, item in statuses.items() if item["status"] == "missing"]
    qualified = bool(reports) and not failed and not missing
    write_json(directory / "reference_bindings.json", bindings)
    artifacts = {
        str(run["manifest_path"]): _artifact(
            root, run["manifest_path"], "candidate_frame_manifest"
        ),
        **reference_artifacts,
    }
    for path in run["input_artifacts"]:
        artifacts[str(path)] = _artifact(root, path, "frozen_worker_input")
    for item in run["descriptors"]:
        for key in ("arrays_path", "metadata_path"):
            artifacts[str(item[key])] = _artifact(root, item[key], "candidate_capture")
    for path in directory.rglob("*"):
        if path.is_file():
            artifacts[str(path)] = _artifact(root, path, "quality_detail")
    for report in reports:
        for artifact in report.get("artifacts", []):
            path = _verify(root, artifact)
            artifacts[str(path)] = _artifact(
                root, path, artifact.get("kind", "quality_detail")
            )
    result = {
        "schema_version": 1,
        **{key: row[key] for key in ("run_id", "backend", "case_id", *_BINDINGS)},
        **contract,
        "quality_status": "qualified" if qualified else "not_qualified",
        "qualification": {
            "passed": qualified,
            "failed_checks": failed,
            "missing_checks": missing,
        },
        "checks": checks,
        "metrics": metrics,
        "coverage": run["coverage"],
        "reference_ids": sorted(reference_ids),
        "reference_id": (
            next(iter(reference_ids))
            if len(reference_ids) == 1
            else stable_hash(sorted(reference_ids)) if reference_ids else None
        ),
        "artifacts": list(artifacts.values()),
        "aggregation": "every required sample must pass; ge=min, le=max; no cross-sample means",
        "sequences" if temporal else "samples": _relative_artifacts(root, reports),
    }
    proof_path = write_json(directory / "aggregate.json", result)
    result["artifacts"].append(_artifact(root, proof_path, "aggregate_quality_proof"))
    return result


def evaluate_experiment(
    root: Path,
    *,
    spp: int = 256,
    threads: int = 4,
    seed: int = 0,
    run_ids: list[str] | None = None,
    calibration_evidence: Path | None = None,
) -> Path:
    """Evaluate selected completed runs, append evidence, and rebuild the report.

    Args:
        root: One formal experiment containing immutable raw execution records.
        spp: Independent reference samples per pixel, default 256.
        threads: CPU reference worker count, default four.
        seed: Frozen independent-reference seed, default zero.
        run_ids: Optional run-ID selection for staged offline evaluation.
        calibration_evidence: Explicit case/backend/reference-bound calibration
            JSON inside the experiment root, referencing exactly one measured
            ``photometric_calibration_receipt``. Its heldout sources and observed
            renderer controls are verified before attachment; a status flag and
            file hash alone cannot qualify calibration.

    Returns:
        Path of the rebuilt Markdown report. Timing fields and raw.jsonl bytes
        remain unchanged; every evaluation event and original snapshot persist.
    """
    root = Path(root).resolve()
    for name, value, minimum in (
        ("spp", spp, 1),
        ("threads", threads, 1),
        ("seed", seed, 0),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    rows = read_json(root / "runs.json")
    if (
        not isinstance(rows, list)
        or any(not isinstance(row, dict) or not row.get("run_id") for row in rows)
        or len({row["run_id"] for row in rows}) != len(rows)
    ):
        raise ValueError("Experiment requires unique run snapshots")
    selected = (
        set(run_ids)
        if run_ids is not None
        else {row["run_id"] for row in rows if row["status"] == "completed"}
    )
    if selected - {row["run_id"] for row in rows}:
        raise ValueError("Unknown requested run_id")
    baseline, _ = _snapshot(root, rows)
    raw_hash = _sha256(root / "raw.jsonl")
    quality_path = root / "quality.json"
    previous = read_json_object(quality_path)
    for record in previous.get("runs", []):
        for artifact in record.get("artifacts", []):
            _verify(root, artifact)
    runs = [
        _load_run(root, row)
        for row in rows
        if row["run_id"] in selected and row["status"] == "completed"
    ]
    calibration = None
    calibration_record = None
    if calibration_evidence is not None:
        path = _inside(root, calibration_evidence)
        calibration = read_json_object(path)
        if not isinstance(calibration.get("records"), list):
            raise ValueError("Calibration evidence requires a records list")
        for record in calibration["records"]:
            for evidence in record.get("photometric_calibration", {}).get(
                "evidence", []
            ):
                _verify(root, evidence)
        calibration_record = _artifact(root, path, "explicit_calibration_input")
    directory = root / "quality_evaluations"
    version = (
        max(
            (
                int(path.name.split("_")[-1])
                for path in directory.glob("evaluation_*")
                if path.is_dir()
            ),
            default=0,
        )
        + 1
    )
    event_dir = directory / f"evaluation_{version:04d}"
    event_dir.mkdir(exist_ok=False)
    settings = {"spp": spp, "threads": threads, "seed": seed}
    evaluation_id = f"evaluation_{version:04d}"
    write_json(
        event_dir / "inputs.json",
        {
            "schema_version": 1,
            "evaluation_id": evaluation_id,
            "baseline": baseline,
            "settings": settings,
            "run_ids": sorted(selected),
            "calibration_evidence": calibration_record,
            "source_sha256": {
                name: _sha256(Path(__file__).with_name(name))
                for name in (
                    "evaluate.py",
                    "quality.py",
                    "quality_protocol.py",
                    "reference.py",
                    "calibration.py",
                )
            },
        },
    )
    records = {record["run_id"]: record for record in previous.get("runs", [])}
    evaluated = {}
    for index, run in enumerate(runs):
        record = _evaluate_run(
            root, event_dir / f"run_{index:04d}", run, settings, calibration
        )
        if calibration_record is not None:
            record["artifacts"].append(calibration_record)
        record["evaluation_id"] = evaluation_id
        records[record["run_id"]] = evaluated[record["run_id"]] = record
    # Freeze all new evidence before changing the derived report view.
    event = {
        "schema_version": 1,
        "evaluation_id": evaluation_id,
        "run_ids": sorted(evaluated),
        "skipped_run_ids": sorted(selected - set(evaluated)),
        "quality": {"schema_version": 1, "runs": list(evaluated.values())},
        "raw_sha256": raw_hash,
    }
    event_path = write_json(event_dir / "event.json", event)
    event_artifacts = {
        item["path"]: item
        for record in evaluated.values()
        for item in record["artifacts"]
    }
    for path in event_dir.rglob("*"):
        if path.is_file():
            event_artifacts[str(path.relative_to(root))] = _artifact(
                root, path, "quality_evaluation_event"
            )
    store = ArtifactStore(root)
    for artifact in event_artifacts.values():
        _verify(root, artifact)
        _register(store, root / artifact["path"], artifact["kind"])
    for row in rows:
        if row["run_id"] in evaluated:
            row["quality_status"] = evaluated[row["run_id"]]["quality_status"]
            row["quality_evaluation_id"] = evaluation_id
            row["quality_evidence"] = event_path.relative_to(root).as_posix()
    if _sha256(root / "raw.jsonl") != raw_hash:
        raise ValueError("Raw execution ledger changed during evaluation")
    store.write_quality({"schema_version": 1, "runs": list(records.values())})
    write_json(root / "runs.json", rows)
    inputs = [
        _artifact(root, root / name, "report_input")
        for name in (
            "runs.json",
            "quality.json",
            "definition.json",
            "manifest.json",
            "raw.jsonl",
        )
        if (root / name).is_file()
    ]
    frozen_inputs = write_json(
        event_dir / "report_inputs.json",
        {"schema_version": 1, "evaluation_id": evaluation_id, "inputs": inputs},
    )
    store.register_artifact(frozen_inputs, kind="frozen_report_inputs")
    event_artifacts[str(frozen_inputs.relative_to(root))] = _artifact(
        root, frozen_inputs, "frozen_report_inputs"
    )
    JsonlLedger(root / "quality_events.jsonl", identity_key="evaluation_id").append(
        {
            "evaluation_id": evaluation_id,
            "event": event_path.relative_to(root).as_posix(),
            "artifacts": list(event_artifacts.values()),
        }
    )
    from scripts.benchmark.rendering.publication_report import rebuild_report

    return rebuild_report(root)


def main(argv: list[str] | None = None) -> int:
    """Run the offline CPU evaluation CLI.

    Args:
        argv: Optional parsed-process argument override.

    Returns:
        Zero after preserving evaluation evidence and rebuilding the report.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--spp", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", action="append", dest="run_ids")
    parser.add_argument("--calibration-evidence", type=Path)
    args = parser.parse_args(argv)
    print(
        evaluate_experiment(
            args.experiment_root,
            spp=args.spp,
            threads=args.threads,
            seed=args.seed,
            run_ids=args.run_ids,
            calibration_evidence=args.calibration_evidence,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
