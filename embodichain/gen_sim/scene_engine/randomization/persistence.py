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

"""Persist grouped layouts in distinct runs with a final completion manifest."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import yaml

from .baseline import (
    PreparedSceneBaseline,
    _data_digest,
    _read_json,
    _read_pose,
    _rigid_entries,
    _source_digest,
)
from .cfg import _BATCH_FIELDS, _UniqueKeyLoader, _validate_policy

if TYPE_CHECKING:
    from .generation import SceneRandomizationResult

__all__ = ["save_layouts"]


def save_layouts(
    result: SceneRandomizationResult,
    *,
    output_root: str | Path,
    layout_format: Literal["json", "packed"] = "json",
) -> Path:
    """Save successful grouped layouts and generation accounting in a new run.

    Layouts reference the existing baseline and source export through relative
    paths. Group centers retain their numeric precision; meshes and full object
    poses are not duplicated. ``run_result.json`` is published last, including
    for runs with no successful candidates. Its presence marks a complete run.
    ``policy_sha256`` hashes canonical sorted JSON of the resolved policy
    independently of YAML formatting. Packed storage writes all candidates in
    one atomic, fsynced JSON container and publishes a version-2 manifest last.

    Args:
        result: Generated candidates and their actual resolved parameter snapshot.
        output_root: Workspace receiving ``scene_layouts/run_NNN/``.
        layout_format: ``json`` retains individual candidate files; ``packed``
            amortizes file creation and durability costs over the complete run.

    Returns:
        Absolute directory of the newly saved run.

    Raises:
        ValueError: Result accounting, candidate provenance or geometry is
            inconsistent, or its source/baseline no longer matches.
        OSError: A source cannot be read or an output cannot be written. A failed
            write may leave an incomplete run without a completion manifest.
    """
    if not isinstance(output_root, (str, Path)) or not str(output_root).strip():
        raise ValueError("output_root must be a nonempty directory path.")
    if layout_format not in ("json", "packed"):
        raise ValueError("layout_format must be json or packed.")
    try:
        policy, candidates = _prepare_result(result)
        export_root = _verify_baseline(result.baseline)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid scene randomization result.") from exc
    policy_hash = hashlib.sha256(
        json.dumps(
            policy, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    runs_root = Path(output_root).expanduser().resolve() / "scene_layouts"
    runs_root.mkdir(parents=True, exist_ok=True)
    ordinal = 1
    while True:
        run = runs_root / f"run_{ordinal:03d}"
        try:
            run.mkdir()
            break
        except FileExistsError:
            ordinal += 1
    (run / "layouts").mkdir()
    _write_atomic(run / "policy_resolved.yaml", policy)
    hierarchical = policy["schema_version"] == 3
    common = {
        "schema_version": 3 if hierarchical else 1,
        "seed": policy["seed"],
        "algorithm_version": result.algorithm_version,
        "policy_sha256": policy_hash,
    }
    layouts = []
    packed = []
    references = _references(result.baseline, export_root, run / "layouts")
    if hierarchical:
        references["support_geometry"] = {
            "path": Path(
                os.path.relpath(result.candidates._geometry_path, run / "layouts")
            ).as_posix(),
            "data_sha256": result.candidates._geometry["data_sha256"],
        }
    for index, candidate in enumerate(candidates):
        name = candidate["candidate_id"]
        path = run / "layouts" / f"{name}.json"
        payload = {
            **common,
            **candidate,
            **references,
            "coordinate_frame": "local_arena_z_up",
            "geometry_model": (
                "hierarchical_support_grid_xy_yaw"
                if hierarchical
                else "fixed_orientation_aabb"
            ),
            "policy_file": "../policy_resolved.yaml",
        }
        if layout_format == "packed":
            path = run / "layouts" / "candidates.json"
            packed.append(payload)
        else:
            _write_atomic(path, payload)
        layouts.append(
            {
                key: candidate[key]
                for key in ("candidate_id", "proposal_ordinal", "variant_id")
            }
            | {"path": path.relative_to(run).as_posix()}
            | ({"index": index} if layout_format == "packed" else {})
        )
    if layout_format == "packed":
        _write_atomic(
            run / "layouts" / "candidates.json",
            {"schema_version": 1, "candidates": packed},
        )
    # Recheck after writing candidates so concurrent source changes cannot
    # publish a completed manifest tied to stale geometry or relative poses.
    _verify_baseline(result.baseline)
    if hierarchical:
        from ._intra_geometry import _load_geometry
        from .cfg import _decode_intra

        _load_geometry(
            result.candidates._geometry_path,
            result.baseline.data,
            _decode_intra(policy["intra_group"]),
            expected_sha256=result.candidates._geometry["data_sha256"],
        )
    _write_atomic(
        run / "run_result.json",
        {
            **common,
            **(
                {
                    "schema_version": 3 if hierarchical else 2,
                    "layout_storage": "packed",
                    "candidate_file": "layouts/candidates.json",
                }
                if layout_format == "packed"
                else {}
            ),
            **(
                {
                    "layout_storage": layout_format,
                    "support_geometry": {
                        "path": Path(
                            os.path.relpath(result.candidates._geometry_path, run)
                        ).as_posix(),
                        "data_sha256": result.candidates._geometry["data_sha256"],
                    },
                }
                if hierarchical
                else {}
            ),
            **_references(result.baseline, export_root, run),
            "status": result.status,
            "requested_count": policy["candidate_count"],
            "generated_count": len(candidates),
            "max_attempts": policy["max_attempts"],
            "attempts": result.attempts,
            "solver_status_counts": dict(result.solver_status_counts),
            "policy_file": "policy_resolved.yaml",
            "layouts": layouts,
        },
    )
    return run


def _prepare_result(result: SceneRandomizationResult) -> tuple[dict, list[dict]]:
    parameters = dict(result.resolved_parameters)
    policy = {
        "schema_version": (
            3
            if "intra_group" in parameters
            else (2 if set(parameters) & set(_BATCH_FIELDS) else 1)
        ),
        **parameters,
    }
    _validate_policy(policy)
    counts = result.solver_status_counts
    count = len(result.candidates)
    if (
        type(result.attempts) is not int
        or not 0 <= result.attempts <= parameters["max_attempts"]
        or set(counts) - {"solved", "no_solution", "budget_exhausted", "unsupported"}
        or any(type(value) is not int or value < 0 for value in counts.values())
        or sum(counts.values()) != result.attempts
        or counts.get("solved", 0) != count
        or count > parameters["candidate_count"]
    ):
        raise ValueError("Generation accounting is inconsistent.")
    completed = count == parameters["candidate_count"] and not counts.get(
        "unsupported", 0
    )
    exhausted = (
        count < parameters["candidate_count"]
        and result.attempts == parameters["max_attempts"]
        and not counts.get("unsupported", 0)
    )
    unsupported = (
        count < parameters["candidate_count"] and counts.get("unsupported", 0) == 1
    )
    if not {
        "completed": completed,
        "budget_exhausted": exhausted,
        "unsupported": unsupported,
    }.get(result.status, False):
        raise ValueError("Run status does not match generation accounting.")
    if (
        not isinstance(result.algorithm_version, str)
        or not result.algorithm_version.strip()
    ):
        raise ValueError("algorithm_version must identify the actual generator.")
    data = result.baseline.data
    from ._batch import _CandidatePool

    if isinstance(result.candidates, _CandidatePool):
        candidates = result.candidates._records(
            result.baseline, parameters, result.algorithm_version
        )
        if candidates and candidates[-1]["proposal_ordinal"] >= result.attempts:
            raise ValueError("Candidate proposal ordinals exceed the attempt budget.")
        return policy, candidates
    groups = {group["root_id"]: group for group in data["groups"]}
    expected_uids = set(groups) | {
        member["entity_id"] for group in groups.values() for member in group["members"]
    }
    candidates = []
    last_ordinal = -1
    for index, candidate in enumerate(result.candidates):
        variant = candidate.variant
        if (
            candidate.layout.status != "solved"
            or not candidate.checks.passed
            or candidate.fragment is None
            or variant is None
            or set(candidate.layout.xy_by_uid) != set(groups)
        ):
            raise ValueError("Only complete, checked, solved candidates can be saved.")
        metadata = dict(variant.metadata)
        proposal_ordinal = int(metadata["proposal_ordinal"])
        if (
            str(proposal_ordinal) != metadata["proposal_ordinal"]
            or not last_ordinal < proposal_ordinal < result.attempts
            or variant.ordinal != index
            or variant.seed != parameters["seed"]
            or variant.parent_scene_id != data["source_scene"]["content_sha256"]
            or metadata.get("generator") != result.algorithm_version
            or metadata.get("baseline_sha256") != data["data_sha256"]
            or {change.entity_uid for change in variant.pose_changes} != expected_uids
        ):
            raise ValueError(
                "Candidate provenance or group membership is inconsistent."
            )
        centers = {}
        for uid in sorted(groups):
            center = np.asarray(candidate.layout.xy_by_uid[uid], dtype=float)
            if center.shape != (2,) or not np.isfinite(center).all():
                raise ValueError("Group centers must contain two finite coordinates.")
            centers[uid] = center
        _check_geometry(data, centers, float(parameters["collision_margin_m"]))
        changes = {
            change.entity_uid: np.asarray(change.arena_pose)
            for change in variant.pose_changes
        }
        for uid, group in groups.items():
            root_pose = _read_pose(group["root_pose"])
            root_pose[:2, 3] += centers[uid] - group["aabb_2d"]["center_xy"]
            expected = {uid: root_pose} | {
                member["entity_id"]: root_pose @ _read_pose(member["pose_in_root"])
                for member in group["members"]
            }
            if any(
                not np.allclose(changes[member], pose, atol=1e-9, rtol=0)
                for member, pose in expected.items()
            ):
                raise ValueError(
                    "Candidate poses do not match the saved group centers."
                )
        candidates.append(
            {
                "candidate_id": f"candidate_{proposal_ordinal:06d}",
                "proposal_ordinal": proposal_ordinal,
                "variant_id": variant.variant_id,
                "variant_ordinal": variant.ordinal,
                "variant_metadata": metadata,
                "groups": [
                    {
                        "root_id": uid,
                        "center_xy": list(map(float, center)),
                        "yaw_delta_deg": 0.0,
                    }
                    for uid, center in centers.items()
                ],
            }
        )
        last_ordinal = proposal_ordinal
    return policy, candidates


def _check_geometry(data: dict, centers: dict[str, np.ndarray], margin: float) -> None:
    table_center = np.asarray(data["table"]["center_xy"])
    table_half = np.asarray(data["table"]["size_xy"]) / 2
    boxes = []
    for group in data["groups"]:
        center = centers[group["root_id"]]
        half = np.asarray(group["aabb_2d"]["size_xy"]) / 2
        box = np.asarray((center - half, center + half))
        if np.any(box[0] < table_center - table_half - 1e-7) or np.any(
            box[1] > table_center + table_half + 1e-7
        ):
            raise ValueError("Candidate group lies outside the table bounds.")
        for other in boxes:
            if np.max(np.maximum(box[0] - other[1], other[0] - box[1])) < margin - 1e-7:
                raise ValueError(
                    "Candidate group clearance is below the required margin."
                )
        boxes.append(box)


def _verify_baseline(baseline: PreparedSceneBaseline) -> Path:
    try:
        persisted = yaml.load(
            baseline.path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader
        )
    except yaml.YAMLError as exc:
        raise ValueError("Persisted baseline is invalid.") from exc
    if (
        persisted != baseline.data
        or _data_digest(persisted) != persisted["data_sha256"]
    ):
        raise ValueError("Persisted baseline changed after generation.")
    source = persisted["source_scene"]
    export_root = (baseline.path.parent / source["export_dir"]).resolve()
    config_bytes, config = _read_json(export_root / "scene_config.json")
    graph_bytes, _ = _read_json(export_root / "scene_graph.json")
    files = [export_root / entry["shape"]["fpath"] for entry in _rigid_entries(config)]
    if (
        _source_digest(export_root, config_bytes, graph_bytes, files)
        != source["content_sha256"]
    ):
        raise ValueError("Source scene changed after generation.")
    return export_root


def _references(
    baseline: PreparedSceneBaseline, export_root: Path, parent: Path
) -> dict:
    source = baseline.data["source_scene"]
    return {
        "source_scene": {
            "scene_id": source["scene_id"],
            "content_sha256": source["content_sha256"],
            "export_dir": Path(os.path.relpath(export_root, parent)).as_posix(),
        },
        "base_data": {
            "path": Path(os.path.relpath(baseline.path, parent)).as_posix(),
            "data_sha256": baseline.data["data_sha256"],
        },
    }


def _write_atomic(path: Path, data: dict[str, Any]) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            if path.suffix == ".yaml":
                yaml.safe_dump(data, stream, sort_keys=False, allow_unicode=True)
            else:
                if path.name == "candidates.json":
                    json.dump(
                        data,
                        stream,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                else:
                    json.dump(
                        data, stream, indent=2, ensure_ascii=False, allow_nan=False
                    )
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
