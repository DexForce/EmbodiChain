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

"""Import measured photometric calibration and bind it to actual target captures.

No renderer or GPU is initialized. Heldout source metrics are recomputed on CPU;
target references must already exist from an initial offline evaluation. A valid
calibration document does not qualify target pixels or temporal sequences.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

from scripts.benchmark.core.artifacts import (
    ArtifactStore,
    read_json,
    read_json_object,
    stable_hash,
    write_json,
)
from .calibration import verify_calibration
from .quality_protocol import protocol_for_modalities, validate_protocol

__all__ = ["bind_calibration_receipt", "main"]

_BINDINGS = ("config_sha256", "scene_sha256", "view_sha256", "profile_sha256")
_KIND = "photometric_calibration_receipt"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root: Path, value: str | Path) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Evidence path escapes its bound root")
    return path


def _verify(root: Path, artifact: dict[str, Any]) -> Path:
    path = _inside(root, artifact["path"])
    if _sha(path) != artifact["sha256"]:
        raise ValueError(f"Evidence checksum mismatch: {artifact['path']}")
    return path


def _artifact(root: Path, path: Path, kind: str) -> dict[str, Any]:
    return {
        "path": _inside(root, path).relative_to(root).as_posix(),
        "sha256": _sha(path),
        "kind": kind,
    }


def _claim(
    receipt: Path, metadata: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    return {
        "status": "qualified",
        "profile_sha256": metadata["profile_sha256"],
        "reference_id": reference["reference_provenance"]["reference_id"],
        "evidence": [{"path": str(receipt), "sha256": _sha(receipt), "kind": _KIND}],
    }


def _check_receipt(receipt_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use real heldout source metadata, never the receipt's own claimed controls."""
    from .reference import load_reference_frame

    receipt = read_json_object(receipt_path)
    if (
        receipt.get("status") != "qualified"
        or receipt.get("kind") != "render_photometric_calibration"
    ):
        raise ValueError(
            "A measured, qualified photometric calibration receipt is required"
        )
    heldout = receipt.get("heldout")
    if not isinstance(heldout, list) or not heldout:
        raise ValueError("Calibration has no heldout source frames")
    capture_path = _verify(receipt_path.parent, heldout[0]["capture"])
    reference_path = _verify(receipt_path.parent, heldout[0]["reference"])
    metadata = deepcopy(read_json_object(capture_path)["metadata"])
    result = read_json_object(
        _inside(receipt_path.parent, capture_path.parent.parent / "result.json")
    )
    for key in ("effective_lighting_controls", "postprocess_evidence"):
        metadata[key] = deepcopy(result.get("renderer", {}).get(key))
    reference = load_reference_frame(reference_path)
    check = verify_calibration(
        _claim(receipt_path, metadata, reference),
        metadata,
        reference,
        protocol_for_modalities(("rgb",)),
    )
    if check.get("status") != "passed":
        raise ValueError(
            "Measured receipt verification failed: " + str(check.get("reason"))
        )
    evidence = check.get("evidence", {}).get("source_evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("Receipt verifier returned no checked source evidence")
    if receipt.get("source_evidence") != evidence:
        raise ValueError(
            "Stored receipt source evidence differs from the verified closure"
        )
    return receipt, check


def _import_receipt(source: Path, target: Path) -> dict[str, Any]:
    """Copy the verifier's complete evidence closure without rewriting JSON paths."""
    receipt, check = _check_receipt(source)
    receipt_sha = _sha(source)
    destination = target / "calibration_imports" / receipt_sha
    expected: dict[str, dict[str, Any]] = {}
    for artifact in check["evidence"]["source_evidence"]:
        if (
            not isinstance(artifact.get("path"), str)
            or Path(artifact["path"]).is_absolute()
        ):
            raise ValueError(
                "Portable calibration source evidence requires relative paths"
            )
        path = _verify(source.parent, artifact)
        relative = path.relative_to(source.parent).as_posix()
        if relative in expected and expected[relative]["sha256"] != artifact["sha256"]:
            raise ValueError("Conflicting source evidence for one relative path")
        expected[relative] = {**artifact, "path": relative}
    if source.name in expected:
        raise ValueError("Receipt filename collides with its source evidence")
    expected[source.name] = {"path": source.name, "sha256": receipt_sha, "kind": _KIND}
    reused = destination.exists()
    if reused:
        for artifact in expected.values():
            _verify(destination, artifact)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(
            tempfile.mkdtemp(prefix=f".{receipt_sha[:12]}-", dir=destination.parent)
        )
        try:
            for relative, artifact in expected.items():
                source_path = (
                    source
                    if relative == source.name
                    else _verify(source.parent, artifact)
                )
                target_path = _inside(stage, relative)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_path, target_path)
                _verify(stage, artifact)
            stage.rename(destination)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    imported = destination / source.name
    # Recompute at the imported location too: relative references must work
    # without depending on the source experiment's directory structure.
    _check_receipt(imported)
    artifacts = [
        _artifact(target, destination / relative, artifact["kind"])
        for relative, artifact in sorted(expected.items())
    ]
    manifest = {
        "schema_version": 1,
        "receipt_sha256": receipt_sha,
        "source_receipt": str(source),
        "imported_receipt": imported.relative_to(target).as_posix(),
        "source_evidence": artifacts,
        "byte_preserving": True,
        "source_verification": "heldout metrics and noise recomputed before and after copying",
        "backend": receipt["backend"],
        "profile_sha256": receipt["profile_sha256"],
    }
    manifest_path = destination.parent / f"{receipt_sha}.manifest.json"
    if manifest_path.exists():
        previous = read_json_object(manifest_path)
        if any(
            previous.get(key) != manifest[key]
            for key in ("receipt_sha256", "imported_receipt", "source_evidence")
        ):
            raise ValueError("Existing calibration import manifest differs")
    else:
        write_json(manifest_path, manifest)
    store = ArtifactStore(target)
    for artifact in artifacts:
        store.register_artifact(target / artifact["path"], kind=artifact["kind"])
    store.register_artifact(manifest_path, kind="calibration_import_manifest")
    return {
        "receipt": imported,
        "receipt_sha256": receipt_sha,
        "manifest": _artifact(target, manifest_path, "calibration_import_manifest"),
        "artifacts": artifacts,
        "reused": reused,
    }


def _raw_rows(root: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for number, line in enumerate(
        (root / "raw.jsonl").read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("run_id"), str):
            raise ValueError(f"Invalid raw run identity at line {number}")
        run_id = row["run_id"]
        if run_id in rows and rows[run_id] != row:
            raise ValueError("Conflicting raw execution records for " + run_id)
        rows[run_id] = row
    return rows


def _binding_records(
    root: Path, row: dict[str, Any], quality_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    matches = [
        record for record in quality_rows if record.get("run_id") == row["run_id"]
    ]
    if len(matches) != 1:
        raise ValueError(
            "Initial offline evaluation with reference_bindings.json is required"
        )
    record = matches[0]
    for key in ("backend", "case_id", *_BINDINGS):
        if record.get(key) != row.get(key):
            raise ValueError(
                "Evaluated target binding differs from its raw run: " + key
            )
    artifacts = [
        artifact
        for artifact in record.get("artifacts", [])
        if Path(artifact.get("path", "")).name == "reference_bindings.json"
    ]
    # One file may carry both a general quality role and a precision-selection
    # role. Verify every descriptor before collapsing only identical identities.
    identities = {
        (_verify(root, artifact), artifact["sha256"]) for artifact in artifacts
    }
    if len(identities) != 1:
        raise ValueError(
            "Expected one checksummed reference_bindings.json for the target run"
        )
    path, _ = next(iter(identities))
    bindings = read_json(path)
    if not isinstance(bindings, list):
        raise ValueError("reference_bindings.json must contain a list")
    return bindings, _artifact(root, path, "target_reference_bindings"), record


def _same(left: Any, right: Any) -> bool:
    if type(left) in (float, int) and type(right) in (float, int):
        return (
            math.isfinite(left)
            and math.isfinite(right)
            and math.isclose(left, right, rel_tol=1e-7, abs_tol=1e-8)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _same(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def _reference(
    root: Path,
    run: dict[str, Any],
    descriptor: dict[str, Any],
    aliases: list[dict[str, Any]],
    cache: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select existing reference pixels by checked geometry and temporal alias."""
    from .reference import load_reference_frame

    metadata = descriptor["metadata"]
    matches = {}
    for alias in aliases:
        if (
            alias.get("status") != "completed"
            or alias.get("frame_id") != metadata["frame_id"]
            or alias.get("time_s") != metadata["time_s"]
            or stable_hash(alias.get("history")) != stable_hash(metadata.get("history"))
        ):
            continue
        artifact = alias.get("binding_artifact")
        if not isinstance(artifact, dict):
            continue
        key = stable_hash(artifact)
        if key not in cache:
            path = _verify(root, artifact)
            binding = read_json_object(path)
            for evidence in binding["artifacts"]:
                _verify(root, evidence)
            cache[key] = binding
        binding = cache[key]
        identity = binding["identity"]
        digest = stable_hash(identity)
        if (
            alias.get("cache_key") != digest
            or alias.get("geometry_identity_sha256") != digest
        ):
            raise ValueError("Reference cache geometry identity checksum mismatch")
        expected = {
            "scene_sha256": metadata["scene_sha256"],
            "camera": {
                key: value
                for key, value in descriptor["canonical_camera"].items()
                if key != "name"
            },
            "object_transforms": metadata["world_from_objects"],
            "width": run["case"].width,
            "height": run["case"].height,
        }
        if any(not _same(identity.get(key), value) for key, value in expected.items()):
            continue
        frame = _inside(root, binding["frame"])
        if frame != _inside(root, alias["reference_frame"]):
            raise ValueError("Reference alias points to different frame pixels")
        if not any(
            _inside(root, evidence["path"]) == frame
            for evidence in binding["artifacts"]
        ):
            raise ValueError(
                "Reference frame is absent from the cache's checked artifacts"
            )
        matches[stable_hash(alias)] = (alias, frame)
    if len(matches) != 1:
        raise ValueError(
            "Missing or ambiguous existing reference for target frame; run initial evaluation first"
        )
    alias, frame = next(iter(matches.values()))
    reference = load_reference_frame(frame)
    provenance = reference["reference_provenance"]
    if (
        provenance["reference_id"] != alias["reference_id"]
        or provenance.get("source_scene_sha256") != metadata["scene_sha256"]
    ):
        raise ValueError(
            "Independent reference identity/scene differs from target binding"
        )
    ref_metadata = reference["metadata"]
    if (
        ref_metadata["frame_id"] != alias["source_frame_id"]
        or not _same(
            ref_metadata.get("world_from_instances"), metadata["world_from_instances"]
        )
        or not _same(
            {k: v for k, v in ref_metadata["camera"].items() if k != "name"},
            {k: v for k, v in descriptor["canonical_camera"].items() if k != "name"},
        )
    ):
        raise ValueError(
            "Independent reference geometry differs from the actual target frame"
        )
    ref_metadata["frame_id"] = ref_metadata["frame_index"] = metadata["frame_id"]
    ref_metadata["time_s"] = metadata["time_s"]
    if "history" in metadata:
        ref_metadata["history"] = deepcopy(metadata["history"])
    return reference, {
        "alias": alias,
        "frame": _artifact(root, frame, "target_reference_frame"),
    }


def _policy(run: dict[str, Any]) -> dict[str, Any]:
    if run.get("history_quality_protocol") is not None:
        return validate_protocol(run["history_quality_protocol"])
    declared = run["config"].get("quality_protocol")
    if declared is not None:
        return validate_protocol(declared)
    case = run["case"]
    temporal = case.temporal_mode == "moving" and (
        set(case.modalities) == {"rgb", "depth", "seg"}
        or run["config"].get("post_measurement_quality_capture")
        == "frozen_sequence_100"
    )
    return protocol_for_modalities(
        case.modalities, temporal_dt_s=1 / 30 if temporal else None
    )


def bind_calibration_receipt(
    receipt: Path,
    target_root: Path,
    *,
    run_ids: Sequence[str] | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    """Import a measured receipt and bind verified target case/reference scopes.

    Args:
        receipt: Source receipt; all source files must remain below its parent.
        target_root: Frozen formal experiment with raw runs and prior references.
        run_ids: Optional explicit target run selection; default completed runs.
        output: New report directory, absolute or relative to the target root.
            Defaults to a timestamped directory under ``calibration_bindings``.

    Returns:
        Status, import and evidence-document paths plus per-frame/scope reasons.
        Rejected receipts and failed scopes emit no calibration claim. This
        function does not change raw/runs/quality files or certify target images.
    """
    from .evaluate import _load_history, _load_run

    root = Path(target_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Target experiment root does not exist: {root}")
    source = Path(receipt).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = _inside(
        root, output if output is not None else Path("calibration_bindings") / stamp
    )
    if directory == root:
        raise ValueError("Use a new calibration binding report subdirectory")
    directory.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "rejected",
        "source_receipt": str(source),
        "target_root": str(root),
        "evidence_document": None,
        "imported_receipt": None,
        "source_verification": None,
        "frames": [],
        "failed_scopes": [],
        "records": [],
        "target_quality_status_changed": False,
        "target_images_require_own_metrics": True,
    }
    input_hashes = {}
    phase = "target_preflight"
    try:
        definition = read_json_object(root / "definition.json")
        if definition.get("definition_version") != "2.0":
            raise ValueError(
                "Target must be a formal definition-version 2.0 experiment"
            )
        input_hashes = {
            name: _sha(root / name)
            for name in ("raw.jsonl", "runs.json", "quality.json")
        }
        raw = _raw_rows(root)
        selected = (
            set(run_ids)
            if run_ids is not None
            else {key for key, row in raw.items() if row.get("status") == "completed"}
        )
        if selected - raw.keys():
            raise ValueError(
                "Unknown target run IDs: " + ", ".join(sorted(selected - raw.keys()))
            )
        phase = "source_import"
        imported = _import_receipt(source, root)
        report.update(
            imported_receipt=imported["receipt"].relative_to(root).as_posix(),
            source_verification={
                "status": "passed",
                "receipt_sha256": imported["receipt_sha256"],
                "reused_import": imported["reused"],
                "manifest": imported["manifest"],
            },
        )
        quality_rows = read_json_object(root / "quality.json").get("runs", [])
        phase = "target_binding"
        scopes: dict[str, dict[str, Any]] = {}
        unknown_failed_cases = set()
        cache: dict[str, Any] = {}
        for run_id in sorted(selected):
            row = raw[run_id]
            identity = (row.get("case_id"), row.get("backend"))
            try:
                if row.get("status") != "completed":
                    raise ValueError(
                        "Only completed raw runs can receive calibration bindings"
                    )
                run = _load_run(root, row)
                aliases, alias_evidence, evaluated = _binding_records(
                    root, row, quality_rows
                )
                history = evaluated.get("timed_history")
                if history is not None:
                    run = _load_history(root, run, _verify(root, history))
                    if any(
                        history.get(key) != run["timed_history"].get(key)
                        for key in (
                            "schedule_sha256",
                            "measured_frames",
                            "sequence_frames",
                        )
                    ):
                        raise ValueError(
                            "Evaluated history identity differs from its verified manifest"
                        )
                elif any(alias.get("history") is not None for alias in aliases):
                    raise ValueError(
                        "History reference aliases require the evaluated timed-history proof"
                    )
                policy = _policy(run)
                if not run["descriptors"]:
                    raise ValueError("Target run has no actual captured frames")
                for descriptor in run["descriptors"]:
                    metadata = deepcopy(descriptor["metadata"])
                    for key in ("effective_lighting_controls", "postprocess_evidence"):
                        metadata[key] = deepcopy(run["renderer"].get(key))
                    frame_check = {
                        "run_id": run_id,
                        "case_id": row["case_id"],
                        "backend": row["backend"],
                        "env_id": metadata["env_id"],
                        "camera_name": metadata["camera_name"],
                        "frame_id": metadata["frame_id"],
                        "status": "failed",
                        "candidate": _artifact(
                            root, descriptor["metadata_path"], "target_capture"
                        ),
                        "reference_bindings": alias_evidence,
                    }
                    if history is not None:
                        frame_check.update(
                            history=deepcopy(metadata["history"]),
                            timed_history=deepcopy(run["timed_history"]),
                        )
                    report["frames"].append(frame_check)
                    try:
                        reference, binding = _reference(
                            root, run, descriptor, aliases, cache
                        )
                        claim = _claim(imported["receipt"], metadata, reference)
                        scope_key = stable_hash(
                            {
                                "case_id": row["case_id"],
                                "backend": row["backend"],
                                "reference_id": claim["reference_id"],
                            }
                        )
                        frame_check.update(
                            reference_id=claim["reference_id"],
                            reference=binding,
                            scope_id=scope_key,
                        )
                        record = {
                            key: row[key] for key in ("case_id", "backend", *_BINDINGS)
                        }
                        record.update(
                            reference_id=claim["reference_id"],
                            photometric_calibration={
                                **claim,
                                "evidence": [
                                    _artifact(root, imported["receipt"], _KIND)
                                ],
                            },
                        )
                        scope = scopes.setdefault(
                            scope_key, {"record": record, "frames": [], "reasons": []}
                        )
                        if scope["record"] != record:
                            scope["reasons"].append("ambiguous_case_reference_bindings")
                        scope["frames"].append(frame_check)
                        check = verify_calibration(claim, metadata, reference, policy)
                        frame_check.update(
                            status=check.get("status", "failed"),
                            reason=check.get("reason"),
                        )
                        if check.get("status") != "passed":
                            scope["reasons"].append(
                                str(
                                    check.get("reason")
                                    or "calibration_verification_failed"
                                )
                            )
                    except (
                        OSError,
                        ValueError,
                        KeyError,
                        TypeError,
                        ImportError,
                        RuntimeError,
                    ) as exc:
                        frame_check["reason"] = f"{type(exc).__name__}: {exc}"
                        unknown_failed_cases.add(identity)
                        report["failed_scopes"].append(
                            {
                                key: frame_check.get(key)
                                for key in (
                                    "run_id",
                                    "case_id",
                                    "backend",
                                    "env_id",
                                    "camera_name",
                                    "frame_id",
                                    "history",
                                    "reason",
                                )
                                if key in frame_check
                            }
                        )
            except (
                OSError,
                ValueError,
                KeyError,
                TypeError,
                ImportError,
                RuntimeError,
            ) as exc:
                unknown_failed_cases.add(identity)
                report["failed_scopes"].append(
                    {
                        "run_id": run_id,
                        "case_id": row.get("case_id"),
                        "backend": row.get("backend"),
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
        records = []
        for scope_id, scope in sorted(scopes.items()):
            record = scope["record"]
            if (record["case_id"], record["backend"]) in unknown_failed_cases:
                scope["reasons"].append(
                    "one_or_more_target_frames_lack_verified_input_or_reference"
                )
            if scope["reasons"] or any(
                frame["status"] != "passed" for frame in scope["frames"]
            ):
                report["failed_scopes"].append(
                    {
                        "scope_id": scope_id,
                        "case_id": record["case_id"],
                        "backend": record["backend"],
                        "reference_id": record["reference_id"],
                        "reasons": sorted(set(scope["reasons"])),
                    }
                )
            else:
                records.append(
                    {
                        **record,
                        "verified_run_ids": sorted(
                            {frame["run_id"] for frame in scope["frames"]}
                        ),
                        "verified_target_frames": len(scope["frames"]),
                        "receipt_import_manifest": imported["manifest"],
                    }
                )
        if any(_sha(root / name) != digest for name, digest in input_hashes.items()):
            raise ValueError(
                "Target raw/runs/quality inputs changed during calibration binding"
            )
        report["records"] = records
        report["status"] = (
            "partially_bound"
            if records and report["failed_scopes"]
            else "bound" if records else "no_valid_bindings"
        )
        if records:
            path = write_json(
                directory / "calibration_evidence.json",
                {
                    "schema_version": 1,
                    "kind": "measured_photometric_calibration_bindings",
                    "records": records,
                    "target_quality_status_changed": False,
                    "target_images_require_own_metrics": True,
                    "target_inputs_sha256": input_hashes,
                },
            )
            report["evidence_document"] = path.relative_to(root).as_posix()
            ArtifactStore(root).register_artifact(
                path, kind="explicit_calibration_input"
            )
    except (OSError, ValueError, KeyError, TypeError, ImportError, RuntimeError) as exc:
        report["status"] = "rejected"
        report["reason"] = f"{type(exc).__name__}: {exc}"
        report["failure_phase"] = phase
        if phase == "source_import":
            report["source_verification"] = {
                "status": "failed",
                "reason": report["reason"],
            }
    report["target_inputs_sha256"] = input_hashes
    report["report_path"] = (
        (directory / "binding_report.json").relative_to(root).as_posix()
    )
    write_json(directory / "binding_report.json", report)
    ArtifactStore(root).register_artifact(
        directory / "binding_report.json", kind="calibration_binding_report"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Import and bind a receipt; print the report and existing-evaluator input path.

    Args:
        argv: Optional command arguments, excluding the module name.

    Returns:
        Zero when at least one valid scope was bound, otherwise one.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--run-id", action="append")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = bind_calibration_receipt(
        args.receipt, args.target_root, run_ids=args.run_id, output=args.output
    )
    print(
        json.dumps(
            {
                key: report.get(key)
                for key in (
                    "status",
                    "reason",
                    "imported_receipt",
                    "evidence_document",
                    "report_path",
                )
            },
            indent=2,
        )
    )
    return int(report["status"] not in {"bound", "partially_bound"})


if __name__ == "__main__":
    raise SystemExit(main())
