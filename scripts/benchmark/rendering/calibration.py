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

"""Recomputable heldout photometric-calibration receipts, outside render profiles.

Receipts never certify from a caller's status bit. They bind source capture and
independent-reference JSON/NPZ files, the exact frozen profile and native lighting
controls, a physical reference-light/renderer signature, and the unchanged RGB
quality policy. Every heldout image must satisfy its SSIM/LPIPS thresholds and an
independently recomputed reference noise budget. Training and heldout captures
must be distinct. Per-frame benchmark quality remains a separate requirement.

Store the receipt at the experiment root: every referenced source must remain
inside its parent directory. Keep ``profile.photometric_calibration`` unchanged
and pass the receipt through the evaluator's external calibration document;
putting a receipt's own profile hash inside the profile would be circular.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scripts.benchmark.core.artifacts import stable_hash, write_json
from scripts.benchmark.rendering.quality_protocol import (
    protocol_for_modalities,
    validate_protocol,
)

__all__ = ["create_calibration_receipt", "verify_calibration"]

_KIND = "photometric_calibration_receipt"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inside(root: Path, path: str | Path) -> Path:
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(
            "Calibration evidence must remain inside the receipt directory"
        )
    return resolved


def _artifact(root: Path, path: Path, kind: str) -> dict[str, Any]:
    path = _inside(root, path)
    return {"path": str(path.relative_to(root)), "sha256": _hash(path), "kind": kind}


def _verify_file(root: Path, artifact: dict[str, Any]) -> Path:
    path = _inside(root, artifact["path"])
    if _hash(path) != artifact["sha256"]:
        raise ValueError(f"Calibration evidence checksum mismatch: {artifact['path']}")
    return path


def _lighting_controls(profile: dict[str, Any], backend: str) -> dict[str, Any]:
    controls = profile.get("lighting_controls", {}).get(backend)
    required = {"directional_intensity", "environment_intensity", "environment_color"}
    if not isinstance(controls, dict) or set(controls) != required:
        raise ValueError(
            "Calibration requires all three explicit native lighting controls in the frozen profile"
        )
    color = controls["environment_color"]
    if not isinstance(color, (list, tuple)) or len(color) != 3:
        raise ValueError(
            "Calibration environment colour must have three linear RGB values"
        )
    for value in [
        controls["directional_intensity"],
        controls["environment_intensity"],
        *color,
    ]:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(
                "Calibration lighting controls must be finite and nonnegative"
            )
    return {**controls, "environment_color": list(color)}


def _reference_signature(provenance: dict[str, Any]) -> dict[str, Any]:
    required = (
        "renderer",
        "renderer_version",
        "variant",
        "material_model",
        "implementation_sha256",
        "lighting",
    )
    if any(key not in provenance for key in required):
        raise ValueError(
            "Independent reference physical/renderer signature is incomplete"
        )
    return {
        **{key: deepcopy(provenance[key]) for key in required},
        "integrator": {
            key: provenance["integrator"][key] for key in ("type", "max_depth")
        },
    }


def _source(
    root: Path, artifact: dict[str, Any], *, reference: bool
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    path = _verify_file(root, artifact)
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema_version") != 1 or not isinstance(
        record.get("metadata"), dict
    ):
        raise ValueError("Calibration sources must use the version-one frame schema")
    arrays = _inside(path.parent, record["arrays"]["path"])
    if _hash(arrays) != record["arrays"]["sha256"]:
        raise ValueError("Calibration source NPZ checksum mismatch")
    evidence = [
        _artifact(root, path, "calibration_frame_metadata"),
        _artifact(root, arrays, "calibration_frame_arrays"),
    ]
    if reference:
        manifest_path = path.parent / "reference.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = next(
            (item for item in manifest["frames"] if item["path"] == path.name), None
        )
        if (
            entry is None
            or entry["sha256"] != _hash(path)
            or manifest["provenance"] != record["reference_provenance"]
        ):
            raise ValueError(
                "Calibration reference metadata/provenance is not bound to its manifest"
            )
        evidence.append(
            _artifact(root, manifest_path, "calibration_reference_manifest")
        )
    return path, record, evidence


def _capture_identity(metadata: dict[str, Any]) -> str:
    fields = (
        "backend",
        "scene_sha256",
        "profile_sha256",
        "view_sha256",
        "frame_id",
        "env_id",
        "camera_name",
    )
    if any(field not in metadata for field in fields):
        raise ValueError("Calibration capture identity is incomplete")
    return stable_hash({field: metadata[field] for field in fields})


def _postprocess_signature(evidence: dict[str, Any], backend: str) -> dict[str, Any]:
    if (
        not isinstance(evidence, dict)
        or not evidence.get("scope")
        or not isinstance(evidence.get("settings"), dict)
    ):
        raise ValueError(
            "Calibration requires captured postprocess state with an explicit observation scope"
        )
    settings = evidence["settings"]
    required = (
        ("tone_mapping_enabled", "tone_mapping_exposure", "auto_exposure_enabled")
        if backend == "embodichain"
        else (
            "/rtx/post/histogram/enabled",
            "/rtx/post/tonemap/op",
            "/rtx/post/tonemap/colorMode",
            "/rtx/post/tonemap/filmIso",
            "/rtx/post/tonemap/exposureTime",
            "/rtx/post/tonemap/fNumber",
            "/rtx/post/tonemap/responsivity",
            "/rtx/post/tonemap/enableSrgbToGamma",
        )
    )
    if any(settings.get(key) is None for key in required):
        raise ValueError(
            "Calibration postprocess exposure/tone/colour state is incomplete"
        )
    return {"scope": evidence["scope"], "settings": deepcopy(settings)}


def _capture_run_proof(
    root: Path, path: Path, record: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Bind pixels to their completed worker, frozen config and frame index."""
    run = path.parent.parent
    if path.parent.name != "frames":
        raise ValueError(
            "Calibration captures must come from a completed benchmark run's frames directory"
        )
    result_path = _inside(root, run / "result.json")
    config_path = _inside(root, run / "worker_config.json")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metadata = record["metadata"]
    if result.get("status") != "completed" or result.get("backend") != metadata.get(
        "backend"
    ):
        raise ValueError("Calibration capture has no matching completed worker result")
    if (
        stable_hash(config["profile"]) != metadata["profile_sha256"]
        or result.get("profile_sha256") != metadata["profile_sha256"]
    ):
        raise ValueError(
            "Calibration pixels are not bound to the worker's actual frozen profile"
        )
    index_path = _verify_file(run, result["frames_manifest"])
    index = json.loads(index_path.read_text(encoding="utf-8"))
    entry = next(
        (item for item in index["frames"] if _inside(run, item["path"]) == path), None
    )
    if (
        entry is None
        or entry["sha256"] != _hash(path)
        or entry["arrays_sha256"] != record["arrays"]["sha256"]
    ):
        raise ValueError(
            "Calibration capture does not match the completed worker's frame index"
        )
    renderer = result["renderer"]
    if renderer.get("effective_lighting_controls") != _lighting_controls(
        config["profile"], metadata["backend"]
    ):
        raise ValueError(
            "Observed native lighting controls differ from the source profile"
        )
    signature = _postprocess_signature(
        renderer.get("postprocess_evidence"), metadata["backend"]
    )
    return signature, [
        _artifact(root, result_path, "calibration_worker_result"),
        _artifact(root, config_path, "calibration_worker_config"),
        _artifact(root, index_path, "calibration_frame_index"),
    ]


@lru_cache(maxsize=128)
def _measured_pair(
    capture_path: str,
    reference_path: str,
    source_hashes: tuple[str, ...],
) -> dict[str, Any]:
    """Cache only numerical work after every source hash is checked by the caller."""
    import numpy as np
    from scripts.benchmark.rendering.quality import _display_rgb, evaluate_frame
    from scripts.benchmark.rendering.reference import load_reference_frame

    path = Path(capture_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    with np.load(path.parent / record["arrays"]["path"], allow_pickle=False) as archive:
        capture = {name: archive[name].copy() for name in archive.files}
    capture["metadata"] = deepcopy(record["metadata"])
    # Calibration validation cannot recursively certify itself.
    capture["metadata"]["photometric_calibration"] = {"status": "not_qualified"}
    ref = load_reference_frame(Path(reference_path))
    ref_record = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    with np.load(
        Path(reference_path).parent / ref_record["arrays"]["path"], allow_pickle=False
    ) as archive:
        if "rgb_repeat" not in archive:
            raise ValueError(
                "Calibration reference requires its independently seeded RGB repeat"
            )
        repeat = {"rgb": archive["rgb_repeat"].copy(), "metadata": ref["metadata"]}
    first, second = _display_rgb(ref), _display_rgb(repeat)
    if first.shape != second.shape:
        raise ValueError("Calibration reference repeat dimensions differ")
    validation = ref_record.get("reference_validation", {})
    seeds = validation.get("seeds", [])
    if len(seeds) != 2 or len(set(seeds)) != 2:
        raise ValueError(
            "Calibration reference needs two distinct recorded sampler seeds"
        )
    noise = float(np.sqrt(np.mean((first - second) ** 2)))
    ref["reference_validation"] = {
        "metric": "independent_seed_display_srgb_rmse",
        "value": noise,
        "seeds": seeds,
        "denominator": int(first.size),
    }
    candidate_poses = capture["metadata"].get("world_from_instances")
    reference_poses = ref["metadata"].get("world_from_instances")
    if (
        not candidate_poses
        or not reference_poses
        or set(candidate_poses) != set(reference_poses)
    ):
        raise ValueError(
            "Calibration sources must bind complete canonical object poses"
        )
    if any(
        np.asarray(candidate_poses[key]).shape != (4, 4)
        or not np.allclose(
            candidate_poses[key], reference_poses[key], atol=1e-5, rtol=0
        )
        for key in candidate_poses
    ):
        raise ValueError("Calibration capture and reference object poses differ")
    policy = protocol_for_modalities(("rgb",))
    report = evaluate_frame(capture, ref, policy)
    checks = {
        name: report["checks"][name]
        for name in report["required_checks"]
        if name != "photometric_calibration"
    }
    metrics = {name: report["metrics"][name] for name in report["required_metrics"]}
    passed = all(
        item["status"] == "passed" for item in [*checks.values(), *metrics.values()]
    )
    return {
        "passed": passed,
        "checks": checks,
        "metrics": metrics,
        "reference_noise_rmse": noise,
        "reference_id": report["reference_id"],
        "capture_rgb_sha256": hashlib.sha256(
            np.ascontiguousarray(capture["rgb"]).tobytes()
        ).hexdigest(),
    }


def _recompute(root: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    if (
        receipt.get("schema_version") != 1
        or receipt.get("kind") != "render_photometric_calibration"
    ):
        raise ValueError(
            "Expected a version-one measured photometric calibration receipt"
        )
    backend = receipt["backend"]
    if backend not in {"embodichain", "isaaclab"}:
        raise ValueError("Calibration backend is not a tested renderer")
    profile = receipt["profile"]
    if stable_hash(profile) != receipt["profile_sha256"]:
        raise ValueError("Calibration profile hash mismatch")
    controls = _lighting_controls(profile, backend)
    if controls != receipt["lighting_controls"]:
        raise ValueError(
            "Calibration native light controls do not match the frozen profile"
        )
    policy = validate_protocol(receipt["protocol"])
    if (
        policy != protocol_for_modalities(("rgb",))
        or stable_hash(policy) != receipt["protocol_sha256"]
    ):
        raise ValueError(
            "Calibration requires the unchanged frozen RGB quality protocol"
        )
    if not receipt.get("training") or not receipt.get("heldout"):
        raise ValueError("Calibration needs nonempty training and heldout capture sets")
    identities: dict[str, set[str]] = {"training": set(), "heldout": set()}
    arrays: dict[str, set[str]] = {"training": set(), "heldout": set()}
    measurements, evidence = {"training": [], "heldout": []}, {}
    for split in ("training", "heldout"):
        for pair in receipt[split]:
            capture_path, capture_record, capture_evidence = _source(
                root, pair["capture"], reference=False
            )
            reference_path, reference_record, reference_evidence = _source(
                root, pair["reference"], reference=True
            )
            metadata = capture_record["metadata"]
            postprocess, run_evidence = _capture_run_proof(
                root, capture_path, capture_record
            )
            if metadata.get("backend") != backend:
                raise ValueError(
                    "Calibration source backend differs from the selected backend"
                )
            if (
                split == "heldout"
                and metadata.get("profile_sha256") != receipt["profile_sha256"]
            ):
                raise ValueError(
                    "Heldout capture did not use the selected frozen profile"
                )
            if split == "heldout" and postprocess != receipt["postprocess_signature"]:
                raise ValueError(
                    "Heldout postprocess state differs from the selected calibration state"
                )
            if (
                _reference_signature(reference_record["reference_provenance"])
                != receipt["reference_signature"]
            ):
                raise ValueError(
                    "Calibration sources use different physical reference lighting or renderer definitions"
                )
            identity = _capture_identity(metadata)
            if identity in identities[split]:
                raise ValueError(
                    "Calibration split contains duplicate capture identities"
                )
            identities[split].add(identity)
            arrays[split].add(capture_record["arrays"]["sha256"])
            sources = capture_evidence + reference_evidence + run_evidence
            for item in sources:
                evidence[(item["path"], item["sha256"])] = item
            measured = _measured_pair(
                str(capture_path),
                str(reference_path),
                tuple(item["sha256"] for item in sources),
            )
            measurements[split].append(
                {"capture_identity": identity, **deepcopy(measured)}
            )
            arrays[split].add("decoded_rgb:" + measured["capture_rgb_sha256"])
    if (
        identities["training"] & identities["heldout"]
        or arrays["training"] & arrays["heldout"]
    ):
        raise ValueError(
            "Heldout calibration captures or image bytes overlap training data"
        )
    qualified = all(item["passed"] for item in measurements["heldout"])
    return {
        "status": "qualified" if qualified else "not_qualified",
        "measurements": measurements,
        "source_evidence": list(evidence.values()),
    }


def create_calibration_receipt(
    path: Path,
    *,
    backend: str,
    profile: dict[str, Any],
    training_pairs: list[dict[str, Path]],
    heldout_pairs: list[dict[str, Path]],
) -> dict[str, Any]:
    """Measure and save a calibration receipt without changing a render profile.

    Args:
        path: New JSON file at the experiment root; no overwrite is permitted.
        backend: ``embodichain`` or ``isaaclab``.
        profile: Exact selected profile, with all backend light controls explicit.
        training_pairs: Nonempty source pairs with ``capture`` and ``reference``
            frame-JSON paths; tuning profiles may differ from the selected one.
            Captures must belong to completed runs with config, frame-index,
            effective-lighting and postprocess evidence retained.
        heldout_pairs: Nonempty distinct source pairs captured with the selected
            profile. Every heldout pair must pass unchanged RGB/noise thresholds.

    Returns:
        Receipt with a status derived from recomputed source metrics. Passing it
        through the evaluator still requires exact case/profile/reference bindings.
    """
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"Calibration receipt already exists: {path}")
    if not training_pairs or not heldout_pairs:
        raise ValueError("Calibration requires training and heldout source pairs")
    root = path.parent
    reference_path = _inside(root, heldout_pairs[0]["reference"])
    reference_record = json.loads(reference_path.read_text(encoding="utf-8"))
    capture_path = _inside(root, heldout_pairs[0]["capture"])
    capture_record = json.loads(capture_path.read_text(encoding="utf-8"))
    postprocess, _ = _capture_run_proof(root, capture_path, capture_record)
    policy = protocol_for_modalities(("rgb",))
    receipt = {
        "schema_version": 1,
        "kind": "render_photometric_calibration",
        "backend": backend,
        "profile": deepcopy(profile),
        "profile_sha256": stable_hash(profile),
        "lighting_controls": _lighting_controls(profile, backend),
        "reference_signature": _reference_signature(
            reference_record["reference_provenance"]
        ),
        "postprocess_signature": postprocess,
        "protocol": policy,
        "protocol_sha256": stable_hash(policy),
    }
    for split, pairs in (("training", training_pairs), ("heldout", heldout_pairs)):
        receipt[split] = [
            {
                name: _artifact(root, Path(pair[name]), "calibration_" + name)
                for name in ("capture", "reference")
            }
            for pair in pairs
        ]
    receipt.update(_recompute(root, receipt))
    write_json(path, receipt)
    return receipt


def verify_calibration(
    calibration: dict[str, Any] | None,
    candidate_metadata: dict[str, Any],
    reference: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Verify an external calibration claim by recomputing heldout evidence.

    Args:
        calibration: Case-bound claim with one evidence descriptor whose kind is
            ``photometric_calibration_receipt``; a status flag alone is invalid.
        candidate_metadata: Actual captured backend/profile identity.
        reference: Independent benchmark reference frame.
        protocol: The caller's frozen quality policy; RGB thresholds must match.

    Returns:
        A quality check with ``passed``, ``failed`` or ``missing`` status and
        explicit reasons. Stored pass bits and stored metric values are not trusted.
    """
    if not isinstance(calibration, dict) or calibration.get("status") != "qualified":
        return {
            "status": "missing",
            "reason": "No measured heldout photometric calibration receipt",
        }
    try:
        provenance = reference["reference_provenance"]
        if calibration.get("reference_id") != provenance.get(
            "reference_id"
        ) or calibration.get("profile_sha256") != candidate_metadata.get(
            "profile_sha256"
        ):
            raise ValueError(
                "Calibration claim is not bound to the candidate profile/reference"
            )
        matches = [
            item
            for item in calibration.get("evidence", [])
            if item.get("kind") == _KIND
        ]
        if len(matches) != 1:
            raise ValueError(
                "Calibration requires exactly one measured photometric_calibration_receipt"
            )
        artifact = matches[0]
        path = Path(artifact["path"]).resolve()
        if _hash(path) != artifact["sha256"]:
            raise ValueError("Calibration receipt checksum mismatch")
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("backend") != candidate_metadata.get("backend") or receipt.get(
            "profile_sha256"
        ) != candidate_metadata.get("profile_sha256"):
            raise ValueError(
                "Calibration receipt backend/profile differs from this capture"
            )
        if receipt.get("reference_signature") != _reference_signature(provenance):
            raise ValueError(
                "Calibration receipt does not cover this physical reference-light/renderer definition"
            )
        if candidate_metadata.get("effective_lighting_controls") != receipt.get(
            "lighting_controls"
        ):
            raise ValueError(
                "Benchmark's observed native light controls differ from the receipt"
            )
        if _postprocess_signature(
            candidate_metadata.get("postprocess_evidence"),
            candidate_metadata["backend"],
        ) != receipt.get("postprocess_signature"):
            raise ValueError(
                "Benchmark postprocess state differs from the calibration state"
            )
        frozen = validate_protocol(protocol)
        rgb_policy = protocol_for_modalities(("rgb",))
        for key in ("ssim", "lpips", "rgb_domain", "reference_noise_rmse_max"):
            if frozen[key] != rgb_policy[key]:
                raise ValueError(
                    "Calibration and candidate RGB metric definitions differ"
                )
        if any(
            frozen["frame_thresholds"][name] != rgb_policy["frame_thresholds"][name]
            for name in ("ssim", "lpips")
        ):
            raise ValueError("Calibration and candidate RGB thresholds differ")
        measured = _recompute(path.parent, receipt)
        if measured["status"] != "qualified":
            return {
                "status": "failed",
                "reason": "Recomputed heldout calibration metrics/noise do not meet the frozen protocol",
                "evidence": measured,
            }
        if (
            receipt.get("status") != measured["status"]
            or receipt.get("measurements") != measured["measurements"]
        ):
            raise ValueError(
                "Stored calibration status/metrics do not match recomputed source evidence"
            )
        return {
            "status": "passed",
            "reason": None,
            "evidence": {
                "receipt": artifact,
                "profile_sha256": receipt["profile_sha256"],
                "lighting_controls": receipt["lighting_controls"],
                "heldout_count": len(measured["measurements"]["heldout"]),
                "source_evidence": measured["source_evidence"],
            },
        }
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        ImportError,
        RuntimeError,
    ) as error:
        return {"status": "failed", "reason": str(error)}
