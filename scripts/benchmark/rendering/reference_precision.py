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

"""Fixed, reference-only precision selection and portable evidence verification.

This module never accepts candidate pixels or quality metrics. NumPy is imported
only to recompute independent reference noise; report verification is stdlib-only.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import hashlib
import math
from pathlib import Path
from typing import Any

from scripts.benchmark.core.artifacts import read_json_object, stable_hash, write_json

__all__ = [
    "PRECISION_POLICY_ID",
    "precision_policy",
    "validate_precision_settings",
    "freeze_precision_policy",
    "measure_reference_noise",
    "select_reference",
    "verify_reference_precision",
    "precision_quality_reasons",
]

PRECISION_POLICY_ID = "reference-noise/4096-16384-v1"
_POLICY = {
    "schema_version": 1,
    "id": PRECISION_POLICY_ID,
    "samples_per_pixel": [4096, 16384],
    "seeds": [0, 1],
    "threads": 8,
    "metric": "independent_seed_display_srgb_rmse",
    "threshold": 0.01,
    "comparison": "le",
    "population": "all pixels and RGB channels in both independent reference repeats",
    "input_encoding": "linear_rec709",
    "transfer": "clip [0,1], IEC 61966-2-1 sRGB OETF",
    "selection": "first passing rung; terminal failed rung retained as unqualified evidence",
    "invalid_evidence": "stop without escalation",
    "candidate_inputs": "none",
    "metadata_agreement": {"relative_tolerance": 1e-6, "absolute_tolerance": 1e-9},
}


def precision_policy(policy_id: str = PRECISION_POLICY_ID) -> dict[str, Any]:
    """Return the sole immutable precision policy, independent of candidate data."""
    if policy_id != PRECISION_POLICY_ID:
        raise ValueError(f"Unknown reference precision policy: {policy_id}")
    return deepcopy(_POLICY)


def validate_precision_settings(
    policy_id: str, settings: dict[str, int]
) -> dict[str, Any]:
    """Require explicitly selected 4096 spp, seed zero and eight CPU threads."""
    policy = precision_policy(policy_id)
    if settings != {"spp": 4096, "seed": 0, "threads": 8} or any(
        type(value) is not int for value in settings.values()
    ):
        raise ValueError("Precision policy requires --spp 4096 --seed 0 --threads 8")
    return policy


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(root: Path, value: str | Path) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Reference precision evidence escapes the experiment root")
    return path


def _artifact(root: Path, path: Path, kind: str) -> dict[str, str]:
    path = _inside(root, path)
    return {
        "path": path.relative_to(root.resolve()).as_posix(),
        "sha256": _sha(path),
        "kind": kind,
    }


def _verified(root: Path, descriptor: dict[str, Any]) -> Path:
    path = _inside(root, descriptor["path"])
    if not path.is_file() or _sha(path) != descriptor.get("sha256"):
        raise ValueError(f"Reference precision evidence checksum mismatch: {path}")
    return path


def freeze_precision_policy(
    root: Path,
    directory: Path,
    settings: dict[str, int],
    policy_id: str = PRECISION_POLICY_ID,
) -> dict[str, str]:
    """Write the fixed policy before reference acquisition, refusing replacement."""
    policy = validate_precision_settings(policy_id, settings)
    path = _inside(root, directory) / "policy.json"
    if path.exists():
        raise ValueError("Reference precision policy already exists")
    write_json(path, policy)
    return _artifact(root, path, "reference_precision_policy")


def _read_policy(root: Path, descriptor: dict[str, Any]) -> dict[str, Any]:
    policy = read_json_object(_verified(root, descriptor))
    if descriptor.get("kind") != "reference_precision_policy" or policy != _POLICY:
        raise ValueError(
            "Reference precision policy differs from the frozen definition"
        )
    return policy


def _noise_evidence(root: Path, cache_binding: Path, spp: int) -> dict[str, Any]:
    """Verify the existing generator/cache envelope without importing NumPy."""
    if type(spp) is not int or spp not in _POLICY["samples_per_pixel"]:
        raise ValueError("Unsupported precision rung")
    path = _inside(root, cache_binding)
    binding = read_json_object(path)
    identity = binding["identity"]
    if identity["settings"] != {"spp": spp, "seed": 0, "threads": 8} or any(
        type(value) is not int for value in identity["settings"].values()
    ):
        raise ValueError("Reference attempt settings differ from the fixed rung")
    key = stable_hash(identity)
    if path.parent.name != key:
        raise ValueError("Reference cache path does not match its identity")
    records = binding["artifacts"]
    paths = [_verified(root, item) for item in records]
    if len(paths) != 3 or len(set(paths)) != 3:
        raise ValueError("Reference cache requires exactly three source artifacts")
    frame_path = _inside(root, binding["frame"])
    frame = read_json_object(frame_path)
    arrays_path = (frame_path.parent / frame["arrays"]["path"]).resolve()
    manifest_path = frame_path.parent / "reference.json"
    if (
        not arrays_path.is_relative_to(frame_path.parent)
        or set(paths) != {frame_path, arrays_path, manifest_path}
        or _sha(arrays_path) != frame["arrays"]["sha256"]
    ):
        raise ValueError("Reference cache source closure is incomplete")
    manifest = read_json_object(manifest_path)
    matched = [item for item in manifest["frames"] if item["path"] == frame_path.name]
    provenance = frame["reference_provenance"]
    if (
        len(matched) != 1
        or matched[0]["sha256"] != _sha(frame_path)
        or manifest["provenance"] != provenance
        or provenance["reference_id"]
        != stable_hash(
            {key: value for key, value in provenance.items() if key != "reference_id"}
        )
    ):
        raise ValueError("Reference manifest/frame provenance is inconsistent")
    if (
        provenance.get("kind") != "independent_offline"
        or provenance.get("renderer") != "mitsuba"
        or provenance.get("device") != "cpu"
        or provenance.get("variant") != "llvm_ad_rgb"
        or provenance.get("assets_verified") is not True
        or provenance.get("implementation_sha256")
        != identity["reference_implementation_sha256"]
        or provenance.get("source_scene_sha256") != identity["scene_sha256"]
        or provenance.get("threads") != 8
        or provenance.get("width") != identity["width"]
        or provenance.get("height") != identity["height"]
        or provenance.get("integrator")
        != {
            "type": "path",
            "max_depth": 8,
            "spp": spp,
            "seed": 0,
            "independent_repeat_count": 2,
        }
        or provenance.get("noise_floor_policy", {}).get("metric") != _POLICY["metric"]
        or provenance.get("noise_floor_policy", {}).get("maximum") != 0.01
        or frame["metadata"].get("rgb_encoding") != "linear_rec709"
    ):
        raise ValueError("Reference attempt has unsupported or inconsistent provenance")
    noise = frame["reference_validation"]
    value = noise.get("value")
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value < 0
        or noise.get("metric") != _POLICY["metric"]
        or noise.get("threshold") != 0.01
        or noise.get("seeds") != [0, 1]
        or noise.get("samples_per_pixel_per_repeat") != spp
        or noise.get("denominator") != identity["width"] * identity["height"] * 3
        or noise.get("passed") is not (value <= 0.01)
    ):
        raise ValueError("Reference noise evidence is missing or inconsistent")
    geometry = deepcopy(identity)
    del geometry["settings"]["spp"]
    physical = deepcopy(provenance)
    for name in ("reference_id", "camera_specs_sha256"):
        physical.pop(name, None)
    del physical["integrator"]["spp"]
    return {
        "schema_version": 1,
        "spp": spp,
        "cache_key": key,
        "geometry_sha256": stable_hash(geometry),
        "physical_signature_sha256": stable_hash(physical),
        "reference_id": provenance["reference_id"],
        "reference_frame": frame_path.relative_to(root.resolve()).as_posix(),
        "binding_artifact": _artifact(
            root, path, "reference_precision_attempt_binding"
        ),
        "source_evidence": [
            _artifact(root, source, "reference_precision_attempt_source")
            for source in paths
        ],
        "noise": deepcopy(noise),
        "arrays_path": arrays_path,
        "shape": [identity["height"], identity["width"], 3],
    }


def measure_reference_noise(
    root: Path, cache_binding: Path, *, spp: int
) -> dict[str, Any]:
    """Recompute noise from both hash-verified independent linear-RGB repeats.

    Args:
        root: Experiment trust boundary containing the immutable reference cache.
        cache_binding: Existing ``cache_binding.json`` under that root.
        spp: One declared rung, 4096 or 16384; seeds and threads stay fixed.

    Returns:
        JSON-safe reference-only evidence, including recomputed noise and hashes.
        Missing, corrupt or inconsistent evidence raises ``ValueError``.
    """
    import numpy as np

    evidence = _noise_evidence(root, cache_binding, spp)
    displays = []
    with np.load(evidence.pop("arrays_path"), allow_pickle=False) as arrays:
        for name in ("rgb", "rgb_repeat"):
            if name not in arrays:
                raise ValueError(f"Independent reference repeat is missing: {name}")
            pixels = arrays[name]
            if (
                pixels.dtype != np.float32
                or list(pixels.shape) != evidence["shape"]
                or not np.all(np.isfinite(pixels))
                or np.any(pixels < 0)
            ):
                raise ValueError(
                    "Reference repeats require finite nonnegative float32 HWC RGB"
                )
            clipped = np.clip(pixels, 0, 1)
            displays.append(
                np.where(
                    clipped <= 0.0031308,
                    12.92 * clipped,
                    1.055 * clipped ** (1 / 2.4) - 0.055,
                )
            )
    value = float(np.sqrt(np.mean((displays[0] - displays[1]) ** 2)))
    _agree(value, evidence["noise"])
    evidence["recomputed_value"] = value
    evidence["recomputed_passed"] = value <= 0.01
    return evidence


def _agree(value: float, noise: dict[str, Any]) -> None:
    tolerance = _POLICY["metadata_agreement"]
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value < 0
        or not math.isclose(
            value,
            noise["value"],
            rel_tol=tolerance["relative_tolerance"],
            abs_tol=tolerance["absolute_tolerance"],
        )
        or (value <= 0.01) is not noise["passed"]
    ):
        raise ValueError(
            "Recomputed reference noise disagrees with the generator evidence"
        )


def verify_reference_precision(
    root: Path,
    descriptor: dict[str, Any],
    *,
    expected_alias: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify all attempts and the first-passing choice without numerical imports.

    Source arrays, generator noise evidence, and the selector's recomputation are
    checksum-bound. This portable verifier checks those values and the selection
    rule; pixel recomputation belongs to :func:`measure_reference_noise`.

    Args:
        root: Experiment evidence trust boundary.
        descriptor: Checksummed selection JSON descriptor.
        expected_alias: Optional selected reference alias to verify against it.

    Returns:
        Verified selection, selected attempt, and the complete artifact closure.
        A valid capped failure has ``passed=False``; malformed proof raises.
    """
    selection = read_json_object(_verified(root, descriptor))
    if descriptor.get("kind") != "reference_precision_selection" or (
        selection.get("schema_version") != 1
        or selection.get("kind") != "reference_precision_selection/v1"
    ):
        raise ValueError("Unknown reference precision selection")
    policy = _read_policy(root, selection["policy"])
    if selection.get("policy_sha256") != stable_hash(policy):
        raise ValueError("Reference precision policy hash mismatch")
    attempts, artifacts = [], [descriptor, selection["policy"]]
    attempt_descriptors = selection["attempts"]
    if (
        not isinstance(attempt_descriptors, list)
        or not 1 <= len(attempt_descriptors) <= 2
    ):
        raise ValueError("Reference precision requires one or two ordered attempts")
    for index, item in enumerate(attempt_descriptors):
        if item.get("kind") != "reference_precision_attempt":
            raise ValueError("Invalid reference attempt descriptor")
        attempt = read_json_object(_verified(root, item))
        binding_path = _verified(root, attempt["binding_artifact"])
        expected = _noise_evidence(
            root, binding_path, policy["samples_per_pixel"][index]
        )
        expected.pop("arrays_path")
        if {
            key: value
            for key, value in attempt.items()
            if key not in {"recomputed_value", "recomputed_passed"}
        } != expected:
            raise ValueError("Reference precision attempt differs from its source")
        _agree(attempt["recomputed_value"], attempt["noise"])
        if attempt.get("recomputed_passed") is not (
            attempt["recomputed_value"] <= 0.01
        ):
            raise ValueError("Reference precision attempt has an inconsistent result")
        if attempts and (
            attempts[-1]["recomputed_passed"]
            or attempt["geometry_sha256"] != attempts[0]["geometry_sha256"]
            or attempt["physical_signature_sha256"]
            != attempts[0]["physical_signature_sha256"]
        ):
            raise ValueError(
                "Precision may change only spp after a valid failed noise check"
            )
        attempts.append(attempt)
        artifacts.extend(
            [item, attempt["binding_artifact"], *attempt["source_evidence"]]
        )
    selected = attempts[-1]
    passed = selected["recomputed_passed"]
    if not passed and len(attempts) != 2:
        raise ValueError("A failed baseline must attempt the declared precision cap")
    if (
        selection.get("selected_index") != len(attempts) - 1
        or selection.get("status") != ("passed" if passed else "noise_budget_exceeded")
        or selection.get("baseline_cache_key") != attempts[0]["cache_key"]
        or selection.get("selected_cache_key") != selected["cache_key"]
        or selection.get("selected_reference_id") != selected["reference_id"]
    ):
        raise ValueError(
            "Reference precision choice violates the frozen selection rule"
        )
    if expected_alias is not None:
        for key in ("cache_key", "reference_id", "reference_frame"):
            if expected_alias.get(key) != selected[key]:
                raise ValueError(
                    f"Selected reference alias differs from precision proof: {key}"
                )
        if (
            expected_alias.get("status") != "completed"
            or expected_alias.get("geometry_identity_sha256") != selected["cache_key"]
            or any(
                expected_alias.get("binding_artifact", {}).get(key)
                != selected["binding_artifact"][key]
                for key in ("path", "sha256")
            )
            or expected_alias.get("reference_precision") != descriptor
        ):
            raise ValueError(
                "Selected reference alias is not bound to its precision proof"
            )
    return {
        "selection": selection,
        "selected": selected,
        "passed": passed,
        "artifacts": artifacts,
    }


def select_reference(
    root: Path,
    directory: Path,
    policy_artifact: dict[str, Any],
    acquire: Callable[[dict[str, int]], tuple[dict[str, Any] | None, dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Acquire the fixed ladder using reference noise only and retain every attempt.

    Args:
        root: Experiment trust boundary.
        directory: Evaluation-local selection directory.
        policy_artifact: Policy frozen before acquisition begins.
        acquire: Existing reference cache/renderer callback accepting only spp,
            seed and threads. Candidate images and quality scores are excluded.

    Returns:
        Selected reference and one alias, with the precision proof attached.
        A failed cap remains a diagnostic reference; corrupt evidence raises
        after preserving a failure record, without trying another rung.
    """
    root = Path(root).resolve()
    directory = _inside(root, directory)
    policy = _read_policy(root, policy_artifact)
    directory.mkdir(parents=True, exist_ok=True)
    attempts = []
    alias = None
    spp = 4096
    try:
        ref, alias = acquire({"spp": 4096, "seed": 0, "threads": 8})
        if ref is None or alias.get("status") != "completed":
            raise ValueError(
                "Baseline reference is unavailable; precision escalation is forbidden"
            )
        key = alias["cache_key"]
        path = directory / f"{key}.json"
        if path.exists():
            choice = _artifact(root, path, "reference_precision_selection")
            verified = verify_reference_precision(root, choice)
            if (
                verified["selection"]["policy"] != policy_artifact
                or verified["selection"]["baseline_cache_key"] != key
            ):
                raise ValueError(
                    "Existing precision choice has different frozen inputs"
                )
            spp = verified["selected"]["spp"]
            if spp != 4096:
                ref, alias = acquire({"spp": spp, "seed": 0, "threads": 8})
        else:
            for index, spp in enumerate(policy["samples_per_pixel"]):
                if index:
                    ref, alias = acquire({"spp": spp, "seed": 0, "threads": 8})
                if ref is None or alias.get("status") != "completed":
                    raise ValueError(
                        "Reference attempt unavailable; no further escalation"
                    )
                attempt = measure_reference_noise(
                    root, _verified(root, alias["binding_artifact"]), spp=spp
                )
                attempt_path = directory / f"{key}.attempt_{index}.json"
                if attempt_path.exists():
                    raise ValueError(
                        "Incomplete prior precision selection; refusing replacement"
                    )
                write_json(attempt_path, attempt)
                attempts.append(
                    _artifact(root, attempt_path, "reference_precision_attempt")
                )
                if attempt["recomputed_passed"]:
                    break
            selection = {
                "schema_version": 1,
                "kind": "reference_precision_selection/v1",
                "policy": policy_artifact,
                "policy_sha256": stable_hash(policy),
                "baseline_cache_key": key,
                "attempts": attempts,
                "selected_index": len(attempts) - 1,
                "selected_cache_key": attempt["cache_key"],
                "selected_reference_id": attempt["reference_id"],
                "status": (
                    "passed"
                    if attempt["recomputed_passed"]
                    else "noise_budget_exceeded"
                ),
            }
            write_json(path, selection)
            choice = _artifact(root, path, "reference_precision_selection")
        alias["reference_precision"] = choice
        verified = verify_reference_precision(root, choice, expected_alias=alias)
        for artifact in verified["artifacts"]:
            ref["artifacts"].append(
                {**artifact, "path": str(_inside(root, artifact["path"]))}
            )
        return ref, alias
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        failure_path = (
            directory
            / f"failure_{len(list(directory.glob('failure_*.json'))):04d}.json"
        )
        write_json(
            failure_path,
            {
                "schema_version": 1,
                "kind": "reference_precision_failure/v1",
                "status": "invalid_evidence",
                "policy": policy_artifact,
                "attempt_spp": spp,
                "completed_attempts": attempts,
                "last_alias": alias,
                "reason": f"{type(error).__name__}: {error}",
                "escalation_stopped": True,
            },
        )
        write_json(
            failure_path.with_suffix(".sha256.json"),
            _artifact(root, failure_path, "reference_precision_failure"),
        )
        raise ValueError(
            f"Reference precision stopped; retained {failure_path}: {error}"
        ) from error


def precision_quality_reasons(root: Path, record: dict[str, Any]) -> list[str]:
    """Return portable report rejection reasons for an opted-in quality record."""
    artifacts = record.get("artifacts", [])
    marked = (
        any(
            isinstance(item, dict)
            and str(item.get("kind", "")).startswith("reference_precision")
            for item in artifacts
        )
        if isinstance(artifacts, list)
        else False
    )
    context = record.get("reference_precision")
    if context is None and not marked:
        return []
    try:
        if not isinstance(context, dict):
            raise ValueError("Missing precision policy/selection binding")
        _read_policy(root, context["policy"])
        inputs = read_json_object(_verified(root, context["evaluation_inputs"]))
        if inputs.get("reference_precision_policy") != context["policy"] or inputs.get(
            "settings"
        ) != {"spp": 4096, "seed": 0, "threads": 8}:
            raise ValueError("Precision policy was not frozen in the evaluation inputs")
        from scripts.benchmark.core.artifacts import read_json

        aliases = read_json(_verified(root, context["bindings"]))
        if (
            not isinstance(aliases, list)
            or not aliases
            or len(aliases) != record.get("coverage", {}).get("evaluated_samples")
        ):
            raise ValueError(
                "Precision reference aliases do not cover every evaluated sample"
            )
        closure = [context["policy"], context["evaluation_inputs"], context["bindings"]]
        references, choices, failed = set(), {}, False
        for alias in aliases:
            descriptor = alias["reference_precision"]
            if descriptor.get("kind") != "reference_precision_selection":
                raise ValueError("Alias requires a reference precision selection")
            identity = (descriptor["path"], descriptor["sha256"])
            if identity not in choices:
                choices[identity] = verify_reference_precision(root, descriptor)
            verified = choices[identity]
            selected = verified["selected"]
            if verified["selection"]["policy"] != context["policy"] or any(
                alias.get(key) != selected[key]
                for key in ("cache_key", "reference_id", "reference_frame")
            ):
                raise ValueError(
                    "Reference alias does not match its selected precision rung"
                )
            if (
                alias.get("status") != "completed"
                or alias.get("geometry_identity_sha256") != selected["cache_key"]
                or any(
                    alias.get("binding_artifact", {}).get(key)
                    != selected["binding_artifact"][key]
                    for key in ("path", "sha256")
                )
            ):
                raise ValueError(
                    "Selected reference cache binding differs from its proof"
                )
            references.add(selected["reference_id"])
            failed |= not verified["passed"]
            closure.extend(verified["artifacts"])
        expected_id = (
            next(iter(references))
            if len(references) == 1
            else stable_hash(sorted(references))
        )
        if (
            record.get("reference_ids") != sorted(references)
            or record.get("reference_id") != expected_id
        ):
            raise ValueError(
                "Quality reference population differs from precision choices"
            )
        retained = {(item["path"], item["sha256"]) for item in artifacts}
        if any((item["path"], item["sha256"]) not in retained for item in closure):
            raise ValueError("Quality proof omits precision attempt source artifacts")
        return ["quality_reference_precision_noise"] if failed else []
    except (OSError, KeyError, TypeError, ValueError, AttributeError) as error:
        return [
            "quality_reference_precision",
            "quality_reference_precision:" + str(error),
        ]
