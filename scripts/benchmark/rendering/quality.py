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

"""R-01/R-02 image quality with independent evidence and conservative gates.

The frozen v1 thresholds are benchmark acceptance policy, not empirical claims:
SSIM >= .95, LPIPS-Squeeze v0.1 <= .05, optical-depth RMSE <= .01 m,
valid-depth coverage >= .99, macro instance IoU >= .98 and silhouette IoU
>= .99. The background ID is zero and is excluded from macro instance IoU.
Temporal gates use all 99 pairs of a fixed 100-frame sequence: worst-pair
static-region residual flicker <= .02, reprojection RGB MAE <= .03, depth
consistency RMSE <= .01 m, and worst-frame silhouette IoU >= .99.

SSIM and LPIPS consume display-sRGB in [0,1]. Scene-linear inputs receive the
standard sRGB transfer after fixed [0,1] clipping; there is no fitted exposure,
colour correction, registration, relabeling, or per-image normalization.
Implementations follow https://github.com/richzhang/PerceptualSimilarity and
https://scikit-image.org/docs/stable/api/skimage.metrics.html.
LPIPS never downloads weights or substitutes random features. Missing evidence,
dependencies, pretrained weights, or photometric calibration cannot qualify.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.metadata
from pathlib import Path
from typing import Any

import numpy as np

from scripts.benchmark.core.artifacts import stable_hash, write_json
from scripts.benchmark.rendering.calibration import verify_calibration
from scripts.benchmark.rendering.quality_protocol import (
    _LPIPS_BACKBONE_SHA256,
    _LPIPS_LINEAR_SHA256,
    default_protocol,
    protocol_for_modalities,
    protocol_requirements,
    validate_protocol,
)
from scripts.benchmark.rendering.reference import (
    DEPTH_SEMANTICS,
    SEGMENTATION_SEMANTICS,
)

__all__ = [
    "default_protocol",
    "protocol_for_modalities",
    "evaluate_frame",
    "evaluate_sequence",
]


def _protocol(protocol: dict[str, Any] | None) -> dict[str, Any]:
    return validate_protocol(protocol)


def _check(passed: bool | None, reason: str = "", **evidence: Any) -> dict[str, Any]:
    return {
        "status": "missing" if passed is None else "passed" if passed else "failed",
        "reason": reason or None,
        **evidence,
    }


def _metric(
    value: float | None,
    rule: list[Any],
    denominator: int,
    reason: str = "",
    **evidence: Any,
) -> dict[str, Any]:
    comparison, threshold, unit = rule
    if value is not None and not np.isfinite(value):
        value, reason = None, "Metric calculation returned a non-finite value"
    passed = (
        None
        if value is None
        else (value >= threshold if comparison == "ge" else value <= threshold)
    )
    return {
        "value": value,
        "threshold": threshold,
        "comparison": comparison,
        "unit": unit,
        "denominator": denominator,
        **_check(passed, reason, **evidence),
    }


def _finish(result: dict[str, Any]) -> dict[str, Any]:
    for category, field in (
        ("checks", "required_checks"),
        ("metrics", "required_metrics"),
    ):
        for name in result.get(field, []):
            result[category].setdefault(
                name, _check(None, "Required protocol evidence is missing")
            )
        for name, item in result[category].items():
            item["required"] = name in result.get(field, result[category])
    statuses = {
        name: item
        for name, item in {**result["checks"], **result["metrics"]}.items()
        if item.get("required", True)
    }
    failed = [name for name, value in statuses.items() if value["status"] == "failed"]
    missing = [
        name
        for name, value in statuses.items()
        if value["status"] not in {"passed", "failed"}
    ]
    passed = bool(statuses) and not failed and not missing
    result["quality_status"] = "qualified" if passed else "not_qualified"
    result["qualification"] = {
        "passed": passed,
        "failed_checks": failed,
        "missing_checks": missing,
        "assessed_modalities": result.get(
            "assessed_modalities", ["rgb", "depth", "seg"]
        ),
    }
    result["modality_quality"] = {
        name: {
            "status": "not_qualified",
            "reason": "No frozen normal-quality threshold; only output shape/type/finite contents checked",
        }
        for name in result.get("unassessed_modalities", [])
    }
    return result


def _display_rgb(frame: dict[str, Any]) -> np.ndarray:
    value = np.asarray(frame["rgb"])
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError("RGB must have shape (height, width, 3)")
    encoding = frame.get("metadata", {}).get("rgb_encoding")
    if encoding == "srgb_uint8":
        if value.dtype != np.uint8:
            raise ValueError("srgb_uint8 requires uint8 RGB storage")
        return value.astype(np.float32) / 255
    value = value.astype(np.float32)
    if not np.isfinite(value).all():
        raise ValueError("RGB contains non-finite values")
    if encoding in ("srgb", "srgb_float"):
        if value.min() < 0 or value.max() > 1:
            raise ValueError("Display-sRGB values must lie in [0,1]")
        return value
    if encoding == "linear_rec709":
        if value.min() < 0:
            raise ValueError("Scene-linear RGB cannot have negative radiance")
        value = np.clip(value, 0, 1)
        return np.where(
            value <= 0.0031308, 12.92 * value, 1.055 * np.power(value, 1 / 2.4) - 0.055
        )
    raise ValueError(
        "RGB encoding is absent or unsupported; cannot infer it from dtype"
    )


def _depth(frame: dict[str, Any]) -> np.ndarray:
    meta = frame.get("metadata", {})
    if meta.get("depth_semantics") not in (
        DEPTH_SEMANTICS,
        DEPTH_SEMANTICS + "; invalid_or_beyond_far_is_infinity",
    ):
        raise ValueError("Depth must explicitly use optical-axis metres")
    depth = np.asarray(frame["depth"], dtype=float)
    if depth.ndim != 2:
        raise ValueError("Depth must have a 2D shape")
    if np.isnan(depth).any() or np.isneginf(depth).any() or np.any(depth <= 0):
        raise ValueError("Depth must be positive metres or +infinity for background")
    return depth


def _geometry(frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    meta = frame.get("metadata", {})
    depth = _depth(frame)
    if meta.get("segmentation_semantics") not in (
        SEGMENTATION_SEMANTICS,
        "canonical_instance_id; background=0",
    ):
        raise ValueError(
            "Segmentation must explicitly use canonical IDs and background zero"
        )
    seg = np.asarray(frame["seg"] if "seg" in frame else frame["segmentation"])
    if depth.ndim != 2 or seg.shape != depth.shape:
        raise ValueError("Depth and segmentation must have matching 2D shapes")
    if not np.issubdtype(seg.dtype, np.integer) or np.any(seg < 0):
        raise ValueError("Canonical instance labels must be nonnegative integers")
    ids = meta.get("instance_ids")
    if ids is None:
        raise ValueError("Canonical instance-ID inventory is missing")
    if not set(np.unique(seg)).issubset({0, *ids}):
        raise ValueError("Segmentation contains IDs outside the frozen scene inventory")
    return depth, seg


def _requested_outputs(
    candidate: dict[str, Any], requested: list[str]
) -> dict[str, Any]:
    missing = [
        name
        for name in requested
        if name not in candidate and not (name == "seg" and "segmentation" in candidate)
    ]
    if missing:
        return _check(None, f"Requested output arrays are missing: {missing}")
    try:
        rgb = _display_rgb(candidate)
        for name in requested:
            if name == "rgb":
                continue
            array = np.asarray(
                candidate["segmentation"]
                if name == "seg" and name not in candidate
                else candidate[name]
            )
            expected = rgb.shape if name == "normals" else rgb.shape[:2]
            if array.shape != expected:
                raise ValueError(f"Requested {name} output shape does not match RGB")
            if name == "normals" and (
                array.dtype != np.float32 or not np.isfinite(array).all()
            ):
                raise ValueError(
                    "Requested normals must be finite float32 three-channel arrays"
                )
        return _check(True, requested_modalities=requested)
    except (KeyError, TypeError, ValueError) as error:
        return _check(False, str(error))


def _depth_metrics(
    depth: np.ndarray, reference: np.ndarray, policy: dict[str, Any]
) -> dict[str, Any]:
    if depth.shape != reference.shape:
        raise ValueError("Candidate/reference depth dimensions differ")
    valid, reference_valid = np.isfinite(depth), np.isfinite(reference)
    common = valid & reference_valid
    count = int(reference_valid.sum())
    coverage = float(common.sum() / count) if count else None
    value = (
        float(np.sqrt(np.mean((depth[common] - reference[common]) ** 2)))
        if common.any()
        else None
    )
    return {
        "depth_rmse_m": _metric(
            value,
            policy["frame_thresholds"]["depth_rmse_m"],
            int(common.sum()),
            "No common valid reference depth" if value is None else "",
        ),
        "depth_valid_coverage": _metric(
            coverage,
            policy["frame_thresholds"]["depth_valid_coverage"],
            count,
            "Reference has no visible geometry" if not count else "",
        ),
    }


def _camera(meta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    camera = meta["camera"]
    if camera.get("convention") != "opencv":
        raise ValueError("Camera convention must be explicitly opencv")
    k = np.asarray(camera["intrinsics"], dtype=float)
    pose = np.asarray(camera["world_from_camera"], dtype=float)
    if (
        k.shape != (3, 3)
        or pose.shape != (4, 4)
        or not np.isfinite(k).all()
        or not np.isfinite(pose).all()
    ):
        raise ValueError("Invalid camera matrices")
    if (
        k[0, 0] <= 0
        or k[1, 1] <= 0
        or not np.allclose(k[2], [0, 0, 1])
        or not np.allclose([k[0, 1], k[1, 0]], 0)
    ):
        raise ValueError("Invalid pinhole intrinsics")
    if (
        not np.allclose(pose[3], [0, 0, 0, 1])
        or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-5)
    ):
        raise ValueError("Camera pose must be a rigid transform")
    return k, pose


def _reference_checks(
    candidate: dict[str, Any], reference: dict[str, Any] | None, policy: dict[str, Any]
) -> dict[str, Any]:
    if reference is None:
        return {
            "independent_reference": _check(
                None, "Independent reference is absent; consistency only"
            )
        }
    meta, refmeta = candidate.get("metadata", {}), reference.get("metadata", {})
    provenance = reference.get("reference_provenance", {})
    identity = {
        key: value for key, value in provenance.items() if key != "reference_id"
    }
    valid = (
        provenance.get("kind") == "independent_offline"
        and provenance.get("renderer") in policy["independent_renderers"]
        and provenance.get("renderer") != meta.get("backend")
        and provenance.get("assets_verified") is True
        and bool(provenance.get("implementation_sha256"))
        and provenance.get("reference_id") == stable_hash(identity)
        and provenance.get("source_scene_sha256") == refmeta.get("scene_sha256")
    )
    checks = {
        "independent_reference": _check(
            valid,
            (
                "Reference must have verified independent renderer and asset provenance"
                if not valid
                else ""
            ),
        )
    }
    artifacts = reference.get("artifacts", [])
    intact = _evidence_intact(artifacts)
    checks["reference_evidence"] = _check(
        True if intact else None,
        (
            "Reference requires accessible checksum-verified frame evidence"
            if not intact
            else ""
        ),
    )
    matched_scene = bool(meta.get("scene_sha256")) and meta.get(
        "scene_sha256"
    ) == refmeta.get("scene_sha256")
    checks["scene_binding"] = _check(
        matched_scene,
        "Candidate/reference scene fingerprints differ" if not matched_scene else "",
    )
    try:
        k, pose = _camera(meta)
        rk, rpose = _camera(refmeta)
        matched = np.allclose(k, rk, atol=1e-5, rtol=0) and np.allclose(
            pose, rpose, atol=1e-5, rtol=0
        )
        checks["camera_binding"] = _check(
            bool(matched), "Camera projection or pose differs" if not matched else ""
        )
    except (ValueError, KeyError, TypeError) as error:
        checks["camera_binding"] = _check(None, str(error))
    calibration = meta.get("photometric_calibration")
    checks["photometric_calibration"] = verify_calibration(
        calibration, meta, reference, policy
    )
    noise = reference.get("reference_validation", {})
    noise_value = noise.get("value")
    measured_noise = (
        noise.get("metric") == "independent_seed_display_srgb_rmse"
        and isinstance(noise_value, (float, int))
        and np.isfinite(noise_value)
        and len(set(noise.get("seeds", []))) == 2
    )
    checks["reference_noise_floor"] = _check(
        noise_value <= policy["reference_noise_rmse_max"] if measured_noise else None,
        (
            "Independent repeat render must establish the reference RGB noise floor"
            if not measured_noise
            else ""
        ),
        evidence=noise,
    )
    return checks


def _evidence_intact(artifacts: list[dict[str, Any]]) -> bool:
    if not artifacts:
        return False
    for artifact in artifacts:
        try:
            path = Path(artifact["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
                return False
        except (OSError, TypeError, KeyError):
            return False
    return True


@lru_cache(maxsize=2)
def _lpips_model(cache_dir: str) -> tuple[Any, dict[str, Any]]:
    import torch
    import lpips

    path = Path(cache_dir) / "checkpoints/squeezenet1_1-b8a52dc0.pth"
    if not path.is_file():
        raise FileNotFoundError(
            "Pretrained SqueezeNet 1.1 backbone is not cached; evaluation never downloads weights"
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != _LPIPS_BACKBONE_SHA256:
        raise ValueError(
            "Pretrained SqueezeNet backbone checksum does not match its official hash"
        )
    head = Path(lpips.__file__).parent / "weights/v0.1/squeeze.pth"
    if not head.is_file():
        raise FileNotFoundError(
            "Learned LPIPS v0.1 SqueezeNet linear weights are missing"
        )
    if hashlib.sha256(head.read_bytes()).hexdigest() != _LPIPS_LINEAR_SHA256:
        raise ValueError(
            "Learned LPIPS linear weights do not match the frozen v0.1 model"
        )
    model = (
        lpips.LPIPS(
            net="squeeze",
            version="0.1",
            pnet_rand=False,
            pretrained=True,
            verbose=False,
        )
        .cpu()
        .eval()
    )
    return model, {
        "package_version": importlib.metadata.version("lpips"),
        "network": "squeeze",
        "version": "0.1",
        "device": "cpu",
        "backbone_sha256": digest,
        "linear_weights_sha256": hashlib.sha256(head.read_bytes()).hexdigest(),
    }


def _lpips(first: np.ndarray, second: np.ndarray) -> tuple[float, dict[str, Any]]:
    import torch

    if min(first.shape[:2]) < 64:
        raise ValueError(
            "LPIPS protocol requires images at least 64 pixels on each axis"
        )
    model, evidence = _lpips_model(torch.hub.get_dir())
    a = torch.from_numpy(np.ascontiguousarray(first.transpose(2, 0, 1))).unsqueeze(0)
    b = torch.from_numpy(np.ascontiguousarray(second.transpose(2, 0, 1))).unsqueeze(0)
    with torch.inference_mode():
        value = float(model(a * 2 - 1, b * 2 - 1).item())
    return value, evidence


def _artifact(path: Path, kind: str) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "kind": kind,
    }


def _save_errors(
    directory: Path,
    first: np.ndarray,
    second: np.ndarray,
    depth: np.ndarray | None,
    reference_depth: np.ndarray | None,
    seg: np.ndarray | None,
    reference_seg: np.ndarray | None,
    policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    rgb_error = np.mean(np.abs(first - second), axis=-1)
    errors = {"rgb_mae": rgb_error}
    if depth is not None and reference_depth is not None:
        depth_error = np.full(depth.shape, np.nan, dtype=np.float32)
        common = np.isfinite(depth) & np.isfinite(reference_depth)
        depth_error[common] = np.abs(depth[common] - reference_depth[common])
        errors.update(
            depth_absolute_m=depth_error,
            reference_valid_depth=np.isfinite(reference_depth),
        )
    if seg is not None and reference_seg is not None:
        errors["instance_mismatch"] = seg != reference_seg
    maps = directory / "error_maps.npz"
    np.savez_compressed(maps, **errors)
    artifacts = [_artifact(maps, "error_maps")]
    images = {
        "candidate": first,
        "reference": second,
        "rgb_error": np.repeat(rgb_error[..., None], 3, axis=-1),
    }
    if seg is not None and reference_seg is not None:
        images["instance_error"] = np.repeat(
            (seg != reference_seg)[..., None], 3, axis=-1
        )
    for name, array in images.items():
        path = directory / (name + ".png")
        Image.fromarray(np.round(np.clip(array, 0, 1) * 255).astype(np.uint8)).save(
            path
        )
        artifacts.append(_artifact(path, name))
    crops = []
    if reference_seg is None:
        height, width = first.shape[:2]
        x0, x1, y0, y1 = width // 4, 3 * width // 4, height // 4, 3 * height // 4
        x1, y1 = max(x0 + 1, x1), max(y0 + 1, y1)
        crop = np.concatenate([first[y0:y1, x0:x1], second[y0:y1, x0:x1]], axis=1)
        path = directory / "crop_central.png"
        Image.fromarray(np.round(crop * 255).astype(np.uint8)).save(path)
        artifacts.append(_artifact(path, "candidate_reference_crop"))
        crops.append(
            {
                "region": "central_half",
                "box_xyxy": [x0, y0, x1, y1],
                "path": str(path.resolve()),
            }
        )
        return artifacts, crops
    ids = sorted(
        (int(v) for v in np.unique(reference_seg) if v),
        key=lambda label: float(rgb_error[reference_seg == label].mean()),
        reverse=True,
    )
    for index, label in enumerate(ids[: policy["crop_count"]]):
        ys, xs = np.where(reference_seg == label)
        x0, x1 = max(0, int(xs.min()) - 4), min(seg.shape[1], int(xs.max()) + 5)
        y0, y1 = max(0, int(ys.min()) - 4), min(seg.shape[0], int(ys.max()) + 5)
        crop = np.concatenate([first[y0:y1, x0:x1], second[y0:y1, x0:x1]], axis=1)
        path = directory / f"crop_{index:02d}_instance_{label}.png"
        Image.fromarray(np.round(crop * 255).astype(np.uint8)).save(path)
        artifacts.append(_artifact(path, "candidate_reference_crop"))
        crops.append(
            {
                "instance_id": label,
                "box_xyxy": [x0, y0, x1, y1],
                "rgb_mae": float(rgb_error[reference_seg == label].mean()),
                "path": str(path.resolve()),
            }
        )
    return artifacts, crops


def evaluate_frame(
    candidate: dict[str, Any],
    reference: dict[str, Any] | None,
    protocol: dict[str, Any] | None = None,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one RGB/depth/instance frame without inventing missing evidence.

    Args:
        candidate: ``rgb``, ``depth``, ``seg`` arrays and ``metadata`` with
            explicit encodings, canonical IDs, camera matrices and scene hash.
        reference: Independent frame from ``load_reference_frame``, or ``None``.
        protocol: Frozen full or explicitly selected modality policy, or ``None``
            for the unchanged full :func:`default_protocol`.
        output_dir: Optional error maps, visual crops, and JSON evidence directory.

    Returns:
        JSON-safe metrics with denominators, per-check status and qualification.
        ``not_qualified`` is independent of benchmark execution success.
    """
    policy = _protocol(protocol)
    requirements = protocol_requirements(policy["id"])
    meta = candidate.get("metadata", {})
    result = {
        "schema_version": 1,
        "protocol_id": policy["id"],
        "protocol_sha256": stable_hash(policy),
        "scene_sha256": meta.get("scene_sha256"),
        "view_sha256": meta.get("view_sha256", stable_hash(meta.get("camera", {}))),
        "profile_sha256": meta.get("profile_sha256"),
        "reference_id": (reference or {})
        .get("reference_provenance", {})
        .get("reference_id"),
        "checks": _reference_checks(candidate, reference, policy),
        "metrics": {},
        "artifacts": list((reference or {}).get("artifacts", [])),
        "crops": [],
    }
    result.update(requirements)
    result["checks"]["requested_outputs"] = _requested_outputs(
        candidate, requirements["requested_modalities"]
    )
    calibration = meta.get("photometric_calibration")
    if isinstance(calibration, dict) and _evidence_intact(
        calibration.get("evidence", [])
    ):
        result["artifacts"].extend(calibration["evidence"])
    metrics = result["metrics"]
    for name, rule in policy["frame_thresholds"].items():
        metrics[name] = _metric(None, rule, 0, "Independent reference is absent")
        if name not in requirements["required_metrics"]:
            metrics[name].update(
                status="not_requested", reason="modality_not_requested", required=False
            )
    if reference is None:
        result["consistency_only"] = True
        _finish(result)
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            write_json(Path(output_dir) / "quality_evidence.json", result)
        return result
    try:
        first, second = _display_rgb(candidate), _display_rgb(reference)
        if first.shape != second.shape:
            raise ValueError("Candidate/reference RGB dimensions differ")
        result["checks"]["rgb_semantics"] = _check(True)
    except (KeyError, TypeError, ValueError) as error:
        first = second = None
        result["checks"]["rgb_semantics"] = _check(False, str(error))
        for name in ("ssim", "lpips"):
            metrics[name] = _metric(
                None, policy["frame_thresholds"][name], 0, str(error)
            )
    if first is not None:
        try:
            from skimage.metrics import structural_similarity

            value = structural_similarity(
                first,
                second,
                channel_axis=-1,
                data_range=1,
                gaussian_weights=True,
                sigma=1.5,
                win_size=11,
                use_sample_covariance=False,
            )
            metrics["ssim"] = _metric(
                float(value),
                policy["frame_thresholds"]["ssim"],
                first.shape[0] * first.shape[1],
                implementation="skimage.metrics.structural_similarity",
                **policy["ssim"],
            )
        except (ImportError, ValueError) as error:
            metrics["ssim"] = _metric(
                None, policy["frame_thresholds"]["ssim"], 0, str(error)
            )
        try:
            value, evidence = _lpips(first, second)
            metrics["lpips"] = _metric(
                value,
                policy["frame_thresholds"]["lpips"],
                1,
                population="one image pair",
                evidence=evidence,
            )
        except (ImportError, OSError, ValueError, RuntimeError) as error:
            metrics["lpips"] = _metric(
                None, policy["frame_thresholds"]["lpips"], 0, str(error)
            )
    if "seg" not in requirements["requested_modalities"]:
        depth = rdepth = None
        if "depth" in requirements["requested_modalities"]:
            for frame, name in (
                (candidate, "depth_semantics"),
                (reference, "reference_depth_semantics"),
            ):
                try:
                    _depth(frame)
                    result["checks"][name] = _check(True)
                except (KeyError, ValueError, TypeError) as error:
                    result["checks"][name] = _check(False, str(error))
            try:
                depth, rdepth = _depth(candidate), _depth(reference)
                if first is not None and depth.shape != first.shape[:2]:
                    raise ValueError("RGB/depth dimensions differ")
                metrics.update(_depth_metrics(depth, rdepth, policy))
            except (KeyError, ValueError, TypeError) as error:
                depth = rdepth = None
                for name in ("depth_rmse_m", "depth_valid_coverage"):
                    metrics[name] = _metric(
                        None, policy["frame_thresholds"][name], 0, str(error)
                    )
        if output_dir is not None and first is not None:
            artifacts, crops = _save_errors(
                Path(output_dir), first, second, depth, rdepth, None, None, policy
            )
            result["artifacts"].extend(artifacts)
            result["crops"] = crops
        _finish(result)
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            write_json(Path(output_dir) / "quality_evidence.json", result)
        return result
    try:
        depth, seg = _geometry(candidate)
        rdepth, rseg = _geometry(reference)
        if depth.shape != rdepth.shape or (
            first is not None and depth.shape != first.shape[:2]
        ):
            raise ValueError("RGB/depth/segmentation dimensions differ")
        if set(meta["instance_ids"]) != set(reference["metadata"]["instance_ids"]):
            raise ValueError("Candidate/reference canonical-ID inventories differ")
        result["checks"]["geometry_semantics"] = _check(True)
        valid, rvalid = np.isfinite(depth), np.isfinite(rdepth)
        reference_integrity = bool(np.all(rvalid == (rseg != 0)))
        result["checks"]["reference_geometry_integrity"] = _check(
            reference_integrity,
            (
                "Reference depth validity and foreground labels disagree"
                if not reference_integrity
                else ""
            ),
        )
        metrics.update(_depth_metrics(depth, rdepth, policy))
        per_instance = []
        for label in sorted((set(np.unique(seg)) | set(np.unique(rseg))) - {0}):
            intersection = int(((seg == label) & (rseg == label)).sum())
            union = int(((seg == label) | (rseg == label)).sum())
            per_instance.append(
                {
                    "instance_id": int(label),
                    "intersection": intersection,
                    "union": union,
                    "iou": intersection / union,
                }
            )
        iou = (
            float(np.mean([item["iou"] for item in per_instance]))
            if per_instance
            else None
        )
        metrics["instance_iou"] = _metric(
            iou,
            policy["frame_thresholds"]["instance_iou"],
            len(per_instance),
            "No foreground instances" if not per_instance else "",
            per_instance=per_instance,
            background_included=False,
        )
        union = int(((seg != 0) | (rseg != 0)).sum())
        silhouette = float(((seg != 0) & (rseg != 0)).sum() / union) if union else None
        metrics["silhouette_iou"] = _metric(
            silhouette,
            policy["frame_thresholds"]["silhouette_iou"],
            union,
            "No foreground silhouette" if not union else "",
        )
        metrics["depth_silhouette_consistency"] = _metric(
            float(np.mean(valid == (seg != 0))),
            policy["frame_thresholds"]["depth_silhouette_consistency"],
            seg.size,
        )
        if output_dir is not None and first is not None:
            artifacts, crops = _save_errors(
                Path(output_dir), first, second, depth, rdepth, seg, rseg, policy
            )
            result["artifacts"].extend(artifacts)
            result["crops"] = crops
    except (KeyError, ValueError, TypeError) as error:
        result["checks"]["geometry_semantics"] = _check(False, str(error))
        for name in policy["frame_thresholds"]:
            if name not in ("ssim", "lpips"):
                metrics[name] = _metric(
                    None, policy["frame_thresholds"][name], 0, str(error)
                )
    _finish(result)
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        write_json(Path(output_dir) / "quality_evidence.json", result)
    return result


def _correspondences(
    previous: dict[str, Any], current: dict[str, Any], policy: dict[str, Any]
) -> tuple[np.ndarray, ...]:
    depth, seg = _geometry(previous)
    next_depth, next_seg = _geometry(current)
    k, pose = _camera(previous["metadata"])
    nk, npose = _camera(current["metadata"])
    rows, cols = np.where(np.isfinite(depth) & (seg != 0))
    points = np.column_stack(
        [
            (cols + 0.5 - k[0, 2]) / k[0, 0],
            (rows + 0.5 - k[1, 2]) / k[1, 1],
            np.ones(len(rows)),
        ]
    )
    world_directions = points @ pose[:3, :3].T
    points *= depth[rows, cols, None]
    world = points @ pose[:3, :3].T + pose[:3, 3]
    labels = seg[rows, cols]
    old_transforms = previous["metadata"].get("world_from_instances", {})
    new_transforms = current["metadata"].get("world_from_instances", {})
    moving = set(previous["metadata"].get("moving_instance_ids", [])) | set(
        current["metadata"].get("moving_instance_ids", [])
    )
    for label in moving:
        if str(label) not in old_transforms or str(label) not in new_transforms:
            raise ValueError(
                "Moving instances require per-frame world transforms for reprojection"
            )
        transform = np.asarray(new_transforms[str(label)]) @ np.linalg.inv(
            old_transforms[str(label)]
        )
        selected = labels == label
        world[selected] = world[selected] @ transform[:3, :3].T + transform[:3, 3]
        world_directions[selected] = world_directions[selected] @ transform[:3, :3].T
    target = (world - npose[:3, 3]) @ npose[:3, :3]
    depth_scale = (world_directions @ npose[:3, :3])[:, 2]
    z = target[:, 2]
    safe_z = np.where(z > 0, z, 1)
    x = np.rint(target[:, 0] / safe_z * nk[0, 0] + nk[0, 2] - 0.5).astype(int)
    y = np.rint(target[:, 1] / safe_z * nk[1, 1] + nk[1, 2] - 0.5).astype(int)
    inside = (
        (z > 0)
        & (x >= 0)
        & (y >= 0)
        & (x < next_depth.shape[1])
        & (y < next_depth.shape[0])
    )
    source = np.flatnonzero(inside)
    target_z = next_depth[y[source], x[source]]
    tolerance = np.maximum(
        policy["occlusion_absolute_m"], policy["occlusion_relative"] * z[source]
    )
    visible = np.isfinite(target_z) & (np.abs(target_z - z[source]) <= tolerance)
    visible &= next_seg[y[source], x[source]] == labels[source]
    selected = source[visible]
    static = ~np.isin(labels[selected], list(moving))
    return (
        rows[selected],
        cols[selected],
        y[selected],
        x[selected],
        static,
        z[selected],
        depth_scale[selected],
    )


def evaluate_sequence(
    candidates: list[dict[str, Any]],
    references: list[dict[str, Any]] | None,
    protocol: dict[str, Any] | None = None,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate fixed-length temporal quality using reference-based visibility.

    Args:
        candidates: Exactly 100 ordered frame dictionaries with ``frame_id``
            (0..99), camera matrices, and optional moving-instance transforms.
        references: Corresponding independent frames, or ``None`` for explicitly
            unqualified consistency diagnostics using candidate visibility.
        protocol: Full frozen policy from :func:`default_protocol`, or an explicit
            full temporal30-v1 selected before rendering. Timestamps are preserved.
        output_dir: Optional directory for pair evidence and the aggregate JSON.

    Returns:
        Worst-pair temporal metrics, every pair's denominator, frame quality
        evidence, and a qualification requiring every required frame/check.
        Reference depth/IDs reject occlusions and disocclusions before scoring;
        reference colour differences remove legitimate view-dependent shading.
    """
    policy = _protocol(protocol)
    requirements = protocol_requirements(policy["id"], kind="sequence")
    result = {
        "schema_version": 1,
        "protocol_id": policy["id"],
        "protocol_sha256": stable_hash(policy),
        "checks": {},
        "metrics": {},
        "frames": [],
        "pairs": [],
        "artifacts": [],
        "consistency_only": references is None,
    }
    result.update(requirements)
    metadata = [frame.get("metadata", {}) for frame in candidates]
    reference_ids = sorted(
        {
            frame.get("reference_provenance", {}).get("reference_id")
            for frame in (references or [])
            if frame.get("reference_provenance", {}).get("reference_id")
        }
    )
    result.update(
        scene_sha256=metadata[0].get("scene_sha256") if metadata else None,
        view_sha256=stable_hash([meta.get("camera", {}) for meta in metadata]),
        profile_sha256=metadata[0].get("profile_sha256") if metadata else None,
        reference_id=(
            (
                reference_ids[0]
                if len(reference_ids) == 1
                else stable_hash(reference_ids)
            )
            if reference_ids
            else None
        ),
        reference_ids=reference_ids,
    )
    checks = result["checks"]
    fixed_profile = (
        bool(metadata)
        and bool(result["profile_sha256"])
        and all(
            meta.get("profile_sha256") == result["profile_sha256"]
            and meta.get("scene_sha256") == result["scene_sha256"]
            for meta in metadata
        )
    )
    checks["scene_profile_fixed"] = _check(
        bool(fixed_profile),
        (
            "Scene and quality profile must stay fixed across the sequence"
            if not fixed_profile
            else ""
        ),
    )
    checks["sequence_length"] = _check(
        len(candidates) == policy["sequence_frames"],
        "Exactly 100 frames are required" if len(candidates) != 100 else "",
        observed=len(candidates),
    )
    ids = [
        frame.get("metadata", {}).get(
            "frame_id", frame.get("metadata", {}).get("frame_index")
        )
        for frame in candidates
    ]
    checks["frame_order"] = _check(
        ids == list(range(len(candidates))),
        (
            "Frame IDs must be consecutive starting at zero"
            if ids != list(range(len(candidates)))
            else ""
        ),
    )
    timestamps = [frame.get("metadata", {}).get("time_s") for frame in candidates]
    valid_times = (
        all(
            isinstance(value, (float, int)) and np.isfinite(value)
            for value in timestamps
        )
        and len(timestamps) > 1
        and np.allclose(np.diff(timestamps), policy["sequence_dt_s"], atol=1e-6, rtol=0)
    )
    checks["fixed_cadence"] = _check(
        bool(valid_times),
        (
            f"Sequence timestamps must use the frozen {1 / policy['sequence_dt_s']:g} Hz cadence"
            if not valid_times
            else ""
        ),
    )
    checks["reference_count"] = _check(
        len(references) == len(candidates) if references is not None else None,
        "Independent reference frame count must match candidate count",
    )
    if references is not None and len(references) != len(candidates):
        checks["reference_count"] = _check(
            False, "Candidate/reference frame counts differ"
        )
        references = None
        result["reference_id"] = None
    result["consistency_only"] = references is None
    checks["independent_reference"] = _check(
        True if references is not None else None,
        (
            "Independent sequence absent; temporal consistency is not quality qualification"
            if references is None
            else ""
        ),
    )
    if references is not None:
        ref_ids = [
            frame.get("metadata", {}).get(
                "frame_id", frame.get("metadata", {}).get("frame_index")
            )
            for frame in references
        ]
        checks["reference_frame_order"] = _check(
            ref_ids == ids,
            "Reference/candidate frame IDs must match" if ref_ids != ids else "",
        )
    for index, frame in enumerate(candidates):
        report = evaluate_frame(
            frame, references[index] if references is not None else None, policy
        )
        result["frames"].append(report)
    artifacts = {}
    for frame in result["frames"]:
        for artifact in frame["artifacts"]:
            artifacts[(artifact["path"], artifact["sha256"])] = artifact
    result["artifacts"] = list(artifacts.values())
    checks["all_frame_quality"] = _check(
        (
            all(frame["quality_status"] == "qualified" for frame in result["frames"])
            if candidates
            else None
        ),
        "Every frame must pass the independent frame protocol",
    )
    geometry_frames = references if references is not None else candidates
    pair_values = {
        name: []
        for name in policy["temporal_thresholds"]
        if name != "temporal_silhouette_iou"
    }
    for index in range(1, len(candidates)):
        pair = {"previous_frame": index - 1, "current_frame": index}
        try:
            rows, cols, ys, xs, static, projected_depth, depth_scale = _correspondences(
                geometry_frames[index - 1], geometry_frames[index], policy
            )
            population = int(
                np.isfinite(np.asarray(geometry_frames[index - 1]["depth"])).sum()
            )
            if (
                not len(rows)
                or len(rows) / max(1, population)
                < policy["minimum_correspondence_fraction"]
            ):
                raise ValueError(
                    "Insufficient visible reference correspondences after occlusion rejection"
                )
            first, second = _display_rgb(candidates[index - 1]), _display_rgb(
                candidates[index]
            )
            rgb_change = second[ys, xs] - first[rows, cols]
            if references is not None:
                a, b = _display_rgb(references[index - 1]), _display_rgb(
                    references[index]
                )
                rgb_change -= b[ys, xs] - a[rows, cols]
            if not static.any():
                raise ValueError("No static-region correspondences for flicker")
            pair["static_region_flicker"] = float(
                np.sqrt(np.mean(rgb_change[static] ** 2))
            )
            pair["reprojection_rgb_mae"] = float(np.mean(np.abs(rgb_change)))
            old_depth, _ = _geometry(candidates[index - 1])
            new_depth, _ = _geometry(candidates[index])
            valid = np.isfinite(old_depth[rows, cols]) & np.isfinite(new_depth[ys, xs])
            if not valid.any():
                raise ValueError(
                    "No candidate depth on reference-visible correspondences"
                )
            # Compare geometry errors along the reference motion, which also
            # avoids counting genuine camera/object translation as instability.
            old_ref = np.asarray(geometry_frames[index - 1]["depth"])[rows, cols]
            new_ref = np.asarray(geometry_frames[index]["depth"])[ys, xs]
            if references is None:
                depth_change = new_depth[ys[valid], xs[valid]] - projected_depth[valid]
            else:
                depth_change = (
                    new_depth[ys[valid], xs[valid]] - new_ref[valid]
                ) - depth_scale[valid] * (
                    old_depth[rows[valid], cols[valid]] - old_ref[valid]
                )
            pair["temporal_depth_rmse_m"] = float(np.sqrt(np.mean(depth_change**2)))
            pair["temporal_depth_coverage"] = float(valid.mean())
            pair.update(
                status="passed",
                visible_correspondences=len(rows),
                static_correspondences=int(static.sum()),
                reference_population=population,
                excluded_occluded_or_out_of_frame=population - len(rows),
            )
            for name in pair_values:
                pair_values[name].append(pair[name])
        except (KeyError, TypeError, ValueError) as error:
            pair.update(status="missing", reason=str(error), visible_correspondences=0)
        result["pairs"].append(pair)
    expected_pairs = max(0, len(candidates) - 1)
    for name, values in pair_values.items():
        rule = policy["temporal_thresholds"][name]
        complete = expected_pairs > 0 and len(values) == expected_pairs
        value = (max(values) if rule[0] == "le" else min(values)) if complete else None
        result["metrics"][name] = _metric(
            value,
            rule,
            len(values),
            "One or more temporal pairs lack evidence" if not complete else "",
            aggregation="worst_pair",
            expected_pairs=expected_pairs,
        )
    silhouettes = [
        frame["metrics"]["silhouette_iou"]["value"] for frame in result["frames"]
    ]
    value = (
        min(silhouettes)
        if silhouettes and all(v is not None for v in silhouettes)
        else None
    )
    result["metrics"]["temporal_silhouette_iou"] = _metric(
        value,
        policy["temporal_thresholds"]["temporal_silhouette_iou"],
        sum(v is not None for v in silhouettes),
        "Independent silhouettes absent" if value is None else "",
        aggregation="worst_frame",
    )
    _finish(result)
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        pairs_path = destination / "temporal_pairs.json"
        write_json(pairs_path, result["pairs"])
        result["artifacts"].append(_artifact(pairs_path, "temporal_pairs"))
        write_json(destination / "quality_evidence.json", result)
    return result
