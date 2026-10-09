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

"""Immutable quality policies and requirements; standard-library-only imports.

All modality variants retain the original numeric thresholds. An unrequested
measurement is not evidence of success. RGB+normals has an explicit RGB quality
scope and requires valid normal output storage, but makes no normal-quality claim.
The original full v1 policy remains unchanged; temporal30-v1 declares a distinct
30 Hz timeline without converting any captured timestamps.
"""

from __future__ import annotations

from typing import Any

from scripts.benchmark.core.artifacts import stable_hash

__all__ = [
    "default_protocol",
    "protocol_by_id",
    "protocol_for_modalities",
    "protocol_requirements",
    "validate_protocol",
]

_LPIPS_BACKBONE_SHA256 = (
    "b8a52dc049b60e4b6ab68ad0df457362afab8b6304b2febdc1650a5dab4d7e7b"
)
_LPIPS_LINEAR_SHA256 = (
    "4a5350f23600cb79923ce65bb07cbf57dca461329894153e05a1346bd531cf76"
)

_MODALITY_POLICIES = {
    "pure-render-quality/rgb-v1": ["rgb"],
    "pure-render-quality/rgbd-v1": ["rgb", "depth"],
    "pure-render-quality/rgb-normals-unassessed-v1": ["rgb", "normals"],
}


def default_protocol() -> dict[str, Any]:
    """Return the unchanged full RGB/depth/segmentation v1 policy.

    Returns:
        Fresh JSON-compatible thresholds and fixed 100-frame, 60 Hz sequence rules.
    """
    return {
        "id": "pure-render-quality/v1",
        "threshold_origin": "prespecified benchmark policy; not empirical conclusions",
        "rgb_domain": "display_srgb_0_1; linear inputs clipped then IEC sRGB transfer",
        "ssim": {"window": 11, "gaussian_sigma": 1.5, "sample_covariance": False},
        "lpips": {
            "network": "squeeze",
            "version": "0.1",
            "pretrained": True,
            "input": "srgb RGB mapped to [-1,1]",
            "device": "cpu",
            "backbone_sha256": _LPIPS_BACKBONE_SHA256,
            "linear_weights_sha256": _LPIPS_LINEAR_SHA256,
        },
        "frame_thresholds": {
            "ssim": ["ge", 0.95, "1"],
            "lpips": ["le", 0.05, "1"],
            "depth_rmse_m": ["le", 0.01, "m"],
            "depth_valid_coverage": ["ge", 0.99, "1"],
            "instance_iou": ["ge", 0.98, "1"],
            "silhouette_iou": ["ge", 0.99, "1"],
            "depth_silhouette_consistency": ["ge", 0.999, "1"],
        },
        "temporal_thresholds": {
            "static_region_flicker": ["le", 0.02, "1"],
            "reprojection_rgb_mae": ["le", 0.03, "1"],
            "temporal_depth_rmse_m": ["le", 0.01, "m"],
            "temporal_silhouette_iou": ["ge", 0.99, "1"],
            "temporal_depth_coverage": ["ge", 0.99, "1"],
        },
        "background_id": 0,
        "instance_reduction": "macro mean over union of visible canonical foreground IDs",
        "depth_population": "all finite positive reference depths; missing candidate depths are coverage failures",
        "sequence_frames": 100,
        "sequence_dt_s": 1 / 60,
        "sequence_aggregation": "worst pair/frame; every pair must have evidence",
        "occlusion_absolute_m": 0.01,
        "occlusion_relative": 0.01,
        "minimum_correspondence_fraction": 0.05,
        "independent_renderers": ["mitsuba"],
        "reference_noise_rmse_max": 0.01,
        "crop_count": 3,
    }


def protocol_by_id(protocol_id: str) -> dict[str, Any]:
    """Resolve a known immutable policy, rejecting unknown identifiers.

    Args:
        protocol_id: Exact versioned policy identifier.

    Returns:
        Fresh complete policy, suitable for an experiment definition and hashing.
    """
    policy = default_protocol()
    if protocol_id == policy["id"]:
        return policy
    policy["id"] = protocol_id
    if protocol_id == "pure-render-quality/temporal30-v1":
        policy["sequence_dt_s"] = 1 / 30
    elif protocol_id == "pure-render-quality/rgb-temporal30-v1":
        policy.update(
            sequence_dt_s=1 / 30,
            requested_modalities=["rgb"],
            assessed_modalities=["rgb"],
            unassessed_modalities=[],
            temporal_geometry_source="independent_reference_only",
        )
    elif protocol_id in _MODALITY_POLICIES:
        requested = _MODALITY_POLICIES[protocol_id]
        policy["requested_modalities"] = list(requested)
        policy["assessed_modalities"] = [
            name for name in requested if name != "normals"
        ]
        policy["unassessed_modalities"] = [
            name for name in requested if name == "normals"
        ]
    else:
        raise ValueError(f"Unknown frozen quality protocol: {protocol_id}")
    return policy


def protocol_for_modalities(
    modalities: tuple[str, ...], *, temporal_dt_s: float | None = None
) -> dict[str, Any]:
    """Select an explicit modality contract and optional full-sequence cadence.

    Args:
        modalities: Requested outputs: RGB, RGB+depth, RGB+normals, or full
            RGB/depth/segmentation. Ordering does not change quality semantics.
        temporal_dt_s: ``None`` for frame quality; full sequences may declare
            exactly ``1/60`` or ``1/30``. RGB-only sequences support ``1/30``
            using independent reference geometry for visibility and motion.
            Captured timestamps are never converted.

    Returns:
        Frozen policy with unchanged numeric thresholds for all required metrics.
    """
    selected = set(modalities)
    if len(selected) != len(modalities):
        raise ValueError("Duplicate requested modalities")
    if selected == {"rgb", "depth", "seg"}:
        if temporal_dt_s in (None, 1 / 60):
            return default_protocol()
        if temporal_dt_s == 1 / 30:
            return protocol_by_id("pure-render-quality/temporal30-v1")
        raise ValueError("Full temporal quality supports only declared 30 or 60 Hz")
    if selected == {"rgb"} and temporal_dt_s == 1 / 30:
        return protocol_by_id("pure-render-quality/rgb-temporal30-v1")
    if temporal_dt_s is not None:
        raise ValueError("Temporal qualification requires full RGB/depth/segmentation")
    for protocol_id, required in _MODALITY_POLICIES.items():
        if selected == set(required):
            return protocol_by_id(protocol_id)
    raise ValueError("No frozen quality policy for requested modalities")


def protocol_requirements(protocol_id: str, kind: str = "frame") -> dict[str, Any]:
    """Return the exact offline-verifiable assessment and qualification contract.

    Args:
        protocol_id: Known policy identifier.
        kind: ``frame`` or ``sequence``. RGB temporal quality uses independent
            reference depth/IDs for correspondences without claiming output
            depth or segmentation quality.

    Returns:
        Protocol hash, exact required metric/check lists, and modality scope.
        Reports must not accept a shorter caller-supplied requirement list.
    """
    policy = protocol_by_id(protocol_id)
    requested = policy.get("requested_modalities", ["rgb", "depth", "seg"])
    assessed = policy.get("assessed_modalities", list(requested))
    unassessed = policy.get("unassessed_modalities", [])
    if kind == "frame":
        metrics = ["ssim", "lpips"]
        checks = [
            "independent_reference",
            "reference_evidence",
            "scene_binding",
            "camera_binding",
            "photometric_calibration",
            "reference_noise_floor",
            "rgb_semantics",
            "requested_outputs",
        ]
        if "depth" in requested:
            metrics.extend(["depth_rmse_m", "depth_valid_coverage"])
        if "seg" in requested:
            metrics.extend(
                ["instance_iou", "silhouette_iou", "depth_silhouette_consistency"]
            )
            checks.extend(["geometry_semantics", "reference_geometry_integrity"])
        elif "depth" in requested:
            checks.extend(["depth_semantics", "reference_depth_semantics"])
    elif kind == "sequence" and (
        set(requested) == {"rgb", "depth", "seg"}
        or protocol_id == "pure-render-quality/rgb-temporal30-v1"
    ):
        metrics = (
            list(policy["temporal_thresholds"])
            if "depth" in requested
            else ["static_region_flicker", "reprojection_rgb_mae"]
        )
        checks = [
            "scene_profile_fixed",
            "sequence_length",
            "frame_order",
            "fixed_cadence",
            "reference_count",
            "independent_reference",
            "reference_frame_order",
            "all_frame_quality",
        ]
    else:
        raise ValueError(
            "Unknown evaluation kind or unsupported temporal modality scope"
        )
    return {
        "protocol_id": protocol_id,
        "protocol_sha256": stable_hash(policy),
        "evaluation_kind": kind,
        "requested_modalities": list(requested),
        "assessed_modalities": list(assessed),
        "unassessed_modalities": list(unassessed),
        "required_metrics": metrics,
        "required_checks": checks,
    }


def validate_protocol(protocol: dict[str, Any] | None) -> dict[str, Any]:
    """Reject altered thresholds or undeclared policy fields.

    Args:
        protocol: Complete frozen policy, or ``None`` for the original full v1.

    Returns:
        A fresh known policy equal to the supplied record.
    """
    if protocol is None:
        return default_protocol()
    policy = protocol_by_id(protocol.get("id", ""))
    if protocol != policy:
        raise ValueError("Modified quality protocol; versioned thresholds are frozen")
    return policy
