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

from typing import Any

import numpy as np

from embodichain.gen_sim.prompt2scene.agent_tools.managers.geometry_manager import (
    GeometryManager,
    LoadMeshRequest,
)
from embodichain.gen_sim.prompt2scene.agent_tools.managers.metric_scale_manager.schemas import (
    EstimateMetricScalesRequest,
    EstimateMetricScalesResult,
    GlobalMetricScaleRequest,
    MetricScaleObjectInput,
)
from embodichain.gen_sim.prompt2scene.utils.io import write_json
from embodichain.gen_sim.prompt2scene.workflows.llm_output import (
    call_structured_json_model_step,
)

__all__ = ["METRIC_SCALE_ENABLED", "MetricScaleManager"]

METRIC_SCALE_ENABLED = True


class MetricScaleManager:
    """Manager for metric scale estimation and scale aggregation."""

    @staticmethod
    def estimate_metric_scales(
        request: EstimateMetricScalesRequest,
    ) -> EstimateMetricScalesResult:
        """Call an LLM and convert bbox-size predictions into scale factors."""
        object_payload = MetricScaleManager.build_object_payload(request.objects)
        raw_model_output_path = (
            request.raw_output_path.expanduser().resolve()
            if request.raw_output_path is not None
            else None
        )
        raw_model_output = call_structured_json_model_step(
            llm=request.llm,
            schema=request.schema,
            messages=request.messages,
            context=request.context,
            step_name=request.step_name,
            output_root=None,
            attempt_count=0,
            raw_output_writer=(
                (lambda payload: write_json(raw_model_output_path, payload))
                if raw_model_output_path is not None
                else None
            ),
        )
        object_scales = MetricScaleManager.apply_model_output(
            object_payload=object_payload,
            raw_model_output=raw_model_output,
            method=request.method,
        )
        return EstimateMetricScalesResult(
            status="ok",
            object_scales=object_scales,
            object_payload=object_payload,
            raw_model_output=raw_model_output,
        )

    @staticmethod
    def build_object_payload(
        objects: list[MetricScaleObjectInput],
    ) -> list[dict[str, Any]]:
        """Build object payload with normalized mesh bbox measurements."""
        geom = GeometryManager()
        payload: list[dict[str, Any]] = []
        for obj in objects:
            mesh = geom.load_mesh(LoadMeshRequest(mesh_path=obj.mesh_path)).mesh
            normalized_bbox_size_m = GeometryManager.mesh_aabb_size(mesh)
            payload.append(
                {
                    "object_id": obj.object_id,
                    "object_name": obj.object_name,
                    "object_description": obj.object_description,
                    "normalized_bbox_size_m": normalized_bbox_size_m.tolist(),
                    "normalized_bbox_ratio": GeometryManager.bbox_ratio(
                        normalized_bbox_size_m
                    ).tolist(),
                }
            )
        return payload

    @staticmethod
    def object_prompt_payload(
        objects: list[MetricScaleObjectInput],
    ) -> list[dict[str, str]]:
        """Return the lightweight object payload intended for LLM prompts."""
        return [
            {
                "object_id": obj.object_id,
                "object_name": obj.object_name,
                "object_description": obj.object_description,
            }
            for obj in objects
        ]

    @staticmethod
    def apply_model_output(
        *,
        object_payload: list[dict[str, Any]],
        raw_model_output: dict[str, Any],
        method: str,
    ) -> list[dict[str, Any]]:
        """Convert model bbox predictions into per-object metric-scale records."""
        model_by_id = {
            str(item.get("object_id", "")): item
            for item in raw_model_output.get("object_scales", [])
            if isinstance(item, dict)
        }
        estimates: list[dict[str, Any]] = []
        for payload in object_payload:
            object_id = str(payload.get("object_id", ""))
            model_item = model_by_id.get(object_id)
            if model_item is None:
                estimates.append(
                    MetricScaleManager.failure(
                        object_id=object_id,
                        reason="missing_object_scale_from_model",
                        method=method,
                    )
                )
                continue
            estimates.append(
                MetricScaleManager.select_candidate(
                    object_id=object_id,
                    object_name=str(payload.get("object_name", "")),
                    object_description=str(payload.get("object_description", "")),
                    bbox_dims_cm=model_item.get("bbox_dims_cm", []),
                    confidence=float(model_item.get("confidence", 0.0)),
                    reason=str(model_item.get("reason", "")),
                    normalized_bbox_size_m=np.asarray(
                        payload["normalized_bbox_size_m"],
                        dtype=np.float64,
                    ),
                    method=method,
                )
            )
        return estimates

    @staticmethod
    def apply_to_objects(
        *,
        objects: list[dict[str, Any]],
        object_scales: list[dict[str, Any]],
    ) -> None:
        """Attach metric-scale records to object dictionaries by object id."""
        scale_by_id = {str(item.get("object_id", "")): item for item in object_scales}
        for obj in objects:
            object_id = str(obj.get("id", ""))
            if object_id in scale_by_id:
                obj["metric_scale"] = scale_by_id[object_id]

    @staticmethod
    def select_candidate(
        *,
        object_id: str,
        object_name: str,
        object_description: str,
        bbox_dims_cm: Any,
        confidence: float,
        reason: str,
        normalized_bbox_size_m: np.ndarray,
        method: str,
    ) -> dict[str, Any]:
        """Select a scale factor from predicted real-world bbox dimensions."""
        try:
            selected = MetricScaleManager.compute_from_bbox_dims(
                bbox_dims_cm=bbox_dims_cm,
                confidence=confidence,
                reason=reason,
                normalized_bbox_size_m=normalized_bbox_size_m,
            )
        except (TypeError, ValueError):
            return MetricScaleManager.failure(
                object_id=object_id,
                reason="invalid_bbox_dims_cm",
                method=method,
            )
        normalized_bbox_size_cm = (
            np.asarray(normalized_bbox_size_m, dtype=np.float64) * 100.0
        )
        return {
            "status": "ok",
            "method": method,
            "object_id": object_id,
            "object_name": object_name,
            "object_description": object_description,
            "normalized_bbox_size_m": normalized_bbox_size_m.tolist(),
            "normalized_bbox_size_cm": normalized_bbox_size_cm.tolist(),
            "normalized_bbox_ratio": GeometryManager.bbox_ratio(
                normalized_bbox_size_m
            ).tolist(),
            "bbox_dims_cm": selected["bbox_dims_cm"],
            "axis_match": selected["axis_match"],
            "scale_factor": selected["scale_factor"],
            "confidence": selected["confidence"],
            "reason": selected["reason"],
            "unit_note": "scale_factor is not baked into this GLB.",
        }

    @staticmethod
    def compute_from_bbox_dims(
        *,
        bbox_dims_cm: Any,
        confidence: float,
        reason: str,
        normalized_bbox_size_m: np.ndarray,
    ) -> dict[str, Any]:
        """Compute one scale candidate from model-predicted bbox dimensions."""
        dims_cm = np.asarray(
            [float(value) for value in bbox_dims_cm],
            dtype=np.float64,
        )
        if dims_cm.shape != (3,) or np.any(dims_cm <= 0.0):
            raise ValueError("bbox_dims_cm must contain three positive values.")
        normalized_bbox_size_cm = (
            np.asarray(normalized_bbox_size_m, dtype=np.float64) * 100.0
        )
        axis_match = GeometryManager.best_axis_bbox_scale_match(
            source_size_cm=normalized_bbox_size_cm,
            target_size_cm=dims_cm,
        )
        return {
            "bbox_dims_cm": dims_cm.tolist(),
            "axis_match": axis_match,
            "scale_factor": float(axis_match["scale_factor"]),
            "confidence": confidence,
            "reason": reason,
        }

    @staticmethod
    def failure(
        *,
        object_id: str,
        reason: str,
        method: str,
    ) -> dict[str, Any]:
        """Build a failed per-object metric-scale record."""
        return {
            "status": "failed",
            "method": method,
            "object_id": object_id,
            "scale_factor": 1.0,
            "reason": reason,
        }

    @staticmethod
    def set_for_all_objects(
        *,
        objects: list[dict[str, Any]],
        status: str,
        reason: str,
        method: str,
    ) -> None:
        """Attach the same fallback metric-scale status to all objects."""
        for obj in objects:
            obj["metric_scale"] = {
                "status": status,
                "method": method,
                "object_id": str(obj.get("id", "")),
                "scale_factor": 1.0,
                "reason": reason,
            }

    @staticmethod
    def compute_global_from_object_scenes(
        request: GlobalMetricScaleRequest,
    ) -> dict[str, Any]:
        """Aggregate object metric scales into one global scale for a scene layout."""
        if not METRIC_SCALE_ENABLED:
            return {
                "status": "disabled",
                "method": "metric_scale_disabled",
                "scale_factor": 1.0,
                "object_count": len(request.objects),
                "used_count": 0,
                "skipped_count": len(request.objects),
                "used": [],
                "skipped": [
                    {"id": str(item.get("id", "")), "reason": "metric_scale_disabled"}
                    for item in request.objects
                ],
                "unit_note": (
                    "Metric scale is disabled; aligned GLBs keep simready "
                    "normalized size."
                ),
            }

        used: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        object_by_id = {str(item.get("id", "")): item for item in request.objects}
        for object_id, scene in request.object_scenes:
            item = object_by_id.get(object_id)
            if item is None:
                skipped.append({"id": object_id, "reason": "missing_object_record"})
                continue
            metric_scale = item.get("metric_scale")
            if not isinstance(metric_scale, dict):
                skipped.append({"id": object_id, "reason": "missing_metric_scale"})
                continue
            if metric_scale.get("status") != "ok":
                skipped.append(
                    {
                        "id": object_id,
                        "reason": str(metric_scale.get("status") or "not_ok"),
                    }
                )
                continue

            scale_factor_simready = float(metric_scale.get("scale_factor", 1.0))
            if not np.isfinite(scale_factor_simready) or scale_factor_simready <= 0.0:
                skipped.append(
                    {"id": object_id, "reason": "invalid_simready_scale_factor"}
                )
                continue
            try:
                simready_size_m = np.asarray(
                    [float(v) for v in metric_scale.get("normalized_bbox_size_m", [])],
                    dtype=np.float64,
                )
            except (TypeError, ValueError):
                skipped.append(
                    {"id": object_id, "reason": "invalid_normalized_bbox_size_m"}
                )
                continue
            if simready_size_m.shape != (3,) or np.any(simready_size_m <= 0.0):
                skipped.append(
                    {"id": object_id, "reason": "invalid_normalized_bbox_size_m"}
                )
                continue

            current_bounds = np.asarray(GeometryManager.scene_to_mesh(scene).bounds)
            current_size_m = current_bounds[1] - current_bounds[0]
            if current_size_m.shape != (3,) or np.any(current_size_m <= 0.0):
                skipped.append({"id": object_id, "reason": "invalid_current_scene_aabb"})
                continue

            geo_ratio = np.sort(current_size_m) / np.sort(simready_size_m)
            geo_scale = float(np.median(geo_ratio))
            if not np.isfinite(geo_scale) or geo_scale <= 0.0:
                skipped.append({"id": object_id, "reason": "non_positive_geo_scale"})
                continue

            effective_scale = scale_factor_simready / geo_scale
            if not np.isfinite(effective_scale) or effective_scale <= 0.0:
                skipped.append(
                    {"id": object_id, "reason": "non_positive_effective_scale"}
                )
                continue

            used.append(
                {
                    "id": object_id,
                    "effective_scale": effective_scale,
                    "scale_factor_simready": scale_factor_simready,
                    "geo_scale": geo_scale,
                    "simready_bbox_size_m": simready_size_m.tolist(),
                    "simready_bbox_size_cm": (simready_size_m * 100.0).tolist(),
                    "current_scene_bbox_size_m": current_size_m.tolist(),
                    "current_scene_bbox_size_cm": (current_size_m * 100.0).tolist(),
                    "target_bbox_dims_cm": metric_scale.get("bbox_dims_cm"),
                    "confidence": metric_scale.get("confidence"),
                }
            )

        if not used:
            return {
                "status": "fallback",
                "method": "simready_reference_geo_ratio_mean_with_clamp",
                "scale_factor": 1.0,
                "raw_scale_factor": 1.0,
                "was_clamped": False,
                "clamp": {"min": request.min_scale, "max": request.max_scale},
                "object_count": len(request.objects),
                "used_count": 0,
                "skipped_count": len(skipped),
                "used": [],
                "skipped": skipped,
                "unit_note": (
                    "No valid metric scale was available; image clutter keeps the "
                    "SAM3D layout scale without an additional metric scale."
                ),
            }

        raw_scale_factor = float(np.mean([item["effective_scale"] for item in used]))
        scale_factor = float(
            np.clip(raw_scale_factor, request.min_scale, request.max_scale)
        )
        return {
            "status": "ok",
            "method": "simready_reference_geo_ratio_mean_with_clamp",
            "scale_factor": scale_factor,
            "raw_scale_factor": raw_scale_factor,
            "was_clamped": bool(scale_factor != raw_scale_factor),
            "clamp": {"min": request.min_scale, "max": request.max_scale},
            "object_count": len(request.objects),
            "used_count": len(used),
            "skipped_count": len(skipped),
            "used": used,
            "skipped": skipped,
            "unit_note": (
                "Global scale derived from scene-level VLM per-object scale_factor "
                "divided by the geometric scale ratio between simready normalized "
                "bbox and current aligned scene bbox (sorted, permutation-invariant). "
                f"Aggregated via mean across objects, clamped to "
                f"[{request.min_scale:.2f}, {request.max_scale:.2f}]."
            ),
        }
