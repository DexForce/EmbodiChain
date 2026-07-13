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

import math
import traceback
from pathlib import Path
from typing import Any

from embodichain.gen_sim.prompt2scene.agent_tools.managers.simready_manager import (
    METRIC_SCALE_ENABLED,
    EstimateMetricScalesRequest,
    MetricScaleManager,
    MetricScaleObjectInput,
)
from embodichain.gen_sim.prompt2scene.utils.io import write_json
from embodichain.gen_sim.prompt2scene.utils.log import log_info, log_warning

__all__ = ["build_metric_scale_inputs", "estimate_text_scene_metric_scale"]

DEFAULT_TEXT_METRIC_SCALE_FALLBACK_FACTOR = 0.10
MIN_TEXT_METRIC_SCALE_FACTOR = 0.02
MAX_TEXT_METRIC_SCALE_FACTOR = 0.50


def estimate_text_scene_metric_scale(
    *,
    object_results: list[dict[str, Any]],
    user_text: str,
    messages: list[dict[str, Any]],
    schema: dict[str, Any],
    output_dir: Path,
    output_root: Path,
    llm: Any | None,
    step_name: str,
) -> dict[str, Any]:
    """Estimate real-world scales for generated text-scene objects."""
    result: dict[str, Any] = {
        "status": "skipped",
        "method": "text_scene_vlm_candidate_shape_ratio_median_scale",
        "user_text": user_text,
        "objects": [],
    }
    try:
        if not object_results:
            result["reason"] = "missing_objects"
            log_warning("text scene metric scale skipped reason=missing_objects")
            return result
        if not METRIC_SCALE_ENABLED:
            result["reason"] = "metric_scale_disabled"
            MetricScaleManager.set_for_all_objects(
                objects=object_results,
                status="skipped",
                reason="metric_scale_disabled",
                method=str(result["method"]),
            )
            _sanitize_metric_scale_factors(
                objects=object_results,
                result=result,
            )
            log_info("text scene metric scale skipped reason=metric_scale_disabled")
            return result
        if llm is None:
            result["reason"] = "missing_llm"
            MetricScaleManager.set_for_all_objects(
                objects=object_results,
                status="skipped",
                reason="missing_llm",
                method=str(result["method"]),
            )
            _sanitize_metric_scale_factors(
                objects=object_results,
                result=result,
            )
            log_warning("text scene metric scale skipped reason=missing_llm")
            return result

        log_info(f"text scene metric scale started count={len(object_results)}")
        metric_objects = build_metric_scale_inputs(
            objects=object_results,
            output_root=output_root,
        )
        result["objects"] = MetricScaleManager.object_prompt_payload(metric_objects)
        metric_result = MetricScaleManager.estimate_metric_scales(
            EstimateMetricScalesRequest(
                objects=metric_objects,
                messages=messages,
                schema=schema,
                llm=llm,
                context="Text scene metric scale estimate",
                method=str(result["method"]),
                step_name=step_name,
                raw_output_path=output_dir / "raw_model_output.json",
            )
        )
        raw_model_output = metric_result.raw_model_output or {}
        if not (output_dir / "raw_model_output.json").is_file():
            try:
                write_json(output_dir / "raw_model_output.json", raw_model_output)
            except Exception as exc:
                log_warning(f"metric scale raw output write failed error={exc}")

        estimates = metric_result.object_scales
        MetricScaleManager.apply_to_objects(
            objects=object_results,
            object_scales=estimates,
        )
        scale_updates = _sanitize_metric_scale_factors(
            objects=object_results,
            result=result,
        )
        result.update(
            {
                "status": "ok",
                "object_scales": _object_metric_scales(object_results),
                "unit_note": (
                    "Per-object scale_factor is not baked into simready GLBs. "
                    "For text input, simready_geometry_path multiplied by this "
                    "scale_factor gives the estimated real-world size."
                ),
            }
        )
        if scale_updates["fallback_count"]:
            result["fallback_count"] = scale_updates["fallback_count"]
            result["fallback_scale_factor"] = DEFAULT_TEXT_METRIC_SCALE_FALLBACK_FACTOR
        if scale_updates["clamped_count"]:
            result["clamped_count"] = scale_updates["clamped_count"]
            result["scale_clamp"] = {
                "min": MIN_TEXT_METRIC_SCALE_FACTOR,
                "max": MAX_TEXT_METRIC_SCALE_FACTOR,
            }
        log_info(f"text scene metric scale completed count={len(estimates)}")
    except Exception as exc:
        result.update({"status": "failed", "reason": traceback.format_exc()})
        MetricScaleManager.set_for_all_objects(
            objects=object_results,
            status="failed",
            reason="text_scene_metric_scale_failed",
            method=str(result["method"]),
        )
        _sanitize_metric_scale_factors(
            objects=object_results,
            result=result,
        )
        log_warning(f"text scene metric scale failed error={exc}")
    return result


def build_metric_scale_inputs(
    *,
    objects: list[dict[str, Any]],
    output_root: Path,
) -> list[MetricScaleObjectInput]:
    inputs: list[MetricScaleObjectInput] = []
    for obj in objects:
        mesh_path = _resolve_generated_path(
            obj.get("simready_geometry_path") or obj.get("mesh_path"),
            output_root,
        )
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Simready object GLB not found: {mesh_path}")
        inputs.append(
            MetricScaleObjectInput(
                object_id=str(obj.get("id", "")),
                object_name=str(obj.get("name", "")),
                object_description=str(obj.get("description", "")),
                mesh_path=mesh_path,
            )
        )
    return inputs


def _resolve_generated_path(value: Any, output_root: Path) -> Path:
    path = Path(str(value or "")).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (output_root / path).resolve()


def _sanitize_metric_scale_factors(
    *,
    objects: list[dict[str, Any]],
    result: dict[str, Any],
) -> dict[str, int]:
    fallback_count = 0
    clamped_count = 0
    for obj in objects:
        metric_scale = obj.get("metric_scale")
        raw_scale_factor = _finite_float_or_none(
            metric_scale.get("scale_factor") if isinstance(metric_scale, dict) else None
        )
        if _has_valid_metric_scale_factor(metric_scale):
            scale_factor = float(raw_scale_factor)
            clamped = min(
                max(scale_factor, MIN_TEXT_METRIC_SCALE_FACTOR),
                MAX_TEXT_METRIC_SCALE_FACTOR,
            )
            if clamped != scale_factor:
                metric_scale["raw_scale_factor"] = scale_factor
                metric_scale["scale_factor"] = clamped
                metric_scale["scale_policy"] = "clamped"
                metric_scale["scale_clamp"] = {
                    "min": MIN_TEXT_METRIC_SCALE_FACTOR,
                    "max": MAX_TEXT_METRIC_SCALE_FACTOR,
                }
                clamped_count += 1
            continue
        previous = dict(metric_scale) if isinstance(metric_scale, dict) else {}
        previous_status = str(previous.get("status") or "").strip()
        previous_reason = str(previous.get("reason") or "").strip()
        previous.update(
            {
                "status": "fallback",
                "method": str(previous.get("method") or result.get("method") or ""),
                "object_id": str(obj.get("id", "")),
                "scale_factor": DEFAULT_TEXT_METRIC_SCALE_FALLBACK_FACTOR,
                "raw_scale_factor": raw_scale_factor,
                "scale_policy": "fallback_10cm",
                "reason": "text_metric_scale_unavailable_default_10cm_longest_edge",
                "fallback_longest_edge_m": DEFAULT_TEXT_METRIC_SCALE_FALLBACK_FACTOR,
                "previous_status": previous_status,
                "previous_reason": previous_reason,
                "unit_note": (
                    "Fallback assumes the generated simready object's longest "
                    "edge was normalized to 1m, then scales it to 10cm."
                ),
            }
        )
        obj["metric_scale"] = previous
        fallback_count += 1
    if fallback_count or clamped_count:
        result["fallback_count"] = fallback_count
        result["clamped_count"] = clamped_count
        result["fallback_scale_factor"] = DEFAULT_TEXT_METRIC_SCALE_FALLBACK_FACTOR
        result["scale_clamp"] = {
            "min": MIN_TEXT_METRIC_SCALE_FACTOR,
            "max": MAX_TEXT_METRIC_SCALE_FACTOR,
        }
        result["object_scales"] = _object_metric_scales(objects)
    return {"fallback_count": fallback_count, "clamped_count": clamped_count}


def _has_valid_metric_scale_factor(metric_scale: Any) -> bool:
    if not isinstance(metric_scale, dict):
        return False
    if str(metric_scale.get("status", "")).strip() != "ok":
        return False
    try:
        scale_factor = float(metric_scale.get("scale_factor"))
    except (TypeError, ValueError):
        return False
    return math.isfinite(scale_factor) and scale_factor > 0.0


def _finite_float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _object_metric_scales(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(obj["metric_scale"])
        for obj in objects
        if isinstance(obj.get("metric_scale"), dict)
    ]
