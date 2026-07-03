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
        result.update(
            {
                "status": "ok",
                "object_scales": estimates,
                "unit_note": (
                    "Per-object scale_factor is not baked into simready GLBs. "
                    "For text input, simready_geometry_path multiplied by this "
                    "scale_factor gives the estimated real-world size."
                ),
            }
        )
        log_info(f"text scene metric scale completed count={len(estimates)}")
    except Exception as exc:
        result.update({"status": "failed", "reason": traceback.format_exc()})
        MetricScaleManager.set_for_all_objects(
            objects=object_results,
            status="failed",
            reason="text_scene_metric_scale_failed",
            method=str(result["method"]),
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
