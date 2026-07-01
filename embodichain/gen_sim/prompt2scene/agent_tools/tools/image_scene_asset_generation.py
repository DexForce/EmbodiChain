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

import shutil
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from embodichain.gen_sim.prompt2scene.utils.log import log_info, log_warning
from embodichain.gen_sim.prompt2scene.agent_tools.clients.image_segmentation_client import (
    decode_rle_mask,
)
from embodichain.gen_sim.prompt2scene.agent_tools.managers.geometry_generation_manager import (
    GeometryGenerationManager,
    RgbaImageToGeometryRequest,
    RgbaImagesToGeometriesRequest,
)
from embodichain.gen_sim.prompt2scene.agent_tools.managers.image_generation_manager import (
    ImageGenerationManager,
    TextToAssetImageRequest,
)
from embodichain.gen_sim.prompt2scene.agent_tools.managers.image_segmentation_manager import (
    AssetImageToRgbaRequest,
    ImageSegmentationManager,
)
from embodichain.gen_sim.prompt2scene.agent_tools.tools.image_layout_alignment import (
    _export_support_aligned_layout_glbs,
)
from embodichain.gen_sim.prompt2scene.agent_tools.managers.simready_manager import (
    MakeAssetSimreadyRequest,
    MakeTableSimreadyRequest,
    SimreadyManager,
)
from embodichain.gen_sim.prompt2scene.agent_tools.managers.simready_manager import (
    METRIC_SCALE_ENABLED,
    EstimateMetricScalesRequest,
    IMAGE_METRIC_SCALE_JSON_SCHEMA,
    MetricScaleManager,
    MetricScaleObjectInput,
    build_image_metric_scale_messages,
)
from embodichain.gen_sim.prompt2scene.agent_tools.managers.geometry_manager import (
    GeometryManager,
)
from embodichain.gen_sim.prompt2scene.agent_tools.tools.layout_manifests import (
    _write_multi_object_layout_manifests,
)
from embodichain.gen_sim.prompt2scene.utils.io import (
    relative_path,
)

__all__ = ["generate_image_scene_assets"]

UNIFIED_SCENE_STEP = "unified_scene"


def generate_image_scene_assets(
    object_specs: list[dict[str, Any]],
    table_spec: dict[str, Any],
    spatial_relations: list[dict[str, Any]],
    segments_data: dict[str, Any],
    image_gen_dir: Path,
    glb_gen_dir: Path,
    debug_dir: Path,
    output_root: Path,
    llm: Any | None = None,
) -> dict[str, Any]:
    """Run layout-aware table/support and object generation from image masks."""
    log_info(f"image object layout generation started count={len(object_specs)}")
    status = "ok"
    failure_reason = ""
    original_image_path = str(segments_data.get("image_path", ""))
    segment_by_id: dict[str, dict[str, Any]] = {
        str(seg["asset_id"]): seg
        for seg in segments_data.get("asset_segments", [])
        if seg.get("asset_id")
    }
    table_segment = segments_data.get("table_segment")
    if not isinstance(table_segment, dict):
        table_segment = None
    debug_subdir = debug_dir / "multi_object_masks"
    masks_dir = debug_subdir / "masks"
    raw_download_dir = glb_gen_dir / "raw_downloads"
    simready_dir = glb_gen_dir / "multi_object_layouts_simready"
    aligned_dir = glb_gen_dir / "multi_object_layouts_aligned"
    masks_dir.mkdir(parents=True, exist_ok=True)
    raw_download_dir.mkdir(parents=True, exist_ok=True)
    simready_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir.mkdir(parents=True, exist_ok=True)

    requested_items: list[dict[str, Any]] = []
    mask_paths: list[Path] = []

    table_id = str(table_spec.get("id", "table")).strip() or "table"
    table_name = str(table_spec.get("name", "table")).strip() or "table"
    is_complete_visible_table = bool(
        table_spec.get("is_complete_visible_table", False)
    )
    skipped_table: dict[str, Any] | None = None
    if table_segment is None:
        skipped_table = {
            "id": table_id,
            "name": table_name,
            "reason": "missing_table_segment",
        }
    else:
        table_mask_rle = table_segment.get("mask_rle")
        if table_mask_rle is None:
            skipped_table = {
                "id": table_id,
                "name": table_name,
                "reason": "missing_table_mask_rle",
            }
        else:
            mask_path = masks_dir / f"{len(requested_items):04d}_{table_id}_mask.png"
            decode_rle_mask(table_mask_rle).save(mask_path)
            mask_paths.append(mask_path)
            requested_items.append(
                {
                    "id": table_id,
                    "name": table_name,
                    "kind": "table",
                    "mask_path": str(mask_path),
                }
            )

    for obj_spec in object_specs:
        obj_id = str(obj_spec.get("id", "")).strip()
        obj_name = str(obj_spec.get("name", "")).strip()
        if not obj_id:
            continue
        segment = segment_by_id.get(obj_id)
        if segment is None:
            continue
        mask_rle = segment.get("mask_rle")
        if mask_rle is None:
            continue

        mask_path = masks_dir / f"{len(requested_items):04d}_{obj_id}_mask.png"
        decode_rle_mask(mask_rle).save(mask_path)
        mask_paths.append(mask_path)
        requested_items.append(
            {
                "id": obj_id,
                "name": obj_name,
                "description": str(obj_spec.get("description", "")),
                "kind": "object",
                "mask_path": str(mask_path),
            }
        )

    generated_objects: list[dict[str, Any]] = []
    generated_table: dict[str, Any] | None = None
    image_manager = ImageGenerationManager()
    segmentation_manager = ImageSegmentationManager()
    geometry_manager = GeometryGenerationManager()
    simready_manager = SimreadyManager()
    try:
        if skipped_table is not None:
            raise ValueError(
                "No valid table/support mask found for image multi-object "
                f"layout generation: {skipped_table['reason']}"
            )
        if not mask_paths:
            raise ValueError(
                "No valid masks found for image multi-object layout generation."
            )

        result = geometry_manager.convert_rgba_images_to_geometries(
            RgbaImagesToGeometriesRequest(
                image_path=Path(original_image_path),
                mask_paths=mask_paths,
                output_dir=raw_download_dir,
            )
        )
        if len(result.objects) != len(requested_items):
            raise RuntimeError(
                "Multi-object SAM3D result count mismatch: "
                f"requested {len(requested_items)}, got {len(result.objects)}"
            )
        for requested, generated in zip(requested_items, result.objects):
            expected_sam3d_name = Path(requested["mask_path"]).stem
            if generated.name != expected_sam3d_name:
                raise RuntimeError(
                    "Multi-object SAM3D result order mismatch: "
                    f"expected {expected_sam3d_name!r}, got {generated.name!r}"
                )
            downloaded_raw_path = Path(generated.geometry_path).expanduser().resolve()
            raw_geometry_path = str(downloaded_raw_path)
            status_parts: list[str] = []
            transform_matrix: list[list[float]] = []
            try:
                transform = GeometryManager.compose_sam3d_multi_object_transform(
                    rotation_quaternion_wxyz=generated.rotation_quaternion_wxyz,
                    translation=generated.translation,
                    scale=generated.scale,
                )
                transform_matrix = transform.tolist()
            except Exception:
                status_parts.append(
                    f"transform_matrix_failed: {traceback.format_exc()}"
                )

            simready_geometry_path = ""
            raw_to_simready_glb_matrix: list[list[float]] = []
            metric_scale: dict[str, Any] | None = None
            try:
                if requested["kind"] == "table":
                    if is_complete_visible_table:
                        table_result = simready_manager.make_table_simready(
                            MakeTableSimreadyRequest(
                                input_path=Path(raw_geometry_path),
                                output_path=simready_dir
                                / f"{requested['id']}_simready.glb",
                            )
                        )
                        simready_geometry_path = str(table_result.output_path)
                        raw_to_simready_glb_matrix = table_result.transform_matrix
                else:
                    asset_result = simready_manager.make_asset_simready(
                        MakeAssetSimreadyRequest(
                            input_path=Path(raw_geometry_path),
                            output_path=simready_dir
                            / f"{requested['id']}_simready.glb",
                        )
                    )
                    simready_geometry_path = str(asset_result.output_path)
                    raw_to_simready_glb_matrix = asset_result.transform_matrix
            except Exception:
                status_parts.append(f"simready_failed: {traceback.format_exc()}")
            item_status = "ok" if not status_parts else "; ".join(status_parts)
            generated_item = {
                "id": requested["id"],
                "name": requested["name"],
                "kind": requested["kind"],
                "description": str(table_spec.get("description", ""))
                if requested["kind"] == "table"
                else str(requested.get("description", "")),
                "complete_table_description": str(
                    table_spec.get("complete_table_description")
                    or table_spec.get("description", "")
                ).strip()
                if requested["kind"] == "table"
                else "",
                "is_complete_visible_table": is_complete_visible_table
                if requested["kind"] == "table"
                else False,
                "status": item_status,
                "mask_path": relative_path(requested["mask_path"], output_root),
                "raw_geometry_path": relative_path(raw_geometry_path, output_root),
                "simready_geometry_path": relative_path(
                    simready_geometry_path, output_root
                )
                if simready_geometry_path
                else "",
                "mesh_path": relative_path(simready_geometry_path, output_root)
                if simready_geometry_path
                else "",
                "sam3d_name": generated.name,
                "downloaded_raw_geometry_path": relative_path(
                    str(downloaded_raw_path), output_root
                ),
                "rotation_quaternion_wxyz": generated.rotation_quaternion_wxyz,
                "translation": generated.translation,
                "scale": generated.scale,
                "transform_matrix": transform_matrix,
                "raw_to_simready_glb_matrix": raw_to_simready_glb_matrix,
                "metric_scale": metric_scale,
            }
            if requested["kind"] == "table":
                support_reference_path = raw_download_dir / "support_surface_raw.glb"
                table_raw_path = raw_download_dir / "table_raw.glb"
                shutil.copy2(downloaded_raw_path, support_reference_path)
                if is_complete_visible_table:
                    shutil.copy2(downloaded_raw_path, table_raw_path)
                    generated_item["raw_geometry_path"] = relative_path(
                        str(table_raw_path),
                        output_root,
                    )
                generated_item["support_reference_geometry_path"] = relative_path(
                    str(support_reference_path),
                    output_root,
                )
                generated_item["support_reference_transform_matrix"] = transform_matrix
                generated_item["support_normal_source"] = "segmented_table"
                generated_item["table_asset_source"] = "segmented_table"
                if not is_complete_visible_table:
                    # Replace partial image table with description-generated table.
                    incomplete_table_id = str(
                        generated_item.get("id")
                        or table_spec.get("id")
                        or "table"
                    )
                    incomplete_table_desc = str(
                        table_spec.get("complete_table_description")
                        or table_spec.get("description", "")
                    ).strip()
                    incomplete_debug_dir = (
                        debug_dir / incomplete_table_id / "description_generated"
                    )
                    incomplete_debug_dir.mkdir(parents=True, exist_ok=True)
                    incomplete_raw_download_dir = glb_gen_dir / "raw_downloads"
                    incomplete_raw_download_dir.mkdir(parents=True, exist_ok=True)
                    incomplete_raw_image = str(
                        image_manager.generate_asset_image_from_text(
                            TextToAssetImageRequest(
                                prompt=incomplete_table_desc,
                                output_path=incomplete_debug_dir
                                / f"{incomplete_table_id}_complete.png",
                            )
                        )
                    )
                    incomplete_rgba = str(
                        segmentation_manager.convert_asset_image_to_rgba(
                            AssetImageToRgbaRequest(
                                image_path=Path(incomplete_raw_image),
                                prompt=incomplete_table_desc
                                if incomplete_table_desc.strip()
                                else "whole table",
                                output_path=image_gen_dir
                                / f"{incomplete_table_id}_complete.png",
                            )
                        )
                    )
                    incomplete_raw_glb = str(
                        geometry_manager.convert_rgba_image_to_geometry(
                            RgbaImageToGeometryRequest(
                                image_path=Path(incomplete_rgba),
                                output_path=incomplete_debug_dir
                                / f"{incomplete_table_id}_complete_raw.glb",
                            )
                        )
                    )
                    incomplete_table_raw_path = (
                        incomplete_raw_download_dir / "table_raw.glb"
                    )
                    shutil.copy2(incomplete_raw_glb, incomplete_table_raw_path)
                    incomplete_simready = simready_manager.make_table_simready(
                        MakeTableSimreadyRequest(
                            input_path=incomplete_table_raw_path,
                            output_path=glb_gen_dir
                            / "multi_object_layouts_simready"
                            / f"{incomplete_table_id}_simready.glb",
                        )
                    )
                    generated_item.update(
                        {
                            "image_path": relative_path(
                                incomplete_rgba, output_root
                            ),
                            "raw_geometry_path": relative_path(
                                str(incomplete_table_raw_path), output_root
                            ),
                            "generated_table_raw_geometry_path": relative_path(
                                incomplete_raw_glb, output_root
                            ),
                            "simready_geometry_path": relative_path(
                                str(incomplete_simready.output_path),
                                output_root,
                            ),
                            "mesh_path": relative_path(
                                str(incomplete_simready.output_path),
                                output_root,
                            ),
                            "raw_to_simready_glb_matrix": (
                                incomplete_simready.transform_matrix
                            ),
                            "transform_matrix": np.eye(
                                4, dtype=np.float64
                            ).tolist(),
                            "table_asset_source": "description_generated",
                            "complete_table_description": incomplete_table_desc,
                        }
                    )
                generated_table = generated_item
            else:
                generated_objects.append(generated_item)
    except Exception as exc:
        status = "failed"
        failure_reason = traceback.format_exc()
        log_warning(f"image object geometry generation failed error={exc}")

    if generated_objects:
        _estimate_image_scene_metric_scales(
            objects=generated_objects,
            bbox_name_image_path=segments_data.get("bbox_name_image_path"),
            output_dir=glb_gen_dir,
            output_root=output_root,
            llm=llm,
        )

    alignment_result: dict[str, Any] | None = None
    if generated_table is not None and generated_objects:
        try:
            alignment_result = _export_support_aligned_layout_glbs(
                table=generated_table,
                objects=generated_objects,
                spatial_relations=spatial_relations,
                original_image_path=Path(original_image_path)
                if original_image_path
                else None,
                llm=llm,
                output_dir=aligned_dir,
                output_root=output_root,
            )
            aligned_object_by_id = {
                item["id"]: item for item in alignment_result["objects"]
            }
            for generated_object in generated_objects:
                aligned_object = aligned_object_by_id.get(generated_object["id"])
                if aligned_object is not None:
                    generated_object["aligned_geometry_path"] = aligned_object[
                        "aligned_geometry_path"
                    ]
        except Exception as exc:
            status = "failed"
            failure_reason = traceback.format_exc()
            log_warning(f"image object alignment failed error={exc}")
            alignment_result = {
                "status": "failed",
                "reason": failure_reason,
            }

    manifest_paths = _write_multi_object_layout_manifests(
        glb_gen_dir=glb_gen_dir,
        output_root=output_root,
        table=generated_table,
        objects=generated_objects,
        alignment=alignment_result,
    )
    table_fields = (
        "id",
        "name",
        "status",
        "is_complete_visible_table",
        "complete_table_description",
        "object_coverage_percent",
        "table_asset_source",
        "support_normal_source",
        "image_path",
        "raw_geometry_path",
        "support_reference_geometry_path",
        "generated_table_raw_geometry_path",
        "transformed_geometry_path",
        "simready_geometry_path",
        "aligned_geometry_path",
        "mesh_path",
    )
    object_fields = (
        "id",
        "name",
        "description",
        "status",
        "image_path",
        "mesh_path",
        "aligned_geometry_path",
        "metric_scale",
    )
    workflow_table = (
        {key: generated_table[key] for key in table_fields if key in generated_table}
        if generated_table is not None
        else None
    )
    workflow_objects = [
        {key: item[key] for key in object_fields if key in item}
        for item in generated_objects
    ]
    if workflow_table is not None and workflow_table.get("status") != "ok":
        workflow_table["status"] = "failed"
    for item in workflow_objects:
        if item.get("status") != "ok":
            item["status"] = "failed"
    workflow_alignment = (
        {
            key: alignment_result[key]
            for key in ("status", "final_clutter_2d_aabb_cm")
            if key in alignment_result
        }
        if alignment_result is not None
        else None
    )
    result = {
        "status": status,
        "table": workflow_table,
        "objects": workflow_objects,
        "alignment": workflow_alignment,
        "manifests": manifest_paths,
    }
    if failure_reason:
        result["reason"] = failure_reason
    log_info(
        "image object layout generation completed "
        f"status={status} generated={len(generated_objects)}"
    )
    return result


def _estimate_image_scene_metric_scales(
    *,
    objects: list[dict[str, Any]],
    bbox_name_image_path: Any,
    output_dir: Path,
    output_root: Path,
    llm: Any | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "skipped",
        "method": "image_scene_bbox_name_vlm_candidate_shape_ratio_median_scale",
        "bbox_name_image_path": str(bbox_name_image_path or ""),
        "objects": [],
    }
    try:
        if not METRIC_SCALE_ENABLED:
            result["reason"] = "metric_scale_disabled"
            MetricScaleManager.set_for_all_objects(
                objects=objects,
                status="skipped",
                reason="metric_scale_disabled",
                method=str(result["method"]),
            )
            return result
        if llm is None:
            result["reason"] = "missing_llm"
            MetricScaleManager.set_for_all_objects(
                objects=objects,
                status="skipped",
                reason="missing_llm",
                method=str(result["method"]),
            )
            return result

        bbox_image = _resolve_generated_path(bbox_name_image_path, output_root)
        if not bbox_image.is_file():
            result["reason"] = "missing_bbox_name_image"
            MetricScaleManager.set_for_all_objects(
                objects=objects,
                status="skipped",
                reason="missing_bbox_name_image",
                method=str(result["method"]),
            )
            return result

        metric_objects = _build_metric_scale_inputs(
            objects=objects,
            output_root=output_root,
        )
        result["objects"] = MetricScaleManager.object_prompt_payload(metric_objects)
        metric_result = MetricScaleManager.estimate_metric_scales(
            EstimateMetricScalesRequest(
                objects=metric_objects,
                messages=build_image_metric_scale_messages(
                    bbox_name_image_path=bbox_image,
                    objects_json=result["objects"],
                ),
                schema=IMAGE_METRIC_SCALE_JSON_SCHEMA,
                llm=llm,
                context="Image scene metric scale estimate",
                method=str(result["method"]),
                step_name=UNIFIED_SCENE_STEP,
                raw_output_path=output_dir / "image_metric_scale_raw_model_output.json",
            )
        )
        estimates = metric_result.object_scales
        MetricScaleManager.apply_to_objects(objects=objects, object_scales=estimates)
        result.update(
            {
                "status": "ok",
                "object_scales": estimates,
                "unit_note": (
                    "Per-object scale_factor is not baked into simready GLBs. "
                    "Image alignment later computes one clamped global clutter "
                    "scale from these per-object estimates, on top of SAM3D "
                    "per-object layout scale."
                ),
            }
        )
    except Exception:
        result.update({"status": "failed", "reason": traceback.format_exc()})
        MetricScaleManager.set_for_all_objects(
            objects=objects,
            status="failed",
            reason="image_scene_metric_scale_failed",
            method=str(result["method"]),
        )
    return result


def _build_metric_scale_inputs(
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
