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

import numpy as np

from embodichain.gen_sim.prompt2scene.llms.llm_output import (
    call_structured_json_model_step,
)
from embodichain.gen_sim.prompt2scene.prompts.builders import (
    build_up_down_flip_check_messages,
)
from embodichain.gen_sim.prompt2scene.agent_tools.managers.simready_manager import (
    GlobalMetricScaleRequest,
    MetricScaleManager,
)
from embodichain.gen_sim.prompt2scene.prompts.schemas import (
    UP_DOWN_FLIP_CHECK_JSON_SCHEMA,
)

UP_DOWN_FLIP_CHECK_CONFIDENCE_THRESHOLD = 0.6
UNIFIED_SCENE_STEP = "unified_scene"
from embodichain.gen_sim.prompt2scene.agent_tools.managers.blender_rendering_manager import (
    BlenderRenderingManager,
    RenderObjectScenesRequest,
)
from embodichain.gen_sim.prompt2scene.agent_tools.managers.matplotlib_manager import (
    MatplotlibManager,
    RenderImageComparisonRequest,
)
from embodichain.gen_sim.prompt2scene.agent_tools.managers.geometry_manager import (
    GeometryManager,
)
from embodichain.gen_sim.prompt2scene.utils.io import (
    relative_path,
)
from embodichain.gen_sim.prompt2scene.agent_tools.managers.layout_manager import (
    LayoutManager,
)

__all__ = ["_export_support_aligned_layout_glbs"]


def _export_support_aligned_layout_glbs(
    *,
    table: dict[str, Any],
    objects: list[dict[str, Any]],
    spatial_relations: list[dict[str, Any]],
    original_image_path: Path | None,
    llm: Any | None,
    output_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Export layout-baked GLBs aligned by support normal and left-right order."""
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("Support-aligned GLB export requires trimesh.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    support_reference_path = _resolve_generated_path(
        table.get("support_reference_geometry_path") or table.get("raw_geometry_path"),
        output_root,
    )
    object_paths = [
        (
            str(item["id"]),
            _resolve_generated_path(item.get("raw_geometry_path"), output_root),
            item.get("transform_matrix"),
        )
        for item in objects
        if item.get("raw_geometry_path") and item.get("transform_matrix")
    ]
    if not support_reference_path.is_file():
        raise FileNotFoundError(
            f"Support reference table GLB not found: {support_reference_path}"
        )
    support_reference_transform = GeometryManager.matrix_from_json(
        table.get("support_reference_transform_matrix")
        or table.get("transform_matrix"),
        name="table.support_reference_transform_matrix",
    )
    if not object_paths:
        raise ValueError("No raw object GLBs with transform matrices available.")

    support_reference_scene = trimesh.load(support_reference_path, force="scene")
    support_reference_scene.apply_transform(support_reference_transform)
    object_scenes = [
        (
            object_id,
            GeometryManager.load_scene_with_transform(
                path=path,
                transform=GeometryManager.matrix_from_json(
                    transform,
                    name=f"{object_id}.transform_matrix",
                ),
                trimesh=trimesh,
            ),
        )
        for object_id, path, transform in object_paths
    ]
    table_mesh = GeometryManager.scene_to_mesh(support_reference_scene, trimesh=trimesh)
    support_normal = GeometryManager.estimate_support_normal(table_mesh)
    normal_alignment = GeometryManager.rotation_between_vectors(
        support_normal,
        np.array([0.0, 0.0, 1.0]),
    )

    for _, scene in object_scenes:
        scene.apply_transform(normal_alignment)

    object_bounds = [
        GeometryManager.scene_to_mesh(scene, trimesh=trimesh).bounds
        for _, scene in object_scenes
    ]
    clutter_bounds = np.vstack(
        [
            np.vstack([bounds[0] for bounds in object_bounds]).min(axis=0),
            np.vstack([bounds[1] for bounds in object_bounds]).max(axis=0),
        ]
    )
    clutter_center = 0.5 * (clutter_bounds[0] + clutter_bounds[1])
    center_transform = np.eye(4, dtype=np.float64)
    center_transform[:3, 3] = [
        -float(clutter_center[0]),
        -float(clutter_center[1]),
        -float(clutter_center[2]),
    ]

    for _, scene in object_scenes:
        scene.apply_transform(center_transform)

    alignment_candidates = _build_up_down_alignment_candidates(
        object_scenes=object_scenes,
        support_normal=support_normal,
        normal_alignment=normal_alignment,
        spatial_relations=spatial_relations,
        trimesh=trimesh,
    )
    vlm_check_dir = output_dir / "vlm_up_down_flip_check"
    up_down_flip_check_result = _run_aligned_up_down_flip_vlm_check(
        llm=llm,
        original_image_path=original_image_path,
        normal_object_scenes=alignment_candidates["normal"]["object_scenes"],
        flipped_object_scenes=alignment_candidates["flipped"]["object_scenes"],
        output_dir=vlm_check_dir,
    )
    selected_variant = str(
        up_down_flip_check_result.get("selected_variant") or "normal"
    )
    if selected_variant not in alignment_candidates:
        selected_variant = "normal"
    selected_candidate = alignment_candidates[selected_variant]
    object_scenes = selected_candidate["object_scenes"]
    selected_extra_transform = selected_candidate["extra_transform"]
    apply_up_down_flip = selected_variant == "flipped"
    complete_table_relative_scale_hint = _complete_table_relative_scale_hint(
        table=table,
        support_reference_scene=support_reference_scene,
        object_scenes=object_scenes,
        table_alignment_matrix=selected_extra_transform
        @ center_transform
        @ normal_alignment,
        trimesh=trimesh,
    )

    global_metric_scale = MetricScaleManager.compute_global_from_object_scenes(
        GlobalMetricScaleRequest(
            objects=objects,
            object_scenes=object_scenes,
        )
    )
    metric_scale_transform = GeometryManager.scale_transform(
        global_metric_scale["scale_factor"]
    )
    if float(global_metric_scale["scale_factor"]) != 1.0:
        for _, scene in object_scenes:
            scene.apply_transform(metric_scale_transform)

    footprint_result = LayoutManager.settle_and_pack_object_footprints(
        object_scenes=object_scenes,
        output_dir=output_dir / "footprint_layout",
        output_root=output_root,
        trimesh=trimesh,
    )
    object_scenes = footprint_result["object_scenes"]

    output_axis_transform = GeometryManager.z_up_to_glb_y_up_transform()
    object_outputs = []
    for object_id, scene in object_scenes:
        object_output = output_dir / f"{object_id}_aligned.glb"
        GeometryManager.copy_scene_with_transform(
            scene,
            output_axis_transform,
        ).export(object_output)
        object_outputs.append(
            {
                "id": object_id,
                "aligned_geometry_path": relative_path(str(object_output), output_root),
            }
        )

    alignment_matrix = selected_extra_transform @ center_transform @ normal_alignment
    scaled_alignment_matrix = metric_scale_transform @ alignment_matrix
    final_clutter_aabb_2d_cm = LayoutManager.object_scenes_xy_aabb_manifest(
        object_scenes=object_scenes,
        trimesh=trimesh,
        unit_scale=100.0,
        unit="cm",
    )
    return {
        "status": "ok",
        "output_dir": relative_path(str(output_dir), output_root),
        "support_normal": support_normal.tolist(),
        "clutter_aabb_center_before_centering": clutter_center.tolist(),
        "alignment_matrix": scaled_alignment_matrix.tolist(),
        "pre_metric_scale_alignment_matrix": alignment_matrix.tolist(),
        "global_metric_scale": global_metric_scale,
        "final_clutter_2d_aabb_cm": final_clutter_aabb_2d_cm,
        "complete_table_relative_scale_hint": complete_table_relative_scale_hint,
        "internal_up_axis": [0.0, 0.0, 1.0],
        "glb_output_up_axis": [0.0, 1.0, 0.0],
        "glb_output_axis_transform": output_axis_transform.tolist(),
        "selected_up_down_variant": selected_variant,
        "applied_up_down_flip": apply_up_down_flip,
        "selected_extra_transform": selected_extra_transform.tolist(),
        "object_alignment_matrices": {
            object_id: (object_transform @ scaled_alignment_matrix).tolist()
            for object_id, object_transform in footprint_result[
                "object_layout_transforms"
            ].items()
        },
        "footprint_layout": footprint_result["manifest"],
        "yaw_sampling": {
            "sample_count_per_variant": 360,
            "score_type": "center_left_of_hard_count",
            "top_view_plane": "XY",
            "yaw_axis": "Z",
            "left_right_axis": "X",
            "front_back_axis": "Y",
            "front_direction": "+Y",
            "normal": alignment_candidates["normal"]["yaw_metadata"],
            "flipped": alignment_candidates["flipped"]["yaw_metadata"],
        },
        "up_down_flip_check": up_down_flip_check_result,
        "objects": object_outputs,
    }


def _complete_table_relative_scale_hint(
    *,
    table: dict[str, Any],
    support_reference_scene: Any,
    object_scenes: list[tuple[str, Any]],
    table_alignment_matrix: np.ndarray,
    trimesh: Any,
) -> dict[str, Any]:
    if not table.get("is_complete_visible_table"):
        return {
            "status": "skipped",
            "reason": "table_is_not_complete_visible",
        }
    if not object_scenes:
        return {
            "status": "skipped",
            "reason": "missing_object_scenes",
        }
    try:
        table_scene = GeometryManager.copy_scene_with_transform(
            support_reference_scene,
            table_alignment_matrix,
        )
        raw_clutter_bounds = GeometryManager.table_fit_scene_union_bounds(
            [scene for _, scene in object_scenes],
            trimesh=trimesh,
        )
        raw_clutter_size_xy = GeometryManager.xy_aabb_size(raw_clutter_bounds)
        raw_table_mesh = GeometryManager.scene_to_mesh(table_scene, trimesh=trimesh)
        raw_table_support = GeometryManager.detect_table_fit_support_quad(
            raw_table_mesh,
            target_aspect=float(
                raw_clutter_size_xy[0] / max(float(raw_clutter_size_xy[1]), 1.0e-6)
            ),
        )
        raw_table_support_size_xy = np.asarray(
            raw_table_support["size_xy"],
            dtype=np.float64,
        )
        ratio_xy = raw_table_support_size_xy / np.maximum(
            raw_clutter_size_xy,
            1.0e-6,
        )
        if not np.all(np.isfinite(ratio_xy)) or np.any(ratio_xy <= 0.0):
            return {
                "status": "skipped",
                "reason": "invalid_raw_relative_size",
            }
        return {
            "status": "ok",
            "method": "complete_table_sam3d_raw_support_to_clutter_ratio",
            "raw_table_support_size_xy": raw_table_support_size_xy.tolist(),
            "raw_clutter_size_xy": raw_clutter_size_xy.tolist(),
            "support_to_clutter_size_ratio_xy": ratio_xy.tolist(),
            "raw_table_support_quad": raw_table_support,
            "note": (
                "Ratio is unitless and is computed before metric scaling; "
                "table fit later applies one uniform XYZ scale to the simready table."
            ),
        }
    except Exception:
        return {
            "status": "failed",
            "reason": traceback.format_exc(),
        }


def _build_up_down_alignment_candidates(
    *,
    object_scenes: list[tuple[str, Any]],
    support_normal: np.ndarray,
    normal_alignment: np.ndarray,
    spatial_relations: list[dict[str, Any]],
    trimesh: Any,
) -> dict[str, dict[str, Any]]:
    flip_transform = GeometryManager.support_normal_flip_transform(
        support_normal=support_normal,
        normal_alignment=normal_alignment,
    )
    directional_relations = _spatial_directional_relations(spatial_relations)
    candidates: dict[str, dict[str, Any]] = {}
    for variant, pre_yaw_transform in [
        ("normal", np.eye(4, dtype=np.float64)),
        ("flipped", flip_transform),
    ]:
        candidate_object_scenes = [
            (
                object_id,
                GeometryManager.copy_scene_with_transform(scene, pre_yaw_transform),
            )
            for object_id, scene in object_scenes
        ]
        object_bounds = {
            object_id: np.asarray(
                GeometryManager.scene_to_mesh(scene, trimesh=trimesh).bounds,
                dtype=np.float64,
            )
            for object_id, scene in candidate_object_scenes
        }
        yaw_metadata = _best_spatial_yaw(
            object_bounds=object_bounds,
            relations=directional_relations,
        )
        yaw_transform = GeometryManager.z_yaw_transform(
            float(yaw_metadata["yaw_degrees"]),
        )
        for _, scene in candidate_object_scenes:
            scene.apply_transform(yaw_transform)
        candidates[variant] = {
            "object_scenes": candidate_object_scenes,
            "pre_yaw_transform": pre_yaw_transform,
            "yaw_transform": yaw_transform,
            "extra_transform": yaw_transform @ pre_yaw_transform,
            "yaw_metadata": yaw_metadata,
        }
    return candidates


def _best_spatial_yaw(
    *,
    object_bounds: dict[str, np.ndarray],
    relations: list[dict[str, str]],
) -> dict[str, Any]:
    if not relations:
        return {
            "yaw_degrees": 0,
            "score": 0,
            "raw_gap_sum": 0.0,
            "relation_count": 0,
            "score_type": "center_left_of_hard_count",
        }

    object_centers = {
        object_id: GeometryManager.aabb_center(bounds)
        for object_id, bounds in object_bounds.items()
    }
    best_yaw = 0
    best_score = -1
    best_raw_gap_sum = float("-inf")
    best_relation_scores: list[dict[str, Any]] = []
    for yaw_degrees in range(360):
        rotation = GeometryManager.z_yaw_transform(float(yaw_degrees))
        rotated_centers = {
            object_id: _transform_point(rotation, center)
            for object_id, center in object_centers.items()
        }
        score, raw_gap_sum, relation_scores = _center_left_of_score(
            centers=rotated_centers,
            relations=relations,
        )
        if score > best_score or (
            score == best_score and raw_gap_sum > best_raw_gap_sum
        ):
            best_yaw = yaw_degrees
            best_score = score
            best_raw_gap_sum = raw_gap_sum
            best_relation_scores = relation_scores
    return {
        "yaw_degrees": best_yaw,
        "score": best_score,
        "raw_gap_sum": best_raw_gap_sum,
        "relation_count": len(relations),
        "score_type": "center_left_of_hard_count",
        "relation_scores": best_relation_scores,
    }


def _spatial_directional_relations(
    spatial_relations: list[dict[str, Any]],
) -> list[dict[str, str]]:
    relations: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for relation in spatial_relations:
        subject = str(relation.get("subject") or "")
        object_id = str(relation.get("object") or "")
        relation_name = str(relation.get("relation") or "")
        if (
            not subject
            or not object_id
            or subject == object_id
            or relation_name != "left_of"
        ):
            continue
        key = (subject, relation_name, object_id)
        if key in seen:
            continue
        seen.add(key)
        relations.append(
            {
                "subject": subject,
                "relation": relation_name,
                "object": object_id,
            }
        )
    return relations


def _center_left_of_score(
    centers: dict[str, np.ndarray],
    relations: list[dict[str, str]],
) -> tuple[int, float, list[dict[str, Any]]]:
    score = 0
    raw_gap_sum = 0.0
    relation_scores: list[dict[str, Any]] = []
    for relation in relations:
        subject = relation["subject"]
        object_id = relation["object"]
        if subject not in centers or object_id not in centers:
            continue
        subject_center = centers[subject]
        object_center = centers[object_id]
        gap = float(object_center[0] - subject_center[0])
        relation_score = 1 if gap > 0.0 else 0
        score += relation_score
        raw_gap_sum += gap
        relation_scores.append(
            {
                "subject": subject,
                "relation": "left_of",
                "object": object_id,
                "gap": gap,
                "score": relation_score,
            }
        )
    return score, raw_gap_sum, relation_scores


def _transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    homogeneous = np.ones(4, dtype=np.float64)
    homogeneous[:3] = point
    return (transform @ homogeneous)[:3]


def _resolve_generated_path(value: Any, output_root: Path) -> Path:
    path = Path(str(value or "")).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (output_root / path).resolve()


def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    try:
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:
        pass


def _run_aligned_up_down_flip_vlm_check(
    *,
    llm: Any | None,
    original_image_path: Path | None,
    normal_object_scenes: list[tuple[str, Any]],
    flipped_object_scenes: list[tuple[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "status": "skipped",
        "applied_up_down_flip": False,
        "confidence_threshold": UP_DOWN_FLIP_CHECK_CONFIDENCE_THRESHOLD,
        "reason": "",
    }
    if not normal_object_scenes or not flipped_object_scenes:
        result["reason"] = "missing_object_scenes"
        _write_json_file(output_dir / "up_down_flip_selection.json", result)
        return result

    try:
        normal_render_path = output_dir / "normal_object_only_front_oblique_view.png"
        flipped_render_path = output_dir / "flipped_object_only_front_oblique_view.png"
        comparison_image_path = output_dir / "numbered_up_down_candidates.png"
        BlenderRenderingManager().render_object_scenes(
            RenderObjectScenesRequest(
                object_scenes=normal_object_scenes,
                output_path=normal_render_path,
            )
        )
        BlenderRenderingManager().render_object_scenes(
            RenderObjectScenesRequest(
                object_scenes=flipped_object_scenes,
                output_path=flipped_render_path,
            )
        )
        MatplotlibManager(figsize=(12, 6), dpi=180).render_image_comparison(
            RenderImageComparisonRequest(
                first_image_path=normal_render_path,
                second_image_path=flipped_render_path,
                output_path=comparison_image_path,
            )
        )
        if llm is None:
            result["reason"] = "missing_llm"
            _write_json_file(output_dir / "up_down_flip_selection.json", result)
            return result
        if original_image_path is None or not original_image_path.is_file():
            result["reason"] = "missing_original_image"
            _write_json_file(output_dir / "up_down_flip_selection.json", result)
            return result

        raw_model_output = call_structured_json_model_step(
            llm=llm,
            schema=UP_DOWN_FLIP_CHECK_JSON_SCHEMA,
            messages=build_up_down_flip_check_messages(
                original_image_path=original_image_path,
                comparison_image_path=comparison_image_path,
            ),
            context="Unified scene aligned up-down flip check",
            attempt_count=0,
            raw_output_writer=lambda payload: _write_json_file(
                output_dir / "vlm_flip_check_result.json",
                payload,
            ),
        )
        confidence = float(raw_model_output.get("confidence", 0.0))
        selected_number = int(raw_model_output.get("selected_number", 1))
        if selected_number not in {1, 2}:
            selected_number = 1
        model_selected_variant = "flipped" if selected_number == 2 else "normal"
        should_apply = (
            model_selected_variant == "flipped"
            and confidence >= UP_DOWN_FLIP_CHECK_CONFIDENCE_THRESHOLD
        )
        selected_variant = "flipped" if should_apply else "normal"
        selected_number = 2 if selected_variant == "flipped" else 1
        result.update(
            {
                "status": "ok",
                "selected_number": selected_number,
                "selected_variant": selected_variant,
                "applied_up_down_flip": should_apply,
                "model_selected_number": raw_model_output.get("selected_number"),
                "model_selected_variant": model_selected_variant,
                "confidence": confidence,
                "reason": str(raw_model_output.get("reason", "")),
            }
        )
        _write_json_file(output_dir / "up_down_flip_selection.json", result)
        return result
    except Exception:
        result.update(
            {
                "status": "failed",
                "reason": traceback.format_exc(),
            }
        )
        _write_json_file(output_dir / "up_down_flip_selection.json", result)
        return result
