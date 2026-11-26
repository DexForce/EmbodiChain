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

import json
from collections.abc import Mapping
from pathlib import Path
from subprocess import TimeoutExpired
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from embodichain.gen_sim.scene_engine.pipeline.utils.scene_generation_utils import (
    layout_object_to_transform_matrix,
    load_glb_mesh,
    transform_matrix_to_layout_object,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_heading_preview import (
    render_scene_heading_comparison,
    SCENE_HEADING_CANDIDATES_FILENAME,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.simready_processor_utils import (
    _strip_json_code_fence,
)
from embodichain.gen_sim.services.compat import vision_for_role
from embodichain.gen_sim.services.errors import ServiceError

if TYPE_CHECKING:
    from embodichain.gen_sim.services.compat import VisionClient

__all__: list[str] = []

_HEADINGS = (0, 90, 180, 270)
_MIN_IMAGE_GAP = 0.05
_MIN_PROJECTED_GAP = 0.02
_MIN_SCORE_SEPARATION = 0.05
_HORIZONTAL_ENDPOINT_RATIO = np.sqrt(0.5)
_SELECTION_FILENAME = "scene_heading_selection.json"
_SYSTEM_PROMPT = """Choose which labeled render most closely matches the overall
viewing direction of the original photograph. Each candidate uses the same fixed
camera and shows the same settled table and all objects rotated together around
the vertical axis. The panel marked Program proposal is the current view.

Compare the tabletop orientation, broad left/right and near/far arrangement of
recognizable object groups, and reliable directional cues such as the long
axis and location of a clearly visible elongated object. A single strong cue
can be enough when there is only one recognizable object. Allow for different
perspective, occlusion, missing objects, and errors in individual reconstructed
objects' positions, poses, shapes, or colors. Ignore robot hardware and its
shadows. Prefer the view that agrees with the overall layout even if some
individual objects disagree. Select a different panel only if it clearly
matches the source better overall. If the evidence is inconclusive, return null.

Return only a JSON object with exactly "yaw_degrees" and "reason". Use the
selected panel's absolute label (0, 90, 180, or 270) or null for "yaw_degrees".
Give a short visual justification in "reason"."""

_REASONS = {
    "ranked": "Selected the highest-scoring eligible heading from the mask centers.",
    "fewer_than_two_masks": "Kept the current heading because fewer than two masks are usable.",
    "insufficient_image_gap": "Kept the current heading because the source mask centers are too close.",
    "insufficient_endpoint_projection": "Kept the current heading because no candidate maintains the visible left-to-right span.",
    "ambiguous": "Kept the current heading because the candidate scores do not support a clear choice.",
}


def review_scene_heading(
    *,
    image_path: Path,
    table_layout: dict[str, object],
    assets_layout: list[dict[str, object]],
    geometry_root: Path,
    debug_output_root: Path,
    mask_paths: Mapping[str, str | Path] | None = None,
    render_candidates: bool = False,
) -> tuple[int, list[float]]:
    """Select a quarter turn from mask geometry, with 0° on ambiguous evidence.

    Candidate renders are optional diagnostics and do not affect the selection.
    The returned XY pivot belongs to the Z-up world frame.
    """
    debug_output_root.mkdir(parents=True, exist_ok=True)
    table_id = str(table_layout["id"])
    table_mesh = load_glb_mesh(geometry_root / f"{table_id}.glb")
    table_mesh.apply_transform(layout_object_to_transform_matrix(table_layout))
    center_x, _, center_z = table_mesh.bounds.mean(axis=0)
    pivot_xy = [float(center_x), -float(center_z)]
    baseline = [table_layout, *assets_layout]
    object_ids = [layout["id"] for layout in baseline]
    audit_path = debug_output_root / _SELECTION_FILENAME
    rotations = {
        angle: Rotation.from_euler("y", angle, degrees=True).as_matrix()
        for angle in _HEADINGS
    }
    evidence = _score_headings_from_masks(
        image_path=image_path,
        assets_layout=assets_layout,
        mask_paths=mask_paths or {},
        geometry_root=geometry_root,
        table_span=float(max(table_mesh.extents[0], table_mesh.extents[2])),
        rotations=rotations,
    )
    suggested = evidence.get("suggested_yaw_degrees")
    angle = suggested if isinstance(suggested, int) else 0
    selection = {
        "yaw_degrees": angle,
        "reason": _REASONS[str(evidence["status"])],
        "method": "mask_geometry",
        "pivot_xy_z_up_m": pivot_xy,
        "object_ids": object_ids,
        "source_image": str(image_path),
        "mask_evidence": evidence,
    }
    if render_candidates:
        preview = render_scene_heading_comparison(
            image_path=image_path,
            candidate_layouts=_heading_candidates(baseline, pivot_xy),
            geometry_root=geometry_root,
            output_path=debug_output_root / "scene_heading_comparison.png",
        )
        selection["candidate_image"] = str(
            preview.with_name(SCENE_HEADING_CANDIDATES_FILENAME)
        )
    _write_selection(audit_path, selection)
    return angle, pivot_xy


def review_settled_scene_heading(
    *,
    image_path: Path,
    table_layout: dict[str, object],
    assets_layout: list[dict[str, object]],
    geometry_root: Path,
    debug_output_root: Path,
    vision_client: VisionClient,
    proposed_yaw_degrees: int,
    pivot_xy: list[float],
) -> int:
    """Review the program's proposal with a shared-camera render of the settled scene."""
    audit_path = debug_output_root / _SELECTION_FILENAME
    baseline = [table_layout, *assets_layout]
    selection = (
        json.loads(audit_path.read_text(encoding="utf-8"))
        if audit_path.is_file()
        else {
            "source_image": str(image_path),
            "pivot_xy_z_up_m": pivot_xy,
            "object_ids": [layout["id"] for layout in baseline],
            "reason": "Kept the program proposal.",
        }
    )
    user_prompt = (
        f"The program proposes {proposed_yaw_degrees} degrees. Compare the original "
        "photograph with the labeled rendered candidates to select the closest overall view."
    )
    review: dict[str, object] = {
        "system_prompt": _SYSTEM_PROMPT,
        "user_prompt": user_prompt,
    }
    selection.update(
        program_yaw_degrees=proposed_yaw_degrees,
        yaw_degrees=proposed_yaw_degrees,
        method="mask_geometry",
        model_images=[],
        vision_review=review,
    )
    debug_output_root.mkdir(parents=True, exist_ok=True)
    try:
        preview = render_scene_heading_comparison(
            image_path=image_path,
            candidate_layouts=_heading_candidates(baseline, pivot_xy),
            geometry_root=geometry_root,
            output_path=debug_output_root / "scene_heading_comparison.png",
            proposed_yaw_degrees=proposed_yaw_degrees,
        )
    except (OSError, RuntimeError, TimeoutExpired, ValueError) as exc:
        review.update(status="render_failed", error=str(exc))
        _write_selection(audit_path, selection)
        return proposed_yaw_degrees

    candidates_path = preview.with_name(SCENE_HEADING_CANDIDATES_FILENAME)
    selection.update(
        candidate_image=str(candidates_path), comparison_image=str(preview)
    )
    client = vision_for_role(vision_client, "scene_understanding")
    complete_images = getattr(client, "complete_images", None)
    yaw = None
    try:
        if callable(complete_images):
            images = [image_path, candidates_path]
            selection["model_images"] = [str(path) for path in images]
            raw = complete_images(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                image_paths=images,
            )
        else:
            selection["model_images"] = [str(preview)]
            raw = client.complete(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                image_path=preview,
            )
    except ServiceError as exc:
        review.update(status="service_error", error=exc.code)
    else:
        review["raw_response"] = raw
        try:
            yaw, reason = _parse_heading_review(raw)
        except ValueError as exc:
            review.update(status="invalid_response", error=str(exc))
        else:
            review.update(
                status="inconclusive" if yaw is None else "selected",
                yaw_degrees=yaw,
                reason=reason,
            )
            if yaw is not None:
                selection.update(
                    yaw_degrees=yaw, reason=reason, method="mask_geometry_vlm"
                )
    _write_selection(audit_path, selection)
    return proposed_yaw_degrees if yaw is None else yaw


def _heading_candidates(
    baseline: list[dict[str, object]], pivot_xy: list[float]
) -> dict[int, list[dict[str, object]]]:
    poses = [layout_object_to_transform_matrix(layout) for layout in baseline]
    pivot = np.array([pivot_xy[0], 0.0, -pivot_xy[1]])
    candidates = {0: baseline}
    for angle in _HEADINGS[1:]:
        rotation = Rotation.from_euler("y", angle, degrees=True).as_matrix()
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = pivot - rotation @ pivot
        candidates[angle] = [
            transform_matrix_to_layout_object(str(layout["id"]), transform @ pose)
            for layout, pose in zip(baseline, poses, strict=True)
        ]
    return candidates


def _parse_heading_review(raw: str) -> tuple[int | None, str]:
    result = json.loads(_strip_json_code_fence(raw))
    if not isinstance(result, dict) or set(result) != {"yaw_degrees", "reason"}:
        raise ValueError("Expected exactly yaw_degrees and reason.")
    yaw, reason = result["yaw_degrees"], result["reason"]
    if yaw is not None and (type(yaw) is not int or yaw not in _HEADINGS):
        raise ValueError("Expected a quarter-turn angle or null.")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Expected a nonempty visual reason.")
    return yaw, reason.strip()


def _score_headings_from_masks(
    *,
    image_path: Path,
    assets_layout: list[dict[str, object]],
    mask_paths: Mapping[str, str | Path],
    geometry_root: Path,
    table_span: float,
    rotations: Mapping[int, np.ndarray],
) -> dict[str, object]:
    """Score mask-center order across both image axes under the preview camera."""
    if len(mask_paths) < 2:
        return {"status": "fewer_than_two_masks", "mask_count": len(mask_paths)}
    with Image.open(image_path) as image:
        size = image.size
    by_id = {str(layout["id"]): layout for layout in assets_layout}
    mask_centers = {}
    for uid, path in sorted(mask_paths.items()):
        if uid not in by_id or not Path(path).is_file():
            continue
        with Image.open(path) as mask:
            if mask.size != size:
                continue
            pixels_mask = np.asarray(mask.convert("L")) != 0
        columns = pixels_mask.sum(axis=0)
        pixels = int(columns.sum())
        if pixels:
            x = int(np.searchsorted(np.cumsum(columns), (pixels + 1) // 2))
            y = int(
                np.searchsorted(np.cumsum(pixels_mask.sum(axis=1)), (pixels + 1) // 2)
            )
            mask_centers[uid] = [x, y]
    evidence: dict[str, object] = {
        "mask_count": len(mask_centers),
        "source_mask_centers_px": mask_centers,
    }
    if len(mask_centers) < 2:
        return {**evidence, "status": "fewer_than_two_masks"}
    ids = list(mask_centers)
    first, second = np.triu_indices(len(ids), k=1)
    image_centers = np.asarray(list(mask_centers.values()), dtype=float) / size
    image_delta = image_centers[second] - image_centers[first]
    informative = np.abs(image_delta) >= _MIN_IMAGE_GAP
    if not np.any(informative):
        return {**evidence, "status": "insufficient_image_gap"}
    world_centers = []
    for uid in mask_centers:
        local_center = load_glb_mesh(geometry_root / f"{uid}.glb").bounds.mean(axis=0)
        transform = layout_object_to_transform_matrix(by_id[uid])
        world_centers.append(transform[:3, :3] @ local_center + transform[:3, 3])
    # The front-elevated preview camera maps Y-up X to image right and maps
    # positive Z toward the foreground. Taller centers project upward.
    elevation = np.deg2rad(25.0)
    camera = np.array([[1.0, 0.0, 0.0], [0.0, -np.cos(elevation), np.sin(elevation)]])
    scores = {}
    axis_scores = {}
    weights = np.where(informative, np.abs(image_delta), 0.0)
    axis_weights = weights.sum(axis=0)
    centers = np.asarray(world_centers)
    projected_gap = max(_MIN_PROJECTED_GAP * table_span, 1e-6)
    eligible = list(rotations)
    left = int(np.argmin(image_centers[:, 0]))
    right = int(np.argmax(image_centers[:, 0]))
    source_dx, source_dy = image_centers[right] - image_centers[left]
    if source_dx >= _MIN_IMAGE_GAP:
        endpoint_delta = centers[right] - centers[left]
        horizontal_gaps = {
            angle: float((rotation @ endpoint_delta)[0])
            for angle, rotation in rotations.items()
        }
        # When the source pair is predominantly horizontal, discard candidates
        # that collapse its span even though their left/right signs agree.
        minimum_gap = projected_gap
        if source_dx >= abs(source_dy):
            minimum_gap = max(
                minimum_gap, _HORIZONTAL_ENDPOINT_RATIO * max(horizontal_gaps.values())
            )
        eligible = [
            angle for angle, gap in horizontal_gaps.items() if gap >= minimum_gap
        ]
        evidence["endpoint_filter"] = {
            "left_id": ids[left],
            "right_id": ids[right],
            "source_x_gap_fraction": float(source_dx),
            "source_y_gap_fraction": float(abs(source_dy)),
            "min_projected_x_gap_m": float(minimum_gap),
            "projected_x_gap_m_by_angle": horizontal_gaps,
            "eligible_yaw_degrees": eligible,
        }
    for angle, rotation in rotations.items():
        projected = centers @ (camera @ rotation).T
        delta = projected[second] - projected[first]
        agreement = np.where(
            np.abs(delta) < projected_gap,
            0.5,
            (delta * image_delta > 0).astype(float),
        )
        support = (agreement * weights).sum(axis=0)
        scores[angle] = float(support.sum() / axis_weights.sum())
        axis_scores[angle] = {
            name: round(float(score / weight), 3) if weight else None
            for name, score, weight in zip(
                ("left_right", "near_far"), support, axis_weights, strict=True
            )
        }
    evidence.update(
        candidate_scores=scores,
        candidate_axis_agreement=axis_scores,
        informative_pair_axes=informative.sum(axis=0).tolist(),
    )
    if not eligible:
        return {**evidence, "status": "insufficient_endpoint_projection"}

    best, *remaining = sorted(eligible, key=scores.__getitem__, reverse=True)
    confident = scores[best] > 0.5 and (
        not remaining or scores[best] - scores[remaining[0]] > _MIN_SCORE_SEPARATION
    )
    return {
        **evidence,
        "status": "ranked" if confident else "ambiguous",
        "suggested_yaw_degrees": best if confident else None,
    }


def _write_selection(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
