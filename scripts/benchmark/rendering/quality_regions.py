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

"""Scene-defined diagnostic crops, selected independently of candidate errors."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

from scripts.benchmark.core.artifacts import stable_hash, write_json

__all__ = ["export_quality_regions"]


def export_quality_regions(
    scene: dict[str, Any],
    candidate: dict[str, Any],
    reference: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Save shadow-fixture, reflective-material and instance-edge comparisons.

    Regions use the frozen geometry/materials and independent reference IDs;
    candidate pixels never select a region. The shadow crop includes box
    casters and their directional-light projection on the table. It is a
    geometric diagnostic region, not a certified visible-shadow mask. A hidden
    material or fixture is explicitly unavailable, never substituted with an
    unrelated high-error crop. All panels use the same unadjusted sRGB scale.

    Args:
        scene: Frozen approved canonical scene.
        candidate: Captured arrays and camera/object metadata.
        reference: Independent RGB and canonical geometry reference.
        output_dir: New diagnostic output directory inside the evaluation.

    Returns:
        ROI definitions, availability reasons and checksummed PNG/JSON artifacts.
    """
    import numpy as np
    from PIL import Image

    from .quality import _artifact, _camera, _display_rgb, _geometry

    first, second = _display_rgb(candidate), _display_rgb(reference)
    _, labels = _geometry(reference)
    if first.shape != second.shape or first.shape[:2] != labels.shape:
        raise ValueError("Semantic crops require matched camera/output dimensions")
    output_dir.mkdir(parents=True, exist_ok=False)
    height, width = labels.shape
    objects = scene["objects"]
    transforms = candidate["metadata"]["world_from_objects"]
    visible = set(int(value) for value in np.unique(labels)) - {0}
    regions: list[dict[str, Any]] = []

    def box(mask: Any, margin: int = 8) -> list[int] | None:
        ys, xs = np.where(mask)
        return (
            [
                max(0, int(xs.min()) - margin),
                max(0, int(ys.min()) - margin),
                min(width, int(xs.max()) + margin + 1),
                min(height, int(ys.max()) + margin + 1),
            ]
            if len(xs)
            else None
        )

    # Reflection regions are selected from the authored material, without
    # inferring that any bright candidate pixel must be a reflection.
    reflectors = [
        obj
        for obj in objects
        if obj["material"]["metallic"] >= 0.5 and obj["material"]["roughness"] <= 0.3
    ]
    reflection_ids = [obj["instance_id"] for obj in reflectors]
    regions.append(
        {
            "name": "reflection",
            "selection": "authored metallic>=0.5 and roughness<=0.3",
            "instance_ids": reflection_ids,
            "box_xyxy": box(np.isin(labels, reflection_ids)),
        }
    )

    # Use the visible reference boundary closest to the image centre. Selection
    # stays deterministic even when the candidate has a large segmentation error.
    edge = np.zeros(labels.shape, dtype=bool)
    edge[1:, :] |= labels[1:, :] != labels[:-1, :]
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    ys, xs = np.where(edge)
    edge_box = None
    if len(xs):
        index = np.argmin((xs - width / 2) ** 2 + (ys - height / 2) ** 2)
        x, y = int(xs[index]), int(ys[index])
        radius = max(8, min(width, height) // 10)
        edge_box = [
            max(0, x - radius),
            max(0, y - radius),
            min(width, x + radius + 1),
            min(height, y + radius + 1),
        ]
    regions.append(
        {
            "name": "edge",
            "selection": "reference instance boundary nearest image centre",
            "box_xyxy": edge_box,
        }
    )

    table = next((obj for obj in objects if obj["uid"] == "table"), None)
    casters = [obj for obj in objects if obj["uid"].startswith("shadow_step_")]
    if not casters:
        casters = [
            obj
            for obj in objects
            if obj["uid"] in {"red_cube", "blue_block", "green_block"}
        ]
    shadow_box = None
    if table is not None and any(obj["instance_id"] in visible for obj in casters):
        table_pose = np.asarray(transforms[table["uid"]], dtype=float)
        plane_normal = table_pose[:3, 2]
        plane_point = (
            table_pose[:3, 3] + plane_normal * table["geometry"]["size"][2] / 2
        )
        direction = np.asarray(scene["lighting"]["direction"], dtype=float)
        divisor = float(direction @ plane_normal)
        points = []
        if divisor < -1e-8:
            for obj in casters:
                if obj["geometry"]["kind"] != "box":
                    continue
                pose = np.asarray(transforms[obj["uid"]], dtype=float)
                corners = np.asarray(list(itertools.product((-0.5, 0.5), repeat=3)))
                corners *= np.asarray(obj["geometry"]["size"])
                world = corners @ pose[:3, :3].T + pose[:3, 3]
                distance = (plane_point - world) @ plane_normal / divisor
                points.extend(world)
                points.extend(
                    world[distance >= 0] + distance[distance >= 0, None] * direction
                )
        if points:
            intrinsics, camera_pose = _camera(candidate["metadata"])
            camera_points = (np.asarray(points) - camera_pose[:3, 3]) @ camera_pose[
                :3, :3
            ]
            camera_points = camera_points[camera_points[:, 2] > 0]
            if len(camera_points):
                projected = camera_points @ intrinsics.T
                projected = projected[:, :2] / projected[:, 2, None]
                x0, y0 = np.floor(projected.min(axis=0) - 8).astype(int)
                x1, y1 = np.ceil(projected.max(axis=0) + 8).astype(int)
                shadow_box = [
                    max(0, int(x0)),
                    max(0, int(y0)),
                    min(width, int(x1)),
                    min(height, int(y1)),
                ]
                x0, y0, x1, y1 = shadow_box
                if (
                    x1 <= x0
                    or y1 <= y0
                    or not np.any(labels[y0:y1, x0:x1] == table["instance_id"])
                ):
                    shadow_box = None
    regions.append(
        {
            "name": "shadow",
            "selection": "box casters and directional projection on table",
            "instance_ids": [obj["instance_id"] for obj in casters],
            "box_xyxy": shadow_box,
            "limitation": "geometric diagnostic crop; not a visible-shadow segmentation or separate quality gate",
        }
    )
    artifacts = []
    for region in regions:
        bounds = region["box_xyxy"]
        region["status"] = (
            "available" if bounds is not None else "not_visible_or_unavailable"
        )
        if bounds is None:
            continue
        x0, y0, x1, y1 = bounds
        a, b = first[y0:y1, x0:x1], second[y0:y1, x0:x1]
        panels = np.concatenate((a, b, np.abs(a - b)), axis=1)
        path = output_dir / f"{region['name']}.png"
        Image.fromarray(np.round(np.clip(panels, 0, 1) * 255).astype(np.uint8)).save(
            path
        )
        region["panels"] = ["candidate", "independent_reference", "absolute_rgb_error"]
        region["artifact"] = _artifact(path, "semantic_quality_crop")
        artifacts.append(region["artifact"])
    result = {
        "schema_version": 1,
        "selection_version": "scene_geometry_v1",
        "scene_sha256": stable_hash(scene),
        "view_sha256": candidate["metadata"].get("view_sha256"),
        "run_id": candidate["metadata"].get("run_id"),
        "candidate_error_used_for_selection": False,
        "regions": regions,
    }
    path = output_dir / "regions.json"
    write_json(path, result)
    return {
        **result,
        "artifacts": [*artifacts, _artifact(path, "semantic_quality_regions")],
    }
