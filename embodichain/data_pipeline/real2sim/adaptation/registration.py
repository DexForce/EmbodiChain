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
"""Metric RGB-D registration of generated meshes, independent of Scene Engine I/O."""

from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
from .inputs import CameraObservation, RegisteredScene

__all__ = ["register_scene"]


def _load_mesh(path: Path) -> trimesh.Trimesh:
    """Load the complete transformed geometry for numerical fitting."""
    scene = trimesh.load(path, force="scene", process=False)
    parts = []
    for node in scene.graph.nodes_geometry:
        matrix, name = scene.graph[node]
        part = scene.geometry[name].copy()
        part.apply_transform(matrix)
        parts.append(part)
    return trimesh.util.concatenate(parts)


def register_scene(
    observation: CameraObservation,
    assets: list[dict],
    object_masks: dict[str, np.ndarray],
    support_surface_z: float,
    *,
    auxiliary_depth: dict[str, CameraObservation] | None = None,
) -> RegisteredScene:
    """Align the entire generated scene with one uniform similarity transform.

    Args:
        observation: Original pixel-aligned RGB-D and optical-to-base calibration.
        assets: Generated Y-up mesh paths with native Z-up poses, scales and categories.
        object_masks: Masks in original image coordinates (or equal-aspect resize).
        support_surface_z: Tabletop height in canonical Z-up mesh coordinates;
            the native table pose and scale are applied before registration.
        auxiliary_depth: Optional calibrated depth views of the same initial scene.

    Returns:
        Registered mesh descriptors and world-frame vertex samples. Geometry files
        and original calibration remain unchanged. Authored shapes, relative poses
        and materials are preserved. No physical success is inferred.
    """
    if observation.depth_m is None or observation.first_image is None:
        raise ValueError(
            "RGB-D registration requires calibrated depth; supply a registered scene otherwise"
        )
    auxiliary_depth = auxiliary_depth or {}
    k, camera = observation.intrinsics, observation.camera_to_base
    cv2.setNumThreads(2)
    invcam = np.linalg.inv(camera)
    depth = observation.depth_m
    yy, xx = np.indices(depth.shape)
    points = (
        np.stack(
            [(xx - k[0, 2]) / k[0, 0], (yy - k[1, 2]) / k[1, 1], np.ones_like(xx)], -1
        )
        * depth[..., None]
    )
    points = points @ camera[:3, :3].T + camera[:3, 3]
    zup = np.eye(4)
    zup[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
    observed, masks, meshes, transforms = {}, {}, {}, {}
    for obj in assets:
        uid = obj["uid"]
        path = Path(obj["mesh_path"])
        mesh = _load_mesh(path)
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler(
            "XYZ", obj["init_rot"], degrees=True
        ).as_matrix() * np.array(obj.get("proxy_body_scale", obj["body_scale"]))
        transform[:3, 3] = obj["init_pos"]
        transform = transform @ zup
        mesh.apply_transform(transform)
        meshes[uid], transforms[uid] = mesh, transform
        mask = np.asarray(object_masks[uid], dtype=bool)
        if mask.shape != depth.shape:
            if not np.isclose(
                mask.shape[1] / mask.shape[0],
                depth.shape[1] / depth.shape[0],
                rtol=0.005,
            ):
                raise ValueError(
                    "Mask requires inverse image preprocessing before registration"
                )
            mask = (
                cv2.resize(
                    mask.astype(np.uint8),
                    (depth.shape[1], depth.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                > 0
            )
        valid = (
            (cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
            & (depth > 0.1)
            & (depth < 3)
        )
        masks[uid] = mask
        p = points[valid]
        if len(p) >= 15:
            z = depth[valid]
            lo, hi = np.quantile(z, [0.05, 0.95])
            observed[uid] = p[(z >= lo) & (z <= hi)]
    ids = [x["uid"] for x in assets if x["uid"] in observed and x["uid"] != "table"]
    if not ids:
        raise ValueError(
            "Need an observed foreground object to register the native scene"
        )
    # A clean pick-and-place scene may contain only one target. The calibrated
    # tabletop is also an observed body and can establish scene scale/yaw.
    alignment_ids = ids
    if len(ids) == 1:
        if "table" not in observed:
            raise ValueError("Single-object registration needs observed tabletop depth")
        alignment_ids = ["table", *ids]
    centers = np.array([meshes[x].bounds.mean(0) for x in alignment_ids])
    a = centers[:, :2]
    b = np.array([np.median(observed[x], 0)[:2] for x in alignment_ids])
    ac, bc = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(ac.T @ bc)
    rot2 = vt.T @ np.diag([1, np.linalg.det(vt.T @ u.T)]) @ u.T
    if min(np.sum(ac * ac), np.sum(bc * bc)) < 1e-8:
        # Centered targets cannot determine yaw from center separation. Retain
        # the native yaw and use the measured tabletop footprint as a scale seed.
        if "table" not in observed:
            raise ValueError("Observed centers cannot establish scene scale")
        extent = np.ptp(
            np.quantile(observed["table"][:, :2], [0.05, 0.95], axis=0), axis=0
        )
        scale = float(
            np.linalg.norm(extent) / np.linalg.norm(meshes["table"].extents[:2])
        )
        rot2 = np.eye(2)
    else:
        scale = float(np.sum((ac @ rot2.T) * bc) / np.sum(ac * ac))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid scene registration scale")
    shared = np.eye(4)
    shared[:2, :2] = rot2 * scale
    shared[2, 2] = scale
    shared[:2, 3] = b.mean(0) - shared[:2, :2] @ a.mean(0)
    # A robust table plane in the robot-base frame supplies vertical alignment.
    table_points = observed.get("table", np.empty((0, 3)))
    support_evidence = {"front_points": len(table_points)}
    if len(table_points) < 100:
        # Dark tabletops can have almost no valid frontal depth. Reproject the
        # calibrated wrist depth into the original table mask, without inventing
        # a plane height or changing the segmentation.
        support_views = [table_points]
        for view, extra in auxiliary_depth.items():
            if extra.depth_m is None:
                continue
            wrist_depth = extra.depth_m
            rows, columns = np.indices(wrist_depth.shape)
            intrinsics = extra.intrinsics
            pose = extra.camera_to_base
            rays = np.stack(
                [
                    (columns - intrinsics[0, 2]) / intrinsics[0, 0],
                    (rows - intrinsics[1, 2]) / intrinsics[1, 1],
                    np.ones_like(columns),
                ],
                -1,
            )
            valid = (wrist_depth > 0.1) & (wrist_depth < 3)
            world = (rays * wrist_depth[..., None])[valid] @ pose[:3, :3].T
            world += pose[:3, 3]
            optical = world @ invcam[:3, :3].T + invcam[:3, 3]
            uv = (optical @ k.T)[:, :2] / np.maximum(optical[:, 2:3], 0.03)
            pixel = np.rint(uv).astype(int)
            inside = (
                (optical[:, 2] > 0.03)
                & (pixel[:, 0] >= 0)
                & (pixel[:, 0] < depth.shape[1])
                & (pixel[:, 1] >= 0)
                & (pixel[:, 1] < depth.shape[0])
            )
            pixel, world = pixel[inside], world[inside]
            selected = world[masks["table"][pixel[:, 1], pixel[:, 0]]]
            support_evidence[view + "_points"] = len(selected)
            support_views.append(selected)
        table_points = np.concatenate(support_views)
    if len(table_points) < 100:
        raise ValueError("Insufficient calibrated depth on the observed tabletop")
    rng = np.random.default_rng(0)
    best = np.zeros(len(table_points), bool)
    for _ in range(400):
        chosen = table_points[rng.choice(len(table_points), 3, replace=False)]
        normal = np.cross(chosen[1] - chosen[0], chosen[2] - chosen[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-8 or abs(normal[2]) / norm < 0.8:
            continue
        normal /= norm
        inliers = abs((table_points - chosen[0]) @ normal) < 0.003
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < 100:
        raise ValueError("Cannot establish observed tabletop plane")
    keep = best
    plane = np.linalg.lstsq(
        np.c_[table_points[keep, :2], np.ones(keep.sum())],
        table_points[keep, 2],
        rcond=None,
    )[0]
    support_evidence.update(
        inliers=int(keep.sum()),
        total_points=len(table_points),
        plane_z_from_xy=plane.tolist(),
    )
    # Rotate the whole scene onto the measured support plane. This changes the
    # coordinate frame, never individual object dimensions or relative placement.
    normal = np.r_[-plane[:2], 1.0]
    normal /= np.linalg.norm(normal)
    # The exporter stores support height before the table's pose and body scale.
    # Remove only the GLB Y-up conversion to transform its canonical Z-up plane.
    table_native = transforms["table"] @ zup.T
    support_point = (table_native @ [0.0, 0.0, float(support_surface_z), 1.0])[:3]
    support_normal = np.linalg.solve(table_native[:3, :3].T, [0.0, 0.0, 1.0])
    support_normal /= np.linalg.norm(support_normal)
    rotated_normal = shared[:3, :3] @ support_normal / scale
    tilt = Rotation.align_vectors([normal], [rotated_normal])[0].as_matrix()
    shared[:3, :3] = tilt @ shared[:3, :3]
    shared[:2, 3] = b.mean(0) - (shared[:3, :3] @ centers.mean(0))[:2]
    shared[2, 3] = (
        plane[2]
        + plane[:2] @ shared[:2, 3]
        - normal @ (shared[:3, :3] @ support_point) / normal[2]
    )
    results = []
    vertices = {}
    for obj in assets:
        uid = obj["uid"]
        base = shared @ transforms[uid]
        path = Path(obj["mesh_path"])
        geo = _load_mesh(path)
        geo.apply_transform(base)
        vertices[uid] = np.asarray(geo.vertices).copy()
        results.append(
            dict(
                id=uid,
                name=obj["name"],
                category=obj["category"],
                path=str(path),
                transform=base.tolist(),
                bounds=geo.bounds.tolist(),
                registration=dict(method="shared scene similarity"),
                articulated=obj.get("is_articulated", False),
            )
        )
    return RegisteredScene(
        dict(
            objects=results,
            source_table_plane=plane.tolist(),
            table_plane_evidence=support_evidence,
            table_appearance="Original generated GLB materials and textures",
            global_similarity=shared.tolist(),
            geometry_policy="One shared uniform similarity; native shapes and relative transforms preserved",
            scope="observation registration for trajectory adaptation, not physical grasp validation",
        ),
        vertices,
    )
