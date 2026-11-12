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
from PIL import Image
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
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
    parents: dict[str, str | None],
    support_surface_z: float,
    *,
    auxiliary_depth: dict[str, CameraObservation] | None = None,
) -> RegisteredScene:
    """Fit a metric tabletop footprint and per-object RGB-D similarities.

    Args:
        observation: Original pixel-aligned RGB-D and optical-to-base calibration.
        assets: Generated mesh paths and declared Y-up poses/scales/categories.
        object_masks: Masks in original image coordinates (or equal-aspect resize).
        parents: Object support relationships from the native scene graph.
        support_surface_z: Exported tabletop height before registration.
        auxiliary_depth: Optional calibrated depth views of the same initial scene.

    Returns:
        Registered mesh descriptors and world-frame vertex samples. Geometry files
        and original calibration remain unchanged. No physical success is inferred.
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
            "xyz", obj["init_rot"], degrees=True
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
    # tabletop is also an observed body and can initialize scene scale/yaw.
    # Foreground RGB-D/silhouette fitting below still determines the target pose.
    alignment_ids = ids
    if len(ids) == 1:
        if "table" not in observed:
            raise ValueError("Single-object registration needs observed tabletop depth")
        alignment_ids = ["table", *ids]
    a = np.array([meshes[x].bounds.mean(0)[:2] for x in alignment_ids])
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
    table_z = float(support_surface_z)
    shared[2, 3] = plane[2] - table_z * scale
    # Include calibrated auxiliary-view support evidence in table fitting too.
    observed["table"] = table_points[keep]
    results = []
    vertices = {}
    for obj in assets:
        uid = obj["uid"]
        native_mesh = meshes[uid]
        base = shared @ transforms[uid]
        report = dict(method="shared scene similarity")
        if uid in observed:
            initialmesh = native_mesh.copy()
            initialmesh.apply_transform(shared)
            center = initialmesh.bounds.mean(0)
            local = np.asarray(initialmesh.vertices) - center
            # Surface sampling avoids attracting observed face interiors to the
            # few corner vertices of low-poly assets. Keep vertices for outlines.
            sample = (
                trimesh.sample.sample_surface(initialmesh, 6500, seed=0)[0] - center
            )
            contour = local[
                np.linspace(0, len(local) - 1, min(900, len(local))).astype(int)
            ]
            tree = cKDTree(sample)
            p = observed[uid][
                np.linspace(
                    0, len(observed[uid]) - 1, min(350, len(observed[uid]))
                ).astype(int)
            ]
            mask = masks[uid]
            ys, xs = np.nonzero(mask)
            box = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
            outside = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
            tilted_support = obj["category"] == "plate" and parents.get(uid) != "table"
            scale_dimensions = 2 if uid == "table" else 1

            def metric_scale(v: np.ndarray) -> np.ndarray:
                if uid == "table":
                    # Calibrate the generated tabletop footprint independently
                    # along both axes: layout-adjusted object center distances
                    # cannot determine its dimensions. Thickness is unobserved.
                    return np.r_[np.exp(v[6:8]), 1.0]
                return np.full(3, np.exp(v[6]))

            def residual(v: np.ndarray) -> np.ndarray:
                r = Rotation.from_rotvec(v[:3]).as_matrix()
                s = metric_scale(v)
                nearest = tree.query((p - v[3:6]) @ r / s)[1]
                dist = np.linalg.norm(sample[nearest] * s @ r.T + v[3:6] - p, axis=1)
                world = contour * s @ r.T + v[3:6]
                optical = world @ invcam[:3, :3].T + invcam[:3, 3]
                uv = (optical @ k.T)[:, :2] / np.maximum(optical[:, 2:3], 0.03)
                silhouette = map_coordinates(
                    outside, [uv[:, 1], uv[:, 0]], order=1, mode="constant", cval=320
                )
                bbox = np.r_[uv.min(0), uv.max(0)]
                height = world[:, 2] - world[:, :2] @ plane[:2] - plane[2]
                support = (
                    np.quantile(height, 0.998) / 0.004
                    if uid == "table"
                    else (
                        0
                        if obj["category"] == "banana" or parents.get(uid) != "table"
                        else np.quantile(height, 0.002) / 0.004
                    )
                )
                normal = np.array([-plane[0], -plane[1], 1.0])
                normal /= np.linalg.norm(normal)
                upright = (
                    np.zeros(3)
                    if obj["category"] == "banana" or tilted_support
                    else np.cross(r[:, 2], normal) / 0.04
                )
                return np.r_[
                    dist / (0.006 * np.sqrt(len(p))),
                    (bbox - box) / 5,
                    silhouette / (5 * np.sqrt(len(contour))),
                    support,
                    upright,
                    v[:3] * 0.15,
                    v[6:] * 0.4,
                ]

            candidates = []
            seeds = [
                np.array([0.0, 0.0, angle])
                for angle in [0, np.pi / 2, -np.pi / 2, np.pi]
            ]
            if tilted_support:
                seeds.extend(
                    Rotation.from_euler("xyz", [tilt, 0, yaw]).as_rotvec()
                    for tilt in [np.pi / 2, -np.pi / 2]
                    for yaw in [0, np.pi / 2, np.pi, -np.pi / 2]
                )
            for seed in seeds:
                # Native layout correction can change inter-object distances.
                # Seed each object's metric scale from its observed silhouette
                # instead of constraining it to the global center-distance fit.
                rotation = Rotation.from_rotvec(seed).as_matrix()
                predicted = contour @ rotation.T + np.median(p, 0)
                optical = predicted @ invcam[:3, :3].T + invcam[:3, 3]
                uv = (optical @ k.T)[:, :2] / np.maximum(optical[:, 2:3], 0.03)
                ratio = (box[2:] - box[:2]) / np.maximum(np.ptp(uv, axis=0), 1)
                log_scale = np.clip(np.log(np.median(ratio)), -2.9, 2.9)
                x = np.r_[seed, np.median(p, 0), np.full(scale_dimensions, log_scale)]
                rotation_bound = 4.0 if tilted_support else 0.6
                fit = least_squares(
                    residual,
                    x,
                    bounds=(
                        np.r_[
                            -rotation_bound,
                            -rotation_bound,
                            -4,
                            np.median(p, 0) - 0.35,
                            np.full(scale_dimensions, -3.0),
                        ],
                        np.r_[
                            rotation_bound,
                            rotation_bound,
                            4,
                            np.median(p, 0) + 0.35,
                            np.full(scale_dimensions, 3.0),
                        ],
                    ),
                    loss="soft_l1",
                    max_nfev=65,
                )
                candidates.append(fit)
            fit = min(candidates, key=lambda f: f.cost)
            delta = np.eye(4)
            fitted_rotation = Rotation.from_rotvec(fit.x[:3]).as_matrix()
            fitted_scale = metric_scale(fit.x)
            delta[:3, :3] = fitted_rotation * fitted_scale
            delta[:3, 3] = fit.x[3:6] - delta[:3, :3] @ center
            base = delta @ base
            nearest = tree.query((p - fit.x[3:6]) @ fitted_rotation / fitted_scale)[1]
            error = np.linalg.norm(
                sample[nearest] * fitted_scale @ fitted_rotation.T + fit.x[3:6] - p,
                axis=1,
            )
            report = dict(
                method=(
                    "automatic RGB-D tabletop footprint fit, unchanged generated mesh"
                    if uid == "table"
                    else "automatic RGB-D similarity fit, unchanged generated geometry"
                ),
                cost=float(fit.cost),
                depth_error_p50_p95_m=np.quantile(error, [0.5, 0.95]).tolist(),
                relative_scale=fitted_scale.tolist(),
            )
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
                registration=report,
                articulated=obj.get("is_articulated", False),
            )
        )
    clean = masks["table"].copy()
    for uid, mask in masks.items():
        if uid != "table":
            clean &= ~cv2.dilate(
                mask.astype(np.uint8), np.ones((7, 7), np.uint8)
            ).astype(bool)
    rgb = np.asarray(Image.open(observation.first_image).convert("RGB")) / 255.0
    if rgb.shape[:2] != depth.shape:
        raise ValueError("Original RGB and depth must be pixel-aligned")
    if not clean.any():
        raise ValueError("No visible table pixels for appearance registration")
    table_rgb = np.median(rgb[clean], axis=0)
    return RegisteredScene(
        dict(
            objects=results,
            source_table_plane=plane.tolist(),
            table_plane_evidence=support_evidence,
            table_color_srgb=table_rgb.tolist(),
            table_appearance="Median color of observed table mask excluding all foreground masks, replaces ghost-containing generated table texture",
            global_similarity=shared.tolist(),
            geometry_policy="All native exported objects retained, no replacement geometry or hand-authored poses",
            scope="observation registration for trajectory adaptation, not physical grasp validation",
        ),
        vertices,
    )
