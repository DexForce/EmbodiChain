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
"""Mesh contact fitting used by the source-near trajectory method.

Migrated numerical objective and tolerances; geometric proximity is not force
closure. Open3D work is CPU-bound with bounded samples and two native threads.
"""

from __future__ import annotations
from dataclasses import dataclass
from collections import OrderedDict
from itertools import product
from pathlib import Path
import xml.etree.ElementTree as ET
import time
import numpy as np
import open3d as o3d
import torch
import trimesh
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from ._kinematics import Robot
from embodichain.compute.geometry.aabb import aabb_separation

__all__: list[str] = []


def _mesh(path: Path) -> trimesh.Trimesh:
    scene = trimesh.load(path, force="scene", process=False)
    parts = []
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        part = scene.geometry[name].copy()
        part.apply_transform(transform)
        parts.append(part)
    return trimesh.util.concatenate(parts)


def _transform(points: np.ndarray, matrices: np.ndarray) -> np.ndarray:
    return (
        np.einsum("fij,pj->fpi", matrices[:, :3, :3], points) + matrices[:, None, :3, 3]
    )


@dataclass
class _Body:
    name: str
    mesh: trimesh.Trimesh
    transforms: np.ndarray
    samples: np.ndarray
    bvh: o3d.t.geometry.RaycastingScene
    bounds: np.ndarray
    closed: bool


def _obstacle_distances(body: _Body, points: np.ndarray) -> np.ndarray:
    """Return exact sampled distances near a body, a safe lower bound far away."""
    bounds = np.stack([points.min(0), points.max(0)]).astype(np.float64)
    gap = float(
        aabb_separation(torch.from_numpy(bounds), torch.from_numpy(body.bounds[0]))
    )
    if gap > 0.001:
        # The whole swept sample set is separated. Both dense and worst-case
        # clearance penalties are exactly zero; preserve their fixed dimensions.
        return np.full(len(points), gap, dtype=np.float32)
    query = (
        body.bvh.compute_signed_distance if body.closed else body.bvh.compute_distance
    )
    return query(o3d.core.Tensor(points.astype(np.float32)), nthreads=2).numpy()


def _body(
    name: str, mesh: trimesh.Trimesh, transforms: np.ndarray, samples: int
) -> _Body:
    # GLB duplicates vertices at UV and normal seams. Merge on a numerical copy;
    # never fill holes, convexify a container, or alter the rendered asset.
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    surface, _ = trimesh.sample.sample_surface(mesh, samples, seed=42)
    vertices = np.asarray(mesh.vertices)
    if len(vertices) > samples:
        vertices = vertices[np.linspace(0, len(vertices) - 1, samples).astype(int)]
    points = np.concatenate([vertices, surface]).astype(np.float32)
    geometry = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.int64)),
    )
    bvh = o3d.t.geometry.RaycastingScene(nthreads=2)
    bvh.add_triangles(geometry)
    corners = np.array(list(product(*zip(*mesh.bounds))), dtype=float)
    world_corners = _transform(corners, transforms)
    bounds = np.stack([world_corners.min(1), world_corners.max(1)], axis=1)
    return _Body(name, mesh, transforms, points, bvh, bounds, bool(mesh.is_watertight))


def _support_height(
    objects: dict[str, dict],
    deltas: dict[str, np.ndarray],
    carried_id: str,
    footprint: np.ndarray,
    table_height: float,
) -> tuple[float, str | None]:
    """Find the actual upward surface below a placement, independent of category."""
    height, support_id = table_height, None
    for uid, obj in objects.items():
        if uid == carried_id:
            continue
        mesh = _mesh(Path(obj["path"]))
        mesh.apply_transform(deltas[uid] @ np.asarray(obj["transform"]))
        bounds = mesh.bounds
        if not np.any(
            np.all((footprint >= bounds[0, :2]) & (footprint <= bounds[1, :2]), axis=1)
        ):
            continue
        bvh = o3d.t.geometry.RaycastingScene(nthreads=2)
        bvh.add_triangles(
            o3d.t.geometry.TriangleMesh(
                o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
                o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.int64)),
            )
        )
        rays = np.zeros((len(footprint), 6), dtype=np.float32)
        rays[:, :2] = footprint
        rays[:, 2], rays[:, 5] = float(bounds[1, 2] + 0.1), -1
        hits = bvh.cast_rays(o3d.core.Tensor(rays), nthreads=2)
        distance = hits["t_hit"].numpy()
        upward = hits["primitive_normals"].numpy()[:, 2] > 0.5
        valid = np.isfinite(distance) & upward
        if valid.any():
            top = float(np.max(rays[valid, 2] - distance[valid]))
            if top > height:
                height, support_id = top, uid
    return height, support_id


def fit_grasp(
    scene: dict,
    visuals: tuple[dict, ...],
    robot: Robot,
    source: np.ndarray,
    side: int,
    scene_object: dict,
    object_delta: np.ndarray,
    object_deltas: dict[str, np.ndarray] | None = None,
    closing_half_open_m: float | None = None,
) -> tuple[np.ndarray, float, dict]:
    """Fit bilateral finger contact with table and neighboring-object clearance.

    Args:
        scene: Registered objects with absolute geometry paths.
        visuals: Robot link meshes with absolute paths and local origins.
        robot: Matching recorded robot model.
        source: Original measured joint state at the grasp anchor.
        side: Arm index in the example's eight-joint arm/jaw groups.
        scene_object: Reconstructed object and initial metric transform.
        object_delta: Current rigid object motion relative to its initial pose.
        object_deltas: Other object pose estimates at this anchor, for obstacle clearance.
        closing_half_open_m: Recorded opening before closure. When provided, check
            neighboring objects throughout the jaw sweep at the fitted arm pose.

    Returns:
        Fitted joint state, paired jaw half width and geometric diagnostics.
        This does not establish force closure or replace full-path screening.
    """
    columns = robot.arm_columns[side]
    d = len(columns)
    links, axes = robot.finger_surfaces(side)
    fingers = []
    pads = []
    for link, axis in zip(links, axes):
        meshes = []
        for visual in visuals:
            if visual["link"] != link:
                continue
            path = Path(visual["path"])
            mesh = _mesh(path)
            mesh.apply_transform(np.asarray(visual["origin"]))
            meshes.append(mesh)
        if not meshes:
            raise ValueError("Missing actual finger visual mesh: " + link)
        finger = _body(link, trimesh.util.concatenate(meshes), np.eye(4)[None], 2048)
        if not finger.closed:
            raise ValueError(
                "Finger mesh is not closed; signed contact fitting unavailable: " + link
            )
        fingers.append(finger)
        finger.mesh.fix_normals(multibody=True)
        points, faces = trimesh.sample.sample_surface(finger.mesh, 4096, seed=42)
        inner = finger.mesh.face_normals[faces] @ axis < -0.8
        if inner.sum() < 20:
            raise ValueError("Cannot identify inward finger surfaces: " + link)
        pads.append(points[inner])
    table = next(o for o in scene["objects"] if o["id"] == "table")
    path = Path(table["path"])
    mesh = _mesh(path)
    mesh.apply_transform(np.asarray(table["transform"]))
    support = _body("table", mesh, np.eye(4)[None], 16)
    path = Path(scene_object["path"])
    mesh = _mesh(path)
    mesh.apply_transform(object_delta @ np.asarray(scene_object["transform"]))
    mesh.fix_normals(multibody=True)
    object_body = _body(scene_object["id"], mesh.copy(), np.eye(4)[None], 16)
    obstacles = []
    if object_deltas is not None:
        for other in scene["objects"]:
            if other["id"] in {"table", scene_object["id"]}:
                continue
            other_mesh = _mesh(Path(other["path"]))
            other_mesh.apply_transform(
                object_deltas.get(other["id"], np.eye(4))
                @ np.asarray(other["transform"])
            )
            obstacles.append(_body(other["id"], other_mesh, np.eye(4)[None], 16))
    surface, surface_faces = trimesh.sample.sample_surface(mesh, 4096, seed=42)
    surface_normals = mesh.face_normals[surface_faces]
    tcp_link = robot.config.get("tcp_links", [p + "_link6" for p in robot.prefixes])[
        side
    ]
    initial_tcp = robot.tcp(source[None], side)[0]
    initial = np.r_[source[columns], float(robot.half_open(source, side))]
    closure_samples = 9 if closing_half_open_m is not None and obstacles else 1
    cache: OrderedDict[bytes, tuple[np.ndarray, dict]] = OrderedDict()

    def evaluate(x: np.ndarray) -> tuple[np.ndarray, dict]:
        key = x.tobytes()
        if key in cache:
            return cache[key]
        q = source.copy()
        q[columns] = x[:d]
        openings = np.linspace(
            x[d], max(x[d], closing_half_open_m or x[d]), closure_samples
        )
        q = robot.set_half_open(
            np.repeat(q[None], closure_samples, axis=0), side, openings
        )
        fk = robot.links(q)
        tcp = fk[tcp_link][0] @ robot.offset
        terms = [
            (x[:d] - initial[:d]) / 0.08,
            np.array([(x[d] - initial[d]) / 0.01]),
            (tcp[:3, 3] - initial_tcp[:3, 3]) / 0.015,
            Rotation.from_matrix(tcp[:3, :3] @ initial_tcp[:3, :3].T).as_rotvec()
            / 0.06,
        ]
        records = []
        for finger, link, pad, axis in zip(fingers, links, pads, axes):
            transform = fk[link][0]
            local = _transform(surface, np.linalg.inv(transform)[None])[0].astype(
                np.float32
            )
            distance = finger.bvh.compute_signed_distance(
                o3d.core.Tensor(local), nthreads=2, nsamples=3
            ).numpy()
            finger_world = _transform(finger.samples, transform[None])[0].astype(
                np.float32
            )
            inverse_distance = (
                object_body.bvh.compute_signed_distance(
                    o3d.core.Tensor(finger_world), nthreads=2, nsamples=3
                ).numpy()
                if object_body.closed
                else np.zeros(len(finger_world))
            )
            if support.closed:
                table_distance = support.bvh.compute_signed_distance(
                    o3d.core.Tensor(finger_world), nthreads=2, nsamples=3
                ).numpy()
            else:
                # A generated tabletop may be an open sheet. Use its actual
                # upward-facing support height, not an invalid inside query.
                origin_z = float(support.mesh.bounds[1, 2] + 0.1)
                rays = np.zeros((len(finger_world), 6), dtype=np.float32)
                rays[:, :2] = finger_world[:, :2]
                rays[:, 2], rays[:, 5] = origin_z, -1
                hit = support.bvh.cast_rays(o3d.core.Tensor(rays), nthreads=2)[
                    "t_hit"
                ].numpy()
                table_distance = finger_world[:, 2] - (origin_z - hit)
                table_distance[~np.isfinite(hit)] = 1.0
            gap = float(np.min(np.abs(distance)))
            outward = transform[:3, :3] @ axis
            facing = surface_normals @ outward > 0.5
            world_pad = _transform(pad, transform[None])[0]
            pad_gap = (
                float(np.quantile(cKDTree(world_pad).query(surface[facing])[0], 0.05))
                if facing.any()
                else 0.1
            )
            terms.append(np.array([gap / 0.001]))
            terms.append(np.array([pad_gap / 0.002]))
            # Dense surface averages can hide a few deeply penetrating points.
            # Penalize the same worst-case metrics used by acceptance, so a
            # low least-squares cost cannot silently miss those constraints.
            terms.append(
                np.array(
                    [
                        max(0.0, pad_gap - 0.003) / 0.0001,
                        max(0.0, -float(distance.min()) - 0.001) / 0.0001,
                        max(0.0, -float(inverse_distance.min()) - 0.001) / 0.0001,
                        max(0.0, -float(table_distance.min()) - 0.001) / 0.0001,
                    ]
                )
            )
            terms.append(np.minimum(distance + 0.0005, 0) / (0.0002 * np.sqrt(32)))
            terms.append(
                np.minimum(inverse_distance + 0.0005, 0) / (0.0002 * np.sqrt(32))
            )
            terms.append(
                np.minimum(table_distance + 0.0005, 0) / (0.0002 * np.sqrt(32))
            )
            clearance = {}
            # Clear endpoints do not imply a clear closure: a finger can cross
            # a plate rim between its open and closed configurations. Keep the
            # target-contact objective at the closed pose, but screen neighbors
            # across the same jaw motion using one batched FK evaluation.
            swept_finger_world = _transform(finger.samples, fk[link]).reshape(-1, 3)
            for obstacle in obstacles:
                distance_to_other = _obstacle_distances(obstacle, swept_finger_world)
                terms.append(
                    np.minimum(distance_to_other - 0.001, 0) / (0.0002 * np.sqrt(32))
                )
                terms.append(
                    np.array(
                        [max(0.0, 0.001 - float(distance_to_other.min())) / 0.0001]
                    )
                )
                clearance[obstacle.name] = float(distance_to_other.min())
            records.append(
                dict(
                    link=link,
                    sampled_object_gap_m=gap,
                    opposing_pad_gap_m=pad_gap,
                    sampled_object_penetration_m=float(
                        -min(distance.min(), inverse_distance.min(), 0)
                    ),
                    sampled_table_penetration_m=float(-min(table_distance.min(), 0)),
                    obstacle_clearance_m=clearance,
                )
            )
        value = np.concatenate(terms), dict(
            fingers=records, tcp_shift_m=(tcp[:3, 3] - initial_tcp[:3, 3]).tolist()
        )
        cache[key] = value
        if len(cache) > 16:
            cache.popitem(last=False)
        return value

    before = evaluate(initial)[1]
    limit = robot.config.get("max_half_open_m", 0.065)
    initial[-1] = np.clip(initial[-1], 0.001, limit)

    def jacobian(x: np.ndarray) -> np.ndarray:
        steps = np.r_[np.full(d, 1e-4), 1e-5]
        return np.column_stack(
            [
                (evaluate(x + step)[0] - evaluate(x - step)[0]) / (2 * size)
                for step, size in zip(np.diag(steps), steps)
            ]
        )

    def accepted(candidate) -> bool:
        return all(
            f["opposing_pad_gap_m"] <= 0.004
            and f["sampled_object_penetration_m"] <= 0.002
            and f["sampled_table_penetration_m"] <= 0.002
            and all(
                value >= (0.0005 if closure_samples > 1 else -0.0005)
                for value in f["obstacle_clearance_m"].values()
            )
            for f in evaluate(candidate.x)[1]["fingers"]
        )

    lower = np.r_[initial[:d] - 0.15, 0.001]
    upper = np.r_[initial[:d] + 0.15, limit]
    # Initialize the same local search from opposing pad geometry. Preserve the
    # demonstrated orientation; this is a seed, not an alternative grasp planner.
    fk = robot.links(source[None])
    centers = np.array(
        [_transform(pad, fk[link])[0].mean(0) for pad, link in zip(pads, links)]
    )
    axis = fk[links[0]][0, :3, :3] @ axes[0]
    axis /= np.linalg.norm(axis)
    span = np.quantile(surface @ axis, [0.01, 0.99])
    seed = initial.copy()
    seed[-1] += (span[1] - span[0] - abs((centers[0] - centers[1]) @ axis)) / 2
    target = initial_tcp.copy()
    delta = np.clip(mesh.bounds.mean(0) - centers.mean(0), -0.08, 0.08)
    lowest_finger = min(
        _transform(finger.samples, fk[link])[0, :, 2].min()
        for finger, link in zip(fingers, links)
    )
    delta[2] = max(delta[2], mesh.bounds[0, 2] + 0.003 - lowest_finger)
    target[:3, 3] += np.clip(delta, -0.08, 0.08)

    def seed_residual(joints: np.ndarray) -> np.ndarray:
        q = source.copy()
        q[columns] = joints
        tcp = robot.tcp(q[None], side)[0]
        return np.r_[
            (tcp[:3, 3] - target[:3, 3]) / 0.003,
            Rotation.from_matrix(tcp[:3, :3] @ target[:3, :3].T).as_rotvec() / 0.03,
            (joints - initial[:d]) / 0.3,
        ]

    seed[:d] = least_squares(
        seed_residual, initial[:d], bounds=(lower[:d], upper[:d]), max_nfev=60
    ).x
    wide = seed.copy()
    wide[-1] = limit * 0.9
    candidates = []
    started = time.monotonic()
    for name, start in [("pad_geometry", seed), ("source", initial), ("wide", wide)]:
        print(f"Contact fit {scene_object['id']}: {name}", flush=True)
        candidate = least_squares(
            lambda x: evaluate(x)[0],
            np.clip(start, lower + 1e-6, upper - 1e-6),
            bounds=(lower, upper),
            jac=jacobian,
            max_nfev=80,
        )
        candidates.append(candidate)
        result = min(candidates, key=lambda item: (not accepted(item), item.cost))
        print(
            f"Contact fit {scene_object['id']}: {name} accepted={accepted(candidate)}, "
            f"elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )
        if accepted(result):
            break
    trial_report = {
        "method": "source-near mesh contact fit with pad-geometry initialization",
        "arm_search_bound_rad": 0.15,
        "target_contact_acceptance_unchanged": True,
        "neighbor_clearance_checked": bool(obstacles),
        "neighbor_clearance_kind": "AABB lower bound when separated, otherwise sampled mesh distance",
        "closure_samples": closure_samples,
        "recorded_closing_half_open_m": closing_half_open_m,
        "minimum_neighbor_clearance_m": 0.0005 if closure_samples > 1 else -0.0005,
        "candidates": [
            dict(
                cost=float(item.cost),
                accepted=accepted(item),
                arm_and_half_open=item.x.tolist(),
                **evaluate(item.x)[1],
            )
            for item in candidates
        ],
    }
    diagnostics = evaluate(result.x)[1]
    diagnostics["search"] = trial_report
    if not accepted(result):
        raise ValueError(
            "No source-near bilateral contact candidate: " + str(diagnostics)
        )
    fitted = source.copy()
    fitted[columns] = result.x[:d]
    fitted = robot.set_half_open(fitted, side, result.x[d])
    diagnostics.update(
        method="Bilateral finger surface fit with object/table and neighboring-object barriers, retaining source joint/orientation priors",
        before=before,
        max_joint_residual_rad=float(np.max(abs(result.x[:d] - source[columns]))),
        evaluations=result.nfev,
        table_barrier=(
            "closed-mesh signed distance"
            if support.closed
            else "open-mesh vertical support rays; no interior classification"
        ),
    )
    return fitted, float(result.x[d]), diagnostics
