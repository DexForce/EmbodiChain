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
import time
import numpy as np
import open3d as o3d
import torch
import trimesh
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from ._kinematics import Robot
from ._hand_geometry import (
    hand_collision_surfaces,
    closed_hand_collision_parts,
    reverse_hand_penetration,
)
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


def _obstacle_distances(
    body: _Body, points: np.ndarray, clearance_m: float = 0.001
) -> np.ndarray:
    """Return exact sampled distances near a body, a safe lower bound far away."""
    bounds = np.stack([points.min(0), points.max(0)]).astype(np.float64)
    gap = float(
        aabb_separation(torch.from_numpy(bounds), torch.from_numpy(body.bounds[0]))
    )
    if gap > clearance_m:
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
    clearance_m: float = 0.001,
    seed_qpos: tuple[np.ndarray, ...] = (),
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
            the complete hand throughout the jaw sweep at the fitted arm pose.
        seed_qpos: Optional same-model optimizer starts selected by the caller.
            When supplied, they replace the default geometric starts. Source
            joint and TCP priors remain the recorded source.
        clearance_m: Caller physical envelope for unintended hand contacts.
            Excess distance violations and pad/target penetration beyond 1 mm
            contribute soft penalties; they do not prevent trajectory return.

    Returns:
        Fitted joint state, paired jaw half width and geometric diagnostics.
        This does not establish force closure or replace full-path screening.
    """
    if not np.isfinite(clearance_m) or not 0 <= clearance_m < 0.15:
        raise ValueError(
            "Hand clearance must be finite and between zero and 0.15 metres"
        )
    hand_surfaces = hand_collision_surfaces(robot, side)
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
    # CAD/GLB normal seams disconnect otherwise closed surfaces. Orient the
    # welded numerical mesh before using its normals to select opposing sides.
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
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
    reverse_parts = (
        closed_hand_collision_parts(
            robot, [link for link in hand_surfaces if link not in links]
        )
        if not object_body.closed
        else {}
    )
    tcp_link = robot.config.get("tcp_links", [p + "_link6" for p in robot.prefixes])[
        side
    ]
    initial_tcp = robot.tcp(source[None], side)[0]
    initial = np.r_[source[columns], float(robot.half_open(source, side))]
    minimum, limit = robot.opening_limits[side]
    minimum = max(minimum, 0.001)
    initial[-1] = np.clip(initial[-1], minimum, limit)
    recorded_closing = closing_half_open_m
    if closing_half_open_m is not None:
        closing_half_open_m = float(np.clip(closing_half_open_m, minimum, limit))
    closure_samples = 9 if closing_half_open_m is not None else 1
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
        hand_clearance = {}
        hand_violation = 0.0
        reverse_penetration = {}
        # Intended target contact is restricted to calibrated pad links. Palm
        # and articulated links must remain clear throughout the same closure.
        # All collision links, including pads, clear table and other objects.
        for link, local_points in hand_surfaces.items():
            transforms = fk[link]
            if np.allclose(transforms, transforms[:1], atol=1e-12, rtol=0):
                transforms = transforms[:1]
            world = _transform(local_points, transforms).reshape(-1, 3)
            bodies = [support, *obstacles, object_body]
            distances = {}
            for body in bodies:
                if body is support and not body.closed:
                    origin_z = float(body.mesh.bounds[1, 2] + 0.1)
                    rays = np.zeros((len(world), 6), dtype=np.float32)
                    rays[:, :2] = world[:, :2]
                    rays[:, 2], rays[:, 5] = origin_z, -1
                    hits = body.bvh.cast_rays(o3d.core.Tensor(rays), nthreads=2)[
                        "t_hit"
                    ].numpy()
                    distance = world[:, 2] - (origin_z - hits)
                    distance[~np.isfinite(hits)] = 1.0
                else:
                    distance = _obstacle_distances(body, world, clearance_m)
                minimum_gap = float(distance.min())
                if body is object_body and link in reverse_parts:
                    depth = reverse_hand_penetration(
                        reverse_parts[link], surface, transforms
                    )
                    reverse_penetration[link] = depth
                    if depth > 0:
                        minimum_gap = min(minimum_gap, -depth)
                required = (
                    -0.001 if body is object_body and link in links else clearance_m
                )
                violation = max(0.0, required - minimum_gap)
                hand_violation = max(hand_violation, violation)
                terms.append(np.array([violation / 0.0001]))
                distances[body.name] = minimum_gap
            hand_clearance[link] = distances
        value = np.concatenate(terms), dict(
            hand_clearance_m=hand_clearance,
            hand_clearance_violation_m=hand_violation,
            open_target_reverse_penetration_m=reverse_penetration,
            fingers=records,
            tcp_shift_m=(tcp[:3, 3] - initial_tcp[:3, 3]).tolist(),
        )
        cache[key] = value
        if len(cache) > 16:
            cache.popitem(last=False)
        return value

    before = evaluate(initial)[1]

    def jacobian(x: np.ndarray) -> np.ndarray:
        steps = np.r_[np.full(d, 1e-4), 1e-5]
        return np.column_stack(
            [
                (evaluate(x + step)[0] - evaluate(x - step)[0]) / (2 * size)
                for step, size in zip(np.diag(steps), steps)
            ]
        )

    arm_lower, arm_upper = robot.arm_limits[side]
    lower = np.r_[arm_lower, minimum]
    upper = np.r_[arm_upper, limit]
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

    if not seed_qpos:
        seed[:d] = robot.ik(source, side, target)[0][columns]
    wide = seed.copy()
    wide[-1] = limit * 0.9
    starts = (
        []
        if seed_qpos
        else [("pad_geometry", seed), ("source", initial), ("wide", wide)]
    )
    for index, q_seed in enumerate(seed_qpos):
        q_seed = np.asarray(q_seed, dtype=float)
        if q_seed.shape != source.shape or not np.isfinite(q_seed).all():
            raise ValueError("Grasp seed must be a finite full joint state")
        starts.insert(
            index,
            (
                f"supplied_{index}",
                np.r_[q_seed[columns], robot.half_open(q_seed, side)],
            ),
        )
    # Derive an axial retreat from the actual non-pad collision surfaces. Keep
    # the demonstrated orientation and physical joint limits; this only seeds the
    # existing source-prior solve, never translates an object or its support.
    seed_q = source.copy()
    seed_q[columns] = seed[:d]
    seed_q = robot.set_half_open(seed_q, side, np.clip(seed[-1], minimum, limit))
    seed_fk = robot.links(seed_q[None])
    seed_tcp = robot.tcp(seed_q[None], side)[0]
    retreat_axis = -seed_tcp[:3, 2]
    nonpad_world = [
        _transform(points, seed_fk[link])[0]
        for link, points in hand_surfaces.items()
        if link not in links
    ]
    axial_retreat_m = None
    if nonpad_world and not seed_qpos:
        cloud = np.concatenate(nonpad_world)
        for distance in np.linspace(0.0, 0.08, 17):
            shifted = cloud + distance * retreat_axis
            if all(
                float(_obstacle_distances(body, shifted, clearance_m).min())
                >= clearance_m
                for body in [object_body, support, *obstacles]
            ):
                if distance > 0:
                    target = seed_tcp.copy()
                    target[:3, 3] += distance * retreat_axis
                    retreat = seed.copy()
                    retreat[:d] = robot.ik(seed_q, side, target)[0][columns]
                    starts.append(("hand_axial_retreat", retreat))
                    axial_retreat_m = float(distance)
                break
    candidates = []
    stopped_after_supplied_seed = False
    started = time.monotonic()
    for name, start in starts:
        print(f"Contact fit {scene_object['id']}: {name}", flush=True)
        candidate = least_squares(
            lambda x: evaluate(x)[0],
            np.clip(start, lower + 1e-6, upper - 1e-6),
            bounds=(lower, upper),
            jac=jacobian,
            max_nfev=80,
        )
        if not np.isfinite(candidate.x).all() or not np.isfinite(candidate.cost):
            raise ValueError(
                "Contact optimizer returned non-finite joints or objective"
            )
        candidates.append(candidate)
        print(
            f"Contact fit {scene_object['id']}: {name} cost={candidate.cost:.4f}, "
            f"elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )
        contact = evaluate(candidate.x)[1]
        if name.startswith("supplied_") and (
            contact["hand_clearance_violation_m"] <= 0.0002
            and all(
                finger["opposing_pad_gap_m"] <= 0.003
                and finger["sampled_object_penetration_m"] <= 0.0012
                for finger in contact["fingers"]
            )
        ):
            stopped_after_supplied_seed = True
            break
    result = min(candidates, key=lambda item: item.cost)
    trial_report = {
        "method": "source-near mesh contact fit with pad-geometry initialization",
        "arm_search_bounds": "URDF physical limits; source retained as soft prior",
        "arm_joint_lower_limits_rad": [
            float(v) if np.isfinite(v) else None for v in arm_lower
        ],
        "arm_joint_upper_limits_rad": [
            float(v) if np.isfinite(v) else None for v in arm_upper
        ],
        "selection": "lowest soft objective among evaluated candidates; geometry diagnostics do not establish task completion",
        "candidate_source": "caller" if seed_qpos else "default_geometry",
        "stopped_after_supplied_seed": stopped_after_supplied_seed,
        "neighbor_clearance_checked": bool(obstacles),
        "neighbor_clearance_kind": "AABB lower bound when separated, otherwise sampled mesh distance",
        "closure_samples": closure_samples,
        "axial_retreat_seed_m": axial_retreat_m,
        "hand_collision_links": list(hand_surfaces),
        "open_target_reverse_sdf_components": {
            link: len(parts) for link, parts in reverse_parts.items()
        },
        "clearance_objective_m": clearance_m,
        "hand_geometry": "authored URDF collision surfaces; sampled closure, not continuous collision certification",
        "recorded_closing_half_open_m": recorded_closing,
        "joint_limited_closing_half_open_m": closing_half_open_m,
        "joint_half_open_limits_m": [minimum, limit],
        "candidates": [
            dict(
                seed=starts[index][0],
                cost=float(item.cost),
                optimizer_converged=bool(item.success),
                arm_and_half_open=item.x.tolist(),
                **evaluate(item.x)[1],
            )
            for index, item in enumerate(candidates)
        ],
    }
    diagnostics = evaluate(result.x)[1]
    diagnostics["search"] = trial_report
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
