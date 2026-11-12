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
"""Apply observed initial resting poses before fitting a real trajectory."""

from __future__ import annotations
from copy import deepcopy
import numpy as np
from .inputs import RegisteredScene

__all__ = ["prepare_settling_scene", "apply_settled_poses"]


def prepare_settling_scene(
    scene: RegisteredScene,
    support_nodes: list[dict],
    *,
    clearance_m: float = 0.02,
) -> tuple[RegisteredScene, dict]:
    """Separate registered on-support groups before native gravity settling.

    Reuse EmbodiChain's group-height proposal with the native scene graph.
    Preserve XY, orientation, scale and descendant relative poses. This is a
    one-time initial placement; final resting poses still come from physics.

    Args:
        scene: RGB-D registered geometry, which may overlap after independent fits.
        support_nodes: Native graph nodes with object_id, parent_id and parent_relation.
        clearance_m: Initial support gap in metres, matching Scene Engine's default.

    Returns:
        Separate initial scene and recorded per-object vertical lifts.
    """
    import torch
    from embodichain.compute.geometry.aabb import group_height_offset

    if not np.isfinite(clearance_m) or not 0 <= clearance_m <= 0.05:
        raise ValueError("Initial support clearance must be between zero and 5 cm")
    description = deepcopy(scene.description)
    vertices = {uid: value.copy() for uid, value in scene.vertices.items()}
    objects = {obj["id"]: obj for obj in description["objects"]}
    nodes = {node["object_id"]: node for node in support_nodes}
    if set(objects) != set(nodes) or "table" not in objects:
        raise ValueError("Support graph must cover the registered scene")
    children = {uid: [] for uid in objects}
    for uid, node in nodes.items():
        parent = node.get("parent_id")
        if parent is not None:
            if parent not in objects:
                raise ValueError("Unknown registered support parent")
            children[parent].append(uid)
    visiting, done = set(), set()
    lifts = dict.fromkeys(objects, 0.0)
    support_heights = {}

    def descendants(uid: str) -> list[str]:
        result, pending = [], [uid]
        while pending:
            item = pending.pop()
            if item in result:
                raise ValueError("Cyclic support graph")
            result.append(item)
            pending.extend(children[item])
        return result

    def place(uid: str) -> None:
        if uid in done:
            return
        if uid in visiting:
            raise ValueError("Cyclic support graph")
        visiting.add(uid)
        node = nodes[uid]
        parent = node.get("parent_id")
        if parent is not None:
            place(parent)
            if node.get("parent_relation") == "on":
                support_z = objects[parent]["bounds"][1][2]
                if parent == "table" and "path" in objects[parent]:
                    from ._contact_fit import _support_height

                    footprint = vertices[uid][
                        np.linspace(
                            0, len(vertices[uid]) - 1, min(256, len(vertices[uid]))
                        ).astype(int),
                        :2,
                    ]
                    support_z, hit = _support_height(
                        {parent: objects[parent]},
                        {parent: np.eye(4)},
                        uid,
                        footprint,
                        -np.inf,
                    )
                    if hit is None or not np.isfinite(support_z):
                        raise ValueError(f"No upward table surface below {uid}")
                support_heights[uid] = float(support_z)
                offset, valid = group_height_offset(
                    torch.tensor([objects[uid]["bounds"]], dtype=torch.float64),
                    torch.tensor(support_z, dtype=torch.float64),
                    clearance=clearance_m,
                )
                if not bool(valid):
                    raise ValueError("Invalid bounds for initial support placement")
                lift = max(0.0, float(offset))
                for member in descendants(uid):
                    lifts[member] += lift
                    if lifts[member] > 0.15:
                        raise ValueError("Initial support separation exceeds 15 cm")
                    obj = objects[member]
                    matrix = np.asarray(obj["transform"], dtype=float)
                    matrix[2, 3] += lift
                    obj["transform"] = matrix.tolist()
                    vertices[member][:, 2] += lift
                    obj["bounds"] = [
                        vertices[member].min(0).tolist(),
                        vertices[member].max(0).tolist(),
                    ]
        visiting.remove(uid)
        done.add(uid)

    for uid in objects:
        place(uid)
    report = dict(
        method="EmbodiChain group_height_offset over native on-support relations",
        clearance_m=clearance_m,
        vertical_lift_m=lifts,
        support_height_m=support_heights,
        preserve_xy_orientation_scale=True,
    )
    description["initial_support_clearance"] = report
    return RegisteredScene(description, vertices), report


def apply_settled_poses(
    scene: RegisteredScene,
    timestamps: np.ndarray,
    object_names: tuple[str, ...],
    object_poses: np.ndarray,
    robot_contacts: np.ndarray,
) -> tuple[RegisteredScene, dict]:
    """Transfer an initial contact-free resting state to registered geometry.

    Args:
        scene: Geometry in robot-base metres, including its registered scale.
        timestamps: Nondecreasing native rollout times in seconds. A final
            checkpoint may repeat the last physics sample.
        object_names: Object order in the native pose array.
        object_poses: Observed rigid transforms, shaped (time, object, 4, 4).
        robot_contacts: Whether any robot contact occurred at each time.

    Returns:
        A separate scene and measured per-object displacement report. Retains
        the previous example's ten-second window and final 30-sample stability
        threshold; this is initial scene preparation, not task validation.
    """
    times = np.asarray(timestamps, dtype=float)
    poses = np.asarray(object_poses, dtype=float)
    contacts = np.asarray(robot_contacts, dtype=bool)
    ids = [obj["id"] for obj in scene.description["objects"]]
    if (
        times.ndim != 1
        or poses.shape != (len(times), len(object_names), 4, 4)
        or contacts.shape != times.shape
        or set(ids) != set(object_names)
        or len(set(object_names)) != len(object_names)
        or not np.isfinite(times).all()
        or not np.isfinite(poses).all()
        or not np.all(np.diff(times) >= 0)
    ):
        raise ValueError("Settling observations must match the registered scene")
    window = np.flatnonzero((times >= 0) & (times <= 10.0))
    if len(window) < 90 or contacts[window].any():
        raise ValueError("Need an initial window without robot contact for settling")
    observed = poses[window]
    if np.max(np.abs(observed[-30:] - observed[-1])) > 0.002:
        raise ValueError("Scene did not settle in the initial ten seconds")
    description = deepcopy(scene.description)
    vertices = {uid: value.copy() for uid, value in scene.vertices.items()}
    changes = {}
    for obj in description["objects"]:
        uid = obj["id"]
        if uid == "table":
            continue
        matrix = np.asarray(obj["transform"], dtype=float)
        initial = matrix.copy()
        initial[:3, :3] /= np.linalg.norm(matrix[:3, :3], axis=0)
        final = observed[-1, object_names.index(uid)]
        delta = final @ np.linalg.inv(initial)
        obj["transform"] = (delta @ matrix).tolist()
        vertices[uid] = vertices[uid] @ delta[:3, :3].T + delta[:3, 3]
        obj["bounds"] = [vertices[uid].min(0).tolist(), vertices[uid].max(0).tolist()]
        changes[uid] = {
            "world_delta": delta.tolist(),
            "origin_displacement_m": float(
                np.linalg.norm(final[:3, 3] - initial[:3, 3])
            ),
        }
    return RegisteredScene(description, vertices), {
        "source_window_seconds": [0, 10],
        "last_sample_time_s": float(times[window[-1]]),
        "changes": changes,
        "purpose": "Use native resting poses as initial scene and geometric grasp targets; no per-frame object commands",
    }
