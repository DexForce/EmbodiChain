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

"""Canonical numeric layout reconstruction and independent certificates."""

from __future__ import annotations
import hashlib
import numpy as np
import torch

__all__: list[str] = []


def _stream(seed: int, stage: str, uid: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}/{stage}/{uid}".encode()).digest()[:4], "little"
    )


def _specification(geometry: dict, policy: dict, device: torch.device) -> tuple:
    nodes = geometry["nodes"]
    index = {n["uid"]: i for i, n in enumerate(nodes)}
    tensor = lambda value: torch.tensor(value, dtype=torch.float64, device=device)
    supports = {
        index[uid]: {
            "frame": tensor(s["parent_from_support"]),
            "integral": torch.tensor(
                s["unsupported_integral"], dtype=torch.int64, device=device
            ),
            "origin": tensor(s["grid_origin"]),
            "resolution": s["resolution"],
            "height_ceiling": s.get("height_ceiling_m", 0.0),
        }
        for uid, s in geometry["supports"].items()
    }
    targets = {}
    for uid, cfg in policy["intra_group"]["targets"].items():
        node = index[uid]
        parent = nodes[nodes[node]["parent"]]["uid"]
        targets[node] = {
            "key": _stream(policy["seed"], "member", uid),
            "xy": (
                None
                if cfg["xy_offset_bounds_m"] is None
                else tensor(cfg["xy_offset_bounds_m"])
            ),
            "yaw": cfg["yaw_delta_range_deg"],
            "min_z": geometry["supports"][parent]["root_min_z"][uid],
        }
    roots = {
        i: _stream(policy["seed"], "group", n["uid"])
        for i, n in enumerate(nodes)
        if n["parent"] < 0
    }
    return (
        tensor([n["relative_pose"] for n in nodes]),
        tensor([n["corners"] for n in nodes]),
        tuple(n["parent"] for n in nodes),
        supports,
        targets,
        roots,
    )


def _target_indices(geometry: dict, policy: dict) -> list[int]:
    return [
        i
        for i, n in enumerate(geometry["nodes"])
        if n["uid"] in policy["intra_group"]["targets"]
    ]


def _sample_poses(geometry: dict, policy: dict, samples: np.ndarray) -> np.ndarray:
    """Canonical CPU float64 reconstruction from addressed XY/yaw samples."""
    selected = []
    nodes = geometry["nodes"]
    for col, node in enumerate(_target_indices(geometry, policy)):
        uid = nodes[node]["uid"]
        parent = nodes[nodes[node]["parent"]]["uid"]
        support = geometry["supports"][parent]
        frame = np.asarray(support["parent_from_support"])
        original = np.linalg.inv(frame) @ np.asarray(nodes[node]["relative_pose"])
        angle = np.deg2rad(samples[:, col, 2])
        rotation = np.zeros((len(samples), 3, 3))
        rotation[:, 0, 0] = rotation[:, 1, 1] = np.cos(angle)
        rotation[:, 1, 0] = np.sin(angle)
        rotation[:, 0, 1] = -np.sin(angle)
        rotation[:, 2, 2] = 1
        pose = np.broadcast_to(np.eye(4), (len(samples), 4, 4)).copy()
        pose[:, :3, :3] = rotation @ original[:3, :3]
        pose[:, :2, 3] = samples[:, col, :2]
        pose[:, 2, 3] = (
            policy["intra_group"]["support_gap_m"]
            + support.get("height_ceiling_m", 0.0)
            - support["root_min_z"][uid]
        )
        selected.append(frame @ pose)
    return np.stack(selected, axis=1) if selected else np.empty((len(samples), 0, 4, 4))


def _numeric_geometry(
    geometry: dict,
    policy: dict,
    deltas: np.ndarray,
    selected: np.ndarray,
    *,
    validate: bool = True,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Independent NumPy certificate over every reconstructed body/subtree."""
    nodes = geometry["nodes"]
    count = len(deltas)
    roots = [i for i, n in enumerate(nodes) if n["parent"] < 0]
    targets = _target_indices(geometry, policy)
    if (
        deltas.shape != (count, len(roots), 2)
        or selected.shape != (count, len(targets), 4, 4)
        or not np.isfinite(deltas).all()
        or not np.isfinite(selected).all()
    ):
        raise ValueError("Invalid hierarchical root translations or member poses.")
    if not policy["randomize_groups"] and np.any(deltas != 0):
        raise ValueError("Intra-only layouts must keep group roots fixed.")
    for column, root in enumerate(roots):
        bounds = policy.get("group_xy_offset_bounds_m", {}).get(nodes[root]["uid"])
        if bounds is not None and (
            np.any(deltas[:, column] < np.asarray(bounds)[0] - 1e-9)
            or np.any(deltas[:, column] > np.asarray(bounds)[1] + 1e-9)
        ):
            raise ValueError("Group root violates explicit XY offset bounds.")
    relative = np.broadcast_to(
        np.array([n["relative_pose"] for n in nodes]), (count, len(nodes), 4, 4)
    ).copy()
    relative[:, targets] = selected
    if validate:
        for col, node in enumerate(targets):
            uid = nodes[node]["uid"]
            parent = nodes[nodes[node]["parent"]]["uid"]
            support = geometry["supports"][parent]
            inv = np.linalg.inv(np.asarray(support["parent_from_support"]))
            original = inv @ np.asarray(nodes[node]["relative_pose"])
            poses = inv @ selected[:, col]
            # Accepted authored matrices can retain float32 roundoff. Use the
            # actual inverse so a pure yaw still cancels their original tilt
            # without weakening the independent tilt-change check.
            rotation = poses[:, :3, :3] @ np.linalg.inv(original[:3, :3])
            angle = np.rad2deg(np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0]))
            rad = np.deg2rad(angle)
            expected = np.zeros_like(rotation)
            expected[:, 0, 0] = expected[:, 1, 1] = np.cos(rad)
            expected[:, 1, 0] = np.sin(rad)
            expected[:, 0, 1] = -np.sin(rad)
            expected[:, 2, 2] = 1
            cfg = policy["intra_group"]["targets"][uid]
            midpoint = (
                cfg["yaw_delta_range_deg"][0]
                + (cfg["yaw_delta_range_deg"][1] - cfg["yaw_delta_range_deg"][0]) / 2
            )
            angle += 360 * np.floor((midpoint - angle) / 360 + 0.5)
            height = (
                policy["intra_group"]["support_gap_m"]
                + support.get("height_ceiling_m", 0.0)
                - support["root_min_z"][uid]
            )
            if (
                not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-10, rtol=0)
                or not np.allclose(rotation, expected, atol=1e-9, rtol=0)
                or np.any(angle < cfg["yaw_delta_range_deg"][0] - 1e-8)
                or np.any(angle > cfg["yaw_delta_range_deg"][1] + 1e-8)
                or np.any(np.abs(poses[:, 2, 3] - height) > 1e-9)
            ):
                raise ValueError(
                    "Member pose violates yaw/tilt/support-height constraints."
                )
            bounds = cfg["xy_offset_bounds_m"]
            if bounds is not None:
                offset = poses[:, :2, 3] - original[:2, 3]
                if np.any(offset < np.array(bounds)[0] - 1e-9) or np.any(
                    offset > np.array(bounds)[1] + 1e-9
                ):
                    raise ValueError(
                        "Member origin violates explicit XY offset bounds."
                    )
    world = []
    for node, n in enumerate(nodes):
        pose = (
            relative[:, node]
            if n["parent"] < 0
            else world[n["parent"]] @ relative[:, node]
        )
        if node in roots:
            pose = pose.copy()
            pose[:, :2, 3] += deltas[:, roots.index(node)]
        world.append(pose)
    points = [
        np.asarray(n["corners"]) @ world[i][:, :3, :3].transpose(0, 2, 1)
        + world[i][:, None, :3, 3]
        for i, n in enumerate(nodes)
    ]

    def descendants(root: int) -> list[int]:
        found = [root]
        for i in range(root + 1, len(nodes)):
            if nodes[i]["parent"] in found:
                found.append(i)
        return found

    centers, group_boxes = [], []
    for root in roots:
        cloud = np.concatenate([points[i] for i in descendants(root)], axis=1)
        box = np.stack((cloud[:, :, :2].min(1), cloud[:, :, :2].max(1)), axis=1)
        group_boxes.append(box)
        centers.append(box.mean(1))
    if validate:
        table = policy["_table"]
        for i, box in enumerate(group_boxes):
            if np.any(box[:, 0] < np.asarray(table)[0] - 1e-8) or np.any(
                box[:, 1] > np.asarray(table)[1] + 1e-8
            ):
                raise ValueError("Hierarchical group lies outside the table bounds.")
            for other in group_boxes[:i]:
                gap = np.maximum(box[:, 0] - other[:, 1], other[:, 0] - box[:, 1]).max(
                    -1
                )
                if np.any(gap < policy["collision_margin_m"] - 1e-8):
                    raise ValueError(
                        "Hierarchical group clearance is below the required margin."
                    )
        for parent_uid, support in geometry["supports"].items():
            parent = next(i for i, n in enumerate(nodes) if n["uid"] == parent_uid)
            support_world = world[parent] @ np.asarray(support["parent_from_support"])
            inverse = np.linalg.inv(support_world)
            sibling_boxes = []
            integral = np.asarray(support["unsupported_integral"], dtype=np.int64)
            for child, n in enumerate(nodes):
                if n["parent"] != parent:
                    continue
                cloud = np.concatenate([points[i] for i in descendants(child)], axis=1)
                local_points = (
                    cloud @ inverse[:, :3, :3].transpose(0, 2, 1)
                    + inverse[:, None, :3, 3]
                )
                bounds = np.stack(
                    (local_points[:, :, :2].min(1), local_points[:, :, :2].max(1)),
                    axis=1,
                )
                coordinates = (bounds - support["grid_origin"]) / support["resolution"]
                lo = np.floor(coordinates[:, 0] + 1e-8).astype(int)
                hi = np.ceil(coordinates[:, 1] - 1e-8).astype(int)
                size = np.array(integral.shape) - 1
                if np.any(lo < 0) or np.any(hi > size) or np.any(hi <= lo):
                    raise ValueError(
                        "Member subtree extends outside its support raster."
                    )
                bad = (
                    integral[hi[:, 0], hi[:, 1]]
                    - integral[lo[:, 0], hi[:, 1]]
                    - integral[hi[:, 0], lo[:, 1]]
                    + integral[lo[:, 0], lo[:, 1]]
                )
                if np.any(bad):
                    raise ValueError(
                        "Member subtree crosses unsupported cells or holes."
                    )
                for other in sibling_boxes:
                    if np.any(
                        np.maximum(
                            bounds[:, 0] - other[:, 1], other[:, 0] - bounds[:, 1]
                        ).max(-1)
                        < policy["intra_group"]["sibling_margin_m"] - 1e-8
                    ):
                        raise ValueError(
                            "Sibling subtree clearance is below the required margin."
                        )
                sibling_boxes.append(bounds)
    movable = (
        set(range(len(nodes)))
        if policy["randomize_groups"]
        else {i for target in targets for i in descendants(target)}
    )
    order = sorted(movable, key=lambda i: nodes[i]["uid"])
    return (
        tuple(nodes[i]["uid"] for i in order),
        np.stack([world[i] for i in order], axis=1),
        np.stack(centers, axis=1),
    )


def _with_table(policy: dict, baseline: dict) -> dict:
    table = baseline["table"]
    center, half = np.asarray(table["center_xy"]), np.asarray(table["size_xy"]) / 2
    return {**policy, "_table": np.stack((center - half, center + half)).tolist()}


def _hierarchy_batch_limit(requested: int, root_count: int) -> int:
    """Bound free-region temporaries to a 256 MiB working allocation budget.

    The bound estimates 40 bytes per edge cell for occupancy chunks, weights
    and cumulative weights. This does not claim a bound on total process RSS.
    """
    cells = (2 * max(0, root_count - 1) + 1) ** 2
    budget = 256 * 1024**2
    if 40 * cells > budget:
        raise ValueError(
            "Group count exceeds the hierarchical free-region memory budget."
        )
    return min(requested, max(1, budget // (40 * cells)))
