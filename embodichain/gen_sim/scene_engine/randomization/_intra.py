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

"""GenSim hierarchy adaptation, canonical poses and lazy v3 candidates."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import torch

from embodichain.compute.geometry._hierarchical import _sample_hierarchy
from embodichain.lab.sim.scene.expansion.contracts import _variant_id
from ._batch import _CandidatePool, _LoadedVariants
from ._intra_geometry import _prepare_geometry, _load_geometry
from .baseline import PreparedSceneBaseline
from .cfg import SceneRandomizationCfg, _INTRA_GENERATOR, _intra_payload

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
    return np.stack(selected, axis=1)


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
            rotation = poses[:, :3, :3] @ original[:3, :3].T
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


def _metadata(
    data: dict, geometry: dict, ordinal: int, *, algorithm: str | None = None
) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            {
                "generator": _INTRA_GENERATOR if algorithm is None else algorithm,
                "baseline_sha256": data["data_sha256"],
                "support_cache_sha256": geometry["data_sha256"],
                "proposal_ordinal": str(ordinal),
                "geometry_scope": "hierarchical_support_grid_xy_yaw",
            }.items()
        )
    )


class _IntraPool(_CandidatePool):
    def __init__(
        self,
        baseline: PreparedSceneBaseline,
        geometry_path: Path,
        geometry: dict,
        deltas: np.ndarray,
        selected: np.ndarray,
        ordinals: np.ndarray,
        parameters: dict,
        entities: tuple,
    ) -> None:
        self._data = deepcopy(baseline.data)
        self._geometry = deepcopy(geometry)
        self._geometry_path = geometry_path
        self._parameters = deepcopy(parameters)
        self._deltas, self._selected, self._ordinals = (
            np.array(x, copy=True) for x in (deltas, selected, ordinals)
        )
        self._entities = deepcopy(entities)
        self._uids, self._poses, self._centers = _numeric_geometry(
            geometry,
            _with_table(parameters, baseline.data),
            self._deltas,
            self._selected,
        )
        # Parent-before-child cache order is sorted by roots, as are layout centers.
        self._data["groups"] = sorted(self._data["groups"], key=lambda g: g["root_id"])
        for array in (
            self._deltas,
            self._selected,
            self._ordinals,
            self._poses,
            self._centers,
        ):
            array.setflags(write=False)

    def _metadata_at(self, index: int) -> tuple[tuple[str, str], ...]:
        return _metadata(self._data, self._geometry, int(self._ordinals[index]))

    def _records(
        self, baseline: PreparedSceneBaseline, parameters: dict, algorithm: str
    ) -> list[dict]:
        if (
            parameters != self._parameters
            or algorithm != _INTRA_GENERATOR
            or sorted(baseline.data["groups"], key=lambda g: g["root_id"])
            != self._data["groups"]
            or baseline.data["data_sha256"] != self._data["data_sha256"]
        ):
            raise ValueError(
                "Hierarchical candidate provenance differs from its snapshot."
            )
        from .cfg import _decode_intra

        _load_geometry(
            self._geometry_path,
            baseline.data,
            _decode_intra(parameters["intra_group"]),
            expected_sha256=self._geometry["data_sha256"],
        )
        _numeric_geometry(
            self._geometry,
            _with_table(parameters, baseline.data),
            self._deltas,
            self._selected,
        )
        records = []
        roots = [n["uid"] for n in self._geometry["nodes"] if n["parent"] < 0]
        targets = [
            self._geometry["nodes"][i]["uid"]
            for i in _target_indices(self._geometry, parameters)
        ]
        for index, ordinal in enumerate(self._ordinals):
            ordinal = int(ordinal)
            metadata = self._metadata_at(index)
            identity = _variant_id(
                self._data["source_scene"]["content_sha256"],
                parameters["seed"],
                index,
                tuple(zip(self._uids, self._poses[index].tolist(), strict=True)),
                metadata,
            )
            records.append(
                {
                    "candidate_id": f"candidate_{ordinal:06d}",
                    "proposal_ordinal": ordinal,
                    "variant_id": identity,
                    "variant_ordinal": index,
                    "variant_metadata": dict(metadata),
                    "root_xy_deltas": dict(
                        zip(roots, self._deltas[index].tolist(), strict=True)
                    ),
                    "member_poses_in_parent": dict(
                        zip(targets, self._selected[index].tolist(), strict=True)
                    ),
                }
            )
        return records


class _LoadedIntraVariants(_LoadedVariants):
    def __init__(
        self, baseline: dict, geometry: dict, records: list[dict], policy: dict
    ) -> None:
        roots = [n["uid"] for n in geometry["nodes"] if n["parent"] < 0]
        targets = [
            geometry["nodes"][i]["uid"] for i in _target_indices(geometry, policy)
        ]
        deltas, selected = [], []
        for record in records:
            if set(record["root_xy_deltas"]) != set(roots) or set(
                record["member_poses_in_parent"]
            ) != set(targets):
                raise ValueError(
                    "Saved roots/selected members do not match the support hierarchy."
                )
            delta = [record["root_xy_deltas"][uid] for uid in roots]
            poses = [record["member_poses_in_parent"][uid] for uid in targets]
            if any(
                not isinstance(row, list)
                or len(row) != 2
                or any(type(v) not in (int, float) for v in row)
                for row in delta
            ):
                raise ValueError("Root XY translations must be numeric pairs.")
            if any(
                not isinstance(pose, list)
                or len(pose) != 4
                or any(
                    not isinstance(row, list)
                    or len(row) != 4
                    or any(type(v) not in (int, float) for v in row)
                    for row in pose
                )
                for pose in poses
            ):
                raise ValueError("Member poses must contain numeric 4x4 matrices.")
            deltas.append(delta)
            selected.append(poses)
        self._source, self._seed = (
            baseline["source_scene"]["content_sha256"],
            policy["seed"],
        )
        self._uids, self._poses, _ = _numeric_geometry(
            geometry,
            _with_table(policy, baseline),
            np.asarray(deltas, dtype=np.float64),
            np.asarray(selected, dtype=np.float64),
        )
        self._poses.setflags(write=False)
        self._metadata = []
        for index, record in enumerate(records):
            metadata = _metadata(
                baseline,
                geometry,
                record["proposal_ordinal"],
                algorithm=record["algorithm_version"],
            )
            if (
                dict(metadata) != record["variant_metadata"]
                or _variant_id(
                    self._source,
                    self._seed,
                    index,
                    tuple(zip(self._uids, self._poses[index].tolist(), strict=True)),
                    metadata,
                )
                != record["variant_id"]
            ):
                raise ValueError(
                    "Reconstructed hierarchical variant identity or provenance does not match."
                )
            self._metadata.append(metadata)


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


def _randomize_intra(
    cfg: SceneRandomizationCfg, baseline: PreparedSceneBaseline, parameters: dict
):
    from .generation import SceneRandomizationResult, _runtime_entities

    device = torch.device(cfg.device)
    if device.type == "cuda":
        if not torch.cuda.is_available() or (
            device.index is not None and device.index >= torch.cuda.device_count()
        ):
            raise ValueError(
                "CUDA hierarchical generation requested but device is unavailable."
            )
        device = torch.device(
            "cuda",
            torch.cuda.current_device() if device.index is None else device.index,
        )
    path, geometry, _ = _prepare_geometry(baseline, cfg.intra_group)
    batch_limit = (
        _hierarchy_batch_limit(
            cfg.batch_size, sum(n["parent"] < 0 for n in geometry["nodes"])
        )
        if cfg.randomize_groups
        else cfg.batch_size
    )
    resolved = {
        **parameters,
        "backend": "batched",
        "device": str(device),
        "batch_size": batch_limit,
        "randomize_groups": cfg.randomize_groups,
        "intra_group": _intra_payload(cfg.intra_group),
    }
    spec = _specification(geometry, resolved, device)
    table = torch.tensor(
        _with_table(resolved, baseline.data)["_table"],
        dtype=torch.float64,
        device=device,
    )
    deltas, samples, ordinals = [], [], []
    accepted = attempts = 0
    while (
        attempts < parameters["max_attempts"]
        and accepted < parameters["candidate_count"]
    ):
        count = min(batch_limit, parameters["max_attempts"] - attempts)
        delta, sample, valid = _sample_hierarchy(
            *spec,
            table,
            start=attempts,
            count=count,
            randomize_groups=cfg.randomize_groups,
            sibling_margin=cfg.intra_group.sibling_margin_m,
            group_margin=parameters["collision_margin_m"],
            support_gap=cfg.intra_group.support_gap_m,
        )
        chosen = torch.nonzero(valid, as_tuple=False).flatten()[
            : parameters["candidate_count"] - accepted
        ]
        picked = chosen.cpu().numpy() + attempts
        deltas.append(delta[chosen].cpu().numpy())
        samples.append(sample[chosen].cpu().numpy())
        ordinals.append(picked)
        accepted += len(picked)
        attempts = (
            int(picked[-1]) + 1
            if accepted == parameters["candidate_count"]
            else attempts + count
        )
    sample_array = np.concatenate(samples)
    selected = _sample_poses(geometry, resolved, sample_array)
    from .persistence import _verify_baseline
    from .baseline import _read_json, _rigid_entries

    export = _verify_baseline(baseline)
    _, config = _read_json(export / "scene_config.json")
    entities = _runtime_entities(_rigid_entries(config), export)
    pool = _IntraPool(
        baseline,
        path,
        geometry,
        np.concatenate(deltas),
        selected,
        np.concatenate(ordinals),
        resolved,
        entities,
    )
    return SceneRandomizationResult(
        baseline,
        pool,
        attempts,
        (
            "completed"
            if accepted == parameters["candidate_count"]
            else "budget_exhausted"
        ),
        {"solved": accepted, "no_solution": attempts - accepted},
        resolved,
        _INTRA_GENERATOR,
    )
