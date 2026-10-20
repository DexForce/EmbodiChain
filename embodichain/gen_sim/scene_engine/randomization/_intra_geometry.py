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

"""Immutable content-addressed geometry preparation for hierarchical proposals."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import itertools
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from embodichain.gen_sim.scene_engine.pipeline.utils.scene_generation_utils import (
    load_glb_mesh,
)
from embodichain.lab.sim.scene._support_grid import _prepare_support
from ._geometry import _Y_UP_TO_Z_UP
from .baseline import PreparedSceneBaseline, _entry_pose, _read_json, _rigid_entries
from .cfg import IntraGroupRandomizationCfg, _intra_payload

__all__: list[str] = []
_EXTRACTOR = "parent_support_grid/v2"
_READABLE_EXTRACTORS = ("parent_support_grid/v1", _EXTRACTOR)


def _digest(data: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            data, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _cache_key(
    baseline: dict, cfg: IntraGroupRandomizationCfg, *, extractor: str | None = None
) -> tuple[str, dict]:
    settings = _intra_payload(cfg)
    # Ranges affect proposals, not geometry. Target membership affects layer choice.
    settings["targets"] = sorted(settings["targets"])
    inputs = {
        "source_sha256": baseline["source_scene"]["content_sha256"],
        "extractor": _EXTRACTOR if extractor is None else extractor,
        "settings": settings,
    }
    inputs = json.loads(json.dumps(inputs, allow_nan=False))
    return _digest(inputs), inputs


def _cache_path(
    baseline: PreparedSceneBaseline,
    cfg: IntraGroupRandomizationCfg,
    *,
    extractor: str | None = None,
) -> Path:
    key, _ = _cache_key(baseline.data, cfg, extractor=extractor)
    return baseline.path.parent / "scene_randomization_geometry" / (key + ".json")


def _load_geometry(
    path: Path,
    baseline: dict,
    cfg: IntraGroupRandomizationCfg,
    *,
    expected_sha256: str | None = None,
) -> dict:
    if path.is_symlink():
        raise ValueError("Support cache must not be a symlink.")
    _, data = _read_json(path)
    extractor = data.get("inputs", {}).get("extractor")
    if extractor not in _READABLE_EXTRACTORS:
        raise ValueError("Unsupported support geometry extraction version.")
    key, inputs = _cache_key(baseline, cfg, extractor=extractor)
    content = {k: v for k, v in data.items() if k != "data_sha256"}
    if (
        path.name != key + ".json"
        or data.get("cache_key") != key
        or data.get("inputs") != inputs
        or _digest(content) != data.get("data_sha256")
    ):
        raise ValueError("Support geometry cache fingerprint or contents changed.")
    if expected_sha256 is not None and data["data_sha256"] != expected_sha256:
        raise ValueError("Support geometry cache differs from the generated snapshot.")
    return data


def _prepare_geometry(
    baseline: PreparedSceneBaseline, cfg: IntraGroupRandomizationCfg
) -> tuple[Path, dict, bool]:
    from .persistence import _verify_baseline, _write_atomic

    export = _verify_baseline(baseline)
    path = _cache_path(baseline, cfg)
    if path.is_symlink():
        raise ValueError("Support cache must not be a symlink.")
    if path.exists():
        return path, _load_geometry(path, baseline.data, cfg), True
    _, config = _read_json(export / "scene_config.json")
    entries = {e["uid"]: e for e in _rigid_entries(config) if e["uid"] != "table"}
    parents = {g["root_id"]: None for g in baseline.data["groups"]}
    parents.update(
        {
            m["entity_id"]: m["parent_id"]
            for g in baseline.data["groups"]
            for m in g["members"]
        }
    )
    for uid in cfg.targets:
        if uid not in parents or parents[uid] is None:
            raise ValueError(
                f"Intra-group target {uid!r} must be an existing non-root member."
            )
    active = set()
    for uid in cfg.targets:
        parent = parents[uid]
        while parent is not None:
            active.add(parent)
            parent = parents[parent]
    if set(cfg.support_overrides) - active:
        raise ValueError(
            "Support overrides must refer to parents affected by intra-group targets."
        )
    # Deterministic parent-before-child order, independent of graph serialization.
    ordered: list[str] = []

    def visit(uid: str) -> None:
        ordered.append(uid)
        for child in sorted(x for x in parents if parents[x] == uid):
            visit(child)

    for uid in sorted(x for x in parents if parents[x] is None):
        visit(uid)
    vertices, faces, poses, bounds = {}, {}, {}, {}
    for uid in ordered:
        entry = entries[uid]
        mesh = load_glb_mesh(export / entry["shape"]["fpath"])
        vertices[uid] = (
            np.asarray(mesh.vertices) * entry.get("body_scale", [1, 1, 1])
        ) @ _Y_UP_TO_Z_UP.T
        faces[uid] = np.asarray(mesh.faces)
        poses[uid] = _entry_pose(entry)
        bounds[uid] = np.array([vertices[uid].min(axis=0), vertices[uid].max(axis=0)])
    nodes = []
    for uid in ordered:
        parent = parents[uid]
        relative = (
            poses[uid] if parent is None else np.linalg.inv(poses[parent]) @ poses[uid]
        )
        nodes.append(
            {
                "uid": uid,
                "parent": -1 if parent is None else ordered.index(parent),
                "relative_pose": relative.tolist(),
                "corners": [
                    list(x)
                    for x in itertools.product(*zip(bounds[uid][0], bounds[uid][1]))
                ],
            }
        )
    supports = {}
    for parent in sorted(active):
        children = [uid for uid in ordered if parents[uid] == parent]
        child_points = []
        for uid in children:
            rel = np.linalg.inv(poses[parent]) @ poses[uid]
            child_points.append(vertices[uid] @ rel[:3, :3].T + rel[:3, 3])
        override = cfg.support_overrides.get(parent)
        try:
            support = _prepare_support(
                vertices[parent],
                faces[parent],
                child_points,
                resolution=cfg.grid_resolution_m,
                margin=cfg.boundary_margin_m,
                override=(
                    None
                    if override is None
                    else {
                        "parent_from_support": override.parent_from_support,
                        "polygon": override.polygon,
                    }
                ),
            )
        except ValueError as exc:
            raise ValueError(f"Parent {parent!r}: {exc}") from exc
        support["root_min_z"] = {}
        inv = np.linalg.inv(np.asarray(support["parent_from_support"]))
        for uid in children:
            rel = inv @ np.linalg.inv(poses[parent]) @ poses[uid]
            support["root_min_z"][uid] = float(
                (vertices[uid] @ rel[:3, :3].T)[:, 2].min()
            )
        supports[parent] = support
    key, inputs = _cache_key(baseline.data, cfg)
    data = {"cache_key": key, "inputs": inputs, "nodes": nodes, "supports": supports}
    data["data_sha256"] = _digest(data)
    _verify_baseline(baseline)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Publish once. Concurrent writers must produce exactly the same artifact.
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{key}.", suffix=".json", delete=False
    ) as staging:
        temporary = Path(staging.name)
    try:
        _write_atomic(temporary, data)
        try:
            os.link(temporary, path)
        except FileExistsError:
            existing = _load_geometry(path, baseline.data, cfg)
            if existing["data_sha256"] != data["data_sha256"]:
                raise ValueError("Conflicting support geometry cache publication.")
    finally:
        temporary.unlink(missing_ok=True)
    return path, deepcopy(data), False
