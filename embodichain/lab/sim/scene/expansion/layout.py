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

"""Source-independent preparation, sampling and selected-row scene application."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import torch

from embodichain.compute.geometry._hierarchical import _sample_hierarchy
from .._support_grid import _prepare_support
from .._validation import _identifier, _rigid_pose
from ..geometry import detect_support_region
from ..query import _backend_tensor
from .contracts import ScenePoseChange, SceneVariant
from ._layout_numeric import (
    _hierarchy_batch_limit,
    _numeric_geometry,
    _sample_poses,
    _specification,
)
from ._policy import (
    IntraGroupRandomizationCfg,
    SceneRandomizationPolicyCfg,
    _decode_intra,
    _intra_payload,
    _validate_backend,
    _validate_generation_parameters,
    _validate_intra,
)

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager

__all__ = [
    "SceneLayoutInput",
    "PreparedSceneRandomization",
    "SceneRandomizationBatch",
    "SceneVariantApplicationResult",
    "scene_layout_input_from_sim",
    "prepare_scene_randomization",
    "generate_scene_variants",
    "apply_scene_variants",
]

_ALGORITHM = "scene.hierarchical_batch_rejection/v1"


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _owned_array(value: object, *, faces: bool = False) -> np.ndarray:
    try:
        raw = np.asarray(value)
        if faces and raw.dtype.kind not in "iu":
            raise ValueError("Triangle indices must be integers.")
        array = np.array(value, dtype=np.int64 if faces else np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Scene geometry must contain finite numeric arrays.") from exc
    if (
        array.ndim != 2
        or array.shape[1] != 3
        or not len(array)
        or not np.isfinite(array).all()
    ):
        raise ValueError("Scene geometry requires nonempty finite (N, 3) arrays.")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class SceneLayoutInput:
    """Own scaled local mesh geometry and a rooted physical support tree.

    Args:
        scene_id: Caller-owned source identity.
        support_uid: Main horizontal support, with an arbitrary native UID.
        poses: Proper entity-to-local-arena transforms, excluding replica offsets.
        vertices: Local Z-up vertices with body scale already applied once.
        faces: Triangle indices for each entity's complete mesh.
        parent_by_uid: Parent UID for every entity except the main support.
        movable_entity_ids: Allowed changed entities; None permits all children.

    Inputs are copied and read-only. Unselected descendants follow their parent;
    entities excluded from the permission set must remain fixed in arena space.
    """

    scene_id: str
    support_uid: str
    poses: Mapping[str, object]
    vertices: Mapping[str, object]
    faces: Mapping[str, object]
    parent_by_uid: Mapping[str, str]
    movable_entity_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        _identifier(self.scene_id, "scene_id")
        _identifier(self.support_uid, "support_uid")
        uids = set(self.poses)
        if (
            self.support_uid not in uids
            or len(uids) < 2
            or set(self.vertices) != uids
            or set(self.faces) != uids
        ):
            raise ValueError(
                "Poses and mesh geometry must describe the same support and entity set."
            )
        parents = dict(self.parent_by_uid)
        if set(parents) != uids - {self.support_uid}:
            raise ValueError("Every non-support entity requires exactly one parent.")
        for uid in sorted(uids):
            _identifier(uid, "entity UID")
            visited = set()
            current = uid
            while current != self.support_uid:
                if (
                    current in visited
                    or current not in parents
                    or parents[current] not in uids
                ):
                    raise ValueError(
                        "Support relationships must form one acyclic rooted tree."
                    )
                visited.add(current)
                current = parents[current]
        permissions = (
            tuple(sorted(parents))
            if self.movable_entity_ids is None
            else tuple(self.movable_entity_ids)
        )
        if len(set(permissions)) != len(permissions) or set(permissions) - set(parents):
            raise ValueError(
                "Movable permissions must name unique non-support entities."
            )
        poses = {uid: _rigid_pose(self.poses[uid]) for uid in sorted(uids)}
        vertices = {uid: _owned_array(self.vertices[uid]) for uid in sorted(uids)}
        faces = {uid: _owned_array(self.faces[uid], faces=True) for uid in sorted(uids)}
        if any(
            np.any(faces[uid] < 0) or np.any(faces[uid] >= len(vertices[uid]))
            for uid in uids
        ):
            raise ValueError("Triangle index is outside its entity's vertices.")
        object.__setattr__(self, "poses", MappingProxyType(poses))
        object.__setattr__(self, "vertices", MappingProxyType(vertices))
        object.__setattr__(self, "faces", MappingProxyType(faces))
        object.__setattr__(self, "parent_by_uid", MappingProxyType(parents))
        object.__setattr__(self, "movable_entity_ids", permissions)
        object.__setattr__(self, "_fingerprint", _digest(self.to_dict()))

    def to_dict(self) -> dict:
        """Return an owned, JSON-compatible source snapshot."""
        return {
            "scene_id": self.scene_id,
            "support_uid": self.support_uid,
            "poses": {
                uid: [list(row) for row in pose] for uid, pose in self.poses.items()
            },
            "vertices": {uid: value.tolist() for uid, value in self.vertices.items()},
            "faces": {uid: value.tolist() for uid, value in self.faces.items()},
            "parent_by_uid": dict(self.parent_by_uid),
            "movable_entity_ids": list(self.movable_entity_ids),
        }

    @property
    def fingerprint(self) -> str:
        """Return the content identity of the complete authored source."""
        return self._fingerprint


def scene_layout_input_from_sim(
    sim: SimulationManager,
    *,
    support_uid: str,
    parent_by_uid: Mapping[str, str],
    movable_entity_ids: Sequence[str] | None = None,
    scene_id: str = "authored-scene",
    env_id: int = 0,
) -> SceneLayoutInput:
    """Snapshot prepared rigid entities through public simulation interfaces.

    Args:
        sim: Prepared caller-owned simulation manager.
        support_uid: Main support entity.
        parent_by_uid: Explicit native UID support tree.
        movable_entity_ids: Optional changed-entity permissions.
        scene_id: Source identity, independent of physical environment slots.
        env_id: Prototype row used only to extract authored arena-local geometry.

    Returns:
        Owned input with scale applied by the rigid-object geometry API.
    """
    if type(env_id) is not int or not 0 <= env_id < sim.num_envs:
        raise ValueError("Prototype environment ID is invalid.")
    uids = sorted({support_uid, *parent_by_uid})
    if set(uids) - set(sim.get_rigid_object_uid_list()):
        raise ValueError("Layout inputs require existing rigid-object UIDs.")
    poses, vertices, faces = {}, {}, {}
    for uid in uids:
        entity = sim.get_rigid_object(uid)
        poses[uid] = (
            entity.get_local_pose(to_matrix=True)[env_id].detach().cpu().numpy()
        )
        vertices[uid] = (
            entity.get_vertices(env_ids=[env_id], scale=True)[0].detach().cpu().numpy()
        )
        faces[uid] = entity.get_triangles(env_ids=[env_id])[0].detach().cpu().numpy()
        if hasattr(entity, "get_body_scale"):
            scales = entity.get_body_scale().detach().cpu().numpy()
            if not np.allclose(scales, scales[env_id], rtol=0, atol=1e-8):
                raise ValueError("Layout replicas must share prepared asset scales.")
    return SceneLayoutInput(
        scene_id,
        support_uid,
        poses,
        vertices,
        faces,
        parent_by_uid,
        None if movable_entity_ids is None else tuple(movable_entity_ids),
    )


def _policy_data(cfg: SceneRandomizationPolicyCfg | Mapping) -> dict:
    if isinstance(cfg, Mapping):
        raw = dict(cfg)
        cfg = SceneRandomizationPolicyCfg(**raw)
    if not isinstance(cfg, SceneRandomizationPolicyCfg):
        raise TypeError("policy must be a SceneRandomizationPolicyCfg or mapping")
    cfg = cfg.copy()
    cfg.validate()
    _validate_backend(backend=cfg.backend, device=cfg.device, batch_size=cfg.batch_size)
    if cfg.backend != "batched":
        raise ValueError(
            "The source-independent layout API supports backend='batched'."
        )
    _validate_generation_parameters(
        seed=0,
        candidate_count=1,
        max_attempts=1,
        collision_margin_m=cfg.collision_margin_m,
    )
    if type(cfg.randomize_groups) is not bool:
        raise ValueError("randomize_groups must be a boolean.")
    intra = cfg.intra_group
    if isinstance(intra, dict) or intra is None:
        intra = _decode_intra(intra)
    if intra is None:
        intra = IntraGroupRandomizationCfg()
    _validate_intra(intra)
    if cfg.group_targets is not None and (
        isinstance(cfg.group_targets, str)
        or len(set(cfg.group_targets)) != len(cfg.group_targets)
    ):
        raise ValueError("group_targets must contain unique root UIDs.")
    return {
        "backend": cfg.backend,
        "device": cfg.device,
        "batch_size": cfg.batch_size,
        "collision_margin_m": float(cfg.collision_margin_m),
        "randomize_groups": cfg.randomize_groups,
        "group_targets": None if cfg.group_targets is None else list(cfg.group_targets),
        "intra_group": _intra_payload(intra),
    }


@dataclass(frozen=True)
class PreparedSceneRandomization:
    """Own reusable numeric geometry, policy, permissions and source identity."""

    source: SceneLayoutInput
    _policy: dict
    geometry_sha256: str
    movable_uids: tuple[str, ...]
    _geometry: dict
    _table: tuple
    _movable_roots: tuple[int, ...]

    @property
    def policy(self) -> dict:
        """Return an owned policy snapshot, preserving prepared geometry ownership."""
        return deepcopy(self._policy)


def prepare_scene_randomization(
    source: SceneLayoutInput,
    policy: SceneRandomizationPolicyCfg | Mapping,
    *,
    cache_dir: str | Path | None = None,
) -> PreparedSceneRandomization:
    """Prepare source-independent hierarchy and conservative support geometry.

    Args:
        source: Owned authored scene geometry and explicit support tree.
        policy: CPU/CUDA sampling policy, without runtime seed or file inputs.
        cache_dir: Optional content-addressed support-cache directory.

    Returns:
        Prepared geometry reusable across generation calls and ordinary resets.
    """
    if not isinstance(source, SceneLayoutInput):
        raise TypeError("source must be a SceneLayoutInput")
    policy = _policy_data(policy)
    parents = dict(source.parent_by_uid)
    ordered = []

    def visit(parent: str) -> None:
        for uid in sorted(uid for uid in parents if parents[uid] == parent):
            ordered.append(uid)
            visit(uid)

    visit(source.support_uid)
    roots = [uid for uid in ordered if parents[uid] == source.support_uid]
    targets = policy["intra_group"]["targets"]
    if set(targets) - (set(ordered) - set(roots)):
        raise ValueError("Intra-group targets must name non-root members.")

    def subtree(uid: str) -> set[str]:
        children = {uid}
        for child in ordered[ordered.index(uid) + 1 :]:
            if parents[child] in children:
                children.add(child)
        return children

    permitted = set(source.movable_entity_ids)
    selected_roots = set(
        roots if policy["group_targets"] is None else policy["group_targets"]
    )
    if selected_roots - set(roots):
        raise ValueError("group_targets must name group roots.")
    if policy["group_targets"] is None:
        selected_roots = {uid for uid in roots if subtree(uid) <= permitted}
    elif any(not subtree(uid) <= permitted for uid in selected_roots):
        raise ValueError("A moving group contains fixed entities.")
    if not policy["randomize_groups"]:
        selected_roots.clear()
    moving = set().union(*(subtree(uid) for uid in selected_roots | set(targets)))
    if not moving or moving - permitted:
        raise ValueError(
            "No permitted layout changes, or a moving subtree contains fixed entities."
        )
    active = set()
    for uid in targets:
        parent = parents[uid]
        while parent != source.support_uid:
            active.add(parent)
            parent = parents[parent]
    if set(policy["intra_group"]["support_overrides"]) - active:
        raise ValueError("Support overrides must reference active member parents.")
    nodes = []
    for uid in ordered:
        parent = parents[uid]
        pose = np.asarray(source.poses[uid])
        relative = (
            pose
            if parent == source.support_uid
            else np.linalg.inv(source.poses[parent]) @ pose
        )
        points = source.vertices[uid]
        nodes.append(
            {
                "uid": uid,
                "parent": -1 if parent == source.support_uid else ordered.index(parent),
                "relative_pose": relative.tolist(),
                "corners": [
                    list(v)
                    for v in itertools.product(*zip(points.min(0), points.max(0)))
                ],
            }
        )
    key = _digest(
        {
            "source": source.fingerprint,
            "extractor": "scene-layout-support/v1",
            "intra": policy["intra_group"],
        }
    )
    geometry = None
    path = None if cache_dir is None else Path(cache_dir) / (key + ".json")
    if path is not None and path.exists():
        from .persistence import _read_json

        cached = _read_json(path)
        if cached.get("key") != key or cached.get("sha256") != _digest(
            cached.get("geometry")
        ):
            raise ValueError("Prepared support cache contents changed.")
        geometry = cached["geometry"]
    if geometry is None:
        supports = {}
        for parent in sorted(active):
            children = [uid for uid in ordered if parents[uid] == parent]
            child_points = []
            for child in children:
                rel = np.linalg.inv(source.poses[parent]) @ np.asarray(
                    source.poses[child]
                )
                child_points.append(source.vertices[child] @ rel[:3, :3].T + rel[:3, 3])
            intra = policy["intra_group"]
            support = _prepare_support(
                source.vertices[parent],
                source.faces[parent],
                child_points,
                resolution=intra["grid_resolution_m"],
                margin=intra["boundary_margin_m"],
                override=intra["support_overrides"].get(parent),
            )
            inverse = np.linalg.inv(support["parent_from_support"])
            support["root_min_z"] = {}
            for child in children:
                rel = (
                    inverse
                    @ np.linalg.inv(source.poses[parent])
                    @ np.asarray(source.poses[child])
                )
                support["root_min_z"][child] = float(
                    (source.vertices[child] @ rel[:3, :3].T)[:, 2].min()
                )
            supports[parent] = support
        geometry = {"nodes": nodes, "supports": supports}
        if path is not None:
            from .persistence import _write_atomic

            _write_atomic(
                path, {"key": key, "sha256": _digest(geometry), "geometry": geometry}
            )
    transform = np.asarray(source.poses[source.support_uid])
    points = (
        source.vertices[source.support_uid] @ transform[:3, :3].T + transform[:3, 3]
    )
    region = detect_support_region(
        source.support_uid, points, source.faces[source.support_uid]
    )
    rectangle = region.optimization_rectangle
    if rectangle is None:
        raise ValueError(
            "Main support has no conservative rectangular sampling region."
        )
    lo_x, lo_y, hi_x, hi_y = rectangle.bounds
    return PreparedSceneRandomization(
        source,
        deepcopy(policy),
        _digest(geometry),
        tuple(sorted(moving)),
        geometry,
        ((lo_x, lo_y), (hi_x, hi_y)),
        tuple(ordered.index(uid) for uid in roots if uid in selected_roots),
    )


class _VariantPool(Sequence[SceneVariant]):
    def __init__(
        self,
        parent: str,
        seed: int,
        uids: tuple[str, ...],
        poses: np.ndarray,
        ordinals: np.ndarray,
        metadata: tuple[tuple[str, str], ...],
    ) -> None:
        self.parent, self.seed, self.uids, self.metadata = parent, seed, uids, metadata
        self.poses, self.ordinals = np.array(poses, copy=True), np.array(
            ordinals, copy=True
        )
        self.poses.setflags(write=False)
        self.ordinals.setflags(write=False)

    def __len__(self) -> int:
        return len(self.ordinals)

    def __getitem__(
        self, index: int | slice
    ) -> SceneVariant | tuple[SceneVariant, ...]:
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        ordinal = int(self.ordinals[index])
        return SceneVariant(
            self.parent,
            self.seed,
            ordinal,
            tuple(
                ScenePoseChange(uid, pose)
                for uid, pose in zip(self.uids, self.poses[index])
            ),
            tuple(sorted((*self.metadata, ("proposal_ordinal", str(ordinal))))),
        )


@dataclass(frozen=True)
class SceneRandomizationBatch:
    """Owned lazy proposals and exact bounded generation accounting."""

    prepared: PreparedSceneRandomization
    variants: Sequence[SceneVariant]
    attempts: int
    status: str
    seed: int
    start_ordinal: int
    next_ordinal: int
    requested_count: int | None = None
    max_attempts: int | None = None


def generate_scene_variants(
    prepared: PreparedSceneRandomization,
    *,
    seed: int = 0,
    candidate_count: int = 10,
    max_attempts: int = 1000,
    start_ordinal: int = 0,
) -> SceneRandomizationBatch:
    """Generate a batch without loading meshes or constructing simulation objects.

    Args:
        prepared: Reusable geometry and policy.
        seed: Addressed random stream seed.
        candidate_count: Requested valid layouts.
        max_attempts: Whole-layout proposal budget.
        start_ordinal: First logical proposal; allows bounded continuation.

    Returns:
        Complete or budget-exhausted proposals with exact continuation ordinal.
    """
    _validate_generation_parameters(
        seed=seed,
        candidate_count=candidate_count,
        max_attempts=max_attempts,
        collision_margin_m=prepared.policy["collision_margin_m"],
    )
    if type(start_ordinal) is not int or start_ordinal < 0:
        raise ValueError("start_ordinal must be nonnegative.")
    policy = {**deepcopy(prepared.policy), "seed": seed, "_table": prepared._table}
    device = torch.device(policy["device"])
    if device.type == "cuda" and (
        not torch.cuda.is_available()
        or (device.index is not None and device.index >= torch.cuda.device_count())
    ):
        raise ValueError("Requested CUDA device is unavailable.")
    spec = _specification(prepared._geometry, policy, device)
    limit = (
        _hierarchy_batch_limit(policy["batch_size"], len(spec[-1]))
        if policy["randomize_groups"]
        else policy["batch_size"]
    )
    deltas, samples, ordinals = [], [], []
    accepted = attempts = 0
    while attempts < max_attempts and accepted < candidate_count:
        count = min(limit, max_attempts - attempts)
        delta, sample, valid = _sample_hierarchy(
            *spec,
            torch.tensor(prepared._table, dtype=torch.float64, device=device),
            start=start_ordinal + attempts,
            count=count,
            randomize_groups=policy["randomize_groups"],
            sibling_margin=policy["intra_group"]["sibling_margin_m"],
            group_margin=policy["collision_margin_m"],
            support_gap=policy["intra_group"]["support_gap_m"],
            movable_roots=prepared._movable_roots,
        )
        selected = torch.nonzero(valid, as_tuple=False).flatten()[
            : candidate_count - accepted
        ]
        picked = selected.cpu().numpy() + start_ordinal + attempts
        deltas.append(delta[selected].cpu().numpy())
        samples.append(sample[selected].cpu().numpy())
        ordinals.append(picked)
        accepted += len(picked)
        attempts = (
            int(picked[-1]) - start_ordinal + 1
            if accepted == candidate_count
            else attempts + count
        )
    delta = np.concatenate(deltas)
    relative = _sample_poses(prepared._geometry, policy, np.concatenate(samples))
    uids, poses, _ = _numeric_geometry(prepared._geometry, policy, delta, relative)
    chosen = [i for i, uid in enumerate(uids) if uid in prepared.movable_uids]
    pool = _VariantPool(
        prepared.source.scene_id,
        seed,
        tuple(uids[i] for i in chosen),
        poses[:, chosen],
        np.concatenate(ordinals),
        (
            ("generator", _ALGORITHM),
            ("source_sha256", prepared.source.fingerprint),
            ("geometry_sha256", prepared.geometry_sha256),
        ),
    )
    return SceneRandomizationBatch(
        prepared,
        pool,
        attempts,
        "completed" if accepted == candidate_count else "budget_exhausted",
        seed,
        start_ordinal,
        start_ordinal + attempts,
        candidate_count,
        max_attempts,
    )


@dataclass(frozen=True)
class SceneVariantApplicationResult:
    """Selected-row pose publication and host-owned cache invalidations."""

    env_ids: tuple[int, ...]
    variant_ids: tuple[str, ...]
    changed_uids: tuple[str, ...]
    refresh_required: tuple[str, ...] = ("geometry", "planning", "observations")


def apply_scene_variants(
    sim: SimulationManager,
    variants: Sequence[SceneVariant],
    *,
    env_ids: Sequence[int] | torch.Tensor,
    movable_entity_ids: Sequence[str],
) -> SceneVariantApplicationResult:
    """Preflight all rows, then batch rigid pose writes without physics steps.

    Args:
        sim: Prepared caller-owned simulation.
        variants: One validated proposal per selected row, in corresponding order.
        env_ids: Unique selected rows; replica offsets never enter the poses.
        movable_entity_ids: Explicit allowed native UID set.

    Returns:
        Applied identities and refresh requests. Native failures propagate and
        do not guarantee rollback of earlier writes.
    """
    raw = (
        env_ids.detach().cpu().tolist()
        if isinstance(env_ids, torch.Tensor)
        else list(env_ids)
    )
    if any(type(i) is not int or not 0 <= i < sim.num_envs for i in raw) or len(
        set(raw)
    ) != len(raw):
        raise ValueError("env_ids must contain unique valid environment rows.")
    if len(raw) != len(variants):
        raise ValueError("One variant is required for each selected row.")
    permitted = set(movable_entity_ids)
    entities = set(sim.get_rigid_object_uid_list())
    writes = {}
    for row, variant in zip(raw, variants):
        if not isinstance(variant, SceneVariant):
            raise TypeError("variants must contain SceneVariant objects")
        for change in variant.pose_changes:
            uid = change.entity_uid
            if uid not in permitted or uid not in entities:
                raise ValueError(
                    f"Scene change requires a permitted existing rigid entity: {uid!r}"
                )
            entity = sim.get_rigid_object(uid)
            if not all(
                callable(getattr(entity, name, None))
                for name in ("set_local_pose", "clear_dynamics")
            ):
                raise ValueError(f"Entity {uid!r} does not support rigid pose writes.")
            writes.setdefault(uid, []).append((row, change.arena_pose))
    tensors = {
        uid: _backend_tensor([pose for _, pose in values], device=sim.device).squeeze(0)
        for uid, values in writes.items()
    }
    for uid, values in writes.items():
        rows = [row for row, _ in values]
        entity = sim.get_rigid_object(uid)
        entity.clear_dynamics(env_ids=rows)
        entity.set_local_pose(tensors[uid], env_ids=rows)
    if writes:
        sim.sync_render_state()
    return SceneVariantApplicationResult(
        tuple(raw), tuple(v.variant_id for v in variants), tuple(sorted(writes))
    )
