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

"""Preflighted deterministic arena-local scene changes."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
import torch
from .query import SceneView, _backend_tensor
from ._validation import _rigid_pose

__all__ = [
    "SceneChangeSet",
    "SceneChangeResult",
    "apply_changes",
    "apply_changes_batch",
    "apply_group_transform",
]


@dataclass(frozen=True)
class SceneChangeSet:
    """Owned rigid/visual pose changes with an explicit movable-entity allowlist.

    Args:
        poses: UID to proper 4x4 entity-to-arena transforms, in meters and Z-up.
        movable_uids: Entities the calling policy permits to move.
    """

    poses: Mapping[str, tuple[tuple[float, ...], ...]]
    movable_uids: tuple[str, ...]

    def __post_init__(self) -> None:
        permissions = tuple(self.movable_uids)
        if any(
            type(uid) is not str or not uid or uid != uid.strip() for uid in permissions
        ) or len(set(permissions)) != len(permissions):
            raise ValueError("movable_uids must contain unique nonempty strings")
        owned = {}
        for uid, pose in self.poses.items():
            if uid not in permissions:
                raise ValueError(f"Scene change lacks movable permission for {uid!r}")
            owned[uid] = _rigid_pose(pose)
        object.__setattr__(self, "poses", MappingProxyType(owned))
        object.__setattr__(self, "movable_uids", permissions)


@dataclass(frozen=True)
class SceneChangeResult:
    """Changed entities and host-owned caches that need invalidation."""

    changed_uids: tuple[str, ...]
    refresh_required: tuple[str, ...] = ("geometry", "planning", "observations")


def _preflight_changes(
    view: SceneView, changes: SceneChangeSet
) -> dict[str, tuple[str, object, torch.Tensor]]:
    view.check_topology()
    if not isinstance(changes, SceneChangeSet):
        raise TypeError("changes must be a SceneChangeSet")
    entities = {uid: (kind, entity) for kind, uid, entity in view.members()}
    selected = {}
    for uid in changes.poses:
        pair = entities.get(uid)
        if pair is None or pair[0] not in ("rigid_object", "visual_object"):
            raise ValueError(
                f"Pose changes require an existing rigid object or visual object: {uid!r}"
            )
        kind, entity = pair
        methods = (
            ("clear_dynamics", "set_local_pose")
            if kind == "rigid_object"
            else ("set_local_pose",)
        )
        if any(not callable(getattr(entity, method, None)) for method in methods):
            raise ValueError(f"Entity {uid!r} does not support pose changes")
        selected[uid] = (
            kind,
            entity,
            _backend_tensor(changes.poses[uid], device=view.sim.device)
            .expand(len(view.env_ids), -1, -1)
            .contiguous(),
        )
    return selected


def apply_changes(view: SceneView, changes: SceneChangeSet) -> SceneChangeResult:
    """Apply allowed poses, clearing rigid dynamics without advancing physics.

    Args:
        view: Explicit scene selection; each selected row receives the declared poses.
        changes: Fully specified arena-local poses and movable permissions.
    Returns:
        Changed UIDs and required host cache invalidations; render state is synced.
    Raises:
        ValueError: For unsupported changes before any entity is modified.
        RuntimeError: For failed native writes; earlier writes remain applied.
    """
    entities = _preflight_changes(view, changes)
    try:
        for kind, entity, pose in entities.values():
            if kind == "rigid_object":
                entity.clear_dynamics(env_ids=list(view.env_ids))
            entity.set_local_pose(pose, env_ids=list(view.env_ids))
        if entities:
            view.sync_render_state()
    except Exception as exc:
        raise RuntimeError(
            "Scene change failed during native writes; earlier writes may remain applied"
        ) from exc
    return SceneChangeResult(tuple(changes.poses))


def apply_changes_batch(
    view: SceneView, changes: Mapping[int, SceneChangeSet]
) -> SceneChangeResult:
    """Apply distinct per-arena rigid/visual changes after preflighting all rows.

    Args:
        view: Ordered set of physical rows to prepare.
        changes: Exactly one change set for every selected environment index.
    Returns:
        Union of changed entities and required host cache invalidations.
    Raises:
        ValueError: For invalid selections or any invalid row, before writes.
        RuntimeError: For native failures; partial writes require host recovery.
    """
    view.check_topology()
    if any(type(index) is not int for index in changes) or set(changes) != set(
        view.env_ids
    ):
        raise ValueError("Changes must exactly match selected env_ids")
    grouped: dict[str, tuple[str, object, list[int], list[torch.Tensor]]] = {}
    for env_id in view.env_ids:
        row_view = SceneView(view.sim, env_ids=(env_id,), entity_uids=view.entity_uids)
        for uid, (kind, entity, pose) in _preflight_changes(
            row_view, changes[env_id]
        ).items():
            group = grouped.setdefault(uid, (kind, entity, [], []))
            group[2].append(env_id)
            group[3].append(pose)
    prepared = [
        (kind, entity, ids, torch.cat(poses, dim=0))
        for kind, entity, ids, poses in grouped.values()
    ]
    try:
        for kind, entity, ids, poses in prepared:
            if kind == "rigid_object":
                entity.clear_dynamics(env_ids=ids)
            entity.set_local_pose(poses, env_ids=ids)
        if prepared:
            view.sync_render_state()
    except Exception as exc:
        raise RuntimeError(
            "Batch scene change failed; world may be partially modified"
        ) from exc
    return SceneChangeResult(tuple(grouped))


def apply_group_transform(
    view: SceneView,
    transform: object,
    *,
    movable_uids: Sequence[str],
) -> SceneChangeResult:
    """Apply one arena-local rigid delta to every selected static/visual member.

    Each target pose is ``transform @ current_pose``. Relative member poses,
    scales and entity types are preserved. The same delta applies to every
    selected environment; arena replication offsets are not part of this frame.
    All targets and permissions are checked before native writes, then render
    state is published once without stepping. Native failures may leave partial
    writes, as for :func:`apply_changes_batch`.

    Args:
        view: Explicit ``entity_uids`` group, containing only static rigid bodies
            and pure visual objects. Dynamic, kinematic and articulated members
            require separate relocation policies and are rejected.
        transform: Finite proper 4x4 delta, as an array, tensor or nested sequence.
            To move a known group root from A to B, pass ``B @ inverse(A)``.
        movable_uids: Explicit permission to move every group member.

    Returns:
        Changed member UIDs and host-owned caches requiring refresh.

    Raises:
        ValueError: If the selection, delta, member type or permission is invalid.
        RuntimeError: If a native write or render publication fails.
    """
    view.check_topology()
    if view.entity_uids is None:
        raise ValueError("Group transforms require explicit entity_uids")
    members = view.members()
    if any(
        kind != "visual_object"
        and (kind != "rigid_object" or entity.body_type != "static")
        for kind, _, entity in members
    ):
        raise ValueError(
            "Group transforms require static rigid objects or visual objects"
        )
    if isinstance(transform, torch.Tensor):
        transform = transform.detach().cpu().tolist()
    delta = torch.tensor(_rigid_pose(transform), dtype=torch.float64)
    targets = {
        uid: delta @ view.pose(uid).detach().to(device="cpu", dtype=torch.float64)
        for _, uid, _ in members
    }
    permissions = tuple(movable_uids)
    changes = {
        env_id: SceneChangeSet(
            {uid: poses[row].tolist() for uid, poses in targets.items()}, permissions
        )
        for row, env_id in enumerate(view.env_ids)
    }
    return apply_changes_batch(view, changes)
