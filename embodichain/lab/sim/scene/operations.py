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
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
import torch
from .query import SceneView, _backend_tensor
from ._validation import _rigid_pose

__all__ = ["SceneChangeSet", "SceneChangeResult", "apply_changes"]


@dataclass(frozen=True)
class SceneChangeSet:
    """Owned rigid-body pose changes with an explicit movable-entity allowlist.

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
) -> dict[str, tuple[object, torch.Tensor]]:
    view.check_topology()
    if not isinstance(changes, SceneChangeSet):
        raise TypeError("changes must be a SceneChangeSet")
    entities = {uid: (kind, entity) for kind, uid, entity in view.entities()}
    selected = {}
    for uid in changes.poses:
        pair = entities.get(uid)
        if pair is None or pair[0] != "rigid_object":
            raise ValueError(f"Pose changes require an existing rigid object: {uid!r}")
        entity = pair[1]
        if any(
            not callable(getattr(entity, method, None))
            for method in ("clear_dynamics", "set_local_pose")
        ):
            raise ValueError(f"Entity {uid!r} does not support pose changes")
        selected[uid] = (
            entity,
            _backend_tensor(changes.poses[uid], device=view.sim.device),
        )
    return selected


def apply_changes(view: SceneView, changes: SceneChangeSet) -> SceneChangeResult:
    """Apply allowed poses and clear dynamics, without advancing physics.

    Args:
        view: Fixed-topology B=1 scene selection.
        changes: Fully specified arena-local poses and movable permissions.
    Returns:
        Changed UIDs and required host cache invalidations; render state is synced.
    Raises:
        ValueError: For unsupported changes before any entity is modified.
        RuntimeError: For failed native writes; earlier writes remain applied.
    """
    entities = _preflight_changes(view, changes)
    try:
        for entity, pose in entities.values():
            entity.clear_dynamics(env_ids=list(view.env_ids))
            entity.set_local_pose(pose, env_ids=list(view.env_ids))
        if entities:
            view.sync_render_state()
    except Exception as exc:
        raise RuntimeError(
            "Scene change failed during native writes; earlier writes may remain applied"
        ) from exc
    return SceneChangeResult(tuple(changes.poses))
