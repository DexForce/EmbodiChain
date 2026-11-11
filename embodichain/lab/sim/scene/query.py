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

"""Thin fixed-topology view over existing simulation entities."""

from __future__ import annotations
from collections.abc import Sequence
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager
__all__ = ["SceneView"]


class SceneView:
    """Select supported entities without creating another resource registry.

    Args:
        sim: Existing prepared simulation manager.
        env_ids: Nonempty unique arena indices, in the desired output order.
        entity_uids: Explicit nonempty group of existing entity UIDs, in member
            order. May include visual objects. None retains the whole physical
            scene selection. This view owns no entities or native resources.
    """

    def __init__(
        self,
        sim: SimulationManager,
        *,
        env_ids: Sequence[int] = (0,),
        entity_uids: Sequence[str] | None = None,
    ) -> None:
        self.sim = sim
        self.env_ids = tuple(env_ids)
        if isinstance(entity_uids, str):
            raise ValueError("entity_uids must be a nonempty sequence of UIDs")
        self.entity_uids = None if entity_uids is None else tuple(entity_uids)
        self._num_envs = getattr(sim, "num_envs", 1)
        self._validate_selection()
        self.topology = self.describe_entities()

    def _validate_selection(self) -> None:
        count = getattr(self.sim, "num_envs", 1)
        if (
            type(count) is not int
            or count < 1
            or not self.env_ids
            or any(type(i) is not int or not 0 <= i < count for i in self.env_ids)
            or len(set(self.env_ids)) != len(self.env_ids)
        ):
            raise ValueError("Scene env_ids must be nonempty, unique and in range")
        if self.entity_uids is not None and (
            not self.entity_uids
            or any(
                not isinstance(uid, str) or not uid or uid.strip() != uid
                for uid in self.entity_uids
            )
            or len(set(self.entity_uids)) != len(self.entity_uids)
        ):
            raise ValueError("entity_uids must be nonempty, unique, stripped UIDs")

    def entities(self) -> tuple[tuple[str, str, object], ...]:
        """Return selected physical bindings; visual members have no physical state.

        Whole-scene views retain kind/UID order. Explicit groups retain member
        order. Physical snapshots intentionally exclude pure visual members.
        """
        return tuple(row for row in self.members() if row[0] != "visual_object")

    def members(self) -> tuple[tuple[str, str, object], ...]:
        """Return current group bindings, including explicitly selected visuals.

        Returns:
            Tuples of entity kind, UID and the original manager-owned facade.
            Unknown or unsupported selected UIDs raise before state is read.
        """
        if self.entity_uids is None:
            for method in (
                "get_rigid_object_group_uid_list",
                "get_deformable_object_uid_list",
                "get_rigid_constraint_uid_list",
            ):
                getter = getattr(self.sim, method, None)
                if callable(getter) and getter():
                    raise ValueError(
                        f"scene snapshots do not support entities from {method}"
                    )
        entities = []
        kinds = ("rigid_object", "robot", "articulation")
        if self.entity_uids is not None:
            kinds += ("visual_object",)
        for kind in kinds:
            getter = getattr(self.sim, f"get_{kind}_uid_list", None)
            for uid in getter() if callable(getter) else ():
                if self.entity_uids is not None and uid not in self.entity_uids:
                    continue
                entity = getattr(self.sim, f"get_{kind}")(uid)
                if entity is None:
                    raise ValueError(f"Scene entity {uid!r} disappeared")
                entities.append((kind, uid, entity))
        if len({uid for _, uid, _ in entities}) != len(entities):
            raise ValueError("Entity UIDs must be unique across scene kinds")
        if self.entity_uids is not None:
            bindings = {row[1]: row for row in entities}
            missing = set(self.entity_uids) - bindings.keys()
            if missing:
                raise ValueError(
                    f"Unknown or unsupported scene group UIDs: {sorted(missing)}"
                )
            return tuple(bindings[uid] for uid in self.entity_uids)
        return tuple(sorted(entities, key=lambda value: (value[0], value[1])))

    def describe_entities(self) -> tuple[tuple[object, ...], ...]:
        """Return topology and binding identities without reading dynamic state."""
        return tuple(
            (
                kind,
                uid,
                getattr(entity, "body_type", ""),
                (
                    ()
                    if kind in ("rigid_object", "visual_object")
                    else tuple(entity.joint_names)
                ),
                id(entity),
            )
            for kind, uid, entity in self.members()
        )

    def check_topology(self) -> None:
        """Reject changed bindings before reads or writes."""
        self._validate_selection()
        if getattr(self.sim, "num_envs", 1) != self._num_envs:
            raise ValueError(
                "Scene environment count changed; reconstruct the scene host"
            )
        if self.describe_entities() != self.topology:
            raise ValueError(
                "Scene entity set or bindings changed; reconstruct the scene host"
            )

    def sync_render_state(self) -> None:
        """Publish direct state edits without stepping when the backend supports it."""
        sync = getattr(self.sim, "sync_render_state", None)
        if callable(sync):
            sync()

    def pose(self, uid: str, *, to_matrix: bool = True) -> torch.Tensor:
        """Read a cloned arena-local pose, retaining the environment dimension.

        Args:
            uid: Existing entity UID.
            to_matrix: Return (B,4,4), otherwise (B,7) with xyzw quaternion.
        Returns:
            Independent pose tensor in meters and Z-up.
        """
        self.check_topology()
        for _, entity_uid, entity in self.members():
            if entity_uid == uid:
                return entity.get_local_pose(to_matrix=to_matrix)[
                    list(self.env_ids)
                ].clone()
        raise KeyError(f"Unknown scene entity {uid!r}")


def _backend_tensor(values: object, *, device: torch.device | str) -> torch.Tensor:
    """Stage a finite backend-compatible tensor before any physical writes."""
    try:
        tensor = torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(0)
    except (ValueError, TypeError, RuntimeError, OverflowError) as exc:
        raise ValueError("Scene state cannot be represented by the backend") from exc
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("Scene state exceeds finite backend float32 range")
    return tensor
