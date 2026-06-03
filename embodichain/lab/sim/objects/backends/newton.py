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
from __future__ import annotations

from typing import Sequence
import torch

from dexsim.models import MeshObject
from dexsim.engine.newton_physics import NewtonPhysicsScene
from embodichain.lab.sim.objects.backends.base import RigidBodyViewBase
from embodichain.utils import logger

__all__ = [
    "NewtonRigidBodyView",
    "apply_collision_filter_for_entities",
    "apply_collision_filter_for_envs",
    "is_newton_scene",
]

_UINT64_MAX = (1 << 64) - 1
_INT32_MAX = (1 << 31) - 1


def _normalize_native_handle(handle: int, owner: str) -> int:
    value = int(handle)
    if value < 0:
        value &= _UINT64_MAX
    if value > _UINT64_MAX:
        logger.log_error(f"{owner} native handle is outside uint64 range: {value}.")
    return value


def _collision_filter_rows(filter_data: torch.Tensor) -> torch.Tensor:
    """Return contiguous ``(N, 4)`` int32 rows for the Newton scene API."""
    rows = filter_data.to(dtype=torch.int32)
    if rows.ndim != 2 or rows.shape[-1] != 4:
        logger.log_error(
            "Collision filter data must have shape (N, 4), " f"got {tuple(rows.shape)}."
        )
    if not rows.is_contiguous():
        rows = rows.contiguous()
    return rows


def _resolve_body_ids_for_entities(
    manager: object,
    entities: Sequence[MeshObject],
) -> torch.Tensor:
    body_ids: list[int] = []
    for entity in entities:
        entity_handle = _normalize_native_handle(
            entity.get_native_handle(), "MeshObject"
        )
        body_id = manager.body_id_for_entity(entity_handle)
        if body_id is None:
            logger.log_error(
                "Newton collision filter batch apply found an entity without a body id."
            )
        body_ids.append(int(body_id))
    return torch.as_tensor(body_ids, dtype=torch.int32)


def apply_collision_filter_for_entities(
    scene: NewtonPhysicsScene,
    entities: Sequence[MeshObject],
    filter_data: torch.Tensor,
) -> None:
    """Batch-apply collision filters for a list of MeshObjects.

    Uses DexSim ``NewtonPhysicsScene.apply_collision_filter`` (vectorized meta
    and shape-group writes on the DexSim side).
    """
    if len(entities) == 0:
        return
    if len(entities) != len(filter_data):
        logger.log_error(
            "Entity count does not match collision filter row count "
            f"({len(entities)} vs {len(filter_data)})."
        )

    rows = _collision_filter_rows(filter_data)
    body_ids = _resolve_body_ids_for_entities(scene.manager, entities)
    body_ids = body_ids.to(device=rows.device)
    scene.apply_collision_filter(body_ids, rows)


def apply_collision_filter_for_envs(
    scene: NewtonPhysicsScene,
    entities_by_env: Sequence[Sequence[MeshObject]],
    filter_data: torch.Tensor,
    env_indices: Sequence[int],
) -> None:
    """Batch-apply collision filters with one filter row per environment.

    Expands each env row to every ``MeshObject`` in that env (e.g. rigid groups).
    """
    entities: list[MeshObject] = []
    rows: list[torch.Tensor] = []
    for i, env_idx in enumerate(env_indices):
        row = filter_data[i]
        for entity in entities_by_env[env_idx]:
            entities.append(entity)
            rows.append(row)
    if not entities:
        return
    stacked = torch.stack(rows, dim=0)
    apply_collision_filter_for_entities(scene, entities, stacked)


def is_newton_scene(scene: object) -> bool:
    """Return whether *scene* looks like a DexSim Newton scene view."""
    return (
        scene is not None
        and hasattr(scene, "manager")
        and hasattr(scene, "batch_fetch_rigid_body_data")
        and hasattr(scene, "batch_apply_rigid_body_data")
        and hasattr(scene, "apply_collision_filter")
        and hasattr(scene, "fetch_collision_filter")
    )


class NewtonRigidBodyView(RigidBodyViewBase):
    """Adapter around DexSim Newton rigid body scene APIs.

    EmbodiChain public rigid-body pose convention is
    ``(x, y, z, qx, qy, qz, qw)``.
    DexSim Newton exposes the same pose convention through its unified rigid
    data API.
    """

    _DATA_TYPE = None  # lazily resolved NewtonRigidDataType

    def __init__(
        self,
        entities: Sequence[MeshObject],
        scene: NewtonPhysicsScene,
        device: torch.device,
    ) -> None:
        self.entities = list(entities)
        self.scene = scene
        self.device = device
        self.entity_handles = [
            _normalize_native_handle(entity.get_native_handle(), "MeshObject")
            for entity in self.entities
        ]
        # Body IDs are resolved lazily because Newton's model is not built
        # until finalization.  Pre-finalization, ``body_id_for_entity()``
        # returns tentative IDs that may differ from the final interleaved
        # layout.  We track whether IDs have been resolved in the READY
        # state and re-resolve once when the manager transitions.
        self._body_ids: list[int] | None = None
        self._body_ids_tensor: torch.Tensor | None = None
        self._body_ids_finalized: bool = False

    # -- Lazy enum access ---------------------------------------------------

    @classmethod
    def _get_data_type(cls):
        """Lazily resolve *NewtonRigidDataType* to avoid eager import."""
        if cls._DATA_TYPE is None:
            from dexsim.engine.newton_physics import NewtonRigidDataType

            cls._DATA_TYPE = NewtonRigidDataType
        return cls._DATA_TYPE

    # -- RigidBodyViewBase: lifecycle ----------------------------------------

    @property
    def is_ready(self) -> bool:
        manager = getattr(self.scene, "manager", None)
        return (
            manager is not None
            and getattr(getattr(manager, "lifecycle_state", None), "name", "")
            == "READY"
        )

    @property
    def _lifecycle_state_name(self) -> str:
        manager = getattr(self.scene, "manager", None)
        return getattr(getattr(manager, "lifecycle_state", None), "name", "")

    @property
    def can_apply_pose(self) -> bool:
        return self._lifecycle_state_name in ("BUILDER", "READY")

    @property
    def can_fetch_pose(self) -> bool:
        return self._lifecycle_state_name in ("BUILDER", "READY")

    # -- RigidBodyViewBase: body IDs -----------------------------------------

    def _ensure_body_ids(self) -> None:
        """Resolve body IDs from the Newton manager.

        Body IDs resolved before finalization may be tentative.  Once the
        manager transitions to READY, re-resolve to get the correct
        interleaved layout.
        """
        if self._body_ids_finalized:
            return
        if self._body_ids is not None and not self.is_ready:
            return
        ids = [self._resolve_body_id(entity) for entity in self.entities]
        if any(bid < 0 or bid > _INT32_MAX for bid in ids):
            logger.log_error(
                "Newton rigid body view found an entity without a Newton body id."
            )
        self._body_ids = ids
        self._body_ids_tensor = torch.as_tensor(
            ids, dtype=torch.int32, device=self.device
        )
        if self.is_ready:
            self._body_ids_finalized = True

    @property
    def body_ids(self) -> list[int]:
        self._ensure_body_ids()
        return self._body_ids  # type: ignore[return-value]

    @property
    def body_ids_tensor(self) -> torch.Tensor:
        self._ensure_body_ids()
        return self._body_ids_tensor  # type: ignore[return-value]

    def select_body_ids(self, indices: Sequence[int] | torch.Tensor) -> torch.Tensor:
        self._ensure_body_ids()
        if not isinstance(indices, torch.Tensor):
            indices = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        return self._body_ids_tensor[indices.to(device=self.device, dtype=torch.long)]

    # -- RigidBodyViewBase: pose ---------------------------------------------

    def fetch_pose(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        self.scene.batch_fetch_rigid_body_data(
            self._fetch_buffer(data),
            self._resolve_body_ids(body_ids),
            self._get_data_type().POSE,
        )

    def apply_pose(self, pose: torch.Tensor, body_ids: torch.Tensor) -> None:
        self._apply_data(body_ids, self._get_data_type().POSE, pose)

    # -- RigidBodyViewBase: center of mass (local) ---------------------------

    def fetch_com_local_pose(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        data_type = getattr(self._get_data_type(), "COM_LOCAL_POSE", None)
        self.scene.batch_fetch_rigid_body_data(
            self._fetch_buffer(data), self._resolve_body_ids(body_ids), data_type
        )

    def apply_com_local_pose(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        data_type = getattr(self._get_data_type(), "COM_LOCAL_POSE", None)
        self._apply_data(body_ids, data_type, data)

    # -- RigidBodyViewBase: velocity -----------------------------------------

    def fetch_linear_velocity(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        self._fetch_vec3(self._get_data_type().LINEAR_VELOCITY, data, body_ids)

    def fetch_angular_velocity(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        self._fetch_vec3(self._get_data_type().ANGULAR_VELOCITY, data, body_ids)

    def apply_linear_velocity(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        self._apply_data(body_ids, self._get_data_type().LINEAR_VELOCITY, data)

    def apply_angular_velocity(
        self, data: torch.Tensor, body_ids: torch.Tensor
    ) -> None:
        self._apply_data(body_ids, self._get_data_type().ANGULAR_VELOCITY, data)

    # -- RigidBodyViewBase: acceleration -------------------------------------

    def fetch_linear_acceleration(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        self._fetch_vec3(self._get_data_type().LINEAR_ACCELERATION, data, body_ids)

    def fetch_angular_acceleration(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        self._fetch_vec3(self._get_data_type().ANGULAR_ACCELERATION, data, body_ids)

    # -- RigidBodyViewBase: force & torque -----------------------------------

    def apply_force(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        self._apply_data(body_ids, self._get_data_type().FORCE, data)

    def apply_torque(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        self._apply_data(body_ids, self._get_data_type().TORQUE, data)

    # -- RigidBodyViewBase: physical properties ------------------------------

    def fetch_mass(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        self._fetch_scalar(self._get_data_type().MASS, data, body_ids)

    def apply_mass(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        self._apply_data(body_ids, self._get_data_type().MASS, data)

    def fetch_inertia_diagonal(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        self._fetch_vec3(self._get_data_type().INERTIA_DIAGONAL, data, body_ids)

    def apply_inertia_diagonal(
        self, data: torch.Tensor, body_ids: torch.Tensor
    ) -> None:
        self._apply_data(body_ids, self._get_data_type().INERTIA_DIAGONAL, data)

    def fetch_friction(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        self._fetch_scalar(self._get_data_type().FRICTION, data, body_ids)

    def apply_friction(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        self._apply_data(body_ids, self._get_data_type().FRICTION, data)

    def fetch_restitution(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        self._fetch_scalar(self._get_data_type().RESTITUTION, data, body_ids)

    def apply_restitution(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        self._apply_data(body_ids, self._get_data_type().RESTITUTION, data)

    # -- Collision filter ----------------------------------------------------

    def fetch_collision_filter(
        self,
        data: torch.Tensor,
        env_indices: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """Fetch collision filter rows into ``data`` with shape ``(N, 4)``."""
        if env_indices is None:
            env_indices = torch.arange(len(self.entities), device=self.device)
        body_ids = self._resolve_body_ids(self.select_body_ids(env_indices))
        out = self._fetch_buffer(data)
        self.scene.fetch_collision_filter(body_ids, out)

    def apply_collision_filter(
        self,
        filter_data: torch.Tensor,
        env_indices: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """Apply DexSim collision filter rows for selected env instances."""
        if env_indices is None:
            env_indices = torch.arange(len(self.entities), device=self.device)
        body_ids = self._resolve_body_ids(self.select_body_ids(env_indices))
        rows = _collision_filter_rows(filter_data.to(device=self.device))
        self.scene.apply_collision_filter(body_ids, rows)

    # -- Internal helpers ----------------------------------------------------

    def _resolve_body_id(self, entity: MeshObject) -> int:
        manager = getattr(self.scene, "manager", None)
        if manager is not None and hasattr(entity, "get_native_handle"):
            entity_handle = _normalize_native_handle(
                entity.get_native_handle(), "MeshObject"
            )
            body_id = manager.body_id_for_entity(entity_handle)
            if body_id is not None:
                return int(body_id)

        if hasattr(entity, "get_gpu_index"):
            body_id = int(entity.get_gpu_index())
            if 0 <= body_id <= _INT32_MAX:
                return body_id
        return -1

    def _resolve_body_ids(self, body_ids: torch.Tensor | None) -> torch.Tensor:
        """Return body IDs as a device int32 tensor for the Newton scene API.

        DexSim's batch API normalizes GPU-resident tensors without a host
        round-trip, so the cached ``body_ids_tensor`` is passed straight
        through.  This avoids a per-call ``cuda -> cpu`` synchronization on the
        per-step fetch/apply hot path.
        """
        if body_ids is None:
            self._ensure_body_ids()
            return self._body_ids_tensor  # type: ignore[return-value]
        if not isinstance(body_ids, torch.Tensor):
            body_ids = torch.as_tensor(body_ids, dtype=torch.int32, device=self.device)
        return body_ids

    def _fetch_buffer(self, data: torch.Tensor) -> torch.Tensor:
        """Validate and forward a caller-owned fetch buffer to the scene API."""
        if not data.is_contiguous():
            logger.log_error("Newton rigid body fetch buffers must be contiguous.")
        return data

    def _fetch_vec3(
        self,
        data_type,
        data: torch.Tensor,
        body_ids: torch.Tensor | None = None,
    ) -> None:
        self.scene.batch_fetch_rigid_body_data(
            self._fetch_buffer(data), self._resolve_body_ids(body_ids), data_type
        )

    # Scalar ``(N, 1)`` fields share the same fetch path as vec3 fields.
    _fetch_scalar = _fetch_vec3

    def _apply_data(
        self, body_ids: torch.Tensor, data_type, data: torch.Tensor
    ) -> None:
        """Apply data to bodies via the unified Newton GPU API."""
        self.scene.batch_apply_rigid_body_data(
            data.to(dtype=torch.float32).contiguous(),
            self._resolve_body_ids(body_ids),
            data_type,
        )
