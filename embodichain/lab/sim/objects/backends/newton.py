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
import numpy as np
import torch

from dexsim.models import MeshObject
from dexsim.engine.newton_physics import NewtonPhysicsScene
from embodichain.lab.sim.objects.backends.base import (
    ArticulationViewBase,
    RigidBodyViewBase,
)
from embodichain.utils import logger
from embodichain.utils.math import matrix_from_quat, quat_from_matrix

__all__ = [
    "NewtonRigidBodyView",
    "NewtonArticulationView",
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


def _resolve_body_ids_and_filter_rows_for_entities(
    manager: object,
    entities: Sequence[MeshObject],
    filter_data: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    body_ids: list[int] = []
    rows: list[torch.Tensor] = []
    for i, entity in enumerate(entities):
        entity_handle = _normalize_native_handle(
            entity.get_native_handle(), "MeshObject"
        )
        body_id = manager.body_id_for_entity(entity_handle)
        if body_id is None:
            entity.set_collision_filter_data(
                filter_data[i].detach().cpu().numpy().astype(np.int64)
            )
            continue
        body_ids.append(int(body_id))
        rows.append(filter_data[i])

    if len(rows) == 0:
        empty_rows = filter_data.new_empty((0, filter_data.shape[-1]))
        return torch.as_tensor(body_ids, dtype=torch.int32), empty_rows

    return torch.as_tensor(body_ids, dtype=torch.int32), torch.stack(rows, dim=0)


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
    body_ids, valid_rows = _resolve_body_ids_and_filter_rows_for_entities(
        scene.manager, entities, rows
    )
    if len(body_ids) == 0:
        return
    body_ids = body_ids.to(device=rows.device)
    scene.apply_collision_filter(body_ids, valid_rows.to(device=rows.device))


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

    def fetch_contact_offset(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        self._fetch_scalar(self._get_data_type().CONTACT_OFFSET, data, body_ids)

    def apply_contact_offset(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        self._apply_data(body_ids, self._get_data_type().CONTACT_OFFSET, data)

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


class NewtonArticulationView(ArticulationViewBase):
    """Adapter around DexSim Newton articulation scene APIs."""

    _DATA_TYPE = None

    def __init__(
        self,
        entities: Sequence[object],
        scene: NewtonPhysicsScene,
        device: torch.device,
    ) -> None:
        self.entities = list(entities)
        self.scene = scene
        self.device = device
        self.dof = self.entities[0].get_dof()
        self.num_links = self.entities[0].get_links_num()
        self.link_names = self.entities[0].get_link_names()
        self._articulation_ids = torch.as_tensor(
            [entity.get_gpu_index() for entity in self.entities],
            dtype=torch.int32,
            device=self.device,
        )
        self._link_body_ids: torch.Tensor | None = None
        self._link_body_ids_finalized = False

    @classmethod
    def _get_data_type(cls):
        if cls._DATA_TYPE is None:
            from dexsim.engine.newton_physics import NewtonArticulationDataType

            cls._DATA_TYPE = NewtonArticulationDataType
        return cls._DATA_TYPE

    @property
    def is_ready(self) -> bool:
        manager = getattr(self.scene, "manager", None)
        return (
            manager is not None
            and getattr(getattr(manager, "lifecycle_state", None), "name", "")
            == "READY"
        )

    @property
    def is_newton_backend(self) -> bool:
        return True

    @property
    def articulation_ids_tensor(self) -> torch.Tensor:
        return self._articulation_ids

    def select_articulation_ids(
        self, env_ids: Sequence[int] | torch.Tensor
    ) -> torch.Tensor:
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        return self._articulation_ids[env_ids.to(device=self.device, dtype=torch.long)]

    def link_body_ids_for(
        self, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        if self._link_body_ids_finalized is False:
            rows = []
            for entity in self.entities:
                row = []
                for link_name in self.link_names:
                    local_link_name = self.entity_link_name(entity, link_name)
                    link_meta = entity.dexsim_meta_links["links"][local_link_name]
                    body_id = (
                        -1 if link_meta.body_id is None else int(link_meta.body_id)
                    )
                    if body_id < 0 or body_id > _INT32_MAX:
                        logger.log_error(
                            f"Newton articulation link '{link_name}' has no valid body id."
                        )
                    row.append(body_id)
                rows.append(row)
            self._link_body_ids = torch.as_tensor(
                rows, dtype=torch.int32, device=self.device
            )
            if self.is_ready:
                self._link_body_ids_finalized = True

        assert self._link_body_ids is not None
        if env_ids is None:
            return self._link_body_ids.reshape(-1)
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        return self._link_body_ids[
            env_ids.to(device=self.device, dtype=torch.long)
        ].reshape(-1)

    def entity_link_name(self, entity: object, link_name: str) -> str:
        if link_name in getattr(entity, "dexsim_meta_links", {}).get("links", {}):
            return link_name
        link_idx = self.link_names.index(link_name)
        return entity.get_link_names()[link_idx]

    def fetch_root_pose(self, data: torch.Tensor) -> torch.Tensor:
        if self.is_ready:
            self._fetch(data, self._get_data_type().ROOT_GLOBAL_POSE)
            return data.clone()

        root_pose = torch.as_tensor(
            np.array([entity.get_local_pose() for entity in self.entities]),
            dtype=torch.float32,
            device=self.device,
        )
        xyzs = root_pose[:, :3, 3]
        quats = quat_from_matrix(root_pose[:, :3, :3])
        return torch.cat((xyzs, quats), dim=-1)

    def fetch_root_linear_velocity(self, data: torch.Tensor) -> torch.Tensor:
        if self.is_ready:
            self._fetch(data, self._get_data_type().ROOT_LINEAR_VELOCITY)
            return data.clone()
        return torch.as_tensor(
            np.array(
                [
                    entity.get_link_general_velocities(entity.get_root_link_name())[
                        0, :3
                    ]
                    for entity in self.entities
                ]
            ),
            dtype=torch.float32,
            device=self.device,
        )

    def fetch_root_angular_velocity(self, data: torch.Tensor) -> torch.Tensor:
        if self.is_ready:
            self._fetch(data, self._get_data_type().ROOT_ANGULAR_VELOCITY)
            return data.clone()
        return torch.as_tensor(
            np.array(
                [
                    entity.get_link_general_velocities(entity.get_root_link_name())[
                        0, 3:
                    ]
                    for entity in self.entities
                ]
            ),
            dtype=torch.float32,
            device=self.device,
        )

    def fetch_qpos(self, data: torch.Tensor) -> torch.Tensor:
        return self._fetch_joint_or_entity(
            data, self._get_data_type().JOINT_POSITION, "get_current_qpos"
        )

    def fetch_target_qpos(self, data: torch.Tensor) -> torch.Tensor:
        return self._fetch_joint_or_entity(
            data, self._get_data_type().JOINT_TARGET_POSITION, "get_target_qpos"
        )

    def fetch_qvel(self, data: torch.Tensor) -> torch.Tensor:
        return self._fetch_joint_or_entity(
            data, self._get_data_type().JOINT_VELOCITY, "get_current_qvel"
        )

    def fetch_target_qvel(self, data: torch.Tensor) -> torch.Tensor:
        return self._fetch_joint_or_entity(
            data, self._get_data_type().JOINT_TARGET_VELOCITY, "get_target_qvel"
        )

    def fetch_qacc(self, data: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (len(self.entities), self.dof), dtype=torch.float32, device=self.device
        )

    def fetch_qf(self, data: torch.Tensor) -> torch.Tensor:
        return self._fetch_joint_or_entity(
            data, self._get_data_type().JOINT_FORCE, "get_current_qf"
        )

    def fetch_link_pose(self, data: torch.Tensor) -> torch.Tensor:
        if self.is_ready:
            flat_pose = data[:, : self.num_links, :].reshape(-1, 7)
            self.scene.batch_fetch_articulation_data(
                flat_pose,
                self.link_body_ids_for(),
                self._get_data_type().LINK_GLOBAL_POSE,
            )
            return data[:, : self.num_links, :].clone()

        from embodichain.lab.sim.utility import get_dexsim_arenas

        arenas = get_dexsim_arenas()
        for j, entity in enumerate(self.entities):
            link_pose = np.zeros((self.num_links, 4, 4), dtype=np.float32)
            for i, link_name in enumerate(self.link_names):
                pose = entity.get_link_pose(self.entity_link_name(entity, link_name))
                arena_pose = arenas[j].get_root_node().get_local_pose()
                pose[:2, 3] -= arena_pose[:2, 3]
                link_pose[i] = pose

            link_pose_tensor = torch.from_numpy(link_pose)
            xyz = link_pose_tensor[:, :3, 3]
            quat = quat_from_matrix(link_pose_tensor[:, :3, :3])
            data[j][: self.num_links, :] = torch.cat((xyz, quat), dim=-1)
        return data[:, : self.num_links, :]

    def fetch_link_velocity(
        self,
        data: torch.Tensor,
        linear_data: torch.Tensor,
        angular_data: torch.Tensor,
    ) -> torch.Tensor:
        if self.is_ready:
            flat_lin = linear_data[:, : self.num_links, :].reshape(-1, 3)
            flat_ang = angular_data[:, : self.num_links, :].reshape(-1, 3)
            link_ids = self.link_body_ids_for()
            self.scene.batch_fetch_articulation_data(
                flat_lin, link_ids, self._get_data_type().LINK_LINEAR_VELOCITY
            )
            self.scene.batch_fetch_articulation_data(
                flat_ang, link_ids, self._get_data_type().LINK_ANGULAR_VELOCITY
            )
            data[..., :3] = linear_data
            data[..., 3:] = angular_data
            return data[:, : self.num_links, :].clone()

        for i, entity in enumerate(self.entities):
            data[i][: self.num_links] = torch.from_numpy(
                entity.get_link_general_velocities()
            )
        return data[:, : self.num_links, :]

    def apply_root_pose(
        self, pose: torch.Tensor, env_ids: Sequence[int] | torch.Tensor
    ) -> None:
        pose_cpu = pose.to(dtype=torch.float32).cpu()
        env_indices = self._env_indices_list(env_ids)
        pose_matrix = torch.eye(4).unsqueeze(0).repeat(len(env_indices), 1, 1)
        pose_matrix[:, :3, 3] = pose_cpu[:, :3]
        pose_matrix[:, :3, :3] = matrix_from_quat(pose_cpu[:, 3:7])
        for i, env_idx in enumerate(env_indices):
            self.entities[env_idx].set_local_pose(pose_matrix[i])

    def apply_qpos(
        self,
        qpos: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor,
        joint_ids: Sequence[int] | torch.Tensor,
        *,
        target: bool,
    ) -> None:
        if self.is_ready:
            data_type = (
                self._get_data_type().JOINT_TARGET_POSITION
                if target
                else self._get_data_type().JOINT_POSITION
            )
            self._apply(qpos, data_type, env_ids, joint_ids)
            return

        joint_ids_np = self._joint_ids_numpy(joint_ids)
        qpos_np = qpos.detach().cpu().numpy()
        for i, env_idx in enumerate(self._env_indices_list(env_ids)):
            setter = (
                self.entities[env_idx].set_target_qpos
                if target
                else self.entities[env_idx].set_current_qpos
            )
            setter(qpos_np[i], joint_ids_np)

    def apply_qvel(
        self,
        qvel: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor,
        joint_ids: Sequence[int] | torch.Tensor,
        *,
        target: bool,
    ) -> None:
        if self.is_ready:
            data_type = (
                self._get_data_type().JOINT_TARGET_VELOCITY
                if target
                else self._get_data_type().JOINT_VELOCITY
            )
            self._apply(qvel, data_type, env_ids, joint_ids)
            return

        joint_ids_np = self._joint_ids_numpy(joint_ids)
        qvel_np = qvel.detach().cpu().numpy()
        for i, env_idx in enumerate(self._env_indices_list(env_ids)):
            setter = (
                self.entities[env_idx].set_target_qvel
                if target
                else self.entities[env_idx].set_current_qvel
            )
            setter(qvel_np[i], joint_ids_np)

    def apply_qf(
        self,
        qf: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor,
        joint_ids: Sequence[int] | torch.Tensor,
    ) -> None:
        if self.is_ready:
            self._apply(qf, self._get_data_type().JOINT_FORCE, env_ids, joint_ids)
            return

        joint_ids_np = self._joint_ids_numpy(joint_ids)
        qf_np = qf.detach().cpu().numpy()
        for i, env_idx in enumerate(self._env_indices_list(env_ids)):
            self.entities[env_idx].set_current_qf(qf_np[i], joint_ids_np)

    def clear_dynamics(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        zeros = torch.zeros(
            (len(env_ids), self.dof), dtype=torch.float32, device=self.device
        )
        joint_ids = torch.arange(self.dof, dtype=torch.int32, device=self.device)
        self.apply_qvel(zeros, env_ids, joint_ids, target=False)
        self.apply_qvel(zeros, env_ids, joint_ids, target=True)
        self.apply_qf(zeros, env_ids, joint_ids)

    def compute_kinematics(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        return

    def _fetch(self, data: torch.Tensor, data_type, joint_ids=None) -> None:
        self.scene.batch_fetch_articulation_data(
            data.contiguous(),
            self._articulation_ids,
            data_type,
            self._joint_ids_numpy(joint_ids) if joint_ids is not None else None,
        )

    def _apply(
        self,
        data: torch.Tensor,
        data_type,
        env_ids: Sequence[int] | torch.Tensor,
        joint_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        self.scene.batch_apply_articulation_data(
            data.to(dtype=torch.float32).contiguous(),
            self.select_articulation_ids(env_ids),
            data_type,
            self._joint_ids_numpy(joint_ids) if joint_ids is not None else None,
        )

    def _fetch_joint_or_entity(
        self, data: torch.Tensor, data_type, entity_method: str
    ) -> torch.Tensor:
        if self.is_ready:
            self._fetch(data, data_type)
            return data[:, : self.dof].clone()
        return torch.as_tensor(
            np.array([getattr(entity, entity_method)() for entity in self.entities]),
            dtype=torch.float32,
            device=self.device,
        )

    def _joint_ids_numpy(
        self, joint_ids: Sequence[int] | torch.Tensor | None
    ) -> np.ndarray | None:
        if joint_ids is None:
            return None
        if isinstance(joint_ids, torch.Tensor):
            return joint_ids.detach().cpu().numpy().astype(np.int32, copy=False)
        return np.asarray(joint_ids, dtype=np.int32)

    def _env_indices_list(self, env_ids: Sequence[int] | torch.Tensor) -> list[int]:
        if isinstance(env_ids, torch.Tensor):
            return env_ids.detach().cpu().to(dtype=torch.long).tolist()
        return [int(env_idx) for env_idx in env_ids]
