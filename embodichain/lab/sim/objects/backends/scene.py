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
"""EmbodiChain views over backend-neutral :mod:`dexsim.scene` batches.

Backend selection, handle rebinding, and topology revision tracking remain
owned by DexSim's ``Scene`` and batch classes. EmbodiChain adapts logical row
selections and its public pose convention ``(x, y, z, qx, qy, qz, qw)``;
backend-specific native compatibility work is delegated to narrow hooks.

Row and DOF selection is delegated to DexSim's public batches.
"""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING, Any, Sequence

import torch

from .base import ArticulationViewBase, RigidBodyViewBase

if TYPE_CHECKING:
    from dexsim.scene import (
        ArticulationBatch,
        RigidBodyBatch,
        Scene,
        SpawnedArticulation,
        SpawnedRigidBody,
    )

__all__ = ["SceneArticulationView", "SceneRigidBodyView"]

_NEWTON_ROOT_POSE_ATOL = 1.0e-6


def _checked_batch_call(
    batch: Any,
    method_name: str,
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Call one Scene batch operation and reject native failure statuses."""
    status = getattr(batch, method_name)(*args, **kwargs)
    if isinstance(status, Integral) and status < 0:
        raise RuntimeError(
            f"DexSim Scene batch operation {method_name!r} failed with "
            f"status {status}."
        )
    return status


def _rows(
    selection: Sequence[int] | torch.Tensor | None,
    count: int,
    device: torch.device,
) -> torch.Tensor:
    if selection is None:
        return torch.arange(count, dtype=torch.long, device=device)
    result = torch.as_tensor(selection, dtype=torch.long, device=device).reshape(-1)
    if torch.any(result < 0) or torch.any(result >= count):
        raise IndexError(f"Batch row selection is outside [0, {count}).")
    return result


def _batch_pose(data: torch.Tensor) -> torch.Tensor:
    """Convert rigid-body ``xyz+xyzw`` poses to batch ``xyzw+xyz``."""
    result = torch.empty_like(data, dtype=torch.float32)
    result[..., 0:4] = data[..., 3:7]
    result[..., 4:7] = data[..., 0:3]
    return result


def _embodichain_pose(data: torch.Tensor) -> torch.Tensor:
    """Convert batch ``xyzw+xyz`` poses to rigid-body ``xyz+xyzw``."""
    result = torch.empty_like(data, dtype=torch.float32)
    result[..., 0:3] = data[..., 4:7]
    result[..., 3:7] = data[..., 0:4]
    return result


class _SceneBatchSelectionAdapter:
    """Shared row-selection support for fixed-size Scene batches."""

    def __init__(self, batch: Any, device: torch.device, row_count: int) -> None:
        self.batch = batch
        self.device = device
        self._row_count = row_count

    def _fetch_rows(
        self,
        method_name: str,
        out: torch.Tensor,
        selection: Sequence[int] | torch.Tensor | None,
        tail_shape: tuple[int, ...],
    ) -> torch.Tensor:
        if selection is None:
            expected_shape = (self._row_count, *tail_shape)
            if tuple(out.shape) != expected_shape:
                raise ValueError(
                    f"Expected batch output shape {expected_shape}, got "
                    f"{tuple(out.shape)}."
                )
            _checked_batch_call(self.batch, method_name, out)
            return out

        rows = _rows(selection, self._row_count, self.device)
        expected_shape = (len(rows), *tail_shape)
        if tuple(out.shape) != expected_shape:
            raise ValueError(
                f"Expected selected output shape {expected_shape}, got "
                f"{tuple(out.shape)}."
            )
        if len(rows):
            _checked_batch_call(self.batch.select(rows), method_name, out)
        return out

    def _apply_rows(
        self,
        method_name: str,
        values: torch.Tensor,
        selection: Sequence[int] | torch.Tensor,
        tail_shape: tuple[int, ...],
    ) -> None:
        rows = _rows(selection, self._row_count, self.device)
        expected_shape = (len(rows), *tail_shape)
        if tuple(values.shape) != expected_shape:
            raise ValueError(
                f"Expected selected data shape {expected_shape}, got "
                f"{tuple(values.shape)}."
            )
        if len(rows):
            _checked_batch_call(self.batch.select(rows), method_name, values)


class SceneRigidBodyView(_SceneBatchSelectionAdapter, RigidBodyViewBase):
    """Backend-neutral rigid-body view backed by ``RigidBodyBatch``."""

    @classmethod
    def from_entities(
        cls,
        scene: Scene,
        entities: Sequence[SpawnedRigidBody],
        device: torch.device,
    ) -> SceneRigidBodyView:
        """Create a view from rigid-body handles owned by ``scene``."""
        return cls(
            scene,
            scene.create_rigid_body_batch(list(entities)),
            device,
        )

    def __init__(
        self,
        scene: Scene,
        batch: RigidBodyBatch,
        device: torch.device,
    ) -> None:
        super().__init__(batch, device, len(batch))
        self.scene = scene
        self._body_ids_tensor = torch.arange(
            len(batch), dtype=torch.int32, device=device
        )
        self._newton_state_sync: tuple[int, Any, Any] | None = None

    @property
    def is_ready(self) -> bool:
        return True

    @property
    def is_newton_backend(self) -> bool:
        return self.scene.backend == "newton"

    @property
    def body_ids(self) -> list[int]:
        return list(range(self._row_count))

    @property
    def body_ids_tensor(self) -> torch.Tensor:
        return self._body_ids_tensor

    def select_body_ids(self, indices: Sequence[int] | torch.Tensor) -> torch.Tensor:
        return self._body_ids_tensor[indices]

    def fetch_pose(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        batch_pose = torch.empty(
            (len(data), 7), dtype=torch.float32, device=self.device
        )
        self._fetch_rows("fetch_pose", batch_pose, body_ids, (7,))
        data.copy_(_embodichain_pose(batch_pose).to(data.device, data.dtype))

    def apply_pose(self, pose: torch.Tensor, body_ids: torch.Tensor) -> None:
        self._apply_rows(
            "apply_pose",
            _batch_pose(pose.to(self.device, torch.float32)),
            body_ids,
            (7,),
        )
        if self.is_newton_backend and len(body_ids):
            self._synchronize_newton_standalone_state()

    def _synchronize_newton_standalone_state(self) -> None:
        """Keep Newton standalone-body FREE joints coherent after state writes.

        DexSim 0.4.3's device ``RigidBodyBatch`` state writes update maximal
        ``body_q`` or ``body_qd`` state, while MuJoCo-Warp advances standalone
        rigid bodies from their reduced FREE-joint state. Cache one selection
        for this stable batch and project both state buffers after each write.
        """
        from .newton import _synchronize_standalone_rigid_body_state

        self._newton_state_sync = _synchronize_standalone_rigid_body_state(
            self.scene,
            self.batch,
            self._newton_state_sync,
        )

    def fetch_com_local_pose(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        batch_pose = torch.empty(
            (len(data), 7), dtype=torch.float32, device=self.device
        )
        self._fetch_rows("fetch_com_local_pose", batch_pose, body_ids, (7,))
        data.copy_(_embodichain_pose(batch_pose).to(data.device, data.dtype))

    def apply_com_local_pose(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        self._apply_rows(
            "apply_com_local_pose",
            _batch_pose(data.to(self.device, torch.float32)),
            body_ids,
            (7,),
        )

    def fetch_linear_velocity(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        self._fetch_rows("fetch_linear_velocity", data, body_ids, (3,))

    def fetch_angular_velocity(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        self._fetch_rows("fetch_angular_velocity", data, body_ids, (3,))

    def apply_linear_velocity(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        self._apply_rows(
            "apply_linear_velocity",
            data,
            body_ids,
            (3,),
        )
        if self.is_newton_backend and len(body_ids):
            self._synchronize_newton_standalone_state()

    def apply_angular_velocity(
        self, data: torch.Tensor, body_ids: torch.Tensor
    ) -> None:
        self._apply_rows(
            "apply_angular_velocity",
            data,
            body_ids,
            (3,),
        )
        if self.is_newton_backend and len(body_ids):
            self._synchronize_newton_standalone_state()

    def fetch_linear_acceleration(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        self._fetch_rows("fetch_linear_acceleration", data, body_ids, (3,))

    def fetch_angular_acceleration(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        self._fetch_rows("fetch_angular_acceleration", data, body_ids, (3,))

    def apply_force(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        self._apply_rows("apply_force", data, body_ids, (3,))

    def apply_torque(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        self._apply_rows("apply_torque", data, body_ids, (3,))

    def fetch_mass(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        self._fetch_rows("fetch_mass", data, body_ids, (1,))

    def apply_mass(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        self._apply_rows("apply_mass", data, body_ids, (1,))

    def fetch_inertia_diagonal(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        self._fetch_rows("fetch_inertia_diagonal", data, body_ids, (3,))

    def apply_inertia_diagonal(
        self, data: torch.Tensor, body_ids: torch.Tensor
    ) -> None:
        self._apply_rows(
            "apply_inertia_diagonal",
            data,
            body_ids,
            (3,),
        )

    def fetch_friction(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        self._fetch_rows("fetch_friction", data, body_ids, (1,))

    def apply_friction(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        self._apply_rows("apply_friction", data, body_ids, (1,))

    def fetch_restitution(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        self._fetch_rows("fetch_restitution", data, body_ids, (1,))

    def apply_restitution(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        self._apply_rows("apply_restitution", data, body_ids, (1,))

    def fetch_contact_offset(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        self._fetch_rows("fetch_contact_offset", data, body_ids, (1,))

    def apply_contact_offset(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        self._apply_rows("apply_contact_offset", data, body_ids, (1,))

    def fetch_damping(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        self._fetch_rows("fetch_damping", data, body_ids, (2,))

    def apply_damping(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        self._apply_rows("apply_damping", data, body_ids, (2,))

    def fetch_collision_filter(
        self,
        data: torch.Tensor,
        body_ids: torch.Tensor | None = None,
    ) -> None:
        self._fetch_rows("fetch_collision_filter", data, body_ids, (4,))

    def apply_collision_filter(
        self,
        data: torch.Tensor,
        body_ids: torch.Tensor,
    ) -> None:
        self._apply_rows("apply_collision_filter", data, body_ids, (4,))


class SceneArticulationView(_SceneBatchSelectionAdapter, ArticulationViewBase):
    """Backend-neutral articulation state view backed by ``ArticulationBatch``.

    Joint selections currently require one scalar DOF per selected joint. The
    public DexSim layout already describes multi-DOF joints; supporting them
    without ambiguity requires a DOF-selection API in DexSim and is therefore
    kept as an explicit boundary rather than guessed here.
    """

    @classmethod
    def from_entities(
        cls,
        scene: Scene,
        entities: Sequence[SpawnedArticulation],
        device: torch.device,
    ) -> SceneArticulationView:
        """Create a view from articulation handles owned by ``scene``."""
        return cls(
            scene,
            scene.create_articulation_batch(list(entities)),
            device,
        )

    def __init__(
        self,
        scene: Scene,
        batch: ArticulationBatch,
        device: torch.device,
    ) -> None:
        super().__init__(batch, device, len(batch))
        self.scene = scene
        self._validate_homogeneous_layout()
        self._articulation_ids = torch.arange(
            len(batch), dtype=torch.int32, device=device
        )

    def _validate_homogeneous_layout(self) -> None:
        """Require the uniform topology promised by one EC Articulation."""
        dof_counts = tuple(self.batch.dof_counts)
        link_counts = tuple(self.batch.link_counts)
        joint_names = tuple(self.batch.joint_names_per_articulation)
        link_names = tuple(self.batch.link_names_per_articulation)
        if dof_counts and len(set(dof_counts)) != 1:
            raise ValueError(
                "One EmbodiChain Articulation cannot bind heterogeneous Scene "
                f"DOF counts: {dof_counts}."
            )
        if link_counts and len(set(link_counts)) != 1:
            raise ValueError(
                "One EmbodiChain Articulation cannot bind heterogeneous Scene "
                f"link counts: {link_counts}."
            )
        if joint_names and any(names != joint_names[0] for names in joint_names[1:]):
            raise ValueError(
                "One EmbodiChain Articulation requires identical active-joint "
                "ordering in every Scene row."
            )
        if link_names and any(names != link_names[0] for names in link_names[1:]):
            raise ValueError(
                "One EmbodiChain Articulation requires identical link ordering "
                "in every Scene row."
            )
        layouts = tuple(self.batch.joint_layouts_per_articulation)
        if layouts and any(layout.dof_count != 1 for layout in layouts[0]):
            raise NotImplementedError(
                "EmbodiChain's Articulation API currently indexes joints and "
                "scalar DOFs interchangeably. Scene multi-DOF joints require "
                "an explicit DOF-selection API before they can be bound safely."
            )

    @property
    def dof(self) -> int:
        """Scalar DOF width shared by every articulation row."""
        return self.batch.dof_width

    @property
    def num_links(self) -> int:
        """Link count shared by every articulation row."""
        return self.batch.link_width

    @property
    def joint_names(self) -> list[str]:
        """Active joints in public flattened-DOF order."""
        rows = self.batch.joint_names_per_articulation
        return [] if not rows else list(rows[0])

    @property
    def link_names(self) -> list[str]:
        """Links in public link-buffer order."""
        rows = self.batch.link_names_per_articulation
        return [] if not rows else list(rows[0])

    @property
    def is_ready(self) -> bool:
        return True

    @property
    def is_newton_backend(self) -> bool:
        return self.scene.backend == "newton"

    @property
    def articulation_ids_tensor(self) -> torch.Tensor:
        return self._articulation_ids

    def select_articulation_ids(
        self, env_ids: Sequence[int] | torch.Tensor
    ) -> torch.Tensor:
        return self._articulation_ids[env_ids]

    def fetch_root_pose(self, data: torch.Tensor) -> torch.Tensor:
        batch_pose = torch.empty_like(data, dtype=torch.float32, device=self.device)
        _checked_batch_call(self.batch, "fetch_root_pose", batch_pose)
        data.copy_(_embodichain_pose(batch_pose).to(data.device, data.dtype))
        return data

    def fetch_root_linear_velocity(self, data: torch.Tensor) -> torch.Tensor:
        _checked_batch_call(self.batch, "fetch_root_linear_velocity", data)
        return data

    def fetch_root_angular_velocity(self, data: torch.Tensor) -> torch.Tensor:
        _checked_batch_call(self.batch, "fetch_root_angular_velocity", data)
        return data

    def fetch_qpos(self, data: torch.Tensor) -> torch.Tensor:
        _checked_batch_call(self.batch, "fetch_joint_position", data)
        return data

    def fetch_target_qpos(self, data: torch.Tensor) -> torch.Tensor:
        _checked_batch_call(self.batch, "fetch_joint_target_position", data)
        return data

    def fetch_qvel(self, data: torch.Tensor) -> torch.Tensor:
        _checked_batch_call(self.batch, "fetch_joint_velocity", data)
        return data

    def fetch_target_qvel(self, data: torch.Tensor) -> torch.Tensor:
        _checked_batch_call(self.batch, "fetch_joint_target_velocity", data)
        return data

    def fetch_qacc(self, data: torch.Tensor) -> torch.Tensor:
        _checked_batch_call(self.batch, "fetch_joint_acceleration", data)
        return data

    def fetch_qf(self, data: torch.Tensor) -> torch.Tensor:
        _checked_batch_call(self.batch, "fetch_joint_force", data)
        return data

    def fetch_link_pose(self, data: torch.Tensor) -> torch.Tensor:
        batch_pose = torch.empty_like(data, dtype=torch.float32, device=self.device)
        _checked_batch_call(self.batch, "fetch_link_pose", batch_pose)
        data.copy_(_embodichain_pose(batch_pose).to(data.device, data.dtype))
        return data

    def fetch_link_velocity(
        self,
        data: torch.Tensor,
        linear_data: torch.Tensor,
        angular_data: torch.Tensor,
    ) -> torch.Tensor:
        _checked_batch_call(self.batch, "fetch_link_linear_velocity", linear_data)
        _checked_batch_call(self.batch, "fetch_link_angular_velocity", angular_data)
        data[..., 0:3] = linear_data
        data[..., 3:6] = angular_data
        return data

    def apply_root_pose(
        self, pose: torch.Tensor, env_ids: Sequence[int] | torch.Tensor
    ) -> None:
        rows = _rows(env_ids, self._row_count, self.device)
        batch_pose = _batch_pose(pose.to(self.device, torch.float32))
        expected_shape = (len(rows), 7)
        if tuple(batch_pose.shape) != expected_shape:
            raise ValueError(
                f"Expected selected data shape {expected_shape}, got "
                f"{tuple(batch_pose.shape)}."
            )
        if self.is_newton_backend and len(rows):
            current_pose = torch.empty_like(batch_pose)
            self._fetch_rows(
                "fetch_root_pose",
                current_pose,
                rows,
                (7,),
            )
            translation_matches = torch.all(
                torch.abs(current_pose[:, 4:7] - batch_pose[:, 4:7])
                <= _NEWTON_ROOT_POSE_ATOL,
                dim=1,
            )
            quaternion_delta = torch.minimum(
                torch.amax(torch.abs(current_pose[:, 0:4] - batch_pose[:, 0:4]), dim=1),
                torch.amax(torch.abs(current_pose[:, 0:4] + batch_pose[:, 0:4]), dim=1),
            )
            changed = ~(
                translation_matches & (quaternion_delta <= _NEWTON_ROOT_POSE_ATOL)
            )
            rows = rows[changed]
            batch_pose = batch_pose[changed]

        self._apply_rows(
            "apply_root_pose",
            batch_pose,
            rows,
            (7,),
        )

    def _joint_columns(self, joint_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
        layouts = self.batch.joint_layouts_per_articulation
        if not layouts:
            return _rows(joint_ids, self.dof, self.device)
        reference = layouts[0]
        ids = _rows(joint_ids, len(reference), self.device)
        columns: list[int] = []
        for joint_id in ids.detach().cpu().tolist():
            layout = reference[joint_id]
            if layout.dof_count != 1:
                raise NotImplementedError(
                    "SceneArticulationView needs DexSim DOF selection for "
                    f"multi-DOF joint {layout.name!r}."
                )
            columns.append(layout.dof_start)
        return torch.as_tensor(columns, dtype=torch.long, device=self.device)

    def _apply_joint_selection(
        self,
        values: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor,
        joint_ids: Sequence[int] | torch.Tensor,
        *,
        apply_method: str,
    ) -> None:
        rows = _rows(env_ids, self._row_count, self.device)
        columns = self._joint_columns(joint_ids)
        values = values.to(device=self.device, dtype=torch.float32)
        expected = (len(rows), len(columns))
        if tuple(values.shape) != expected:
            raise ValueError(
                f"Expected selected joint data shape {expected}, got "
                f"{tuple(values.shape)}."
            )
        if len(rows) and len(columns):
            _checked_batch_call(
                self.batch.select(rows),
                apply_method,
                values,
                dof_ids=columns,
            )

    def apply_qpos(
        self,
        qpos: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor,
        joint_ids: Sequence[int] | torch.Tensor,
        *,
        target: bool,
    ) -> None:
        self._apply_joint_selection(
            qpos,
            env_ids,
            joint_ids,
            apply_method=(
                "apply_joint_target_position" if target else "apply_joint_position"
            ),
        )

    def apply_qvel(
        self,
        qvel: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor,
        joint_ids: Sequence[int] | torch.Tensor,
        *,
        target: bool,
    ) -> None:
        self._apply_joint_selection(
            qvel,
            env_ids,
            joint_ids,
            apply_method=(
                "apply_joint_target_velocity" if target else "apply_joint_velocity"
            ),
        )

    def apply_qf(
        self,
        qf: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor,
        joint_ids: Sequence[int] | torch.Tensor,
    ) -> None:
        self._apply_joint_selection(
            qf,
            env_ids,
            joint_ids,
            apply_method="apply_joint_force",
        )

    def clear_dynamics(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        rows = _rows(env_ids, self._row_count, self.device)
        if not len(rows):
            return
        zeros = torch.zeros(
            (len(rows), self.batch.dof_width),
            dtype=torch.float32,
            device=self.device,
        )
        selected = self.batch.select(rows)
        _checked_batch_call(selected, "apply_joint_velocity", zeros)
        _checked_batch_call(selected, "apply_joint_target_velocity", zeros)
        _checked_batch_call(selected, "apply_joint_force", zeros)

    def compute_kinematics(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        rows = _rows(env_ids, self._row_count, self.device)
        if not len(rows):
            return
        _checked_batch_call(self.batch.select(rows), "compute_kinematics")
