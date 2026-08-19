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
"""EmbodiChain tensor-layout adapters for :mod:`dexsim.spawn` batches.

The classes in this module deliberately know nothing about PhysX scenes or
Newton runtime objects. Backend selection, handle rebinding, and topology
revision tracking remain owned by DexSim's ``SpawnResult`` and batch classes.
EmbodiChain only adapts logical row selections and its public pose convention
``(x, y, z, qx, qy, qz, qw)``.

DexSim does not yet expose lightweight row/DOF/link selections on its public
batches. Until that API lands, partial writes use a correctness-first
read/modify/write fallback. The fallback is kept here, at the boundary, so it
can be deleted without changing object or environment APIs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import torch

from .base import ArticulationViewBase, RigidBodyViewBase

if TYPE_CHECKING:
    from dexsim.spawn import ArticulationBatch, RigidBodyBatch, SpawnResult

__all__ = ["SpawnArticulationView", "SpawnRigidBodyView"]


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


def _spawn_pose(data: torch.Tensor) -> torch.Tensor:
    """Convert rigid-body ``xyz+xyzw`` poses to Spawn ``xyzw+xyz``."""
    result = torch.empty_like(data, dtype=torch.float32)
    result[..., 0:4] = data[..., 3:7]
    result[..., 4:7] = data[..., 0:3]
    return result


def _embodichain_pose(data: torch.Tensor) -> torch.Tensor:
    """Convert Spawn ``xyzw+xyz`` poses to rigid-body ``xyz+xyzw``."""
    result = torch.empty_like(data, dtype=torch.float32)
    result[..., 0:3] = data[..., 4:7]
    result[..., 3:7] = data[..., 0:4]
    return result


def _spawn_articulation_pose(data: torch.Tensor) -> torch.Tensor:
    """Convert articulation ``xyz+wxyz`` poses to Spawn ``xyzw+xyz``."""
    result = torch.empty_like(data, dtype=torch.float32)
    result[..., 0:3] = data[..., 4:7]
    result[..., 3] = data[..., 3]
    result[..., 4:7] = data[..., 0:3]
    return result


def _embodichain_articulation_pose(data: torch.Tensor) -> torch.Tensor:
    """Convert Spawn ``xyzw+xyz`` poses to articulation ``xyz+wxyz``."""
    result = torch.empty_like(data, dtype=torch.float32)
    result[..., 0:3] = data[..., 4:7]
    result[..., 3] = data[..., 3]
    result[..., 4:7] = data[..., 0:3]
    return result


class _SpawnSelectionAdapter:
    """Shared correctness-first selection support for fixed-size Spawn batches."""

    def __init__(self, batch: Any, device: torch.device, row_count: int) -> None:
        self._batch = batch
        self.device = device
        self._row_count = row_count

    def _fetch_rows(
        self,
        method_name: str,
        out: torch.Tensor,
        selection: Sequence[int] | torch.Tensor | None,
        tail_shape: tuple[int, ...],
    ) -> torch.Tensor:
        rows = _rows(selection, self._row_count, self.device)
        full = torch.empty(
            (self._row_count, *tail_shape),
            dtype=torch.float32,
            device=self.device,
        )
        getattr(self._batch, method_name)(full)
        selected = full.index_select(0, rows)
        out.copy_(selected.to(device=out.device, dtype=out.dtype))
        return out

    def _apply_rows(
        self,
        method_name: str,
        values: torch.Tensor,
        selection: Sequence[int] | torch.Tensor,
        tail_shape: tuple[int, ...],
        *,
        fetch_method_name: str | None,
    ) -> None:
        rows = _rows(selection, self._row_count, self.device)
        values = values.to(device=self.device, dtype=torch.float32)
        expected_shape = (len(rows), *tail_shape)
        if tuple(values.shape) != expected_shape:
            raise ValueError(
                f"Expected selected data shape {expected_shape}, got "
                f"{tuple(values.shape)}."
            )

        if fetch_method_name is None:
            full = torch.zeros(
                (self._row_count, *tail_shape),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            full = torch.empty(
                (self._row_count, *tail_shape),
                dtype=torch.float32,
                device=self.device,
            )
            getattr(self._batch, fetch_method_name)(full)
        full.index_copy_(0, rows, values)
        getattr(self._batch, method_name)(full)


class SpawnRigidBodyView(_SpawnSelectionAdapter, RigidBodyViewBase):
    """Backend-neutral rigid-body view backed by ``RigidBodyBatch``."""

    def __init__(
        self,
        result: SpawnResult,
        batch: RigidBodyBatch,
        device: torch.device,
    ) -> None:
        super().__init__(batch, device, len(batch))
        self.result = result
        self.batch = batch
        self._body_ids_tensor = torch.arange(
            len(batch), dtype=torch.int32, device=device
        )

    @property
    def is_ready(self) -> bool:
        return True

    @property
    def is_newton_backend(self) -> bool:
        return self.result.backend == "newton"

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
        spawn = torch.empty((len(data), 7), dtype=torch.float32, device=self.device)
        self._fetch_rows("fetch_pose", spawn, body_ids, (7,))
        data.copy_(_embodichain_pose(spawn).to(data.device, data.dtype))

    def apply_pose(self, pose: torch.Tensor, body_ids: torch.Tensor) -> None:
        self._apply_rows(
            "apply_pose",
            _spawn_pose(pose.to(self.device, torch.float32)),
            body_ids,
            (7,),
            fetch_method_name="fetch_pose",
        )

    def fetch_com_local_pose(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        spawn = torch.empty((len(data), 7), dtype=torch.float32, device=self.device)
        self._fetch_rows("fetch_com_local_pose", spawn, body_ids, (7,))
        data.copy_(_embodichain_pose(spawn).to(data.device, data.dtype))

    def apply_com_local_pose(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        self._apply_rows(
            "apply_com_local_pose",
            _spawn_pose(data.to(self.device, torch.float32)),
            body_ids,
            (7,),
            fetch_method_name="fetch_com_local_pose",
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
            fetch_method_name="fetch_linear_velocity",
        )

    def apply_angular_velocity(
        self, data: torch.Tensor, body_ids: torch.Tensor
    ) -> None:
        self._apply_rows(
            "apply_angular_velocity",
            data,
            body_ids,
            (3,),
            fetch_method_name="fetch_angular_velocity",
        )

    def fetch_linear_acceleration(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        self._fetch_rows("fetch_linear_acceleration", data, body_ids, (3,))

    def fetch_angular_acceleration(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        self._fetch_rows("fetch_angular_acceleration", data, body_ids, (3,))

    def apply_force(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        self._apply_rows("apply_force", data, body_ids, (3,), fetch_method_name=None)

    def apply_torque(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        self._apply_rows("apply_torque", data, body_ids, (3,), fetch_method_name=None)

    def fetch_mass(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        self._fetch_rows("fetch_mass", data, body_ids, (1,))

    def apply_mass(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        self._apply_rows(
            "apply_mass", data, body_ids, (1,), fetch_method_name="fetch_mass"
        )

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
            fetch_method_name="fetch_inertia_diagonal",
        )

    @staticmethod
    def _unsupported_property(name: str) -> None:
        raise NotImplementedError(
            f"DexSim Spawn RigidBodyBatch does not expose the {name} property yet. "
            "Extend the public Spawn batch instead of accessing backend internals."
        )

    def fetch_friction(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        del data, body_ids
        self._unsupported_property("friction")

    def apply_friction(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        del data, body_ids
        self._unsupported_property("friction")

    def fetch_restitution(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        del data, body_ids
        self._unsupported_property("restitution")

    def apply_restitution(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        del data, body_ids
        self._unsupported_property("restitution")

    def fetch_contact_offset(
        self, data: torch.Tensor, body_ids: torch.Tensor | None = None
    ) -> None:
        del data, body_ids
        self._unsupported_property("contact_offset")

    def apply_contact_offset(self, data: torch.Tensor, body_ids: torch.Tensor) -> None:
        del data, body_ids
        self._unsupported_property("contact_offset")


class SpawnArticulationView(_SpawnSelectionAdapter, ArticulationViewBase):
    """Backend-neutral articulation state view backed by ``ArticulationBatch``.

    Joint selections currently require one scalar DOF per selected joint. The
    public DexSim layout already describes multi-DOF joints; supporting them
    without ambiguity requires a DOF-selection API in DexSim and is therefore
    kept as an explicit boundary rather than guessed here.
    """

    def __init__(
        self,
        result: SpawnResult,
        batch: ArticulationBatch,
        device: torch.device,
    ) -> None:
        super().__init__(batch, device, len(batch))
        self.result = result
        self.batch = batch
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
                "One EmbodiChain Articulation cannot bind heterogeneous Spawn "
                f"DOF counts: {dof_counts}."
            )
        if link_counts and len(set(link_counts)) != 1:
            raise ValueError(
                "One EmbodiChain Articulation cannot bind heterogeneous Spawn "
                f"link counts: {link_counts}."
            )
        if joint_names and any(names != joint_names[0] for names in joint_names[1:]):
            raise ValueError(
                "One EmbodiChain Articulation requires identical active-joint "
                "ordering in every Spawn row."
            )
        if link_names and any(names != link_names[0] for names in link_names[1:]):
            raise ValueError(
                "One EmbodiChain Articulation requires identical link ordering "
                "in every Spawn row."
            )
        layouts = tuple(self.batch.joint_layouts_per_articulation)
        if layouts and any(layout.dof_count != 1 for layout in layouts[0]):
            raise NotImplementedError(
                "EmbodiChain's Articulation API currently indexes joints and "
                "scalar DOFs interchangeably. Spawn multi-DOF joints require "
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
        return self.result.backend == "newton"

    @property
    def articulation_ids_tensor(self) -> torch.Tensor:
        return self._articulation_ids

    def select_articulation_ids(
        self, env_ids: Sequence[int] | torch.Tensor
    ) -> torch.Tensor:
        return self._articulation_ids[env_ids]

    def fetch_root_pose(self, data: torch.Tensor) -> torch.Tensor:
        spawn = torch.empty_like(data, dtype=torch.float32, device=self.device)
        self.batch.fetch_root_pose(spawn)
        data.copy_(_embodichain_articulation_pose(spawn).to(data.device, data.dtype))
        return data

    def fetch_root_linear_velocity(self, data: torch.Tensor) -> torch.Tensor:
        self.batch.fetch_root_linear_velocity(data)
        return data

    def fetch_root_angular_velocity(self, data: torch.Tensor) -> torch.Tensor:
        self.batch.fetch_root_angular_velocity(data)
        return data

    def fetch_qpos(self, data: torch.Tensor) -> torch.Tensor:
        self.batch.fetch_joint_position(data)
        return data

    def fetch_target_qpos(self, data: torch.Tensor) -> torch.Tensor:
        self.batch.fetch_joint_target_position(data)
        return data

    def fetch_qvel(self, data: torch.Tensor) -> torch.Tensor:
        self.batch.fetch_joint_velocity(data)
        return data

    def fetch_target_qvel(self, data: torch.Tensor) -> torch.Tensor:
        self.batch.fetch_joint_target_velocity(data)
        return data

    def fetch_qacc(self, data: torch.Tensor) -> torch.Tensor:
        self.batch.fetch_joint_acceleration(data)
        return data

    def fetch_qf(self, data: torch.Tensor) -> torch.Tensor:
        self.batch.fetch_joint_force(data)
        return data

    def fetch_link_pose(self, data: torch.Tensor) -> torch.Tensor:
        spawn = torch.empty_like(data, dtype=torch.float32, device=self.device)
        self.batch.fetch_link_pose(spawn)
        data.copy_(_embodichain_articulation_pose(spawn).to(data.device, data.dtype))
        return data

    def fetch_link_velocity(
        self,
        data: torch.Tensor,
        linear_data: torch.Tensor,
        angular_data: torch.Tensor,
    ) -> torch.Tensor:
        self.batch.fetch_link_linear_velocity(linear_data)
        self.batch.fetch_link_angular_velocity(angular_data)
        data[..., 0:3] = linear_data
        data[..., 3:6] = angular_data
        return data

    def apply_root_pose(
        self, pose: torch.Tensor, env_ids: Sequence[int] | torch.Tensor
    ) -> None:
        self._apply_rows(
            "apply_root_pose",
            _spawn_articulation_pose(pose.to(self.device, torch.float32)),
            env_ids,
            (7,),
            fetch_method_name="fetch_root_pose",
        )

    def _joint_columns(self, joint_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
        ids = torch.as_tensor(joint_ids, dtype=torch.long, device=self.device)
        layouts = self.batch.joint_layouts_per_articulation
        if not layouts:
            return ids
        reference = layouts[0]
        columns: list[int] = []
        for joint_id in ids.detach().cpu().tolist():
            layout = reference[joint_id]
            if layout.dof_count != 1:
                raise NotImplementedError(
                    "SpawnArticulationView needs DexSim DOF selection for "
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
        fetch_method: str | None,
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
        width = self.batch.dof_width
        if fetch_method is None:
            full = torch.zeros(
                (self._row_count, width),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            full = torch.empty(
                (self._row_count, width),
                dtype=torch.float32,
                device=self.device,
            )
            getattr(self.batch, fetch_method)(full)
        full[rows[:, None], columns] = values
        getattr(self.batch, apply_method)(full)

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
            fetch_method=(
                "fetch_joint_target_position" if target else "fetch_joint_position"
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
            fetch_method=(
                "fetch_joint_target_velocity" if target else "fetch_joint_velocity"
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
            fetch_method=None,
        )

    def clear_dynamics(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        rows = _rows(env_ids, self._row_count, self.device)
        zeros = torch.zeros(
            (len(rows), self.batch.dof_width),
            dtype=torch.float32,
            device=self.device,
        )
        self._apply_rows(
            "apply_joint_velocity",
            zeros,
            rows,
            (self.batch.dof_width,),
            fetch_method_name="fetch_joint_velocity",
        )
        self._apply_rows(
            "apply_joint_target_velocity",
            zeros,
            rows,
            (self.batch.dof_width,),
            fetch_method_name="fetch_joint_target_velocity",
        )
        self._apply_rows(
            "apply_joint_force",
            zeros,
            rows,
            (self.batch.dof_width,),
            fetch_method_name=None,
        )

    def compute_kinematics(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        # DexSim currently refreshes the complete batch. Since this operation
        # only propagates already-authored state, that is equivalent to a row
        # selection and keeps selection details out of EmbodiChain.
        del env_ids
        if self.batch.compute_kinematics() < 0:
            raise RuntimeError("DexSim Spawn articulation kinematics update failed.")
