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

from types import SimpleNamespace

import pytest
import torch

import embodichain.lab.sim.objects.backends.spawn as spawn_backend
from embodichain.lab.sim.objects.backends.spawn import (
    SpawnArticulationView,
    SpawnRigidBodyView,
    _embodichain_articulation_pose,
    _embodichain_pose,
    _spawn_articulation_pose,
    _spawn_pose,
)

pytestmark = pytest.mark.no_sim


def test_spawn_pose_adapters_preserve_embodichain_xyzw_order() -> None:
    pose = torch.tensor([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9]])
    expected_spawn = torch.tensor([[0.1, 0.2, 0.3, 0.9, 1.0, 2.0, 3.0]])

    torch.testing.assert_close(_spawn_pose(pose), expected_spawn)
    torch.testing.assert_close(_spawn_articulation_pose(pose), expected_spawn)
    torch.testing.assert_close(_embodichain_pose(expected_spawn), pose)
    torch.testing.assert_close(_embodichain_articulation_pose(expected_spawn), pose)


class _SelectedRigidBatch:
    def __init__(self, owner: _RigidBatch, rows: torch.Tensor) -> None:
        self.owner = owner
        self.rows = rows

    def apply_force(self, values: torch.Tensor) -> int:
        self.owner.force[self.rows] = values
        return len(self.rows)

    def apply_pose(self, values: torch.Tensor) -> int:
        self.owner.pose[self.rows] = values
        return len(self.rows)

    def apply_linear_velocity(self, values: torch.Tensor) -> int:
        self.owner.linear_velocity[self.rows] = values
        return len(self.rows)

    def apply_angular_velocity(self, values: torch.Tensor) -> int:
        self.owner.angular_velocity[self.rows] = values
        return len(self.rows)

    def apply_friction(self, values: torch.Tensor) -> int:
        self.owner.friction[self.rows] = values
        return len(self.rows)

    def fetch_friction(self, out: torch.Tensor) -> int:
        out.copy_(self.owner.friction[self.rows])
        return len(self.rows)


class _RigidBatch:
    def __init__(self) -> None:
        self.force = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        self.pose = torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0],
            ]
        )
        self.friction = torch.tensor([[0.1], [0.2], [0.3]])
        self.linear_velocity = torch.zeros((3, 3))
        self.angular_velocity = torch.zeros((3, 3))
        self.selections: list[tuple[int, ...]] = []

    def __len__(self) -> int:
        return len(self.force)

    def select(self, rows: torch.Tensor) -> _SelectedRigidBatch:
        selected = rows.detach().cpu().to(dtype=torch.long)
        self.selections.append(tuple(selected.tolist()))
        return _SelectedRigidBatch(self, selected)


class _SelectedArticulationBatch:
    def __init__(self, owner: _ArticulationBatch, rows: torch.Tensor) -> None:
        self.owner = owner
        self.rows = rows

    def apply_joint_force(
        self,
        values: torch.Tensor,
        *,
        dof_ids: torch.Tensor,
    ) -> int:
        columns = dof_ids.detach().cpu().to(dtype=torch.long)
        self.owner.force[self.rows[:, None], columns] = values
        self.owner.last_dof_ids = tuple(columns.tolist())
        return len(self.rows)

    def fetch_root_pose(self, out: torch.Tensor) -> int:
        self.owner.root_pose_fetch_rows.append(tuple(self.rows.tolist()))
        out.copy_(self.owner.root_pose[self.rows])
        return len(self.rows)

    def apply_root_pose(self, values: torch.Tensor) -> int:
        self.owner.root_pose_apply_rows.append(tuple(self.rows.tolist()))
        self.owner.root_pose[self.rows] = values
        return len(self.rows)


class _ArticulationBatch:
    def __init__(self) -> None:
        layouts = tuple(
            SimpleNamespace(name=f"joint_{index}", dof_start=index, dof_count=1)
            for index in range(3)
        )
        self.dof_counts = (3, 3)
        self.link_counts = (1, 1)
        self.joint_names_per_articulation = (("joint_0", "joint_1", "joint_2"),) * 2
        self.link_names_per_articulation = (("root",),) * 2
        self.joint_layouts_per_articulation = (layouts,) * 2
        self.dof_width = 3
        self.link_width = 1
        self.force = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        # Spawn articulation poses use xyzw + xyz layout.
        self.root_pose = torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 1.0],
            ]
        )
        self.last_dof_ids: tuple[int, ...] | None = None
        self.root_pose_fetch_rows: list[tuple[int, ...]] = []
        self.root_pose_apply_rows: list[tuple[int, ...]] = []
        self.selections: list[tuple[int, ...]] = []

    def __len__(self) -> int:
        return len(self.force)

    def select(self, rows: torch.Tensor) -> _SelectedArticulationBatch:
        selected = rows.detach().cpu().to(dtype=torch.long)
        self.selections.append(tuple(selected.tolist()))
        return _SelectedArticulationBatch(self, selected)


def test_rigid_partial_writes_delegate_to_selected_batch() -> None:
    batch = _RigidBatch()
    view = SpawnRigidBodyView(
        SimpleNamespace(backend="newton"),
        batch,
        torch.device("cpu"),
    )

    view.apply_force(torch.tensor([[10.0, 20.0, 30.0]]), torch.tensor([1]))
    view.apply_friction(torch.tensor([[0.9]]), torch.tensor([2]))

    assert torch.equal(
        batch.force,
        torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [7.0, 8.0, 9.0]]),
    )
    assert torch.equal(batch.friction, torch.tensor([[0.1], [0.2], [0.9]]))
    assert batch.selections == [(1,), (2,)]


def test_rigid_partial_fetch_reads_only_selected_batch() -> None:
    batch = _RigidBatch()
    view = SpawnRigidBodyView(
        SimpleNamespace(backend="dexsim"),
        batch,
        torch.device("cpu"),
    )
    out = torch.empty((2, 1))

    view.fetch_friction(out, torch.tensor([2, 0]))

    assert torch.equal(out, torch.tensor([[0.3], [0.1]]))
    assert batch.selections == [(2, 0)]


def test_rigid_batch_failure_status_is_not_silently_ignored() -> None:
    batch = _RigidBatch()
    view = SpawnRigidBodyView(
        SimpleNamespace(backend="dexsim"),
        batch,
        torch.device("cpu"),
    )
    selected = batch.select(torch.tensor([0]))
    selected.fetch_friction = lambda _out: -2
    batch.select = lambda _rows: selected

    with pytest.raises(RuntimeError, match="fetch_friction.*status -2"):
        view.fetch_friction(torch.empty((1, 1)), torch.tensor([0]))


def test_newton_rigid_pose_write_synchronizes_free_joint_state(monkeypatch) -> None:
    batch = _RigidBatch()
    current_state = object()
    other_state = object()
    runtime = SimpleNamespace(
        model=object(),
        current_state=current_state,
        other_state=other_state,
    )
    batch._binding = SimpleNamespace(
        _runtime=runtime,
        _indices=torch.tensor([10, 11, 12]),
    )
    synchronized_states: list[tuple[object, object]] = []
    created_body_ids: list[tuple[int, ...]] = []

    class _StateSync:
        def synchronize(self, states: tuple[object, object]) -> None:
            synchronized_states.append(states)

    def _create_state_sync(_model: object, body_ids: list[int]) -> _StateSync:
        created_body_ids.append(tuple(body_ids))
        return _StateSync()

    monkeypatch.setattr(
        spawn_backend,
        "_create_newton_standalone_state_sync",
        _create_state_sync,
    )
    view = SpawnRigidBodyView(
        SimpleNamespace(backend="newton", topology_revision=3),
        batch,
        torch.device("cpu"),
    )

    view.apply_pose(
        torch.tensor([[4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0]]),
        torch.tensor([1]),
    )
    view.apply_pose(
        torch.tensor([[7.0, 8.0, 9.0, 0.0, 0.0, 0.0, 1.0]]),
        torch.tensor([2]),
    )

    assert created_body_ids == [(10, 11, 12)]
    assert synchronized_states == [
        (current_state, other_state),
        (current_state, other_state),
    ]
    assert torch.equal(
        batch.pose[1:],
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0, 4.0, 5.0, 6.0],
                [0.0, 0.0, 0.0, 1.0, 7.0, 8.0, 9.0],
            ]
        ),
    )


def test_newton_rigid_velocity_writes_synchronize_free_joint_state(
    monkeypatch,
) -> None:
    batch = _RigidBatch()
    current_state = object()
    other_state = object()
    runtime = SimpleNamespace(
        model=object(),
        current_state=current_state,
        other_state=other_state,
    )
    batch._binding = SimpleNamespace(
        _runtime=runtime,
        _indices=torch.tensor([10, 11, 12]),
    )
    synchronized_states: list[tuple[object, object]] = []
    created_body_ids: list[tuple[int, ...]] = []

    class _StateSync:
        def synchronize(self, states: tuple[object, object]) -> None:
            synchronized_states.append(states)

    def _create_state_sync(_model: object, body_ids: list[int]) -> _StateSync:
        created_body_ids.append(tuple(body_ids))
        return _StateSync()

    monkeypatch.setattr(
        spawn_backend,
        "_create_newton_standalone_state_sync",
        _create_state_sync,
    )
    view = SpawnRigidBodyView(
        SimpleNamespace(backend="newton", topology_revision=3),
        batch,
        torch.device("cpu"),
    )

    view.apply_linear_velocity(
        torch.tensor([[1.0, 2.0, 3.0]]),
        torch.tensor([1]),
    )
    view.apply_angular_velocity(
        torch.tensor([[4.0, 5.0, 6.0]]),
        torch.tensor([2]),
    )

    assert created_body_ids == [(10, 11, 12)]
    assert synchronized_states == [
        (current_state, other_state),
        (current_state, other_state),
    ]
    assert torch.equal(batch.linear_velocity[1], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(batch.angular_velocity[2], torch.tensor([4.0, 5.0, 6.0]))


def test_articulation_partial_force_preserves_other_rows_and_dofs() -> None:
    batch = _ArticulationBatch()
    view = SpawnArticulationView(
        SimpleNamespace(backend="newton"),
        batch,
        torch.device("cpu"),
    )

    view.apply_qf(
        torch.tensor([[50.0]]),
        env_ids=torch.tensor([1]),
        joint_ids=torch.tensor([1]),
    )

    assert torch.equal(
        batch.force,
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 50.0, 6.0]]),
    )
    assert batch.selections == [(1,)]
    assert batch.last_dof_ids == (1,)


def test_newton_idempotent_root_pose_write_is_skipped() -> None:
    batch = _ArticulationBatch()
    view = SpawnArticulationView(
        SimpleNamespace(backend="newton"),
        batch,
        torch.device("cpu"),
    )
    current_pose = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [2.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
        ]
    )

    view.apply_root_pose(current_pose, env_ids=torch.tensor([0, 1]))

    assert batch.root_pose_fetch_rows == [(0, 1)]
    assert batch.root_pose_apply_rows == []


def test_newton_root_pose_write_keeps_only_changed_rows() -> None:
    batch = _ArticulationBatch()
    view = SpawnArticulationView(
        SimpleNamespace(backend="newton"),
        batch,
        torch.device("cpu"),
    )
    target_pose = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [3.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )

    view.apply_root_pose(target_pose, env_ids=torch.tensor([0, 1]))

    assert batch.root_pose_fetch_rows == [(0, 1)]
    assert batch.root_pose_apply_rows == [(1,)]
    assert torch.equal(
        batch.root_pose,
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 3.0, 0.0, 1.0],
            ]
        ),
    )
