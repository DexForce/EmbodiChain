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

import importlib.util
import inspect
from types import SimpleNamespace

import pytest
import torch
from dexsim.scene import ArticulationBatch, RigidBodyBatch, Scene

import embodichain.lab.sim.objects.backends as backends
import embodichain.lab.sim.objects.backends.newton as newton_backend
from embodichain.lab.sim.objects.articulation import Articulation, ArticulationData
from embodichain.lab.sim.objects.backends.scene import (
    SceneArticulationView,
    SceneRigidBodyView,
    _batch_pose,
    _embodichain_pose,
)
from embodichain.lab.sim.objects.rigid_object import RigidBodyData, RigidObject

pytestmark = pytest.mark.no_sim


def test_deprecated_batch_adapters_are_removed() -> None:
    deprecated_exports = {
        "DefaultArticulationView",
        "DefaultRigidBodyView",
        "NewtonArticulationView",
        "NewtonRigidBodyView",
        "SpawnArticulationView",
        "SpawnRigidBodyView",
        "apply_collision_filter_for_entities",
        "apply_collision_filter_for_envs",
    }

    assert deprecated_exports.isdisjoint(backends.__all__)
    assert all(not hasattr(backends, name) for name in deprecated_exports)
    assert (
        importlib.util.find_spec("embodichain.lab.sim.objects.backends.spawn") is None
    )
    assert (
        importlib.util.find_spec("embodichain.lab.sim.objects.backends.default") is None
    )


def test_scene_views_match_installed_dexsim_batch_surface() -> None:
    rigid_methods = {
        "select",
        "apply_pose",
        "fetch_pose",
        "apply_com_local_pose",
        "fetch_com_local_pose",
        "apply_linear_velocity",
        "fetch_linear_velocity",
        "apply_angular_velocity",
        "fetch_angular_velocity",
        "fetch_linear_acceleration",
        "fetch_angular_acceleration",
        "apply_force",
        "apply_torque",
        "apply_mass",
        "fetch_mass",
        "apply_inertia_diagonal",
        "fetch_inertia_diagonal",
        "apply_friction",
        "fetch_friction",
        "apply_restitution",
        "fetch_restitution",
        "apply_contact_offset",
        "fetch_contact_offset",
        "apply_damping",
        "fetch_damping",
        "apply_collision_filter",
        "fetch_collision_filter",
    }
    articulation_methods = {
        "select",
        "apply_root_pose",
        "fetch_root_pose",
        "fetch_root_linear_velocity",
        "fetch_root_angular_velocity",
        "apply_joint_position",
        "fetch_joint_position",
        "apply_joint_target_position",
        "fetch_joint_target_position",
        "apply_joint_velocity",
        "fetch_joint_velocity",
        "apply_joint_target_velocity",
        "fetch_joint_target_velocity",
        "apply_joint_force",
        "fetch_joint_force",
        "fetch_joint_acceleration",
        "fetch_link_pose",
        "fetch_link_linear_velocity",
        "fetch_link_angular_velocity",
        "compute_kinematics",
    }
    articulation_metadata = {
        "dof_counts",
        "link_counts",
        "joint_names_per_articulation",
        "link_names_per_articulation",
        "joint_layouts_per_articulation",
        "dof_width",
        "link_width",
    }

    assert rigid_methods <= set(dir(RigidBodyBatch))
    assert articulation_methods <= set(dir(ArticulationBatch))
    assert articulation_metadata <= set(dir(ArticulationBatch))
    assert callable(Scene.create_rigid_body_batch)
    assert callable(Scene.create_articulation_batch)
    for method_name in (
        "apply_joint_position",
        "apply_joint_target_position",
        "apply_joint_velocity",
        "apply_joint_target_velocity",
        "apply_joint_force",
    ):
        assert (
            "dof_ids"
            in inspect.signature(getattr(ArticulationBatch, method_name)).parameters
        )


def test_scene_pose_adapters_preserve_embodichain_xyzw_order() -> None:
    pose = torch.tensor([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9]])
    expected_batch = torch.tensor([[0.1, 0.2, 0.3, 0.9, 1.0, 2.0, 3.0]])

    torch.testing.assert_close(_batch_pose(pose), expected_batch)
    torch.testing.assert_close(_embodichain_pose(expected_batch), pose)


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

    def apply_collision_filter(self, values: torch.Tensor) -> int:
        self.owner.collision_filter[self.rows] = values
        return len(self.rows)

    def fetch_friction(self, out: torch.Tensor) -> int:
        out.copy_(self.owner.friction[self.rows])
        return len(self.rows)

    def fetch_collision_filter(self, out: torch.Tensor) -> int:
        out.copy_(self.owner.collision_filter[self.rows])
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
        self.collision_filter = torch.zeros((3, 4), dtype=torch.int32)
        self.linear_velocity = torch.zeros((3, 3))
        self.angular_velocity = torch.zeros((3, 3))
        self.selections: list[tuple[int, ...]] = []

    def __len__(self) -> int:
        return len(self.force)

    def select(self, rows: torch.Tensor) -> _SelectedRigidBatch:
        selected = rows.detach().cpu().to(dtype=torch.long)
        self.selections.append(tuple(selected.tolist()))
        return _SelectedRigidBatch(self, selected)

    def apply_collision_filter(self, values: torch.Tensor) -> int:
        self.collision_filter.copy_(values.to(dtype=self.collision_filter.dtype))
        return len(self)

    def fetch_friction(self, out: torch.Tensor) -> int:
        out.copy_(self.friction)
        return len(self)

    def fetch_collision_filter(self, out: torch.Tensor) -> int:
        out.copy_(self.collision_filter)
        return len(self)


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
        # Scene articulation batches use xyzw + xyz layout.
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


class _Scene(Scene):
    def __init__(self, backend: str = "newton") -> None:
        self.backend = backend
        self._topology_revision = 0
        self.rigid_batch = _RigidBatch()
        self.articulation_batch = _ArticulationBatch()
        self.rigid_batch_objects: list[object] | None = None
        self.articulation_batch_objects: list[object] | None = None

    def create_rigid_body_batch(self, objects: list[object]) -> _RigidBatch:
        self.rigid_batch_objects = objects
        return self.rigid_batch

    def create_articulation_batch(
        self,
        articulations: list[object],
    ) -> _ArticulationBatch:
        self.articulation_batch_objects = articulations
        return self.articulation_batch


class _ArticulationEntity:
    def get_joint_position_limits(self) -> list[list[float]]:
        return [[-1.0, 1.0]] * 3

    def get_joint_velocity_limit(self) -> list[float]:
        return [2.0] * 3

    def get_joint_effort_limit(self) -> list[float]:
        return [3.0] * 3


def test_is_newton_scene_requires_current_scene_batch_api() -> None:
    assert newton_backend.is_newton_scene(_Scene("newton")) is True
    assert newton_backend.is_newton_scene(_Scene("dexsim")) is False
    assert newton_backend.is_newton_scene(SimpleNamespace(backend="newton")) is False


@pytest.mark.parametrize("backend", ["dexsim", "newton"])
def test_rigid_body_data_uses_scene_view(backend: str) -> None:
    scene = _Scene(backend)
    entities = [object(), object(), object()]

    data = RigidBodyData(entities, scene, torch.device("cpu"))

    assert isinstance(data.body_view, SceneRigidBodyView)
    assert data.is_newton_backend is (backend == "newton")
    assert scene.rigid_batch_objects == entities


def test_rigid_body_data_rejects_raw_physics_scene_path() -> None:
    with pytest.raises(TypeError, match="requires a finalized DexSim Scene"):
        RigidBodyData([object()], SimpleNamespace(), torch.device("cpu"))


@pytest.mark.parametrize("object_type", [RigidObject, Articulation])
def test_materialized_object_construction_requires_scene(object_type: type) -> None:
    with pytest.raises(TypeError, match="requires a finalized DexSim Scene"):
        object_type(SimpleNamespace(), [object()])


@pytest.mark.parametrize("backend", ["dexsim", "newton"])
def test_articulation_data_uses_scene_view(backend: str) -> None:
    scene = _Scene(backend)
    entities = [_ArticulationEntity(), _ArticulationEntity()]

    data = ArticulationData(entities, scene, torch.device("cpu"))

    assert isinstance(data.articulation_view, SceneArticulationView)
    assert data.is_newton_backend is (backend == "newton")
    assert scene.articulation_batch_objects == entities


def test_articulation_data_rejects_raw_physics_scene_path() -> None:
    with pytest.raises(TypeError, match="requires a finalized DexSim Scene"):
        ArticulationData(
            [_ArticulationEntity()],
            SimpleNamespace(),
            torch.device("cpu"),
        )


def test_rigid_partial_writes_delegate_to_selected_batch() -> None:
    batch = _RigidBatch()
    view = SceneRigidBodyView(
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
    view = SceneRigidBodyView(
        SimpleNamespace(backend="dexsim"),
        batch,
        torch.device("cpu"),
    )
    out = torch.empty((2, 1))

    view.fetch_friction(out, torch.tensor([2, 0]))

    assert torch.equal(out, torch.tensor([[0.3], [0.1]]))
    assert batch.selections == [(2, 0)]


def test_rigid_full_fetch_uses_original_batch() -> None:
    batch = _RigidBatch()
    view = SceneRigidBodyView(
        SimpleNamespace(backend="dexsim"),
        batch,
        torch.device("cpu"),
    )
    out = torch.empty((3, 1))

    view.fetch_friction(out)

    assert torch.equal(out, batch.friction)
    assert batch.selections == []


def test_rigid_batch_failure_status_is_not_silently_ignored() -> None:
    batch = _RigidBatch()
    view = SceneRigidBodyView(
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
        newton_backend,
        "_create_newton_standalone_state_sync",
        _create_state_sync,
    )
    view = SceneRigidBodyView(
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
        newton_backend,
        "_create_newton_standalone_state_sync",
        _create_state_sync,
    )
    view = SceneRigidBodyView(
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
    view = SceneArticulationView(
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
    view = SceneArticulationView(
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
    view = SceneArticulationView(
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


def test_newton_root_pose_rejects_mismatched_selection_shape() -> None:
    view = SceneArticulationView(
        SimpleNamespace(backend="newton"),
        _ArticulationBatch(),
        torch.device("cpu"),
    )

    with pytest.raises(ValueError, match="Expected selected data shape \\(2, 7\\)"):
        view.apply_root_pose(
            torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]),
            env_ids=torch.tensor([0, 1]),
        )


def test_articulation_joint_selection_rejects_negative_indices() -> None:
    view = SceneArticulationView(
        SimpleNamespace(backend="dexsim"),
        _ArticulationBatch(),
        torch.device("cpu"),
    )

    with pytest.raises(IndexError, match="outside \\[0, 3\\)"):
        view.apply_qf(
            torch.tensor([[1.0]]),
            env_ids=torch.tensor([0]),
            joint_ids=torch.tensor([-1]),
        )


def test_scene_views_use_current_scene_batch_factories() -> None:
    scene = _Scene()
    rigid_entities = [object(), object(), object()]
    articulation_entities = [object(), object()]

    rigid_view = SceneRigidBodyView.from_entities(
        scene, rigid_entities, torch.device("cpu")
    )
    articulation_view = SceneArticulationView.from_entities(
        scene, articulation_entities, torch.device("cpu")
    )

    rigid_view.apply_friction(torch.tensor([[0.8]]), torch.tensor([1]))
    articulation_view.apply_qf(
        torch.tensor([[20.0]]),
        env_ids=torch.tensor([1]),
        joint_ids=torch.tensor([2]),
    )

    assert scene.rigid_batch_objects == rigid_entities
    assert scene.articulation_batch_objects == articulation_entities
    assert torch.equal(scene.rigid_batch.friction, torch.tensor([[0.1], [0.8], [0.3]]))
    assert torch.equal(
        scene.articulation_batch.force,
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 20.0]]),
    )


def test_scene_rigid_collision_filter_delegates_to_batch() -> None:
    scene = _Scene()
    view = SceneRigidBodyView.from_entities(
        scene, [object(), object(), object()], torch.device("cpu")
    )
    expected = torch.tensor(
        [[0, 1, 0, 0], [1, 1, 0, 0], [2, 1, 0, 0]], dtype=torch.int32
    )

    view.apply_collision_filter(expected, torch.tensor([0, 1, 2]))
    actual = torch.empty_like(expected)
    view.fetch_collision_filter(actual)

    assert torch.equal(actual, expected)
