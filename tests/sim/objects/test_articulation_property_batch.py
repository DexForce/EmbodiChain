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
from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.sim.objects.articulation import Articulation, ArticulationData
from embodichain.lab.sim.objects.backends.scene import SceneArticulationView

pytestmark = pytest.mark.no_sim


class PropertyBatch:
    def __init__(self) -> None:
        self.mass = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.inertia = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
        self.pose = torch.arange(28, dtype=torch.float32).reshape(2, 2, 7)
        self.pose_buffers: list[torch.Tensor] = []
        self.status = 2

    def fetch_link_mass(self, out: torch.Tensor) -> int:
        out.copy_(self.mass)
        return self.status

    def fetch_link_inertia_diagonal(self, out: torch.Tensor) -> int:
        out.copy_(self.inertia)
        return self.status

    def fetch_link_com_local_pose(self, out: torch.Tensor) -> int:
        self.pose_buffers.append(out)
        out.copy_(self.pose)
        return self.status


def make_data(newton: bool = True) -> ArticulationData:
    data = object.__new__(ArticulationData)
    data.device = torch.device("cpu")
    data.num_instances = 2
    data.num_links = 2
    data.link_names = ["base", "tip"]
    data.entities = [object(), object()]
    data._mass = torch.empty(2, 2)
    data._inertia = torch.empty(2, 2, 3)
    data._com_pose = torch.empty(2, 2, 7)
    view = object.__new__(SceneArticulationView)
    view.scene = SimpleNamespace(backend="newton" if newton else "dexsim")
    view.device = data.device
    view.batch = PropertyBatch()
    view._link_com_scratch = None
    data.articulation_view = view
    return data


def test_newton_property_read_reuses_buffers_and_reads_current_values() -> None:
    data = make_data()
    data._entity_link_properties = Mock(side_effect=AssertionError("scalar getter"))
    original = data._mass, data._inertia, data._com_pose
    batch = data.articulation_view.batch
    for delta in (0.0, 10.0):
        batch.mass.add_(delta)
        batch.inertia.add_(delta)
        batch.pose.add_(delta)
        result = data.read_physical_properties()
        assert all(a is b for a, b in zip(result, original, strict=True))
        torch.testing.assert_close(result[0], batch.mass)
        torch.testing.assert_close(result[1], batch.inertia)
        torch.testing.assert_close(result[2][..., :3], batch.pose[..., 4:])
        torch.testing.assert_close(result[2][..., 3:], batch.pose[..., :4])
    assert batch.pose_buffers[0] is batch.pose_buffers[1]
    data._entity_link_properties.assert_not_called()


def test_property_read_reports_batch_failure() -> None:
    data = make_data()
    data.articulation_view.batch.status = -1
    with pytest.raises(RuntimeError, match="fetch_link_mass"):
        data.read_physical_properties()


def test_other_backend_keeps_one_scalar_read_per_link() -> None:
    data = make_data(False)
    original = data._mass, data._inertia, data._com_pose
    getter = Mock(
        return_value=SimpleNamespace(
            mass=3.0,
            inertia=[1.0, 2.0, 3.0],
            com_position=[4.0, 5.0, 6.0],
            com_quaternion=[1.0, 0.0, 0.0, 0.0],
        )
    )
    data._entity_link_properties = getter
    result = data.read_physical_properties()
    assert all(a is b for a, b in zip(result, original, strict=True))
    mass, inertia, pose = result
    assert getter.call_count == 4
    torch.testing.assert_close(mass, torch.full((2, 2), 3.0))
    torch.testing.assert_close(inertia, torch.tensor([1.0, 2.0, 3.0]).expand(2, 2, 3))
    torch.testing.assert_close(
        pose, torch.tensor([4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0]).expand(2, 2, 7)
    )
    getter.return_value.mass = 5.0
    result = data.read_physical_properties()
    assert all(a is b for a, b in zip(result, original, strict=True))
    torch.testing.assert_close(mass, torch.full((2, 2), 5.0))
    assert getter.call_count == 8


@pytest.mark.parametrize("newton", [False, True])
@pytest.mark.parametrize("env_ids", [(1, 0, 1), (1,), ()])
def test_selected_property_read_preserves_order_freshness_and_ownership(
    newton: bool, env_ids: tuple[int, ...]
) -> None:
    data = make_data(newton)
    batch = data.articulation_view.batch
    buffers = data._mass, data._inertia, data._com_pose
    for buffer in buffers:
        buffer.fill_(-1.0)

    def read_link(entity: object, link_name: str) -> SimpleNamespace:
        row = data.entities.index(entity)
        link = data.link_names.index(link_name)
        native_pose = batch.pose[row, link]
        return SimpleNamespace(
            mass=batch.mass[row, link].item(),
            inertia=batch.inertia[row, link].numpy(),
            com_position=native_pose[4:].numpy(),
            com_quaternion=native_pose[[3, 0, 1, 2]].numpy(),
        )

    data._entity_link_properties = Mock(side_effect=read_link)
    expected = (
        batch.mass.clone(),
        batch.inertia.clone(),
        torch.cat((batch.pose[..., 4:], batch.pose[..., :4]), dim=-1),
    )
    selected = list(env_ids)
    result = data.read_physical_properties(env_ids)
    for actual, values, buffer in zip(result, expected, buffers, strict=True):
        torch.testing.assert_close(actual, values[selected])
        if newton:
            torch.testing.assert_close(buffer, values)
        else:
            for row in range(data.num_instances):
                torch.testing.assert_close(
                    buffer[row],
                    (
                        values[row]
                        if row in selected
                        else torch.full_like(buffer[row], -1)
                    ),
                )
        actual.add_(100.0)
        torch.testing.assert_close(buffer[selected], values[selected])

    getter = data._entity_link_properties
    if newton or not selected:
        getter.assert_not_called()
    else:
        assert {call.args[0] for call in getter.call_args_list} == {
            data.entities[row] for row in selected
        }
    batch.mass.add_(10.0)
    batch.inertia.add_(10.0)
    batch.pose.add_(10.0)
    refreshed = data.read_physical_properties(env_ids)
    for actual, previous, values in zip(refreshed, result, expected, strict=True):
        torch.testing.assert_close(actual, values[selected] + 10.0)
        torch.testing.assert_close(previous, values[selected] + 100.0)


@pytest.mark.parametrize("status", [2, -1, -2])
def test_newton_com_write_preserves_selection_pose_and_native_status(
    status: int,
) -> None:
    art = object.__new__(Articulation)
    art.device = torch.device("cpu")
    art._spawn_result = object()
    art._declared_num_instances = 3
    art._entities = [Mock(), Mock(), Mock()]
    view = object.__new__(SceneArticulationView)
    view.device = art.device
    view.batch = Mock()
    view._row_indices = torch.arange(3)
    view.batch.apply_link_com_local_pose.return_value = status
    art._data = SimpleNamespace(
        is_newton_backend=True,
        link_names=["base", "tip"],
        articulation_view=view,
    )
    pose = torch.tensor([1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.4]).repeat(2, 2, 1)
    pose[..., 3:] /= torch.linalg.vector_norm(pose[..., 3:], dim=-1, keepdim=True)
    original = pose.clone()
    if status < 0:
        with pytest.raises(RuntimeError, match=f"apply_link_com_local_pose.*{status}"):
            art.set_com_pose(pose, link_names=["tip", "base"], env_ids=[2, 0])
    else:
        art.set_com_pose(pose, link_names=["tip", "base"], env_ids=[2, 0])
    args, kwargs = view.batch.apply_link_com_local_pose.call_args
    assert torch.as_tensor(kwargs["link_ids"]).tolist() == [1, 0]
    assert torch.as_tensor(kwargs["rows"]).tolist() == [2, 0]
    view.batch.select.assert_not_called()
    torch.testing.assert_close(args[0][..., :4], original[..., 3:])
    torch.testing.assert_close(args[0][..., 4:], original[..., :3])
    torch.testing.assert_close(pose, original)
    assert all(not entity.mock_calls for entity in art._entities)
