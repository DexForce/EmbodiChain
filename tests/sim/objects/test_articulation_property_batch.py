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

from embodichain.lab.sim.objects.articulation import ArticulationData
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
    getter = Mock(
        return_value=SimpleNamespace(
            mass=3.0,
            inertia=[1.0, 2.0, 3.0],
            com_position=[4.0, 5.0, 6.0],
            com_quaternion=[1.0, 0.0, 0.0, 0.0],
        )
    )
    data._entity_link_properties = getter
    mass, inertia, pose = data.read_physical_properties()
    assert getter.call_count == 4
    torch.testing.assert_close(mass, torch.full((2, 2), 3.0))
    torch.testing.assert_close(inertia, torch.tensor([1.0, 2.0, 3.0]).expand(2, 2, 3))
    torch.testing.assert_close(
        pose, torch.tensor([4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0]).expand(2, 2, 7)
    )
