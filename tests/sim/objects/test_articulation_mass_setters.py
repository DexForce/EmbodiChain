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

import numpy as np
import pytest
import torch

from embodichain.lab.sim.objects import Articulation

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize("newton", [False, True])
@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("property_name,width", [("inertia", 3), ("com_pose", 7)])
def test_spawn_mass_setters_preserve_selection_and_quaternion_order(
    newton: bool, selected: bool, property_name: str, width: int
) -> None:
    articulation = object.__new__(Articulation)
    articulation.device = torch.device("cpu")
    articulation._spawn_result = object()
    articulation._data = SimpleNamespace(
        is_newton_backend=newton, link_names=["base", "tip"]
    )
    articulation._entities = [
        Mock(spec=["set_link_inertia", "set_link_com_pose"]) for _ in range(3)
    ]
    for entity in articulation._entities:
        entity.set_link_inertia.return_value = 0
        entity.set_link_com_pose.return_value = 0
    articulation._declared_num_instances = 3
    env_ids = [2, 0] if selected else None
    link_names = ["tip", "base"] if selected else None
    rows = env_ids if selected else [0, 1, 2]
    names = link_names if selected else ["base", "tip"]
    values = (
        torch.arange(len(rows) * len(names) * width, dtype=torch.float32).reshape(
            len(rows), len(names), width
        )
        + 1
    )
    if property_name == "com_pose":
        values[..., 3:] /= torch.linalg.vector_norm(
            values[..., 3:], dim=-1, keepdim=True
        )
    original = values.clone()

    getattr(articulation, "set_" + property_name)(
        values, link_names=link_names, env_ids=env_ids
    )

    for i, env_id in enumerate(rows):
        calls = getattr(
            articulation._entities[env_id], "set_link_" + property_name
        ).call_args_list
        assert len(calls) == len(names)
        for j, name in enumerate(names):
            args = calls[j].args
            assert args[0] == name
            np.testing.assert_array_equal(args[1], values[i, j, :3].numpy())
            if property_name == "com_pose":
                np.testing.assert_array_equal(
                    args[2], values[i, j, [6, 3, 4, 5]].numpy()
                )
    if selected:
        assert articulation._entities[1].mock_calls == []
    torch.testing.assert_close(values, original)


@pytest.mark.parametrize("property_name,width", [("inertia", 3), ("com_pose", 7)])
def test_spawn_mass_setters_reject_incorrect_shape_before_writing(
    property_name: str, width: int
) -> None:
    articulation = object.__new__(Articulation)
    articulation.device = torch.device("cpu")
    articulation._data = SimpleNamespace(link_names=["base"])
    articulation._entities = [Mock()]
    with pytest.raises(ValueError, match="Expected"):
        getattr(articulation, "set_" + property_name)(
            torch.zeros(1, width), link_names="base", env_ids=[0]
        )
    assert articulation._entities[0].mock_calls == []


@pytest.mark.parametrize("newton", [False, True])
@pytest.mark.parametrize("status", [-1, -2])
@pytest.mark.parametrize("property_name,width", [("inertia", 3), ("com_pose", 7)])
def test_spawn_mass_setters_raise_for_negative_native_status(
    newton: bool, status: int, property_name: str, width: int
) -> None:
    articulation = object.__new__(Articulation)
    articulation.device = torch.device("cpu")
    articulation._spawn_result = object()
    articulation._data = SimpleNamespace(is_newton_backend=newton, link_names=["base"])
    entity = Mock(spec=["set_link_inertia", "set_link_com_pose"])
    getattr(entity, "set_link_" + property_name).return_value = status
    articulation._entities = [Mock(), entity]

    with pytest.raises(
        RuntimeError, match=f"set_link_{property_name}.*env 1.*base.*{status}"
    ):
        getattr(articulation, "set_" + property_name)(
            torch.ones(1, 1, width), link_names="base", env_ids=[1]
        )
    assert articulation._entities[0].mock_calls == []
