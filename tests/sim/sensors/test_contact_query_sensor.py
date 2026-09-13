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

import torch
import warp as wp

from dexsim.scene import ContactActorInfo, ContactBuffer, ContactQueryCapabilities
from embodichain.lab.sim.sensors import ContactSensor, ContactSensorCfg


class _FakeQuery:
    def __init__(self) -> None:
        self.capabilities = ContactQueryCapabilities(True, True, True)
        self.selected_actor_ids = (4, 7)
        self._actors = {
            4: ContactActorInfo(4, "arena_0/cube", None, "arena_0", 0),
            7: ContactActorInfo(7, "arena_1/cube", None, "arena_1", 1),
        }
        self.buffer = ContactBuffer.allocate(4, "cpu")
        self.buffer.count = 2
        self.buffer.data[0] = torch.tensor(
            [1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3, 0.4, -0.01]
        )
        self.buffer.data[1] = torch.tensor(
            [4.0, 5.0, 6.0, 0.0, 1.0, 0.0, 0.4, 0.5, 0.6, 0.7, -0.02]
        )
        self.buffer.actor_ids[:2] = torch.tensor([[4, -1], [-1, 7]], dtype=torch.int32)
        self.buffer.env_ids[:2] = torch.tensor([0, 1], dtype=torch.int32)

    def actor_info(self, actor_id: int) -> ContactActorInfo:
        return self._actors[actor_id]

    def fetch(self) -> ContactBuffer:
        return self.buffer


def test_contact_sensor_consumes_scene_query_and_explicit_env_ids() -> None:
    wp.init()
    query = _FakeQuery()
    captured = {}
    result = SimpleNamespace()

    def create_contact_query(targets, **kwargs):
        captured["targets"] = tuple(targets)
        captured.update(kwargs)
        return query

    result.create_contact_query = create_contact_query
    handles = (
        SimpleNamespace(path="arena_0/cube"),
        SimpleNamespace(path="arena_1/cube"),
    )
    owner = SimpleNamespace(
        num_envs=2,
        spawn_result=result,
        _spawn_scene=SimpleNamespace(handles=lambda uid: handles),
        arena_offsets=torch.zeros((2, 3)),
    )
    cfg = ContactSensorCfg(
        uid="contacts",
        rigid_uid_list=["cube"],
        filter_need_both_actor=False,
        max_contacts_per_env=2,
    )

    sensor = ContactSensor(cfg, torch.device("cpu"), owner=owner)
    sensor.update()
    data = sensor.get_data()

    assert captured["targets"] == handles
    assert captured["match"] == "any"
    assert captured["frame"] == "arena"
    assert captured["capacity"] == 4
    assert captured["capacity_per_env"] == 2
    assert sensor.total_current_contacts == 2
    assert data["is_valid"][:, 0].all()
    assert data["position"][0, 0].tolist() == [1.0, 2.0, 3.0]
    assert data["position"][1, 0].tolist() == [4.0, 5.0, 6.0]
    assert data["user_ids"][1, 0].tolist() == [-1, 7]
    assert sensor.get_actor_info(7).path == "arena_1/cube"
    assert sensor.contact_capabilities.impulse
