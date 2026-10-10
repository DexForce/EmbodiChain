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
import warp as wp

from dexsim.scene import ContactActorInfo, ContactBuffer, ContactQueryCapabilities
from embodichain.lab.sim.sensors import ContactSensor, ContactSensorCfg

pytestmark = [pytest.mark.gpu, pytest.mark.no_sim]


class _CudaQuery:
    def __init__(self, device: torch.device) -> None:
        self.capabilities = ContactQueryCapabilities(True, True, True)
        self.selected_actor_ids = (10, 20)
        self.actors = (
            ContactActorInfo(10, "arena_0/cube", None, "arena_0", 0),
            ContactActorInfo(20, "arena_1/cube", None, "arena_1", 1),
        )
        self.buffer = ContactBuffer.allocate(4, str(device))
        self.buffer.data.zero_()
        self.buffer.data[:2, 5] = 1.0
        self.buffer.data[:2, 9] = 0.2
        self.buffer.actor_ids[:2] = torch.tensor(
            [[0, 10], [0, 20]], device=device, dtype=torch.int32
        )
        self.buffer.env_ids[:2] = torch.tensor([0, 1], device=device, dtype=torch.int32)
        self.buffer.device_counts.zero_()

    def actor_info(self, actor_id: int) -> ContactActorInfo:
        return next(actor for actor in self.actors if actor.actor_id == actor_id)

    def fetch_async(self) -> ContactBuffer:
        return self.buffer


@pytest.mark.parametrize("caller_kind", ["warp", "torch"])
def test_sensor_graph_samples_once_reuses_and_recaptures(caller_kind: str) -> None:
    """Capture and replay retain timing, reset isolation and stream ordering."""
    wp.init()
    device = torch.device("cuda:0")
    caller = (
        wp.stream_to_torch(wp.get_stream(str(device)))
        if caller_kind == "warp"
        else torch.cuda.Stream(device=device)
    )
    with torch.cuda.stream(caller):
        query = _CudaQuery(device)
        owner = SimpleNamespace(
            num_envs=2,
            arena_offsets=torch.zeros((2, 3), device=device),
            _spawn_scene=SimpleNamespace(
                handles=lambda uid: tuple(
                    SimpleNamespace(path=actor.path) for actor in query.actors
                )
            ),
            spawn_result=SimpleNamespace(
                create_contact_query=lambda *args, **kwargs: query
            ),
        )
        sensor = ContactSensor(
            ContactSensorCfg(
                uid="contacts", rigid_uid_list=["cube"], max_contacts_per_env=2
            ),
            device,
            owner=owner,
        )
        history = sensor.create_history(
            "feet", torch.tensor([[10], [20]], device=device)
        )
        sensor.begin_control_step()

        def assert_air_time(value: float) -> None:
            torch.testing.assert_close(
                history.current_air_time,
                torch.full((2, 1), value, device=device),
            )

        sensor.update_physics_step(0.1)
        assert sensor._sample_graph is None
        assert_air_time(0.1)
        sensor.update_physics_step(0.1)
        graph = sensor._sample_graph
        assert graph is not None
        assert_air_time(0.2)
        sensor.update_physics_step(0.1)
        assert sensor._sample_graph is graph
        assert_air_time(0.3)

        query.buffer.device_counts[0] = 2
        sensor.update_physics_step(0.1)
        assert sensor._sample_graph is graph
        assert history.contact.all()
        assert history.found.all()
        assert history.first_contact.all()
        assert_air_time(0.0)
        torch.testing.assert_close(
            history.last_air_time, torch.full((2, 1), 0.4, device=device)
        )
        torch.testing.assert_close(
            history.force,
            torch.tensor([[[0.0, 0.0, 2.0]], [[0.0, 0.0, 2.0]]], device=device),
        )
        torch.testing.assert_close(history.contact_count, torch.ones(2, device=device))
        assert sensor.get_data()["is_valid"].tolist() == [[True, False], [True, False]]
        fields = (
            "contact",
            "found",
            "first_contact",
            "force",
            "peak_force",
            "current_air_time",
            "last_air_time",
            "contact_count",
        )
        untouched = {name: getattr(history, name)[1].clone() for name in fields}
        valid = sensor.get_data()["is_valid"][1].clone()
        sensor.reset((0,))
        for name in fields:
            assert not getattr(history, name)[0].any()
            torch.testing.assert_close(getattr(history, name)[1], untouched[name])
        assert not sensor.get_data()["is_valid"][0].any()
        torch.testing.assert_close(sensor.get_data()["is_valid"][1], valid)
        assert sensor._num_contacts_per_env.tolist() == [0, 1]
        assert sensor._sample_graph is graph

        sensor.begin_control_step()
        history.force_threshold = 3.0
        sensor.update_physics_step(0.1)
        assert sensor._sample_graph is None
        assert not history.contact.any()
        assert not history.found.any()
        assert_air_time(0.1)
        sensor.update_physics_step(0.1)
        recaptured = sensor._sample_graph
        assert recaptured is not None and recaptured is not graph
        assert_air_time(0.2)
        sensor.update_physics_step(0.1)
        assert sensor._sample_graph is recaptured
        assert not history.contact.any()
        assert_air_time(0.3)
    caller.synchronize()
