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
        if hasattr(self.buffer, "quantity_valid"):
            self.buffer.quantity_valid.fill_(True)
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

    def fetch_solver_report_async(self) -> ContactBuffer:
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
            physics_step_index=0,
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

        def sample() -> None:
            owner.physics_step_index += 1
            sensor.update_physics_step(0.1)
            sensor.update_physics_step(0.1)

        def assert_air_time(value: float) -> None:
            torch.testing.assert_close(
                history.current_air_time,
                torch.full((2, 1), value, device=device),
            )

        sample()
        assert sensor._sample_graph is None
        assert_air_time(0.1)
        sample()
        graph = sensor._sample_graph
        assert graph is not None
        assert_air_time(0.2)
        sample()
        assert sensor._sample_graph is graph
        assert_air_time(0.3)

        query.buffer.device_counts[0] = 2
        sample()
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
        sample()
        assert sensor._sample_graph is None
        assert not history.contact.any()
        assert not history.found.any()
        assert_air_time(0.1)
        sample()
        recaptured = sensor._sample_graph
        assert recaptured is not None and recaptured is not graph
        assert_air_time(0.2)
        sample()
        assert sensor._sample_graph is recaptured
        assert not history.contact.any()
        assert_air_time(0.3)

        if hasattr(query.buffer, "quantity_valid"):
            # Replay reads changed masks without recapture and never evaluates
            # unavailable poisoned force components.
            query.buffer.quantity_valid.zero_()
            query.buffer.data[:2, 6:10] = float("nan")
            sample()
            assert sensor._sample_graph is recaptured
            assert not history.force.any()
            assert not sensor.get_data()["impulse_valid"].any()
            assert not sensor.get_data()["friction_valid"].any()

            # A topology refresh can replace the native mask storage. The
            # captured pointer must be refreshed before the next sample.
            query.buffer.quantity_valid = torch.ones_like(query.buffer.quantity_valid)
            query.buffer.data[:2, 6:9] = 0.0
            query.buffer.data[:2, 9] = 0.4
            sample()
            assert sensor._sample_graph is None
            assert history.contact.all()
            torch.testing.assert_close(
                history.force,
                torch.tensor([[[0.0, 0.0, 4.0]], [[0.0, 0.0, 4.0]]], device=device),
            )
            sample()
            assert sensor._sample_graph is not None
            assert sensor._sample_graph is not recaptured
            assert history.contact.all()
        # The graph writes query loss even when there are no valid contacts.
        query.buffer.device_counts.zero_()
        query.buffer.device_counts[1] = 3
        sensor.begin_control_step()
        sample()
        assert sensor.latest_sample_dropped_contacts == 3
        assert sensor.evidence_dropped_contacts == 3
        query.buffer.device_counts[1] = 0
        sample()
        assert sensor.latest_sample_dropped_contacts == 0
        assert sensor.evidence_dropped_contacts == 3  # Earlier interval loss survives.
    caller.synchronize()


@pytest.mark.parametrize("history_sample", [False, True])
def test_cached_sample_is_ready_on_another_cuda_stream(history_sample: bool) -> None:
    """A cached update must wait for the producer's outstanding device work."""
    wp.init()
    device = torch.device("cuda:0")
    query = _CudaQuery(device)
    query.buffer.device_counts[0] = 2
    owner = SimpleNamespace(
        num_envs=2,
        physics_step_index=0,
        arena_offsets=torch.zeros((2, 3), device=device),
        _spawn_scene=SimpleNamespace(
            handles=lambda uid: tuple(
                SimpleNamespace(path=actor.path) for actor in query.actors
            )
        ),
        spawn_result=SimpleNamespace(create_contact_query=lambda *a, **kw: query),
    )
    sensor = ContactSensor(
        ContactSensorCfg(
            uid="contacts", rigid_uid_list=["cube"], max_contacts_per_env=2
        ),
        device,
        owner=owner,
    )
    history = (
        sensor.create_history("feet", torch.tensor([[10], [20]], device=device))
        if history_sample
        else None
    )

    def sample() -> None:
        if history_sample:
            sensor.update_physics_step(0.1)
        else:
            sensor.update()

    sample()  # Compile kernels before arranging the delayed producer.
    sensor.reset()
    torch.cuda.synchronize(device)
    owner.physics_step_index += 1
    producer = torch.cuda.Stream(device=device)
    consumer = torch.cuda.Stream(device=device)
    with torch.cuda.stream(producer):
        torch.cuda._sleep(200_000_000)
        sample()
    with torch.cuda.stream(consumer):
        sample()
        observed = sensor.get_data()["is_valid"].clone()
        force = history.force.clone() if history is not None else None
    consumer.synchronize()
    assert observed.tolist() == [[True, False], [True, False]]
    if force is not None:
        torch.testing.assert_close(
            force,
            torch.tensor([[[0.0, 0.0, 2.0]], [[0.0, 0.0, 2.0]]], device=device),
        )
    producer.synchronize()


@pytest.mark.parametrize("history_sample", [False, True])
@pytest.mark.parametrize("capture_after_advance", [False, True])
@pytest.mark.parametrize("capture_mode", ["same_stream", "other_stream", "lifecycle"])
def test_outer_capture_preserves_sampling_and_eager_refresh(
    history_sample: bool, capture_after_advance: bool, capture_mode: str
) -> None:
    """Warmup must not suppress captured fetch_solver_report/scatter/history or eager refresh."""
    wp.init()
    device = torch.device("cuda:0")
    query = _CudaQuery(device)
    source = query.buffer.data.clone()

    def fetch_solver_report_async() -> ContactBuffer:
        # Model a backend whose query must copy fresh device data on each replay.
        query.buffer.data.copy_(source)
        return query.buffer

    query.fetch_solver_report_async = fetch_solver_report_async
    owner = SimpleNamespace(
        num_envs=2,
        physics_step_index=0,
        arena_offsets=torch.zeros((2, 3), device=device),
        _spawn_scene=SimpleNamespace(
            handles=lambda uid: tuple(
                SimpleNamespace(path=actor.path) for actor in query.actors
            )
        ),
        spawn_result=SimpleNamespace(create_contact_query=lambda *a, **kw: query),
    )
    sensor = ContactSensor(
        ContactSensorCfg(
            uid="contacts", rigid_uid_list=["cube"], max_contacts_per_env=2
        ),
        device,
        owner=owner,
    )
    history = (
        sensor.create_history("feet", torch.tensor([[10], [20]], device=device))
        if history_sample
        else None
    )

    def sample() -> None:
        if history_sample:
            sensor.update_physics_step(0.1)
        else:
            sensor.update()

    def assert_sample(impulse: float) -> None:
        assert sensor.get_data()["is_valid"].tolist() == [[True, False], [True, False]]
        torch.testing.assert_close(
            sensor.get_data()["impulse"][:, 0],
            torch.full_like(sensor.get_data()["impulse"][:, 0], impulse),
        )
        if history is not None:
            torch.testing.assert_close(
                history.force,
                torch.tensor(
                    [[[0.0, 0.0, impulse / 0.1]], [[0.0, 0.0, impulse / 0.1]]],
                    device=device,
                ),
            )

    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    producer = (
        stream if capture_mode == "same_stream" else torch.cuda.Stream(device=device)
    )
    producer.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(producer):
        sample()
    # Even a completed producer must not introduce an uncaptured event into capture.
    producer.synchronize()
    with torch.cuda.stream(stream):
        if capture_after_advance:
            owner.physics_step_index += 1
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            if capture_mode == "lifecycle":
                sensor.reset()
                sensor.begin_control_step()
            sample()
        # Capture has not executed a new sample. Same-step eager calls must work.
        query.buffer.device_counts[0] = 2
        source[:2, 9] = 0.3
        sample()
        assert_sample(0.3)
        sensor.reset()
        for impulse in (0.2, 0.5):
            owner.physics_step_index += 1
            source[:2, 9] = impulse
            graph.replay()
            assert_sample(impulse)
        if history is not None:
            torch.testing.assert_close(
                history.contact_count,
                torch.full_like(
                    history.contact_count, 1 if capture_mode == "lifecycle" else 2
                ),
            )
        # Caller-owned replay must restore latest loss after full reset too.
        query.buffer.device_counts[0] = 0
        for loss in (3, 0):
            sensor.reset()
            assert sensor.evidence_dropped_contacts == 0
            query.buffer.device_counts[1] = loss
            graph.replay()
            assert sensor.latest_sample_dropped_contacts == loss
            assert sensor.evidence_dropped_contacts == loss
    stream.synchronize()
