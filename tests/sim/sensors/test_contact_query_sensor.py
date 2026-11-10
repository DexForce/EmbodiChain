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

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import warp as wp

from dexsim.scene import ContactActorInfo, ContactBuffer, ContactQueryCapabilities
from embodichain.lab.gym.envs import BaseEnv
from embodichain.lab.sim.sensors import ContactSensor, ContactSensorCfg


class _FakeQuery:
    def __init__(self) -> None:
        self.capabilities = ContactQueryCapabilities(True, True, True)
        self.selected_actor_ids = (4, 7)
        self._actors = {
            4: ContactActorInfo(4, "arena_0/cube", None, "arena_0", 0),
            7: ContactActorInfo(7, "arena_1/cube", None, "arena_1", 1),
        }
        self.actors = tuple(self._actors.values())
        self.actor_info_calls = 0
        self.fetch_calls = 0
        self.available = True
        self.buffer = ContactBuffer.allocate(4, "cpu")
        self.buffer.count = 2
        if hasattr(self.buffer, "quantity_valid"):
            self.buffer.quantity_valid.fill_(True)
        self.buffer.data[0] = torch.tensor(
            [1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3, 0.4, -0.01]
        )
        self.buffer.data[1] = torch.tensor(
            [4.0, 5.0, 6.0, 0.0, 1.0, 0.0, 0.4, 0.5, 0.6, 0.7, -0.02]
        )
        self.buffer.actor_ids[:2] = torch.tensor([[4, -1], [-1, 7]], dtype=torch.int32)
        self.buffer.env_ids[:2] = torch.tensor([0, 1], dtype=torch.int32)

    def actor_info(self, actor_id: int) -> ContactActorInfo:
        self.actor_info_calls += 1
        return self._actors[actor_id]

    def fetch_solver_report(self) -> ContactBuffer:
        assert self.available, "SDK contact snapshot is unavailable after reset"
        self.fetch_calls += 1
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
        physics_step_index=0,
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
    assert query.fetch_calls == 1
    original_ids = sensor.item_user_ids
    original_env_ids = sensor.item_env_ids
    assert query.actor_info_calls == len(query.selected_actor_ids)
    sensor.update()
    assert query.fetch_calls == 1
    assert sensor.item_user_ids is original_ids
    assert sensor.item_env_ids is original_env_ids
    assert query.actor_info_calls == len(query.selected_actor_ids)
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


def test_contact_metadata_refreshes_when_actor_table_changes() -> None:
    """A topology refresh may change environment IDs while retaining actor IDs."""
    query = _FakeQuery()
    sensor = ContactSensor.__new__(ContactSensor)
    sensor.device = torch.device("cpu")
    sensor._query = query
    sensor._metadata_actor_table = None
    sensor._metadata_selected_ids = None
    sensor._sync_filter_actor_metadata()
    assert sensor.item_user_env_ids_map[7].item() == 1

    query._actors[7] = ContactActorInfo(7, "arena_0/other", None, "arena_0", 0)
    query.actors = tuple(query._actors.values())
    sensor._sync_filter_actor_metadata()
    assert sensor.item_env_ids.tolist() == [0, 0]
    assert sensor.item_user_env_ids_map[7].item() == 0

    query.selected_actor_ids = (7,)
    sensor._sync_filter_actor_metadata()
    assert sensor.item_user_ids.tolist() == [7]
    assert sensor.item_user_env_ids_map[4].item() == -1


def _sensor_for_query(query, capacity=2):
    handles = (
        SimpleNamespace(path="arena_0/cube"),
        SimpleNamespace(path="arena_1/cube"),
    )
    owner = SimpleNamespace(
        num_envs=2,
        physics_step_index=0,
        arena_offsets=torch.zeros((2, 3)),
        _spawn_scene=SimpleNamespace(handles=lambda uid: handles),
        spawn_result=SimpleNamespace(
            create_contact_query=lambda *args, **kwargs: query
        ),
    )
    return ContactSensor(
        ContactSensorCfg(
            uid="contacts", rigid_uid_list=["cube"], max_contacts_per_env=capacity
        ),
        owner=owner,
    )


def test_contact_report_errors_other_than_reset_are_not_hidden():
    query = _FakeQuery()
    sensor = _sensor_for_query(query)

    def fail():
        raise RuntimeError("contact device failure")

    query.fetch_solver_report = fail
    with pytest.raises(RuntimeError, match="contact device failure"):
        sensor.update()


def test_default_history_enables_substep_sampling_and_preserves_interval_drops():
    query = _FakeQuery()
    sensor = _sensor_for_query(query)
    history = sensor.create_history("feet", torch.tensor([[4], [7]]))
    assert sensor.requires_substep_update
    sensor.begin_control_step()
    query.buffer.dropped_count = 3
    sensor._sim.physics_step_index += 1
    sensor.update_physics_step(0.01)
    assert history.found.all()
    sensor.update_physics_step(0.01)
    assert query.fetch_calls == 1
    query.buffer.count = 0
    query.buffer.dropped_count = 0
    sensor._sim.physics_step_index += 1
    sensor.update_physics_step(0.01)
    assert not history.contact.any()
    assert history.found.all()
    assert sensor.dropped_contacts == 3
    assert sensor.dropped_contacts_mask.tolist() == [True, True]
    sensor.update()  # Observation refresh must not replace interval diagnostics.
    assert query.fetch_calls == 2
    assert sensor.dropped_contacts == 3
    sensor.begin_control_step()
    assert sensor.dropped_contacts == 0
    assert not sensor.dropped_contacts_mask.any()


def test_scatter_overflow_is_counted_separately_and_reset_by_selected_row():
    query = _FakeQuery()
    query.buffer.count = 3
    query.buffer.env_ids[:3] = torch.tensor([0, 0, 1])
    query.buffer.actor_ids[:3] = torch.tensor([[4, -1], [4, -1], [7, -1]])
    query.buffer.data[2] = query.buffer.data[1]
    sensor = _sensor_for_query(query, capacity=1)
    sensor.create_history("feet", torch.tensor([[4], [7]]))
    sensor.begin_control_step()
    sensor.update_physics_step(0.01)
    assert sensor.dropped_contacts_mask.tolist() == [True, False]
    sensor.begin_control_step()
    query.buffer.dropped_count = 2
    sensor._sim.physics_step_index += 1
    sensor.update_physics_step(0.01)
    assert sensor.total_current_contacts == 2
    assert sensor._scatter_dropped_count.tolist() == [1, 0]
    assert sensor.dropped_contacts == 3
    query.buffer.count = 0
    query.buffer.dropped_count = 0
    sensor._sim.physics_step_index += 1
    sensor.update_physics_step(0.01)
    assert sensor.dropped_contacts == 3
    sensor.reset([1])
    assert sensor.dropped_contacts == 3
    sensor.reset()
    assert sensor.dropped_contacts == 0


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
)
def test_mixed_contact_quantities_survive_scatter_filter_and_history(device):
    query = _FakeQuery()
    if not hasattr(query.buffer, "quantity_valid"):
        pytest.skip("Per-row contact quantities require DexSim's current dev API.")
    query.capabilities = ContactQueryCapabilities(True, False, False)
    query.buffer.quantity_valid[:2] = torch.tensor([[True, False], [False, True]])
    # Poison unsupported quantities; neither the observation mask nor the
    # history reducer may promote them to measured forces.
    query.buffer.data[0, 6:9] = float("nan")
    query.buffer.data[1, 9] = float("nan")
    query.buffer.data[1, 6:9] = torch.tensor([0.0, 0.5, 0.0])
    if device != "cpu":
        old = query.buffer
        query.buffer = ContactBuffer.allocate(old.capacity, device)
        for name in ("data", "actor_ids", "env_ids", "quantity_valid"):
            getattr(query.buffer, name).copy_(getattr(old, name))
        query.buffer.count = old.count
        query.buffer.device_counts[0] = old.count
        query.fetch_solver_report_async = query.fetch_solver_report
    handles = (
        SimpleNamespace(path="arena_0/cube"),
        SimpleNamespace(path="arena_1/cube"),
    )
    owner = SimpleNamespace(
        num_envs=2,
        physics_step_index=0,
        arena_offsets=torch.zeros((2, 3), device=device),
        _spawn_scene=SimpleNamespace(handles=lambda uid: handles),
        spawn_result=SimpleNamespace(
            create_contact_query=lambda *args, **kwargs: query
        ),
    )
    sensor = ContactSensor(
        ContactSensorCfg(
            uid="contacts",
            rigid_uid_list=["cube"],
            filter_need_both_actor=False,
            max_contacts_per_env=2,
        ),
        torch.device(device),
        owner=owner,
    )
    history = sensor.create_history(
        "forces", torch.tensor([[4], [7]], device=device), force_threshold=1.0
    )
    sensor.update_physics_step(0.1)
    data = sensor.get_data()
    assert data["impulse_valid"].cpu().tolist() == [[True, False], [False, False]]
    assert data["friction_valid"].cpu().tolist() == [[False, False], [True, False]]
    torch.testing.assert_close(
        history.force.cpu(), torch.tensor([[[0.0, 0.0, -4.0]], [[0.0, -5.0, 0.0]]])
    )
    assert history.contact.all()
    filtered = sensor.filter_by_user_ids(torch.tensor([4]))
    assert filtered["impulse_valid"].tolist() == [True]
    assert filtered["friction_valid"].tolist() == [False]
    sensor.reset([0])
    assert not data["impulse_valid"].any()
    assert data["friction_valid"][1, 0]
    query.buffer.count = 0
    query.buffer.device_counts.zero_()
    sensor.update()
    assert not data["is_valid"].any()
    assert not data["friction_valid"].any()


def test_legacy_contact_buffer_uses_query_quantity_capabilities():
    query = _FakeQuery()
    old = query.buffer
    query.buffer = SimpleNamespace(
        data=old.data,
        actor_ids=old.actor_ids,
        env_ids=old.env_ids,
        device_counts=old.device_counts,
        count_device=old.count_device,
        dropped_count_device=old.dropped_count_device,
        capacity=old.capacity,
        count=old.count,
        dropped_count=old.dropped_count,
    )
    query.capabilities = ContactQueryCapabilities(True, False, False)
    query.buffer.data[:2, 6:10] = float("nan")
    sensor = _sensor_for_query(query)
    history = sensor.create_history(
        "forces", torch.tensor([[4], [7]]), force_threshold=1.0
    )
    sensor.update_physics_step(0.1)
    data = sensor.get_data()
    assert data["is_valid"][:, 0].all()
    assert not data["impulse_valid"].any()
    assert not data["friction_valid"].any()
    assert not history.force.any()
    assert not history.found.any()


@pytest.mark.parametrize("with_history", [False, True])
def test_sampling_observers_are_instance_owned_and_bracket_native_fetch(with_history):
    query = _FakeQuery()
    sensor = _sensor_for_query(query)
    if with_history:
        sensor.create_history("existing_history", torch.tensor([[4], [7]]))
    other = _sensor_for_query(_FakeQuery())
    events = []
    original_fetch = query.fetch_solver_report

    def fetch_solver_report():
        events.append("fetch_solver_report")
        return original_fetch()

    query.fetch_solver_report = fetch_solver_report
    observer = SimpleNamespace(
        begin_control_step=lambda: events.append("begin"),
        before_sample=lambda: (events.append("before"), "stamp")[1],
        after_sample=lambda dt, token: events.append(
            ("after", dt, token, int(sensor.get_data()["is_valid"].sum()))
        ),
        reset=lambda ids: events.append(("reset", None if ids is None else list(ids))),
    )
    sensor.add_sampling_observer("audit", observer)
    assert sensor.requires_substep_update
    assert not other.requires_substep_update
    with pytest.raises(ValueError, match="already exists"):
        sensor.add_sampling_observer("audit", observer)
    sensor.begin_control_step()
    sensor.update_physics_step(0.01)
    sensor.reset([0])
    assert events == [
        "begin",
        "before",
        "fetch_solver_report",
        ("after", 0.01, "stamp", 2),
        ("reset", [0]),
    ]
    query.buffer.dropped_count = 3
    sensor._sim.physics_step_index += 1
    sensor.update_physics_step(0.01)
    query.buffer.dropped_count = 0
    sensor.update()
    assert sensor.dropped_contacts == 3
    assert sensor.remove_sampling_observer("audit") is observer
    assert sensor.requires_substep_update is with_history
    events.clear()
    sensor._sim.physics_step_index += 1
    sensor.update_physics_step(0.01)
    assert events == ["fetch_solver_report"]


def test_sampling_observers_reject_external_capture_before_callbacks(monkeypatch):
    sensor = _sensor_for_query(_FakeQuery())
    calls = []
    observer = SimpleNamespace(
        begin_control_step=lambda: None,
        before_sample=lambda: calls.append("before"),
        after_sample=lambda *args: None,
        reset=lambda *args: None,
    )
    sensor.add_sampling_observer("audit", observer)
    sensor.device = SimpleNamespace(type="cuda")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="external CUDA capture"):
        sensor.update_physics_step(0.01)
    assert calls == []


def test_contact_sampler_normal_update_fetches_once_before_reading_data():
    from embodichain.lab.sim.atomic_skills.evidence import PhysicalEvidenceRequest
    from embodichain.lab.sim.sensors.contact.sampling import (
        ContactEvidenceSampler,
        CONTACT_VALID_MASK,
    )

    query = _FakeQuery()
    sensor = _sensor_for_query(query)
    calls = []
    original = query.fetch_solver_report

    def fetch_solver_report():
        calls.append("fetch_solver_report")
        return original()

    query.fetch_solver_report = fetch_solver_report
    sample = ContactEvidenceSampler(sensor).sample(
        PhysicalEvidenceRequest(0.0, 0, torch.tensor([1, 0]))
    )
    assert calls == ["fetch_solver_report"]
    assert sample.frame.get(CONTACT_VALID_MASK).values.sum().item() == 2
    assert all(batch.valid.all() for batch in sample.frame.batches.values())


def test_sampling_observer_exception_is_not_replaced_with_empty_success():
    sensor = _sensor_for_query(_FakeQuery())

    def fail(*args):
        raise RuntimeError("observer failed")

    observer = SimpleNamespace(
        begin_control_step=lambda: None,
        before_sample=lambda: None,
        after_sample=fail,
        reset=lambda ids: None,
    )
    sensor.add_sampling_observer("broken", observer)
    with pytest.raises(RuntimeError, match="observer failed"):
        sensor.update_physics_step(0.01)


def test_contact_cache_preserves_gym_and_standalone_refresh_after_reset():
    """Reset invalidates the cache, while empty reset and repeated reads do not."""
    query = _FakeQuery()
    sensor = _sensor_for_query(query)
    owner = sensor._sim
    sensors = {"contact": sensor}

    def advance(dt, count, after_substep=None, **kwargs):
        for _ in range(count):
            owner.physics_step_index += 1
            query.available = True
            if after_substep is not None:
                after_substep(dt)

    owner.update = advance
    owner.render_camera_group = lambda ids: None
    env = SimpleNamespace(
        sensors=sensors,
        sim=owner,
        physics_dt=0.001,
        cfg=SimpleNamespace(sim_steps_per_control=20),
        num_envs=2,
        device=torch.device("cpu"),
        _camera_group_ids=[],
        _profiler=SimpleNamespace(section=lambda *args: nullcontext()),
    )
    BaseEnv._get_sensor_obs(env)
    BaseEnv._get_sensor_obs(env)
    assert query.fetch_calls == 1
    BaseEnv._advance_physics(env)
    assert owner.physics_step_index == 20 and query.fetch_calls == 1
    BaseEnv._get_sensor_obs(env)
    sensor.update()
    assert query.fetch_calls == 2
    untouched = sensor.get_data()[1].clone()

    sensor.reset([0])
    assert not sensor.get_data()["is_valid"][0].any()
    for name, value in untouched.items():
        torch.testing.assert_close(sensor.get_data()[name][1], value)
    # Scene preparation clears sensor history after settling, then explicitly
    # refreshes raw evidence from the still-valid completed physics step.
    BaseEnv._get_sensor_obs(env)
    assert query.fetch_calls == 3 and owner.physics_step_index == 20
    assert sensor.get_data()["is_valid"][:, 0].all()
    sensor.reset([])
    sensor.update()
    assert query.fetch_calls == 3

    # Replay advances the manager directly rather than through env.step().
    owner.update(0.001, 1)
    BaseEnv._get_sensor_obs(env)
    assert query.fetch_calls == 4
    assert sensor.get_data()["is_valid"][:, 0].all()

    sensor.reset()
    assert not sensor.get_data()["is_valid"].any()
    sensor.update()
    assert query.fetch_calls == 5
    assert sensor.get_data()["is_valid"][:, 0].all()


def test_history_reuses_raw_sample_and_advances_only_once_per_physics_step():
    query = _FakeQuery()
    sensor = _sensor_for_query(query)
    history = sensor.create_history("feet", torch.tensor([[4], [7]]))
    sensor.begin_control_step()
    sensor._sim.physics_step_index += 1
    sensor.update()
    sensor.update_physics_step(0.01)
    sensor.update_physics_step(0.01)
    assert query.fetch_calls == 1
    assert history.contact_count.tolist() == [1, 1]


def test_failed_contact_query_does_not_poison_step_cache():
    query = _FakeQuery()
    sensor = _sensor_for_query(query)
    sensor._sim.physics_step_index += 1
    query.available = False
    with pytest.raises(AssertionError, match="snapshot is unavailable"):
        sensor.update()
    query.available = True
    sensor.update()
    assert query.fetch_calls == 1
    assert sensor.get_data()["is_valid"][:, 0].all()


def test_same_step_topology_change_refreshes_sample_and_actor_metadata():
    query = _FakeQuery()
    sensor = _sensor_for_query(query)
    sensor.update()
    assert sensor.total_current_contacts == 2
    query._actors[7] = ContactActorInfo(7, "arena_0/other", None, "arena_0", 0)
    query.actors = tuple(query._actors.values())
    query.buffer.count = 0  # A rebuilt model has no solved contacts yet.
    sensor.update()
    assert query.fetch_calls == 2
    assert sensor.total_current_contacts == 0
    assert not sensor.get_data()["is_valid"].any()
    assert sensor.item_env_ids.tolist() == [0, 0]


@pytest.mark.parametrize("raw_first", [False, True])
def test_sampling_observers_share_the_physics_step_cache_with_history(raw_first):
    query = _FakeQuery()
    sensor = _sensor_for_query(query)
    history = sensor.create_history("feet", torch.tensor([[4], [7]]))
    events = []
    sensor.add_sampling_observer(
        "audit",
        SimpleNamespace(
            begin_control_step=lambda: None,
            before_sample=lambda: events.append("before"),
            after_sample=lambda dt, token: events.append(("after", dt)),
            reset=lambda ids: None,
        ),
    )
    sensor.begin_control_step()
    for index in range(2):
        sensor._sim.physics_step_index += 1
        if raw_first:
            sensor.update()
        sensor.update_physics_step(0.01)
        sensor.update_physics_step(0.01)
        sensor.update()
        assert query.fetch_calls == index + 1
        assert events == ["before", ("after", 0.01)] * (index + 1)
    assert history.contact_count.tolist() == [2, 2]


def test_contact_package_exports_and_sensor_config_registration():
    from embodichain.lab.sim.sensors import SensorCfg
    from embodichain.lab.sim.sensors.contact import (
        ContactSensor as PackagedSensor,
        ContactSensorCfg as PackagedCfg,
    )
    from embodichain.lab.sim.sensors.contact.sensor import (
        ContactSensor as Implementation,
    )
    from importlib.util import find_spec

    assert ContactSensor is PackagedSensor is Implementation
    assert find_spec("embodichain.lab.sim.sensors.contact_sensor") is None
    assert find_spec("embodichain.lab.sim.sensors.contact_history") is None
    config = SensorCfg.from_dict(
        {"sensor_type": "ContactSensor", "uid": "contacts", "rigid_uid_list": ["cube"]}
    )
    assert type(config) is ContactSensorCfg is PackagedCfg


def test_disabled_observers_do_not_build_or_iterate_subscription_snapshots():
    class DisabledObservers(dict):
        def values(self):
            raise AssertionError("Disabled observer path must not enumerate callbacks.")

    query = _FakeQuery()
    sensor = _sensor_for_query(query)
    sensor._sampling_observers = DisabledObservers()
    history = sensor.create_history("feet", torch.tensor([[4], [7]]))
    sensor.begin_control_step()
    sensor._sim.physics_step_index += 1
    sensor.update_physics_step(0.01)
    sensor.update_physics_step(0.01)
    sensor.update()
    assert query.fetch_calls == 1
    assert history.contact_count.tolist() == [1, 1]


def test_observer_snapshot_survives_subscription_removal_during_sampling():
    query = _FakeQuery()
    sensor = _sensor_for_query(query)
    events = []

    def before():
        sensor.remove_sampling_observer("audit")
        return "prepared"

    sensor.add_sampling_observer(
        "audit",
        SimpleNamespace(
            begin_control_step=lambda: None,
            before_sample=before,
            after_sample=lambda dt, token: events.append((dt, token)),
            reset=lambda ids: None,
        ),
    )
    sensor.update_physics_step(0.01)
    assert events == [(0.01, "prepared")]
    sensor._sim.physics_step_index += 1
    sensor.update_physics_step(0.01)
    assert events == [(0.01, "prepared")]
    assert query.fetch_calls == 2


@pytest.mark.parametrize("loss", ["none", "query", "scatter"])
def test_first_standalone_sample_with_observer_preserves_loss_unavailability(loss):
    """The observer's interval counter has not yet seen this standalone fetch_solver_report."""
    from embodichain.lab.sim.atomic_skills.evidence import PhysicalEvidenceRequest
    from embodichain.lab.sim.sensors.contact.sampling import (
        ContactEvidenceSampler,
    )

    wp.init()
    query = _FakeQuery()
    sensor = _sensor_for_query(query, capacity=1)
    sensor.add_sampling_observer(
        "audit",
        SimpleNamespace(
            begin_control_step=lambda: None,
            before_sample=lambda: None,
            after_sample=lambda dt, token: None,
            reset=lambda ids: None,
        ),
    )
    if loss == "query":
        query.buffer.dropped_count = 3
    elif loss == "scatter":
        query.buffer.count = 3
        query.buffer.env_ids[:3] = torch.tensor([0, 0, 1])
        query.buffer.actor_ids[:3] = torch.tensor([[4, -1], [4, -1], [7, -1]])
        query.buffer.data[2] = query.buffer.data[1]
    fetches = []
    original_fetch = query.fetch_solver_report

    def fetch_solver_report():
        fetches.append(True)
        return original_fetch()

    query.fetch_solver_report = fetch_solver_report
    sample = ContactEvidenceSampler(sensor).sample(
        PhysicalEvidenceRequest(1.0, 0, torch.tensor([1, 0]))
    )
    assert fetches == [True]
    assert sensor.dropped_contacts == 0
    if loss == "none":
        assert all(batch.valid.all() for batch in sample.frame.batches.values())
    else:
        assert all(not batch.valid.any() for batch in sample.frame.batches.values())
        assert all(
            "capacity" in error
            for batch in sample.frame.batches.values()
            for error in batch.acquisition_errors
        )
        assert sample.dropped_contacts == (3 if loss == "query" else 1)


def test_latest_contact_loss_belongs_to_one_query_and_respects_selected_reset():
    wp.init()
    first_query, second_query = _FakeQuery(), _FakeQuery()
    first, second = _sensor_for_query(first_query), _sensor_for_query(second_query)
    first_query.buffer.dropped_count = 3
    first.update()
    second.update()
    assert first.latest_sample_dropped_contacts == 3
    assert second.latest_sample_dropped_contacts == 0
    first.reset([1])
    assert first.latest_sample_dropped_contacts == 3
    first.reset()
    assert first.latest_sample_dropped_contacts == 0
    first.update()
    assert first.latest_sample_dropped_contacts == 3
    first_query.buffer.dropped_count = 0
    first._sim.physics_step_index += 1
    first.update()
    assert first.latest_sample_dropped_contacts == 0
