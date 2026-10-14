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

from dataclasses import dataclass
from types import SimpleNamespace as NS

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program.support_evidence import (
    ContactSensor,
    NativeSupportObservation,
    SupportContactSensor,
)
from embodichain.lab.gym.envs.task_program.bridge import EnvironmentStepClock


@dataclass(frozen=True)
class _ActorInfo:
    actor_id: int
    path: str
    link_name: str | None
    arena_name: str
    env_id: int


@pytest.fixture
def support_pair(monkeypatch):
    count = 2
    sensor = object.__new__(SupportContactSensor)
    sensor._updates, sensor._fetches, sensor._dropped = 0, 0, 0
    data = {
        "user_ids": torch.tensor([[[1, 2]] * 4, [[1, 4]] * 4]),
        "position": torch.tensor(
            [[[-0.02, -0.02, 0], [-0.02, 0.02, 0], [0.02, -0.02, 0], [0.02, 0.02, 0]]]
            * count
        ),
        "normal": torch.tensor([[[0.0, 0.0, 1.0]] * 4] * count),
        "impulse": torch.full((count, 4), 0.001),
        "is_valid": torch.ones((count, 4), dtype=torch.bool),
    }
    info = {
        1: _ActorInfo(1, "/global/tray", None, "global", -1),
        2: _ActorInfo(2, "/arena0/item", None, "arena0", 0),
        4: _ActorInfo(4, "/arena1/item", None, "arena1", 1),
    }
    sim = NS(revision=0, spawn_result=NS(topology_revision=0, backend="dexsim"))
    physics = NS(_get_contact_data_revision=lambda: sim.revision)
    sim.get_world = lambda: NS(get_physics_scene=lambda: physics)
    sim.get_sensor = lambda uid: sensor
    capabilities = NS(geometry=True, impulse=True)
    monkeypatch.setattr(
        ContactSensor, "contact_capabilities", property(lambda s: capabilities)
    )
    monkeypatch.setattr(
        ContactSensor, "dropped_contacts", property(lambda s: s._dropped)
    )
    monkeypatch.setattr(ContactSensor, "get_data", lambda s: data)
    monkeypatch.setattr(ContactSensor, "get_actor_info", lambda s, actor: info[actor])
    monkeypatch.setattr(
        ContactSensor,
        "get_actor_ids",
        lambda s, uid: torch.tensor([[2], [4]] if uid == "item" else [[1], [1]]),
    )

    def update(s):
        s._updates += 1
        s._dropped = 0

    def substep(s, dt):
        s._fetches += 1

    def reset(s, env_ids):
        data["is_valid"][slice(None) if env_ids is None else env_ids] = False

    monkeypatch.setattr(ContactSensor, "update", update)
    monkeypatch.setattr(ContactSensor, "update_physics_step", substep)
    monkeypatch.setattr(
        ContactSensor, "begin_control_step", lambda s: setattr(s, "_dropped", 0)
    )
    monkeypatch.setattr(ContactSensor, "reset", reset)
    pose = torch.eye(4).repeat(count, 1, 1)
    object_pose = pose.clone()
    object_pose[:, 2, 3] = 0.1

    def body(uid, matrix):
        return NS(
            uid=uid,
            num_instances=count,
            _entities=[NS(get_physical_attr=lambda: NS(rest_offset=0.001))] * count,
            body_data=NS(com_pose=torch.zeros((count, 7))),
            get_local_pose=lambda to_matrix: matrix,
        )

    obj, ref = body("item", object_pose), body("tray", pose)
    reader = NativeSupportObservation(
        sim, obj, ref, entity_id="item", reference_id="tray"
    )
    clock = EnvironmentStepClock(0.04)
    reader.bind_clock(clock)

    def step(*, valid=True, new_revision=True):
        sensor.begin_control_step()
        data["is_valid"][:] = valid
        for _ in range(2):
            sim.revision += int(new_revision)
            sensor.update_physics_step(0.02)
        clock.advance_after_env_step()

    def observe():
        return reader.observe(object_pose, pose, support_contract_revision=1)

    return NS(
        reader=reader,
        sensor=sensor,
        sim=sim,
        physics=physics,
        capabilities=capabilities,
        data=data,
        info=info,
        object_pose=object_pose,
        reference_pose=pose,
        clock=clock,
        step=step,
        observe=observe,
        obj=obj,
        ref=ref,
    )


def test_clocked_observe_twice_never_fetches_or_recaptures(support_pair, monkeypatch):
    pair = support_pair
    pair.step()
    monkeypatch.setattr(
        pair.reader,
        "capture",
        lambda *a, **k: pytest.fail("observation captured contacts"),
    )
    first, receipt = pair.observe()
    second, again = pair.observe()
    assert first.tolist() == second.tolist() == [True, True]
    assert pair.sensor._updates == 0 and pair.sensor._fetches == 2
    assert receipt["rows"][0]["sample"] == again["rows"][0]["sample"]
    assert receipt["rows"][0]["sample"]["timestamp"] == pytest.approx(0.04)
    assert receipt["actor_pairs"][0]["reference_info"]["env_id"] == -1


def test_clocked_observation_cannot_clear_prior_interval_overflow(support_pair):
    pair = support_pair
    pair.step()
    pair.sensor._dropped = 1
    for _ in range(2):
        with pytest.raises(ValueError):
            pair.observe()
        assert pair.sensor._dropped == 1
    assert pair.sensor._updates == 0
    assert all(witness.value is None for witness in pair.reader.witnesses)


@pytest.mark.parametrize("new_revision", [False, True])
def test_sleeping_retention_preserves_original_native_sample_time(
    support_pair, new_revision
):
    pair = support_pair
    pair.step()
    _, original = pair.observe()
    pair.step(valid=False, new_revision=new_revision)
    supported, retained = pair.observe()
    assert supported.tolist() == [True, True]
    assert retained["rows"][0]["sample"] == original["rows"][0]["sample"]
    assert retained["rows"][0]["retained"] is True
    assert retained["observed_at"] > retained["rows"][0]["sample"]["timestamp"]


def test_repeated_native_revision_cannot_retimestamp_positive_rows(support_pair):
    pair = support_pair
    pair.step()
    _, original = pair.observe()
    pair.step(valid=True, new_revision=False)
    _, retained = pair.observe()
    assert retained["rows"][0]["sample"] == original["rows"][0]["sample"]
    assert retained["rows"][0]["retained"] is True


def test_reset_clears_only_selected_generation_and_blocks_cached_snapshot(support_pair):
    pair = support_pair
    pair.step()
    _, original = pair.observe()
    pair.sensor.reset([0])
    pair.step(valid=True, new_revision=False)
    supported, receipt = pair.observe()
    assert supported.tolist() == [False, True]
    assert receipt["rows"][0]["sample"] is None
    assert receipt["episode_generations"][0] == original["episode_generations"][0] + 1
    assert receipt["episode_generations"][1] == original["episode_generations"][1]
    assert receipt["rows"][1]["sample"] == original["rows"][1]["sample"]


def test_new_window_does_not_promote_previous_load_sample(support_pair):
    pair = support_pair
    pair.step()
    pair.step(valid=False)
    release_started_at = pair.clock.now()
    pair.step(valid=False)
    _, receipt = pair.observe()
    # Consumer must reject this receipt for the new release window.
    assert receipt["rows"][0]["sample"]["timestamp"] < release_started_at
    assert receipt["rows"][0]["sample"]["control_step"] < receipt["observed_step"]


@pytest.mark.parametrize(
    "change", ["object", "reference", "clock", "topology", "actor"]
)
def test_changed_configuration_never_revives_cached_patch(support_pair, change):
    pair = support_pair
    pair.step()
    if change in {"object", "reference"}:
        pose = pair.object_pose if change == "object" else pair.reference_pose
        pose[:, 0, 3] += 0.01
        pair.step(new_revision=False)
        pose[:, 0, 3] -= 0.01
    elif change == "clock":
        pair.reader.bind_clock(
            EnvironmentStepClock(0.04, initial_step=pair.clock.step_index)
        )
        with pytest.raises(ValueError, match="unobserved step"):
            pair.observe()
        return
    elif change == "topology":
        pair.sim.spawn_result.topology_revision += 1
    else:
        pair.info[2] = _ActorInfo(2, "/replacement/item", None, "arena0", 0)
    supported, _ = pair.observe()
    assert supported[0].item() is False


@pytest.mark.parametrize(
    "change", ["geometry", "impulse", "unknown", "nonfinite", "overflow", "revision"]
)
def test_unqualified_clocked_native_sources_fail_closed(support_pair, change):
    pair = support_pair
    pair.step()
    if change in {"geometry", "impulse"}:
        setattr(pair.capabilities, change, False)
    elif change == "unknown":
        pair.data["user_ids"][0, 0, 0] = -1
    elif change == "nonfinite":
        pair.data["normal"][0, 0, 0] = float("nan")
    elif change == "revision":
        del pair.physics._get_contact_data_revision
    if change == "overflow":
        pair.sensor._dropped = 1
    else:
        pair.step(new_revision=False)
    with pytest.raises(ValueError):
        pair.observe()


def test_alias_observer_reads_the_canonical_owner(support_pair):
    pair = support_pair
    pair.step()
    alias = NativeSupportObservation(
        pair.sim, pair.obj, pair.ref, entity_id="item", reference_id="tray"
    )
    _, original = pair.observe()
    supported, receipt = alias.observe(
        pair.object_pose, pair.reference_pose, support_contract_revision=1
    )
    assert supported.tolist() == [True, True]
    assert receipt["rows"] == original["rows"]
    assert pair.sensor.support_reader is pair.reader


def test_legacy_observation_keeps_its_standalone_update_behavior(support_pair):
    pair = support_pair
    pair.reader.bind_clock(None)
    supported, receipt = pair.reader.observe(pair.object_pose, pair.reference_pose)
    assert supported.tolist() == [True, True]
    assert pair.sensor._updates == 1
    assert "support_contract_revision" not in receipt


def test_legacy_read_of_clocked_pair_does_not_poison_strong_receipt(support_pair):
    pair = support_pair
    pair.step()
    _, before = pair.observe()
    pair.reader.observe(pair.object_pose, pair.reference_pose)
    _, after = pair.observe()
    assert before["rows"] == after["rows"]
    assert pair.sensor._updates == 0


@pytest.mark.parametrize("partial", [False, True])
def test_clock_ack_without_complete_sensor_interval_is_unavailable(
    support_pair, partial
):
    pair = support_pair
    pair.step()
    if partial:
        pair.sensor.begin_control_step()
        pair.sim.revision += 1
        pair.sensor.update_physics_step(0.02)
    pair.clock.advance_after_env_step()
    with pytest.raises(ValueError, match="unobserved step"):
        pair.observe()
