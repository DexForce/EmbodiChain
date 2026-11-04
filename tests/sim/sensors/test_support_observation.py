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

"""CPU native-support contracts on the real ContactSensor observer lifecycle."""

from __future__ import annotations

from copy import deepcopy
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

from dexsim.scene import ContactActorInfo, ContactBuffer, ContactQueryCapabilities
from embodichain.lab.gym.envs.task_program.bridge import EnvironmentStepClock
from embodichain.lab.gym.envs.base_env import BaseEnv
from embodichain.lab.sim.sensors.contact.sensor import ContactSensor, ContactSensorCfg
from embodichain.lab.sim.sensors.support_observation import (
    ContactWitness,
    NativeSupportObservation,
    NativeSupportResult,
    SupportObservationRegistry,
)


class _Body:
    def __init__(self, uid: str, count: int, rest_offset: float = 0.001) -> None:
        self.uid, self.num_instances = uid, count
        self.pose = torch.eye(4).repeat(count, 1, 1)
        self.body_data = SimpleNamespace(com_pose=torch.zeros(count, 7))
        self.body_data.com_pose[:, 2] = 0.1
        self._entities = [
            SimpleNamespace(
                get_physical_attr=lambda: SimpleNamespace(rest_offset=rest_offset)
            )
            for _ in range(count)
        ]
        self.pose_reads = 0

    def get_local_pose(self, *, to_matrix):
        assert to_matrix
        self.pose_reads += 1
        return self.pose.clone()

    def get_vertices(self, *, scale=True):
        import trimesh

        return torch.tensor(
            trimesh.creation.box(extents=[0.2, 0.2, 0.2]).vertices, dtype=torch.float32
        ).repeat(self.num_instances, 1, 1)

    def get_triangles(self):
        import trimesh

        return torch.tensor(
            trimesh.creation.box(extents=[0.2, 0.2, 0.2]).faces, dtype=torch.int64
        ).repeat(self.num_instances, 1, 1)


class _Query:
    def __init__(self, count: int) -> None:
        self.capabilities = ContactQueryCapabilities(True, True, True)
        self.actors = tuple(
            ContactActorInfo(
                10 * row + offset, f"arena_{row}/{uid}", None, f"arena_{row}", row
            )
            for row in range(count)
            for offset, uid in ((1, "payload"), (2, "support"))
        )
        self.selected_actor_ids = tuple(actor.actor_id for actor in self.actors)
        self.buffer = ContactBuffer.allocate(4 * count, "cpu")
        self.buffer.data.zero_()
        self.buffer.count = 4 * count
        patch = torch.tensor(
            [[-0.02, -0.02, 0], [-0.02, 0.02, 0], [0.02, -0.02, 0], [0.02, 0.02, 0]]
        )
        for row in range(count):
            start = 4 * row
            self.buffer.data[start : start + 4, :3] = patch
            self.buffer.data[start : start + 4, 5] = 1.0
            self.buffer.data[start : start + 4, 9] = 0.001
            self.buffer.actor_ids[start : start + 4] = torch.tensor(
                [10 * row + 2, 10 * row + 1]
            )
            self.buffer.env_ids[start : start + 4] = row
        self.fetches = 0
        self.on_fetch = lambda: None

    def actor_info(self, actor_id):
        return next(actor for actor in self.actors if actor.actor_id == actor_id)

    def fetch(self):
        self.fetches += 1
        self.on_fetch()
        return self.buffer


class _Simulation:
    def __init__(self, count=2) -> None:
        self.num_envs = count
        self.arena_offsets = torch.zeros(count, 3)
        self.objects = {uid: _Body(uid, count) for uid in ("payload", "support")}
        self.sensors = {}
        self.query = _Query(count)
        self.time = 0.0
        self.revision = 1
        self.sensor_creations = 0
        self.spawn_result = SimpleNamespace(
            backend="dexsim",
            topology_revision=1,
            create_contact_query=lambda *args, **kwargs: self.query,
        )
        self._spawn_scene = SimpleNamespace(
            handles=lambda uid: tuple(
                SimpleNamespace(path=f"arena_{row}/{uid}") for row in range(count)
            )
        )

    @property
    def physics_step_index(self):
        return self.revision

    def get_world(self):
        physics = SimpleNamespace(_get_contact_data_revision=lambda: self.revision)
        return SimpleNamespace(get_physics_scene=lambda: physics)

    def get_rigid_object(self, uid):
        return self.objects.get(uid)

    def get_sensor(self, uid):
        return self.sensors.get(uid)

    def add_sensor(self, cfg):
        self.sensor_creations += 1
        sensor = ContactSensor(deepcopy(cfg), owner=self)
        self.sensors[cfg.uid] = sensor
        return sensor


def _setup(*, strong=False, count=2, capacity=4):
    wp.init()
    simulation = _Simulation(count)
    registry = SupportObservationRegistry(
        simulation, acquisition_clock=lambda: simulation.time
    )
    cfg = ContactSensorCfg(
        uid="declared_support",
        rigid_uid_list=["payload", "support"],
        max_contacts_per_env=capacity,
    )
    observer = registry.acquire(
        simulation.objects["payload"],
        simulation.objects["support"],
        entity_id="item",
        reference_id="surface",
        observer_name="support_observation",
        sensor_cfg=cfg,
    )
    clock = EnvironmentStepClock(0.04)
    if strong:
        observer.bind_clock(clock)
    return simulation, registry, observer, clock, cfg


def _substep(simulation, observer, dt=0.01):
    simulation.revision += 1
    simulation.time += dt
    observer.sensor.update_physics_step(dt)


def _window(simulation, observer, clock):
    observer.sensor.begin_control_step()
    for _ in range(4):
        _substep(simulation, observer)
    clock.advance_after_env_step()


def _observe(observer, revision=None):
    return observer.observe(
        observer.object.pose.clone(),
        observer.reference.pose.clone(),
        support_contract_revision=revision,
    )


def test_legacy_support_owns_metadata_and_uses_actual_acquisition_clock():
    sim, _, observer, _, _ = _setup()
    sim.time = 7.25
    result = _observe(observer)
    assert result.support.values.tolist() == [True, True]
    assert result.support.valid.tolist() == [True, True]
    assert result.support.timestamp == 7.25
    assert result.support.observation_revision == 1
    assert observer.tolerance == pytest.approx(0.003)
    assert set(result.metadata) == {
        "scope",
        "entity",
        "reference",
        "object_pose",
        "reference_pose",
        "supported",
        "env_ids",
        "rows",
    }
    assert result.metadata["rows"][0]["com_patch_distance_m"] == 0.0
    assert result.metadata["rows"][0]["upward_impulse"] == pytest.approx(0.004)
    result.support.values.zero_()
    result.metadata["rows"][0]["contacts_world"][0][0] = 100.0
    again = _observe(observer)
    assert again.support.values.all()
    assert again.metadata["rows"][0]["contacts_world"][0][0] == pytest.approx(-0.02)
    assert (
        sim.query.fetches == 1
    )  # Same completed physics step reuses its contact snapshot.


@pytest.mark.parametrize("moving", ["payload", "support"])
def test_positive_witness_survives_sleep_but_not_either_body_motion(moving):
    sim, _, observer, _, _ = _setup()
    assert _observe(observer).support.values.all()
    sim.query.buffer.count = 0
    assert _observe(observer).support.values.all()
    sim.objects[moving].pose[:, 0, 3] = 0.000099
    assert _observe(observer).support.values.all()
    sim.objects[moving].pose[:, 0, 3] = 0.000101
    result = _observe(observer)
    assert result.support.valid.all()
    assert not result.support.values.any()
    sim.query.buffer.count = 4 * sim.num_envs
    sim.revision += 1
    assert _observe(observer).support.values.all()
    assert (
        sim.query.fetches == 2
    )  # New completed physics step can establish fresh support.


def test_strong_support_requires_clock_complete_window_and_positive_witness():
    sim, _, observer, clock, _ = _setup()
    with pytest.raises(ValueError, match="execution clock"):
        _observe(observer, 1)
    assert sim.query.fetches == 0
    observer.bind_clock(clock)
    _window(sim, observer, clock)
    result = _observe(observer, 1)
    assert result.support.values.all() and result.support.valid.all()
    assert sim.query.fetches == 4
    assert result.metadata["support_contract_revision"] == 1
    assert result.metadata["observed_step"] == 1
    assert result.metadata["observed_at"] == 0.04
    info = deepcopy(result.metadata["rows"][0]["sample"])
    assert info == dict(
        timestamp=0.04,
        control_step=1,
        substep=4,
        serial=4,
        episode_generation=1,
        native_revision=5,
        topology_revision=1,
    )
    result.metadata["rows"][0]["sample"]["serial"] = 999
    result.metadata["actor_pairs"][0]["object_info"]["actor_id"] = -1
    again = _observe(observer, 1)
    assert again.support.valid.all()
    assert again.metadata["rows"][0]["sample"] == info
    assert again.metadata["actor_pairs"][0]["object_info"]["actor_id"] == 1
    legacy = _observe(observer)
    assert legacy.support.values.all()
    assert "support_contract_revision" not in legacy.metadata
    assert sim.query.fetches == 4
    sim.query.buffer.count = 0
    _window(sim, observer, clock)
    sleeping = _observe(observer, 1)
    assert sleeping.support.values.all() and sleeping.support.valid.all()
    assert sleeping.metadata["rows"][0]["retained"] is True
    assert sleeping.metadata["rows"][0]["sample"] == info


def test_public_support_result_constructor_copies_mutable_external_inputs():
    _, _, observer, _, _ = _setup()
    source = _observe(observer)
    result = NativeSupportResult(source.support, source.metadata)
    source.support.values.zero_()
    source.metadata["rows"][0]["contacts_world"][0][0] = 100.0
    assert result.support.values.all()
    assert result.metadata["rows"][0]["contacts_world"][0][0] == pytest.approx(-0.02)
    source.support.env_ids[0] = source.support.env_ids[1]
    with pytest.raises(ValueError, match="unique"):
        NativeSupportResult(source.support, source.metadata)


def test_complete_empty_contact_sample_is_only_an_instantaneous_negative():
    sim, _, observer, clock, _ = _setup(strong=True)
    sim.query.buffer.count = 0
    _window(sim, observer, clock)
    result = _observe(observer, 1)
    assert result.support.valid.all()
    assert not result.support.values.any()
    assert all(row["sample"] is None for row in result.metadata["rows"])


def test_binding_strong_clock_keeps_source_serial_separate_from_all_acquisitions():
    sim, _, observer, clock, _ = _setup()
    assert _observe(observer).support.observation_revision == 1
    observer.bind_clock(clock)
    _window(sim, observer, clock)
    result = _observe(observer, 1)
    assert result.support.observation_revision == 5
    assert result.metadata["rows"][0]["sample"]["serial"] == 4


def test_strong_time_domain_tracks_an_explicit_warmup_offset():
    sim, _, observer, clock, _ = _setup()
    sim.time = 7.25
    legacy = _observe(observer)
    assert legacy.support.timestamp == 7.25
    observer.bind_clock(clock)
    _window(sim, observer, clock)
    strong = _observe(observer, 1)
    assert strong.support.valid.all() and strong.support.values.all()
    assert strong.support.timestamp == 0.04
    assert strong.metadata["observed_at"] == 0.04
    assert strong.metadata["rows"][0]["sample"]["timestamp"] == 0.04
    assert observer._last_sample.frame.timestamp == 0.04
    assert _observe(observer).support.timestamp == pytest.approx(7.29)


def test_clock_mapping_rejects_unobserved_time_between_complete_windows():
    sim, _, observer, clock, _ = _setup(strong=True)
    _window(sim, observer, clock)
    assert _observe(observer, 1).support.values.all()
    sim.time += 0.01
    _window(sim, observer, clock)
    result = _observe(observer, 1)
    assert not result.support.valid.any()
    assert all("clock mapping" in error for error in result.support.acquisition_errors)


@pytest.mark.parametrize("substeps", [2, 5])
def test_partial_or_overlong_window_is_unavailable(substeps):
    sim, _, observer, clock, _ = _setup(strong=True)
    observer.sensor.begin_control_step()
    for _ in range(substeps):
        _substep(sim, observer)
    clock.advance_after_env_step()
    result = _observe(observer, 1)
    assert not result.support.valid.any()
    assert all(result.support.acquisition_errors)


def test_missing_acquisition_time_never_becomes_a_fabricated_zero_timestamp():
    sim, _, observer, _, _ = _setup()
    sim.time = float("nan")
    with pytest.raises(ValueError, match="acquisition time"):
        _observe(observer)


def test_public_base_env_step_samples_support_once_per_physics_substep():
    sim, _, observer, clock, _ = _setup(strong=True, count=1)
    env = BaseEnv.__new__(BaseEnv)
    env.cfg = SimpleNamespace(sim_steps_per_control=4)
    env.sim_cfg = SimpleNamespace(physics_dt=0.01)
    env.sensors = {observer.sensor.cfg.uid: observer.sensor}

    def update(dt, steps, **kwargs):
        for _ in range(steps):
            sim.revision += 1
            sim.time += dt
            kwargs["after_substep"](dt)

    env.sim = SimpleNamespace(update=update, render_frame=lambda: nullcontext())
    env._profiler = SimpleNamespace(section=lambda *args, **kwargs: nullcontext())
    env._elapsed_steps = torch.zeros(1, dtype=torch.long)
    for name in ("_preprocess_action", "_step_action", "_postprocess_action"):
        setattr(env, name, lambda action: action)
    env._update_sim_state = lambda **kwargs: None
    env._update_physical_objective = lambda: None
    env.get_obs = lambda **kwargs: {}
    env.get_info = lambda **kwargs: {}
    env.get_reward = lambda **kwargs: torch.zeros(1)
    env._extend_reward = lambda rewards, **kwargs: rewards
    env._get_termination_flags = lambda *args: (
        torch.zeros(1, dtype=torch.bool),
        torch.zeros(1, dtype=torch.bool),
    )
    env._hook_after_sim_step = lambda **kwargs: None
    env.step(None)
    clock.advance_after_env_step()
    result = _observe(observer, 1)
    assert result.support.valid.tolist() == result.support.values.tolist() == [True]
    assert sim.query.fetches == 4
    assert result.metadata["rows"][0]["sample"]["substep"] == 4


@pytest.mark.parametrize(
    "damage",
    ["nonfinite_row", "overflow", "fetch_revision", "topology", "unobserved_step"],
)
def test_acquisition_failure_invalidates_every_witness(damage):
    sim, _, observer, clock, _ = _setup(strong=True)
    _window(sim, observer, clock)
    assert _observe(observer, 1).support.values.all()
    if damage == "nonfinite_row":
        sim.query.buffer.data[0, 5] = float("nan")
    elif damage == "overflow":
        sim.query.buffer.dropped_count = 1
    elif damage == "fetch_revision":
        sim.query.on_fetch = lambda: setattr(sim, "revision", sim.revision + 1)
    elif damage == "topology":
        sim.query.on_fetch = lambda: setattr(
            sim.spawn_result,
            "topology_revision",
            sim.spawn_result.topology_revision + 1,
        )
    if damage == "unobserved_step":
        sim.revision += 1
    else:
        _window(sim, observer, clock)
    result = _observe(observer, 1)
    assert not result.support.valid.any()
    assert all(result.support.acquisition_errors)
    sim.query.buffer.data[0, 5] = 1.0
    sim.query.buffer.count = 0
    sim.query.buffer.dropped_count = 0
    sim.query.on_fetch = lambda: None
    _window(sim, observer, clock)
    clean_empty = _observe(observer, 1)
    assert clean_empty.support.valid.all()
    assert not clean_empty.support.values.any()


def test_partial_reset_requires_the_next_complete_window_for_selected_rows():
    sim, _, observer, clock, _ = _setup(strong=True)
    observer.sensor.begin_control_step()
    for _ in range(2):
        _substep(sim, observer)
    observer.sensor.reset([0, 0])
    for _ in range(2):
        _substep(sim, observer)
    clock.advance_after_env_step()
    result = _observe(observer, 1)
    assert result.support.valid.tolist() == [False, True]
    assert result.metadata["episode_generations"] == [2, 1]
    _window(sim, observer, clock)
    next_result = _observe(observer, 1)
    assert next_result.support.valid.all() and next_result.support.values.all()


def test_registry_reuses_exact_owners_and_requires_explicit_native_rebinding():
    sim, registry, observer, _, cfg = _setup()
    arguments = dict(
        entity_id="item",
        reference_id="surface",
        observer_name="support_observation",
        sensor_cfg=cfg,
    )
    assert (
        registry.acquire(observer.object, observer.reference, **arguments) is observer
    )
    assert sim.sensor_creations == 1
    replacement = _Body("payload", 2)
    sim.objects["payload"] = replacement
    with pytest.raises(ValueError, match="rebind"):
        registry.acquire(replacement, observer.reference, **arguments)
    assert not _observe(observer).support.valid.any()
    registry.release("payload", "support")
    new = registry.acquire(replacement, observer.reference, **arguments)
    assert new is not observer
    assert sim.sensor_creations == 1
    registry.close()
    assert sim.get_sensor(cfg.uid) is observer.sensor
    assert sim.get_rigid_object("payload") is replacement
    assert not observer.sensor.requires_substep_update


def test_registry_does_not_silently_reuse_a_changed_sensor_declaration():
    _, registry, observer, _, cfg = _setup()
    cfg.max_contacts_per_env = 8
    assert observer.sensor.cfg.max_contacts_per_env == 4
    with pytest.raises(ValueError, match="declaration changed"):
        registry.acquire(
            observer.object,
            observer.reference,
            entity_id="item",
            reference_id="surface",
            observer_name="support_observation",
            sensor_cfg=cfg,
        )
    assert _observe(observer).support.values.all()


def test_close_cannot_remove_a_replacement_hook_and_is_idempotent():
    _, _, observer, _, _ = _setup()
    sensor = observer.sensor
    sensor.remove_sampling_observer("support_observation", expected=observer)
    replacement = SimpleNamespace(
        begin_control_step=lambda: None,
        before_sample=lambda: None,
        after_sample=lambda dt, token: None,
        reset=lambda ids: None,
    )
    sensor.add_sampling_observer("support_observation", replacement)
    with pytest.raises(ValueError, match="different owner"):
        sensor.remove_sampling_observer("support_observation", expected=observer)
    observer.close()
    observer.close()
    assert sensor.remove_sampling_observer("support_observation") is replacement


def test_delayed_after_sample_from_sensor_token_snapshot_does_not_revive_closed_owner():
    sim, _, observer, _, _ = _setup()
    sensor = observer.sensor
    sensor.remove_sampling_observer("support_observation", expected=observer)
    closer = SimpleNamespace(
        begin_control_step=lambda: None,
        before_sample=lambda: None,
        after_sample=lambda dt, token: observer.close(),
        reset=lambda ids: None,
    )
    sensor.add_sampling_observer("closer", closer)
    sensor.add_sampling_observer("support_observation", observer)
    _substep(sim, observer)
    assert observer.object.pose_reads == observer.reference.pose_reads == 0
    assert not _observe(observer).support.valid.any()
    sensor.remove_sampling_observer("closer")


def test_native_rest_envelope_bound_is_checked_without_installing_hook():
    sim, registry, observer, _, cfg = _setup()
    registry.release("payload", "support")
    sim.objects["payload"] = _Body("payload", 2, rest_offset=0.01)
    with pytest.raises(ValueError, match="rest envelopes"):
        registry.acquire(
            sim.objects["payload"],
            sim.objects["support"],
            entity_id="item",
            reference_id="surface",
            observer_name="support_observation",
            sensor_cfg=cfg,
        )
    assert not observer.sensor.requires_substep_update


def test_contact_witness_owns_patch_and_pose_snapshot():
    pose = torch.eye(4)
    patch = (
        np.array([[1, 2]]),
        np.array([[0.0, 0, 0]]),
        np.array([[0.0, 0, 1]]),
        np.array([0.001]),
    )
    witness = ContactWitness()
    witness.update(pose, pose, patch)
    patch[1][0, 0] = 1.0
    assert witness.update(pose, pose, None)[1][0, 0] == 0.0
    assert witness.update(pose, pose, None, invalid=True) is None


def test_support_registry_close_does_not_allow_reassembly_of_same_native_pair():
    _, registry, observer, _, cfg = _setup()
    registry.close()
    registry.close()
    with pytest.raises(RuntimeError, match="closed"):
        registry.acquire(
            observer.object,
            observer.reference,
            entity_id="item",
            reference_id="surface",
            observer_name="support_observation",
            sensor_cfg=cfg,
        )


@pytest.mark.parametrize(
    "kind,startup",
    [
        ("stack", "complete"),
        ("supported_placement", "complete"),
        ("supported_region", "complete"),
        ("stack", "cold"),
        ("stack", "row_reset"),
        ("stack", "clock_drift"),
    ],
)
def test_standard_condition_port_consumes_qualified_support_on_the_original_grid(
    kind, startup
):
    from embodichain.lab.task_program import decode_task_program, TaskProgramCompiler
    from embodichain.lab.gym.envs.task_program.bridge import EnvironmentStepClock
    from embodichain.lab.sim.sensors.support_observation import (
        SupportObservationRegistry,
    )
    from embodichain.lab.task_program.integrations.simulation.bindings import (
        SimulationSceneBinding,
        SimulationRigidObjectBinding,
    )
    from embodichain.lab.task_program.integrations.simulation.policies import (
        SimulationSegmentPolicyPort,
    )
    from embodichain.lab.task_program.integrations.simulation.stability import (
        SimulationStabilityConditionCfg,
        SupportSensorDeclarationCfg,
    )

    wp.init()
    sim = _Simulation(2)
    sim.objects["payload"].pose[:, 2, 3] = 0.2
    sim.query.buffer.data[:, 2] = 0.1
    binding = SimulationSceneBinding(
        registry_id="support_scene",
        rigid_objects=(
            SimulationRigidObjectBinding(entity_id="item", simulation_uid="payload"),
            SimulationRigidObjectBinding(entity_id="surface", simulation_uid="support"),
        ),
    )
    cfg = SimulationStabilityConditionCfg(
        entity="item",
        kind=kind,
        reference="surface",
        reference_half_extents=(0.1, 0.1),
        displacement=None if kind == "stack" else (0.0, 0.0, 0.2),
        object_bottom=-0.1,
        reference_top=0.1,
        local_axis=(0, 0, 1),
        reference_axis=(0, 0, 1),
        duration=0.08,
        timeout=0.24,
        support_contract_revision=1,
        support_sensor=SupportSensorDeclarationCfg(
            uid="declared_support", max_contacts_per_env=4
        ),
    )
    robot = SimpleNamespace(get_qpos=lambda target=False: torch.zeros(2, 1))
    registry = SupportObservationRegistry(sim, acquisition_clock=lambda: sim.time)
    port = SimulationSegmentPolicyPort(
        sim,
        robot,
        binding,
        settle_presets={"held_surface": cfg},
        step_dt=0.04,
        support_registry=registry,
        sensor_suite=sim.sensors,
    )
    compiled = TaskProgramCompiler(binding.declare()).compile(
        decode_task_program(
            {
                "program_id": "support_policy",
                "targets": {},
                "integration": {
                    "robot_profile": "robot",
                    "scene_registry": "support_scene",
                    "runtime_preset": "default",
                },
                "program": {
                    "kind": "segment",
                    "name": "support",
                    "steps": {
                        "kind": "invoke",
                        "call": {"kind": "pick", "object": "item"},
                    },
                    "post": [
                        {
                            "kind": "wait_stable",
                            "entity": "item",
                            "preset": "held_surface",
                        }
                    ],
                },
            }
        )
    )
    segment = next(compiled.iter_segments())
    policy = segment.post_policies[0]
    clock = EnvironmentStepClock(0.04)
    port.bind_clock(clock)
    sensor = sim.get_sensor("declared_support")

    def step():
        sensor.begin_control_step()
        for _ in range(4):
            sim.revision += 1
            sim.time += 0.01
            sensor.update_physics_step(0.01)
        clock.advance_after_env_step()

    if startup != "cold":
        step()  # Last complete action interval precedes the post-policy.
    if startup == "clock_drift":
        sim.time += 0.005  # A source-time jump cannot silently rebase the mapping.
    commands = 0
    for command in port.actions(
        policy, segment=segment, active_mask=torch.ones(2, dtype=torch.bool)
    ):
        assert command.shape == (2, 1)
        commands += 1
        if startup == "cold" and commands == 1:
            initial = port.post_policy_metadata(policy, segment=segment)
            acquisition = initial["measurements"]["support_acquisition"]
            assert acquisition["valid"] == [False, False]
            assert all(acquisition["acquisition_errors"])
            assert acquisition["timestamp"] == 0.0
            assert acquisition["env_ids"] == [0, 1]
            assert initial["stable_seconds"] == [0.0, 0.0]
        if startup == "clock_drift" and commands == 1:
            missing = port.post_policy_metadata(policy, segment=segment)
            assert all(
                "clock mapping" in error
                for error in missing["measurements"]["support_acquisition"][
                    "acquisition_errors"
                ]
            )
        if startup == "row_reset" and commands == 2:
            partial = port.post_policy_metadata(policy, segment=segment)
            acquisition = partial["measurements"]["support_acquisition"]
            assert acquisition["valid"] == [False, True]
            assert acquisition["acquisition_errors"][0] is not None
            assert acquisition["acquisition_errors"][1] is None
            assert partial["stable_seconds"] == [0.0, 0.04]
        if startup == "row_reset" and commands == 3:
            assert port.post_policy_result(policy, segment=segment).tolist() == [
                False,
                True,
            ]
        step()
        if startup == "row_reset" and commands == 1:
            sensor.reset(torch.tensor([0]))
    if startup == "clock_drift":
        assert commands == 6
        assert not port.post_policy_result(policy, segment=segment).any()
        metadata = port.post_policy_metadata(policy, segment=segment)
        assert metadata["support_result_available"] == [False, False]
        assert metadata["support_unavailable_observations"] == [7, 7]
        assert all(
            "clock" in error for error in metadata["support_last_acquisition_errors"]
        )
        assert metadata["measurements"]["support_acquisition"]["valid"] == [
            False,
            False,
        ]
        assert metadata["stable_seconds"] == [0.0, 0.0]
        assert (
            sim.query.fetches == 1
        )  # Same completed physics step reuses its contact snapshot.8
        registry.close()
        return
    assert commands == (3 if startup == "row_reset" else 2)
    assert port.post_policy_result(policy, segment=segment).tolist() == [True, True]
    metadata = port.post_policy_metadata(policy, segment=segment)
    assert metadata["clock_started_at"] == (0.0 if startup == "cold" else 0.04)
    assert metadata["clock_ended_at"] == pytest.approx(
        0.08 if startup == "cold" else 0.16 if startup == "row_reset" else 0.12
    )
    assert metadata["stable_seconds"] == pytest.approx(
        [0.08, 0.12] if startup == "row_reset" else [0.08, 0.08]
    )
    if startup != "complete":
        assert metadata["support_unavailable_observations"] == (
            [1, 1] if startup == "cold" else [1, 0]
        )
        assert metadata["support_last_acquisition_errors"][0] is not None
    assert metadata["support_result_available"] == [True, True]
    assert metadata["measurements"]["native_support"]["support_contract_revision"] == 1
    assert sim.query.fetches == (
        8 if startup == "cold" else 16 if startup == "row_reset" else 12
    )
    if kind == "supported_region":
        native = metadata["measurements"]["native_support"]
        # Historical projection consumes the captured poses and original load,
        # not a new contact read or a live pose after the boundary.
        before_fetches = sim.query.fetches

        def no_live_pose(**kwargs):
            raise AssertionError("historical geometry reread a live pose")

        for body in sim.objects.values():
            body.get_local_pose = no_live_pose
        measured = port.measure_support_geometry(
            "held_surface",
            native_support=native,
            object_pose=native["object_pose"],
            reference_pose=native["reference_pose"],
        )
        assert measured["landing_region"] == metadata["measurements"]["landing_region"]
        assert measured["support_gap_m"] == metadata["measurements"]["support_gap"]
        assert sim.query.fetches == before_fetches
        changed = deepcopy(native["object_pose"])
        changed[0][0][3] += 0.02
        other = port.measure_support_geometry(
            "held_surface",
            native_support=native,
            object_pose=changed,
            reference_pose=native["reference_pose"],
        )
        assert (
            other["landing_region"]["object_pose"]
            != measured["landing_region"]["object_pose"]
        )
        assert sim.query.fetches == before_fetches
    registry.close()


def _support_policy_fixture(*, com_tolerance=None, com_x=0.0, extra_preset=False):
    from embodichain.lab.task_program.integrations.simulation.bindings import (
        SimulationSceneBinding,
        SimulationRigidObjectBinding,
    )
    from embodichain.lab.task_program.integrations.simulation.policies import (
        SimulationSegmentPolicyPort,
    )
    from embodichain.lab.task_program.integrations.simulation.stability import (
        SimulationStabilityConditionCfg,
        SupportSensorDeclarationCfg,
    )

    wp.init()
    sim = _Simulation(2)
    sim.objects["payload"].pose[:, 2, 3] = 0.2
    sim.objects["payload"].body_data.com_pose[:, 0] = com_x
    sim.query.buffer.data[:, 2] = 0.1
    binding = SimulationSceneBinding(
        registry_id="support_scene",
        rigid_objects=(
            SimulationRigidObjectBinding(entity_id="item", simulation_uid="payload"),
            SimulationRigidObjectBinding(entity_id="surface", simulation_uid="support"),
        ),
    )
    cfg = SimulationStabilityConditionCfg(
        entity="item",
        reference="surface",
        kind="supported_placement",
        displacement=(0.0, 0.0, 0.2),
        com_tolerance=com_tolerance,
        duration=0.04,
        timeout=0.12,
        support_sensor=SupportSensorDeclarationCfg(
            uid="declared_support", max_contacts_per_env=4
        ),
    )
    robot = SimpleNamespace(get_qpos=lambda target=False: torch.zeros(2, 1))
    registry = SupportObservationRegistry(sim, acquisition_clock=lambda: sim.time)
    port = SimulationSegmentPolicyPort(
        sim,
        robot,
        binding,
        settle_presets={"support": cfg, **({"unused": cfg} if extra_preset else {})},
        step_dt=0.04,
        support_registry=registry,
        sensor_suite=sim.sensors,
    )
    return sim, robot, registry, port, binding, cfg


def test_current_capture_rejects_superseded_session_but_same_clock_bind_is_idempotent():
    first = _support_policy_fixture()
    second = _support_policy_fixture()
    clock, other_clock = EnvironmentStepClock(0.04), EnvironmentStepClock(0.04)
    first[3].bind_clock(clock)
    session = first[3].new_task_program_session()
    second[3].bind_clock(other_clock)
    first[3].bind_clock(clock)
    assert session.capture_support_observation(
        "support", clock=clock
    ).support.valid.all()
    first[3].bind_clock(EnvironmentStepClock(0.04))
    with pytest.raises(ValueError, match="another runtime/session"):
        session.capture_support_observation("support", clock=clock)
    assert (
        second[3]
        .capture_support_observation("support", clock=other_clock)
        .support.valid.all()
    )
    first[2].close()
    second[2].close()


@pytest.mark.parametrize(
    "declaration_case",
    ["normal", "foreign_scene", "missing_preset", "equivalent_snapshot"],
)
def test_collector_uses_actual_bridge_session_and_never_resamples_recorded_support(
    declaration_case,
):
    from types import MethodType
    from dataclasses import replace
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
    from embodichain.lab.gym.envs.episode_context import EpisodeContext
    from embodichain.lab.gym.envs.task_program.bridge import (
        TaskProgramDemoBridge,
        BufferedGymCommandSink,
        RuntimeCommandFrameEncoder,
    )
    from embodichain.lab.task_program import TaskProgramCompiler, decode_task_program
    from embodichain.lab.task_program.integrations import (
        SimulationTaskProgramRegistration,
        SimulationRobotSkillProfileBinding,
        ControlPartResourceBinding,
        ControlPartEndpointBinding,
    )
    from embodichain.lab.task_program.semantics import (
        SkillPolicyPreset,
        EffectAssurance,
    )
    from embodichain.lab.sim.atomic_actions import CARTESIAN_POSE_CAPABILITY
    from embodichain.lab.task_program.integrations.simulation.observations import (
        SimulationSupportObservationCollector,
    )

    sim, robot, registry, template, binding, cfg = _support_policy_fixture(
        com_tolerance=0.1, com_x=0.04, extra_preset=True
    )
    profile = SimulationRobotSkillProfileBinding(
        "robot",
        resources=(
            ControlPartResourceBinding(
                resource_id="primary",
                endpoints=(
                    ControlPartEndpointBinding(
                        endpoint_id="motion",
                        control_part="arm",
                        capabilities=frozenset({CARTESIAN_POSE_CAPABILITY}),
                    ),
                ),
            ),
        ),
        presets=(
            SkillPolicyPreset(
                "default",
                action_option_templates={},
                effect_assurance=EffectAssurance.PROJECTED,
            ),
        ),
        default_preset="default",
    )
    registration = SimulationTaskProgramRegistration(
        binding, profile, settle_presets={"support": cfg, "unused": cfg}
    )
    compiled = TaskProgramCompiler(binding.declare()).compile(
        decode_task_program(
            {
                "program_id": "support",
                "targets": {},
                "integration": {
                    "robot_profile": "robot",
                    "scene_registry": "support_scene",
                    "runtime_preset": "default",
                },
                "program": {
                    "kind": "segment",
                    "name": "observe",
                    "steps": {
                        "kind": "invoke",
                        "call": {"kind": "pick", "object": "item"},
                    },
                    "post": [
                        {"kind": "wait_stable", "entity": "item", "preset": "support"}
                    ],
                },
            }
        )
    )
    clock = EnvironmentStepClock(0.04)
    template.bind_clock(clock)
    encoder = RuntimeCommandFrameEncoder(
        SimpleNamespace(current_qpos=lambda ids: torch.zeros(len(ids), 1))
    )
    sink = BufferedGymCommandSink(encoder, clock)
    runtime = SimpleNamespace(
        result=None,
        status=None,
        start=lambda *a, **k: None,
        step=lambda: None,
        cancel=lambda reason: None,
        adopt_verified_task_state=lambda state: None,
    )
    bridge = TaskProgramDemoBridge(
        compiled,
        runtime,
        sink,
        clock,
        post_policy_port=template,
        isolate_policy_sessions=True,
    )
    context = EpisodeContext(2)
    context._begin_episode(torch.tensor([0, 1]))
    context._finish_initialization(torch.tensor([0, 1]))
    values = {}
    for uid, body in sim.objects.items():
        pose = torch.zeros(2, 3, 7)
        pose[..., 6] = 1.0
        pose[:, 1, :3] = body.pose[:, :3, 3]
        pose[:, 0, 0] = -50.0
        values[uid] = {"pose": pose}
    env = SimpleNamespace(
        sim=sim,
        robot=robot,
        num_envs=2,
        episode_context=context,
        _traj_buffer={"states": {"rigid_objects": values}},
        _traj_steps=torch.ones(2, dtype=torch.long),
        _closed=False,
    )
    env.capture_recorded_rigid_poses = MethodType(
        EmbodiedEnv.capture_recorded_rigid_poses, env
    )
    if declaration_case == "foreign_scene":
        foreign = replace(
            binding,
            rigid_objects=(
                replace(binding.rigid_objects[0], simulation_uid="other_payload"),
                *binding.rigid_objects[1:],
            ),
        )
        registration = replace(registration, scene_binding=foreign)
        with pytest.raises(ValueError, match="scene binding differs"):
            SimulationSupportObservationCollector(env, bridge, registration)
        registry.close()
        return
    if declaration_case == "missing_preset":
        registration = replace(registration, settle_presets={"support": cfg})
        with pytest.raises(ValueError, match="complete preset selection"):
            SimulationSupportObservationCollector(env, bridge, registration)
        registry.close()
        return
    if declaration_case == "equivalent_snapshot":
        registration = replace(registration, scene_binding=replace(binding))
        assert registration.scene_binding is not binding
    collector = SimulationSupportObservationCollector(env, bridge, registration)
    assert bridge.post_policy_port is not template
    segment = next(compiled.iter_segments())
    policy = segment.post_policies[0]
    sensor = sim.get_sensor("declared_support")
    for _ in bridge.post_policy_port.actions(
        policy, segment=segment, active_mask=torch.ones(2, dtype=torch.bool)
    ):
        sensor.begin_control_step()
        for _ in range(4):
            sim.time += 0.01
            sim.revision += 1
            sensor.update_physics_step(0.01)
        clock.advance_after_env_step()
    with pytest.raises((KeyError, RuntimeError)):
        template.post_policy_metadata(policy, segment=segment)
    before = sim.query.fetches
    recorded = collector.capture_recorded(
        "support",
        segment_id=segment.name,
        source_path=segment.source_path,
        frame_indices=[1, 1],
    )
    assert sim.query.fetches == before
    assert recorded["boundary_poses"]["item"][0][0][3] == 0.0
    assert recorded["boundary_frames"] == [1, 1]
    assert recorded["native_support"]["supported"] == [True, True]
    assert recorded["native_support"]["rows"][0]["com_tolerance_m"] == 0.1
    recorded["native_support"]["rows"][0]["contacts_world"][0][0] = 100.0
    again = collector.capture_recorded(
        "support",
        segment_id=segment.name,
        source_path=segment.source_path,
        frame_indices=[1, 1],
    )
    assert again["native_support"]["rows"][0]["contacts_world"][0][0] != 100.0
    current = collector.capture_current("support")
    assert sim.query.fetches == before + 1
    assert current["support_acquisition"]["clock_domain"] == "simulation_time"
    assert current["native_support"]["supported"] == [False, False]
    assert current["native_support"]["rows"][0]["com_tolerance_m"] == pytest.approx(
        0.003
    )
    assert "support_contract_revision" not in current["native_support"]
    with pytest.raises(ValueError, match="before the first"):
        SimulationSupportObservationCollector(env, bridge, registration)
    context._begin_episode(torch.tensor([0]))
    with pytest.raises(ValueError, match="another episode"):
        collector.capture_current("support")
    assert sim.query.fetches == before + 1
    template.close_runtime_resources()
    with pytest.raises(RuntimeError, match="closed"):
        collector.capture_recorded(
            "support",
            segment_id=segment.name,
            source_path=segment.source_path,
            frame_indices=[1, 1],
        )
    assert sim.query.fetches == before + 1
    registry.close()


def test_current_observation_rejects_replaced_native_body_before_reading_old_pose():
    sim, registry, observer, _, _ = _setup()
    original = observer.object
    sim.objects["payload"] = _Body("payload", 2)
    with pytest.raises(ValueError, match="bodies changed"):
        observer.observe_current()
    assert original.pose_reads == 0
    registry.close()
