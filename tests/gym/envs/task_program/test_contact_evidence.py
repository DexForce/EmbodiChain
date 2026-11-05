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

"""Shared contact acquisition and grip contracts without Task Engine imports."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.gym.envs.base_env import BaseEnv
from embodichain.lab.sim.atomic_actions.evidence import PhysicalEvidenceRequest
from embodichain.lab.sim.sensors.contact.sensor import ContactSensor
from embodichain.lab.sim.sensors.contact.sampling import (
    CONTACT_ACTOR_IDS,
    CONTACT_VALID_MASK,
    CONTACT_NORMALS,
    CONTACT_IMPULSES,
    ContactEvidenceSampler,
)
from embodichain.lab.sim.sensors.contact.timeline import (
    ContactTimeline,
    ContactTimelineCfg,
)
from embodichain.lab.sim.atomic_actions.grip_evidence import DriverOpeningBinding
from embodichain.lab.task_program.integrations.simulation.grip_evidence import (
    NativeGripEvidenceCfg,
    NativeGripEvidenceFactory,
    NativeGripObserver,
)
from embodichain.lab.task_program.semantics.evidence import (
    BinaryEffectEvidenceQuery,
    EffectEvidenceCollectionContext,
)
from embodichain.lab.task_program.semantics.effects import (
    BinaryEffectClause,
    BinaryEvidenceKind,
    CONTROL_PART_EVIDENCE_PROVIDER_ID,
    CONTROL_PART_EVIDENCE_PROVIDER_REVISION,
    CONSTRAINT_EFFECT_CHANNEL,
    ControlPartEvidenceAddress,
    EffectEvidenceSourceRef,
    HeldObjectRelation,
    HeldObjectStateExpectation,
)
from embodichain.lab.task_program.semantics.scene import (
    SceneRegistry,
    SceneObjectRef,
    SceneEntityRegistration,
)


class ContactSource:
    def __init__(self, count=1, capacity=2):
        self.num_instances = count
        self.contact_capabilities = SimpleNamespace(geometry=True, impulse=True)
        self.dropped_contacts = 0
        self.update_count = 0
        self.revision = 1
        self.topology = 1
        self.data = {
            "is_valid": torch.zeros(count, capacity, dtype=torch.bool),
            "user_ids": torch.zeros(count, capacity, 2, dtype=torch.long),
            "normal": torch.zeros(count, capacity, 3),
            "position": torch.zeros(count, capacity, 3),
            "impulse": torch.zeros(count, capacity),
        }
        if capacity >= 2:
            for row in range(count):
                target = 10 * row + 1
                self.data["is_valid"][row, :2] = True
                self.data["user_ids"][row, :2] = torch.tensor(
                    [[target, target + 1], [target, target + 2]]
                )
                self.data["normal"][row, :2] = torch.tensor(
                    [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
                )
                self.data["impulse"][row, :2] = 0.1

    def update(self):
        self.update_count += 1

    def get_data(self):
        return self.data

    def get_actor_ids(self, uid, names=None):
        if names is None:
            return torch.arange(self.num_instances).reshape(-1, 1) * 10 + 1
        return torch.arange(self.num_instances).reshape(-1, 1) * 10 + torch.tensor(
            [[2, 3]]
        )


def query(
    *,
    expected=True,
    relation=HeldObjectRelation.ATTACHED,
    part="hand",
    evidence_id="constraint",
    object_id="subject",
):
    source = EffectEvidenceSourceRef(
        CONTROL_PART_EVIDENCE_PROVIDER_ID,
        CONTROL_PART_EVIDENCE_PROVIDER_REVISION,
        ControlPartEvidenceAddress(part, CONSTRAINT_EFFECT_CHANNEL),
    )
    return BinaryEffectEvidenceQuery(
        BinaryEffectClause(
            evidence_id, "held", source, BinaryEvidenceKind.CONSTRAINT, expected
        ),
        HeldObjectStateExpectation("held", relation, object_id, "primary", part, part),
    )


def observer(source, *, revision=1, opening=None, capacity=4096):
    return NativeGripObserver(
        ContactEvidenceSampler(source),
        {"subject": source.get_actor_ids("object")},
        {"hand": source.get_actor_ids("robot", ("pad_a", "pad_b"))},
        sensor_uid="explicit_contacts",
        algorithm_revision=revision,
        release_opening=opening,
        trace_capacity=capacity,
    )


def context(ids=(0,), revision=1):
    return EffectEvidenceCollectionContext(
        1.0, revision, torch.tensor(ids, dtype=torch.long)
    )


@pytest.mark.parametrize(
    "enabled,opened,expected,relation,other_pad,dropped,finite,valid,value",
    [
        (False, True, False, HeldObjectRelation.DETACHED, False, 0, True, False, False),
        (True, True, False, HeldObjectRelation.DETACHED, False, 0, True, True, False),
        (True, False, False, HeldObjectRelation.DETACHED, False, 0, True, False, False),
        (True, True, True, HeldObjectRelation.ATTACHED, False, 0, True, False, False),
        (True, True, False, HeldObjectRelation.ATTACHED, False, 0, True, False, False),
        (True, True, False, HeldObjectRelation.DETACHED, True, 0, True, True, True),
        (True, True, False, HeldObjectRelation.DETACHED, False, 1, True, False, False),
        (True, True, False, HeldObjectRelation.DETACHED, False, 0, False, False, False),
    ],
)
def test_explicit_open_release_revision_preserves_physical_negative_controls(
    enabled, opened, expected, relation, other_pad, dropped, finite, valid, value
):
    source = ContactSource()
    source.data["is_valid"][0, 1] = other_pad
    source.dropped_contacts = dropped
    if not finite:
        source.data["normal"][0, 0, 0] = float("nan")
    reader = observer(
        source,
        revision=2 if enabled else 1,
        opening=lambda part, env: {"finite": True, "open": opened},
    )
    result = reader(query(expected=expected, relation=relation), context())
    assert result.valid.tolist() == [valid]
    assert result.values.tolist() == [value]
    assert reader.snapshot_trace()["recorded_query_count"] == 1


@pytest.mark.parametrize("damage", ["zero_normal", "nonopposing", "negative_impulse"])
def test_opening_does_not_resolve_invalid_or_nonopposing_contacts(damage):
    source = ContactSource()
    source.data["normal"].fill_(0)
    source.data["normal"][:, :, 0] = 1
    if damage != "nonopposing":
        source.data["user_ids"][0, 1] = torch.tensor([4, 5])
    if damage == "zero_normal":
        source.data["normal"][0, 0].zero_()
    elif damage == "negative_impulse":
        source.data["impulse"][0, 1] = -0.1
    result = observer(
        source, revision=2, opening=lambda *args: {"finite": True, "open": True}
    )(query(expected=False, relation=HeldObjectRelation.DETACHED), context())
    assert result.values.tolist() == [False]
    assert result.valid.tolist() == [False]


@pytest.mark.parametrize(
    "driver,mimic,opened",
    [
        (0.0, 2.0, True),
        (0.049, 0.0, True),
        (0.05, 0.0, False),
        (0.051, 0.0, False),
        (float("nan"), 0.0, False),
        (0.0, float("nan"), False),
    ],
)
def test_declared_driver_opening_keeps_strict_boundary_and_finite_full_row(
    driver, mimic, opened
):
    reads = []

    def measured_qpos(name):
        reads.append(name)
        return torch.tensor([[driver, mimic]])

    robot = SimpleNamespace(
        joint_names=["drive", "compliant"],
        get_joint_ids=lambda name: [0, 1],
        get_qpos=measured_qpos,
    )
    binding = DriverOpeningBinding(
        "hand", ("drive", "compliant"), (0,), (0.0, 0.0), 0.05
    )
    reader = binding.bind(robot)
    assert reads == []
    value = reader(0)
    assert reads == ["hand"]
    assert value["open"] is opened
    assert value["driving_joint_names"] == ["drive"]
    assert "not_contact_free_separation" in value["scope"]


def test_sampler_owns_masked_frames_in_requested_environment_order():
    source = ContactSource(count=3, capacity=4)
    source.data["normal"][:, 2:].fill_(float("nan"))
    source.data["impulse"][:, 2:].fill_(float("nan"))
    sampler = ContactEvidenceSampler(source)
    request = PhysicalEvidenceRequest(
        2.0,
        7,
        torch.tensor([2, 0]),
        (CONTACT_ACTOR_IDS, CONTACT_NORMALS, CONTACT_IMPULSES),
    )
    sample = sampler.sample(request)
    assert sample.frame.env_ids.tolist() == [2, 0]
    assert sample.frame.get(CONTACT_ACTOR_IDS).values[:, 0, 0].tolist() == [21, 1]
    assert (
        sample.frame.get(CONTACT_VALID_MASK).values.tolist()
        == [[True, True, False, False]] * 2
    )
    assert all(batch.valid.all() for batch in sample.frame.batches.values())
    sample.frame.batches[CONTACT_NORMALS].env_ids[0] = 1
    assert sample.frame.get(CONTACT_IMPULSES).env_ids.tolist() == [2, 0]
    assert sample.frame.env_ids.tolist() == [2, 0]
    sample.frame.batches[CONTACT_NORMALS].env_ids[0] = 2
    old = sample.frame.get(CONTACT_NORMALS).values.clone()
    source.data["normal"].zero_()
    assert torch.equal(sample.frame.get(CONTACT_NORMALS).values, old)
    request.env_ids[0] = 1
    assert sample.frame.env_ids.tolist() == [2, 0]
    sample.frame.env_ids[0] = 99
    assert sample.frame.batches[CONTACT_NORMALS].env_ids.tolist() == [2, 0]
    sample.frame.env_ids[0] = 2
    sample.frame.batches[CONTACT_NORMALS].values[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        sample.frame.snapshot()
    assert source.update_count == 1


def test_sampler_revalidates_request_scope_before_touching_the_source():
    source = ContactSource(count=3)
    request = PhysicalEvidenceRequest(2.0, 7, torch.tensor([2, 0]))
    request.env_ids[1] = 2
    with pytest.raises(ValueError, match="unique"):
        ContactEvidenceSampler(source).sample(request)
    assert source.update_count == 0


@pytest.mark.parametrize(
    "damage", ["dropped", "active_nan", "revision", "topology", "fetch_error"]
)
def test_sampler_does_not_convert_unavailable_acquisition_to_negative_evidence(damage):
    source = ContactSource()
    sampler = ContactEvidenceSampler(
        source,
        native_revision_reader=lambda: source.revision,
        topology_revision_reader=lambda: source.topology,
    )
    if damage == "dropped":
        source.dropped_contacts = 3
    elif damage == "active_nan":
        source.data["normal"][0, 0, 0] = float("nan")
    elif damage == "topology":
        source.topology += 1
    elif damage == "revision":
        source.update = lambda: setattr(source, "revision", source.revision + 1)
    else:

        def fail():
            raise RuntimeError("fetch failed")

        source.update = fail
    sample = sampler.sample(PhysicalEvidenceRequest(1.0, 2, torch.tensor([0])))
    assert sample.frame.get(CONTACT_NORMALS).valid.tolist() == [False]
    assert sample.frame.get(CONTACT_NORMALS).acquisition_errors[0]


def test_grip_trace_keeps_explicit_loss_and_nonfinite_accounting():
    source = ContactSource()
    reader = observer(source, capacity=1)
    reader(query(), context())
    source.data["normal"][0, 0, 0] = float("nan")
    reader(query(), context(revision=2))
    snapshot = reader.snapshot_trace()
    assert snapshot["cumulative_query_count"] == 2
    assert snapshot["retained_query_count"] == 1
    assert snapshot["evicted_query_count"] == 1
    assert not snapshot["trace_complete"]
    assert snapshot["finite_audit"]["nonfinite_query_count"] == 1
    assert snapshot["finite_audit"]["retained_nonfinite_locations"]
    assert snapshot["trace"][0]["rows"][0]["raw_contacts"]["normals"][0][0] == {
        "nonfinite": "NaN"
    }

    def fail():
        raise RuntimeError("missing query")

    source.update = fail
    result = reader(query(), context(revision=3))
    assert not result.valid.any()
    assert reader.snapshot_trace()["unrecorded_query_count"] == 1


@pytest.mark.parametrize("revision", [1, 2])
def test_shared_factory_uses_explicit_bindings_and_keeps_provider_identity_and_instance_state(
    revision,
):
    source = ContactSource(count=3)
    source.data["is_valid"][0] = False
    sensors = {}

    def add_sensor(cfg):
        source.cfg = cfg
        sensors[cfg.uid] = source
        return source

    simulation = SimpleNamespace(
        get_sensor=sensors.get,
        add_sensor=add_sensor,
        spawn_result=SimpleNamespace(topology_revision=1),
    )
    robot = SimpleNamespace(
        uid="robot",
        joint_names=["drive", "compliant"],
        get_joint_ids=lambda name: [0, 1],
        get_qpos=lambda **kwargs: torch.zeros(3, 2),
        get_qvel=lambda **kwargs: torch.zeros(3, 2),
        compute_fk=lambda **kwargs: torch.eye(4).repeat(3, 1, 1),
    )
    state_provider = SimpleNamespace(observe=lambda **kwargs: None)
    registry = SceneRegistry(
        (
            SceneEntityRegistration(
                SceneObjectRef("subject"), state_provider=state_provider
            ),
        )
    )
    cfg = NativeGripEvidenceCfg(
        sensor_uid="custom_sensor",
        object_bindings=(("subject", "native_body"),),
        pad_bindings=(("precision_clamp", ("first_pad", "second_pad")),),
        algorithm_revision=revision,
        opening_bindings=(
            (
                DriverOpeningBinding(
                    "precision_clamp", ("drive", "compliant"), (0,), (0.0, 0.0), 0.05
                ),
            )
            if revision == 2
            else ()
        ),
    )
    factory = NativeGripEvidenceFactory(cfg)
    from embodichain.lab.task_program.integrations.extensions import (
        declare_control_part_evidence_factory,
    )

    declaration = declare_control_part_evidence_factory(factory)
    arguments = dict(
        simulation=simulation,
        robot=robot,
        scene_registry=registry,
        engine=SimpleNamespace(robot=robot),
        scene_provider=None,
    )
    if revision == 2:
        wrong_robot = SimpleNamespace(**vars(robot))
        wrong_robot.joint_names = ["compliant", "drive"]
        with pytest.raises(ValueError, match="joint order"):
            factory.create(
                **{
                    **arguments,
                    "robot": wrong_robot,
                    "engine": SimpleNamespace(robot=wrong_robot),
                }
            )
        assert sensors == {}
    first, second = factory.create(**arguments), factory.create(**arguments)
    observation = first.collect(
        (query(part="precision_clamp", evidence_id="left-lane:grip"),),
        context(ids=(2, 0)),
    )
    batch = observation["left-lane:grip"]
    assert batch.evidence_id == "left-lane:grip"
    assert batch.values.tolist() == [True, False]
    assert batch.valid.tolist() == [True, True]
    assert declare_control_part_evidence_factory(factory) == declaration
    assert first.provider_id == CONTROL_PART_EVIDENCE_PROVIDER_ID
    assert first.revision == CONTROL_PART_EVIDENCE_PROVIDER_REVISION
    assert first.grip_observer.snapshot_trace()["recorded_query_count"] == 1
    assert second.grip_observer.snapshot_trace()["recorded_query_count"] == 0
    assert source.cfg.articulation_cfg_list[0].link_name_list == [
        "first_pad",
        "second_pad",
    ]
    # Keep only one loaded pad in row 2. Only the explicitly configured second
    # revision may use the measured open driver to resolve release availability.
    source.data["is_valid"][2, 1] = False
    released = first.collect(
        (
            query(
                part="precision_clamp",
                evidence_id="release",
                expected=False,
                relation=HeldObjectRelation.DETACHED,
            ),
        ),
        context(ids=(2,)),
    )["release"]
    assert released.values.tolist() == [False]
    assert released.valid.tolist() == [revision == 2]
    source.cfg.filter_need_both_actor = False
    with pytest.raises(ValueError, match="contact scope"):
        factory.create(**arguments)


def make_timeline(source, *, max_steps=30000, max_rows=1_000_000, max_substeps=256):
    clock = SimpleNamespace(step_index=0)
    sampler = ContactEvidenceSampler(
        source,
        native_revision_reader=lambda: source.revision,
        topology_revision_reader=lambda: source.topology,
    )
    timeline = ContactTimeline(
        sampler,
        ContactTimelineCfg(
            step_dt=0.04,
            max_control_steps=max_steps,
            max_contact_rows=max_rows,
            max_substeps_per_control=max_substeps,
        ),
        clock=lambda: (clock.step_index, clock.step_index * 0.04),
    )
    return timeline, clock


def capture_step(timeline, source, count=4):
    timeline.begin_control_step()
    for _ in range(count):
        source.revision += 1
        timeline.after_sample(0.01, timeline.before_sample())


def test_timeline_partial_reset_preserves_other_rows_and_owned_snapshots():
    source = ContactSource(count=2)
    timeline, clock = make_timeline(source)
    capture_step(timeline, source)
    before = timeline.snapshot()
    timeline.reset([1])
    reset = timeline.snapshot()
    assert reset["environments"][0] == before["environments"][0]
    assert reset["environments"][1]["samples"] == []
    assert reset["environments"][1]["episode_generation"] == 1
    clock.step_index = 1
    capture_step(timeline, source)
    after = timeline.snapshot()
    assert len(after["environments"][0]["samples"]) == 2
    assert len(after["environments"][1]["samples"]) == 1
    assert len(before["environments"][0]["samples"]) == 1


@pytest.mark.parametrize("limit", ["steps", "contacts", "substeps"])
def test_timeline_exposes_capacity_loss_without_turning_it_into_evidence(limit):
    source = ContactSource()
    timeline, clock = make_timeline(
        source,
        max_steps=1 if limit == "steps" else 10,
        max_rows=1 if limit == "contacts" else 100,
        max_substeps=1 if limit == "substeps" else 256,
    )
    capture_step(timeline, source)
    clock.step_index = 1
    capture_step(timeline, source)
    row = timeline.snapshot()["environments"][0]
    assert row["errors"]
    assert len(row["samples"]) <= 1
    assert (
        row["retained_contact_rows"] <= 8
        if limit == "steps"
        else row["retained_contact_rows"] <= 4
    )


def test_empty_contact_timeline_does_not_retain_sensor_padding():
    source = ContactSource(count=2, capacity=4096)
    source.data["is_valid"].zero_()
    timeline, _ = make_timeline(source)
    capture_step(timeline, source)
    result = timeline.snapshot()
    assert result["retained_contact_rows"] == 0
    for row in result["environments"]:
        assert len(row["samples"][0]["substeps"]) == 4
        assert all(
            not step["contacts"]["pairs"] and not step["contacts"]["normals"]
            for step in row["samples"][0]["substeps"]
        )


def test_public_base_env_step_drives_actual_sensor_observer_port():
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path

    # Reuse the complete real-sensor fixture rather than mirror its private buffers.
    fixture_path = (
        Path(__file__).parents[3] / "sim/sensors/test_contact_query_sensor.py"
    )
    spec = spec_from_file_location("contact_query_fixture", fixture_path)
    fixture = module_from_spec(spec)
    spec.loader.exec_module(fixture)
    _FakeQuery, _sensor_for_query = fixture._FakeQuery, fixture._sensor_for_query
    import warp as wp

    wp.init()
    source = ContactSource()
    query = _FakeQuery()
    sensor = _sensor_for_query(query)
    clock = SimpleNamespace(step_index=0)
    sampled = []
    sampler = ContactEvidenceSampler(
        sensor,
        native_revision_reader=lambda: source.revision,
        topology_revision_reader=lambda: source.topology,
    )
    timeline = ContactTimeline(
        sampler,
        ContactTimelineCfg(step_dt=0.04),
        clock=lambda: (clock.step_index, clock.step_index * 0.04),
    )

    def forbid_refresh():
        raise AssertionError("Substep evidence must not fetch a second time.")

    sensor.update = forbid_refresh
    sensor.add_sampling_observer("physical_contacts", timeline)
    env = BaseEnv.__new__(BaseEnv)
    env.cfg = SimpleNamespace(sim_steps_per_control=4)
    env.sim_cfg = SimpleNamespace(physics_dt=0.01)
    env.sensors = {"contacts": sensor}

    def update(dt, steps, **kwargs):
        for step in range(steps):
            source.revision += 1
            query.buffer.data[:2, 9].fill_((step + 1) * 0.01)
            sensor._sim.physics_step_index += 1
            sampled.append(dt)
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
    clock.step_index = 1
    env.step(None)
    rows = timeline.snapshot()["environments"][0]["samples"]
    assert [row["step"] for row in rows] == [1, 2]
    assert rows[1]["time"] == 0.08
    assert len(rows[1]["substeps"]) == 4
    assert (
        rows[1]["substeps"][0]["contacts"]["impulses"][0]
        < rows[1]["substeps"][-1]["contacts"]["impulses"][0]
    )
    assert len(sampled) == 8
    assert source.update_count == 0  # Substep observer never fetches a second time.
    assert timeline.snapshot()["retained_contact_rows"] == 16


def test_geometry_only_contact_backend_cannot_certify_grip_or_clear_release():
    source = ContactSource()
    source.contact_capabilities.impulse = False
    source.data["impulse"].zero_()
    sample = ContactEvidenceSampler(source).sample(
        PhysicalEvidenceRequest(1.0, 1, torch.tensor([0]))
    )
    assert sample.frame.get(CONTACT_NORMALS).valid.tolist() == [True]
    assert sample.frame.get(CONTACT_IMPULSES).valid.tolist() == [False]
    result = observer(source)(
        query(expected=False, relation=HeldObjectRelation.DETACHED), context()
    )
    assert result.values.tolist() == [False]
    assert result.valid.tolist() == [False]


def test_reset_during_substeps_suppresses_only_selected_rows_until_next_interval():
    source = ContactSource(count=2)
    timeline, clock = make_timeline(source)
    timeline.begin_control_step()
    timeline.after_sample(0.01, timeline.before_sample())
    assert timeline.snapshot()["environments"][0]["incomplete_control_step"] == 1
    timeline.reset([1])
    for _ in range(3):
        timeline.after_sample(0.01, timeline.before_sample())
    rows = timeline.snapshot()["environments"]
    assert len(rows[0]["samples"]) == 1
    assert len(rows[0]["samples"][0]["substeps"]) == 4
    assert rows[1]["samples"] == []
    clock.step_index = 1
    capture_step(timeline, source)
    assert (
        timeline.snapshot()["environments"][1]["samples"][0]["episode_generation"] == 1
    )


@pytest.mark.parametrize(
    "damage", ["missing_revision", "revision_changes_during_fetch", "clock"]
)
def test_timeline_keeps_missing_or_misaligned_samples_unavailable(damage):
    source = ContactSource()
    timeline, clock = make_timeline(source)
    if damage == "missing_revision":
        timeline = ContactTimeline(
            ContactEvidenceSampler(source),
            ContactTimelineCfg(step_dt=0.04),
            clock=lambda: (0, 0.0),
        )
    elif damage == "clock":
        timeline = ContactTimeline(
            timeline.sampler, ContactTimelineCfg(step_dt=0.04), clock=lambda: (0, 0.1)
        )
    timeline.begin_control_step()
    for _ in range(4):
        token = timeline.before_sample()
        if damage == "revision_changes_during_fetch":
            source.revision += 1
        timeline.after_sample(0.01, token)
    result = timeline.snapshot()["environments"][0]
    assert result["errors"]
    assert not any(record["available"] for record in result["samples"])


def test_factory_freezes_caller_configuration_without_enabling_release_by_presence():
    objects, pads = [["subject", "body"]], [["hand", ["pad_a", "pad_b"]]]
    cfg = NativeGripEvidenceCfg(
        sensor_uid="contact", object_bindings=objects, pad_bindings=pads
    )
    factory = NativeGripEvidenceFactory(cfg)
    objects[0][1] = "changed"
    pads[0][1][0] = "changed"
    cfg.algorithm_revision = 2
    assert factory.object_bindings == (("subject", "body"),)
    assert factory.pad_bindings == (("hand", ("pad_a", "pad_b")),)
    assert factory.algorithm_revision == 1
    with pytest.raises(ValueError, match="opening bindings"):
        NativeGripEvidenceFactory(cfg)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "cosine", [-0.5001, -0.5, -0.4999, -0.5000000596046448, -0.4999999701976776]
)
@pytest.mark.parametrize("impulse", [0.0, 0.1])
def test_host_frame_round_trip_preserves_cuda_threshold_facts(cosine, impulse):
    import math
    from embodichain.compute.geometry.grip_contacts import (
        classify_contacts,
    )

    device = torch.device("cuda:0")
    source = ContactSource(capacity=4)
    source.data["normal"][0, 1] = torch.tensor(
        [cosine, math.sqrt(1 - cosine * cosine), 0.0]
    )
    source.data["impulse"][0, :2] = impulse
    source.data["normal"][0, 2:] = float("nan")
    source.data["impulse"][0, 2:] = -1.0
    source.data = {
        name: value.to(
            device=device, dtype=torch.int32 if name == "user_ids" else value.dtype
        )
        for name, value in source.data.items()
    }
    active = source.data["is_valid"][0]
    expected = classify_contacts(
        source.data["user_ids"][0, active],
        source.data["normal"][0, active],
        source.data["impulse"][0, active],
        target=1,
        pads=(2, 3),
    )
    # Independent negative controls away from the floating-point boundary.
    if impulse == 0:
        assert expected == (False, True)
    elif cosine in (-0.5001, -0.4999):
        assert expected == ((True, True) if cosine < -0.5 else (False, False))
    reader = observer(source)
    collection = EffectEvidenceCollectionContext(
        1.0, 1, torch.tensor([0], device=device)
    )
    result = reader(query(), collection)
    assert (result.values.item(), result.valid.item()) == expected
    assert result.values.device == result.valid.device == device
    assert reader.snapshot_trace()["finite_audit"]["valid"]
    assert torch.isnan(source.data["normal"][0, 2:]).all()
    request = PhysicalEvidenceRequest(1.0, 1, collection.env_ids, (CONTACT_NORMALS,))
    frame = reader.sampler.collect(
        SimpleNamespace(timestamp=1.0, env_ids=collection.env_ids), request
    )
    assert frame.env_ids.device == frame.get(CONTACT_NORMALS).values.device == device


def test_shared_provider_keeps_distinct_parallel_evidence_channels():
    source = ContactSource(capacity=4)
    source.data["is_valid"].fill_(True)
    source.data["user_ids"][0] = torch.tensor([[1, 2], [1, 3], [4, 5], [4, 6]])
    source.data["normal"][0] = torch.tensor(
        [[1.0, 0, 0], [-1.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0]]
    )
    source.data["impulse"].fill_(0.1)
    actors = {
        ("first_body", None): [[1]],
        ("second_body", None): [[4]],
        ("robot", ("a", "b")): [[2, 3]],
        ("robot", ("c", "d")): [[5, 6]],
    }
    source.get_actor_ids = lambda uid, names=None: torch.tensor(
        actors[(uid, None if names is None else tuple(names))]
    )

    def create_sensor(cfg):
        source.cfg = cfg
        return source

    simulation = SimpleNamespace(
        get_sensor=lambda uid: None,
        add_sensor=create_sensor,
        spawn_result=SimpleNamespace(topology_revision=1),
    )
    robot = SimpleNamespace(
        uid="robot",
        get_qpos=lambda **kwargs: torch.zeros(1, 2),
        get_qvel=lambda **kwargs: torch.zeros(1, 2),
        compute_fk=lambda **kwargs: torch.eye(4).unsqueeze(0),
    )
    registry = SceneRegistry(
        tuple(
            SceneEntityRegistration(
                SceneObjectRef(name),
                state_provider=SimpleNamespace(observe=lambda **kwargs: None),
            )
            for name in ("first", "second")
        )
    )
    cfg = NativeGripEvidenceCfg(
        sensor_uid="contacts",
        object_bindings=(("first", "first_body"), ("second", "second_body")),
        pad_bindings=(("alpha", ("a", "b")), ("beta", ("c", "d"))),
    )
    provider = NativeGripEvidenceFactory(cfg).create(
        simulation=simulation,
        robot=robot,
        scene_registry=registry,
        engine=SimpleNamespace(robot=robot),
        scene_provider=None,
    )
    results = provider.collect(
        (
            query(object_id="first", part="alpha", evidence_id="lane-a:contact"),
            query(object_id="second", part="beta", evidence_id="lane-b:contact"),
        ),
        context(),
    )
    assert results["lane-a:contact"].values.tolist() == [True]
    assert results["lane-a:contact"].valid.tolist() == [True]
    assert results["lane-b:contact"].values.tolist() == [False]
    assert results["lane-b:contact"].valid.tolist() == [False]
    assert list(results) == ["lane-a:contact", "lane-b:contact"]
    assert source.update_count == 2


def test_timeline_contact_budget_keeps_one_complete_batch_then_reports_unavailable():
    source = ContactSource(count=3, capacity=2)
    clock = SimpleNamespace(step_index=0)
    sampler = ContactEvidenceSampler(
        source,
        native_revision_reader=lambda: source.revision,
        topology_revision_reader=lambda: source.topology,
    )
    timeline = ContactTimeline(
        sampler,
        ContactTimelineCfg(step_dt=0.01, max_contact_rows=5),
        clock=lambda: (clock.step_index, clock.step_index * 0.01),
    )
    timeline.begin_control_step()
    timeline.after_sample(0.01, timeline.before_sample())
    first = timeline.snapshot()
    assert first["retained_contact_rows"] == 6
    assert first["retained_contact_rows"] < 5 + 3 * 2
    assert all(
        len(row["samples"][0]["contacts"]["pairs"]) == 2
        for row in first["environments"]
    )
    assert all(row["samples"][0]["available"] for row in first["environments"])
    clock.step_index = 1
    timeline.begin_control_step()
    timeline.after_sample(0.01, timeline.before_sample())
    stopped = timeline.snapshot()
    assert stopped["retained_contact_rows"] == 6
    assert all(len(row["samples"]) == 1 for row in stopped["environments"])
    assert all(
        any("capacity exceeded" in error for error in row["errors"])
        for row in stopped["environments"]
    )


@pytest.mark.parametrize(
    ("indices", "generations"),
    [
        ([-1, 2, -1], [0, 0, 1]),
        (torch.tensor(1), [0, 1, 0]),
        ([1.9, True], [0, 1, 0]),
        ([], [0, 0, 0]),
    ],
)
def test_timeline_reset_matches_sensor_indices_without_double_generation(
    indices, generations
):
    source = ContactSource(count=3)
    timeline, _ = make_timeline(source)
    capture_step(timeline, source)
    before = timeline.snapshot()
    sensor = ContactSensor.__new__(ContactSensor)
    sensor._sample_ready_stream = None
    sensor.device = torch.device("cpu")
    sensor._data_buffer = source.data
    sensor._num_contacts_per_env = source.data["is_valid"].sum(1).to(torch.int32)
    sensor._query_dropped_count = torch.zeros(1, dtype=torch.int32)
    sensor._scatter_dropped_count = torch.zeros(3, dtype=torch.int32)
    sensor._sample_scatter_dropped_count = torch.zeros(3, dtype=torch.int32)
    sensor._latest_contact_buffer = None
    sensor._histories = {}
    sensor._sampling_observers = {}
    sensor.add_sampling_observer("timeline", timeline)
    sensor.reset(indices)
    after = timeline.snapshot()
    assert [row["episode_generation"] for row in after["environments"]] == generations
    for row, generation in enumerate(generations):
        if generation:
            assert after["environments"][row]["samples"] == []
            assert not source.data["is_valid"][row].any()
        else:
            assert after["environments"][row] == before["environments"][row]


@pytest.mark.parametrize("indices", [[3], [-4], [[0, 1]], ["bad"]])
def test_timeline_reset_propagates_sensor_index_errors_before_mutating_history(indices):
    source = ContactSource(count=3)
    timeline, _ = make_timeline(source)
    capture_step(timeline, source)
    before = timeline.snapshot()
    with pytest.raises((TypeError, ValueError, IndexError, RuntimeError)) as expected:
        torch.zeros(3).index_fill_(0, torch.as_tensor(indices, dtype=torch.long), 1)
    with pytest.raises(type(expected.value)):
        timeline.reset(indices)
    assert timeline.snapshot() == before


@pytest.mark.parametrize("loss", [0, 3, -1])
def test_sampler_combined_loss_diagnostic_does_not_read_legacy_counters(loss):
    source = ContactSource()

    class CombinedSource:
        def __getattr__(self, name):
            return getattr(source, name)

        @property
        def evidence_dropped_contacts(self):
            return loss

        @property
        def dropped_contacts(self):
            raise AssertionError("Combined acquisition must not synchronize twice.")

        @property
        def latest_sample_dropped_contacts(self):
            raise AssertionError("Combined acquisition must not synchronize twice.")

    sample = ContactEvidenceSampler(CombinedSource()).sample(
        PhysicalEvidenceRequest(1.0, 0, torch.tensor([0]))
    )
    assert sample.frame.get(CONTACT_NORMALS).valid.tolist() == [loss == 0]
    if loss:
        assert sample.frame.get(CONTACT_NORMALS).acquisition_errors[0]


@pytest.mark.parametrize("loss", [3, -1])
def test_custom_source_latest_loss_fallback_retains_unavailable_evidence(loss):
    source = ContactSource()
    source.latest_sample_dropped_contacts = loss
    sample = ContactEvidenceSampler(source).sample(
        PhysicalEvidenceRequest(1.0, 0, torch.tensor([0]))
    )
    assert sample.frame.get(CONTACT_NORMALS).valid.tolist() == [False]
    assert sample.frame.get(CONTACT_NORMALS).acquisition_errors[0]


def test_optional_contact_distances_preserve_default_channels_and_row_validity():
    from embodichain.lab.sim.sensors.contact.sampling import (
        CONTACT_DISTANCES,
        CONTACT_POSITIONS,
        DEFAULT_CHANNELS,
    )

    source = ContactSource(count=2, capacity=3)
    sampler = ContactEvidenceSampler(source)
    default = sampler.sample(PhysicalEvidenceRequest(1.0, 1, torch.tensor([1, 0])))
    assert tuple(default.frame.batches) == DEFAULT_CHANNELS
    assert len(DEFAULT_CHANNELS) == 5
    assert all(batch.valid.all() for batch in default.frame.batches.values())
    missing = sampler.sample(
        PhysicalEvidenceRequest(
            1.0,
            2,
            torch.tensor([1, 0]),
            (CONTACT_POSITIONS, CONTACT_DISTANCES),
        )
    )
    assert missing.frame.get(CONTACT_POSITIONS).valid.all()
    assert missing.frame.get(CONTACT_DISTANCES).valid.tolist() == [False, False]
    assert all(missing.frame.get(CONTACT_DISTANCES).acquisition_errors)
    source.data["distance"] = torch.tensor(
        [
            [-0.001, 0.002, float("nan")],
            [float("nan"), 0.004, float("nan")],
        ]
    )
    sampled = sampler.sample(
        PhysicalEvidenceRequest(
            2.0,
            3,
            torch.tensor([1, 0]),
            (CONTACT_DISTANCES,),
        )
    )
    distance = sampled.frame.get(CONTACT_DISTANCES)
    assert distance.valid.tolist() == [False, True]
    assert distance.values[1].tolist() == pytest.approx([-0.001, 0.002, 0.0])
    assert distance.values.device == sampled.frame.env_ids.device
    source.data["distance"].zero_()
    assert distance.values[1, 0] == pytest.approx(-0.001)
    assert source.update_count == 3


def test_joint_contact_observer_reuses_substep_sample_and_owns_window_records():
    from embodichain.lab.sim.sensors.contact import (
        JointContactObserver,
    )
    from embodichain.lab.sim.sensors.contact.sampling import CONTACT_DISTANCES

    source = ContactSource(capacity=5)
    source.data["distance"] = torch.tensor([[-0.001, 0.002, float("nan"), 0.0, 0.0]])
    qpos, qvel = torch.tensor([0.12]), torch.tensor([0.01])
    clock = [0.0]
    observer = JointContactObserver(
        ContactEvidenceSampler(source),
        lambda: qpos,
        lambda: clock[0],
        joint_velocity=lambda: qvel,
    )
    generation = observer.begin_window("prepare")
    observer.capture(refresh=True)
    initial = observer.snapshot()[0]
    assert source.update_count == 1
    observer.set_phase("twist")
    token = observer.before_sample()
    clock[0] = 0.004
    qpos[0] = 0.13
    observer.after_sample(0.004, token)
    records = observer.snapshot()
    assert len(records) == 2
    assert len(observer.snapshot(1)) == 1
    assert observer.snapshot(2) == ()
    with pytest.raises(ValueError, match="window prefix"):
        observer.snapshot(-1)
    assert [record.phase for record in records] == ["prepare", "twist"]
    assert records[-1].frame.timestamp == 0.004
    assert records[-1].frame.observation_revision == 2
    assert records[-1].frame.get(
        "articulation.joint_position"
    ).values.tolist() == pytest.approx([0.13])
    assert initial.frame.get(
        "articulation.joint_position"
    ).values.tolist() == pytest.approx([0.12])
    assert records[-1].frame.get(CONTACT_DISTANCES).values.tolist()[0] == pytest.approx(
        [-0.001, 0.002]
    )
    assert source.update_count == 1
    records[-1].frame.batches["articulation.joint_position"].values.fill_(100)
    assert observer.snapshot()[-1].frame.get("articulation.joint_position").values[
        0
    ] == pytest.approx(0.13)
    with pytest.raises(ValueError, match="consecutive substeps"):
        observer.after_sample(0.004, observer.before_sample())
    observer.reset([])
    assert observer.generation == generation
    observer.begin_window("next", max_samples=1)
    observer.after_sample(0.004, token)
    assert observer.snapshot() == ()
    observer.capture(refresh=True)
    observer.capture(refresh=True)
    assert len(observer.snapshot()) == 1
    assert observer.overflow
    assert source.update_count == 2
    late = observer.before_sample()
    observer.close()
    observer.after_sample(0.004, late)
    assert source.update_count == 2


def test_joint_contact_channels_keep_missing_and_nonfinite_observations_unavailable():
    from embodichain.lab.sim.sensors.contact import (
        JointContactObserver,
    )
    from embodichain.lab.sim.sensors.contact.sampling import CONTACT_DISTANCES

    source = ContactSource()
    observer = JointContactObserver(
        ContactEvidenceSampler(source),
        lambda: torch.tensor([float("nan")]),
        lambda: 1.0,
    )
    observer.begin_window("press")
    observer.capture(refresh=True)
    record = observer.snapshot()[0]
    assert not record.frame.get("articulation.joint_position").valid.any()
    assert not record.frame.get(CONTACT_DISTANCES).valid.any()
    assert record.frame.get(CONTACT_NORMALS).valid.all()
    with pytest.raises(ValueError, match="B=1"):
        JointContactObserver(
            ContactEvidenceSampler(ContactSource(count=2)),
            lambda: torch.zeros(2),
            lambda: 1.0,
        )


@pytest.mark.parametrize("change", ["window", "clock"])
def test_joint_contact_observer_rejects_changed_window_or_clock_during_acquisition(
    change,
):
    from embodichain.lab.sim.sensors.contact import (
        JointContactObserver,
    )

    source = ContactSource()
    holder, clock = [], [1.0]

    def read_joint():
        if change == "window":
            holder[0].begin_window("replacement")
        else:
            clock[0] += 0.004
        return torch.zeros(1)

    observer = JointContactObserver(
        ContactEvidenceSampler(source), read_joint, lambda: clock[0]
    )
    holder.append(observer)
    observer.begin_window("original")
    message = "window changed" if change == "window" else "Simulation time changed"
    with pytest.raises(ValueError, match=message):
        observer.capture(refresh=True)
    assert observer.snapshot() == ()


def test_joint_contact_duration_limit_preserves_owned_prefix_and_stops_acquisition():
    from embodichain.lab.sim.sensors.contact import JointContactObserver

    source, clock = ContactSource(), [0.0]
    observer = JointContactObserver(
        ContactEvidenceSampler(source), lambda: torch.zeros(1), lambda: clock[0]
    )
    generation = observer.begin_window("bounded", max_duration=0.01)
    observer.capture(refresh=True)
    retained = observer.snapshot()
    clock[0] = 0.01
    observer.capture(refresh=True)
    assert len(observer.snapshot()) == 2
    clock[0] = 0.02
    observer.capture(refresh=True)
    assert observer.overflow and source.update_count == 2
    assert len(observer.snapshot()) == 2
    observer.begin_window("new")
    assert observer.generation != generation and observer.snapshot() == ()
    assert len(retained) == 1 and retained[0].frame.timestamp == 0.0
