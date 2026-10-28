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

"""CPU flow-control contracts, including bounded child-process publication."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
import os
import threading
import time
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from embodichain.data_pipeline import runtime as runtime_module
from embodichain.data_pipeline.configuration import (
    ODSReuseProfileCfg,
    ODSRuntimeCfg,
    configure_ods,
)
from embodichain.data_pipeline.datasets import (
    DataBudgetExhaustedError,
    DataUnavailableError,
    OnlineDataSource,
    SequenceSpecCfg,
)
from embodichain.data_pipeline.engine import data as engine_module
from embodichain.data_pipeline.engine.data import (
    OnlineDataEngine,
    OnlineDataEngineCfg,
    OnlineDataEngineState,
)
from embodichain.data_pipeline.runtime import ODSFlowController, ODSFlowControllerCfg


def _plan():
    # Two chunks every quarter second; two trajectories with four credits each.
    return configure_ods(
        2,
        8,
        profile=ODSReuseProfileCfg(
            profile_id="runtime-cpu",
            scope="deterministic flow tests",
            qualified=False,
            min_chunks_per_episode=4,
            max_chunks_per_episode=4,
            hard_max_chunks_per_episode=8,
            latency_safety_factor=1,
        ),
        runtime=ODSRuntimeCfg(
            batch_size=2,
            producer_envs=1,
            first_publication_p95_seconds=0.5,
        ),
    )


class _Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []
        self.scheduled = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        assert seconds >= 0, "Polling must never request a negative sleep"
        self.sleeps.append(seconds)
        self.now += seconds
        due = [item for item in self.scheduled if item[0] <= self.now]
        self.scheduled = [item for item in self.scheduled if item[0] > self.now]
        for _, callback in due:
            callback()


@pytest.fixture
def clock(monkeypatch):
    result = _Clock()
    monkeypatch.setattr(runtime_module, "time", result)
    return result


class _Engine:
    def __init__(self, clock):
        plan = _plan()
        self.clock = clock
        self.buffer_size = plan.buffer_size
        self.cfg = SimpleNamespace(
            refill_policy="manual",
            episode_sample_budget=plan.episode_sample_budget,
            episode_sample_hard_limit=plan.episode_sample_hard_limit,
            gym_config={"num_envs": 1},
        )
        self.owner = os.getpid()
        self.events = []
        self.error = None
        self.health_checks = 0
        self.refills = []
        self.waits = []
        self.status = {
            "requested_generation": 1,
            "completed_generation": 1,
            "successful_episodes": self.buffer_size,
            "unlocked_remaining_chunks": 8,
            "total_delivered_chunks": 0,
        }

    def _require_owner_process(self, operation):
        if os.getpid() != self.owner:
            raise RuntimeError(f"{operation} requires owner process")

    def check_health(self):
        self.health_checks += 1
        if self.error is not None:
            raise self.error

    def drain_lifecycle_events(self):
        events, self.events = self.events, []
        return events

    def get_flow_status(self):
        self.check_health()
        return dict(self.status)

    def request_refill(self):
        if self.status["requested_generation"] == self.status["completed_generation"]:
            self.status["requested_generation"] += 1
        generation = self.status["requested_generation"]
        self.refills.append(generation)
        return generation

    def publish(self):
        generation = self.status["requested_generation"]
        self.status.update(
            completed_generation=generation,
            successful_episodes=self.status["successful_episodes"] + self.buffer_size,
            unlocked_remaining_chunks=8,
        )
        self.events.append({"kind": "published", "generation": generation})

    def wait_for_generation(self, generation, timeout):
        self.waits.append((generation, timeout))
        if generation > self.status["requested_generation"]:
            raise ValueError("generation has not been requested")
        if self.status["completed_generation"] >= generation:
            return
        self.clock.sleep(timeout)
        if self.status["completed_generation"] < generation:
            raise TimeoutError("publication pending")


class _Source:
    def __init__(self, engine, responses):
        self.engine = engine
        self.responses = deque(responses)
        self.calls = 0

    def sample_context(self, *args, **kwargs):
        self.calls += 1
        response = (
            self.responses[0] if len(self.responses) == 1 else self.responses.popleft()
        )
        if isinstance(response, BaseException):
            raise response
        return response


@pytest.mark.parametrize(
    "field", ["data_wait_timeout", "generation_timeout", "poll_interval"]
)
@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf")])
def test_wait_bounds_must_be_positive_finite_numbers(field, value):
    with pytest.raises(ValueError, match=field):
        ODSFlowControllerCfg(**{field: value})


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_production_limit_requires_a_positive_integer(value):
    with pytest.raises(ValueError, match="max_generated_episodes"):
        ODSFlowControllerCfg(max_generated_episodes=value)


def test_controller_requires_creator_engine_manual_refill_and_matching_plan(clock):
    engine = _Engine(clock)
    engine.owner = -1
    with pytest.raises(RuntimeError, match="owner"):
        ODSFlowController(engine)
    engine.owner = os.getpid()
    engine.cfg.refill_policy = "sample_threshold"
    with pytest.raises(ValueError, match="manual"):
        ODSFlowController(engine)
    engine.cfg.refill_policy = "manual"
    with pytest.raises(ValueError, match="initial cache"):
        ODSFlowController(engine, ODSFlowControllerCfg(max_generated_episodes=1))
    with pytest.raises(ValueError, match="match the plan"):
        ODSFlowController(engine, plan=replace(_plan(), buffer_size=3))
    with pytest.raises(ValueError, match="match the plan"):
        ODSFlowController(engine, plan=replace(_plan(), episode_sample_budget=3))


@pytest.mark.parametrize("planned", [True, False])
@pytest.mark.parametrize("producers", [None, 0, True, 1.5, 2])
def test_protected_flow_requires_explicit_divisible_producer_count(
    clock, planned, producers
):
    engine = _Engine(clock)
    engine.buffer_size = 3  # A two-environment writer would overlap the final row.
    engine.cfg.gym_config = {} if producers is None else {"num_envs": producers}
    with pytest.raises(ValueError, match="explicit num_envs dividing the cache"):
        ODSFlowController(
            engine,
            ODSFlowControllerCfg(max_generated_episodes=None if planned else 6),
            plan=replace(_plan(), buffer_size=3) if planned else None,
        )


def test_divisible_producer_count_must_still_match_measured_plan(clock):
    engine = _Engine(clock)
    # Both one and two producers divide this cache, but the plan measured one.
    engine.cfg.gym_config["num_envs"] = 2
    with pytest.raises(ValueError, match="producer count.*match the plan"):
        ODSFlowController(engine, plan=_plan())


@pytest.mark.parametrize(
    "operation",
    ["poll", "drain_events", "request_refill", "prepare_batch", "wait_for_generation"],
)
def test_control_operations_reject_other_threads_before_touching_engine(
    clock, operation
):
    engine = _Engine(clock)
    flow = ODSFlowController(engine)
    errors = []

    def other_thread():
        try:
            method = getattr(flow, operation)
            (
                method(1)
                if operation in {"prepare_batch", "wait_for_generation"}
                else method()
            )
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=other_thread)
    thread.start()
    thread.join(2)
    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], RuntimeError)
    assert "owner-process/thread" in str(errors[0])
    assert engine.health_checks == 0 and engine.refills == [] and engine.waits == []


def test_unplanned_prepare_does_not_implicitly_request_data(clock):
    engine = _Engine(clock)
    engine.status["unlocked_remaining_chunks"] = 0
    flow = ODSFlowController(engine)
    assert flow.prepare_batch(3)
    assert engine.refills == [] and clock.sleeps == []


def test_controller_rejects_inherited_process_ownership(clock, monkeypatch):
    engine = _Engine(clock)
    flow = ODSFlowController(engine)
    monkeypatch.setattr(
        runtime_module, "os", SimpleNamespace(getpid=lambda: os.getpid() + 1)
    )
    with pytest.raises(RuntimeError, match="owner-process/thread"):
        flow.poll()
    assert engine.health_checks == 0


def test_planned_batch_size_mismatch_fails_before_requesting_data(clock):
    engine = _Engine(clock)
    flow = ODSFlowController(engine, plan=_plan())
    with pytest.raises(ValueError, match="measured plan"):
        flow.prepare_batch(3)
    assert engine.refills == [] and engine.health_checks == 0


def test_refill_observer_reports_each_coalesced_generation_once(clock):
    engine = _Engine(clock)
    observed = []
    flow = ODSFlowController(engine, on_refill=lambda *event: observed.append(event))
    assert flow.request_refill() == flow.request_refill() == 2
    engine.publish()
    clock.sleep(1)
    assert flow.request_refill() == 3
    assert observed == [(2, 0), (3, 1)]


def test_pending_refill_completion_race_does_not_start_an_extra_generation(
    clock, monkeypatch
):
    engine = _Engine(clock)
    engine.request_refill()
    engine.refills.clear()
    observed = []
    flow = ODSFlowController(
        engine,
        ODSFlowControllerCfg(max_generated_episodes=4),
        on_refill=lambda *event: observed.append(event),
    )
    status = engine.get_flow_status

    def complete_after_snapshot():
        snapshot = status()
        if snapshot["requested_generation"] > snapshot["completed_generation"]:
            engine.publish()
        return snapshot

    monkeypatch.setattr(engine, "get_flow_status", complete_after_snapshot)
    assert flow.request_refill() == 2
    assert engine.status["completed_generation"] == 2
    assert engine.status["requested_generation"] == 2
    assert engine.status["successful_episodes"] == 4
    assert engine.refills == []
    # Adopting another caller's earlier request must not fabricate its timestamp.
    assert observed == []
    assert flow.request_refill() is None


def test_adopting_pending_refill_preserves_only_actual_request_timestamps(clock):
    engine = _Engine(clock)
    engine.request_refill()
    observed = []
    flow = ODSFlowController(engine, on_refill=lambda *event: observed.append(event))
    clock.sleep(3)
    assert flow.request_refill() == 2
    assert flow.request_refill() == 2
    assert observed == []
    engine.publish()
    clock.sleep(1)
    assert flow.request_refill() == 3
    assert observed == [(3, 4)]


def test_partial_pending_fill_is_coalesced_even_when_another_cache_would_not_fit(clock):
    engine = _Engine(clock)
    flow = ODSFlowController(engine, ODSFlowControllerCfg(max_generated_episodes=4))
    assert flow.request_refill() == 2
    engine.status["successful_episodes"] = 3
    assert flow.request_refill() == 2
    assert engine.status["requested_generation"] == 2


def test_production_ceiling_allows_only_complete_caches(clock):
    engine = _Engine(clock)
    # Initial2 + one refill2 fits5; the unused final capacity1 must not be filled.
    flow = ODSFlowController(engine, ODSFlowControllerCfg(max_generated_episodes=5))
    assert flow.request_refill() == 2
    assert flow.request_refill() == 2
    engine.publish()
    assert flow.request_refill() is None
    assert engine.status["successful_episodes"] == 4


def test_remaining_final_supply_is_consumed_before_stopping(clock):
    engine = _Engine(clock)
    flow = ODSFlowController(
        engine, ODSFlowControllerCfg(max_generated_episodes=2), plan=_plan()
    )
    engine.status["unlocked_remaining_chunks"] = 2
    assert flow.prepare_batch(2)
    engine.status["unlocked_remaining_chunks"] = 1
    assert flow.prepare_batch(2) is False
    assert engine.refills == [] and clock.sleeps == []


def test_quota_wait_drains_publications_then_funds_a_batch(clock):
    engine = _Engine(clock)
    engine.status["unlocked_remaining_chunks"] = 0
    publications, intervals = [], []
    flow = ODSFlowController(
        engine,
        ODSFlowControllerCfg(data_wait_timeout=1, poll_interval=0.1),
        plan=_plan(),
        on_events=publications.extend,
        on_interval=lambda *interval: intervals.append(interval),
    )
    clock.scheduled.append((0.2, engine.publish))
    assert flow.prepare_batch(2)
    assert publications == [{"kind": "published", "generation": 2}]
    assert intervals[0] == ("quota_wait", 0, pytest.approx(0.2))
    assert engine.refills == [2]


def test_quota_wait_has_one_total_deadline(clock):
    engine = _Engine(clock)
    engine.status["unlocked_remaining_chunks"] = 0
    intervals = []
    flow = ODSFlowController(
        engine,
        ODSFlowControllerCfg(data_wait_timeout=0.25, poll_interval=0.1),
        plan=_plan(),
        on_interval=lambda *interval: intervals.append(interval),
    )
    with pytest.raises(TimeoutError, match="quota"):
        flow.prepare_batch(2)
    assert clock.now == pytest.approx(0.25)
    assert intervals == [("quota_wait", 0, pytest.approx(0.25))]


def test_pacing_uses_slower_cap_without_catch_up_after_slow_service(clock):
    plan = replace(_plan(), active_chunks_per_second_cap=4)
    flow = ODSFlowController(_Engine(clock), plan=plan)
    assert flow.prepare_batch(2)
    clock.sleep(10)  # Long optimizer/evaluation pause earns no burst allowance.
    assert flow.prepare_batch(2)
    assert clock.now == 10
    assert flow.prepare_batch(2)
    assert clock.now == pytest.approx(10.5)


def test_poll_callback_crossing_pacing_deadline_never_sleeps_negative(clock):
    callbacks = []
    armed = False

    def progress():
        if armed:
            callbacks.append(clock.now)
            if len(callbacks) == 2:
                clock.sleep(0.3)

    flow = ODSFlowController(_Engine(clock), plan=_plan(), progress=progress)
    assert flow.prepare_batch(2)
    armed = True
    assert flow.prepare_batch(2)
    assert clock.now == pytest.approx(0.3)
    assert flow.prepare_batch(2)
    assert clock.now == pytest.approx(0.55)


@pytest.mark.parametrize("external_deadline,expected", [(None, 0.4), (0.15, 0.15)])
def test_generation_wait_uses_total_or_shorter_absolute_deadline(
    clock, external_deadline, expected
):
    engine = _Engine(clock)
    engine.request_refill()
    flow = ODSFlowController(
        engine, ODSFlowControllerCfg(generation_timeout=0.4, poll_interval=0.1)
    )
    with pytest.raises(TimeoutError, match="deadline"):
        flow.wait_for_generation(2, deadline=external_deadline)
    assert clock.now == pytest.approx(expected)
    assert all(0 < duration <= 0.1 for _, duration in engine.waits)


def test_generation_wait_releases_events_and_propagates_invalid_generation(clock):
    engine = _Engine(clock)
    engine.request_refill()
    events = []
    flow = ODSFlowController(
        engine, ODSFlowControllerCfg(poll_interval=0.1), on_events=events.extend
    )
    clock.scheduled.append((0.2, engine.publish))
    flow.wait_for_generation(2)
    assert events == [{"kind": "published", "generation": 2}]
    with pytest.raises(ValueError, match="not been requested"):
        flow.wait_for_generation(3)


@pytest.mark.parametrize("wait", ["quota", "pacing", "generation", "sample"])
def test_application_progress_can_interrupt_every_wait(clock, wait):
    class Cancelled(Exception):
        pass

    def progress():
        if clock.now >= 0.1:
            raise Cancelled("application deadline")

    engine = _Engine(clock)
    flow = ODSFlowController(engine, plan=_plan(), progress=progress)
    if wait == "quota":
        engine.status["unlocked_remaining_chunks"] = 0
        operation = lambda: flow.prepare_batch(2)
    elif wait == "pacing":
        flow.prepare_batch(2)
        operation = lambda: flow.prepare_batch(2)
    elif wait == "generation":
        engine.request_refill()
        operation = lambda: flow.wait_for_generation(2)
    else:
        source = _Source(engine, [DataUnavailableError("pending")])
        operation = lambda: flow.sample_context(source, 2, SequenceSpecCfg())
    with pytest.raises(Cancelled, match="application deadline"):
        operation()
    assert clock.now <= 0.12


def test_transient_sampling_retries_drain_events_without_changing_request(clock):
    engine = _Engine(clock)
    events, intervals = [], []
    batch = object()
    source = _Source(
        engine,
        [
            DataUnavailableError("locked"),
            DataBudgetExhaustedError("pending quota"),
            batch,
        ],
    )
    flow = ODSFlowController(
        engine,
        ODSFlowControllerCfg(poll_interval=0.1),
        on_events=events.extend,
        on_interval=lambda *interval: intervals.append(interval),
    )
    clock.scheduled.append((0.1, lambda: engine.events.append({"kind": "published"})))
    assert flow.sample_context(source, 2, SequenceSpecCfg()) is batch
    assert source.calls == 3 and engine.refills == []
    assert events == [{"kind": "published"}]
    assert intervals == [("data_wait", 0, pytest.approx(0.2))]


def test_sampling_retry_deadline_does_not_reset_after_each_transient_error(clock):
    engine = _Engine(clock)
    source = _Source(engine, [DataUnavailableError("no eligible window")])
    flow = ODSFlowController(
        engine, ODSFlowControllerCfg(data_wait_timeout=0.25, poll_interval=0.1)
    )
    with pytest.raises(TimeoutError, match="context"):
        flow.sample_context(source, 2, SequenceSpecCfg())
    assert clock.now == pytest.approx(0.25)
    assert source.calls == 4


@pytest.mark.parametrize("worker_failure", [True, False])
def test_worker_and_schema_errors_are_propagated_without_retry(clock, worker_failure):
    engine = _Engine(clock)
    original = ValueError("worker failed" if worker_failure else "schema mismatch")
    source = _Source(engine, [original])
    if worker_failure:
        engine.error = original
    flow = ODSFlowController(engine)
    with pytest.raises(ValueError) as caught:
        flow.sample_context(source, 2, SequenceSpecCfg())
    assert caught.value is original
    assert source.calls == (0 if worker_failure else 1)
    assert clock.sleeps == []


def test_interval_observer_failure_retains_original_worker_error(clock):
    engine = _Engine(clock)
    original = ValueError("worker planner failed")
    engine.error = original
    source = _Source(engine, [object()])

    def observer(*interval):
        raise RuntimeError("interval export failed")

    flow = ODSFlowController(engine, on_interval=observer)
    with pytest.raises(ValueError) as caught:
        flow.sample_context(source, 2, SequenceSpecCfg())
    assert caught.value is original
    assert source.calls == 0
    assert any("interval export failed" in note for note in original.__notes__)


def test_interval_observer_failure_without_primary_error_is_propagated(clock):
    engine = _Engine(clock)
    source = _Source(engine, [object()])
    original = RuntimeError("interval export failed")

    def observer(*interval):
        raise original

    flow = ODSFlowController(engine, on_interval=observer)
    with pytest.raises(RuntimeError) as caught:
        flow.sample_context(source, 2, SequenceSpecCfg())
    assert caught.value is original
    assert source.calls == 1
    assert clock.sleeps == []


def test_diagnostic_events_remain_drainable_after_worker_failure(clock):
    engine = _Engine(clock)
    original = RuntimeError("worker exited")
    engine.error = original
    engine.events.append({"kind": "published", "generation": 1})
    events = []
    flow = ODSFlowController(engine, on_events=events.extend)
    with pytest.raises(RuntimeError) as caught:
        flow.poll()
    assert caught.value is original
    engine.events.append({"kind": "evicted", "generation": 1})
    assert flow.drain_events() == [{"kind": "evicted", "generation": 1}]
    assert events == [
        {"kind": "published", "generation": 1},
        {"kind": "evicted", "generation": 1},
    ]


def test_sampling_rejects_an_uncontrolled_source(clock):
    engine = _Engine(clock)
    source = _Source(_Engine(clock), [object()])
    with pytest.raises(ValueError, match="controlled live engine"):
        ODSFlowController(engine).sample_context(source, 2, SequenceSpecCfg())
    assert source.calls == 0


def _buffer(engine):
    rows, steps = engine.cfg.buffer_size, 4
    shape = rows, steps
    row_ids = torch.arange(rows, dtype=torch.int64)[:, None].expand(shape)
    return TensorDict(
        {
            "obs": torch.arange(rows * steps * 2, dtype=torch.float32).reshape(
                *shape, 2
            ),
            "actions": torch.zeros(*shape, 1),
            "valid": torch.ones(shape, dtype=torch.bool),
            "continuity_id": torch.zeros(shape, dtype=torch.int64),
            "segment_id": torch.zeros(shape, dtype=torch.int64),
            "segment_accepted": torch.ones(shape, dtype=torch.bool),
            "ods_run_id": torch.full(shape, engine.run_id, dtype=torch.int64),
            "ods_episode_id": row_ids.clone(),
            "ods_generation": torch.ones(shape, dtype=torch.int64),
            "ods_frame_index": torch.arange(steps).expand(shape).clone(),
        },
        batch_size=shape,
        device="cpu",
    ).share_memory_()


_SCHEMA = {
    "observation": {"names": ["position", "speed"]},
    "action": {"names": ["request"], "semantics": "policy_request", "step_dt": 1 / 30},
}


def _engine_cfg(**changes):
    values = dict(
        buffer_size=2,
        max_episode_steps=4,
        state_dim=2,
        action_dim=1,
        action_semantics="policy_request",
        action_contracts=("gripper.parallel@1",),
        gym_config={"num_envs": 1},
        include_provenance=True,
        refill_policy="manual",
        initialization_timeout=15,
        shutdown_timeout=2,
    )
    return OnlineDataEngineCfg(**(values | changes))


def test_flow_preserves_actual_source_batches_padding_and_independent_rng(monkeypatch):
    monkeypatch.setattr(OnlineDataEngine, "_create_buffer", _buffer)
    engine = OnlineDataEngine(_engine_cfg())
    # This test exercises source sampling, while worker liveness is covered below.
    monkeypatch.setattr(engine, "_ensure_worker_alive", lambda: None)
    engine._set_state(OnlineDataEngineState.READY)
    engine._init_signal.set()
    with engine._lock_index.get_lock():
        engine._lock_index[:] = [-1, -1]
    try:
        source = OnlineDataSource(engine, schema=_SCHEMA)
        flow = ODSFlowController(engine)
        sequence = SequenceSpecCfg(history_length=2, prediction_horizon=4)
        direct_rng = torch.Generator().manual_seed(91)
        flow_rng = torch.Generator().manual_seed(91)
        global_rng = torch.get_rng_state().clone()
        direct = source.sample_context(64, sequence, generator=direct_rng)
        actual = flow.sample_context(source, 64, sequence, generator=flow_rng)
        assert set(direct.keys()) == set(actual.keys())
        assert all(torch.equal(direct[key], actual[key]) for key in direct.keys())
        assert torch.equal(direct_rng.get_state(), flow_rng.get_state())
        assert torch.equal(torch.get_rng_state(), global_rng)
        assert not actual["loss_mask"][:, 0].any()
        assert not actual["valid"].all()
    finally:
        engine._set_state(OnlineDataEngineState.CREATED)
        engine.stop()


def _numeric_worker(
    cfg,
    shared,
    lock,
    fill,
    init,
    close,
    error_buffer,
    error_length,
    failed,
    state,
    language,
    progress,
    run_id,
    events,
    quota,
):
    """Publish numeric trajectories through real engine IPC, without a simulator."""
    previous = {}
    try:
        while not close.is_set():
            if not fill.wait(0.05):
                continue
            fill.clear()
            with progress.values.get_lock():
                generation = int(progress.values[0])
            for slot in range(len(shared)):
                required = 2 if slot in previous else 1
                with events.condition:
                    if (
                        events.cursors[1] - events.cursors[0] + required
                        > events.capacity
                    ):
                        cfg.gym_config["backpressure_observed"].set()
                if not events.wait_for_space(required, close):
                    return
                with lock.get_lock():
                    lock[:] = [slot, slot + 1]
                    if slot in previous:
                        events.append(previous[slot] | {"kind": "evicted"})
                    shared["obs"][slot].fill_(generation)
                    shared["ods_episode_id"][slot].fill_(
                        generation * len(shared) + slot
                    )
                    shared["ods_generation"][slot].fill_(generation)
                    event = engine_module._publication_event(
                        shared[slot],
                        run_id=run_id,
                        episode_id=generation * len(shared) + slot,
                        generation=generation,
                        slot=slot,
                        monotonic_ns=time.monotonic_ns(),
                    )
                    events.append(event)
                    previous[slot] = event
                    lock[:] = [-1, -1]
                with progress.values.get_lock():
                    progress.values[2] += 1
            with progress.values.get_lock():
                progress.values[1] = generation
            init.set()
    except BaseException as error:
        engine_module._publish_worker_error(
            error_buffer, error_length, failed, state, error
        )


def test_generation_wait_drains_a_full_bounded_log_from_a_real_child(monkeypatch):
    import multiprocessing as mp

    context = mp.get_context("forkserver")
    observed = context.Event()
    monkeypatch.setattr(OnlineDataEngine, "_create_buffer", _buffer)
    monkeypatch.setattr(engine_module, "_sim_worker_fn", _numeric_worker)
    engine = OnlineDataEngine(_engine_cfg(lifecycle_event_capacity=4))
    # Test-only IPC handle: configclass copies ordinary authored configuration.
    engine.cfg.gym_config["backpressure_observed"] = observed
    events = []
    try:
        engine.start()
        # Request directly so the initial publication records remain in the log.
        generation = engine.request_refill()
        assert observed.wait(10), "Producer did not reach the bounded-log barrier"
        flow = ODSFlowController(
            engine, ODSFlowControllerCfg(generation_timeout=5), on_events=events.extend
        )
        flow.wait_for_generation(generation)
        assert [event["sequence"] for event in events] == list(range(6))
        assert [event["kind"] for event in events].count("evicted") == 2
        source = OnlineDataSource(engine, schema=_SCHEMA)
        batch = flow.sample_context(source, 4, SequenceSpecCfg())
        assert (batch["generations"][batch["valid"]] == generation).all()
        assert (batch["obs"][batch["valid"]] == generation).all()
    finally:
        worker_pid = None if engine._sim_process is None else engine._sim_process.pid
        engine.stop()
    assert engine.state is OnlineDataEngineState.STOPPED
    assert worker_pid not in {child.pid for child in mp.active_children()}
