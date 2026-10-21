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

"""Unit tests for OnlineDataset and OnlineDataEngine.

These tests do **not** start a real simulation subprocess. ``_make_fake_engine``
injects a pre-filled shared buffer for sampling tests, while lifecycle tests
drive ``start()`` with a lightweight process mock and a real multiprocessing
error pipe.

This exercises all public logic in ``sample_batch``,
``_trigger_refill_if_needed``, and ``OnlineDataset.__iter__`` without GPU or
sim dependencies.
"""

from __future__ import annotations

import multiprocessing as mp
from unittest.mock import MagicMock, patch

import pytest
import torch
from tensordict import TensorDict

from embodichain.data_pipeline.engine import data as data_module
from embodichain.data_pipeline.engine.data import (
    OnlineDataBudgetExhaustedError,
    OnlineDataEngine,
    OnlineDataEngineCfg,
    OnlineDataEngineState,
)


def _buffer(engine: OnlineDataEngine) -> TensorDict:
    rows, steps = engine.cfg.buffer_size, 4
    row_ids = torch.arange(rows)[:, None].expand(rows, steps).clone()
    result = TensorDict(
        {
            "obs": row_ids[..., None].expand(rows, steps, 2).float().clone(),
            "actions": torch.zeros(rows, steps, 2),
            "valid": torch.ones(rows, steps, dtype=torch.bool),
            "continuity_id": torch.zeros(rows, steps, dtype=torch.int64),
            "segment_id": torch.tensor([0, 0, 1, 1]).expand(rows, steps).clone(),
            "segment_accepted": torch.ones(rows, steps, dtype=torch.bool),
            "ods_run_id": torch.full((rows, steps), engine.run_id, dtype=torch.int64),
            "ods_episode_id": row_ids,
        },
        batch_size=[rows, steps],
        device="cpu",
    )
    return result.share_memory_()


def _engine(
    rows: int = 2, budget: int | None = 3, hard: int | None = 3
) -> OnlineDataEngine:
    cfg = OnlineDataEngineCfg(
        buffer_size=rows,
        state_dim=2,
        max_episode_steps=4,
        gym_config={"num_envs": 1},
        refill_policy="manual",
        episode_sample_budget=budget,
        episode_sample_hard_limit=hard,
    )
    with patch.object(OnlineDataEngine, "_create_buffer", _buffer):
        engine = OnlineDataEngine(cfg)
    # CPU-only consumer fixture: no simulation process exists or is required.
    engine._owner_pid = -1
    engine._set_state(OnlineDataEngineState.READY)
    engine._init_signal.set()
    with engine._lock_index.get_lock():
        engine._lock_index[:] = [-1, -1]
        if engine._sampling_budget is not None:
            engine._sampling_budget.publish(0, rows, 0, 1)
    return engine


def _sources(batch: TensorDict) -> list[int]:
    return batch["ods_episode_id"].amax(dim=1).tolist()


@pytest.mark.parametrize(
    "budget,hard",
    [
        (None, 3),
        (3, None),
        (0, 1),
        (2, 1),
        (True, 3),
        (1, False),
        (1.5, 3),
        (1, 1 << 63),
    ],
)
def test_configuration_requires_bounded_integer_pair(budget, hard) -> None:
    with pytest.raises(ValueError, match="episode_sample"):
        _engine(budget=budget, hard=hard)


def test_quota_mode_rejects_non_cpu_buffer_before_allocating() -> None:
    with pytest.raises(ValueError, match="CPU|cpu"):
        OnlineDataEngine(
            OnlineDataEngineCfg(
                buffer_device="cuda:0",
                episode_sample_budget=1,
                episode_sample_hard_limit=1,
            )
        )


def test_duplicate_chunks_spend_source_quota_and_exhaust_without_partial_charge() -> (
    None
):
    engine = _engine(rows=1)
    with pytest.raises(OnlineDataBudgetExhaustedError):
        engine.sample_batch(4, 1)
    assert engine.get_sampling_budget_snapshot()["total_delivered_chunks"] == 0
    assert _sources(engine.sample_batch(3, 1)) == [0, 0, 0]
    snapshot = engine.get_sampling_budget_snapshot()
    assert snapshot["rows"][0]["delivered"] == 3
    assert snapshot["rows"][0]["remaining"] == 0
    with pytest.raises(OnlineDataBudgetExhaustedError):
        engine.sample_batch(1, 1)
    assert engine.get_sampling_budget_snapshot()["total_delivered_chunks"] == 3


@pytest.mark.parametrize("method", ["plain", "context", "segment", "boundary"])
def test_every_sampling_api_obeys_shared_episode_budget(method: str) -> None:
    engine = _engine(rows=2, budget=1, hard=1)
    if method == "plain":
        batch = engine.sample_batch(2, 2)
    elif method == "context":
        batch = engine.sample_batch_with_context(2, 2, 3)
    else:
        batch = engine.sample_batch(2, 2, sampling_mode=method)
    assert sorted(_sources(batch)) == [0, 1]
    assert [
        row["delivered"] for row in engine.get_sampling_budget_snapshot()["rows"]
    ] == [1, 1]
    with pytest.raises(OnlineDataBudgetExhaustedError):
        engine.sample_batch_with_context(1, 2, 3)


def test_failed_payload_copy_does_not_spend_quota(monkeypatch) -> None:
    engine = _engine()
    before = engine.get_sampling_budget_snapshot()
    original_getitem = TensorDict.__getitem__

    def fail_payload_gather(buffer, index):
        if (
            buffer is engine.shared_buffer
            and isinstance(index, tuple)
            and all(isinstance(value, torch.Tensor) for value in index)
        ):
            raise RuntimeError("copy failed")
        return original_getitem(buffer, index)

    with monkeypatch.context() as context:
        context.setattr(TensorDict, "__getitem__", fail_payload_gather)
        with pytest.raises(RuntimeError, match="copy failed"):
            engine.sample_batch(2, 2)
    assert engine.get_sampling_budget_snapshot() == before
    assert len(engine.sample_batch(2, 2)) == 2


def test_replacement_assigns_only_fresh_source_and_keeps_lifetime_delivered() -> None:
    engine = _engine(rows=2, budget=2, hard=3)
    engine.sample_batch(4, 1)
    with engine._lock_index.get_lock():
        engine._sampling_budget.publish(0, 1, 2, 1)  # Same generation, fresh ID.
        engine.shared_buffer["ods_episode_id"][0].fill_(2)
    rows = engine.get_sampling_budget_snapshot()["rows"]
    assert [
        (row["episode_id"], row["delivered"], row["remaining"]) for row in rows
    ] == [(2, 0, 2), (1, 2, 0)]
    assert _sources(engine.sample_batch(2, 1)) == [2, 2]
    with engine._lock_index.get_lock():
        with pytest.raises(ValueError, match="fresh"):
            engine._sampling_budget.publish(
                0, 1, 2, 99
            )  # New generation grants nothing.
        with pytest.raises(ValueError, match="fresh"):
            engine._sampling_budget.publish(
                1, 1, 0, 99
            )  # Retired ID cannot be recycled.
    snapshot = engine.get_sampling_budget_snapshot()
    assert snapshot["total_delivered_chunks"] == 6
    assert snapshot["unlocked_remaining_chunks"] == 0


def test_locked_or_ineligible_rows_do_not_lend_remaining_quota() -> None:
    engine = _engine(rows=3, budget=2, hard=2)
    with engine._lock_index.get_lock():
        engine._lock_index[:] = [0, 1]
        engine.shared_buffer["continuity_id"][1].fill_(-1)
    assert engine.get_sampling_budget_snapshot()["unlocked_remaining_chunks"] == 2
    with pytest.raises(OnlineDataBudgetExhaustedError):
        engine.sample_batch(3, 1)
    assert _sources(engine.sample_batch(2, 1)) == [2, 2]
    assert [
        row["delivered"] for row in engine.get_sampling_budget_snapshot()["rows"]
    ] == [0, 0, 2]


def test_copy_after_delivery_is_charged_even_if_optimizer_fails() -> None:
    engine = _engine(rows=1, budget=2, hard=2)
    batch = engine.sample_batch(1, 1)
    del batch  # The consumer rejects/discards delivery; quota cannot be refunded.
    assert engine.get_sampling_budget_snapshot()["rows"][0]["remaining"] == 1


def test_defaults_are_unrestricted_and_stopped_snapshot_is_detached() -> None:
    engine = _engine(rows=1, budget=None, hard=None)
    assert len(engine.sample_batch(99, 1)) == 99
    assert engine.get_sampling_budget_snapshot()["enabled"] is False
    assert engine.get_sampling_budget_snapshot()["total_delivered_chunks"] is None
    budgeted = _engine()
    budgeted.sample_batch(1, 1)
    budgeted._set_state(OnlineDataEngineState.STOPPED)
    first = budgeted.get_sampling_budget_snapshot()
    first["rows"][0]["remaining"] = -999
    assert budgeted.get_sampling_budget_snapshot()["rows"][0]["remaining"] >= 0


def test_worker_error_has_priority_over_exhaustion_and_snapshot() -> None:
    engine = _engine(rows=1, budget=1, hard=1)
    engine.sample_batch(1, 1)
    engine._record_worker_error(RuntimeError("producer exploded"))
    with pytest.raises(RuntimeError, match="producer exploded") as caught:
        engine.sample_batch(1, 1)
    assert not isinstance(caught.value, OnlineDataBudgetExhaustedError)
    with pytest.raises(RuntimeError, match="producer exploded"):
        engine.get_sampling_budget_snapshot()


def test_worker_failure_during_selection_beats_budget_error(monkeypatch) -> None:
    engine = _engine(rows=1, budget=1, hard=1)
    engine.sample_batch(1, 1)
    original = engine._sampling_budget.remaining

    def fail_then_remaining(rows):
        engine._record_worker_error(RuntimeError("producer failed during selection"))
        return original(rows)

    monkeypatch.setattr(engine._sampling_budget, "remaining", fail_then_remaining)
    with pytest.raises(RuntimeError, match="during selection") as caught:
        engine.sample_batch(1, 1)
    assert not isinstance(caught.value, OnlineDataBudgetExhaustedError)


def _consume_budget(engine, ready, output) -> None:
    ready.wait(timeout=20)
    counts = {}
    while True:
        try:
            batch = engine.sample_batch_with_context(2, 2, 3)
        except OnlineDataBudgetExhaustedError:
            break
        for episode in _sources(batch):
            counts[episode] = counts.get(episode, 0) + 1
    output.put(counts)


def test_real_concurrent_consumers_never_overspend_source_quota() -> None:
    engine = _engine(rows=3, budget=7, hard=7)
    context = mp.get_context("forkserver")
    ready = context.Barrier(4)
    output = context.Queue()
    processes = [
        context.Process(target=_consume_budget, args=(engine, ready, output))
        for _ in range(3)
    ]
    try:
        for process in processes:
            process.start()
        ready.wait(timeout=20)
        counts = {0: 0, 1: 0, 2: 0}
        for _ in processes:
            for source, count in output.get(timeout=20).items():
                counts[source] += count
        for process in processes:
            process.join(10)
            assert process.exitcode == 0
        assert sum(counts.values()) == 20  # One unused credit cannot fund batch=2.
        assert all(value <= 7 for value in counts.values())
        snapshot = engine.get_sampling_budget_snapshot()
        assert snapshot["total_delivered_chunks"] == 20
        assert snapshot["unlocked_remaining_chunks"] == 1
        assert {
            row["episode_id"]: row["delivered"] for row in snapshot["rows"]
        } == counts
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
        output.close()
        output.join_thread()


def test_snapshot_raises_local_startup_failure_without_shared_envelope() -> None:
    engine = _engine()
    engine._worker_error = TimeoutError("initial fill timed out")
    engine._set_state(OnlineDataEngineState.FAILED)
    with pytest.raises(TimeoutError, match="initial fill"):
        engine.get_sampling_budget_snapshot()


def test_snapshot_checks_ready_worker_liveness(monkeypatch) -> None:
    engine = _engine()

    def died() -> None:
        raise RuntimeError("worker exited before monitor notification")

    monkeypatch.setattr(engine, "_ensure_worker_alive", died)
    with pytest.raises(RuntimeError, match="before monitor"):
        engine.get_sampling_budget_snapshot()


def test_real_producer_loop_publishes_fresh_quotas_and_preserves_lifetime(
    monkeypatch,
) -> None:
    import sys
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    engine = _engine(rows=2, budget=2, hard=2)
    context = engine._mp_ctx
    engine._sampling_budget = data_module._EpisodeSamplingBudget(
        context, 2, 2, 2, engine.run_id
    )
    original_publish = engine._sampling_budget.publish

    def publish_while_locked(*args, **kwargs):
        assert engine._lock_index.get_lock()._semlock._is_mine()
        original_publish(*args, **kwargs)

    monkeypatch.setattr(engine._sampling_budget, "publish", publish_while_locked)
    env = MagicMock()
    config = SimpleNamespace(
        num_envs=1,
        sim_cfg=SimpleNamespace(
            headless=True,
            render_cfg=SimpleNamespace(renderer=None),
            gpu_id=0,
            device=None,
        ),
    )
    monkeypatch.setitem(
        sys.modules, "gymnasium", SimpleNamespace(make=lambda **kwargs: env)
    )
    monkeypatch.setitem(
        sys.modules,
        "embodichain.lab.gym.utils.gym_utils",
        SimpleNamespace(
            config_to_cfg=lambda *args, **kwargs: config, get_manager_modules=lambda: []
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "embodichain.lab.gym.utils.registration",
        SimpleNamespace(discover_task_packages=lambda: []),
    )
    calls = 0

    def execute(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            engine._close_signal.set()
            engine._fill_signal.set()
        return SimpleNamespace(completed=True, all_success=True)

    monkeypatch.setitem(
        sys.modules,
        "embodichain.lab.gym.envs.demo",
        SimpleNamespace(execute_demo_episode=execute),
    )
    monkeypatch.delenv("EMBODICHAIN_SIM_EXIT_PROCESS", raising=False)
    initialized = MagicMock()
    initialized.is_set.side_effect = [False, True]

    def consume_initial() -> None:
        engine.sample_batch(4, 1)
        assert engine.get_sampling_budget_snapshot()["unlocked_remaining_chunks"] == 0
        engine._fill_signal.set()

    initialized.set.side_effect = consume_initial
    engine._fill_signal.set()
    cfg = OnlineDataEngineCfg(
        gym_config={"id": "Fake-v1", "num_envs": 1},
        episode_sample_budget=2,
        episode_sample_hard_limit=2,
    )
    data_module._run_sim_worker(
        cfg,
        engine.shared_buffer,
        engine._lock_index,
        engine._fill_signal,
        initialized,
        engine._close_signal,
        engine._worker_error_buffer,
        engine._worker_error_length,
        engine._worker_failed_signal,
        engine._state_value,
        [False],
        progress=engine._progress,
        run_id=engine.run_id,
        sampling_budget=engine._sampling_budget,
    )
    engine._set_state(OnlineDataEngineState.STOPPED)
    snapshot = engine.get_sampling_budget_snapshot()
    assert snapshot["total_delivered_chunks"] == 4
    assert snapshot["unlocked_remaining_chunks"] == 4
    assert [
        (row["episode_id"], row["generation"], row["delivered"])
        for row in snapshot["rows"]
    ] == [(2, 2, 0), (3, 2, 0)]
    env.close.assert_called_once()


def test_nonbinding_quota_preserves_episode_uniform_sampling_seed() -> None:
    unrestricted = _engine(rows=3, budget=None, hard=None)
    budgeted = _engine(rows=3, budget=100, hard=100)
    torch.manual_seed(77)
    baseline = unrestricted.sample_batch_with_context(16, 2, 3)
    torch.manual_seed(77)
    constrained = budgeted.sample_batch_with_context(16, 2, 3)
    for key in ("obs", "actions", "valid", "loss_mask", "ods_episode_id"):
        assert torch.equal(baseline[key], constrained[key])
    assert budgeted.get_sampling_budget_snapshot()["total_delivered_chunks"] == 16


@pytest.mark.parametrize("budget,depletes", [(8, False), (1, True)])
def test_cpu_quota_sampling_ignores_non_cpu_default_device(
    budget: int, depletes: bool
) -> None:
    engine = _engine(rows=2, budget=budget, hard=budget)
    default_device = torch.get_default_device()
    with torch.random.fork_rng(devices=[]):
        # Seed 1 proposes the same episode twice. With budget=1, selection must
        # reject the duplicate and draw again from the remaining episode.
        torch.manual_seed(1)
        try:
            torch.set_default_device("meta")
            with patch.object(torch, "randint", wraps=torch.randint) as draw:
                batch = engine.sample_batch_with_context(2, 2, 3)
            assert (draw.call_count > 1) is depletes
            assert batch.device == torch.device("cpu")
            assert len(batch) == 2
            if depletes:
                assert sorted(_sources(batch)) == [0, 1]
            snapshot = engine.get_sampling_budget_snapshot()
            assert snapshot["total_delivered_chunks"] == 2
            assert all(row["delivered"] <= budget for row in snapshot["rows"])
        finally:
            torch.set_default_device(default_device)


def test_lifetime_counter_overflow_fails_without_partial_quota_spend() -> None:
    engine = _engine(rows=1, budget=3, hard=3)
    engine._sampling_budget.meta[1] = (1 << 63) - 2
    before = engine.get_sampling_budget_snapshot()
    with pytest.raises(OverflowError, match="Lifetime"):
        engine.sample_batch(2, 1)
    assert engine.get_sampling_budget_snapshot() == before
    engine.sample_batch(1, 1)
    assert (
        engine.get_sampling_budget_snapshot()["total_delivered_chunks"] == (1 << 63) - 1
    )


def test_shared_quota_draw_preserves_explicit_generator_and_global_rng() -> None:
    from embodichain.data_pipeline.datasets._sampling import _draw_budgeted_rows

    rows = torch.tensor([4, 8])
    remaining = torch.tensor([2, 3])
    global_before = torch.get_rng_state()
    first = torch.Generator().manual_seed(71)
    second = torch.Generator().manual_seed(71)
    result = _draw_budgeted_rows(rows, remaining, 5, generator=first)
    assert sorted(result.tolist()) == [4, 4, 8, 8, 8]
    assert torch.equal(
        result, _draw_budgeted_rows(rows, remaining, 5, generator=second)
    )
    assert torch.equal(first.get_state(), second.get_state())
    assert torch.equal(global_before, torch.get_rng_state())


def test_flow_status_matches_audit_across_ineligible_and_replaced_sources() -> None:
    engine = _engine(rows=6, budget=3, hard=3)

    def compare(expected_remaining: int, expected_delivered: int) -> None:
        compact = engine.get_flow_status()
        audit = engine.get_sampling_budget_snapshot()
        assert compact["unlocked_remaining_chunks"] == expected_remaining
        assert compact["total_delivered_chunks"] == expected_delivered
        assert (
            compact["unlocked_remaining_chunks"] == audit["unlocked_remaining_chunks"]
        )
        assert compact["total_delivered_chunks"] == audit["total_delivered_chunks"]

    compare(18, 0)
    with engine._lock_index.get_lock():
        engine._sampling_budget.debit(torch.tensor([0, 0, 5]))
        engine._lock_index[:] = [0, 1]
        engine.shared_buffer["valid"][1].zero_()
        engine.shared_buffer["continuity_id"][2].fill_(-1)
        # An unpublished row with stale valid payload cannot lend credits.
        engine._sampling_budget.rows[3 * 4] = -1
    compare(5, 3)
    with engine._lock_index.get_lock():
        engine._sampling_budget.publish(5, 1, 6, 2)
        engine.shared_buffer["ods_episode_id"][5].fill_(6)
    # The replacement retires unused credits and resets only its own delivery.
    compare(6, 3)
    with engine._lock_index.get_lock():
        engine._sampling_budget.debit(torch.tensor([5, 5, 5]))
    compare(3, 6)


@pytest.mark.parametrize("with_continuity", [False, True])
def test_flow_status_requires_a_joint_valid_continuity_anchor(
    with_continuity: bool,
) -> None:
    engine = _engine(rows=2, budget=3, hard=3)
    with engine._lock_index.get_lock():
        if with_continuity:
            # Separate valid frames and continuity-qualified frames do not
            # constitute one usable anchor.
            engine.shared_buffer["valid"][0] = torch.tensor([True, False, False, False])
            engine.shared_buffer["continuity_id"][0] = torch.tensor([-1, 0, 0, 0])
        else:
            engine.shared_buffer = engine.shared_buffer.exclude("continuity_id")
            engine.shared_buffer["valid"][0].zero_()
    assert engine.get_flow_status()["unlocked_remaining_chunks"] == 3
    assert engine.get_sampling_budget_snapshot()["unlocked_remaining_chunks"] == 3


@pytest.mark.parametrize("budget", [None, 3])
def test_flow_status_is_detached_and_does_not_build_full_audits(
    monkeypatch, budget: int | None
) -> None:
    engine = _engine(rows=2, budget=budget, hard=budget)
    with engine._progress.values.get_lock():
        engine._progress.values[:3] = [3, 2, 7]

    def unexpected_audit():
        pytest.fail("Compact flow status must not invoke full diagnostics")

    monkeypatch.setattr(engine, "get_stats", unexpected_audit)
    monkeypatch.setattr(engine, "get_sampling_budget_snapshot", unexpected_audit)
    assert engine.check_health() is None
    with torch.device("meta"):
        status = engine.get_flow_status()
    assert status == {
        "requested_generation": 3,
        "completed_generation": 2,
        "successful_episodes": 7,
        "unlocked_remaining_chunks": None if budget is None else 6,
        "total_delivered_chunks": None if budget is None else 0,
    }
    assert all(value is None or type(value) is int for value in status.values())
    status["requested_generation"] = -99
    status["unlocked_remaining_chunks"] = -99
    assert engine.get_flow_status()["requested_generation"] == 3
    assert engine.get_flow_status()["unlocked_remaining_chunks"] == (
        None if budget is None else 6
    )
    assert not engine._fill_signal.is_set()


@pytest.mark.parametrize(
    "state",
    [
        OnlineDataEngineState.CREATED,
        OnlineDataEngineState.STARTING,
        OnlineDataEngineState.STOPPED,
    ],
)
@pytest.mark.parametrize("method", ["check_health", "get_flow_status"])
def test_flow_apis_require_ready_without_removing_stopped_audits(
    state: OnlineDataEngineState, method: str
) -> None:
    engine = _engine()
    engine._set_state(state)
    with pytest.raises(RuntimeError, match=f"current state is {state.value}"):
        getattr(engine, method)()
    if state is OnlineDataEngineState.STOPPED:
        assert engine.get_sampling_budget_snapshot()["unlocked_remaining_chunks"] == 6


@pytest.mark.parametrize("method", ["check_health", "get_flow_status"])
@pytest.mark.parametrize("shared_error", [False, True])
def test_flow_apis_preserve_worker_and_local_failure_exceptions(
    method: str, shared_error: bool
) -> None:
    engine = _engine(rows=1, budget=1, hard=1)
    engine.sample_batch(1, 1)
    error = TimeoutError("producer refill failed")
    if shared_error:
        engine._record_worker_error(error)
    else:
        engine._worker_error = error
        engine._set_state(OnlineDataEngineState.FAILED)
    with pytest.raises(TimeoutError, match="producer refill failed") as captured:
        getattr(engine, method)()
    assert captured.value is error


@pytest.mark.parametrize("method", ["check_health", "get_flow_status"])
def test_flow_apis_detect_ready_worker_exit(monkeypatch, method: str) -> None:
    engine = _engine()
    process = MagicMock(exitcode=17)
    process.is_alive.return_value = False
    engine._sim_process = process
    monkeypatch.setattr(engine, "_is_owner_process", lambda: True)
    with pytest.raises(RuntimeError, match="exit code 17"):
        getattr(engine, method)()
    assert engine.state is OnlineDataEngineState.FAILED
    process.join.assert_called_once_with(timeout=0)


@pytest.mark.parametrize("method", ["check_health", "get_flow_status"])
def test_flow_apis_reject_failed_engine_without_error_envelope(method: str) -> None:
    engine = _engine()
    engine._set_state(OnlineDataEngineState.FAILED)
    with pytest.raises(
        data_module.OnlineDataWorkerError, match="without an error envelope"
    ):
        getattr(engine, method)()
