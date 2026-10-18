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

"""Independent conservation and feasibility checks across ODS operating regimes."""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict

from embodichain.data_pipeline.configuration import (
    ODSConfiguration,
    ODSReuseProfileCfg,
    ODSRuntimeCfg,
    configure_ods,
)
from embodichain.data_pipeline.engine.data import (
    OnlineDataBudgetExhaustedError,
    OnlineDataEngine,
    OnlineDataEngineCfg,
    OnlineDataEngineState,
)

# Odd quotas and batches deliberately exercise indivisible final batches.
_PROFILE_MIN = 7
_PROFILE_MAX = 31
_PROFILE_HARD = 47
_EPISODE_BYTES = 103
_CASES = list(
    product(
        (0.125, 0.3),  # Binary-exact and inexact successful publication rates.
        (3.5, 17.25, 31.0, 124.0),  # Below, within, at, above profile bounds.
        (1, 3),
        (1, 17, 64),
        (0.1, 2.5, 13.7),
        (None, 2, 4, 12),  # Storage measured in writer batches, plus spare bytes.
    )
)


def _profile() -> ODSReuseProfileCfg:
    return ODSReuseProfileCfg(
        profile_id="regime-contract-v1",
        scope="synthetic conservation tests; not a learning qualification",
        qualified=False,
        min_chunks_per_episode=_PROFILE_MIN,
        max_chunks_per_episode=_PROFILE_MAX,
        hard_max_chunks_per_episode=_PROFILE_HARD,
        latency_safety_factor=1.25,
    )


def _assert_conservation(
    plan: ODSConfiguration,
    g: float,
    t: float,
    profile: ODSReuseProfileCfg,
    runtime: ODSRuntimeCfg,
) -> None:
    """Check physical credit/storage inequalities using exact binary input values."""
    quota = plan.episode_sample_budget
    target = Fraction(plan.training_chunks_per_second_target)
    active = Fraction(plan.active_chunks_per_second_cap)
    lead = Fraction(runtime.first_publication_p95_seconds) * Fraction(
        profile.latency_safety_factor
    )
    unlocked = (plan.buffer_size - runtime.producer_envs) * quota
    assert profile.min_chunks_per_episode <= quota <= profile.max_chunks_per_episode
    assert (
        quota <= plan.episode_sample_hard_limit == profile.hard_max_chunks_per_episode
    )
    assert plan.buffer_size % runtime.producer_envs == 0
    assert plan.buffer_size >= max(
        2 * runtime.producer_envs, runtime.min_cache_episodes
    )
    assert unlocked >= runtime.batch_size
    assert 0 < target <= active <= Fraction(t)
    assert target <= Fraction(g) * quota
    assert active * lead <= unlocked
    assert runtime.batch_size <= plan.refill_remaining_chunks <= unlocked
    assert Fraction(plan.publication_episodes_per_second_target) <= Fraction(g)
    assert (
        Fraction(plan.training_updates_per_second_target)
        <= Fraction(t) / runtime.batch_size
    )
    assert 0 < plan.training_duty_fraction <= 1
    assert 0 < plan.producer_duty_fraction <= 1
    assert (
        plan.training_duty_fraction + plan.expected_training_throttle_fraction
        == pytest.approx(1)
    )
    assert (
        plan.producer_duty_fraction + plan.expected_producer_throttle_fraction
        == pytest.approx(1)
    )
    assert plan.initial_chunk_budget == plan.buffer_size * quota
    assert plan.initial_chunk_budget == (
        plan.initial_max_updates * runtime.batch_size + plan.initial_remainder_chunks
    )
    assert 0 <= plan.initial_remainder_chunks < runtime.batch_size
    if runtime.cache_memory_bytes is not None:
        assert plan.cache_bytes == plan.buffer_size * runtime.bytes_per_episode
        assert plan.cache_bytes <= runtime.cache_memory_bytes
    assert not plan.profile_qualified


@pytest.mark.parametrize("g,ratio,producers,batch,latency,memory_batches", _CASES)
def test_configuration_regime_conservation(
    g: float,
    ratio: float,
    producers: int,
    batch: int,
    latency: float,
    memory_batches: int | None,
) -> None:
    """Check a deterministic matrix without reproducing the cache-sizing heuristic."""
    t = g * ratio
    profile = _profile()
    # Independent finite search over the profile: maximize affordable reuse,
    # retaining the minimum if the learner cannot afford any allowed integer.
    affordable = [
        value
        for value in range(_PROFILE_MIN, _PROFILE_MAX + 1)
        if Fraction(g) * value <= Fraction(t)
    ]
    quota = max(affordable, default=_PROFILE_MIN)
    memory = (
        None
        if memory_batches is None
        else memory_batches * producers * _EPISODE_BYTES + _EPISODE_BYTES - 1
    )
    runtime = ODSRuntimeCfg(
        batch_size=batch,
        producer_envs=producers,
        first_publication_p95_seconds=latency,
        min_cache_episodes=5,
        cache_memory_bytes=memory,
        bytes_per_episode=None if memory is None else _EPISODE_BYTES,
        measurement_scope="synthetic concurrent capacities",
    )
    # Enumerate legal storage layouts: no dependency on configure_ods internals.
    feasible = (
        None
        if memory is None
        else [
            count
            for count in range(producers, memory // _EPISODE_BYTES + 1, producers)
            if count >= 2 * producers
            and count >= runtime.min_cache_episodes
            and (count - producers) * quota >= batch
        ]
    )
    if feasible == []:
        with pytest.raises(ValueError, match="Infeasible cache memory"):
            configure_ods(g, t, profile=profile, runtime=runtime)
        return
    plan = configure_ods(g, t, profile=profile, runtime=runtime)
    assert plan.episode_sample_budget == quota
    _assert_conservation(plan, g, t, profile, runtime)


def test_initial_burst_does_not_define_sustainable_training_capacity() -> None:
    """Cached credits can fund an initial fast burst without increasing G*K."""
    profile = _profile()
    runtime = ODSRuntimeCfg(
        batch_size=17,
        producer_envs=1,
        first_publication_p95_seconds=2,
        measurement_scope="synthetic supply-limited",
    )
    plan = configure_ods(0.25, 100, profile=profile, runtime=runtime)
    assert plan.active_chunks_per_second_cap == 100
    assert plan.training_chunks_per_second_target == 7.75
    initial_delivered = plan.initial_max_updates * runtime.batch_size
    burst_seconds = Fraction(initial_delivered) / Fraction(
        plan.active_chunks_per_second_cap
    )
    assert Fraction(initial_delivered) / burst_seconds > Fraction(
        plan.training_chunks_per_second_target
    )
    # Over any horizon only initial credits plus published credits are available.
    for horizon_seconds in (10, 100, 10000):
        maximum_chunks = (
            Fraction(plan.initial_chunk_budget)
            + Fraction(0.25) * horizon_seconds * plan.episode_sample_budget
        )
        assert maximum_chunks / horizon_seconds - Fraction(
            plan.training_chunks_per_second_target
        ) == Fraction(plan.initial_chunk_budget, horizon_seconds)


def _consumer_engine(plan: ODSConfiguration) -> OnlineDataEngine:
    """Use the public sampler with CPU payloads and real shared quota bookkeeping."""
    cfg = OnlineDataEngineCfg(
        buffer_size=plan.buffer_size,
        state_dim=2,
        max_episode_steps=4,
        gym_config={"num_envs": plan.producer_envs},
        refill_policy="manual",
        episode_sample_budget=plan.episode_sample_budget,
        episode_sample_hard_limit=plan.episode_sample_hard_limit,
    )

    def create_buffer(engine: OnlineDataEngine) -> TensorDict:
        rows, steps = plan.buffer_size, cfg.max_episode_steps
        row_ids = torch.arange(rows)[:, None].expand(rows, steps).clone()
        return TensorDict(
            {
                "obs": torch.zeros(rows, steps, 2),
                "actions": torch.zeros(rows, steps, 2),
                "valid": torch.ones(rows, steps, dtype=torch.bool),
                "continuity_id": torch.zeros(rows, steps, dtype=torch.int64),
                "segment_id": torch.zeros(rows, steps, dtype=torch.int64),
                "segment_accepted": torch.ones(rows, steps, dtype=torch.bool),
                "ods_run_id": torch.full(
                    (rows, steps), engine.run_id, dtype=torch.int64
                ),
                "ods_episode_id": row_ids,
            },
            batch_size=[rows, steps],
            device="cpu",
        ).share_memory_()

    with patch.object(OnlineDataEngine, "_create_buffer", create_buffer):
        engine = OnlineDataEngine(cfg)
    engine._owner_pid = -1  # No producer exists; bypass producer-liveness checks.
    engine._set_state(OnlineDataEngineState.READY)
    engine._init_signal.set()
    with engine._lock_index.get_lock():
        engine._lock_index[:] = [-1, -1]
        engine._sampling_budget.publish(0, plan.buffer_size, 0, 1)
    return engine


@pytest.mark.parametrize(
    "g,t,batch,locked", [(1, 3, 17, False), (1, 100, 17, False), (1, 17, 64, True)]
)
def test_derived_budget_depletes_into_exact_full_batches(
    g: float,
    t: float,
    batch: int,
    locked: bool,
) -> None:
    """Quota safety and divmod leftovers hold under real sampling, including locks."""
    plan = configure_ods(
        g,
        t,
        profile=_profile(),
        runtime=ODSRuntimeCfg(
            batch_size=batch,
            producer_envs=2,
            first_publication_p95_seconds=0.1,
            measurement_scope="synthetic consumer depletion",
        ),
    )
    engine = _consumer_engine(plan)
    if locked:
        with engine._lock_index.get_lock():
            engine._lock_index[:] = [0, plan.producer_envs]
    available_rows = plan.buffer_size - (plan.producer_envs if locked else 0)
    expected_updates, expected_leftover = divmod(
        available_rows * plan.episode_sample_budget, batch
    )
    sources: dict[int, int] = {}
    try:
        for _ in range(expected_updates):
            result = engine.sample_batch(batch, 1)
            for episode in result["ods_episode_id"].flatten().tolist():
                sources[episode] = sources.get(episode, 0) + 1
        before = engine.get_sampling_budget_snapshot()
        with pytest.raises(OnlineDataBudgetExhaustedError):
            engine.sample_batch(batch, 1)
        after = engine.get_sampling_budget_snapshot()
        assert after == before
        assert after["total_delivered_chunks"] == expected_updates * batch
        assert after["unlocked_remaining_chunks"] == expected_leftover
        assert max(sources.values()) <= plan.episode_sample_budget
        assert {
            row["episode_id"]: row["delivered"]
            for row in after["rows"]
            if row["delivered"]
        } == sources
        if locked:
            assert set(sources).isdisjoint(range(plan.producer_envs))
    finally:
        engine._set_state(OnlineDataEngineState.STOPPED)
