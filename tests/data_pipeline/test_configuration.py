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

"""CPU contracts for ODS rate, memory and selection-budget configuration."""

from __future__ import annotations

from dataclasses import asdict
from fractions import Fraction

import pytest

from embodichain.data_pipeline.configuration import (
    ODSRateMeasurementCfg,
    ODSReuseProfileCfg,
    ODSRuntimeCfg,
    configure_ods,
    measure_ods_throughput,
)


def profile(**changes):
    values = dict(
        profile_id="test-v1",
        scope="synthetic model/horizon/length profile",
        qualified=False,
        min_chunks_per_episode=200,
        max_chunks_per_episode=600,
        hard_max_chunks_per_episode=1000,
        latency_safety_factor=1.25,
    )
    values.update(changes)
    return ODSReuseProfileCfg(**values)


def runtime(**changes):
    values = dict(
        batch_size=64,
        producer_envs=2,
        first_publication_p95_seconds=2.0,
        measurement_scope="joint synthetic capacity",
    )
    values.update(changes)
    return ODSRuntimeCfg(**values)


def test_supply_limited_target_preserves_recommended_cap_and_metadata():
    result = configure_ods(1, 1000, profile=profile(), runtime=runtime())
    assert result.episode_sample_budget == 600
    assert result.episode_sample_hard_limit == 1000
    assert result.training_chunks_per_second_target == 600
    assert result.publication_episodes_per_second_target == 1
    assert result.training_duty_fraction == pytest.approx(0.6)
    assert result.buffer_size == 8  # next multiple of2 above2+ceil(2500/600)
    assert result.refill_remaining_chunks == 2500
    assert (result.batch_size, result.producer_envs) == (64, 2)
    assert result.profile_qualified is False
    assert any("Provisional" in reason for reason in result.reasons)
    assert asdict(result)["episode_sample_budget"] == 600


def test_fast_producer_is_paced_without_lowering_profile_minimum():
    result = configure_ods(10, 1000, profile=profile(qualified=True), runtime=runtime())
    assert result.episode_sample_budget == 200
    assert result.training_chunks_per_second_target == 1000
    assert result.publication_episodes_per_second_target == 5
    assert result.producer_duty_fraction == 0.5
    assert result.profile_qualified is True


def test_balance_budget_floors_before_clamping():
    result = configure_ods(3, 1000, profile=profile(), runtime=runtime())
    assert result.episode_sample_budget == 333
    assert result.training_chunks_per_second_target == 999


def test_memory_constrains_active_not_merely_average_training_rate():
    result = configure_ods(
        1,
        1000,
        profile=profile(),
        runtime=runtime(cache_memory_bytes=400, bytes_per_episode=100),
    )
    assert result.buffer_size == 4
    # Two unlocked episodes have1200 selections, covering2.5s at480 chunks/s.
    assert result.active_chunks_per_second_cap == 480
    assert result.training_chunks_per_second_target == 480
    assert result.publication_episodes_per_second_target == pytest.approx(0.8)
    assert result.training_duty_fraction == pytest.approx(0.48)
    assert result.refill_remaining_chunks == 1200
    assert result.cache_bytes == 400
    assert result.initial_max_updates == 37
    assert result.initial_remainder_chunks == 32


def test_memory_capacity_rounds_down_to_whole_writer_batches():
    result = configure_ods(
        1,
        1000,
        profile=profile(),
        runtime=runtime(cache_memory_bytes=750, bytes_per_episode=100),
    )
    assert result.buffer_size == 6
    assert result.cache_bytes == 600
    assert result.active_chunks_per_second_cap == 960


def test_short_window_memory_bound_is_conservatively_represented():
    result = configure_ods(
        1,
        1000,
        profile=profile(latency_safety_factor=1),
        runtime=runtime(
            first_publication_p95_seconds=7,
            cache_memory_bytes=400,
            bytes_per_episode=100,
        ),
    )
    exact_limit = Fraction(1200, 7)
    assert Fraction(result.active_chunks_per_second_cap) <= exact_limit
    assert result.refill_remaining_chunks <= 1200


def test_cache_preserves_minimum_diversity_and_one_full_unlocked_batch():
    result = configure_ods(
        1,
        1,
        profile=profile(min_chunks_per_episode=1, max_chunks_per_episode=1),
        runtime=runtime(batch_size=64, min_cache_episodes=5),
    )
    assert result.buffer_size == 66
    assert result.refill_remaining_chunks == 64
    assert result.initial_max_updates == 1


@pytest.mark.parametrize(
    "changes",
    [
        dict(cache_memory_bytes=300, bytes_per_episode=100),
        dict(cache_memory_bytes=500, bytes_per_episode=100, min_cache_episodes=6),
    ],
)
def test_memory_infeasibility_is_explicit(changes):
    with pytest.raises(ValueError, match="Infeasible"):
        configure_ods(1, 1000, profile=profile(), runtime=runtime(**changes))


def test_memory_cannot_fit_one_unlocked_batch():
    with pytest.raises(ValueError, match="Infeasible"):
        configure_ods(
            1,
            1,
            profile=profile(min_chunks_per_episode=1, max_chunks_per_episode=1),
            runtime=runtime(cache_memory_bytes=400, bytes_per_episode=100),
        )


@pytest.mark.parametrize("rate", [0, -1, True, float("nan"), float("inf"), "1"])
def test_invalid_capacity_requires_diagnosis(rate):
    with pytest.raises(ValueError):
        configure_ods(rate, 1000, profile=profile(), runtime=runtime())
    with pytest.raises(ValueError):
        configure_ods(1, rate, profile=profile(), runtime=runtime())


@pytest.mark.parametrize(
    "changes",
    [
        dict(min_chunks_per_episode=True),
        dict(max_chunks_per_episode=1.5),
        dict(min_chunks_per_episode=601),
        dict(hard_max_chunks_per_episode=599),
        dict(qualified=1),
        dict(profile_id=" "),
        dict(scope=""),
        dict(latency_safety_factor=0.9),
        dict(latency_safety_factor=float("inf")),
    ],
)
def test_invalid_profile_is_rejected(changes):
    with pytest.raises(ValueError):
        configure_ods(1, 1000, profile=profile(**changes), runtime=runtime())


@pytest.mark.parametrize(
    "changes",
    [
        dict(batch_size=True),
        dict(producer_envs=0),
        dict(first_publication_p95_seconds=0),
        dict(first_publication_p95_seconds=float("nan")),
        dict(cache_memory_bytes=100),
        dict(bytes_per_episode=100),
        dict(cache_memory_bytes=True, bytes_per_episode=100),
        dict(min_cache_episodes=-1),
        dict(measurement_scope=""),
    ],
)
def test_invalid_runtime_is_rejected(changes):
    with pytest.raises(ValueError):
        configure_ods(1, 1000, profile=profile(), runtime=runtime(**changes))


def test_exact_initial_integer_budget_above_float_precision():
    budget = (1 << 53) + 1
    result = configure_ods(
        1,
        budget,
        profile=profile(
            min_chunks_per_episode=budget,
            max_chunks_per_episode=budget,
            hard_max_chunks_per_episode=budget,
            latency_safety_factor=1,
        ),
        runtime=runtime(batch_size=3, producer_envs=1, first_publication_p95_seconds=1),
    )
    assert result.episode_sample_budget == budget
    assert result.initial_chunk_budget == 2 * budget
    assert (result.initial_max_updates, result.initial_remainder_chunks) == divmod(
        2 * budget, 3
    )


@pytest.mark.parametrize("rates", [(1e-300, 1e300), (1e300, 1e-300)])
def test_unrepresentable_derived_rates_are_rejected(rates):
    with pytest.raises(ValueError):
        configure_ods(*rates, profile=profile(), runtime=runtime())


def test_derived_integer_overflow_is_rejected():
    huge = (1 << 63) - 1
    with pytest.raises(ValueError):
        configure_ods(
            1,
            huge,
            profile=profile(
                min_chunks_per_episode=huge,
                max_chunks_per_episode=huge,
                hard_max_chunks_per_episode=huge,
            ),
            runtime=runtime(),
        )


def measurement(unit, **changes):
    values = dict(
        unit=unit,
        count=20,
        start_monotonic_seconds=100.0,
        end_monotonic_seconds=110.0,
        active_seconds=5.0,
    )
    values.update(changes)
    return ODSRateMeasurementCfg(**values)


def test_measurement_preserves_active_and_wall_units():
    result = measure_ods_throughput(
        measurement("accepted_episodes"),
        measurement("completed_chunks", count=1000, active_seconds=2),
    )
    assert result.accepted_episodes_per_active_second == 4
    assert result.accepted_episodes_per_wall_second == 2
    assert result.training_chunks_per_active_second == 500
    assert result.training_chunks_per_wall_second == 100
    assert result.window_seconds == 10


def test_zero_count_is_reported_without_inventing_capacity():
    result = measure_ods_throughput(
        measurement("accepted_episodes", count=0),
        measurement("completed_chunks", count=0),
    )
    assert (
        result.accepted_episodes_per_active_second
        == result.training_chunks_per_wall_second
        == 0
    )


@pytest.mark.parametrize(
    "changes",
    [
        dict(unit="frames"),
        dict(count=True),
        dict(count=-1),
        dict(active_seconds=0),
        dict(active_seconds=11),
        dict(start_monotonic_seconds=-1),
        dict(end_monotonic_seconds=100),
        dict(end_monotonic_seconds=111),
        dict(active_seconds=float("inf")),
    ],
)
def test_invalid_or_unaligned_measurements_are_rejected(changes):
    production = measurement("accepted_episodes")
    training = (
        measurement("completed_chunks", **changes)
        if "unit" not in changes
        else measurement(changes["unit"])
    )
    with pytest.raises(ValueError):
        measure_ods_throughput(production, training)


def test_missing_qualification_is_not_inferred_from_rates():
    incomplete = ODSReuseProfileCfg(
        profile_id="incomplete",
        scope="synthetic",
        min_chunks_per_episode=1,
        max_chunks_per_episode=2,
        hard_max_chunks_per_episode=3,
    )
    with pytest.raises(TypeError, match="qualified"):
        configure_ods(1, 1000, profile=incomplete, runtime=runtime())


def test_configuration_does_not_mutate_profile_or_runtime():
    policy = profile()
    settings = runtime(cache_memory_bytes=400, bytes_per_episode=100)
    before = (asdict(policy), asdict(settings))
    configure_ods(1, 1000, profile=policy, runtime=settings)
    assert (asdict(policy), asdict(settings)) == before


def test_mutated_metadata_is_revalidated_at_configuration_boundary():
    settings = runtime()
    settings.batch_size = False
    with pytest.raises(ValueError, match="batch_size"):
        configure_ods(1, 1000, profile=profile(), runtime=settings)
