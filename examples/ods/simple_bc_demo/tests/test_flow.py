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

"""CPU checks for aligned throughput reports and excluded publication cohorts."""

from __future__ import annotations

import pytest

from examples.ods.simple_bc_demo.flow import summarize_flow


def episode(generation, offset, *, anchors=10, identity=None):
    result = {
        "generation": generation,
        "published_at_monotonic_ns": int((1000 + offset) * 1_000_000_000),
        "eligible_anchors": anchors,
    }
    if identity is not None:
        result.update(run_id=9, episode_id=identity)
    return result


def request(generation, seconds):
    return {"kind": "refill_requested", "generation": generation, "seconds": seconds}


def inputs(**changes):
    values = dict(
        monotonic_origin_seconds=1000,
        intervals={
            "training_service": [(10, 14), (12, 18), (20, 26)],
            "data_wait": [(13, 15), (22, 23)],
            "usage_throttle": [(18, 20)],
            "quota_wait": [(26, 28)],
        },
        refill_requests=[request(2, 11), request(3, 15), request(4, 23)],
        episode_records=[
            episode(1, 1),
            episode(1, 2),
            episode(2, 12),
            episode(2, 16, anchors=20),
            episode(3, 17, anchors=20),
            episode(3, 21, anchors=20),
            episode(4, 24),
            episode(5, 24),
            episode(5, 25),
        ],
        batch_size=4,
        updates=22,
        loop_start=10,
        loop_end=30,
        buffer_size=2,
    )
    values.update(changes)
    return values


def test_active_wall_and_classified_waits_have_distinct_denominators():
    result = summarize_flow(**inputs())
    learner = result["learner"]
    assert learner["completed_chunks"] == 88
    assert learner["service_seconds"] == 14
    assert learner["service_data_wait_seconds"] == 3
    assert learner["active_seconds"] == 11
    assert learner["active_chunks_per_second"] == 8
    assert learner["wall_chunks_per_second"] == 4.4
    assert learner["unattributed_wall_seconds"] == 2
    assert result["waits"] == {
        "usage_throttle_seconds": 2,
        "quota_wait_seconds": 2,
        "data_wait_seconds": 3,
        "union_seconds": 7,
    }


def test_productive_rate_uses_union_of_complete_refill_durations():
    producer = summarize_flow(**inputs())["producer"]
    # [11,16] and[15,21] overlap: count4 episodes over10s, not11s.
    assert producer["accepted_episodes"] == 4
    assert producer["eligible_anchors"] == 70
    assert producer["productive_seconds"] == 10
    assert producer["accepted_episodes_per_productive_second"] == 0.4
    assert producer["eligible_anchors_per_productive_second"] == 7
    assert [row["generation"] for row in producer["included_generations"]] == [2, 3]


def test_initial_incomplete_and_unmatched_generations_are_explicit():
    producer = summarize_flow(**inputs())["producer"]
    excluded = {
        row["generation"]: row["reasons"] for row in producer["excluded_generations"]
    }
    assert excluded == {
        1: ["initial_fill"],
        4: ["incomplete_generation"],
        5: ["unmatched_request"],
    }
    assert producer["first_publication_latencies"] == [
        {"generation": 2, "seconds": 1},
        {"generation": 3, "seconds": 2},
        {"generation": 4, "seconds": 1},
    ]
    assert producer["first_publication_latency_observations"] == 3


def test_absolute_clock_translation_preserves_rates():
    original = inputs()
    translated = inputs(monotonic_origin_seconds=2000)
    translated["episode_records"] = [
        {
            **row,
            "published_at_monotonic_ns": row["published_at_monotonic_ns"]
            + 1_000_000_000_000,
        }
        for row in original["episode_records"]
    ]
    assert (
        summarize_flow(**original)["producer"]
        == summarize_flow(**translated)["producer"]
    )


def test_completed_refill_outside_loop_is_not_labeled_training_era_supply():
    result = summarize_flow(
        **inputs(
            refill_requests=[request(2, 25)],
            episode_records=[episode(2, 26), episode(2, 31)],
        )
    )["producer"]
    assert result["accepted_episodes"] == 0
    assert result["accepted_episodes_per_productive_second"] is None
    assert result["excluded_generations"][0]["reasons"] == ["outside_training_window"]


def test_requested_generation_with_no_publications_is_reported():
    result = summarize_flow(
        **inputs(refill_requests=[request(2, 20)], episode_records=[])
    )["producer"]
    assert result["excluded_generations"][0]["reasons"] == ["incomplete_generation"]
    assert result["excluded_generations"][0]["first_publication_seconds"] is None


def test_no_completed_work_has_no_invented_active_capacity():
    result = summarize_flow(
        **inputs(
            intervals={},
            updates=0,
            loop_start=10,
            loop_end=10,
            refill_requests=[],
            episode_records=[],
        )
    )
    assert result["learner"]["active_chunks_per_second"] is None
    assert result["learner"]["wall_updates_per_second"] is None
    assert result["producer"]["productive_seconds"] == 0


def test_waits_outside_window_do_not_inflate_loop_wait():
    result = summarize_flow(
        **inputs(
            intervals={"training_service": [(10, 20)], "data_wait": [(0, 12), (28, 40)]}
        )
    )
    assert result["learner"]["active_seconds"] == 8
    assert result["waits"]["data_wait_seconds"] == 4


@pytest.mark.parametrize(
    "changes",
    [
        dict(updates=True),
        dict(updates=-1),
        dict(batch_size=0),
        dict(buffer_size=True),
        dict(loop_start=-1),
        dict(loop_end=9),
        dict(monotonic_origin_seconds=float("nan")),
        dict(loop_end=float("inf")),
        dict(intervals=None),
        dict(refill_requests=None),
        dict(episode_records=None),
    ],
)
def test_invalid_scalar_and_container_inputs_are_rejected(changes):
    with pytest.raises(ValueError):
        summarize_flow(**inputs(**changes))


@pytest.mark.parametrize(
    "services",
    [[], [(9, 12)], [(29, 31)], [(15, 12)], [(10, float("nan"))], [(10, 11, 12)]],
)
def test_invalid_or_missing_completed_service_is_rejected(services):
    with pytest.raises(ValueError):
        summarize_flow(**inputs(intervals={"training_service": services}))


def test_wait_only_service_cannot_support_completed_updates():
    with pytest.raises(ValueError, match="positive completed-service"):
        summarize_flow(
            **inputs(
                intervals={"training_service": [(10, 20)], "data_wait": [(10, 20)]}
            )
        )


@pytest.mark.parametrize(
    "events",
    [
        [request(2, 11), request(2, 12)],
        [request(1, 11)],
        [request(2, -1)],
        [{"kind": "other", "generation": 2, "seconds": 11}],
    ],
)
def test_invalid_or_ambiguous_request_events_are_rejected(events):
    with pytest.raises(ValueError):
        summarize_flow(**inputs(refill_requests=events))


@pytest.mark.parametrize(
    "episodes",
    [
        [episode(2, -1)],
        [episode(2, 10)],
        [episode(2, 12, anchors=-1)],
        [episode(2, 12), episode(2, 13), episode(2, 14)],
    ],
)
def test_clock_or_episode_accounting_inconsistencies_are_rejected(episodes):
    with pytest.raises(ValueError):
        summarize_flow(
            **inputs(episode_records=episodes, refill_requests=[request(2, 11)])
        )


def test_duplicate_immutable_identity_is_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_flow(
            **inputs(
                episode_records=[episode(2, 12, identity=1), episode(2, 13, identity=1)]
            )
        )


def test_zero_duration_complete_refill_is_rejected():
    with pytest.raises(ValueError, match="positive request-to-publication"):
        summarize_flow(
            **inputs(
                refill_requests=[request(2, 11)],
                episode_records=[episode(2, 11), episode(2, 11)],
            )
        )


def test_completed_chunk_counter_has_explicit_integer_bound():
    with pytest.raises(ValueError, match="completed chunks"):
        summarize_flow(**inputs(updates=(1 << 63) - 1, batch_size=2))
