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

"""Public source and Dataset contracts without simulator initialization."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader

from embodichain.data_pipeline.datasets import (
    BatchSource,
    DataBudgetExhaustedError,
    DataUnavailableError,
    OfflineDataset,
    OnlineDataset,
    OnlineDataSource,
    SequenceSpecCfg,
    SequenceDataset,
    TensorDataSource,
)
from embodichain.data_pipeline.engine.data import (
    OnlineDataEngine,
    OnlineDataEngineCfg,
    OnlineDataEngineState,
)

_SCHEMA = {
    "observation": {"names": ["position", "speed"], "dtype": "float32"},
    "action": {"names": ["request"], "semantics": "policy_request", "step_dt": 1 / 30},
}


def _data() -> TensorDict:
    shape = (2, 5)
    return TensorDict(
        {
            "obs": torch.arange(20, dtype=torch.float32).reshape(2, 5, 2),
            "actions": -torch.arange(10, dtype=torch.float32).reshape(2, 5, 1),
            "valid": torch.tensor([[True] * 5, [True, False, True, True, False]]),
            "continuity_id": torch.tensor([[0, 0, 1, 1, -1], [0, -1, 2, 2, -1]]),
            "segment_id": torch.tensor([[0, 0, 1, 1, 1], [0, -1, 0, 1, -1]]),
            "segment_accepted": torch.tensor(
                [[True, True, True, False, True], [True] * 5]
            ),
            "task_index": torch.zeros(shape, dtype=torch.int64),
            "subtask_index": torch.ones(shape, dtype=torch.int64),
            "generations": torch.tensor([[1] * 5, [2] * 5]),
            "accepted_at_ns": torch.full(shape, 1000000000000000000),
            "published_at_monotonic_ns": torch.full(shape, 123456789),
            "frame_ids": torch.stack(
                (
                    torch.full(shape, 17),
                    torch.arange(2)[:, None].expand(shape),
                    torch.arange(5)[None].expand(shape),
                ),
                -1,
            ),
        },
        batch_size=shape,
        device="cpu",
    )


@pytest.fixture
def paired_sources(monkeypatch):
    data = _data()
    raw = data.exclude(
        "frame_ids", "generations", "accepted_at_ns", "published_at_monotonic_ns"
    ).clone()
    raw["ods_accepted_at_ns"] = data["accepted_at_ns"].clone()
    raw["ods_published_at_monotonic_ns"] = data["published_at_monotonic_ns"].clone()
    for index, key in enumerate(("ods_run_id", "ods_episode_id", "ods_frame_index")):
        raw[key] = data["frame_ids"][..., index].clone()
    raw["ods_generation"] = data["generations"].clone()
    monkeypatch.setattr(
        OnlineDataEngine, "_create_buffer", lambda self: raw.share_memory_()
    )
    engine = OnlineDataEngine(
        OnlineDataEngineCfg(
            gym_config={"num_envs": 1},
            state_dim=2,
            action_dim=1,
            action_semantics="policy_request",
            action_contracts=("gripper.parallel@1",),
            buffer_size=2,
            max_episode_steps=5,
            refill_policy="manual",
            include_provenance=True,
        )
    )
    engine._owner_pid = -1
    engine._set_state(OnlineDataEngineState.READY)
    engine._init_signal.set()
    with engine._lock_index.get_lock():
        engine._lock_index[:] = [-1, -1]
    try:
        yield OnlineDataSource(engine, schema=_SCHEMA), TensorDataSource(
            data, schema=_SCHEMA, source_id="recorded-17"
        )
    finally:
        engine._set_state(OnlineDataEngineState.CREATED)
        engine._owner_pid = os.getpid()
        engine.stop()


@pytest.mark.parametrize("mode", ["episode", "segment"])
@pytest.mark.parametrize("generations", [None, (1,), (2,)])
def test_live_and_resident_sources_share_windows_masks_and_rng(
    paired_sources, mode, generations
):
    live, resident = paired_sources
    assert isinstance(live, BatchSource) and isinstance(resident, BatchSource)
    sequence = SequenceSpecCfg(2, 4, mode, generations)
    first = torch.Generator().manual_seed(210)
    second = torch.Generator().manual_seed(210)
    global_state = torch.get_rng_state()
    online = live.sample_context(128, sequence, generator=first)
    offline = resident.sample_context(128, sequence, generator=second)
    assert torch.equal(first.get_state(), second.get_state())
    assert torch.equal(torch.get_rng_state(), global_state)
    assert set(online.keys()) == set(offline.keys())
    assert all(torch.equal(online[key], offline[key]) for key in online.keys())
    assert online["valid"][:, 1].all() and not online["loss_mask"][:, 0].any()
    for key in ("accepted_at_ns", "published_at_monotonic_ns"):
        assert (online[key][~online["valid"]] == -1).all()
    if mode == "segment":
        assert (online["segment_id"] == online["segment_id"][:, 1:2])[
            online["valid"]
        ].all()
        assert online["segment_accepted"][online["valid"]].all()
    if generations is not None:
        assert (online["generations"][online["valid"]] == generations[0]).all()
    online["obs"].fill_(-100)
    assert not (live.engine.shared_buffer["obs"] == -100).any()
    assert not (resident.data["obs"] == -100).any()


@pytest.mark.parametrize("with_continuity", [False, True])
@pytest.mark.parametrize("generations", [None, (1,)])
def test_segment_filter_preserves_source_validity_and_episode_anchors(
    with_continuity, generations
):
    data = _data()[:1].exclude("continuity_id")
    if with_continuity:
        data["continuity_id"] = torch.zeros(data.batch_size, dtype=torch.int64)
    source = TensorDataSource(data, schema=_SCHEMA, source_id="segment-filter")
    original = source.data.clone()
    history_length = 2
    segment_batch = source.sample_context(
        32,
        SequenceSpecCfg(history_length, 4, "segment", generations),
        generator=torch.Generator().manual_seed(210),
    )
    assert segment_batch["segment_accepted"][segment_batch["valid"]].all()
    assert all(torch.equal(source.data[key], original[key]) for key in original.keys())

    episode_batch = source.sample_context(
        64,
        SequenceSpecCfg(history_length, 4),
        generator=torch.Generator().manual_seed(210),
    )
    # Frame 3 is rejected only by the segment filter; episode mode retains it.
    anchors = episode_batch["frame_ids"][:, history_length - 1, 2]
    assert (anchors == 3).any()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"history_length": 0},
        {"history_length": True},
        {"prediction_horizon": 0},
        {"sampling_mode": "boundary"},
        {"generations": [1]},
        {"generations": (0,)},
    ],
)
def test_invalid_sequence_is_rejected(kwargs):
    with pytest.raises(ValueError):
        SequenceSpecCfg(**kwargs)


def test_missing_source_capabilities_are_not_silently_ignored():
    source = TensorDataSource(
        _data().exclude("generations", "segment_id"), schema=_SCHEMA, source_id="test"
    )
    with pytest.raises(ValueError, match="generation"):
        source.sample_context(1, SequenceSpecCfg(generations=(1,)))
    with pytest.raises(ValueError, match="Segment"):
        source.sample_context(1, SequenceSpecCfg(sampling_mode="segment"))


def test_constructor_owns_input_and_nested_observation_storage():
    data = _data()
    source = TensorDataSource(data, schema=_SCHEMA, source_id="test")
    data["obs"].fill_(-100)
    assert not (source.data["obs"] == -100).any()
    nested = _data()
    nested["obs"] = TensorDict({"joint": nested["obs"]}, batch_size=nested.batch_size)
    source = TensorDataSource(
        nested, schema={"action": _SCHEMA["action"]}, source_id="test"
    )
    source.configure_sampling_budget(3, 3)
    with torch.device("meta"):
        batch = source.sample_context(
            2, SequenceSpecCfg(), generator=torch.Generator(device="cpu").manual_seed(2)
        )
    assert batch["obs", "joint"].device.type == "cpu"


def test_source_local_language_ids_require_the_correct_dictionary():
    source = TensorDataSource(
        _data(),
        schema=_SCHEMA,
        source_id="test",
        metadata={"language": {"task": {0: "pick"}, "subtask": {"1": "release"}}},
    )
    item = source.data[0, :2]
    assert source.resolve_language(item) == {
        "task": ["pick"] * 2,
        "subtask": ["release"] * 2,
    }
    item["task_index"][0] = -1
    with pytest.raises(ValueError, match="padded"):
        source.resolve_language(item)


def test_ineligible_continuity_does_not_advertise_spendable_credits():
    data = _data()
    data["continuity_id"][0].fill_(-1)
    source = TensorDataSource(data, schema=_SCHEMA, source_id="test")
    source.configure_sampling_budget(3, 3)
    snapshot = source.get_sampling_budget_snapshot()
    assert snapshot["remaining_chunks"] == 6
    assert snapshot["unlocked_remaining_chunks"] == 3


def test_unmarked_source_frame_jumps_cannot_become_training_transitions():
    data = _data().exclude("continuity_id")
    data["frame_ids"][0, 1:, 2] += 1
    with pytest.raises(ValueError, match="frame gaps"):
        TensorDataSource(data, schema=_SCHEMA, source_id="test")


@pytest.mark.parametrize("batch_size", [None, 3])
def test_dataset_wrappers_share_context_iteration(paired_sources, batch_size):
    live, resident = paired_sources
    sequence = SequenceSpecCfg(2, 3)
    online = OnlineDataset(
        live.engine,
        sequence=sequence,
        schema=_SCHEMA,
        batch_size=batch_size,
        generator=torch.Generator().manual_seed(88),
    )
    datasets = [
        SequenceDataset(
            source,
            sequence,
            batch_size=batch_size,
            max_batches=2,
            generator=torch.Generator().manual_seed(88),
        )
        for source in (live, resident)
    ]
    # Caller mutations must not change a Dataset's established window.
    sequence.prediction_horizon = 1
    reference = iter(online)
    left, right = map(iter, datasets)
    for _ in range(2):
        expected, a, b = next(reference), next(left), next(right)
        assert expected.batch_size == a.batch_size == b.batch_size
        assert a.batch_size[-1] == 4
        assert all(
            torch.equal(expected[key], a[key]) and torch.equal(a[key], b[key])
            for key in a.keys()
        )
    for iterator in (left, right):
        with pytest.raises(StopIteration):
            next(iterator)
    dataset = OfflineDataset(resident, SequenceSpecCfg(2, 3), max_batches=4)
    loader = DataLoader(
        dataset, batch_size=2, collate_fn=OfflineDataset.collate_fn, num_workers=0
    )
    assert next(iter(loader)).batch_size == (2, 4)


@pytest.mark.parametrize(
    "options", [{"generator": torch.Generator()}, {"max_batches": 2}]
)
def test_explicit_generator_cannot_be_copied_into_workers(monkeypatch, options):
    from embodichain.data_pipeline.datasets import sequence_data

    monkeypatch.setattr(sequence_data, "get_worker_info", lambda: object())
    source = TensorDataSource(_data(), schema=_SCHEMA, source_id="test")
    with pytest.raises(RuntimeError, match="owner-process"):
        next(iter(OfflineDataset(source, SequenceSpecCfg(), **options)))


def test_offline_public_imports_do_not_import_simulation_or_sdk():
    root = Path(__file__).resolve().parents[2]
    program = "from embodichain.data_pipeline.datasets import LeRobotDataSource, TensorDataSource, SequenceDataset, OfflineDataset, DataUnavailableError, validate_data_schema, data_schema_fingerprint; import sys; assert not any(k.startswith(('embodichain.lab', 'gymnasium', 'dexsim', 'lerobot')) for k in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_offline_dataset_is_only_a_compatibility_name():
    from embodichain.data_pipeline.datasets.offline_data import OfflineDataset as legacy

    assert OfflineDataset is SequenceDataset is legacy


def test_new_dataset_iterator_does_not_replenish_source_budget():
    source = TensorDataSource(_data(), schema=_SCHEMA, source_id="test")
    source.configure_sampling_budget(2, 2)
    dataset = SequenceDataset(source, SequenceSpecCfg(), batch_size=2, max_batches=1)
    assert len(list(dataset)) == len(list(dataset)) == 1
    with pytest.raises(DataBudgetExhaustedError):
        next(iter(dataset))
    assert source.get_sampling_budget_snapshot()["total_delivered_chunks"] == 4


@pytest.mark.parametrize(
    "error_type", [DataUnavailableError, DataBudgetExhaustedError, RuntimeError]
)
def test_dataset_propagates_source_errors_without_retry(monkeypatch, error_type):
    source = TensorDataSource(_data(), schema=_SCHEMA, source_id="test")
    calls = []
    error = error_type("source failure")

    def fail(*args, **kwargs):
        calls.append(1)
        raise error

    monkeypatch.setattr(source, "sample_context", fail)
    with pytest.raises(error_type) as captured:
        next(iter(SequenceDataset(source, SequenceSpecCfg())))
    assert captured.value is error and len(calls) == 1


def test_empty_window_error_is_distinct_from_failure_and_quota(paired_sources):
    live, resident = paired_sources
    live.engine.shared_buffer["valid"].zero_()
    resident.data["valid"].zero_()
    for source in (live, resident):
        with pytest.raises(DataUnavailableError) as captured:
            source.sample_context(2, SequenceSpecCfg())
        assert isinstance(captured.value, RuntimeError)
        assert not isinstance(captured.value, DataBudgetExhaustedError)


def test_failed_transform_keeps_delivery_charged():
    source = TensorDataSource(_data(), schema=_SCHEMA, source_id="test")
    source.configure_sampling_budget(2, 2)

    def fail(batch):
        raise ValueError("feature transform failed")

    dataset = SequenceDataset(source, SequenceSpecCfg(), batch_size=2, transform=fail)
    with pytest.raises(ValueError, match="feature transform"):
        next(iter(dataset))
    assert source.get_sampling_budget_snapshot()["total_delivered_chunks"] == 2


def test_common_transform_restores_canonical_padding():
    source = TensorDataSource(_data(), schema=_SCHEMA, source_id="test")

    def convert(batch):
        batch["obs"] += 1
        return batch

    dataset = SequenceDataset(
        source,
        SequenceSpecCfg(4, 8),
        batch_size=16,
        transform=convert,
        generator=torch.Generator().manual_seed(4),
    )
    batch = next(iter(dataset))
    assert (~batch["valid"]).any()
    assert (batch["obs"][~batch["valid"]] == 0).all()
    assert not batch["loss_mask"][:, :3].any()
