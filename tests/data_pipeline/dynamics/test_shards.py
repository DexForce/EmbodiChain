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

"""Persistence, split and bounded-window contracts."""

from __future__ import annotations

import pickle
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from embodichain.data_pipeline.dynamics.shards import (
    DynamicsShardWriter,
    DynamicsWindowDataset,
    load_manifest,
    trajectory_split,
)


def trajectory(offset=0, length=5):
    q = np.arange(length + 1, dtype=np.float32)[:, None] + offset
    return {"q": q, "qd": q * 2}, {
        "joint_f": q[:-1],
        "contact_depths": np.zeros((length, 0), dtype=np.float32),
    }


def test_commit_resume_and_final_partial_shard(tmp_path):
    writer = DynamicsShardWriter(tmp_path, {"dt": 1 / 60}, trajectories_per_shard=2)
    writer.add_trajectory("a", *trajectory(), split="train")
    assert not load_manifest(tmp_path)["shards"]
    writer.add_trajectory("b", *trajectory(20), split="train")
    writer.add_trajectory("c", *trajectory(40), split="train")
    writer.abort()
    resumed = DynamicsShardWriter(
        tmp_path, {"dt": 1 / 60}, trajectories_per_shard=2, resume=True
    )
    assert resumed.committed_ids == {"a", "b"}
    resumed.add_trajectory("c", *trajectory(40), split="test")
    resumed.close()
    manifest = load_manifest(tmp_path, verify=True)
    assert [len(shard["trajectories"]) for shard in manifest["shards"]] == [2, 1]
    assert len(DynamicsWindowDataset(tmp_path, "train", 3)) == 6
    assert len(DynamicsWindowDataset(tmp_path, "test", 3)) == 3


def test_windows_do_not_cross_trajectory_and_prefix_masks(tmp_path):
    with DynamicsShardWriter(tmp_path, {}) as writer:
        writer.add_trajectory("a", *trajectory(), split="train")
        writer.add_trajectory("b", *trajectory(100), split="validation")
    dataset = DynamicsWindowDataset(tmp_path, "train", 3, include_prefix=True)
    assert len(dataset) == 5
    assert dataset[0]["valid_mask"].tolist() == [True, False, False]
    assert dataset[0]["states"]["q"].flatten().tolist() == [0.0, 1.0, 1.0, 1.0]
    assert dataset[-1]["states"]["q"].flatten().tolist() == [2.0, 3.0, 4.0, 5.0]
    assert all(dataset[i]["trajectory_id"] == "a" for i in range(len(dataset)))
    with pytest.raises(IndexError):
        dataset[5]


def test_independent_worker_handles_bounded_lru(tmp_path):
    with DynamicsShardWriter(tmp_path, {}, trajectories_per_shard=1) as writer:
        for index in range(4):
            writer.add_trajectory(str(index), *trajectory(index * 10), split="train")
    dataset = DynamicsWindowDataset(tmp_path, "train", 2, max_open_shards=1)
    for index in range(len(dataset)):
        dataset[index]
        assert len(dataset._handles) == 1
    copy = pickle.loads(pickle.dumps(dataset))
    assert not copy._handles
    assert copy[0]["states"]["q"].shape == (3, 1)
    batches = list(DataLoader(dataset, batch_size=3, num_workers=2))
    assert sum(len(batch["length"]) for batch in batches) == len(dataset)
    dataset.close()
    copy.close()


def test_validation_and_integrity(tmp_path):
    with DynamicsShardWriter(tmp_path, {}) as writer:
        writer.add_trajectory("a", *trajectory(), split="train")
        with pytest.raises(ValueError, match="duplicate"):
            writer.add_trajectory("a", *trajectory())
        states, transitions = trajectory()
        states["q"][0] = np.nan
        with pytest.raises(ValueError, match="Non-finite"):
            writer.add_trajectory("invalid", states, transitions)
    with pytest.raises(ValueError, match="metadata"):
        DynamicsShardWriter(tmp_path, {"changed": True}, resume=True)
    shard = tmp_path / load_manifest(tmp_path)["shards"][0]["file"]
    with shard.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_manifest(tmp_path, verify=True)


def test_progress_only_publication_preserves_data_identity_and_pending_rows(tmp_path):
    from embodichain.data_pipeline.dynamics.shards import dataset_fingerprint

    with DynamicsShardWriter(tmp_path, {}, trajectories_per_shard=4) as writer:
        writer.add_trajectory("accepted", *trajectory(), split="train")
        progress = {"next_candidate": 1, "accepted": 1, "rejected": 0, "rejections": []}
        writer.manifest["generation_progress"] = progress
        writer.commit()
        first = load_manifest(tmp_path, verify=True)
        assert first["shards"][0]["trajectories"][0]["id"] == "accepted"
        legacy = {
            key: value for key, value in first.items() if key != "generation_progress"
        }
        assert dataset_fingerprint(first) == dataset_fingerprint(legacy)
        progress.update(
            next_candidate=3,
            rejected=2,
            rejections=[{"id": "rejected-1"}, {"id": "rejected-2"}],
        )
        writer.commit()
        final = load_manifest(tmp_path, verify=True)
        assert final["generation_progress"] == progress
        assert final["shards"] == first["shards"]
        assert dataset_fingerprint(final) == dataset_fingerprint(first)


def test_stable_split_not_ingestion_order():
    forward = {str(i): trajectory_split(str(i), 7) for i in range(100)}
    backward = {str(i): trajectory_split(str(i), 7) for i in reversed(range(100))}
    assert forward == backward
    assert set(forward.values()) == {"train", "validation", "test"}


@pytest.mark.parametrize("prefix", [False, True])
def test_resident_windows_preserve_order_padding_ownership_and_bounded_staging(
    tmp_path, monkeypatch, prefix
):
    with DynamicsShardWriter(tmp_path, {}, trajectories_per_shard=2) as writer:
        for index, length in enumerate([3, 5, 11]):
            writer.add_trajectory(
                str(index), *trajectory(index * 100, length), split="train"
            )
    dataset = DynamicsWindowDataset(
        tmp_path, "train", 3, include_prefix=prefix, read_page_steps=2
    )
    requests = [-1, 0, 3, 0, len(dataset) // 2]
    expected = dataset.get_batch(requests)
    original = dataset._read_block
    spans = []

    def tracked(source, name, start, end, shape, dtype):
        spans.append(end - start)
        return original(source, name, start, end, shape, dtype)

    monkeypatch.setattr(dataset, "_read_block", tracked)
    used = dataset.enable_device_cache("cpu", 1024 * 1024)
    assert used > 0 and max(spans) <= 2

    def forbidden(*args, **kwargs):
        raise AssertionError("Resident batches must not reopen trajectory data")

    monkeypatch.setattr(dataset, "_read_block", forbidden)
    actual = dataset.get_batch(requests)
    _assert_same_batch(actual, expected)
    actual["states"]["q"][0].zero_()
    _assert_same_batch(dataset.get_batch(requests), expected)
    monkeypatch.delattr(dataset, "_read_block")
    copied = pickle.loads(pickle.dumps(dataset))
    assert copied._resident is None
    result = dataset.get_batch(requests)
    dataset.close()
    _assert_same_batch(result, expected)


def test_resident_budget_failure_keeps_lazy_dataset_usable(tmp_path):
    with DynamicsShardWriter(tmp_path, {}) as writer:
        writer.add_trajectory("a", *trajectory(), split="train")
    dataset = DynamicsWindowDataset(tmp_path, "train", 3)
    with pytest.raises(ValueError, match="exceeding cache budget"):
        dataset.enable_device_cache("cpu", 1)
    assert dataset._resident is None
    assert dataset[0]["length"] == 3


@pytest.mark.gpu
def test_resident_cuda_windows_match_cpu_and_survive_close(tmp_path):
    with DynamicsShardWriter(tmp_path, {}) as writer:
        writer.add_trajectory("a", *trajectory(), split="train")
    dataset = DynamicsWindowDataset(tmp_path, "train", 3, include_prefix=True)
    expected = dataset.get_batch([0, -1, 0])
    dataset.enable_device_cache("cuda:0", 1024 * 1024)
    actual = dataset.get_batch([0, -1, 0])
    dataset.close()
    for kind in ["states", "transitions"]:
        for name, value in actual[kind].items():
            assert value.device.type == "cuda"
            torch.testing.assert_close(
                value.cpu(), expected[kind][name], atol=0, rtol=0
            )


def test_finalizer_is_best_effort_but_explicit_close_surfaces_errors():
    from collections import OrderedDict
    from types import SimpleNamespace
    from unittest.mock import Mock

    handle = SimpleNamespace(close=Mock(side_effect=RuntimeError("h5py finalized")))
    dataset = object.__new__(DynamicsWindowDataset)
    dataset._handles = OrderedDict([("shard.h5", handle)])
    with pytest.raises(RuntimeError, match="h5py finalized"):
        dataset.close()
    dataset.__del__()
    assert handle.close.call_count == 2


def _assert_same_batch(actual, expected):
    import torch

    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_same_batch(actual[key], expected[key])
    elif isinstance(actual, torch.Tensor):
        assert actual.dtype == expected.dtype
        assert torch.equal(actual, expected)
    else:
        assert actual == expected


@pytest.mark.parametrize("history", [1, 3, 10])
@pytest.mark.parametrize("include_prefix", [False, True])
def test_grouped_batch_matches_individual_windows_across_shards(
    tmp_path, history, include_prefix
):
    import torch
    from torch.utils.data import default_collate

    with DynamicsShardWriter(tmp_path, {}, trajectories_per_shard=1) as writer:
        for index, length in enumerate([1, 3, 9, 25, 30, 11, 7, 41]):
            writer.add_trajectory(
                str(index),
                *trajectory(index * 100, length),
                split="train" if index % 3 else "validation",
            )
    reader = DynamicsWindowDataset(
        tmp_path,
        "train",
        history,
        include_prefix=include_prefix,
        max_open_shards=2,
        read_page_steps=8,
    )
    baseline = DynamicsWindowDataset(
        tmp_path, "train", history, include_prefix=include_prefix, max_open_shards=2
    )
    generator = torch.Generator().manual_seed(7)
    for size in (1, 16, 512):
        indices = torch.randint(len(reader), (size,), generator=generator)
        indices[0] = -1
        _assert_same_batch(
            reader.get_batch(indices),
            default_collate([baseline[int(index)] for index in indices]),
        )
        assert len(reader._handles) <= 2
    reader.close()
    baseline.close()


def test_grouped_batch_preserves_duplicate_and_cross_call_ownership(tmp_path):
    import torch

    with DynamicsShardWriter(tmp_path, {}) as writer:
        writer.add_trajectory("a", *trajectory(), split="train")
    dataset = DynamicsWindowDataset(tmp_path, "train", 3, include_prefix=True)
    first = dataset.get_batch([0, 0, 4])
    untouched = first["states"]["q"][1].clone()
    first["states"]["q"][0].fill_(-999)
    assert torch.equal(first["states"]["q"][1], untouched)
    second = dataset.get_batch([0, 4])
    assert torch.equal(second["states"]["q"][0], untouched)
    assert torch.all(first["states"]["q"][0] == -999)
    dataset.close()
    assert torch.equal(second["states"]["q"][0], untouched)


def test_grouped_reads_are_bounded_for_sparse_long_trajectory(tmp_path, monkeypatch):
    import h5py
    from torch.utils.data import default_collate

    with DynamicsShardWriter(tmp_path, {}) as writer:
        writer.add_trajectory("long", *trajectory(length=4097), split="train")
    dataset = DynamicsWindowDataset(tmp_path, "train", 7, read_page_steps=32)
    selections = []
    original = dataset._read_block

    def observed(group, name, start, stop, shape, dtype):
        block = original(group, name, start, stop, shape, dtype)
        assert block.shape == (stop - start, *shape)
        selections.append(slice(start, stop))
        return block

    monkeypatch.setattr(dataset, "_read_block", observed)
    indices = [0, 31, 32, 63, 2048, len(dataset) - 1, -1, 1]
    batch = dataset.get_batch(indices)
    assert selections
    assert max(selection.stop - selection.start for selection in selections) <= 39
    _assert_same_batch(batch, default_collate([dataset[index] for index in indices]))
    dataset.close()


def test_grouped_batch_validates_indices_before_io(tmp_path):
    with DynamicsShardWriter(tmp_path, {}) as writer:
        writer.add_trajectory("one", *trajectory(), split="train")
    dataset = DynamicsWindowDataset(tmp_path, "train", 3)
    with pytest.raises(ValueError, match="at least one"):
        dataset.get_batch([])
    with pytest.raises(TypeError):
        dataset.get_batch([1.5])
    with pytest.raises(IndexError):
        dataset.get_batch([len(dataset)])
    assert not dataset._handles


def test_grouped_low_level_read_checks_shape_before_buffer_write(tmp_path):
    import h5py

    with DynamicsShardWriter(tmp_path, {}) as writer:
        writer.add_trajectory("one", *trajectory(), split="train")
    manifest = load_manifest(tmp_path)
    with h5py.File(tmp_path / manifest["shards"][0]["file"], "r+") as file:
        del file["0/states/q"]
        file["0/states"].create_dataset("q", data=np.zeros((6, 2), np.float32))
    reader = DynamicsWindowDataset(tmp_path, "train", 3, verify=False)
    with pytest.raises(ValueError, match="shape/range mismatch"):
        reader.get_batch([0])
    reader.close()


def test_grouped_hyperslabs_preserve_dtype_multiaxis_and_utf8_fields(tmp_path):
    from torch.utils.data import default_collate

    states = {
        "q": np.arange(12, dtype=np.float64).reshape(6, 2),
        "qd": np.arange(6, dtype=np.int16).reshape(6, 1),
        "额外值": np.arange(36, dtype=np.float32).reshape(6, 2, 3),
    }
    transitions = {
        "joint_f": np.arange(5, dtype=np.uint8).reshape(5, 1),
        "contact_masks": np.array([[True], [False], [True], [True], [False]]),
        "empty": np.zeros((5, 0), np.float32),
    }
    with DynamicsShardWriter(tmp_path, {}) as writer:
        writer.add_trajectory("typed", states, transitions, split="train")
    reader = DynamicsWindowDataset(tmp_path, "train", 3)
    indices = [2, 0, 1, 1]
    _assert_same_batch(
        reader.get_batch(indices), default_collate([reader[index] for index in indices])
    )
    reader.close()
