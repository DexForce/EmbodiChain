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

"""Exact, bounded shared minibatch order and tamper detection."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from embodichain.learning.dynamics._sample_order import (
    _CheckpointedOrder,
    _OfficialShuffleOrder,
    _open_order,
    _permutation_sha256,
    _write_checkpointed_order,
    _write_official_shuffle_order,
)


def test_checkpointed_stream_matches_the_original_matrix_and_random_seeks(tmp_path):
    steps, batch, block_steps = 2071, 16, 1024
    generator = torch.Generator().manual_seed(19)
    expected = torch.cat(
        [
            torch.randint(
                1037, (min(block_steps, steps - start), batch), generator=generator
            )
            for start in range(0, steps, block_steps)
        ]
    ).numpy()
    path, rng = _write_checkpointed_order(
        tmp_path / "order.json",
        dataset_size=1037,
        steps=steps,
        batch_size=batch,
        seed=19,
        block_steps=block_steps,
    )
    actual = _open_order(path)
    assert isinstance(actual, _CheckpointedOrder)
    for index in [0, 1023, 1024, 2070, 2, 2048, 1024]:
        np.testing.assert_array_equal(actual[index], expected[index])
    row = actual[0]
    row[:] = -1
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual._block.nbytes <= block_steps * batch * 8
    assert rng.stat().st_size < expected.nbytes
    with pytest.raises(IndexError):
        actual[steps]
    with pytest.raises(IndexError):
        actual[-1]


def test_checkpoint_stream_first_block_is_independent_of_declared_budget(tmp_path):
    for name, steps in [("a", 8), ("b", 3000)]:
        _write_checkpointed_order(
            tmp_path / (name + ".json"),
            dataset_size=91,
            steps=steps,
            batch_size=8,
            seed=7,
        )
    left, right = [_open_order(tmp_path / (name + ".json")) for name in ["a", "b"]]
    for step in range(8):
        np.testing.assert_array_equal(left[step], right[step])


def test_changed_rng_artifact_or_torch_version_fails_before_training(tmp_path):
    path, rng = _write_checkpointed_order(
        tmp_path / "order.json",
        dataset_size=97,
        steps=40,
        batch_size=4,
        seed=3,
        block_steps=8,
    )
    spec = json.loads(path.read_text())
    spec["torch_version"] = "other"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="Torch version"):
        _open_order(path)
    spec["torch_version"] = str(torch.__version__)
    path.write_text(json.dumps(spec))
    values = np.load(rng).copy()
    values[-1, 20] ^= 1
    np.save(rng, values)
    with pytest.raises(ValueError, match="checksum"):
        _open_order(path)


def test_legacy_matrix_still_opens_read_only(tmp_path):
    values = np.arange(24).reshape(6, 4)
    np.save(tmp_path / "order.npy", values)
    actual = _open_order(tmp_path / "order.npy")
    assert isinstance(actual, np.memmap)
    np.testing.assert_array_equal(actual, values)


def _official_order(tmp_path: Path) -> tuple[Path, Path, Path, np.ndarray]:
    # Five complete batches/pass; each twelve-update epoch discards the unused
    # tail of its third pass. The final three-update prefix uses one more pass.
    size, batch, epoch_steps, steps = 23, 4, 12, 27
    seeds = np.arange(101, 108, dtype=np.int64)
    hashes = [
        _permutation_sha256(
            torch.randperm(
                size, generator=torch.Generator().manual_seed(int(seed))
            ).numpy()
        )
        for seed in seeds
    ]
    paths = _write_official_shuffle_order(
        tmp_path / "official.json",
        dataset_size=size,
        steps=steps,
        batch_size=batch,
        epoch_steps=epoch_steps,
        pass_seeds=seeds,
        permutation_sha256=hashes,
        provenance={
            "source_rng_evidence": "Synthetic conditional RandomSampler fixture",
            "producer_torch_version": "2.2.2+cpu",
        },
    )
    # Independent oracle: actual Torch RandomSampler/DataLoader, dropping
    # incomplete minibatches and abandoning the final iterator each epoch.
    from torch.utils.data import DataLoader, RandomSampler

    result = []
    seed_iterator = iter(seeds)
    while len(result) < steps:
        remaining = min(epoch_steps, steps - len(result))
        while remaining:
            sampler = RandomSampler(
                range(size),
                replacement=False,
                generator=torch.Generator().manual_seed(int(next(seed_iterator))),
            )
            loader = DataLoader(
                range(size),
                batch_size=batch,
                sampler=sampler,
                drop_last=True,
                num_workers=0,
                generator=torch.Generator().manual_seed(8),
            )
            for indices in loader:
                result.append(indices.numpy())
                remaining -= 1
                if not remaining:
                    break
    return *paths, np.stack(result)


def test_official_order_matches_real_random_sampler_and_epoch_resets(
    tmp_path: Path,
) -> None:
    path, _, _, expected = _official_order(tmp_path)
    actual = _open_order(path)
    assert isinstance(actual, _OfficialShuffleOrder)
    assert actual.shape == expected.shape
    for index in range(len(expected)):
        np.testing.assert_array_equal(actual[index], expected[index])
    for index in (26, 0, 5, 11, 12, 9, 25, 1):
        np.testing.assert_array_equal(actual[index], expected[index])
    assert actual._permutation.nbytes == actual.dataset_size * 8
    assert not actual.seeds.flags.writeable
    for index in (-1, len(expected)):
        with pytest.raises(IndexError):
            actual[index]


def test_official_rows_remain_owned_across_seeks_and_reopen(tmp_path: Path) -> None:
    path, _, _, expected = _official_order(tmp_path)
    actual = _open_order(path)
    row = actual[0]
    actual[26]
    np.testing.assert_array_equal(row, expected[0])
    row[:] = -1
    np.testing.assert_array_equal(actual[0], expected[0])
    resumed = _open_order(path)
    for index in (12, 25, 26):
        np.testing.assert_array_equal(resumed[index], expected[index])


def test_trainer_admits_official_order_and_rejects_a_different_dataset_size(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace
    from embodichain.learning.dynamics.trainer import DynamicsTrainer

    path, _, _, expected = _official_order(tmp_path)
    # Exercise the production order admission boundary without creating a
    # model, optimizer, simulation or allocating CUDA memory.
    owner = DynamicsTrainer.__new__(DynamicsTrainer)
    owner.cfg = SimpleNamespace(total_steps=27, batch_size=4)
    owner.train_dataset = range(23)
    order = owner._prepare_order(path)
    assert isinstance(order, _OfficialShuffleOrder)
    np.testing.assert_array_equal(order[12], expected[12])
    assert owner._sample_order_identity["path"] == str(path.resolve())
    owner.train_dataset = range(24)
    with pytest.raises(ValueError, match="dataset size mismatch"):
        owner._prepare_order(path)


@pytest.mark.parametrize("payload", ["seeds", "hashes"])
def test_official_payload_tampering_is_rejected(tmp_path: Path, payload: str) -> None:
    path, seeds, hashes, _ = _official_order(tmp_path)
    target = seeds if payload == "seeds" else hashes
    values = np.load(target, allow_pickle=False).copy()
    values[0] = int(values[0]) ^ 1 if payload == "seeds" else b"0" * 64
    np.save(target, values, allow_pickle=False)
    with pytest.raises(ValueError, match="payload checksum"):
        _open_order(path)


def test_official_rehashed_wrong_permutation_still_fails_before_yield(
    tmp_path: Path,
) -> None:
    import hashlib

    path, _, hashes, _ = _official_order(tmp_path)
    values = np.load(hashes, allow_pickle=False).copy()
    values[0] = b"0" * 64
    np.save(hashes, values, allow_pickle=False)
    spec = json.loads(path.read_text())
    spec["permutation_hashes_sha256"] = hashlib.sha256(hashes.read_bytes()).hexdigest()
    path.write_text(json.dumps(spec))
    reader = _open_order(path)
    with pytest.raises(ValueError, match="permutation checksum"):
        reader[0]


@pytest.mark.parametrize(
    "change",
    [
        {"drop_last": False},
        {"discard_epoch_tail": False},
        {"permutation_algorithm": "other"},
        {"seeds_file": "../outside.npy"},
    ],
)
def test_official_unknown_or_escaping_contract_is_rejected(
    tmp_path: Path, change: dict
) -> None:
    path, _, _, _ = _official_order(tmp_path)
    spec = json.loads(path.read_text())
    spec.update(change)
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="contract|sibling file"):
        _open_order(path)


def test_official_full_budget_storage_is_not_a_dense_minibatch_matrix(
    tmp_path: Path,
) -> None:
    size, batch, steps, epoch_steps = 931840, 512, 4995000, 5000
    passes = 2997
    # This is a scale fixture, not claimed original RNG capture: repeated seeds
    # let the real reader exercise the full formal shape with one hash oracle.
    seeds = np.full(passes, 107, dtype=np.int64)
    expected = torch.randperm(
        size, generator=torch.Generator().manual_seed(107)
    ).numpy()
    digest = _permutation_sha256(expected)
    path, seed_path, hash_path = _write_official_shuffle_order(
        tmp_path / "full.json",
        dataset_size=size,
        steps=steps,
        batch_size=batch,
        epoch_steps=epoch_steps,
        pass_seeds=seeds,
        permutation_sha256=[digest] * passes,
        provenance={"source_rng_evidence": "Synthetic storage/shape fixture"},
    )
    reader = _open_order(path)
    np.testing.assert_array_equal(reader[0], expected[:batch])
    # The final epoch's third pass consumes only 1360 minibatches.
    start = (1360 - 1) * batch
    np.testing.assert_array_equal(reader[steps - 1], expected[start : start + batch])
    assert reader._permutation.nbytes == size * 8
    assert sum(p.stat().st_size for p in (path, seed_path, hash_path)) < 256 * 1024


@pytest.mark.parametrize(
    "change",
    [
        {"dataset_size": 0},
        {"dataset_size": 4194305},
        {"epoch_steps": 1},
        {"pass_seeds": [1]},
        {"permutation_sha256": ["wrong"]},
        {"provenance": {}},
    ],
)
def test_official_writer_rejects_invalid_or_unbounded_inputs(
    tmp_path: Path, change: dict
) -> None:
    options = dict(
        dataset_size=23,
        steps=27,
        batch_size=4,
        epoch_steps=12,
        pass_seeds=np.arange(7, dtype=np.int64),
        permutation_sha256=["0" * 64] * 7,
        provenance={"source_rng_evidence": "Synthetic negative fixture"},
    )
    options.update(change)
    with pytest.raises(ValueError):
        _write_official_shuffle_order(tmp_path / "invalid.json", **options)
