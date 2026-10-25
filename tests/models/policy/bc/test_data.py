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

"""Fixed episode boundaries, causal camera reads and stable sampling."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import h5py
import numpy as np
import pytest
import torch

from embodichain_models.policy.bc.data import BCWindowDataset
from embodichain_models.policy.bc.supervised import HDF5BCSource, PrefetchedBCSource
from embodichain_models.policy.bc.model import ChunkBCModel, ChunkBCModelCfg
from embodichain.learning.sl import SupervisedBatch, SupervisedLearner
from embodichain_models.policy.bc.supervised import ChunkBCObjective
from embodichain.learning.sl.trainer import SupervisedTrainer, SupervisedTrainerCfg


def _corpus(tmp_path, *, cameras=True, lengths=(8, 11)):
    episodes = []
    for ordinal, length in enumerate(lengths):
        path = tmp_path / f"episode{ordinal}.h5"
        with h5py.File(path, "w") as handle:
            values = np.arange(length, dtype=np.float32)[:, None] + ordinal * 100
            handle["observations"] = np.concatenate((values, values + 10), 1)
            handle["targets"] = values * 2
            if cameras:
                rgb = np.broadcast_to(
                    np.arange(length, dtype=np.uint8)[:, None, None, None],
                    (length, 256, 256, 3),
                )
                handle.create_dataset(
                    "images/front", data=rgb, chunks=(1, 256, 256, 3), compression="lzf"
                )
        episodes.append(
            {
                "path": path.name,
                "length": length,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "complete": True,
                "cameras": ["front"] if cameras else [],
                "episodes": episodes,
            }
        )
    )
    return manifest


def test_fixed_source_reads_only_causal_camera_history_and_matching_targets(tmp_path):
    manifest = _corpus(tmp_path)
    dataset = BCWindowDataset(
        manifest, history_length=2, prediction_horizon=3, max_open_files=1
    )
    try:
        source = HDF5BCSource(dataset)
        batch = source.sample(12, generator=torch.Generator().manual_seed(4))
        assert batch.inputs["history"].shape == (12, 2, 2)
        assert batch.inputs["images"]["front"].shape == (12, 2, 3, 256, 256)
        for row, (ordinal, anchor) in enumerate(
            zip(batch.metadata["episode_ids"], batch.metadata["anchors"])
        ):
            expected = torch.arange(
                int(anchor) - 1, int(anchor) + 1
            ).float() + 100 * int(ordinal)
            torch.testing.assert_close(batch.inputs["history"][row, :, 0], expected)
            torch.testing.assert_close(
                batch.targets[row, :, 0],
                (
                    torch.arange(int(anchor), int(anchor) + 3) + 100 * int(ordinal)
                ).float()
                * 2,
            )
            torch.testing.assert_close(
                batch.inputs["images"]["front"][row, :, 0, 0, 0],
                (expected - 100 * int(ordinal)) / 255,
            )
        copy = batch.inputs["images"]["front"].clone()
        source.sample(2, generator=torch.Generator().manual_seed(5))
        dataset.close()
        torch.testing.assert_close(batch.inputs["images"]["front"], copy)
    finally:
        dataset.close()


def test_sampling_rng_and_transformed_targets_are_reproducible(tmp_path):
    manifest = _corpus(tmp_path, cameras=False)
    dataset = BCWindowDataset(
        manifest,
        history_length=1,
        prediction_horizon=2,
        target_encoder=lambda fields: fields["targets"] + 7,
        target_fields=("targets",),
        target_encoder_id="targets-plus7@1",
    )
    try:
        first = dataset.sample(8, generator=torch.Generator().manual_seed(3))
        second = dataset.sample(8, generator=torch.Generator().manual_seed(3))
        torch.testing.assert_close(
            first["inputs"]["history"], second["inputs"]["history"], atol=0, rtol=0
        )
        torch.testing.assert_close(first["targets"], second["targets"], atol=0, rtol=0)
        torch.testing.assert_close(
            first["targets"][:, 0], first["inputs"]["history"][:, 0, :1] * 2 + 7
        )
    finally:
        dataset.close()


@pytest.mark.parametrize("encoded", [False, True])
def test_tail_target_keeps_complete_causal_history_and_finite_owned_padding(
    tmp_path, encoded
):
    manifest = _corpus(tmp_path, lengths=(2,))
    with pytest.raises(ValueError, match="complete requested window"):
        BCWindowDataset(manifest, history_length=2, prediction_horizon=4)

    def encode(fields):
        # Padding before encoding would divide by zero. Only real targets enter.
        assert fields["targets"].shape == (1, 1, 1)
        return fields["targets"].reciprocal()

    dataset = BCWindowDataset(
        manifest,
        history_length=2,
        prediction_horizon=4,
        include_partial_targets=True,
        **(
            {
                "target_encoder": encode,
                "target_fields": ("targets",),
                "target_encoder_id": "reciprocal-valid-targets@1",
            }
            if encoded
            else {}
        ),
    )
    try:
        batch = HDF5BCSource(dataset).sample(
            1, generator=torch.Generator().manual_seed(4)
        )
        torch.testing.assert_close(
            batch.inputs["history"][0, :, 0], torch.tensor([0.0, 1.0])
        )
        torch.testing.assert_close(
            batch.inputs["images"]["front"][0, :, 0, 0, 0],
            torch.tensor([0.0, 1.0 / 255]),
        )
        assert batch.masks["target_valid"].tolist() == [[True, False, False, False]]
        expected = 0.5 if encoded else 2.0
        torch.testing.assert_close(
            batch.targets[0, :, 0], torch.tensor([expected, 0.0, 0.0, 0.0])
        )
        assert torch.isfinite(batch.targets).all() and not batch.targets.requires_grad

        model = ChunkBCModel(
            ChunkBCModelCfg(
                observation_dim=2,
                target_dim=1,
                history_length=2,
                prediction_horizon=4,
                hidden_dims=(4,),
            )
        )
        loss = (
            ChunkBCObjective()
            .compute_loss(
                model,
                SupervisedBatch(
                    {"history": batch.inputs["history"]}, batch.targets, batch.masks
                ),
            )
            .loss
        )
        loss.backward()
        assert all(
            torch.isfinite(parameter.grad).all() for parameter in model.parameters()
        )
        torch.testing.assert_close(model.head[-1].bias.grad[1:], torch.zeros(3))

        previous = batch.targets.clone()
        another = dataset.sample(1, generator=torch.Generator().manual_seed(5))
        another["targets"].fill_(99)
        dataset.close()
        torch.testing.assert_close(batch.targets, previous, atol=0, rtol=0)
    finally:
        dataset.close()
    short = tmp_path / "short"
    short.mkdir()
    with pytest.raises(ValueError, match="complete requested history"):
        BCWindowDataset(
            _corpus(short, cameras=False, lengths=(1,)),
            history_length=2,
            prediction_horizon=4,
            include_partial_targets=True,
        )


def test_partial_encoder_preserves_row_order_and_only_masks_unavailable_futures(
    tmp_path,
):
    manifest = _corpus(tmp_path, cameras=False)
    dataset = BCWindowDataset(
        manifest,
        history_length=2,
        prediction_horizon=4,
        include_partial_targets=True,
        target_encoder=lambda fields: fields["targets"] + 7,
        target_fields=("targets",),
        target_encoder_id="targets-plus7@1",
    )
    try:
        batch = dataset.sample(32, generator=torch.Generator().manual_seed(3))
        counts = batch["masks"]["target_valid"].sum(1)
        assert set(counts.tolist()) == {1, 2, 3, 4}
        for row, (ordinal, anchor, count) in enumerate(
            zip(batch["metadata"]["episode_ids"], batch["metadata"]["anchors"], counts)
        ):
            count, anchor, ordinal = int(count), int(anchor), int(ordinal)
            expected = (
                torch.arange(anchor, anchor + count) + 100 * ordinal
            ).float() * 2 + 7
            torch.testing.assert_close(batch["targets"][row, :count, 0], expected)
            torch.testing.assert_close(
                batch["targets"][row, count:, 0], torch.zeros(4 - count)
            )
            assert batch["masks"]["target_valid"][row].tolist() == [True] * count + [
                False
            ] * (4 - count)
    finally:
        dataset.close()


def test_partial_or_modified_corpus_is_rejected(tmp_path):
    manifest = _corpus(tmp_path, cameras=False)
    data = json.loads(manifest.read_text())
    data["complete"] = False
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="completed"):
        BCWindowDataset(manifest)
    data["complete"] = True
    manifest.write_text(json.dumps(data))
    dataset = BCWindowDataset(manifest)
    with h5py.File(tmp_path / data["episodes"][0]["path"], "a") as handle:
        handle["targets"][0] = -999
    with pytest.raises(ValueError, match="checksum"):
        dataset.read_episode(0, ("targets",))


def test_view_identity_rejects_changed_feature_slice_before_resume_mutation(tmp_path):
    manifest = _corpus(tmp_path, cameras=False)
    left = BCWindowDataset(manifest, observation_slice=slice(0, 1))
    right = BCWindowDataset(manifest, observation_slice=slice(1, 2))
    assert left.dataset_id == right.dataset_id and left.source_id != right.source_id
    assert not torch.equal(
        left.sample(1, generator=torch.Generator().manual_seed(3))["inputs"]["history"],
        right.sample(1, generator=torch.Generator().manual_seed(3))["inputs"][
            "history"
        ],
    )
    cfg = SupervisedTrainerCfg(
        total_updates=2,
        batch_size=2,
        output_dir=str(tmp_path / "run"),
        checkpoint_interval=0,
    )

    def trainer(dataset):
        model = ChunkBCModel(
            ChunkBCModelCfg(observation_dim=1, target_dim=1, hidden_dims=(4,))
        )
        learner = SupervisedLearner(
            model, ChunkBCObjective(), torch.optim.Adam(model.parameters(), lr=1e-3)
        )
        return SupervisedTrainer(learner, HDF5BCSource(dataset), cfg)

    original = trainer(left)
    original.fit(stop_after=1)
    candidate = trainer(right)
    before = deepcopy(candidate.learner.model.state_dict())
    with pytest.raises(ValueError, match="dataset"):
        candidate.resume(tmp_path / "run" / "last.pt")
    assert candidate.learner.num_updates == 0
    for name, value in before.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(
                candidate.learner.model.state_dict()[name], value, atol=0, rtol=0
            )
    left.close()
    right.close()


def test_target_transform_and_window_contract_are_bound_to_source_identity(tmp_path):
    manifest = _corpus(tmp_path, cameras=False)
    encode = lambda fields: fields["targets"]
    with pytest.raises(ValueError, match="transform identity"):
        BCWindowDataset(manifest, target_encoder=encode, target_fields=("targets",))
    first = BCWindowDataset(
        manifest,
        target_encoder=encode,
        target_fields=("targets",),
        target_encoder_id="position-targets@1",
    )
    second = BCWindowDataset(
        manifest,
        target_encoder=encode,
        target_fields=("targets",),
        target_encoder_id="effort-targets@1",
    )
    future = BCWindowDataset(
        manifest,
        history_length=2,
        prediction_horizon=2,
        target_encoder=encode,
        target_fields=("targets",),
        target_encoder_id="position-targets@1",
    )
    assert len({first.source_id, second.source_id, future.source_id}) == 3


def test_one_batch_prefetch_preserves_sampling_cursor_and_retry_batches(tmp_path):
    manifest = _corpus(tmp_path, cameras=False)
    direct = HDF5BCSource(BCWindowDataset(manifest, prediction_horizon=2))
    prefetched = PrefetchedBCSource(
        HDF5BCSource(BCWindowDataset(manifest, prediction_horizon=2))
    )
    expected_rng = torch.Generator().manual_seed(9)
    actual_rng = torch.Generator().manual_seed(9)
    try:
        for size in (3, 3, 5):
            before = actual_rng.get_state().clone()
            expected = direct.sample(size, generator=expected_rng)
            actual = prefetched.sample(size, generator=actual_rng)
            torch.testing.assert_close(
                actual.inputs["history"], expected.inputs["history"], atol=0, rtol=0
            )
            torch.testing.assert_close(actual.targets, expected.targets, atol=0, rtol=0)
            torch.testing.assert_close(
                actual_rng.get_state(), expected_rng.get_state(), atol=0, rtol=0
            )
            # A rejected optimizer update retries its old committed cursor.
            actual_rng.set_state(before)
            retry = prefetched.sample(size, generator=actual_rng)
            torch.testing.assert_close(retry.targets, actual.targets, atol=0, rtol=0)
            torch.testing.assert_close(
                actual_rng.get_state(), expected_rng.get_state(), atol=0, rtol=0
            )
    finally:
        direct.close()
        prefetched.close()


@pytest.mark.parametrize("include_partial_targets", [False, True])
def test_prefetched_training_resume_rebuilds_unused_batch_exactly(
    tmp_path, include_partial_targets
):
    manifest = _corpus(tmp_path, cameras=False)
    from embodichain_models.policy.bc import build_supervised_trainer

    statistics = tmp_path / "normalization.pt"
    torch.save(
        {
            "observation_mean": torch.zeros(1),
            "observation_std": torch.ones(1),
            "target_mean": torch.zeros(1),
            "target_std": torch.ones(1),
        },
        statistics,
    )

    def trainer(name, *, partial=include_partial_targets):
        return build_supervised_trainer(
            {
                "model": {
                    "observation_dim": 1,
                    "target_dim": 1,
                    "hidden_dims": [4],
                    "history_length": 2,
                    "prediction_horizon": 3,
                },
                "data": {
                    "manifest": str(manifest),
                    "normalization": str(statistics),
                    "reader": {
                        "observation_slice": [0, 1],
                        "include_partial_targets": partial,
                    },
                },
                "trainer": {
                    "total_updates": 4,
                    "batch_size": 2,
                    "output_dir": str(tmp_path / name),
                    "checkpoint_interval": 0,
                    "seed": 19,
                },
            }
        )

    full = trainer("full")
    resumed = trainer("resumed")
    try:
        full.fit(stop_after=2)
        prefix = tmp_path / "prefix.pt"
        full.save(prefix)
        full.fit()
        resumed.resume(prefix)
        resumed.fit()
        for name, value in full.learner.model.state_dict().items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(
                    resumed.learner.model.state_dict()[name], value, atol=0, rtol=0
                )
        torch.testing.assert_close(
            full.sampler.get_state(), resumed.sampler.get_state(), atol=0, rtol=0
        )
        # Recipe identity must reject changed optimizer semantics before mutation.
        checkpoint = torch.load(prefix, weights_only=False)
        checkpoint["signature"]["optimizer"]["learning_rate"] = 0.1
        changed = tmp_path / "changed.pt"
        torch.save(checkpoint, changed)
        before = deepcopy(resumed.learner.model.state_dict())
        with pytest.raises(ValueError, match="recipe"):
            resumed.resume(changed)
        for name, value in before.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(
                    resumed.learner.model.state_dict()[name], value, atol=0, rtol=0
                )
        opposite = trainer("opposite", partial=not include_partial_targets)
        try:
            assert opposite.source.source_id != full.source.source_id
            before = deepcopy(opposite.learner.model.state_dict())
            with pytest.raises(ValueError, match="recipe or dataset"):
                opposite.resume(prefix)
            assert opposite.learner.num_updates == 0
            for name, value in before.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(
                        opposite.learner.model.state_dict()[name], value, atol=0, rtol=0
                    )
        finally:
            opposite.source.close()
    finally:
        full.source.close()
        resumed.source.close()


def test_rgb_storage_remains_fp32_under_different_default_dtype(tmp_path):
    manifest = _corpus(tmp_path)
    dataset = BCWindowDataset(manifest)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        batch = dataset.sample(1, generator=torch.Generator().manual_seed(2))
        assert batch["inputs"]["history"].dtype == torch.float32
        assert batch["inputs"]["images"]["front"].dtype == torch.float32
    finally:
        torch.set_default_dtype(previous)
        dataset.close()


def test_encoded_targets_own_storage_and_do_not_retain_teacher_graph(tmp_path):
    manifest = _corpus(tmp_path, cameras=False)
    reusable = torch.ones(2, 1, 1, requires_grad=True)
    dataset = BCWindowDataset(
        manifest,
        target_encoder=lambda fields: reusable,
        target_fields=("targets",),
        target_encoder_id="constant-reference@1",
    )
    try:
        batch = dataset.sample(2, generator=torch.Generator().manual_seed(2))
        with torch.no_grad():
            reusable.fill_(9)
        assert not batch["targets"].requires_grad
        torch.testing.assert_close(batch["targets"], torch.ones(2, 1, 1))
    finally:
        dataset.close()
