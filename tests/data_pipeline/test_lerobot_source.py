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

"""Official SDK integration for the public numeric LeRobot source."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from embodichain.data_pipeline.datasets import (
    LeRobotDataSource,
    LeRobotFieldMappingCfg,
    SequenceSpecCfg,
)

_SCHEMA = {
    "observation": {"names": ["position", "speed"], "dtype": "float32"},
    "action": {"names": ["request"], "semantics": "policy_request", "step_dt": 1 / 30},
}


@pytest.fixture
def official_dataset(tmp_path: Path) -> Path:
    sdk = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    root = tmp_path / "dataset"
    dataset = sdk.LeRobotDataset.create(
        repo_id="local/numeric-test",
        root=root,
        fps=30,
        use_videos=False,
        features={
            "observation.state": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["position", "speed"],
            },
            "action": {"dtype": "float32", "shape": (1,), "names": ["request"]},
            "source.trace": {
                "dtype": "int64",
                "shape": (4,),
                "names": ["origin", "episode", "frame", "generation"],
            },
            "source.continuity": {
                "dtype": "int64",
                "shape": (1,),
                "names": ["continuity"],
            },
            "source.subtask": {"dtype": "int64", "shape": (1,), "names": ["subtask"]},
        },
    )
    for row, frames in enumerate(([0, 1, 3, 4], [0, 1])):
        for frame in frames:
            dataset.add_frame(
                {
                    "observation.state": np.array([row, frame], dtype=np.float32),
                    "action": np.array([frame / 10], dtype=np.float32),
                    "source.trace": np.array([7, row, frame, 1], dtype=np.int64),
                    "source.continuity": np.array([int(frame >= 3)], dtype=np.int64),
                    "source.subtask": np.array([int(frame >= 3)], dtype=np.int64),
                    "task": "pick",
                }
            )
        # LeRobot 0.4.x declares shape-(1,) fields as Arrow scalars. Normalize
        # its pending scalar buffers for NumPy 2's strict scalar conversion.
        for key in ("action", "source.continuity", "source.subtask"):
            dataset.episode_buffer[key] = [
                value.item() for value in dataset.episode_buffer[key]
            ]
        dataset.save_episode()
    dataset.finalize()
    return root


def _load(root: Path, **kwargs) -> LeRobotDataSource:
    return LeRobotDataSource(
        root,
        schema=_SCHEMA,
        source_id="numeric-test-v1",
        repo_id="local/numeric-test",
        memory_limit_bytes=1024 * 1024,
        **kwargs,
    )


def test_external_dataset_without_showcase_sidecar_is_usable(official_dataset):
    source = _load(official_dataset, read_batch_size=2)
    assert source.data.batch_size == (2, 4)
    assert source.data["valid"].sum() == 6
    assert source.data["obs"][0, :, 1].tolist() == [0, 1, 3, 4]
    assert source.metadata["storage"]["resident_tensor_bytes"] <= 1024 * 1024
    assert not (official_dataset / "meta/embodichain_ods_corpus.json").exists()
    batch = source.sample_context(16, SequenceSpecCfg(2, 4))
    assert batch["valid"][:, 1].all()


def test_explicit_provenance_restores_gaps_continuity_and_language(official_dataset):
    mapping = LeRobotFieldMappingCfg(
        provenance_key="source.trace",
        metadata_keys={
            "continuity_id": "source.continuity",
            "task_index": "task_index",
            "subtask_index": "source.subtask",
        },
    )
    source = _load(
        official_dataset,
        mapping=mapping,
        read_batch_size=2,
        language={"task": {0: "pick"}, "subtask": {0: "approach", 1: "release"}},
    )
    assert source.data.batch_size == (2, 5)
    assert source.data["valid"][0].tolist() == [True, True, False, True, True]
    assert source.data["frame_ids"][0, 3].tolist() == [7, 0, 3]
    assert source.resolve_language(source.data[source.data["valid"]])["subtask"] == [
        "approach",
        "approach",
        "release",
        "release",
        "approach",
        "approach",
    ]
    first = source.sample_context(
        128, SequenceSpecCfg(2, 4), generator=torch.Generator().manual_seed(9)
    )
    spans = first["continuity_id"] == first["continuity_id"][:, 1:2]
    assert spans[first["valid"]].all()


def test_resident_limit_rejects_before_reading_payloads(official_dataset, monkeypatch):
    from embodichain.data_pipeline.datasets import lerobot

    def unexpected_read(*args):
        raise AssertionError(
            "Resident allocation must be rejected before numeric reads"
        )

    monkeypatch.setattr(lerobot, "_column_batch", unexpected_read)
    with pytest.raises(MemoryError, match="memory_limit_bytes"):
        LeRobotDataSource(
            official_dataset, schema=_SCHEMA, source_id="test", memory_limit_bytes=1
        )


def test_padding_capacity_cannot_exceed_resident_allowance(official_dataset):
    with pytest.raises(MemoryError, match="padded resident"):
        _load(official_dataset, max_episode_steps=10**9)


def test_fixed_capacity_and_read_chunking_preserve_numeric_results(official_dataset):
    from embodichain.learning.sl import SequenceBatchSource, SupervisedBatch

    mapping = LeRobotFieldMappingCfg(provenance_key="source.trace")
    first = _load(
        official_dataset, mapping=mapping, max_episode_steps=8, read_batch_size=1
    )
    second = _load(
        official_dataset, mapping=mapping, max_episode_steps=8, read_batch_size=3
    )
    assert first.data.batch_size == (2, 8)
    assert all(
        torch.equal(first.data[key], second.data[key]) for key in first.data.keys()
    )
    assert first.source_id == second.source_id

    def encode(context):
        return SupervisedBatch(
            context["obs"][:, :2],
            context["actions"][:, 1:],
            {"target": context["loss_mask"][:, 1:]},
        )

    bridges = [
        SequenceBatchSource(
            source,
            SequenceSpecCfg(2, 4),
            encoder=encode,
            encoder_version="sdk-toy-actions-v1",
            immutable=True,
        )
        for source in (first, second)
    ]
    assert bridges[0].source_id == bridges[1].source_id
    batches = [
        bridge.sample(16, generator=torch.Generator().manual_seed(13))
        for bridge in bridges
    ]
    torch.testing.assert_close(batches[0].inputs, batches[1].inputs)
    torch.testing.assert_close(batches[0].targets, batches[1].targets)
    for key in ("valid", "loss_mask", "frame_ids", "generations"):
        torch.testing.assert_close(
            batches[0].metadata["sequence_source"]["context"][key],
            batches[1].metadata["sequence_source"]["context"][key],
        )


def test_ordered_feature_names_are_checked(official_dataset):
    schema = copy.deepcopy(_SCHEMA)
    schema["observation"]["names"].reverse()
    with pytest.raises(ValueError, match="ordered names"):
        LeRobotDataSource(
            official_dataset,
            schema=schema,
            source_id="test",
            memory_limit_bytes=1024 * 1024,
        )


def test_missing_action_semantics_are_not_inferred_from_shape(tmp_path):
    schema = copy.deepcopy(_SCHEMA)
    del schema["action"]["semantics"]
    with pytest.raises(ValueError, match="Explicit action semantics"):
        LeRobotDataSource(
            tmp_path, schema=schema, source_id="test", memory_limit_bytes=1024
        )


def test_timestamp_corruption_is_rejected(official_dataset, monkeypatch):
    from embodichain.data_pipeline.datasets import lerobot

    original = lerobot._column_batch

    def corrupted(*args):
        result = original(*args)
        result["timestamp"] = result["timestamp"] + 0.02
        return result

    monkeypatch.setattr(lerobot, "_column_batch", corrupted)
    with pytest.raises(ValueError, match="timestamps"):
        _load(official_dataset)


def test_reads_stay_bounded_and_on_cpu(official_dataset, monkeypatch):
    from embodichain.data_pipeline.datasets import lerobot

    original = lerobot._column_batch
    sizes = []

    def tracked(dataset, keys, start, count):
        sizes.append(count)
        return original(dataset, keys, start, count)

    monkeypatch.setattr(lerobot, "_column_batch", tracked)
    with torch.device("meta"):
        source = _load(official_dataset, read_batch_size=2)
    assert sizes and max(sizes) <= 2
    assert all(value.device.type == "cpu" for value in source.data.values())
