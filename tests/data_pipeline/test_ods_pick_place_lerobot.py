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

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict
from embodichain.data_pipeline.datasets import (
    TensorDataSource,
)

from examples.data_pipeline._ods_pick_place_benchmark import (
    _content_manifest,
)
from examples.data_pipeline._ods_pick_place_lerobot import (
    DirectLeRobotWriter,
    load_lerobot_corpus,
)

lerobot_module = pytest.importorskip("lerobot.datasets.lerobot_dataset")

_SCHEMA = {
    "observation": {"names": ["measured_joint", "cube_height"]},
    "action": {
        "semantics": "test_request",
        "names": ["arm_request", "gripper_request"],
        "step_dt": 1.0 / 30,
    },
    "version": 1,
}


def _make_corpus(destination: Path) -> TensorDataSource:
    valid = torch.tensor([[True, True, False], [True, False, True]])
    ids = torch.stack(
        (
            torch.full((2, 3), 11),
            torch.tensor([5, 7])[:, None].expand(2, 3),
            torch.arange(3)[None].expand(2, 3),
        ),
        -1,
    )
    tensors = {
        "observations": torch.arange(12).float().reshape(2, 3, 2),
        "actions": -torch.arange(12).float().reshape(2, 3, 2) - 1,
        "valid": valid,
        "frame_ids": ids,
        "generations": torch.tensor([[1] * 3, [3] * 3]),
    }
    for key in tensors.keys() - {"valid"}:
        tensors[key][~valid] = 0
    with DirectLeRobotWriter(
        destination, schema=_SCHEMA, episode_count=2, max_episode_steps=3
    ) as writer:
        for row in range(2):
            writer.submit({key: value[row] for key, value in tensors.items()})
        writer.finalize()
    return TensorDataSource(
        TensorDict(
            {
                ("obs" if key == "observations" else key): value
                for key, value in tensors.items()
            },
            batch_size=tensors["valid"].shape,
        ),
        schema=_SCHEMA,
        source_id="test-corpus",
        metadata={"schema": _SCHEMA, **_content_manifest(tensors, _SCHEMA)},
    )


def test_official_lerobot_roundtrip_preserves_padded_numeric_corpus(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "lerobot"
    original = _make_corpus(destination)
    restored = load_lerobot_corpus(destination, expected_schema=_SCHEMA)
    for field in ("observations", "actions", "valid", "frame_ids"):
        assert torch.equal(
            original.data["obs" if field == "observations" else field],
            restored.data["obs" if field == "observations" else field],
        )
    assert torch.equal(original.data["generations"], restored.data["generations"])
    assert restored.metadata["content_id"] == original.metadata["content_id"]
    assert restored.metadata["storage"]["per_minibatch_lerobot_io"] is False
    assert restored.metadata["storage"]["format"] == "lerobot"
    direct = lerobot_module.LeRobotDataset(
        "local/ods-pick-place", root=destination, download_videos=False
    )
    assert len(direct) == 4
    assert torch.equal(direct[3]["ods.provenance"][:3], torch.tensor([11, 7, 2]))


def test_lerobot_export_rejects_wrong_control_frequency(tmp_path: Path) -> None:
    destination = tmp_path / "lerobot"
    with pytest.raises(ValueError, match="fps must match"):
        DirectLeRobotWriter(destination, schema=_SCHEMA, episode_count=2, fps=20)
    assert not destination.exists()


def test_lerobot_reader_rejects_schema_and_content_hash_changes(tmp_path: Path) -> None:
    destination = tmp_path / "lerobot"
    original = _make_corpus(destination)
    with pytest.raises(ValueError, match="schema mismatch"):
        load_lerobot_corpus(destination, expected_schema={})
    sidecar_path = destination / "meta/embodichain_ods_corpus.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["source_manifest"]["tensors"]["actions"]["sha256"] = "changed"
    sidecar_path.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="content/hash mismatch"):
        load_lerobot_corpus(destination, expected_schema=_SCHEMA)


@pytest.mark.parametrize("read_batch_size", [1, 2, 3, 1024])
def test_bulk_read_boundaries_preserve_holes_and_hashes(
    tmp_path: Path, read_batch_size: int
) -> None:
    destination = tmp_path / "lerobot"
    original = _make_corpus(destination)
    expected = original
    actual = load_lerobot_corpus(
        destination, expected_schema=_SCHEMA, read_batch_size=read_batch_size
    )
    assert actual.metadata["content_id"] == expected.metadata["content_id"]
    assert actual.metadata["tensors"] == expected.metadata["tensors"]
    assert actual.metadata["storage"]["numeric_read_mode"] == "sdk_hf_dataset_batch"
    for field in ("observations", "actions", "valid", "frame_ids", "generations"):
        assert torch.equal(
            actual.data["obs" if field == "observations" else field],
            expected.data["obs" if field == "observations" else field],
        )


def test_bulk_read_owns_cpu_device_even_with_non_cpu_default(tmp_path: Path) -> None:
    destination = tmp_path / "lerobot"
    original = _make_corpus(destination)
    with torch.device("meta"):
        actual = load_lerobot_corpus(destination, expected_schema=_SCHEMA)
    expected = original
    assert actual.metadata["tensors"] == expected.metadata["tensors"]
    for field in ("observations", "actions", "valid", "frame_ids", "generations"):
        assert (
            actual.data["obs" if field == "observations" else field].device.type
            == "cpu"
        )
        assert torch.equal(
            actual.data["obs" if field == "observations" else field],
            expected.data["obs" if field == "observations" else field],
        )


def test_scalar_fallback_only_for_absent_backing_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "lerobot"
    original = _make_corpus(destination)
    sdk_class = lerobot_module.LeRobotDataset

    class ScalarOnly:
        def __init__(self, *args, **kwargs):
            self.inner = sdk_class(*args, **kwargs)
            self.hf_dataset = None

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def __len__(self):
            return len(self.inner)

        def __getitem__(self, index):
            return self.inner[index]

    monkeypatch.setattr(lerobot_module, "LeRobotDataset", ScalarOnly)
    actual = load_lerobot_corpus(destination, expected_schema=_SCHEMA)
    assert actual.metadata["tensors"] == original.metadata["tensors"]
    assert actual.metadata["storage"]["numeric_read_mode"] == "sdk_scalar_fallback"
    assert actual.metadata["storage"]["scalar_fallback_reason"]


@pytest.mark.parametrize("corruption", ["episode", "frame", "dtype", "failure"])
def test_bulk_integrity_errors_do_not_fall_back_to_scalar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    destination = tmp_path / "lerobot"
    original = _make_corpus(destination)
    sdk_class = lerobot_module.LeRobotDataset

    class CorruptBacking:
        def __init__(self, original):
            self.original = original

        def __len__(self):
            return len(self.original)

        def __getitem__(self, index):
            if corruption == "failure":
                raise OSError("batch read failed")
            result = self.original[index]
            if corruption == "episode":
                result["episode_index"][0] += 1
            elif corruption == "frame":
                result["frame_index"][0] += 1
            else:
                result["action"] = [item.double() for item in result["action"]]
            return result

    class CorruptSDK:
        def __init__(self, *args, **kwargs):
            self.inner = sdk_class(*args, **kwargs)
            self.hf_dataset = CorruptBacking(self.inner.hf_dataset)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def __len__(self):
            return len(self.inner)

        def __getitem__(self, index):
            raise AssertionError("Integrity failures must not silently fall back")

    monkeypatch.setattr(lerobot_module, "LeRobotDataset", CorruptSDK)
    with pytest.raises((ValueError, OSError)):
        load_lerobot_corpus(destination, expected_schema=_SCHEMA)
