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

from examples.data_pipeline._ods_pick_place_benchmark import export_corpus, load_corpus
from examples.data_pipeline._ods_pick_place_lerobot import (
    export_lerobot_corpus,
    load_lerobot_corpus,
)

lerobot_module = pytest.importorskip("lerobot.datasets.lerobot_dataset")

_SCHEMA = {
    "observation": {"names": ["measured_joint", "cube_height"]},
    "action": {"names": ["arm_request", "gripper_request"], "step_dt": 1.0 / 30},
    "version": 1,
}


def _make_corpus(path: Path, *, generations: bool = True) -> None:
    # An empty padded row and a hole exercise preservation of source histories.
    valid = torch.tensor([[True, True, False], [False] * 3, [True, False, True]])
    observations = torch.arange(18, dtype=torch.float32).reshape(3, 3, 2)
    actions = -torch.arange(18, dtype=torch.float32).reshape(3, 3, 2) - 1
    ids = torch.zeros(3, 3, 3, dtype=torch.int64)
    for row in range(3):
        ids[row, :, 0] = 11
        ids[row, :, 1] = row + 5
        ids[row, :, 2] = torch.arange(3)
    export_corpus(
        path,
        observations,
        actions,
        valid,
        ids,
        schema=_SCHEMA,
        generations=(
            torch.tensor([[1] * 3, [2] * 3, [3] * 3]) if generations else None
        ),
        provenance={"teacher": "qualified-v1"},
    )


@pytest.mark.parametrize("generations", [True, False])
def test_official_lerobot_roundtrip_preserves_padded_numeric_corpus(
    tmp_path: Path, generations: bool
) -> None:
    source, destination = tmp_path / "source.pt", tmp_path / "lerobot"
    _make_corpus(source, generations=generations)
    export_lerobot_corpus(source, destination)
    original = load_corpus(source, expected_schema=_SCHEMA)
    restored = load_lerobot_corpus(destination, expected_schema=_SCHEMA)
    for field in ("observations", "actions", "valid", "frame_ids"):
        assert torch.equal(getattr(original, field), getattr(restored, field))
    if generations:
        assert torch.equal(original.generations, restored.generations)
    else:
        assert restored.generations is None
    assert restored.manifest["content_id"] == original.manifest["content_id"]
    assert restored.manifest["storage"]["per_minibatch_lerobot_io"] is False
    assert restored.manifest["storage"]["format"] == "lerobot"
    direct = lerobot_module.LeRobotDataset(
        "local/ods-pick-place", root=destination, download_videos=False
    )
    assert len(direct) == 4
    assert torch.equal(direct[3]["ods.provenance"][:3], torch.tensor([11, 7, 2]))
    selected = torch.tensor([[11, 5, 0], [11, 7, 2]])
    assert torch.equal(
        original.sample_frame_ids(selected)["actions"],
        restored.sample_frame_ids(selected)["actions"],
    )


def test_lerobot_export_refuses_existing_destination(tmp_path: Path) -> None:
    source, destination = tmp_path / "source.pt", tmp_path / "lerobot"
    _make_corpus(source)
    destination.mkdir()
    with pytest.raises(FileExistsError):
        export_lerobot_corpus(source, destination)


def test_lerobot_export_rejects_wrong_control_frequency(tmp_path: Path) -> None:
    source, destination = tmp_path / "source.pt", tmp_path / "lerobot"
    _make_corpus(source)
    with pytest.raises(ValueError, match="fps must match"):
        export_lerobot_corpus(source, destination, fps=20)
    assert not destination.exists()


def test_lerobot_reader_rejects_schema_and_content_hash_changes(tmp_path: Path) -> None:
    source, destination = tmp_path / "source.pt", tmp_path / "lerobot"
    _make_corpus(source)
    export_lerobot_corpus(source, destination)
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
    source, destination = tmp_path / "source.pt", tmp_path / "lerobot"
    _make_corpus(source)
    export_lerobot_corpus(source, destination)
    expected = load_corpus(source, expected_schema=_SCHEMA)
    actual = load_lerobot_corpus(
        destination, expected_schema=_SCHEMA, read_batch_size=read_batch_size
    )
    assert actual.manifest["content_id"] == expected.manifest["content_id"]
    assert actual.manifest["tensors"] == expected.manifest["tensors"]
    assert actual.manifest["storage"]["numeric_read_mode"] == "sdk_hf_dataset_batch"
    for field in ("observations", "actions", "valid", "frame_ids", "generations"):
        assert torch.equal(getattr(actual, field), getattr(expected, field))


def test_bulk_read_owns_cpu_device_even_with_non_cpu_default(tmp_path: Path) -> None:
    source, destination = tmp_path / "source.pt", tmp_path / "lerobot"
    _make_corpus(source)
    export_lerobot_corpus(source, destination)
    with torch.device("meta"):
        actual = load_lerobot_corpus(destination, expected_schema=_SCHEMA)
    expected = load_corpus(source, expected_schema=_SCHEMA)
    assert actual.manifest["tensors"] == expected.manifest["tensors"]
    for field in ("observations", "actions", "valid", "frame_ids", "generations"):
        assert getattr(actual, field).device.type == "cpu"
        assert torch.equal(getattr(actual, field), getattr(expected, field))


def test_scalar_fallback_only_for_absent_backing_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, destination = tmp_path / "source.pt", tmp_path / "lerobot"
    _make_corpus(source)
    export_lerobot_corpus(source, destination)
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
    assert (
        actual.manifest["tensors"]
        == load_corpus(source, expected_schema=_SCHEMA).manifest["tensors"]
    )
    assert actual.manifest["storage"]["numeric_read_mode"] == "sdk_scalar_fallback"
    assert actual.manifest["storage"]["scalar_fallback_reason"]


@pytest.mark.parametrize("corruption", ["episode", "frame", "dtype", "failure"])
def test_bulk_integrity_errors_do_not_fall_back_to_scalar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    source, destination = tmp_path / "source.pt", tmp_path / "lerobot"
    _make_corpus(source)
    export_lerobot_corpus(source, destination)
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
