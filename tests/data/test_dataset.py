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

"""Regression tests for data-path resolution and cache synchronization."""

from __future__ import annotations

import multiprocessing as mp
import time
from types import SimpleNamespace
from pathlib import Path

import pytest

from embodichain.data import constants
from embodichain.data import dataset as dataset_module
from embodichain.data.dataset import _dataset_download_lock

LOCK_HOLD_SECONDS = 0.2
PROCESS_JOIN_TIMEOUT_SECONDS = 5.0
PROCESS_START_TIMEOUT_SECONDS = 60.0


def _check_cache(
    root: Path, prefix: str = "TestAsset", expected: str = "valid"
) -> None:
    """Run cache preparation without starting a network download."""
    descriptor = SimpleNamespace(
        urls=["https://example.invalid/model.zip"], md5=expected
    )
    dataset = SimpleNamespace(calculate_md5=lambda path: "valid")
    dataset_module.EmbodiChainDataset.check_zip(dataset, prefix, descriptor, str(root))


@pytest.mark.no_sim
@pytest.mark.parametrize("has_zip", [False, True])
def test_custom_cache_removes_stale_extraction(tmp_path: Path, has_zip: bool) -> None:
    """An orphaned or invalid download clears only its own extraction."""
    root = tmp_path / "asset-cache"
    extracted = root / "extract/TestAsset"
    extracted.mkdir(parents=True)
    (extracted / "stale.txt").write_text("stale")
    neighbor = root / "extract/AnotherAsset"
    neighbor.mkdir()
    (neighbor / "keep.txt").write_text("keep")
    download = root / "download/TestAsset/model.zip"
    if has_zip:
        download.parent.mkdir(parents=True)
        download.write_bytes(b"invalid")

    _check_cache(root, expected="different")

    assert not extracted.exists()
    assert not download.exists()
    assert (neighbor / "keep.txt").read_text() == "keep"


@pytest.mark.no_sim
def test_custom_cache_preserves_valid_download(tmp_path: Path) -> None:
    """A valid archive with extracted contents is reused."""
    root = tmp_path / "asset-cache"
    download = root / "download/TestAsset/model.zip"
    download.parent.mkdir(parents=True)
    download.write_bytes(b"archive")
    extracted = root / "extract/TestAsset"
    extracted.mkdir(parents=True)
    (extracted / "model.xml").write_text("model")

    _check_cache(root)

    assert download.read_bytes() == b"archive"
    assert (extracted / "model.xml").read_text() == "model"


@pytest.mark.no_sim
def test_cache_rejects_extraction_symlink_outside_root(tmp_path: Path) -> None:
    """Invalid extraction paths fail before an archive or outside data is removed."""
    root = tmp_path / "asset-cache"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    (root / "extract").mkdir(parents=True)
    (root / "extract/TestAsset").symlink_to(outside, target_is_directory=True)
    download = root / "download/TestAsset/model.zip"
    download.parent.mkdir(parents=True)
    download.write_bytes(b"archive")

    with pytest.raises(ValueError, match="cache"):
        _check_cache(root, expected="different")

    assert download.read_bytes() == b"archive"
    assert (outside / "keep.txt").read_text() == "keep"


@pytest.mark.no_sim
@pytest.mark.parametrize(
    "prefix",
    ["", ".", "..", "../outside", "/absolute", "nested/name", "nested\\name", ".locks"],
)
def test_cache_rejects_non_entry_prefix(tmp_path: Path, prefix: str) -> None:
    """A dataset owns one cache entry, never a parent or reserved lock directory."""
    with pytest.raises(ValueError, match="prefix"):
        _check_cache(tmp_path, prefix)


@pytest.mark.no_sim
def test_get_data_path_accepts_dataset_root_without_subpath(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare dataset name resolves to the downloaded extraction root."""
    extract_dir = tmp_path / "extract" / "SceneData"

    class FakeDataset:
        def __init__(self) -> None:
            self.extract_dir = str(extract_dir)

    def get_fake_data_class(dataset_name: str) -> type[FakeDataset]:
        assert dataset_name == "SceneData"
        return FakeDataset

    monkeypatch.setattr(constants, "EMBODICHAIN_DEFAULT_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(dataset_module, "get_data_class", get_fake_data_class)

    assert dataset_module.get_data_path("SceneData") == str(extract_dir)


def _hold_dataset_download_lock(
    data_root: str,
    start_event,
    ready_event,
    active_initializers,
    maximum_active_initializers,
) -> None:
    """Enter the same dataset lock and hold it long enough to detect overlap."""
    ready_event.set()
    if not start_event.wait(PROCESS_START_TIMEOUT_SECONDS):
        raise TimeoutError("Dataset lock test start signal was not received.")
    with _dataset_download_lock(data_root, "ConcurrentDataset"):
        with active_initializers.get_lock():
            active_initializers.value += 1
            with maximum_active_initializers.get_lock():
                maximum_active_initializers.value = max(
                    maximum_active_initializers.value,
                    active_initializers.value,
                )
        try:
            time.sleep(LOCK_HOLD_SECONDS)
        finally:
            with active_initializers.get_lock():
                active_initializers.value -= 1


@pytest.mark.no_sim
def test_dataset_initialization_is_serialized_across_processes(tmp_path) -> None:
    """One worker completes cache preparation before another enters it."""
    context = mp.get_context("spawn")
    active_initializers = context.Value("i", 0)
    maximum_active_initializers = context.Value("i", 0)
    start_event = context.Event()
    ready_events = [context.Event() for _ in range(2)]

    workers = [
        context.Process(
            target=_hold_dataset_download_lock,
            args=(
                str(tmp_path),
                start_event,
                ready_event,
                active_initializers,
                maximum_active_initializers,
            ),
        )
        for ready_event in ready_events
    ]
    started_workers = []
    try:
        for worker in workers:
            worker.start()
            started_workers.append(worker)
        # Spawn imports native dependencies before entering the worker. Give
        # startup its own budget, then exercise contention with both ready.
        deadline = time.monotonic() + PROCESS_START_TIMEOUT_SECONDS
        for ready_event in ready_events:
            assert ready_event.wait(
                max(0.0, deadline - time.monotonic())
            ), "Dataset lock worker did not finish startup."
        start_event.set()
        for worker in workers:
            worker.join(PROCESS_JOIN_TIMEOUT_SECONDS)
    finally:
        start_event.set()
        for worker in started_workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(PROCESS_JOIN_TIMEOUT_SECONDS)

    assert [worker.exitcode for worker in workers] == [0, 0]
    assert maximum_active_initializers.value == 1
