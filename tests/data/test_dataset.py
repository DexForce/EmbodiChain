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

import hashlib
import multiprocessing as mp
import time
from types import SimpleNamespace
from pathlib import Path
from zipfile import ZipFile

import pytest

from embodichain.data import constants
from embodichain.data import dataset as dataset_module
from embodichain.data.dataset import _dataset_download_lock

LOCK_HOLD_SECONDS = 0.2
PROCESS_JOIN_TIMEOUT_SECONDS = 5.0
PROCESS_START_TIMEOUT_SECONDS = 60.0


@pytest.mark.no_sim
@pytest.mark.parametrize(
    "urls, expected",
    [
        (
            [
                "https://hf-mirror.com/datasets/DexForceAI/embodichain_data/resolve/v1/model.zip"
            ],
            [
                "https://hf-mirror.com/datasets/DexForceAI/embodichain_data/resolve/v1/model.zip",
                "https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/v1/model.zip",
            ],
        ),
        (
            ["https://assets.example/model.zip"],
            ["https://assets.example/model.zip"],
        ),
        (
            ["https://hf-mirror.com/datasets/Other/data/resolve/main/model.zip"],
            ["https://hf-mirror.com/datasets/Other/data/resolve/main/model.zip"],
        ),
        (
            [
                "https://hf-mirror.com/datasets/DexForceAI/embodichain_data/resolve/main/model.zip",
                "https://assets.example/model.zip",
                "https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/main/model.zip",
            ],
            [
                "https://hf-mirror.com/datasets/DexForceAI/embodichain_data/resolve/main/model.zip",
                "https://assets.example/model.zip",
                "https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/main/model.zip",
            ],
        ),
    ],
)
def test_asset_origin_fallback_preserves_sources_and_metadata(urls, expected) -> None:
    """Only the known asset mirror gains an origin, without changing its revision."""
    descriptor = dataset_module.o3d.data.DataDescriptor(urls, "a" * 32, "mounted")
    result = dataset_module._with_asset_origin_fallback(descriptor)

    assert result.urls == expected
    assert result.md5 == "a" * 32
    assert result.extract_in_subdir == "mounted"
    assert descriptor.urls == urls


@pytest.mark.no_sim
@pytest.mark.parametrize(
    "mirror_content, origin_valid",
    [(None, True), (b"corrupt", True), (b"corrupt", False)],
)
def test_dataset_download_falls_back_with_integrity_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mirror_content, origin_valid: bool
) -> None:
    """The real downloader recovers missing/corrupt mirrors and rejects bad origins."""
    mirror = tmp_path / "mirror"
    origin = tmp_path / "origin"
    mirror.mkdir()
    origin.mkdir()
    archive = origin / "model.zip"
    with ZipFile(archive, "w") as bundle:
        bundle.writestr("model.urdf", "verified model")
    expected_md5 = hashlib.md5(archive.read_bytes()).hexdigest()
    if mirror_content is not None:
        (mirror / "model.zip").write_bytes(mirror_content)
    if not origin_valid:
        archive.write_bytes(b"invalid origin")
    monkeypatch.setattr(dataset_module, "_HF_ASSET_MIRROR", mirror.as_uri() + "/")
    monkeypatch.setattr(dataset_module, "_HF_ASSET_ORIGIN", origin.as_uri() + "/")
    descriptor = dataset_module.o3d.data.DataDescriptor(
        (mirror / "model.zip").as_uri(), expected_md5, "mounted"
    )
    cache = str(tmp_path / "cache")

    if not origin_valid:
        with pytest.raises(RuntimeError, match="Downloading failed"):
            dataset_module.EmbodiChainDataset("Model", descriptor, cache)
        return

    dataset = dataset_module.EmbodiChainDataset("Model", descriptor, cache)
    model = Path(dataset.extract_dir) / "mounted/model.urdf"
    assert model.read_text() == "verified model"
    archive.unlink()
    cached = dataset_module.EmbodiChainDataset("Model", descriptor, cache)
    assert (
        Path(cached.extract_dir) / "mounted/model.urdf"
    ).read_text() == "verified model"


@pytest.mark.no_sim
def test_default_data_modules_remains_a_public_export() -> None:
    from embodichain.data import DEFAULT_DATA_MODULES

    assert DEFAULT_DATA_MODULES is dataset_module.DEFAULT_DATA_MODULES


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


@pytest.mark.no_sim
def test_registered_file_reference_uses_verified_asset_root(
    tmp_path, monkeypatch
) -> None:
    from types import SimpleNamespace
    from embodichain.data._references import _resolve_data_reference

    root = tmp_path / "bundle"
    root.mkdir()
    (root / "trajectory.json").write_text("verified")
    calls = []

    def asset_type(name):
        calls.append(name)
        return lambda: SimpleNamespace(extract_dir=str(root))

    monkeypatch.setattr("embodichain.data.get_data_class", asset_type)
    resolved = _resolve_data_reference(
        {"asset": "DemoInputs", "path": "trajectory.json"}, base_dir=tmp_path / "owner"
    )
    assert resolved == root / "trajectory.json"
    assert calls == ["DemoInputs"]
    assert (
        _resolve_data_reference("source.json", base_dir=tmp_path)
        == tmp_path / "source.json"
    )


@pytest.mark.no_sim
@pytest.mark.parametrize(
    "reference",
    [
        {"asset": "DemoInputs", "path": "../secret"},
        {"asset": "DemoInputs", "path": "/secret"},
        {"asset": "DemoInputs", "path": "a\\secret"},
        {"asset": "DemoInputs", "path": "a.json", "import": "arbitrary"},
        {"asset": "https://example.test", "path": "a.json"},
    ],
)
def test_asset_file_reference_rejects_invalid_input_before_download(
    tmp_path, monkeypatch, reference
) -> None:
    from embodichain.data._references import _resolve_data_reference

    def forbidden(*args):
        raise AssertionError("invalid reference reached asset initialization")

    monkeypatch.setattr("embodichain.data.get_data_class", forbidden)
    with pytest.raises(ValueError):
        _resolve_data_reference(reference, base_dir=tmp_path)


@pytest.mark.no_sim
def test_asset_file_reference_rejects_symlink_escape(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace
    from embodichain.data._references import _resolve_data_reference

    root = tmp_path / "bundle"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("outside")
    (root / "escape.json").symlink_to(outside)
    monkeypatch.setattr(
        "embodichain.data.get_data_class",
        lambda _: lambda: SimpleNamespace(extract_dir=str(root)),
    )
    with pytest.raises(ValueError, match="escapes"):
        _resolve_data_reference(
            {"asset": "DemoInputs", "path": "escape.json"}, base_dir=tmp_path
        )
