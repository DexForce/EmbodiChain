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
import sys
import threading
import time
import types
from pathlib import Path

import pytest
import torch

from examples.ods.simple_bc_demo.data import _content_manifest
from examples.ods.simple_bc_demo.lerobot import (
    DirectLeRobotWriter,
    load_lerobot_corpus,
)

_SCHEMA = {
    "observation": {"names": ["a", "b"]},
    "action": {"names": ["a", "b"], "step_dt": 1 / 30},
}


def _row(index: int = 0) -> dict[str, torch.Tensor]:
    return {
        "observations": torch.arange(8, dtype=torch.float32).reshape(4, 2) + index,
        "actions": torch.arange(8, dtype=torch.float32).reshape(4, 2),
        "valid": torch.tensor([True, False, True, False]),
        "frame_ids": torch.tensor([[17, index, step] for step in range(4)]),
        "generations": torch.ones(4, dtype=torch.int64),
    }


@pytest.fixture
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> type:
    class Dataset:
        calls: list[tuple[str, int]] = []
        rows: list[dict] = []
        gate: threading.Event | None = None
        entered = threading.Event()
        failure = False
        cleanup_failure = False

        @classmethod
        def create(cls, **kwargs):
            cls.calls.append(("create", threading.get_ident()))
            (kwargs["root"] / "meta").mkdir(parents=True)
            cls.entered.set()
            if cls.gate is not None:
                assert cls.gate.wait(5)
            return cls()

        def add_frame(self, frame):
            self.calls.append(("add", threading.get_ident()))
            if self.failure:
                raise OSError("disk write failed")
            self.rows.append(frame)

        def save_episode(self):
            self.calls.append(("save", threading.get_ident()))

        def finalize(self):
            self.calls.append(("finalize", threading.get_ident()))
            if self.cleanup_failure:
                raise OSError("cleanup failed")

    module = types.ModuleType("lerobot.datasets.lerobot_dataset")
    module.LeRobotDataset = Dataset
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return Dataset


def test_streaming_hashes_match_full_corpus_and_sdk_single_thread(tmp_path, fake_sdk):
    rows = [_row(0), _row(1)]
    writer = DirectLeRobotWriter(
        tmp_path / "dataset", schema=_SCHEMA, episode_count=2, max_episode_steps=4
    )
    with writer:
        for row in rows:
            writer.submit(row)
        sidecar = writer.finalize()
    tensors = {key: torch.stack([row[key] for row in rows]) for key in rows[0]}
    for key in tensors.keys() - {"valid"}:
        tensors[key][~tensors["valid"]] = 0
    expected = _content_manifest(tensors, _SCHEMA)
    assert all(
        sidecar["source_manifest"][key] == value for key, value in expected.items()
    )
    assert sidecar["prepublication_readback"] is False
    assert len(fake_sdk.rows) == 4
    threads = {thread for _, thread in fake_sdk.calls}
    assert len(threads) == 1 and threading.get_ident() not in threads
    assert writer.stats["accepted_rows"] == 2
    assert writer.stats["worker_joined"] and writer.stats["published"]
    assert not list(tmp_path.glob("*.pt"))


def test_bounded_queue_clones_before_handoff_and_polls_progress(tmp_path, fake_sdk):
    fake_sdk.gate = threading.Event()
    writer = DirectLeRobotWriter(
        tmp_path / "dataset",
        schema=_SCHEMA,
        episode_count=2,
        max_episode_steps=4,
        queue_size=1,
    )
    calls = []
    try:
        assert fake_sdk.entered.wait(2)
        original = _row(0)
        expected = original["observations"][0].clone()
        writer.submit(original)
        original["observations"].fill_(999)

        def progress():
            calls.append(True)
            fake_sdk.gate.set()

        writer.submit(_row(1), progress=progress)
        writer.finalize()
        assert calls
        assert torch.equal(fake_sdk.rows[0]["observation.state"], expected)
        assert writer.stats["max_queued_rows"] <= 1
        assert writer.stats["queue_wait_seconds"] >= 0.04
    finally:
        fake_sdk.gate.set()
        writer.abort()


def test_failed_writer_unblocks_submit_preserves_original_error(tmp_path, fake_sdk):
    fake_sdk.failure = True
    fake_sdk.cleanup_failure = True
    writer = DirectLeRobotWriter(
        tmp_path / "dataset",
        schema=_SCHEMA,
        episode_count=5,
        max_episode_steps=4,
        queue_size=1,
    )
    try:
        with pytest.raises(RuntimeError, match="writer failed") as error:
            for i in range(5):
                writer.submit(_row(i))
            writer.finalize()
        assert isinstance(error.value.__cause__, OSError)
        assert str(error.value.__cause__) == "disk write failed"
        with pytest.raises(RuntimeError, match="writer failed"):
            writer.submit(_row(6))
    finally:
        writer.abort()
    assert not (tmp_path / "dataset").exists()
    assert (
        json.loads((writer.temporary / "FAILED.json").read_text())["error"]
        == "disk write failed"
    )
    assert "cleanup failed" in writer.stats["cleanup_error"]
    assert not writer._thread.is_alive()


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda row: row.pop("generations"), "five numeric"),
        (lambda row: row.update(actions=row["actions"].double()), "float32"),
        (lambda row: row["frame_ids"].__setitem__((2, 2), 3), "identities"),
        (lambda row: row["generations"].__setitem__(2, 2), "one generation"),
        (lambda row: row["observations"].__setitem__((0, 0), float("nan")), "finite"),
    ],
)
def test_rejects_bad_row_before_acceptance(tmp_path, fake_sdk, mutation, match):
    with DirectLeRobotWriter(
        tmp_path / "dataset", schema=_SCHEMA, episode_count=1, max_episode_steps=4
    ) as writer:
        row = _row()
        mutation(row)
        with pytest.raises(ValueError, match=match):
            writer.submit(row)
        assert writer.stats["submitted_rows"] == 0
    assert not (tmp_path / "dataset").exists()


def test_counts_identity_and_creator_thread_are_enforced(tmp_path, fake_sdk):
    with DirectLeRobotWriter(
        tmp_path / "dataset", schema=_SCHEMA, episode_count=2, max_episode_steps=4
    ) as writer:
        writer.submit(_row())
        with pytest.raises(ValueError, match="Duplicate"):
            writer.submit(_row())
        with pytest.raises(ValueError, match="row count"):
            writer.finalize()
        errors = []

        def foreign():
            try:
                writer.submit(_row(1))
            except RuntimeError as error:
                errors.append(str(error))

        thread = threading.Thread(target=foreign)
        thread.start()
        thread.join()
        assert errors and "creator thread" in errors[0]
        writer.submit(_row(1))
        with pytest.raises(ValueError, match="More rows"):
            writer.submit(_row(2))
        writer.finalize()
        with pytest.raises(RuntimeError, match="closed"):
            writer.submit(_row(2))


def test_existing_destination_preserved_before_or_during_write(tmp_path, fake_sdk):
    destination = tmp_path / "dataset"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        DirectLeRobotWriter(destination, schema=_SCHEMA, episode_count=1)
    destination.rmdir()
    with DirectLeRobotWriter(
        destination, schema=_SCHEMA, episode_count=1, max_episode_steps=4
    ) as writer:
        writer.submit(_row())
        destination.mkdir()
        (destination / "owned").write_text("existing")
        with pytest.raises(RuntimeError) as error:
            writer.finalize()
        assert isinstance(error.value.__cause__, FileExistsError)
    assert (destination / "owned").read_text() == "existing"


def test_progress_exception_aborts_without_masking_primary(tmp_path, fake_sdk):
    fake_sdk.gate = threading.Event()
    writer = DirectLeRobotWriter(
        tmp_path / "dataset",
        schema=_SCHEMA,
        episode_count=2,
        max_episode_steps=4,
        queue_size=1,
    )

    def progress():
        fake_sdk.gate.set()
        raise LookupError("producer failed")

    with pytest.raises(LookupError, match="producer failed"):
        with writer:
            writer.submit(_row())
            writer.submit(_row(1), progress=progress)
    assert not writer._thread.is_alive()
    assert not (tmp_path / "dataset").exists()


def test_official_sdk_direct_roundtrip(tmp_path):
    pytest.importorskip("lerobot.datasets.lerobot_dataset")
    with DirectLeRobotWriter(
        tmp_path / "dataset", schema=_SCHEMA, episode_count=2, max_episode_steps=4
    ) as writer:
        writer.submit(_row(0))
        writer.submit(_row(1))
        sidecar = writer.finalize()
    loaded = load_lerobot_corpus(tmp_path / "dataset", expected_schema=_SCHEMA)
    assert loaded.manifest["content_id"] == sidecar["source_manifest"]["content_id"]
    assert torch.equal(loaded.valid, torch.stack([_row()["valid"]] * 2))
    assert loaded.observations[0, 1].count_nonzero() == 0
    assert torch.equal(loaded.observations[1, 2], _row(1)["observations"][2])


def test_finalize_progress_failure_never_publishes(tmp_path, fake_sdk):
    def slow_finalize(self):
        time.sleep(0.15)

    fake_sdk.finalize = slow_finalize
    writer = DirectLeRobotWriter(
        tmp_path / "dataset", schema=_SCHEMA, episode_count=1, max_episode_steps=4
    )

    def progress():
        raise LookupError("producer fault during finalize")

    with pytest.raises(LookupError, match="producer fault"):
        with writer:
            writer.submit(_row())
            writer.finalize(progress=progress)
    assert not writer._thread.is_alive()
    assert not (tmp_path / "dataset").exists()
    assert (writer.temporary / "FAILED.json").exists()


def test_atomic_publication_refuses_existing_empty_directory(tmp_path):
    from examples.ods.simple_bc_demo.lerobot import (
        _publish_without_overwrite,
    )

    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    with pytest.raises(FileExistsError):
        _publish_without_overwrite(source, destination)
    assert source.is_dir() and destination.is_dir()


@pytest.mark.parametrize("name", ["operation_timeout", "shutdown_timeout"])
@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True])
def test_timeouts_must_be_finite_positive(tmp_path, name, value):
    with pytest.raises(ValueError, match="finite and positive"):
        DirectLeRobotWriter(
            tmp_path / "dataset", schema=_SCHEMA, episode_count=1, **{name: value}
        )
    assert not list(tmp_path.iterdir())


def test_blocked_sdk_bounds_queue_and_context_shutdown(tmp_path, fake_sdk):
    fake_sdk.gate = threading.Event()
    writer = DirectLeRobotWriter(
        tmp_path / "dataset",
        schema=_SCHEMA,
        episode_count=2,
        max_episode_steps=4,
        queue_size=1,
        operation_timeout=0.06,
        shutdown_timeout=0.04,
    )
    try:
        assert fake_sdk.entered.wait(2)
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="queue handoff") as error:
            with writer:
                writer.submit(_row())
                writer.submit(_row(1))
        assert time.monotonic() - started < 0.5
        assert writer._thread.daemon and writer._thread.is_alive()
        assert not writer.stats["worker_joined"]
        assert "TimeoutError" in writer.stats["cleanup_error"]
        assert any("worker_joined=False" in note for note in error.value.__notes__)
        assert not writer.stats["published"]
        assert not (tmp_path / "dataset").exists()
        with pytest.raises(RuntimeError, match="writer failed") as sticky:
            writer.submit(_row(1))
        assert sticky.value.__cause__ is error.value
    finally:
        fake_sdk.gate.set()
        writer._thread.join(timeout=2)
        writer.abort()
    assert not (tmp_path / "dataset").exists()
    assert (
        "queue handoff"
        in json.loads((writer.temporary / "FAILED.json").read_text())["error"]
    )


def test_blocked_sdk_finalize_has_deadline_and_cannot_publish_later(tmp_path, fake_sdk):
    finalize_gate = threading.Event()
    entered = threading.Event()

    def blocked_finalize(self):
        entered.set()
        assert finalize_gate.wait(5)

    fake_sdk.finalize = blocked_finalize
    writer = DirectLeRobotWriter(
        tmp_path / "dataset",
        schema=_SCHEMA,
        episode_count=1,
        max_episode_steps=4,
        operation_timeout=0.06,
        shutdown_timeout=0.04,
    )
    try:
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="did not stop"):
            with writer:
                writer.submit(_row())
                writer.finalize()
        assert entered.is_set()
        assert time.monotonic() - started < 0.5
        assert not writer.stats["worker_joined"] and not writer.stats["published"]
        assert "cleanup_error" in writer.stats
        assert not (tmp_path / "dataset").exists()
    finally:
        finalize_gate.set()
        writer._thread.join(timeout=2)
        writer.abort()
    assert not (tmp_path / "dataset").exists()


def test_primary_error_survives_unjoined_cleanup_with_visible_note(tmp_path, fake_sdk):
    fake_sdk.gate = threading.Event()
    writer = DirectLeRobotWriter(
        tmp_path / "dataset",
        schema=_SCHEMA,
        episode_count=1,
        max_episode_steps=4,
        shutdown_timeout=0.04,
    )
    try:
        assert fake_sdk.entered.wait(2)
        started = time.monotonic()
        with pytest.raises(LookupError, match="original producer failure") as error:
            with writer:
                raise LookupError("original producer failure")
        assert time.monotonic() - started < 0.5
        assert any("cleanup failed" in note for note in error.value.__notes__)
        assert not writer.stats["worker_joined"] and writer.stats["cleanup_error"]
        assert not (tmp_path / "dataset").exists()
    finally:
        fake_sdk.gate.set()
        writer._thread.join(timeout=2)
        writer.abort()


def test_explicit_abort_raises_when_sdk_remains_blocked(tmp_path, fake_sdk):
    fake_sdk.gate = threading.Event()
    writer = DirectLeRobotWriter(
        tmp_path / "dataset",
        schema=_SCHEMA,
        episode_count=1,
        max_episode_steps=4,
        shutdown_timeout=0.04,
    )
    try:
        assert fake_sdk.entered.wait(2)
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="did not stop"):
            writer.abort()
        assert time.monotonic() - started < 0.5
        assert not writer.stats["worker_joined"]
        assert writer.stats["cleanup_error"]
    finally:
        fake_sdk.gate.set()
        writer._thread.join(timeout=2)
        writer.abort()
