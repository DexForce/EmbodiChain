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

import multiprocessing as mp
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from embodichain.data_pipeline.engine.data import (
    OnlineDataEngine,
    OnlineDataEngineCfg,
    _populate_language_indices,
)
from embodichain.data_pipeline.engine.language import SharedLanguageRegistry


def _registry_consumer(
    registry: SharedLanguageRegistry, ready: Event, release: Event, output: Queue
) -> None:
    try:
        before = registry.resolve(torch.tensor([0]))
        added = registry.intern("Child task")
        ready.set()
        if not release.wait(10):
            raise TimeoutError("Parent did not release language consumer")
        after = registry.resolve(torch.tensor([[0, added, 2]]))
        output.put((before, after))
    except BaseException as error:
        ready.set()
        output.put((type(error).__name__, str(error)))


def _buffer(lengths: tuple[int, ...], capacity: int = 5) -> TensorDict:
    rows = len(lengths)
    valid = torch.arange(capacity)[None, :] < torch.tensor(lengths)[:, None]
    return TensorDict(
        {
            "task_index": torch.full((rows, capacity), -1, dtype=torch.int64),
            "subtask_index": torch.full((rows, capacity), -1, dtype=torch.int64),
            "valid": valid,
        },
        batch_size=[rows, capacity],
    )


def _segment(
    start: int, end: int, instruction: str | None, **kwargs: object
) -> SimpleNamespace:
    return SimpleNamespace(
        start_step=start, end_step=end, instruction=instruction, **kwargs
    )


def _result(
    instruction: str | None, lengths: tuple[int, ...], *segments: SimpleNamespace
) -> SimpleNamespace:
    return SimpleNamespace(
        instruction=instruction, lengths=lengths, length=max(lengths), segments=segments
    )


def test_shared_registry_preserves_utf8_and_separate_language_levels() -> None:
    registry = SharedLanguageRegistry()
    task = registry.intern("把方块放到目标\n位置")
    subtask = registry.intern("抓起方块", kind="subtask")
    assert task == subtask == 0  # Independent task/subtask dictionaries.
    assert registry.intern("把方块放到目标\n位置") == task
    assert registry.resolve(torch.tensor([[task, task]])) == [
        ["把方块放到目标\n位置"] * 2
    ]
    assert registry.resolve(torch.tensor(subtask), kind="subtask") == "抓起方块"


def test_shared_registry_entries_are_visible_across_forkserver_processes() -> None:
    context = mp.get_context("forkserver")
    registry = SharedLanguageRegistry(context=context)
    assert registry.intern("Parent task") == 0
    ready, release, output = context.Event(), context.Event(), context.Queue()
    consumer = context.Process(
        target=_registry_consumer, args=(registry, ready, release, output)
    )
    consumer.start()
    try:
        assert ready.wait(20)
        assert registry.resolve(torch.tensor([1])) == ["Child task"]
        assert registry.intern("Later task") == 2
        release.set()
        assert output.get(timeout=20) == (
            ["Parent task"],
            [["Parent task", "Child task", "Later task"]],
        )
        consumer.join(timeout=5)
        assert consumer.exitcode == 0
    finally:
        release.set()
        if consumer.is_alive():
            consumer.terminate()
            consumer.join(timeout=5)
        output.close()
        output.join_thread()


def test_registry_overflow_keeps_all_previous_indices_immutable() -> None:
    registry = SharedLanguageRegistry(capacity_bytes=96)
    first = registry.intern("Original task")
    with pytest.raises(OverflowError, match="language_buffer_bytes=96"):
        registry.intern("Unique description " * 20)
    assert registry.resolve(torch.tensor([first])) == ["Original task"]
    assert registry.intern("Original task") == first


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_engine_rejects_invalid_registry_capacity_before_environment_creation(
    capacity: object,
) -> None:
    with pytest.raises(ValueError, match="language_buffer_bytes"):
        OnlineDataEngine(OnlineDataEngineCfg(language_buffer_bytes=capacity))


@pytest.mark.parametrize(
    "indices", [torch.tensor([-1]), torch.tensor([0]), torch.tensor([1.0])]
)
def test_registry_rejects_unknown_or_nonnumeric_indices(indices: torch.Tensor) -> None:
    registry = SharedLanguageRegistry()
    with pytest.raises((ValueError, TypeError)):
        registry.resolve(indices)


def test_language_mapping_uses_row_specific_half_open_spans_and_episode_fallback() -> (
    None
):
    registry = SharedLanguageRegistry()
    buffer = _buffer((5, 3))
    result = _result(
        "Move the block",
        (5, 3),
        _segment(0, 2, "Pick the block", start_steps=(0, 0), end_steps=(2, 1)),
        _segment(2, 5, "Place the block", start_steps=(2, 1), end_steps=(5, 3)),
    )
    _populate_language_indices(buffer, result, registry)
    assert registry.resolve(buffer["task_index"][0]) == ["Move the block"] * 5
    assert (
        registry.resolve(buffer["subtask_index"][0], kind="subtask")
        == ["Pick the block"] * 2 + ["Place the block"] * 3
    )
    assert registry.resolve(buffer["subtask_index"][1, :3], kind="subtask") == [
        "Pick the block",
        "Place the block",
        "Place the block",
    ]
    assert (buffer["task_index"][1, 3:] == -1).all()
    assert (buffer["subtask_index"][1, 3:] == -1).all()


def test_missing_segment_text_and_uncovered_frames_inherit_overall_task() -> None:
    registry = SharedLanguageRegistry()
    buffer = _buffer((5,))
    _populate_language_indices(
        buffer, _result("Overall", (5,), _segment(1, 3, None)), registry
    )
    assert (
        registry.resolve(buffer["subtask_index"][0], kind="subtask") == ["Overall"] * 5
    )
    _populate_language_indices(buffer, _result(None, (5,)), registry)
    assert registry.resolve(buffer["task_index"][0]) == ["unknown_task"] * 5


def test_inactive_segments_do_not_annotate_other_rows_and_empty_segments_do_not_allocate_labels() -> (
    None
):
    registry = SharedLanguageRegistry(capacity_bytes=128)
    buffer = _buffer((3, 3))
    result = _result(
        "Overall",
        (3, 3),
        _segment(0, 2, "Row one", active=(True, False)),
        _segment(3, 3, "Unused " * 100),
    )
    _populate_language_indices(buffer, result, registry)
    assert (
        registry.resolve(buffer["subtask_index"][1, :3], kind="subtask")
        == ["Overall"] * 3
    )
    assert registry.resolve(buffer["subtask_index"][0, :3], kind="subtask") == [
        "Row one",
        "Row one",
        "Overall",
    ]


def test_cloned_language_indices_survive_ring_slot_reuse() -> None:
    registry = SharedLanguageRegistry()
    buffer = _buffer((5,))
    first = _result("Original", (5,), _segment(0, 5, "Original subtask"))
    _populate_language_indices(buffer, first, registry)
    snapshot = buffer.clone()
    _populate_language_indices(
        buffer,
        _result("Replacement", (5,), _segment(0, 5, "Replacement subtask")),
        registry,
    )
    assert registry.resolve(snapshot["task_index"][0]) == ["Original"] * 5
    assert (
        registry.resolve(snapshot["subtask_index"][0], kind="subtask")
        == ["Original subtask"] * 5
    )
    assert registry.resolve(buffer["task_index"][0]) == ["Replacement"] * 5


@pytest.mark.parametrize(
    "segments",
    [(_segment(0, 4, "A"), _segment(3, 5, "B")), (_segment(0, 6, "Outside"),)],
)
def test_language_mapping_rejects_ambiguous_or_out_of_bounds_spans(
    segments: tuple[SimpleNamespace, ...],
) -> None:
    with pytest.raises(ValueError, match="overlap|outside"):
        _populate_language_indices(
            _buffer((5,)), _result("Overall", (5,), *segments), SharedLanguageRegistry()
        )
