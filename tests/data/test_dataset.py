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

"""Regression tests for data-cache synchronization."""

from __future__ import annotations

import multiprocessing as mp
import time

import pytest

from embodichain.data.dataset import _dataset_download_lock

LOCK_HOLD_SECONDS = 0.2
PROCESS_JOIN_TIMEOUT_SECONDS = 5.0
PROCESS_START_TIMEOUT_SECONDS = 60.0


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
