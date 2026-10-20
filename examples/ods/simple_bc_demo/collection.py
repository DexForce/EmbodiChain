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

"""Collect accepted ODS generations directly into bounded LeRobot persistence."""

from __future__ import annotations

import time
import math
from pathlib import Path
from typing import Any, Callable

from examples.ods.simple_bc_demo.lerobot import DirectLeRobotWriter


def collect_lerobot(
    engine: Any,
    initial: dict[str, Any],
    destination: Path,
    *,
    schema: dict[str, Any],
    episodes: int,
    snapshot: Callable[[Any], dict[str, Any]],
    request_refill: Callable[[], int],
    progress: Callable[[], None],
    clock: Callable[[], float],
    record_interval: Callable[[str, float, float], None],
    generation_timeout: float = 1800.0,
) -> dict[str, Any]:
    """Overlap generations with a bounded writer using cache-sized snapshots.

    The caller may retain the initial snapshot for normalization while this
    collector holds the current one. Numeric staging scales with cache capacity
    and the bounded writer queue, rather than total collected episodes.

    The caller owns engine shutdown and the common launch clock. Initial fill is
    already complete. No learner samples or intermediate numeric file are used.
    """
    rows, steps = initial["valid"].shape
    if episodes < rows or episodes % rows:
        raise ValueError("Collection requires a positive whole-cache episode count")
    if not math.isfinite(generation_timeout) or generation_timeout <= 0:
        raise ValueError("Generation timeout must be finite and positive")
    started = clock()
    writer = None
    try:
        with DirectLeRobotWriter(
            destination,
            schema=schema,
            episode_count=episodes,
            max_episode_steps=steps,
            queue_size=2,
            provenance={
                "source_run_id": engine.run_id,
                "collection": "direct_accepted_generations",
            },
        ) as writer:
            current = initial
            for index in range(episodes // rows):
                # The current snapshot is an independent copy. Producing the next
                # cache can safely overlap enqueue/write of these accepted rows.
                generation = request_refill() if (index + 1) * rows < episodes else None
                deadline = time.monotonic() + generation_timeout
                for row in range(rows):
                    progress()
                    writer.submit(
                        {
                            key: current[key][row]
                            for key in (
                                "observations",
                                "actions",
                                "valid",
                                "frame_ids",
                                "generations",
                            )
                        },
                        progress=progress,
                    )
                if generation is not None:
                    before = clock()
                    while engine.get_stats()["completed_generation"] < generation:
                        progress()
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                "Offline refill did not complete before its deadline"
                            )
                        time.sleep(0.02)
                    progress()
                    record_interval("offline_refill_wait", before, clock())
                    current = snapshot(engine)
            before = clock()
            writer.finalize(progress=progress)
            record_interval("offline_writer_finalize", before, clock())
            result = {
                "destination": str(destination),
                "writer": writer.stats,
                "collected_episodes": episodes,
                "intermediate_pt_file": False,
                "collection_seconds_after_initial_fill": clock() - started,
                "timing_scope": "generation and writer intervals overlap; do not add service durations",
            }
    except BaseException as error:
        if writer is not None and not writer.stats.get("worker_joined", True):
            raise RuntimeError(
                "Direct LeRobot writer did not join during cleanup"
            ) from error
        raise
    record_interval("offline_collection_and_persistence", started, clock())
    return result
