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

"""Collect native Cube observations with disjoint episode membership."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from multiprocessing.util import Finalize
from pathlib import Path
import time
from typing import Any

import h5py
import numpy as np

from embodichain.lab.sim import SimulationManager

from .._official_eval import _native_state
from .._official_native import _make_cube, _NATIVE_RENDER, _pixels, _query_metadata
from ._resources import _CubeResources

_env: Any = None
_source: h5py.File | None = None
_pair_root: Path | None = None


def _episode_splits(resources: _CubeResources) -> dict[str, list[int]]:
    queries = _query_metadata(resources.queries)
    official = set(queries["episode_ids"])
    with h5py.File(resources.data, "r") as source:
        order = [
            int(i)
            for i in np.random.default_rng(23071).permutation(len(source["ep_offset"]))
            if int(i) not in official
        ]
    # Reserve the same development cases in both dataset profiles.
    return {
        "excluded": sorted(official | set(order[:50])),
        "paired_train": order[50:274],
        "paired_validation": order[274:322],
        "expert_train": order[322:2522],
        "expert_validation": order[2522:2822],
    }


def _close_pairs() -> None:
    _env.close()
    SimulationManager.flush_cleanup_queue()
    _source.close()


def _init_pairs(
    resources: _CubeResources, root: Path, gpu_ids: list[int], rank: Any
) -> None:
    global _env, _source, _pair_root
    with rank.get_lock():
        gpu_id = gpu_ids[rank.value % len(gpu_ids)]
        rank.value += 1
    _env = _make_cube(resources.reference, 1, gpu_id)
    _source = h5py.File(resources.data, "r")
    _pair_root = root
    Finalize(None, _close_pairs, exitpriority=10)


def _capture_pairs(episode: int) -> None:
    assert _env is not None and _source is not None and _pair_root is not None
    start = int(_source["ep_offset"][episode])
    steps = np.arange(episode % 5, 200, 5, dtype=np.int64)
    qpos, qvel = (_source[key][start : start + 201] for key in ("qpos", "qvel"))
    goal = _source["privileged_target_block_pos"][start]
    images = []
    before = _env.sim.physics_step_index
    for step in steps:
        obs, _ = _env.reset(
            seed=42, options={"qpos": qpos[step], "qvel": qvel[step], "goal_pos": goal}
        )
        images.append(_pixels(obs)[0])
    if _env.sim.physics_step_index != before:
        raise RuntimeError("Restoring the paired Cube poses advanced physics")
    actual, _ = _native_state(_env)
    np.testing.assert_allclose(actual[0, :17], qpos[steps[-1], :17], atol=2e-6, rtol=0)
    destination = _pair_root / f"{episode}.h5"
    partial = destination.with_suffix(".partial")
    with h5py.File(partial, "x") as file:
        file["rows"] = start + steps
        file.create_dataset(
            "pixels",
            data=np.asarray(images),
            compression="lzf",
            chunks=(1, 224, 224, 3),
        )
    partial.rename(destination)


def collect_pairs(
    resources: _CubeResources,
    output: Path,
    *,
    train_episodes: int,
    validation_episodes: int,
    gpu_ids: list[int],
    workers: int,
) -> dict[str, Any]:
    """Render archived poses with the task's camera for visual supervision."""
    split = _episode_splits(resources)
    ids = {
        "train": split["paired_train"][:train_episodes],
        "validation": split["paired_validation"][:validation_episodes],
    }
    if not 0 < train_episodes <= 224 or not 0 < validation_episodes <= 48:
        raise ValueError("Expected up to 224 training and 48 validation pair episodes")
    if workers < 1 or not gpu_ids:
        raise ValueError("Cube collection requires positive workers and render GPUs")
    output.mkdir(parents=True)
    pair_root = output / "pairs"
    pair_root.mkdir()
    start = time.monotonic()
    todo = sorted(ids["train"] + ids["validation"])
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(workers, len(todo)),
        mp_context=context,
        initializer=_init_pairs,
        initargs=(resources, pair_root, gpu_ids, context.Value("i", 0)),
    ) as pool:
        for count, _ in enumerate(pool.map(_capture_pairs, todo), 1):
            if count % 16 == 0 or count == len(todo):
                print(f"Rendered {count}/{len(todo)} Cube pair episodes", flush=True)
    queries = _query_metadata(resources.queries)
    report = {
        "kind": "cube-native-paired-poses",
        "episodes": ids,
        "render": _NATIVE_RENDER,
        "environment": "OGBenchCubeDexUni-v0",
        "physics": {"solver": "dexuni", "physics_dt": 0.002, "substeps": 25},
        "queries_sha256": queries["queries_sha256"],
        "source_sha256": queries["source_sha256"],
        "eval_episodes_excluded": split["excluded"],
        "frames_per_episode": 40,
        "seconds": time.monotonic() - start,
    }
    (output / "dataset.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
