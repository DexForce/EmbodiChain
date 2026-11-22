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

"""Prepare the literal official Cube evaluation queries from verified data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401 -- registers the corpus's Blosc filter.
import numpy as np
from sklearn.preprocessing import StandardScaler

from ._reference import _sha

HDF5_SHA256 = "0664d507c4ff12009010644c9ae950836f954e700c172ccf22e7423af1a55625"


def prepare_queries(data: str | Path, output: str | Path) -> dict[str, Any]:
    """Verify the official corpus and reproduce seed-42, 25-step queries."""
    data, output = Path(data), Path(output)
    before = data.stat()
    if before.st_size != 101942558720 or _sha(data) != HDF5_SHA256:
        raise ValueError("official Cube HDF5 checksum mismatch")
    with h5py.File(data, "r") as file:
        episodes = file["ep_idx"][:]
        steps = file["step_idx"][:]
        lengths = file["ep_len"][:]
        offsets = file["ep_offset"][:]
        # Equivalent to eval.py's per-episode max(step_idx)+1, without rescanning
        # the two-million-row column once for every episode.
        measured = np.zeros(len(lengths), dtype=np.int64)
        np.maximum.at(measured, episodes, steps + 1)
        if not np.array_equal(measured, lengths):
            raise ValueError("episode-length metadata disagrees with source rows")
        valid = np.flatnonzero(steps <= (lengths - 25 - 1)[episodes])
        generator = np.random.default_rng(42)
        # The original script excludes the final valid row from choice().
        selected = np.sort(
            valid[generator.choice(len(valid) - 1, size=50, replace=False)]
        )
        future = selected + 25
        if not np.array_equal(episodes[selected], episodes[future]):
            raise ValueError("official queries cross an episode boundary")
        actions = file["action"][:]
        finite = actions[~np.isnan(actions).any(axis=1)]
        scaler = StandardScaler().fit(finite)
        columns = {key: file[key][selected] for key in ("qpos", "qvel", "pixels")}
        columns.update(
            {
                "goal_" + key: file[key][future]
                for key in (
                    "qpos",
                    "qvel",
                    "pixels",
                    "privileged_block_0_pos",
                    "privileged_block_0_quat",
                )
            }
        )
        if not np.allclose(
            columns["goal_qpos"][:, 14:17],
            columns["goal_privileged_block_0_pos"],
            atol=1e-12,
        ):
            raise ValueError("future-state and privileged goal positions disagree")
        distance = np.linalg.norm(
            columns["qpos"][:, 14:17] - columns["goal_privileged_block_0_pos"], axis=1
        )
        report = {
            "source_sha256": HDF5_SHA256,
            "source_rows": len(episodes),
            "episodes": len(lengths),
            "valid_start_rows": len(valid),
            "seed": 42,
            "num_eval": 50,
            "goal_offset_steps": 25,
            "selected_rows": selected.tolist(),
            "goal_rows": future.tolist(),
            "episode_ids": episodes[selected].tolist(),
            "start_steps": steps[selected].tolist(),
            "initial_goal_distances": distance.tolist(),
            "already_within_success_threshold": int(np.count_nonzero(distance <= 0.04)),
            "action_mean": scaler.mean_.tolist(),
            "action_std": scaler.scale_.tolist(),
            "scaler_rows": len(finite),
            "history_len": 1,
            "action_block": 5,
            "horizon": 5,
            "receding_horizon": 5,
            "eval_budget": 50,
            "max_episode_steps": 100,
        }
    after = data.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("official corpus changed during query preparation")
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as file:
        file.attrs["metadata"] = json.dumps(report)
        for key, value in columns.items():
            file.create_dataset(key, data=value)
        file.create_dataset("rows", data=selected)
        file.create_dataset("episode_ids", data=episodes[selected])
        file.create_dataset("start_steps", data=steps[selected])
    report["queries_sha256"] = _sha(output)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report
