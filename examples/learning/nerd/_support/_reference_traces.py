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

"""Bounded replay of supported-environment original-Warp evaluation traces."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from examples.learning.nerd._support._evaluation import _teacher_row_sha256
from embodichain_models.dynamics.nerd.data import load_manifest
from ._reference import OFFICIAL_COMMIT

__all__: list[str] = []


def _load_reference_trace(
    directory: Path,
    spec: Mapping[str, Any],
    source: Mapping[str, Any],
    model_metadata: Mapping[str, Any],
    policy_identity: Mapping[str, Any] | None,
    count: int,
    horizon: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = load_manifest(directory, verify=True)
    metadata, evaluation = manifest.get("metadata", {}), manifest.get("evaluation", {})
    environment = metadata.get("generation_environment", {})
    robot = spec["robot"]
    origin = metadata.get("teacher_origin", {})
    expected_warp = "1.5.1" if robot == "ant" else "1.8.0"
    if (
        manifest.get("trace_type") != "evaluation_including_failures"
        or metadata.get("robot") != robot
        or origin.get("kind") != "official_warp"
        or origin.get("source_commit") != OFFICIAL_COMMIT
        or origin.get("warp_version") != expected_warp
        or environment.get("python") != "3.8.20"
        or str(environment.get("torch", "")).split("+")[0] != "2.2.2"
        or environment.get("numpy") != "1.24.4"
        or environment.get("warp") != expected_warp
        or any(
            environment.get("runtime_imports_" + name) is not False
            for name in ("embodichain", "dexsim", "newton")
        )
    ):
        raise ValueError(
            "Teacher trace must be original-Warp evaluation data generated in the supported isolated environment"
        )
    if (
        metadata.get("asset_sha256") != model_metadata.get("asset_sha256")
        or metadata.get("codec") != model_metadata.get("codec")
        or metadata.get("dt") != model_metadata.get("dt")
        or metadata.get("world_rotation_xyzw")
        != model_metadata.get("world_rotation_xyzw")
        or metadata.get("source_runtime_files") != source.get("runtime_files")
    ):
        raise ValueError(
            "Teacher source, asset or coordinate contract differs from the model/runtime"
        )
    if (
        evaluation.get("motion_mode") != spec.get("motion_mode", "random_effort")
        or evaluation.get("seed") != spec.get("seed", 500)
        or evaluation.get("initial_states") != spec.get("initial_states", "sample")
        or evaluation.get("policy") != policy_identity
        or type(evaluation.get("horizon")) is not int
        or evaluation["horizon"] < horizon
    ):
        raise ValueError(
            "Teacher evaluation seed, initial state, horizon, motion or policy mismatch"
        )
    records = [
        {**row, "file": shard["file"]}
        for shard in manifest["shards"]
        for row in shard["trajectories"]
    ]
    if evaluation.get("trajectories") != len(records) or len(records) < count:
        raise ValueError("Teacher corpus omits requested evaluation trajectories")
    for index, row in enumerate(records):
        if (
            row.get("split") != "test"
            or row.get("length") != evaluation["horizon"]
            or row.get("metadata", {}).get("sampling_seed")
            != evaluation["seed"] + index
            or row.get("id") != f"{evaluation['seed']}:{index}"
        ):
            raise ValueError(
                "Teacher corpus reorders, filters or mislabels evaluation trajectories"
            )
    return manifest, records[:count]


def _reference_trace_batches(
    directory: Path,
    records: list[dict[str, Any]],
    batch: int,
    horizon: int,
    robot: str,
    device: str,
) -> Iterator[tuple[list[dict[str, Any]], dict[str, torch.Tensor]]]:
    fields = {
        "raw": "states/raw",
        "q": "states/q",
        "qd": "states/qd",
        "joint_f": "transitions/joint_f",
        "raw_joint_f": "transitions/raw_joint_f",
    }
    q_dim, qd_dim, force_dim = (15, 14, 8) if robot == "ant" else (2, 2, 2)
    widths = {
        "raw": q_dim + qd_dim,
        "q": q_dim,
        "qd": qd_dim,
        "joint_f": qd_dim,
        "raw_joint_f": force_dim,
    }
    handles: OrderedDict[str, h5py.File] = OrderedDict()
    try:
        for start in range(0, len(records), batch):
            selected = records[start : start + batch]
            values = []
            for row in selected:
                filename = row["file"]
                if filename not in handles:
                    if len(handles) >= 4:
                        _, stream = handles.popitem(last=False)
                        stream.close()
                    handles[filename] = h5py.File(directory / filename, "r")
                handles.move_to_end(filename)
                group = handles[filename][row["group"]]
                arrays = {}
                for name, key in fields.items():
                    if key not in group:
                        raise ValueError(
                            "Original teacher trace is missing raw/canonical fields"
                        )
                    steps = horizon + 1 if name in {"raw", "q", "qd"} else horizon
                    value = group[key][:steps]
                    if value.dtype != np.dtype("float32") or value.shape != (
                        steps,
                        widths[name],
                    ):
                        raise ValueError(
                            "Original teacher trace has invalid dtype, shape or length"
                        )
                    arrays[name] = value
                if not np.isfinite(arrays["raw_joint_f"]).all():
                    raise ValueError(
                        "Original teacher input efforts must remain finite"
                    )
                expected = (
                    np.concatenate(
                        (
                            np.zeros((horizon, 6), dtype=np.float32),
                            arrays["raw_joint_f"],
                        ),
                        -1,
                    )
                    if robot == "ant"
                    else arrays["raw_joint_f"]
                )
                if not np.array_equal(expected, arrays["joint_f"]):
                    raise ValueError(
                        "Recorded canonical and original teacher controls differ"
                    )
                row["reference_sha256"] = _teacher_row_sha256(
                    row, [arrays["q"], arrays["qd"], arrays["joint_f"]]
                )
                values.append(arrays)
            values.extend([values[-1]] * (batch - len(values)))
            yield selected, {
                name: torch.from_numpy(np.stack([value[name] for value in values])).to(
                    device
                )
                for name in fields
            }
    finally:
        for stream in handles.values():
            stream.close()
