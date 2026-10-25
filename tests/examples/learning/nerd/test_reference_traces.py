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

"""Original trace provenance, ordering, bounded batching and failure retention."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import torch

from examples.learning.nerd._support._reference_traces import (
    _load_reference_trace,
    _reference_trace_batches,
)
from examples.learning.nerd._support._reference import OFFICIAL_COMMIT


def _fixture(
    root: Path, robot: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root.mkdir()
    ant = robot == "ant"
    q_dim, qd_dim, force_dim = (15, 14, 8) if ant else (2, 2, 2)
    codec = {
        "robot": robot,
        "base_com": [0.0, 0.0, 0.0],
        "continuous_q_indices": [] if ant else [1],
    }
    shared = {
        "robot": robot,
        "asset_sha256": "fixture-asset",
        "dt": 1 / 60,
        "codec": codec,
        "world_rotation_xyzw": [1.0, 0.0, 0.0, 0.0],
    }
    source = {"runtime_files": {"fixture-source.py": "fixture-source-hash"}}
    version = "1.5.1" if ant else "1.8.0"
    meta = {
        **shared,
        "source_runtime_files": source["runtime_files"],
        "teacher_origin": {
            "kind": "official_warp",
            "source_commit": OFFICIAL_COMMIT,
            "warp_version": version,
        },
        "generation_environment": {
            "python": "3.8.20",
            "torch": "2.2.2+cpu",
            "numpy": "1.24.4",
            "warp": version,
            "runtime_imports_embodichain": False,
            "runtime_imports_dexsim": False,
            "runtime_imports_newton": False,
        },
    }
    rows = []
    file = root / "teacher.h5"
    with h5py.File(file, "w") as stream:
        for index in range(3):
            group = stream.create_group(str(index))
            q = np.full((5, q_dim), index + 0.25, dtype=np.float32)
            qd = np.zeros((5, qd_dim), dtype=np.float32)
            if index == 2:
                q[-1, 0] = np.nan
            raw = np.concatenate((q, qd), -1)
            force = np.full((4, force_dim), index, dtype=np.float32)
            canonical = (
                np.concatenate((np.zeros((4, 6), dtype=np.float32), force), -1)
                if ant
                else force
            )
            for key, value in {
                "states/raw": raw,
                "states/q": q,
                "states/qd": qd,
                "transitions/raw_joint_f": force,
                "transitions/joint_f": canonical,
            }.items():
                group.create_dataset(key, data=value)
            rows.append(
                {
                    "id": f"71:{index}",
                    "group": str(index),
                    "length": 4,
                    "split": "test",
                    "metadata": {"sampling_seed": 71 + index},
                }
            )
    manifest = {
        "schema_version": 1,
        "trace_type": "evaluation_including_failures",
        "metadata": meta,
        "evaluation": {
            "seed": 71,
            "trajectories": 3,
            "horizon": 4,
            "initial_states": "sample",
            "motion_mode": "random_effort",
            "policy": None,
        },
        "shards": [
            {
                "file": file.name,
                "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                "trajectories": rows,
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return {"robot": robot, "seed": 71, "motion_mode": "random_effort"}, source, shared


@pytest.mark.parametrize("robot", ["cartpole", "ant"])
def test_original_trace_preserves_order_padding_failure_and_ownership(
    tmp_path: Path, robot: str
) -> None:
    directory = tmp_path / "teacher"
    spec, source, metadata = _fixture(directory, robot)
    _, records = _load_reference_trace(directory, spec, source, metadata, None, 3, 4)
    batches = list(_reference_trace_batches(directory, records, 2, 4, robot, "cpu"))
    assert [[row["id"] for row in selected] for selected, _ in batches] == [
        ["71:0", "71:1"],
        ["71:2"],
    ]
    assert batches[0][1]["q"][0, 0, 0] == 0.25
    assert batches[1][1]["q"][0, 0, 0] == 2.25
    assert torch.isnan(batches[1][1]["q"][:, -1, 0]).all()
    assert all(row["reference_sha256"] for row in records)
    batches[1][1]["q"][0, 0, 0] = 99
    assert batches[1][1]["q"][1, 0, 0] == 2.25
    assert batches[0][1]["q"][0, 0, 0] == 0.25


@pytest.mark.parametrize(
    "field,value",
    [
        ("python", "3.11.14"),
        ("torch", "2.7.0"),
        ("warp", "1.17.0"),
        ("runtime_imports_embodichain", True),
        ("runtime_imports_newton", True),
    ],
)
def test_original_trace_rejects_wrong_generation_environment(
    tmp_path: Path, field: str, value: Any
) -> None:
    directory = tmp_path / "teacher"
    spec, source, metadata = _fixture(directory, "ant")
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["metadata"]["generation_environment"][field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="supported isolated environment"):
        _load_reference_trace(directory, spec, source, metadata, None, 3, 4)


@pytest.mark.parametrize("change", ["seed", "order", "count", "asset", "policy"])
def test_original_trace_rejects_mispaired_evaluation(
    tmp_path: Path, change: str
) -> None:
    directory = tmp_path / "teacher"
    spec, source, metadata = _fixture(directory, "cartpole")
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    if change == "seed":
        manifest["evaluation"]["seed"] += 1
    elif change == "order":
        manifest["shards"][0]["trajectories"].reverse()
    elif change == "count":
        manifest["evaluation"]["trajectories"] += 1
    elif change == "asset":
        manifest["metadata"]["asset_sha256"] = "another-asset"
    else:
        manifest["evaluation"]["policy"] = {"kind": "pd"}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        _load_reference_trace(directory, spec, source, metadata, None, 3, 4)


def test_original_trace_rejects_rehashed_control_mismatch(tmp_path: Path) -> None:
    directory = tmp_path / "teacher"
    spec, source, metadata = _fixture(directory, "ant")
    file = directory / "teacher.h5"
    with h5py.File(file, "a") as stream:
        stream["0/transitions/joint_f"][1, 0] = 1
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["shards"][0]["sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    _, records = _load_reference_trace(directory, spec, source, metadata, None, 3, 4)
    with pytest.raises(ValueError, match="controls differ"):
        list(_reference_trace_batches(directory, records, 2, 4, "ant", "cpu"))


def test_original_trace_rejects_raw_shard_tampering(tmp_path: Path) -> None:
    directory = tmp_path / "teacher"
    spec, source, metadata = _fixture(directory, "cartpole")
    with (directory / "teacher.h5").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="checksum mismatch"):
        _load_reference_trace(directory, spec, source, metadata, None, 3, 4)


@pytest.mark.parametrize("change", ["missing", "dtype", "length"])
def test_original_trace_rejects_rehashed_invalid_state_schema(
    tmp_path: Path, change: str
) -> None:
    directory = tmp_path / "teacher"
    spec, source, metadata = _fixture(directory, "cartpole")
    file = directory / "teacher.h5"
    with h5py.File(file, "a") as stream:
        value = stream["0/states/raw"][()]
        del stream["0/states/raw"]
        if change != "missing":
            stream.create_dataset(
                "0/states/raw",
                data=value.astype(np.float64) if change == "dtype" else value[:-1],
            )
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["shards"][0]["sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    _, records = _load_reference_trace(directory, spec, source, metadata, None, 3, 4)
    with pytest.raises(
        ValueError, match="missing raw/canonical fields|dtype, shape or length"
    ):
        list(_reference_trace_batches(directory, records, 2, 4, "cartpole", "cpu"))
