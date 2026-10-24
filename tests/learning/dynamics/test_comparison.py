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

"""Immutable experiment preparation and truthful pending qualification reports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from embodichain.data_pipeline.dynamics.shards import DynamicsShardWriter
from embodichain.learning.dynamics.comparison import (
    prepare_comparison,
    run_comparison,
    run_candidate,
    _read_plan,
)
from embodichain.learning.dynamics.report import write_comparison_report


def configuration(tmp_path, monkeypatch):
    import embodichain.learning.dynamics.comparison as comparison
    from embodichain.learning.dynamics.reference import OFFICIAL_COMMIT

    monkeypatch.setattr(
        comparison,
        "reference_identity",
        lambda path: {
            "root": str(tmp_path / "source"),
            "commit": OFFICIAL_COMMIT,
            "files": {},
            "source_patches": [],
        },
    )
    with DynamicsShardWriter(
        tmp_path / "data", {"codec": {"robot": "cartpole"}, "dt": 1 / 60}
    ) as writer:
        for split in ("train", "validation", "test"):
            q = np.zeros((6, 2), np.float32)
            qd = np.zeros_like(q)
            writer.add_trajectory(
                split,
                {"q": q, "qd": qd},
                {"joint_f": np.ones((5, 2), np.float32)},
                split=split,
            )
    return {
        "output_dir": str(tmp_path / "experiment"),
        "source": str(tmp_path / "source"),
        "datasets": {"official_warp": None, "current_newton": str(tmp_path / "data")},
        "scope": "smoke",
        "seeds": [0],
        "model": {
            "input_dims": {"states_embedding": 4, "joint_f": 2},
            "output_dim": 4,
            "n_embd": 12,
            "n_head": 3,
            "n_layer": 1,
            "history_length": 3,
            "block_size": 3,
            "head_hidden_dims": [12],
        },
        "train": {
            "total_steps": 2,
            "batch_size": 2,
            "validation_interval": 1,
            "device": "cpu",
        },
    }


def test_preparation_shared_artifacts_and_missing_row_are_explicit(
    tmp_path, monkeypatch
):
    torch.set_num_threads(1)
    path = prepare_comparison(configuration(tmp_path, monkeypatch))
    root, plan = _read_plan(path)
    assert len(plan["cells"]) == 4
    assert sum(cell["status"] == "pending" for cell in plan["cells"]) == 2
    specs = [
        json.loads(Path(cell["spec"]).read_text())
        for cell in plan["cells"]
        if cell["status"] == "prepared"
    ]
    assert specs[0]["initial_model_sha256"] == specs[1]["initial_model_sha256"]
    assert specs[0]["sample_order_sha256"] == specs[1]["sample_order_sha256"]
    with pytest.raises(ValueError, match="trust_external_code"):
        run_comparison(path)
    report = write_comparison_report(path)
    assert report["status"] == "pending"
    assert any("Five" in reason for reason in report["pending"])
    assert (root / "report" / "training-curves.png").is_file()
    assert (root / "report" / "REPORT.md").is_file()


def test_changed_common_artifact_fails_before_execution(tmp_path, monkeypatch):
    path = prepare_comparison(configuration(tmp_path, monkeypatch))
    _, plan = _read_plan(path)
    name = next(iter(plan["artifacts"]))
    with (path.parent / name).open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="Immutable comparison artifact"):
        _read_plan(path)


def test_candidate_worker_uses_frozen_artifacts(tmp_path, monkeypatch):
    path = prepare_comparison(configuration(tmp_path, monkeypatch))
    _, plan = _read_plan(path)
    cell = next(
        cell
        for cell in plan["cells"]
        if cell["status"] == "prepared" and cell["trainer"] == "embodichain"
    )
    result = run_candidate(json.loads(Path(cell["spec"]).read_text()))
    assert result["completed_updates"] == 2
    assert result["test"]["split"] == "test"
    assert result["qualification"] == "matched_training_run_only"


def test_rollout_attachment_preserves_measurements_and_requires_identity(tmp_path):
    from embodichain.learning.dynamics.report import annotate_rollout_evidence

    source = tmp_path / "source.json"
    source.write_text(
        json.dumps({"trajectories": 2048, "horizons": {"1": {"q_mae": 0.1}}})
    )
    with pytest.raises(ValueError, match="model_sha256"):
        annotate_rollout_evidence(source, "run", tmp_path / "bad.json")
    raw = {
        "model_sha256": "abc",
        "motion_mode": "passive",
        "trajectories": 2048,
        "horizons": {
            "1": {
                "q_mae": 0.1,
                "qd_mae": 0.2,
                "failure_rate": 0.0,
                "nonfinite_rate": 0.0,
            },
            "10": {
                "q_mae": 0.2,
                "qd_mae": 0.3,
                "failure_rate": 0.1,
                "nonfinite_rate": 0.05,
            },
        },
    }
    source.write_text(json.dumps(raw))
    result = annotate_rollout_evidence(source, "run", tmp_path / "attached.json")
    assert result["horizons"] == [1, 10]
    assert result["errors"] == raw["horizons"]
    assert result["failure_rate"] == 0.1
    assert "policy_return" not in result


def test_preparation_keeps_selected_virtualenv_interpreter_symlink(
    tmp_path, monkeypatch
):
    import sys

    chosen = tmp_path / "selected-env/bin/python"
    chosen.parent.mkdir(parents=True)
    chosen.symlink_to(sys.executable)
    config = configuration(tmp_path, monkeypatch)
    config["python"] = str(chosen)
    path = prepare_comparison(config)
    _, plan = _read_plan(path)
    assert plan["python"] == str(chosen)
    assert all(
        cell["command"][0] == str(chosen)
        for cell in plan["cells"]
        if cell["status"] == "prepared"
    )
