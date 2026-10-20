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

"""Qualification cannot substitute pass flags or mismatched rollout populations."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from embodichain.learning.dynamics.report import (
    _matched_rollouts,
    _numerical_evidence,
    _rollout_evidence,
    annotate_rollout_evidence,
)
from embodichain.learning.dynamics.reference import _hash


def numerical_fixture() -> dict:
    return {
        "robot": "Cartpole",
        "reference_commit": "476dc67fccc6a99be293bf611f897b1c06f13e30",
        "checkpoint_sha256": "a" * 64,
        "codec_fixtures_sha256": "b" * 64,
        "history_lengths": [1, 9, 10],
        "adam_updates": 3,
        "precision": "float32",
        "atol": 1e-5,
        "rtol": 1e-4,
        **{
            key: 1e-7
            for key in (
                "max_codec_feature_error",
                "max_codec_target_error",
                "max_codec_decode_error",
                "max_forward_error",
                "max_input_gradient_error",
                "max_parameter_gradient_error",
                "max_adam_parameter_error",
            )
        },
    }


def test_numerical_receipt_requires_measurements_and_matching_identity(tmp_path):
    path = tmp_path / "numerical.json"
    path.write_text(json.dumps(numerical_fixture()))
    assert _numerical_evidence(path, "cartpole")["status"] == "passed"
    assert _numerical_evidence(path, "ant")["status"] == "failed"
    path.write_text(
        json.dumps(
            {
                "checks": {
                    key: {"passed": True}
                    for key in (
                        "preprocessing",
                        "forward",
                        "input_gradients",
                        "parameter_gradients",
                        "adam_update",
                    )
                }
            }
        )
    )
    assert _numerical_evidence(path, "cartpole")["status"] == "pending"


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_forward_error", float("nan")),
        ("max_codec_decode_error", -1.0),
        ("atol", 1.0),
        ("reference_commit", "unrelated"),
    ],
)
def test_invalid_numerical_measurements_cannot_pass(tmp_path, field, value):
    raw = numerical_fixture()
    raw[field] = value
    path = tmp_path / "numerical.json"
    path.write_text(json.dumps(raw))
    assert _numerical_evidence(path, "cartpole")["status"] == "failed"


def rollout_fixture(tmp_path: Path, *, motion: str = "passive"):
    model = tmp_path / "run" / "best-export"
    model.mkdir(parents=True)
    (model / "weights.pt").write_bytes(b"synthetic test artifact")
    (model / "manifest.json").write_text("{}")
    trace = tmp_path / "trace.h5"
    trace.write_bytes(b"synthetic test trace")
    trajectories = tmp_path / "rows.jsonl"
    trajectories.write_bytes(b"synthetic test row index")
    raw = {
        "robot": "cartpole",
        "motion_mode": motion,
        "state_dtype": "float32",
        "teacher_origin": {"kind": "current_newton"},
        "teacher_split": "test",
        "selected_teacher_sha256": "a" * 64,
        "selected_teacher_hash_schema": "ordered-teacher-content-v1",
        "runtime_identity": {
            "device": "cpu",
            "engine": "current_newton",
            "state_dtype": "float32",
            "python": "3.11",
            "versions": {"torch": "2.7.1", "newton": "1.6.0", "warp-lang": "1.17.0"},
        },
        "trajectories": 2048,
        "horizons": {str(h): {"q_mae": 0.1, "qd_mae": 0.2} for h in (1, 10, 100, 1000)},
        "model_sha256": _hash(model / "weights.pt"),
        "model_manifest_sha256": _hash(model / "manifest.json"),
        "trace_evidence": str(trace),
        "trace_evidence_sha256": _hash(trace),
        "trajectory_evidence": str(trajectories),
        "trajectory_evidence_sha256": _hash(trajectories),
        "failure_rate": 0.0,
        "nonfinite_rate": 0.0,
        "policy_return": 1.0 if motion == "fixed_policy" else None,
        "policy_sha256": "b" * 64 if motion == "fixed_policy" else None,
    }
    source = tmp_path / "source.json"
    source.write_text(json.dumps(raw))
    record = annotate_rollout_evidence(
        source, "current_newton-official-0", tmp_path / "attached.json"
    )
    cell = {"teacher": "current_newton", "output_dir": str(model.parent)}
    return record, cell


def test_rollout_source_model_and_trace_identity_are_verified(tmp_path):
    record, cell = rollout_fixture(tmp_path)
    assert _rollout_evidence(record, cell, "cartpole")[0] == "available"
    record["errors"]["1"]["q_mae"] = 0.0
    assert "Annotated" in _rollout_evidence(record, cell, "cartpole")[1]
    record["errors"]["1"]["q_mae"] = 0.1
    Path(record["trace_evidence"]).write_bytes(b"changed")
    assert "changed" in _rollout_evidence(record, cell, "cartpole")[1]


@pytest.mark.parametrize(
    "field,value",
    [
        ("teacher_origin", {"kind": "official_warp"}),
        ("teacher_split", "train"),
        ("failure_rate", -1.0),
        ("nonfinite_rate", 1.5),
        ("robot", "ant"),
    ],
)
def test_wrong_domain_split_or_invalid_rates_fail(tmp_path, field, value):
    record, cell = rollout_fixture(tmp_path)
    record[field] = value
    assert _rollout_evidence(record, cell, "cartpole")[0] == "failed"


def test_undefined_error_is_failed_instead_of_crashing_statistics(tmp_path):
    record, cell = rollout_fixture(tmp_path)
    record["errors"]["1000"]["q_mae"] = None
    assert _rollout_evidence(record, cell, "cartpole")[0] == "failed"


def test_unmatched_teacher_traces_or_fixed_policy_never_form_seed_pair(tmp_path):
    record, cell = rollout_fixture(tmp_path, motion="fixed_policy")
    assert _rollout_evidence(record, cell, "cartpole")[0] == "available"
    other = copy.deepcopy(record)
    assert _matched_rollouts(record, other)
    other["selected_teacher_sha256"] = "c" * 64
    assert not _matched_rollouts(record, other)
    other = copy.deepcopy(record)
    other["policy_sha256"] = "d" * 64
    assert not _matched_rollouts(record, other)


def test_historical_rollout_uses_pinned_warp_identity_without_newton(tmp_path):
    from embodichain.learning.dynamics.reference import OFFICIAL_COMMIT

    record, cell = rollout_fixture(tmp_path)
    source = Path(record["source_report"])
    raw = json.loads(source.read_text())
    raw["teacher_origin"] = {
        "kind": "official_warp",
        "source_commit": OFFICIAL_COMMIT,
        "warp_version": "1.8.0",
    }
    raw["runtime_identity"] = {
        "device": "cpu",
        "engine": "official_warp",
        "state_dtype": "float32",
        "python": "3.11",
        "reference_commit": OFFICIAL_COMMIT,
        "versions": {"torch": "2.7.1", "warp-lang": "1.8.0"},
    }
    raw["physical_verification"] = {
        "schema": "official-warp-physical-arrays-v2",
        "complete": True,
        "matched_recorded_fields": True,
        "missing_recorded_fields": [],
        "matched_fields": ["gravity"],
        "sha256": "f" * 64,
    }
    source.write_text(json.dumps(raw))
    record = annotate_rollout_evidence(
        source, "official_warp-official-0", tmp_path / "historical.json"
    )
    cell["teacher"] = "official_warp"
    assert _rollout_evidence(record, cell, "cartpole")[0] == "available"
    record["physical_verification"]["missing_recorded_fields"] = ["gravity"]
    assert _rollout_evidence(record, cell, "cartpole")[0] == "pending"
    record["physical_verification"]["missing_recorded_fields"] = []
    record["runtime_identity"]["versions"]["warp-lang"] = "1.17.0"
    assert "pinned original Warp" in _rollout_evidence(record, cell, "cartpole")[1]


def test_full_profile_checks_actual_feature_order_precision_and_training_controls(
    tmp_path,
):
    import torch
    from embodichain.learning.dynamics.report import _full_profile_issues
    from embodichain.learning.dynamics.trainer import DynamicsTrainCfg
    from embodichain.learning.models.nerd import NeRDModelCfg

    model = NeRDModelCfg(input_dims={"states_embedding": 4, "joint_f": 2}).to_dict()
    root = tmp_path / "initial"
    root.mkdir()
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"model_config": model}))
    torch.save({"weight": torch.ones(1)}, root / "weights.pt")
    spec = tmp_path / "run.json"
    spec.write_text(json.dumps({"initial_model": str(root)}))
    train = DynamicsTrainCfg(
        total_steps=4995000,
        batch_size=512,
        validation_interval=5000,
        validation_batches=100,
        lr_update_interval=5000,
        lr_offset=1,
    ).to_dict()
    # Plan JSON sorts dict keys. This must not reorder the actual network inputs.
    plan = {
        "model": json.loads(json.dumps(model, sort_keys=True)),
        "train": train,
        "cells": [{"id": "fixture", "status": "prepared", "spec": str(spec)}],
    }
    assert _full_profile_issues(plan) == []
    model["input_dims"] = {"joint_f": 2, "states_embedding": 4}
    manifest.write_text(json.dumps({"model_config": model}))
    assert any("feature order" in issue for issue in _full_profile_issues(plan))
    model["input_dims"] = {"states_embedding": 4, "joint_f": 2}
    model["head_hidden_dims"] = [128]
    manifest.write_text(json.dumps({"model_config": model}))
    torch.save({"weight": torch.ones(1, dtype=torch.float64)}, root / "weights.pt")
    plan["train"]["batch_size"] = 16
    issues = _full_profile_issues(plan)
    assert any("head_hidden_dims" in issue for issue in issues)
    assert any("FP32" in issue for issue in issues)
    assert any("batch_size" in issue for issue in issues)
