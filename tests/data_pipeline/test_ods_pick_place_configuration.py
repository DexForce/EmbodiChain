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

"""CPU contracts for ODS rate, memory and selection-budget configuration."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from examples.data_pipeline._ods_pick_place_configuration import (
    apply_configuration,
    load_configuration,
)
from examples.data_pipeline.ods_pick_place_showcase import _parse_args


def _inputs() -> dict:
    return {
        "accepted_episodes_per_second": 1.0,
        "training_chunks_per_second": 1000.0,
        "profile": {
            "profile_id": "test-provisional-v1",
            "scope": "test fixture only",
            "qualified": False,
            "min_chunks_per_episode": 200,
            "max_chunks_per_episode": 600,
            "hard_max_chunks_per_episode": 1000,
        },
        "runtime": {
            "batch_size": 64,
            "producer_envs": 2,
            "first_publication_p95_seconds": 0.1,
            "measurement_scope": "CPU integration fixture, not a measured benchmark",
        },
        "binding": {
            "task": "PickPlace-Franka-v1",
            "policy": "chunk-pose",
            "prediction_horizon": 64,
            "action_horizon": 16,
            "pose_ik": "dls",
        },
    }


def _path(tmp_path: Path, inputs: dict | None = None) -> Path:
    path = tmp_path / "input.json"
    path.write_text(json.dumps(_inputs() if inputs is None else inputs))
    return path


def test_cli_applies_rate_plan_before_cache_and_production_validation(
    tmp_path, monkeypatch
):
    path = _path(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["showcase", "--policy", "chunk-pose", "--auto-config-input", str(path)],
    )
    args = _parse_args()
    plan = args.auto_configuration["configuration"]
    assert args.buffer_size == 4
    assert args.batch_size == 64
    assert args.producer_envs == 2
    assert args.max_generated_episodes == args.rounds * 4
    assert args.usage_statistics
    assert args.defer_evaluation
    assert plan["episode_sample_budget"] == 600
    assert plan["episode_sample_hard_limit"] == 1000
    assert plan["profile_qualified"] is False
    assert args.auto_configuration["training_budget"]["quota_update_ceiling"] == 75
    assert (
        args.auto_configuration["training_budget"]["maximum_additional_updates"] == 75
    )


@pytest.mark.parametrize("option", ["--smoke", "--prediction-horizon"])
def test_cli_rejects_overrides_that_change_profile_contract(
    tmp_path, monkeypatch, option
):
    extra = {
        "--smoke": [],
        "--prediction-horizon": ["32"],
    }[option]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "showcase",
            "--policy",
            "chunk-pose",
            "--auto-config-input",
            str(_path(tmp_path)),
            option,
            *extra,
        ],
    )
    with pytest.raises(SystemExit) as error:
        _parse_args()
    assert error.value.code == 2


def test_configuration_report_retains_exact_input_and_provisional_status(tmp_path):
    inputs = _inputs()
    report = load_configuration(_path(tmp_path, inputs))
    assert report["input"] == inputs
    assert len(report["input_sha256"]) == 64
    assert report["configuration"]["training_chunks_per_second_target"] == 600
    assert report["configuration"]["training_duty_fraction"] == pytest.approx(0.6)


def test_unknown_root_setting_is_rejected_instead_of_ignored(tmp_path):
    inputs = _inputs()
    inputs["episode_sample_budget"] = 999999
    with pytest.raises(ValueError, match="exactly"):
        load_configuration(_path(tmp_path, inputs))


@pytest.mark.parametrize(
    "flag,value",
    [("--batch-size", "256"), ("--producer-envs", "8"), ("--buffer-size", "32")],
)
def test_explicit_runtime_conflict_is_rejected(tmp_path, monkeypatch, flag, value):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "showcase",
            "--policy",
            "chunk-pose",
            "--auto-config-input",
            str(_path(tmp_path)),
            flag,
            value,
        ],
    )
    with pytest.raises(SystemExit) as error:
        _parse_args()
    assert error.value.code == 2
