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

"""Validate usage-study commands, source freezing and safe resume behavior."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest
import torch

from scripts.benchmark.data_pipeline import run_ods_usage_study as study


def _args(output: Path, **kwargs: object) -> argparse.Namespace:
    values = dict(
        output=output,
        stage="screen",
        cells=None,
        producer_envs=1,
        gpu_uuid="GPU-12345678-1234-1234-1234-123456789abc",
        cpus=str(min(os.sched_getaffinity(0))),
        python=sys.executable,
        runtime_overlay=None,
        minimum_free_mib=16384,
        execute=False,
        resume=False,
    )
    values.update(kwargs)
    return argparse.Namespace(**values)


def _flag(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_matrix_freezes_shared_budgets_and_excludes_formal_test(tmp_path) -> None:
    manifest = study.build_manifest(_args(tmp_path))
    assert len(manifest["jobs"]) == 10
    for job in manifest["jobs"]:
        command = job["argv"]
        assert _flag(command, "--max-generated-episodes") == "256"
        assert _flag(command, "--rounds") == "10"
        assert _flag(command, "--updates-per-round") == "5000"
        assert _flag(command, "--wall-time-budget") == "5400"
        assert _flag(command, "--final-test-scenes") == "0"
        assert _flag(command, "--dev-seed-base") == "810000"
        assert _flag(command, "--sampling-seed") == "23001"
        assert _flag(command, "--train-device") == "cuda:0"
        assert "610000" not in command
    reference = manifest["jobs"][-1]
    assert _flag(reference["argv"], "--usage-schedule") == "fixed"
    assert "keep training last cache" in reference["stop_semantics"]
    assert manifest["environment"]["CUDA_VISIBLE_DEVICES"].startswith("GPU-")


def test_confirmation_requires_selection_and_uses_three_new_seeds(tmp_path) -> None:
    with pytest.raises(ValueError, match="explicitly selected"):
        study.build_manifest(_args(tmp_path, stage="confirmation"))
    manifest = study.build_manifest(
        _args(tmp_path, stage="confirmation", cells=["c32-r32", "reference"])
    )
    assert len(manifest["jobs"]) == 6
    assert {job["seed"] for job in manifest["jobs"]} == {14001, 14002, 14003}
    assert all(
        _flag(job["argv"], "--dev-seed-base") == "810000" for job in manifest["jobs"]
    )
    assert all(_flag(job["argv"], "--dev-scenes") == "20" for job in manifest["jobs"])


def test_dry_run_never_launches_and_manifest_is_immutable(
    tmp_path, monkeypatch
) -> None:
    def metadata_read(command: list[str], **kwargs: object) -> str:
        assert command == ["git", "rev-parse", "HEAD"]
        return "test-revision\n"

    monkeypatch.setattr(study.subprocess, "check_output", metadata_read)
    launch = Mock(side_effect=AssertionError("dry run must not launch experiments"))
    monkeypatch.setattr(study.subprocess, "Popen", launch)
    args = _args(tmp_path, cells=["c16-r8"])
    study.run_study(args)
    assert (tmp_path / "manifest.json").exists()
    assert not (tmp_path / "runs").exists()
    launch.assert_not_called()
    with pytest.raises(FileExistsError):
        study.run_study(args)
    args.resume = True
    study.run_study(args)
    args.producer_envs = 2
    with pytest.raises(FileExistsError):
        study.run_study(args)


def test_partial_and_failed_cleanup_cannot_be_resumed(tmp_path) -> None:
    job = {"id": "test", "output": str(tmp_path)}
    with pytest.raises(RuntimeError, match="Partial"):
        study._completed(job, {"status": "running"})
    status = {"status": "completed", "returncode": 0, "process_seconds": 4}
    (tmp_path / "cleanup.json").write_text(
        json.dumps({"engine_cleanup_error": "forced", "evaluator_exitcode": 0})
    )
    with pytest.raises(RuntimeError, match="cleanup"):
        study._completed(job, status)
    (tmp_path / "cleanup.json").write_text(
        json.dumps({"engine_cleanup_error": None, "evaluator_exitcode": 0})
    )
    (tmp_path / "summary.json").write_text(
        json.dumps({"final_success_rate": None, "training_step": 12})
    )
    (tmp_path / "deferred_evaluation.json").write_text(
        json.dumps(
            {"training_finished_seconds": 3, "last_report": {"success_rate": 0.5}}
        )
    )
    result = study._completed(job, status)
    assert result["latest_development_success"] == 0.5
    (tmp_path / "failure.json").write_text("{}")
    with pytest.raises(RuntimeError, match="failure evidence"):
        study._completed(job, status)


def test_cpu_parser_and_uuid_reject_ambiguous_resources(tmp_path) -> None:
    assert study.parse_cpu_affinity("1-3,8,2") == [1, 2, 3, 8]
    for value in ("", "3-1", "-1", "1-2-3"):
        with pytest.raises(ValueError):
            study.parse_cpu_affinity(value)
    with pytest.raises(ValueError, match="full physical GPU"):
        study.build_manifest(_args(tmp_path, gpu_uuid="0"))


def _confirmation_job(root: Path, *, same_checkpoint: bool = False) -> dict:
    root.mkdir()
    best = root / "best.pt"
    torch.save(
        {
            "model": {"weight": torch.tensor([1.0]), "mean": torch.tensor([0.0])},
            "policy_config": {"action_horizon": 16},
            "config": {"pose_ik": "dls"},
        },
        best,
    )
    latest = best if same_checkpoint else root / "latest.pt"
    if not same_checkpoint:
        torch.save(
            {
                "model": {"weight": torch.tensor([2.0]), "mean": torch.tensor([0.0])},
                "policy_config": {"action_horizon": 16},
                "config": {"pose_ik": "dls"},
            },
            latest,
        )
    (root / "task_config.json").write_text(
        json.dumps({"physics": "default", "controller": "dls"})
    )
    (root / "summary.json").write_text(json.dumps({"best_checkpoint": str(best)}))
    (root / "deferred_evaluation.json").write_text(
        json.dumps({"last_checkpoint": str(latest)})
    )
    return {
        "output": str(root),
        "confirmation_seeds": list(range(820000, 820100)),
        "argv": ["taskset", "--cpu-list", "1", sys.executable],
    }


def _healthy_panel(output: Path, checkpoint: Path) -> None:
    output.mkdir()
    (output / "cleanup.json").write_text(json.dumps({"forced": False, "exitcode": 0}))
    (output / "report.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": study._sha256(checkpoint),
                "seeds": list(range(820000, 820100)),
                "records": [{"seed": seed} for seed in range(820000, 820100)],
                "success_rate": 0.8,
            }
        )
    )


def test_confirmation_freezes_best_latest_and_isolates_each_child(
    tmp_path, monkeypatch
) -> None:
    job = _confirmation_job(tmp_path / "run")
    launched = []
    environment = {"CUDA_VISIBLE_DEVICES": "GPU-frozen"}
    manifest = {
        "environment": environment,
        "gpu_uuid": "GPU-frozen",
        "cwd": str(tmp_path),
    }

    def launch(command, **kwargs):
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-frozen"
        assert "_confirm-one" in command and "--execute" in command
        output = Path(_flag(command, "--output"))
        checkpoint = Path(_flag(command, "--checkpoint"))
        _healthy_panel(output, checkpoint)
        launched.append(command)
        return Mock(wait=Mock(return_value=0))

    monkeypatch.setattr(study.subprocess, "Popen", launch)
    result = study._run_confirmation(job, manifest)
    assert len(launched) == 2
    assert result["confirmation_best_success"] == 0.8
    assert result["confirmation_latest_success"] == 0.8
    assert result["confirmation_scenes"] == 100
    assert study._confirmation_results(job) == result
    # The confirmation score cannot change which checkpoint has the best role.
    frozen = json.loads(
        (Path(job["output"]) / "confirmation/selection.json").read_text()
    )
    assert frozen["panels"][0]["roles"] == ["best"]
    assert frozen["selection_panel_seeds"] == list(range(810000, 810020))
    (Path(job["output"]) / "summary.json").write_text(
        json.dumps({"best_checkpoint": str(Path(job["output"]) / "latest.pt")})
    )
    with pytest.raises(RuntimeError, match="selection changed"):
        study._confirmation_results(job)


def test_confirmation_deduplicates_same_frozen_checkpoint(tmp_path) -> None:
    selection = study._confirmation_selection(
        _confirmation_job(tmp_path / "run", same_checkpoint=True)
    )
    assert len(selection["panels"]) == 1
    assert selection["panels"][0]["roles"] == ["best", "latest"]


@pytest.mark.parametrize(
    "cleanup", [{}, {"forced": True, "exitcode": 0}, {"forced": False, "exitcode": -15}]
)
def test_confirmation_resume_rejects_unknown_or_failed_cleanup(
    tmp_path, cleanup
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"weights")
    output = tmp_path / "panel"
    _healthy_panel(output, checkpoint)
    (output / "cleanup.json").write_text(json.dumps(cleanup))
    with pytest.raises(RuntimeError, match="cleanup"):
        study._read_confirmation_panel(
            output,
            {"checkpoint_sha256": study._sha256(checkpoint)},
            list(range(820000, 820100)),
        )


def test_confirmation_resume_rejects_missing_or_wrong_panel(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"weights")
    output = tmp_path / "panel"
    _healthy_panel(output, checkpoint)
    panel = {"checkpoint_sha256": study._sha256(checkpoint)}
    with pytest.raises(RuntimeError, match="mismatch"):
        study._read_confirmation_panel(output, panel, list(range(810000, 810020)))
    (output / "report.json").unlink()
    with pytest.raises(FileNotFoundError):
        study._read_confirmation_panel(output, panel, list(range(820000, 820100)))


def test_confirmation_deduplicates_policy_ignoring_checkpoint_metadata(
    tmp_path, monkeypatch
) -> None:
    job = _confirmation_job(tmp_path / "run")
    root = Path(job["output"])
    saved = torch.load(root / "best.pt", map_location="cpu", weights_only=False)
    saved.update(
        step=999, optimizer={"different": [1, 2, 3]}, torch_rng=torch.tensor([8])
    )
    torch.save(saved, root / "latest.pt")
    assert study._sha256(root / "best.pt") != study._sha256(root / "latest.pt")
    launches = []

    def launch(command, **kwargs):
        launches.append(command)
        _healthy_panel(
            Path(_flag(command, "--output")), Path(_flag(command, "--checkpoint"))
        )
        return Mock(wait=Mock(return_value=0))

    monkeypatch.setattr(study.subprocess, "Popen", launch)
    result = study._run_confirmation(
        job,
        {
            "environment": {"CUDA_VISIBLE_DEVICES": "GPU-isolated"},
            "gpu_uuid": "GPU-isolated",
            "cwd": str(tmp_path),
        },
    )
    assert len(launches) == 1
    roles = result["confirmation_roles"]
    assert roles["best"]["report"] == roles["latest"]["report"]
    assert roles["best"]["checkpoint"] != roles["latest"]["checkpoint"]
    assert roles["best"]["checkpoint_sha256"] != roles["latest"]["checkpoint_sha256"]
    assert roles["best"]["policy_sha256"] == roles["latest"]["policy_sha256"]


@pytest.mark.parametrize("change", ["weight", "mean", "horizon", "controller", "task"])
def test_policy_identity_changes_with_executable_state(tmp_path, change) -> None:
    job = _confirmation_job(tmp_path / "run")
    root = Path(job["output"])
    checkpoint = root / "best.pt"
    task = json.loads((root / "task_config.json").read_text())
    before = study._policy_identity(checkpoint, task)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if change in {"weight", "mean"}:
        saved["model"][change] += 1
    elif change == "horizon":
        saved["policy_config"]["action_horizon"] = 8
    elif change == "controller":
        saved["config"]["pose_ik"] = "solver"
    else:
        task["physics"] = "newton"
    torch.save(saved, checkpoint)
    assert study._policy_identity(checkpoint, task) != before
