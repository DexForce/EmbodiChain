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

"""Reproducibility and raw-result artifacts for planner benchmarks."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from .config import SuiteCfg, suite_to_dict
from .models import BenchmarkCase, TrialRecord

__all__ = [
    "TrialJsonlWriter",
    "create_run_directory",
    "environment_metadata",
    "write_case_manifest",
    "write_json",
    "write_resolved_suite",
]


def create_run_directory(output_root: str | Path, suite_name: str) -> Path:
    """Create one timestamped benchmark run directory."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(output_root) / suite_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _package_version(package: str) -> str | None:
    """Return an installed distribution version when available."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    """Return the current repository commit without failing outside git."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None


def environment_metadata() -> dict[str, object]:
    """Collect software and hardware metadata needed to interpret results."""
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor() or None,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu_name,
        "curobo": _package_version("nvidia-curobo"),
        "dexsim_engine": _package_version("dexsim-engine"),
        "embodichain": _package_version("embodichain"),
    }


def write_json(path: str | Path, value: object) -> Path:
    """Write one UTF-8 JSON artifact with stable formatting."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        + "\n",
        encoding="utf-8",
    )
    return output


def write_resolved_suite(path: str | Path, suite: SuiteCfg) -> Path:
    """Write the fully resolved suite as YAML."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(suite_to_dict(suite), sort_keys=False), encoding="utf-8"
    )
    return output


def _case_to_dict(case: BenchmarkCase) -> dict[str, Any]:
    """Serialize a fixed case without losing tensor numeric values."""
    return {
        "suite_version": case.suite_version,
        "track": case.track,
        "scenario_id": case.scenario_id,
        "case_id": case.case_id,
        "seed": case.seed,
        "batch_size": case.batch_size,
        "num_waypoints": case.num_waypoints,
        "path_shape": case.path_shape,
        "start_state_bin": case.start_state_bin,
        "robot_id": case.robot_id,
        "skill_id": case.skill_id,
        "object_id": case.object_id,
        "task_difficulty": case.task_difficulty,
        "primary_success": case.primary_success,
        "start_qpos": case.start_qpos.detach().cpu().tolist(),
        "full_start_qpos": (
            None
            if case.full_start_qpos is None
            else case.full_start_qpos.detach().cpu().tolist()
        ),
        "target_waypoints": case.target_waypoints.detach().cpu().tolist(),
        "case_parameters": _to_json_value(case.case_parameters),
        "validity_evidence": {
            "method": (
                "reference_qpos_fk"
                if case.skill_id == "N/A"
                else (
                    "joint_limits_and_reference_fk"
                    if case.skill_id == "move_joints"
                    else "independent_sequential_ik"
                )
            ),
            "reference_qpos": case.reference_qpos.detach().cpu().tolist(),
        },
    }


def _to_json_value(value: object) -> object:
    """Recursively preserve tensors and numeric case configuration values."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]
    return value


def write_case_manifest(path: str | Path, cases: list[BenchmarkCase]) -> Path:
    """Write the algorithm-independent case manifest."""
    return write_json(
        path,
        {
            "case_schema_version": 2,
            "cases": [_case_to_dict(case) for case in cases],
        },
    )


class TrialJsonlWriter:
    """Append numeric raw trial records to one JSONL artifact."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def append(self, record: TrialRecord) -> None:
        """Append one record and flush it immediately for recoverability."""
        with self.path.open("a", encoding="utf-8") as file:
            file.write(
                json.dumps(record.to_dict(), ensure_ascii=False, default=str) + "\n"
            )
