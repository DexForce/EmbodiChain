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

"""Record the inputs and execution environment of one motion-training invocation."""

from __future__ import annotations

import csv
from datetime import datetime
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import torch

__all__ = ["write_run_metadata"]


def _command(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=3, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _software() -> dict[str, Any]:
    modules = {}
    for name in (
        "embodichain",
        "dexsim",
        "newton",
        "mujoco",
        "mujoco_warp",
        "warp",
        "torch",
        "numpy",
        "tensordict",
    ):
        module = sys.modules.get(name)
        modules[name] = {
            "loaded": module is not None,
            "version": (
                str(getattr(module, "__version__", "unknown")) if module else None
            ),
            "file": getattr(module, "__file__", None),
        }
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cuda_runtime": torch.version.cuda,
        "modules": modules,
    }


def _hardware(device: torch.device) -> dict[str, Any]:
    cpu = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu,
        )
    memory_bytes = None
    if hasattr(os, "sysconf"):
        try:
            memory_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (OSError, ValueError):
            pass
    result: dict[str, Any] = {
        "host": platform.node(),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "ram_bytes": memory_bytes,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result["selected_gpu"] = {
            "logical_device": str(device),
            "name": properties.name,
            "memory_bytes": properties.total_memory,
            "uuid": str(getattr(properties, "uuid", "unknown")),
        }
        inventory = _command(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ]
        )
        result["physical_gpus"] = (
            [
                dict(zip(("index", "uuid", "name", "memory_mib", "driver"), row))
                for row in csv.reader(io.StringIO(inventory), skipinitialspace=True)
            ]
            if inventory
            else None
        )
    return result


def _source(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(root),
        "git_available": (root / ".git").exists(),
    }
    if result["git_available"]:
        result.update(
            head=_command(["git", "-C", str(root), "rev-parse", "HEAD"]),
            branch=_command(["git", "-C", str(root), "branch", "--show-current"]),
            working_tree_status=_command(["git", "-C", str(root), "status", "--short"]),
        )
    return result


def write_run_metadata(
    output: Path,
    *,
    invocation_started_at: str,
    config: dict[str, Any],
    gym_config: dict[str, Any],
    motion: dict[str, Any],
    motion_path: Path,
    device: torch.device,
    resume_checkpoint: Path | None,
) -> Path:
    """Save one invocation's actual inputs and loaded runtime before training.

    Args:
        output: Existing training output directory.
        invocation_started_at: UTC ISO timestamp shared with metric records.
        config: Resolved training configuration, including budget and seed.
        gym_config: Resolved physical-environment configuration.
        motion: Selected motion names, lengths and sample rate.
        motion_path: Resolved archive used by the physical environment.
        device: Actual training device.
        resume_checkpoint: Successfully restored checkpoint, or None for fresh runs.

    Returns:
        Path to the new metadata file. Resumed invocations retain earlier files.
    """
    timestamp = datetime.fromisoformat(invocation_started_at).strftime(
        "%Y%m%dT%H%M%S.%fZ"
    )
    path = output / f"environment-{timestamp}.json"
    result = {
        "schema": 1,
        "invocation_started_at": invocation_started_at,
        "kind": config["kind"],
        "seed": config["seed"],
        "resume_checkpoint": (
            str(resume_checkpoint.resolve()) if resume_checkpoint else None
        ),
        "config": config,
        "gym_config": gym_config,
        "dataset": {
            **motion,
            "path": str(motion_path.resolve()),
            "total_frames": sum(motion["lengths"]),
        },
        "hardware": _hardware(device),
        "software": _software(),
        "precision": {
            "network_autocast": "bfloat16" if config["amp"] else "float32",
            "state": "float32",
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        },
        "source": _source(Path(__file__).resolve().parents[3]),
        "runtime_environment": {
            name: os.environ.get(name)
            for name in (
                "SOURCE_ROOT",
                "REFERENCE_ROOT",
                "EMBODICHAIN_DATA_ROOT",
                "DEXSIM_NEWTON_ARTIFACT_MODE",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
    }
    with path.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return path
