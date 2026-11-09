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

"""Read-only execution provenance without importing any simulation runtime."""

from __future__ import annotations

import csv
import io
import platform
import shutil
import subprocess
import sys
from typing import Any

import torch

__all__: list[str] = []


def _driver_inventory() -> dict[str, Any]:
    """Query physical GPUs independently of Torch's logical device numbering."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"status": "unavailable", "reason": "nvidia-smi not found"}
    try:
        process = subprocess.run(
            [
                executable,
                "--query-gpu=index,uuid,name,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if process.returncode:
            return {
                "status": "error",
                "reason": "nonzero_exit",
                "returncode": process.returncode,
                "stderr": process.stderr[:1024],
            }
        if len(process.stdout) > 65536:
            return {"status": "error", "reason": "inventory_too_large"}
        rows = list(csv.reader(io.StringIO(process.stdout), skipinitialspace=True))
        if (
            not rows
            or len(rows) > 128
            or any(
                len(row) != 4 or any(not value.strip() for value in row) for row in rows
            )
        ):
            return {"status": "error", "reason": "invalid_inventory"}
        return {
            "status": "available",
            "physical_gpus": [
                dict(
                    zip(
                        ("index", "uuid", "name", "driver_version"),
                        (value.strip() for value in row),
                    )
                )
                for row in rows
            ],
        }
    except subprocess.TimeoutExpired:
        return {"status": "error", "reason": "timeout", "timeout_seconds": 2.0}
    except (OSError, UnicodeError) as error:
        return {
            "status": "error",
            "reason": type(error).__name__,
            "message": str(error)[:1024],
        }


def _training_runtime_identity(model: torch.nn.Module) -> dict[str, Any]:
    """Describe actual parameter placement without moving tensors or consuming RNG.

    CUDA driver discovery is best effort and limited to two seconds. The driver
    inventory uses physical GPU identifiers; it is deliberately not equated with
    Torch logical indices, which may be remapped by the process environment.
    """
    placements = sorted(
        {
            (str(parameter.device), str(parameter.dtype))
            for parameter in model.parameters()
        }
    )
    cuda_devices = sorted(
        {device for device, _ in placements if torch.device(device).type == "cuda"}
    )
    devices = []
    for device in cuda_devices:
        try:
            devices.append(
                {
                    "device": device,
                    "status": "available",
                    "name": torch.cuda.get_device_name(torch.device(device)),
                }
            )
        except (RuntimeError, AssertionError, ValueError, OSError) as error:
            devices.append(
                {
                    "device": device,
                    "status": "error",
                    "reason": type(error).__name__,
                    "message": str(error)[:1024],
                }
            )
    result = {
        "schema_version": 1,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "parameter_placements": [
            {"device": device, "dtype": dtype} for device, dtype in placements
        ],
        "parameter_placement_status": "available" if placements else "no_parameters",
        "cuda": {
            "used": bool(cuda_devices),
            "runtime_version": torch.version.cuda,
            "devices": devices,
            "driver_inventory": (
                _driver_inventory() if cuda_devices else {"status": "not_used"}
            ),
        },
    }
    warp = sys.modules.get("warp")
    result["warp"] = (
        {
            "status": "already_loaded",
            "version": str(getattr(warp, "__version__", "unknown")),
        }
        if warp is not None
        else {"status": "not_loaded"}
    )
    return result
