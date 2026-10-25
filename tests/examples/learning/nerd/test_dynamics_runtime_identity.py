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

"""Execution provenance is read-only, simulation-free and best effort."""

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from examples.learning.nerd._support import _runtime


def test_cpu_identity_reflects_actual_parameters_without_side_effects(monkeypatch):
    model = torch.nn.Linear(2, 1, dtype=torch.float64)
    rng = torch.get_rng_state().clone()
    modules = set(sys.modules)
    monkeypatch.delitem(sys.modules, "warp", raising=False)

    def unexpected(*args, **kwargs):
        pytest.fail("CPU identity must not query CUDA or nvidia-smi")

    monkeypatch.setattr(_runtime, "_driver_inventory", unexpected)
    monkeypatch.setattr(torch.cuda, "get_device_name", unexpected)
    result = _runtime._training_runtime_identity(model)
    assert result["parameter_placements"] == [
        {"device": "cpu", "dtype": "torch.float64"}
    ]
    assert result["cuda"]["used"] is False
    assert result["cuda"]["driver_inventory"] == {"status": "not_used"}
    assert result["torch_version"] == str(torch.__version__)
    assert result["python"] and result["platform"]
    assert result["warp"] == {"status": "not_loaded"}
    assert not any(
        name.split(".")[0] in {"warp", "dexsim", "newton"}
        for name in set(sys.modules) - modules
    )
    assert torch.equal(rng, torch.get_rng_state())
    assert json.loads(json.dumps(result)) == result


def test_empty_and_mixed_parameter_metadata_and_loaded_warp(monkeypatch):
    assert (
        _runtime._training_runtime_identity(torch.nn.Identity())[
            "parameter_placement_status"
        ]
        == "no_parameters"
    )
    model = torch.nn.Module()
    model.register_parameter(
        "a", torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
    )
    model.register_parameter(
        "b", torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    )
    monkeypatch.setitem(
        sys.modules, "warp", SimpleNamespace(__version__="test-version")
    )
    result = _runtime._training_runtime_identity(model)
    assert result["parameter_placements"] == [
        {"device": "cpu", "dtype": "torch.float32"},
        {"device": "cpu", "dtype": "torch.float64"},
    ]
    assert result["warp"] == {"status": "already_loaded", "version": "test-version"}


def _mock_cuda_model():
    # Metadata only: no CUDA allocations or initialization in these tests.
    return SimpleNamespace(
        parameters=lambda: iter(
            [
                SimpleNamespace(device=torch.device("cuda:1"), dtype=torch.float32),
                SimpleNamespace(device=torch.device("cuda:1"), dtype=torch.float32),
            ]
        )
    )


def test_cuda_inventory_is_bounded_and_keeps_physical_ids_separate(monkeypatch):
    monkeypatch.setattr(_runtime.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0, stdout='0, GPU-physical, "Example, GPU", 595.1\n', stderr=""
        )

    monkeypatch.setattr(_runtime.subprocess, "run", run)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "Logical GPU")
    result = _runtime._training_runtime_identity(_mock_cuda_model())
    assert result["cuda"]["devices"] == [
        {"device": "cuda:1", "status": "available", "name": "Logical GPU"}
    ]
    inventory = result["cuda"]["driver_inventory"]
    assert inventory == {
        "status": "available",
        "physical_gpus": [
            {
                "index": "0",
                "uuid": "GPU-physical",
                "name": "Example, GPU",
                "driver_version": "595.1",
            }
        ],
    }
    assert calls[0][0][1:] == [
        "--query-gpu=index,uuid,name,driver_version",
        "--format=csv,noheader,nounits",
    ]
    assert calls[0][1]["timeout"] == 2.0
    assert not calls[0][1].get("shell", False)


@pytest.mark.parametrize(
    "failure, expected",
    [
        ("missing", "unavailable"),
        ("timeout", "error"),
        ("exit", "error"),
        ("oserror", "error"),
        ("malformed", "error"),
        ("oversized", "error"),
    ],
)
def test_cuda_inventory_failure_does_not_abort_receipt(monkeypatch, failure, expected):
    monkeypatch.setattr(
        _runtime.shutil,
        "which",
        lambda name: None if failure == "missing" else "/mock/nvidia-smi",
    )

    def run(command, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if failure == "oserror":
            raise OSError("driver unavailable")
        return SimpleNamespace(
            returncode=1 if failure == "exit" else 0,
            stdout="x" * 65537 if failure == "oversized" else "bad,row",
            stderr="driver error",
        )

    monkeypatch.setattr(_runtime.subprocess, "run", run)

    def device_name(device):
        raise RuntimeError("device unavailable")

    monkeypatch.setattr(torch.cuda, "get_device_name", device_name)
    result = _runtime._training_runtime_identity(_mock_cuda_model())
    assert result["cuda"]["devices"][0]["status"] == "error"
    assert result["cuda"]["driver_inventory"]["status"] == expected
    json.dumps(result, allow_nan=False)


def test_fresh_process_identity_does_not_import_simulation():
    code = """
import sys
import torch
from examples.learning.nerd._support._runtime import _training_runtime_identity
_training_runtime_identity(torch.nn.Linear(1, 1))
assert not any(name.split('.')[0] in {'warp', 'dexsim', 'newton'} for name in sys.modules)
assert not any(name.startswith('embodichain.lab.sim') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)
