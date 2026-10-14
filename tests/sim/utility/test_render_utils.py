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

from __future__ import annotations

from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytestmark = pytest.mark.no_sim


@pytest.fixture
def renderer_selector(monkeypatch):
    # Load the pure selector without importing DexSim's native runtime.
    sim_module = ModuleType("embodichain.lab.sim")
    sim_module.cfg = SimpleNamespace(DEFAULT_RENDERER="auto")
    monkeypatch.setitem(sys.modules, "embodichain.lab.sim", sim_module)
    path = (
        Path(__file__).resolve().parents[3]
        / "embodichain/lab/sim/utility/render_utils.py"
    )
    spec = importlib.util.spec_from_file_location("renderer_selector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, sim_module.cfg


@pytest.mark.parametrize(
    ("available", "device_name", "expected"),
    [
        (True, "NVIDIA GeForce RTX 4090", "hybrid"),
        (True, "NVIDIA RTX A6000", "hybrid"),
        (True, "NVIDIA A100", "fast-rt"),
        (True, "NVIDIA H200", "fast-rt"),
        (True, "Unknown GPU", "hybrid"),
        (False, None, "hybrid"),
    ],
    ids=["consumer", "workstation", "ampere", "hopper", "unknown", "unavailable"],
)
def test_standalone_preserves_hardware_selection(
    monkeypatch, renderer_selector, available, device_name, expected
) -> None:
    module, cfg = renderer_selector
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    name_probe = Mock(return_value=device_name)
    monkeypatch.setattr(torch.cuda, "get_device_name", name_probe)

    assert module.select_default_renderer(2) == expected
    if available:
        name_probe.assert_called_once_with(2)
    else:
        name_probe.assert_not_called()
    assert cfg.DEFAULT_RENDERER == "auto"


def test_standalone_falls_back_when_device_query_fails(monkeypatch, renderer_selector):
    module, _ = renderer_selector
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", Mock(side_effect=RuntimeError))
    assert module.select_default_renderer() == "hybrid"


@pytest.mark.parametrize("gpu_id", [0, 2])
def test_task_defaults_to_fast_rt_without_hardware_detection(
    monkeypatch, renderer_selector, gpu_id: int
) -> None:
    module, cfg = renderer_selector
    hardware_probe = Mock(
        side_effect=AssertionError("Task default must not query CUDA")
    )
    monkeypatch.setattr(torch.cuda, "is_available", hardware_probe)
    monkeypatch.setattr(torch.cuda, "get_device_name", hardware_probe)

    with module._native_auto_renderer("fast-rt"):
        assert module.select_default_renderer(gpu_id) == "fast-rt"
    hardware_probe.assert_not_called()
    assert cfg.DEFAULT_RENDERER == "auto"


@pytest.mark.parametrize("renderer", ["no-render", "hybrid", "fast-rt", "rt"])
def test_explicit_global_renderer_takes_precedence(renderer_selector, renderer: str):
    module, cfg = renderer_selector
    cfg.DEFAULT_RENDERER = renderer

    assert module.select_default_renderer() == renderer
    with module._native_auto_renderer("fast-rt"):
        assert module.select_default_renderer() == renderer


@pytest.mark.parametrize("failure", [False, True])
def test_task_default_is_restored_after_exit(monkeypatch, renderer_selector, failure):
    module, _ = renderer_selector
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda _: "NVIDIA GeForce RTX 4090"
    )

    outcome = (
        pytest.raises(RuntimeError, match="task failed") if failure else nullcontext()
    )
    with outcome:
        with module._native_auto_renderer("fast-rt"):
            assert module.select_default_renderer() == "fast-rt"
            if failure:
                raise RuntimeError("task failed")

    assert module.select_default_renderer() == "hybrid"


def test_nested_defaults_restore_the_outer_scope(renderer_selector):
    module, _ = renderer_selector
    with module._native_auto_renderer("fast-rt"):
        with module._native_auto_renderer("rt"):
            assert module.select_default_renderer() == "rt"
        assert module.select_default_renderer() == "fast-rt"
