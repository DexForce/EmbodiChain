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
    return module.select_default_renderer, sim_module.cfg


@pytest.mark.parametrize("gpu_id", [0, 2])
def test_auto_defaults_to_fast_rt_without_hardware_detection(
    monkeypatch, renderer_selector, gpu_id: int
) -> None:
    select_default_renderer, cfg = renderer_selector
    cuda_probe = Mock(
        side_effect=AssertionError("Renderer selection must not query CUDA")
    )
    monkeypatch.setattr(torch.cuda, "is_available", cuda_probe)
    monkeypatch.setattr(torch.cuda, "get_device_name", cuda_probe)

    assert select_default_renderer(gpu_id) == "fast-rt"
    cuda_probe.assert_not_called()
    assert cfg.DEFAULT_RENDERER == "auto"


@pytest.mark.parametrize("renderer", ["no-render", "hybrid", "fast-rt", "rt"])
def test_explicit_global_renderer_takes_precedence(
    renderer_selector, renderer: str
) -> None:
    select_default_renderer, cfg = renderer_selector
    cfg.DEFAULT_RENDERER = renderer

    assert select_default_renderer() == renderer
