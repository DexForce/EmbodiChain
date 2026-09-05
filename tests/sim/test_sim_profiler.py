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

import types

import pytest
import torch

from embodichain.lab.sim import Profiler, ProfilerCfg, SimulationManager

pytestmark = pytest.mark.no_sim


class _WorldUpdateProbe:
    """Minimal world interface used by ``SimulationManager.update``."""

    def __init__(self) -> None:
        self.update_calls = 0

    def update(self, physics_dt: float) -> None:
        del physics_dt
        self.update_calls += 1


def _make_sim_update_probe(profiler: Profiler) -> SimulationManager:
    """Build a manager probe without starting the simulation backend."""

    sim = object.__new__(SimulationManager)
    sim.profiler = profiler
    sim.device = torch.device("cpu")
    sim._is_initialized_gpu_physics = False
    sim._world = _WorldUpdateProbe()
    sim._window_record_state = None
    sim._visualization_runtime = None
    sim._visualization_sim_step = 0
    sim._visualization_sim_time = 0.0
    sim.sim_config = types.SimpleNamespace(
        physics_dt=0.01,
        visualization=types.SimpleNamespace(backend="none"),
    )
    return sim


def test_standalone_sim_update_is_profile_root() -> None:
    profiler = Profiler(
        ProfilerCfg(enable_time=True, warmup_steps=0), torch.device("cpu")
    )
    sim = _make_sim_update_probe(profiler)

    sim.update(step=2)

    assert "sim_update" in profiler._stats
    assert "sim_update.gpu_physics_check" in profiler._stats
    assert "sim_update.physics_steps" in profiler._stats
    assert profiler._stats["sim_update.physics_steps.gizmo_update"].n == 2
    assert profiler._stats["sim_update.physics_steps.world_update"].n == 2


def test_sim_update_composes_with_env_profile_hierarchy() -> None:
    profiler = Profiler(
        ProfilerCfg(enable_time=True, warmup_steps=0), torch.device("cpu")
    )
    sim = _make_sim_update_probe(profiler)

    with profiler.section("step", is_root=True):
        with profiler.section("sim_update"):
            sim.update(step=1)

    assert "step.sim_update.gpu_physics_check" in profiler._stats
    assert "step.sim_update.physics_steps.gizmo_update" in profiler._stats
    assert "step.sim_update.physics_steps.world_update" in profiler._stats
    assert "step.sim_update.sim_update" not in profiler._stats
    assert "sim_update" not in profiler._stats


def test_visualization_capture_is_profiled_per_sim_step() -> None:
    profiler = Profiler(
        ProfilerCfg(enable_time=True, warmup_steps=0), torch.device("cpu")
    )
    sim = _make_sim_update_probe(profiler)
    sim.sim_config.visualization.backend = "viser"
    camera_capture_flags: list[bool] = []
    sim.capture_visualization_safely = lambda *, capture_camera_images: (
        camera_capture_flags.append(capture_camera_images)
    )

    sim.update(step=2)

    stats = profiler._stats["sim_update.physics_steps.visualization_capture"]
    assert stats.n == 2
    assert camera_capture_flags == [False, True]
