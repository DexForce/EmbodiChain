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

"""Batch fluid declarations and row-local reset through the public manager."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import pytest
import torch
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    FluidObjectCfg,
    FluidSpawnCfg,
    FluidRenderCfg,
    NewtonPhysicsCfg,
    RenderCfg,
)

pytestmark = [pytest.mark.gpu, pytest.mark.subprocess_sim]


def _run_fluid_partial_reset():
    from dexsim.engine.newton_physics.solver_runtime import ensure_solver_available

    if not getattr(ensure_solver_available("sph"), "supports_world_reset", False):
        pytest.skip("requires world-isolated SPH build")
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device="cuda",
            num_envs=2,
            arena_space=1,
            physics_dt=0.001,
            render_cfg=RenderCfg(renderer="no-render"),
            physics_cfg=NewtonPhysicsCfg(
                num_substeps=1,
                collision_cfg=None,
                solver_cfg={
                    "solver_type": "dexuni",
                    "sph_cfg": {
                        "sph_method": "dfsph",
                        "volume_map_boundary_enabled": False,
                    },
                },
            ),
        )
    )
    fluid = None
    try:
        fluid = sim.add_fluid_object(
            FluidObjectCfg(
                uid="water",
                particle_spacing=0.01,
                init_pos=(0.0, 0.0, 1.0),
                spawn=FluidSpawnCfg(kind="box", size=(0.04,) * 3),
                render=FluidRenderCfg(mode="none"),
            )
        )
        sim.prepare()
        initial = fluid.data.default_particle_state_w
        offsets = torch.as_tensor(
            sim.arena_offsets, device=sim.device, dtype=torch.float32
        )
        torch.testing.assert_close(
            initial[:, :, :3].mean(1),
            offsets + torch.tensor([0.0, 0.0, 1.0], device=sim.device),
        )
        changed = initial[[1]].clone()
        changed[..., 3] = 1
        fluid.write_particle_state_to_sim(changed, [1])
        sim.update(step=3, render_final_step=False)
        before = fluid.data.particle_state_w
        density = sim.get_fluid_diagnostics("water")["particle_density"]
        fluid.reset([1])
        torch.testing.assert_close(
            fluid.data.particle_state_w[0], before[0], rtol=0, atol=0
        )
        torch.testing.assert_close(
            fluid.data.particle_state_w[1], initial[1], rtol=0, atol=0
        )
        torch.testing.assert_close(
            sim.get_fluid_diagnostics("water")["particle_density"][0],
            density[0],
            rtol=0,
            atol=0,
        )
        assert not sim.get_fluid_diagnostics("water")["particle_density"][1].any()
        sim.update(step=1, render_final_step=False)
        assert torch.isfinite(fluid.data.particle_state_w).all()
    finally:
        fluid = None
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


def test_fluid_partial_reset_preserves_other_world():
    from dexsim.engine.newton_physics.solver_runtime import ensure_solver_available

    if not getattr(ensure_solver_available("sph"), "supports_world_reset", False):
        pytest.skip("requires world-isolated SPH build")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _run_fluid_partial_reset()
