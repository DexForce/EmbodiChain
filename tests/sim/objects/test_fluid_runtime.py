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

"""Small CUDA integration test of the public fluid lifecycle."""

from __future__ import annotations

import json
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


class TestFluidRuntime:
    def test_prepare_write_reset_and_repeat(self, tmp_path: Path) -> None:
        result_path = tmp_path / "lifecycle.json"
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), str(result_path)],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(result_path.read_text()) == {"num_particles": 64}


def _run_fluid_lifecycle() -> int:
    """Run the complete NoRender lifecycle in a fresh native process."""
    from dexsim.engine.newton_physics import SPHSolverCfg
    from embodichain.lab.sim.cfg import RigidObjectCfg
    from embodichain.lab.sim.shapes import CubeCfg

    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device="cuda",
            num_envs=1,
            physics_dt=0.001,
            render_cfg=RenderCfg(renderer="no-render"),
            physics_cfg=NewtonPhysicsCfg(
                num_substeps=2,
                collision_cfg=None,
                solver_cfg=SPHSolverCfg(
                    sph_method="dfsph",
                    particle_support_radius=0.0225,
                    volume_map_boundary_enabled=False,
                ),
            ),
        )
    )
    fluid = rigid = None
    try:
        rigid = sim.add_rigid_object(
            RigidObjectCfg(
                uid="boundary", body_type="kinematic", shape=CubeCfg(size=[0.1] * 3)
            )
        )
        fluid = sim.add_fluid_object(
            FluidObjectCfg(
                uid="water",
                particle_spacing=0.01,
                init_pos=(0.0, 0.0, 1.0),
                spawn=FluidSpawnCfg(kind="box", size=(0.04, 0.04, 0.04)),
                render=FluidRenderCfg(mode="none"),
            )
        )
        with pytest.raises(RuntimeError, match="prepare"):
            _ = fluid.data
        sim.prepare()
        assert sim.get_fluid_object("water") is fluid
        assert sim.get_asset("water") is fluid
        assert fluid.num_particles == 64
        original = fluid.data.default_particle_state_w
        sim.update(step=3, render_final_step=False)
        first = fluid.data.particle_state_w
        assert torch.isfinite(first).all()
        assert first[..., 2].mean() < original[..., 2].mean()
        sim.reset_objects_state()
        torch.testing.assert_close(fluid.data.particle_state_w, original)
        sim.update(step=3, render_final_step=False)
        torch.testing.assert_close(
            fluid.data.particle_state_w, first, rtol=1e-5, atol=1e-6
        )
        with pytest.raises(NotImplementedError, match="before prepare"):
            sim.add_fluid_object(FluidObjectCfg(uid="late"))
        return fluid.num_particles
    finally:
        fluid = rigid = None
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(
        json.dumps({"num_particles": _run_fluid_lifecycle()}) + "\n"
    )
