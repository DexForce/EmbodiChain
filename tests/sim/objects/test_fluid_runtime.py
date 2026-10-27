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
    @pytest.mark.parametrize("num_envs", [1, 2])
    def test_prepare_write_reset_and_repeat(
        self, tmp_path: Path, num_envs: int
    ) -> None:
        if num_envs > 1:
            from dexsim.engine.newton_physics.solver_runtime import (
                ensure_solver_available,
            )

            solver = ensure_solver_available("sph")
            if not (
                getattr(solver, "supports_world_isolation", False)
                and getattr(solver, "supports_world_reset", False)
            ):
                pytest.skip("This DexSim SPH build supports only a single world.")

        def run_worlds(count: int) -> dict:
            result_path = tmp_path / f"lifecycle-{count}.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    str(result_path),
                    str(count),
                ],
                capture_output=True,
                text=True,
                timeout=180,
            )
            assert result.returncode == 0, result.stdout + result.stderr
            state = json.loads(result_path.read_text())
            assert state["num_particles"] == 64
            assert state["num_envs"] == count
            return state

        result = run_worlds(num_envs)
        if num_envs > 1:
            reference = run_worlds(1)
            torch.testing.assert_close(
                torch.tensor(result["first_velocity"]),
                torch.tensor(reference["first_velocity"]),
                rtol=1e-4,
                atol=1e-5,
            )


def _run_fluid_lifecycle(num_envs: int) -> dict[str, object]:
    """Run the complete NoRender lifecycle in a fresh native process."""
    from dexsim.engine.newton_physics import SPHSolverCfg
    from embodichain.lab.sim.cfg import RigidObjectCfg
    from embodichain.lab.sim.shapes import CubeCfg

    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device="cuda",
            num_envs=num_envs,
            # The two fluid volumes overlap in world space. SPH worlds must
            # remain isolated even when their particles are nearby.
            arena_space=0.02,
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
        assert original.shape == (num_envs, 64, 6)
        if num_envs > 1:
            assert torch.cdist(original[0, :, :3], original[1, :, :3]).min() < 0.0225
        sim.update(step=3, render_final_step=False)
        first = fluid.data.particle_state_w
        assert torch.isfinite(first).all()
        assert torch.all(first[..., 2].mean(dim=1) < original[..., 2].mean(dim=1))
        if num_envs > 1:
            torch.testing.assert_close(
                first[0, :, 3:], first[1, :, 3:], rtol=1e-4, atol=1e-5
            )
            fluid.reset(env_ids=[0])
            selected = fluid.data.particle_state_w
            torch.testing.assert_close(selected[0], original[0], rtol=0, atol=0)
            torch.testing.assert_close(selected[1:], first[1:], rtol=0, atol=0)
            sim.update(step=3, render_final_step=False)
            continued = fluid.data.particle_state_w
            torch.testing.assert_close(continued[0], first[0], rtol=1e-5, atol=1e-6)
            assert continued[1, :, 2].mean() < first[1, :, 2].mean()
        sim.reset_objects_state()
        torch.testing.assert_close(fluid.data.particle_state_w, original)
        sim.update(step=3, render_final_step=False)
        torch.testing.assert_close(
            fluid.data.particle_state_w, first, rtol=1e-5, atol=1e-6
        )
        with pytest.raises(NotImplementedError, match="before prepare"):
            sim.add_fluid_object(FluidObjectCfg(uid="late"))
        return {
            "num_particles": fluid.num_particles,
            "num_envs": num_envs,
            "first_velocity": first[0, :, 3:].cpu().tolist(),
        }
    finally:
        fluid = rigid = None
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(
        json.dumps(_run_fluid_lifecycle(int(sys.argv[2]))) + "\n"
    )
