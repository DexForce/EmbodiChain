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

"""Online rigid boundaries, mass ownership and selected fluid resets."""

from __future__ import annotations

import pytest
from pathlib import Path
import subprocess
import sys
import torch
import trimesh

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    FluidObjectCfg,
    FluidSpawnCfg,
    FluidRenderCfg,
    NewtonPhysicsCfg,
    RenderCfg,
    RigidObjectCfg,
)
from embodichain.lab.sim.shapes import CubeCfg, MeshCfg, MeshCollisionCfg

pytestmark = [pytest.mark.gpu, pytest.mark.subprocess_sim]


class _FluidOnlineRuntime:
    def test_live_dynamic_boundary_and_selected_reset(self):
        from dexsim.engine.newton_physics.backend_registry import get_newton_backend

        mesh = trimesh.creation.box(extents=[0.1] * 3)
        sim = SimulationManager(
            SimulationManagerCfg(
                headless=True,
                device="cuda",
                num_envs=2,
                arena_space=0,
                physics_dt=0.001,
                render_cfg=RenderCfg(renderer="no-render"),
                physics_cfg=NewtonPhysicsCfg(
                    num_substeps=1,
                    collision_cfg=None,
                    solver_cfg={
                        "solver_type": "sph",
                        "sph_method": "dfsph",
                        "rigid_solver_cfg": {"rne_mode": "disabled"},
                        "rigid_solver_substeps": 2,
                        "particle_support_radius": 0.0225,
                        "volume_map_resolution": 16,
                        "volume_map_quadrature_samples": 2,
                    },
                ),
            )
        )
        body = fluid = None
        try:
            body = sim.add_rigid_object(
                RigidObjectCfg.from_dict(
                    {
                        "uid": "boundary",
                        "shape": CubeCfg(size=[0.1] * 3).to_dict(),
                        "particle_collision_shape": MeshCfg(
                            vertices=mesh.vertices.tolist(),
                            triangles=mesh.faces.tolist(),
                            collision=MeshCollisionCfg(
                                approximation="sdf",
                                sdf_resolution=32,
                                sdf_narrow_band_range=(-0.05, 0.05),
                                sdf_padding=0.02,
                            ),
                        ).to_dict(),
                        "body_type": "dynamic",
                        "init_pos": [0, 0, 1],
                        "attrs": {"mass_props": {"mass": 0.3}},
                    }
                )
            )
            fluid = sim.add_fluid_object(
                FluidObjectCfg(
                    uid="water",
                    particle_spacing=0.01,
                    particle_radius=0.003,
                    spawn=FluidSpawnCfg(
                        kind="points",
                        positions=[
                            [0.055, -0.005, 0.995],
                            [0.055, 0.005, 0.995],
                            [0.055, -0.005, 1.005],
                            [0.055, 0.005, 1.005],
                        ],
                    ),
                    render=FluidRenderCfg(mode="none"),
                )
            )
            sim.prepare()
            backend = get_newton_backend(sim._world)
            assert backend.solver.sph_solver.sdf_shape_inventory.shape_count == 2
            torch.testing.assert_close(
                torch.as_tensor(backend.model.body_mass.numpy()),
                torch.tensor([0.3, 0.3]),
            )
            torch.testing.assert_close(
                torch.as_tensor(backend.model.body_inertia.numpy())[
                    :, [0, 1, 2], [0, 1, 2]
                ],
                torch.full((2, 3), 0.0005),
                rtol=0.001,
                atol=1e-7,
            )
            body.set_velocity(
                lin_vel=torch.tensor([[0.5, 0, 0], [0.5, 0, 0]], device=sim.device)
            )
            sim.update(step=200, render_final_step=False)
            assert backend.sim_time == pytest.approx(0.2)
            assert sim.physics.cuda_graph_status == "captured"
            pose = body.get_local_pose()
            assert bool((pose[:, 0] > 0.095).all())
            assert bool((fluid.data.particle_pos_w[..., 0] > 0.12).all())
            # Water has no reaction force on the prescribed initial rigid momentum.
            torch.testing.assert_close(
                pose[:, 0],
                torch.full((2,), 0.1, device=sim.device),
                rtol=0.01,
                atol=0.001,
            )
            particles = fluid.data.particle_state_w.clone()
            fluid.reset([0])
            torch.testing.assert_close(
                fluid.data.particle_state_w[1], particles[1], rtol=0, atol=0
            )
            torch.testing.assert_close(body.get_local_pose(), pose, rtol=0, atol=0)
            sim.reset_objects_state(env_ids=[0])
            torch.testing.assert_close(
                body.get_local_pose()[1], pose[1], rtol=0, atol=0
            )
            sim.update(step=1, render_final_step=False)
            assert bool(torch.isfinite(fluid.data.particle_state_w).all())
        finally:
            body = fluid = None
            sim.destroy(exit_process=False)
            SimulationManager.flush_cleanup_queue()

    def test_mimic_follower_uses_inner_rigid_solver(self):
        from embodichain_robots import FrankaPandaCfg

        sim = SimulationManager(
            SimulationManagerCfg(
                headless=True,
                device="cuda",
                num_envs=1,
                physics_dt=0.001,
                render_cfg=RenderCfg(renderer="no-render"),
                physics_cfg=NewtonPhysicsCfg(
                    num_substeps=1,
                    collision_cfg=None,
                    solver_cfg={
                        "solver_type": "sph",
                        "rigid_solver_cfg": {"rne_mode": "disabled"},
                        "rigid_solver_substeps": 2,
                        "volume_map_boundary_enabled": False,
                        "sdf_boundary_enabled": False,
                    },
                ),
            )
        )
        robot = fluid = None
        try:
            robot = sim.add_robot(FrankaPandaCfg.from_dict({}))
            fluid = sim.add_fluid_object(
                FluidObjectCfg(
                    uid="water",
                    particle_spacing=0.01,
                    init_pos=(2, 0, 2),
                    spawn=FluidSpawnCfg(kind="box", size=(0.02,) * 3),
                    render=FluidRenderCfg(mode="none"),
                )
            )
            sim.prepare()
            assert robot._newton_mimic_compliance_configured
            joint_ids = robot.get_joint_ids("eef", remove_mimic=True)
            robot.set_qpos(
                torch.zeros((1, len(joint_ids)), device=sim.device), joint_ids=joint_ids
            )
            sim.update(step=500, render_final_step=False)
            fingers = robot.get_qpos(name="eef")
            assert bool((fingers.abs() < 0.004).all()), fingers
            assert float((fingers[:, 0] - fingers[:, 1]).abs().max()) < 0.002
        finally:
            robot = fluid = None
            sim.destroy(exit_process=False)
            SimulationManager.flush_cleanup_queue()

    def test_normal_gym_partial_reset_does_not_advance_peer(self):
        from embodichain.lab.gym.utils.gym_utils import config_to_cfg
        from embodichain.lab.gym.utils.registration import discover_task_packages, make
        from embodichain.utils.utility import load_config

        task = (
            Path(__file__).resolve().parents[3]
            / "embodichain_tasks/configs/tasks/manipulation/tableware/pour_cup/task.franka.fluid.yaml"
        )
        discover_task_packages()
        declaration = load_config(task)
        cfg = config_to_cfg(declaration, source_path=task)
        cfg.num_envs = 2
        cfg.enable_sensor = False
        cfg.sim_cfg.render_cfg = RenderCfg(renderer="no-render")
        cfg.sim_cfg.headless = True
        cfg.sim_steps_per_control = 1
        cfg.events.record_camera = None
        cfg.sim_cfg.physics_cfg.solver_cfg.update(
            volume_map_boundary_enabled=False, sdf_boundary_enabled=False
        )
        for obj in cfg.background + cfg.rigid_object:
            for shape in (obj.shape, obj.particle_collision_shape):
                collision = getattr(shape, "collision", None)
                if collision is not None and collision.approximation == "sdf":
                    collision.sdf_resolution = 16
        fluid_cfg = cfg.fluid_object[0]
        fluid_cfg.particle_spacing = 0.01
        fluid_cfg.particle_radius = 0.003
        fluid_cfg.spawn = FluidSpawnCfg(
            kind="box", size=(0.02,) * 3, reference_uid="cup"
        )
        fluid_cfg.render = FluidRenderCfg(mode="none")
        env = make(declaration["id"], cfg=cfg)
        try:
            env.reset(seed=0)
            action = env.robot.get_qpos()[:, env.active_joint_ids].clone()
            for _ in range(3):
                env.step(action)
            fluid = env.sim.get_fluid_object("water")
            before_water = fluid.data.particle_state_w[1].clone()
            before_robot = env.robot.get_qpos()[1].clone()
            before_cup = env.sim.get_rigid_object("cup").get_local_pose()[1].clone()
            env.reset(options={"reset_ids": [0], "save_data": False})
            torch.testing.assert_close(
                fluid.data.particle_state_w[1], before_water, rtol=0, atol=0
            )
            torch.testing.assert_close(
                env.robot.get_qpos()[1], before_robot, rtol=0, atol=0
            )
            torch.testing.assert_close(
                env.sim.get_rigid_object("cup").get_local_pose()[1],
                before_cup,
                rtol=0,
                atol=0,
            )
            assert env.elapsed_steps.tolist() == [0, 3]
        finally:
            env.close(exit_process=False)
            del env
            SimulationManager.flush_cleanup_queue()


def test_online_fluid_boundaries_and_mimic_reset_in_fresh_process():
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    worker = _FluidOnlineRuntime()
    worker.test_live_dynamic_boundary_and_selected_reset()
    worker.test_mimic_follower_uses_inner_rigid_solver()
    worker.test_normal_gym_partial_reset_does_not_advance_peer()
