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
"""Run ordinary and functional stepping through public EmbodiChain/Scene APIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    CollisionPropertiesCfg,
    NewtonPhysicsCfg,
    RigidBodyPhysicsCfg,
)
from embodichain.lab.visualization import VisualizationCfg


def run(device: str) -> dict[str, object]:
    """Demonstrate discovery, parity, gradients, publication and partial reset."""
    config = SimulationManagerCfg(
        headless=True,
        visualization=VisualizationCfg(backend="none"),
        num_envs=2,
        enable_entity_gizmo=False,
        startup_summary="off",
        physics_cfg=NewtonPhysicsCfg(
            device=device,
            gravity=(0.0, 0.0, 0.0),
            physics_dt=0.02,
            num_substeps=2,
            requires_grad=True,
            use_cuda_graph=False,
            collision_cfg=None,
            solver_cfg={"solver_type": "simple", "stiffness": 2.0, "damping": 0.2},
        ),
    )
    sim = SimulationManager(config, requires_native_renderer=False)
    try:
        slider = sim.add_articulation(
            ArticulationCfg(
                uid="slider",
                fpath=str(Path(__file__).with_name("slider.urdf")),
                asset_physics_mode="overlay",
                attrs=RigidBodyPhysicsCfg(
                    collision_props=CollisionPropertiesCfg(collision_enabled=False),
                ),
            )
        )
        sim.prepare()
        scene = sim.spawn_result
        initial = scene.create_functional_state()
        params = scene.functional_parameters()
        force = torch.tensor([4.0, 6.0], device=device, requires_grad=True)

        predicted = scene.functional_step(initial, force, publish=False)
        torch.testing.assert_close(
            slider.get_qpos().flatten(), initial.physical["joint_q"]
        )
        slider.set_qf(force.detach().reshape(2, 1))
        sim.update(step=1)
        torch.testing.assert_close(
            slider.get_qpos().flatten(), predicted.physical["joint_q"]
        )

        trajectory = initial
        for _ in range(6):
            trajectory = scene.functional_step(trajectory, force, publish=False)
        saved = trajectory.physical["joint_q"].clone()
        published = scene.functional_step(trajectory, force, publish=True)
        torch.testing.assert_close(
            slider.get_qpos().flatten(), published.physical["joint_q"]
        )
        torch.testing.assert_close(trajectory.physical["joint_q"], saved)

        reset = scene.reset_functional_state(published, arenas=["arena_0"])
        torch.testing.assert_close(
            reset.physical["joint_q"][0], initial.physical["joint_q"][0]
        )
        torch.testing.assert_close(
            reset.physical["joint_q"][1], published.physical["joint_q"][1]
        )
        loss = (
            trajectory.physical["joint_q"].square().sum()
            + reset.physical["joint_q"].square().sum()
        )
        grad_force, grad_scale = torch.autograd.grad(
            loss, (force, params["force_scale"])
        )
        assert torch.isfinite(grad_force).all() and (grad_force.abs() > 0).all()
        assert torch.isfinite(grad_scale) and grad_scale.abs() > 0
        slider.reset(env_ids=[0])
        torch.testing.assert_close(
            slider.get_qpos()[0].flatten(), initial.physical["joint_q"][:1]
        )
        return {
            "device": device,
            "native_functional_parity": True,
            "published_position": published.physical["joint_q"].detach().tolist(),
            "partial_reset_position": reset.physical["joint_q"].detach().tolist(),
            "force_gradient": grad_force.tolist(),
            "force_scale_gradient": grad_scale.item(),
        }
    finally:
        sim.destroy(exit_process=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.device), indent=2))
    finally:
        SimulationManager.flush_cleanup_queue()
