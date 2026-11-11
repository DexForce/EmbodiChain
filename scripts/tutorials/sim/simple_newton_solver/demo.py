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
from dexsim.engine.newton_physics import ProviderNewtonSolverCfg

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    CollisionPropertiesCfg,
    NewtonPhysicsCfg,
    RigidBodyPhysicsCfg,
)
from embodichain.lab.visualization import VisualizationCfg
from solver import SimpleSolverProvider


def _save_preview(
    positions: list[torch.Tensor], reset_position: torch.Tensor, path: Path
) -> None:
    """Plot the measured functional trajectory and selected-arena reset."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = torch.stack(positions).detach().cpu().numpy()
    reset_values = reset_position.detach().cpu().numpy()
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for arena_id, color in enumerate(("#2563eb", "#ea580c")):
        axis.plot(
            range(len(values)),
            values[:, arena_id],
            marker="o",
            color=color,
            label=f"arena_{arena_id}",
        )
        axis.plot(
            (len(values) - 1, len(values)),
            (values[-1, arena_id], reset_values[arena_id]),
            linestyle="--",
            marker="o",
            color=color,
        )
    axis.set_xticks(range(len(values) + 1))
    axis.set_xticklabels([*(str(step) for step in range(len(values))), "reset"])
    axis.set_xlabel("Functional transition")
    axis.set_ylabel("Joint position (m)")
    axis.set_title("Spring-slider tutorial: rollout and selected-arena reset")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run(device: str, preview_path: Path | None = None) -> dict[str, object]:
    """Demonstrate direct provider creation, parity, gradients and partial reset."""
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
            solver_cfg=ProviderNewtonSolverCfg(
                provider_factory=SimpleSolverProvider,
                options={"stiffness": 2.0, "damping": 0.2},
            ),
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
        positions = [initial.physical["joint_q"]]
        for _ in range(6):
            trajectory = scene.functional_step(trajectory, force, publish=False)
            positions.append(trajectory.physical["joint_q"])
        saved = trajectory.physical["joint_q"].clone()
        published = scene.functional_step(trajectory, force, publish=True)
        positions.append(published.physical["joint_q"])
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
        if preview_path is not None:
            _save_preview(positions, reset.physical["joint_q"], preview_path)
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
        result = {
            "device": device,
            "native_functional_parity": True,
            "published_position": published.physical["joint_q"].detach().tolist(),
            "partial_reset_position": reset.physical["joint_q"].detach().tolist(),
            "force_gradient": grad_force.tolist(),
            "force_scale_gradient": grad_scale.item(),
        }
        if preview_path is not None:
            result["preview_path"] = str(preview_path)
        return result
    finally:
        sim.destroy(exit_process=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    parser.add_argument(
        "--preview-path", type=Path, help="save the rollout and reset plot as PNG"
    )
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.device, args.preview_path), indent=2))
    finally:
        SimulationManager.flush_cleanup_queue()
