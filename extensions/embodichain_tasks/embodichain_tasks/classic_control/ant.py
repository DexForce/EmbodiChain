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

"""Ant control through the registered NeRD dynamics solver."""

from __future__ import annotations
from collections.abc import Sequence
from typing import Any, TYPE_CHECKING
import gymnasium as gym
import numpy as np
import torch
from embodichain.lab.gym.envs.differentiable_dynamics_env import (
    DifferentiableDynamicsEnv,
)
from embodichain.lab.gym.envs.embodied_env import EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim.cfg import NewtonPhysicsCfg, RenderCfg
from embodichain.lab.sim.sim_manager import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects.articulation import Articulation
from embodichain.lab.sim.physics.solvers._reference_assets import (
    make_reference_articulation,
    reference_profile,
)
from embodichain.lab.sim.physics.solvers._nerd_binding import NeRDBinding
from embodichain.compute.dynamics import quat_rotate, reference_task_values
from embodichain.learning.rl.env import register_learning_env

if TYPE_CHECKING:
    from dexsim.engine.newton_physics import FunctionalSolverState

__all__ = ["AntDynamicsEnv", "make_ant_dynamics_env"]


@register_env("AntDynamics-v1", max_episode_steps=1000, supports_rl=True)
class AntDynamicsEnv(DifferentiableDynamicsEnv):
    """Differentiable ant task with normalized effort actions.

    The pinned reference articulation is task-owned. A compatible exported model
    must be selected by physics_config.solver_cfg; model accuracy is determined
    by its independent validation, not by this task's registration.
    """

    def __init__(self, cfg: EmbodiedEnvCfg, **kwargs: Any) -> None:
        self.random_reset = (cfg.extensions or {}).get("random_reset", True)
        if not isinstance(self.random_reset, bool):
            raise TypeError("random_reset must be boolean")
        self._profile = reference_profile("ant")
        self._binding: NeRDBinding | None = None
        if cfg.robot is not None:
            raise ValueError(
                "AntDynamics owns its reference robot; configure robot: null"
            )
        if cfg.sim_cfg is not None and bool(cfg.sim_cfg.enable_ground_plane) != True:
            raise ValueError("AntDynamics requires enable_ground_plane: true")
        super().__init__(cfg, **kwargs)

    def _declare_robot(self, **kwargs: Any) -> Articulation:
        del kwargs
        return self.sim.add_articulation_descriptor(make_reference_articulation("ant"))

    def _setup_robot(self, **kwargs: Any) -> Articulation:
        del kwargs
        self._binding = NeRDBinding.from_metadata(
            dict(self.sim.spawn_result.dynamics_metadata()), "ant"
        )
        self._joint_ids = [
            self.robot.joint_names.index(name) for name in self._binding.joint_names
        ]
        self.active_joint_ids = list(self._joint_ids)
        self.single_action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(8,), dtype=np.float32
        )
        self.single_observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(27,), dtype=np.float32
        )
        return self.robot

    def _action_to_control(
        self, action: torch.Tensor, state: FunctionalSolverState
    ) -> dict[str, torch.Tensor]:
        if action.shape != (self.num_envs, 8):
            raise ValueError(
                f"Expected action shape {(self.num_envs,8)}, got {tuple(action.shape)}"
            )
        indices = self._binding.qd_indices[:, 6:]
        effort = action.clamp(-1.0, 1.0) * 400.0
        return {
            "joint_f": torch.zeros_like(state.physical["joint_qd"]).scatter(
                0, indices.flatten(), effort.flatten()
            )
        }

    def _initialize_episode(
        self, env_ids: Sequence[int] | torch.Tensor | None = None, **kwargs: Any
    ) -> None:
        super()._initialize_episode(env_ids, **kwargs)
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        )
        count = ids.numel()
        generator = self.get_generator("ant_dynamics_reset")
        q = (
            torch.tensor(self._profile["default_q"], device=self.device)
            .expand(count, -1)
            .clone()
        )
        qd = torch.zeros(count, self._profile["qd_dim"], device=self.device)
        if self.random_reset:
            q[:, :3] += (
                2 * torch.rand(count, 3, device=self.device, generator=generator) - 1
            ) * q.new_tensor((0.2, 0.2, 0.1))
            axis = 2 * torch.rand(count, 3, device=self.device, generator=generator) - 1
            axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            angle = (
                2 * torch.rand(count, 1, device=self.device, generator=generator) - 1
            ) * (torch.pi / 6)
            q[:, 3:7] = torch.cat(
                (axis * torch.sin(angle / 2), torch.cos(angle / 2)), -1
            )
            q[:, 7:] += 0.2 * (
                2 * torch.rand(count, 8, device=self.device, generator=generator) - 1
            )
            qd[:, :3] = quat_rotate(
                q[:, 3:7],
                0.1
                * (
                    2 * torch.rand(count, 3, device=self.device, generator=generator)
                    - 1
                ),
            )
            qd[:, 3:6] = quat_rotate(
                q[:, 3:7],
                0.25
                * (
                    2 * torch.rand(count, 3, device=self.device, generator=generator)
                    - 1
                ),
            )
            qd[:, 6:] = 0.25 * (
                2 * torch.rand(count, 8, device=self.device, generator=generator) - 1
            )
        self.robot.set_state(
            root_pose=q[:, :7],
            root_velocity=qd[:, :6],
            qpos=q[:, 7:],
            qvel=qd[:, 6:],
            env_ids=ids,
            joint_ids=self._joint_ids,
            clear_dynamics=True,
        )

    def _read_outputs(self, state: FunctionalSolverState) -> dict[str, Any]:
        q, qd = self._binding.read(state.physical)
        observation, reward, failed = reference_task_values("ant", q, qd)
        return {
            "obs": observation,
            "reward": reward,
            "terminated": failed,
            "truncated": torch.zeros_like(failed),
            "info": {
                "fail": failed,
                "metrics": {
                    "height": q[:, 2].detach(),
                    "forward_velocity": qd[:, 1].detach(),
                },
            },
        }

    def close(self, *, exit_process: bool = False) -> None:
        """Close the task without terminating the trainer process by default."""
        super().close(exit_process=exit_process)
        if not exit_process:
            SimulationManager.flush_cleanup_queue()


@register_learning_env("AntDynamics-v1")
def make_ant_dynamics_env(
    num_envs: int,
    device: torch.device | str,
    *,
    checkpoint: str,
    max_episode_steps: int = 1000,
    random_reset: bool = True,
    seed: int | None = None,
) -> AntDynamicsEnv:
    """Build the same task for the existing differentiable RL trainer.

    Args:
        num_envs: Number of parallel homogeneous robots.
        device: Simulation and policy device.
        checkpoint: Export directory, absolute or relative to the working directory.
        max_episode_steps: Control-step episode limit.
        random_reset: Sample task-specific initial states when true.
        seed: Optional initial RNG seed; explicit reset seeds take precedence.

    Returns:
        A registered task implementing the DifferentiableVecEnv protocol.
    """
    physics = NewtonPhysicsCfg(
        device=device,
        physics_dt=1 / 60,
        num_substeps=1,
        requires_grad=True,
        use_cuda_graph=False,
        collision_cfg=None,
        solver_cfg={"solver_type": "nerd", "options": {"checkpoint": checkpoint}},
    )
    sim_cfg = SimulationManagerCfg(
        headless=True,
        num_envs=num_envs,
        device=device,
        arena_space=0.0,
        enable_ground_plane=True,
        enable_entity_gizmo=False,
        robot_ik_gizmo=None,
        render_cfg=RenderCfg(renderer="no-render"),
        startup_summary="off",
        physics_cfg=physics,
    )
    cfg = EmbodiedEnvCfg(
        num_envs=num_envs,
        sim_cfg=sim_cfg,
        sim_steps_per_control=1,
        max_episode_steps=max_episode_steps,
        seed=seed,
        robot=None,
        extensions={"random_reset": random_reset},
    )
    return AntDynamicsEnv(cfg)
