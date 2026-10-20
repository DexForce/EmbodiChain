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

"""Newton solver dynamics with immutable functional state for analytic policy gradients."""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any, TYPE_CHECKING
import torch
from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.sim.cfg import NewtonPhysicsCfg
from embodichain.lab.sim.diff.dynamics import FunctionalDynamics, detach_dynamics_state

if TYPE_CHECKING:
    from dexsim.engine.newton_physics import FunctionalSolverState

__all__ = ["DifferentiableDynamicsEnv"]


class DifferentiableDynamicsEnv(EmbodiedEnv):
    """Environment for a registered solver's functional dynamics implementation.

    Task subclasses implement ``_action_to_control(action, state)`` and
    ``_read_outputs(state)``. Observations and rewards must be derived from the
    supplied functional tensors, never read back from detached native buffers.
    Normal EmbodiedEnv scene, robot and initialization hooks still apply.

    Finished rows auto-reset before the next observation, preserving terminal
    reward graphs and unselected state/history graphs. ``detach_state()`` is the
    explicit rollout boundary. Retained history costs O(rollout length); callers
    own that boundary and must backward before updating model parameters.
    """

    def __init__(self, cfg: EmbodiedEnvCfg, *args: Any, **kwargs: Any) -> None:
        self._validate_dynamics_cfg(cfg)
        self._dynamics: FunctionalDynamics | None = None
        self._functional_state: FunctionalSolverState | None = None
        self._resetting_functional = False
        super().__init__(cfg, *args, **kwargs)
        try:
            if (
                getattr(self, "event_manager", None) is not None
                and "interval" in self.event_manager.available_modes
            ):
                raise ValueError(
                    "DifferentiableDynamicsEnv does not support native interval events; implement changes in the functional transition"
                )
            self._ensure_dynamics()
        except Exception:
            self._cleanup_failed_initialization()
            raise

    @staticmethod
    def _validate_dynamics_cfg(cfg: EmbodiedEnvCfg) -> None:
        if cfg.sim_cfg is None or not isinstance(
            cfg.sim_cfg.physics_cfg, NewtonPhysicsCfg
        ):
            raise ValueError("DifferentiableDynamicsEnv requires NewtonPhysicsCfg")
        for name in ("actions", "observations", "rewards"):
            configured = getattr(cfg, name, None)
            terms = (
                configured
                if isinstance(configured, Mapping)
                else vars(configured) if configured is not None else {}
            )
            if any(
                value is not None
                for key, value in terms.items()
                if not key.startswith("_")
            ):
                raise ValueError(
                    f"DifferentiableDynamicsEnv uses functional hooks instead of configured {name} functors"
                )
        if not cfg.sim_cfg.physics_cfg.requires_grad:
            raise ValueError("DifferentiableDynamicsEnv requires requires_grad=True")

    def _ensure_dynamics(self) -> FunctionalDynamics:
        if self._dynamics is None:
            self._dynamics = FunctionalDynamics(self.sim)
        if self._functional_state is None:
            self._functional_state = self._dynamics.create_state()
        return self._dynamics

    @property
    def functional_state(self) -> FunctionalSolverState:
        """Return current graph-connected state; callers must treat it as immutable."""
        self._ensure_dynamics()
        return self._functional_state

    def dynamics_parameters(self) -> Mapping[str, torch.Tensor]:
        """Return named solver parameters for explicit optimization or freezing."""
        return self._ensure_dynamics().parameters()

    def _action_to_control(
        self, action: torch.Tensor, state: FunctionalSolverState
    ) -> Any:
        """Map actions to differentiable controls held for one control interval.

        Args:
            action: Batched task action on the environment device.
            state: Current graph-connected physical state and history.

        Returns:
            The selected provider's control structure, retaining action gradients.
        """
        raise NotImplementedError("Implement _action_to_control(action, state)")

    def _read_outputs(self, state: FunctionalSolverState) -> dict[str, Any]:
        """Compute obs, reward, terminated, truncated and optional info from tensors.

        The observation's leading axis is environment. Reward and boolean done
        fields have shape (num_envs,). This hook must be pure: it is also called
        for reset, get_obs and detach_state without a physics transition.
        """
        raise NotImplementedError("Implement _read_outputs(state)")

    def _outputs(self, state: FunctionalSolverState) -> dict[str, Any]:
        outputs = self._read_outputs(state)
        for name in ("obs", "reward", "terminated", "truncated"):
            if name not in outputs or not isinstance(outputs[name], torch.Tensor):
                raise TypeError(f"_read_outputs must return tensor {name!r}")
        if outputs["obs"].ndim < 1 or outputs["obs"].shape[0] != self.num_envs:
            raise ValueError("Observation must have one leading row per environment")
        for name in ("reward", "terminated", "truncated"):
            if outputs[name].shape != (self.num_envs,):
                raise ValueError(f"{name} must have shape (num_envs,)")
        if any(
            outputs[name].dtype != torch.bool for name in ("terminated", "truncated")
        ):
            raise TypeError("terminated and truncated must be boolean tensors")
        return outputs

    def get_obs(self, **kwargs: Any) -> torch.Tensor:
        """Read observations from graph state, or fresh native state during reset."""
        del kwargs
        dynamics = self._ensure_dynamics()
        state = (
            dynamics.create_state()
            if self._resetting_functional
            else self._functional_state
        )
        return self._outputs(state)["obs"]

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run normal reset events then merge selected fresh rows into graph state."""
        dynamics = self._ensure_dynamics()
        old_state = self._functional_state
        options = {} if options is None else dict(options)
        mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        if "reset_ids" in options:
            ids = torch.as_tensor(
                options["reset_ids"], dtype=torch.long, device=self.device
            ).reshape(-1)
            mask = torch.zeros_like(mask).scatter(0, ids, True)
        # Base reset fills selected counters in place. Preserve any old counter
        # tensor referenced by a terminal reward expression before entering it.
        self._elapsed_steps = self._elapsed_steps.clone()
        self._resetting_functional = True
        try:
            _, info = super().reset(seed=seed, options=options)
        finally:
            self._resetting_functional = False
        self._functional_state = dynamics.reset(old_state, mask)
        return self._outputs(self._functional_state)["obs"], info

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Advance configured physics macro steps without a second World.update."""
        dynamics = self._ensure_dynamics()
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        control = self._action_to_control(action, self._functional_state)
        sensors = [
            sensor for sensor in self.sensors.values() if sensor.requires_substep_update
        ]
        for sensor in sensors:
            sensor.begin_control_step()
        for _ in range(int(self.cfg.sim_steps_per_control)):
            self._functional_state = dynamics.step(
                self._functional_state, control, physics_dt=self.physics_dt
            )
            for sensor in sensors:
                sensor.update_physics_step(self.physics_dt)
        self._update_physical_objective()
        # Never mutate a counter tensor retained by a task's reward expression.
        self._elapsed_steps = self._elapsed_steps + 1
        with self.sim.render_frame():
            outputs = self._outputs(self._functional_state)
        obs, reward = outputs["obs"], outputs["reward"]
        terminated = outputs["terminated"]
        if self.cfg.ignore_terminations:
            terminated = torch.zeros_like(terminated)
        truncated = outputs["truncated"] | (
            self._elapsed_steps >= self.max_episode_steps
        )
        info = dict(outputs.get("info", {}))
        done = terminated | truncated
        self._hook_after_sim_step(
            obs=obs,
            action=action,
            rewards=reward,
            dones=done,
            info=info,
            terminateds=terminated,
            truncateds=truncated,
        )
        reset_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if reset_ids.numel():
            info["terminal_observation"] = obs
            info["terminal_mask"] = done
            fresh, _ = self.reset(options={"reset_ids": reset_ids})
            row_mask = done.reshape(self.num_envs, *([1] * (obs.ndim - 1)))
            obs = torch.where(row_mask, fresh, obs)
        return obs, reward, terminated, truncated, info

    def detach_state(self) -> torch.Tensor:
        """Detach current physical/history state and observation without resetting."""
        self._ensure_dynamics()
        self._functional_state = detach_dynamics_state(self._functional_state)
        return self._outputs(self._functional_state)["obs"].detach()
