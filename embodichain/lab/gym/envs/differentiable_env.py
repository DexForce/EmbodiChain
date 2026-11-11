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
"""Differentiable simulation with explicit kinematic and functional dynamics modes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Literal, TYPE_CHECKING

import torch

from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.sim.cfg import NewtonPhysicsCfg
from embodichain.lab.sim.types import EnvObs
from embodichain.lab.sim.diff import NewtonStepFunc
from embodichain.lab.sim.diff.dynamics import FunctionalDynamics, detach_dynamics_state
from embodichain.utils import logger

if TYPE_CHECKING:
    from dexsim.engine.newton_physics import FunctionalSolverState

__all__ = ["DifferentiableEnv"]


class DifferentiableEnv(EmbodiedEnv):
    """Common environment entry point for differentiable simulation tasks.

    Args:
        cfg: Simulation and task configuration using Newton gradient mode.
        *args: Positional arguments passed to :class:`EmbodiedEnv`.
        step_mode: ``"kinematic"`` records task-defined Warp hooks without
            advancing physics. ``"dynamics"`` advances the configured solver
            through the Scene functional interface. Selection is explicit and
            does not depend on the solver name.
        **kwargs: Keyword arguments passed to :class:`EmbodiedEnv`.

    Kinematic tasks implement ``_apply_action_kernel``,
    ``_make_kinematic_step_fn`` and ``_read_outputs``. Tracked terminal steps
    defer reset until backward has consumed the Warp tape. Dynamics tasks
    implement ``_action_to_control`` and ``_read_outputs`` over independent
    functional state. They auto-reset selected rows while preserving terminal
    observation/reward graphs for value bootstrap.
    """

    def __init__(
        self,
        cfg: EmbodiedEnvCfg,
        *args: Any,
        step_mode: Literal["kinematic", "dynamics"] = "kinematic",
        **kwargs: Any,
    ) -> None:
        self._validate_diff_cfg(cfg, step_mode=step_mode)
        self._step_mode = step_mode
        self._dynamics: FunctionalDynamics | None = None
        self._functional_state: FunctionalSolverState | None = None
        self._resetting_functional = False
        super().__init__(cfg, *args, **kwargs)
        if step_mode == "dynamics":
            try:
                if (
                    getattr(self, "event_manager", None) is not None
                    and "interval" in self.event_manager.available_modes
                ):
                    raise ValueError(
                        "DifferentiableEnv dynamics mode does not support native interval events; "
                        "implement changes in the functional transition"
                    )
                self._ensure_dynamics()
            except Exception:
                self._cleanup_failed_initialization()
                raise

    @staticmethod
    def _validate_diff_cfg(
        cfg: EmbodiedEnvCfg,
        *,
        step_mode: Literal["kinematic", "dynamics"] = "kinematic",
    ) -> None:
        if step_mode not in {"kinematic", "dynamics"}:
            raise ValueError("step_mode must be kinematic or dynamics")
        physics_cfg = None if cfg.sim_cfg is None else cfg.sim_cfg.physics_cfg
        if not isinstance(physics_cfg, NewtonPhysicsCfg):
            logger.log_error(
                "DifferentiableEnv requires NewtonPhysicsCfg, "
                f"got {type(physics_cfg).__name__}."
            )
        if not physics_cfg.requires_grad:
            logger.log_error(
                "DifferentiableEnv requires requires_grad=True on the NewtonPhysicsCfg."
            )
        if step_mode == "dynamics":
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
                        f"DifferentiableEnv dynamics mode uses functional hooks instead of configured {name} functors"
                    )

    def _functional_state_tensors(self) -> tuple[torch.Tensor, ...]:
        """Return recurrent state tensors forwarded through autograd.

        The default environment has no functional state. Subclasses that keep
        differentiable state between steps return it here in the same order as
        the Warp arrays accepted by :meth:`_apply_action_kernel`. Their output
        hook must expose each next-state tensor through ``_order`` and
        ``_grad_track`` before retaining that same tensor for the next step.

        Returns:
            Functional state tensors for the current control step.
        """
        return ()

    def _apply_action_kernel(
        self,
        action_wp: Any,
        *state_wps: Any,
        tape: Any,
    ) -> None:
        """Write an action for the task-defined kinematics callback.

        Functional state arrays follow ``action_wp`` and ``tape`` is always
        keyword-only, so state cannot collide with the tape argument. The hook
        receives no solver control in kinematic mode.
        """
        del state_wps
        raise NotImplementedError(
            "DifferentiableEnv subclasses must implement "
            "_apply_action_kernel(action_wp, *state_wps, tape=tape)."
        )

    def _read_outputs(self, final_state: Any) -> dict[str, Any]:
        """Compute observations, rewards, termination flags and optional info.

        Args:
            final_state: Kinematic callback result or a functional solver state,
                according to the selected mode.

        Returns:
            Tensor ``obs``, ``reward``, ``terminated`` and ``truncated`` fields,
            with optional ``info``. Dynamics hooks must be pure because they
            also run during reset, observation and detach. Kinematic hooks run
            inside the Warp tape and additionally supply ``_order`` and
            ``_grad_track`` for :class:`NewtonStepFunc`.
        """
        raise NotImplementedError("Implement _read_outputs(final_state)")

    def _make_kinematic_step_fn(self) -> Callable[[], Any]:
        """Return the task-defined kinematics callback.

        Raises:
            NotImplementedError: If the subclass has no kinematics hook.
        """
        raise NotImplementedError(
            "DifferentiableEnv requires _make_kinematic_step_fn()."
        )

    def _ensure_dynamics(self) -> FunctionalDynamics:
        if self._step_mode != "dynamics":
            raise RuntimeError("Functional solver state requires step_mode='dynamics'")
        if self._dynamics is None:
            self._dynamics = FunctionalDynamics(self.sim)
        if self._functional_state is None:
            self._functional_state = self._dynamics.create_state()
        return self._dynamics

    @property
    def dynamics_state(self) -> FunctionalSolverState:
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

    def get_obs(self, **kwargs: Any) -> torch.Tensor | EnvObs:
        """Read observations from graph state, or fresh native state during reset."""
        if self._step_mode == "kinematic":
            return super().get_obs(**kwargs)
        dynamics = self._ensure_dynamics()
        state = (
            dynamics.create_state()
            if self._resetting_functional
            else self._functional_state
        )
        return self._outputs(state)["obs"]

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[torch.Tensor | EnvObs, dict[str, Any]]:
        """Reset task state, merging fresh dynamics rows without changing other graphs.

        Args:
            seed: Optional reset RNG seed.
            options: Base environment reset options, including selected reset_ids.

        Returns:
            Initial observations and reset metadata. Kinematics delegates to the
            normal task lifecycle; dynamics additionally merges solver history.
        """
        if self._step_mode == "kinematic":
            return super().reset(seed=seed, options=options)
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
        """Advance the selected path and apply its graph-safe terminal policy.

        Args:
            action: Batched task action. Kinematic callbacks consume one action
                per control step; dynamics holds controls across the configured
                physics macro steps without invoking a second World update.

        Returns:
            Observation, reward, termination, truncation and info. Dynamics
            auto-resets finished rows. Kinematics reports deferred reset IDs
            when the action or recurrent input retains a Warp tape.
        """
        advance = (
            self._step_kinematic
            if self._step_mode == "kinematic"
            else self._step_dynamics
        )
        obs, reward, terminated, truncated, info, defer_reset = advance(action)
        done = terminated | truncated
        reset_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if reset_ids.numel():
            info["terminal_observation"] = obs
            info["terminal_mask"] = done
            if defer_reset:
                info["requires_reset_after_backward"] = True
                info["deferred_reset_ids"] = reset_ids.detach().clone()
            else:
                fresh, _ = self.reset(options={"reset_ids": reset_ids})
                row_mask = done.reshape(done.shape[0], *([1] * (obs.ndim - 1)))
                obs = torch.where(row_mask, fresh, obs)
        return obs, reward, terminated, truncated, info

    def _step_kinematic(
        self, action: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any], bool
    ]:
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32)
        functional_state = tuple(self._functional_state_tensors())
        retains_tape_for_backward = bool(
            torch.is_grad_enabled()
            and (
                action.requires_grad
                or any(
                    bool(getattr(state, "requires_grad", False))
                    for state in functional_state
                )
            )
        )
        sim_state = self._build_sim_state_dict(action)
        outputs = NewtonStepFunc.apply(action, sim_state, *functional_state)
        obs, reward, terminated, truncated = outputs[:4]
        info = sim_state["last_info"]

        return obs, reward, terminated, truncated, info, retains_tape_for_backward

    def _step_dynamics(
        self, action: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any], bool
    ]:
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
        return obs, reward, terminated, truncated, info, False

    def detach_state(self) -> torch.Tensor:
        """Detach functional solver state without resetting or resampling.

        Returns:
            The current detached observation in dynamics mode.

        Raises:
            NotImplementedError: Kinematic tasks own their recurrent tensors and
                must implement their own detach boundary before using a
                segmented differentiable trainer.
        """
        if self._step_mode == "kinematic":
            raise NotImplementedError(
                "Kinematic tasks must implement detach_state() for their task-owned recurrent state"
            )
        self._ensure_dynamics()
        self._functional_state = detach_dynamics_state(self._functional_state)
        return self._outputs(self._functional_state)["obs"].detach()

    def _build_sim_state_dict(self, action: torch.Tensor) -> dict:
        del action
        return {
            "manager": self.sim,
            "action_kernel": self._wrap_action_kernel(),
            "kernel_args": (),
            "obs_reward_fn": self._read_outputs,
            "last_info": {},
            "step_fn": self._make_kinematic_step_fn(),
        }

    def _wrap_action_kernel(self) -> Callable[..., None]:
        """Adapt the task action hook to :class:`NewtonStepFunc`."""
        env = self

        def _inner(action_wp: Any, tape: Any, *state_wps: Any) -> None:
            env._apply_action_kernel(action_wp, *state_wps, tape=tape)

        return _inner
