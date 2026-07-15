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
"""Differentiable Newton-backed EmbodiedEnv for analytic policy gradient.

Wraps the standard :class:`EmbodiedEnv` step pipeline in a Warp tape and
bridges autograd into PyTorch via
:class:`embodichain.lab.sim.diff.NewtonStepFunc`. Subclasses define how
actions become Newton control writes and how observations/rewards are
read from the post-step state; the bridge handles the tape lifecycle
and the backward pass.

Usage:

    class MyTask(DifferentiableEmbodiedEnv):
        def _apply_dynamics_action_kernel(self, action_wp, control, tape): ...
        def _read_outputs(self, final_state) -> dict: ...
"""

from __future__ import annotations

from typing import Any, Callable, Literal

import torch

from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.sim.cfg import NewtonPhysicsCfg
from embodichain.lab.sim.diff import NewtonStepFunc
from embodichain.utils import logger

__all__ = ["DifferentiableEmbodiedEnv"]


class DifferentiableEmbodiedEnv(EmbodiedEnv):
    """EmbodiedEnv variant that exposes APG-ready :py:meth:`step`.

    Dynamics subclasses must implement :meth:`_apply_dynamics_action_kernel`
    and :meth:`_read_outputs`; the rest of the EmbodiedEnv contract (reset,
    observation managers, reward functors) carries over. The default
    ``dynamics`` route invokes the Newton solver through
    :class:`NewtonStepFunc` using a detached trajectory-local control buffer.
    Subclasses that intentionally use FK-only stepping must explicitly select
    ``kinematics`` and implement :meth:`_make_kinematic_step_fn` together with
    the legacy :meth:`_apply_action_kernel` hook.
    """

    differentiable_step_mode: Literal["dynamics", "kinematics"] = "dynamics"
    """Stepping route used by :meth:`_build_sim_state_dict`."""

    def __init__(self, cfg: EmbodiedEnvCfg, *args, **kwargs) -> None:
        self._validate_diff_cfg(cfg)
        super().__init__(cfg, *args, **kwargs)
        self._truncate_backward_at: int | None = getattr(
            cfg, "truncate_backward_at", None
        )

    @staticmethod
    def _validate_diff_cfg(cfg: EmbodiedEnvCfg) -> None:
        physics_cfg = cfg.sim_cfg.physics_cfg
        if not isinstance(physics_cfg, NewtonPhysicsCfg):
            logger.log_error(
                "DifferentiableEmbodiedEnv requires NewtonPhysicsCfg, "
                f"got {type(physics_cfg).__name__}."
            )
        if not physics_cfg.requires_grad:
            logger.log_error(
                "DifferentiableEmbodiedEnv requires requires_grad=True on "
                "the NewtonPhysicsCfg."
            )

    # -- subclass contract ------------------------------------------------ #

    def _apply_dynamics_action_kernel(
        self,
        action_wp: Any,
        control: Any,
        tape: Any,
    ) -> None:
        """Write an action into a detached dynamics trajectory control buffer.

        Implementations launch a Warp kernel that reads ``action_wp``
        (a ``wp.array(dtype=wp.float32, requires_grad=True)`` of shape
        ``[num_envs * action_dim]``) and writes into the supplied ``control``.
        It is the isolated control owned by the active manager trajectory; do
        not write ``self.sim.physics.newton_manager._control`` while the tape
        is active. ``tape`` is the caller-owned active Warp tape for this
        callback only; the bridge clears the per-step binding after tape exit.
        """
        raise NotImplementedError(
            "Dynamics subclasses of DifferentiableEmbodiedEnv must migrate "
            "their legacy _apply_action_kernel(action_wp, tape) hook to "
            "_apply_dynamics_action_kernel(action_wp, control, tape)."
        )

    def _apply_action_kernel(self, action_wp: Any, tape: Any) -> None:
        """Write an action for the explicitly selected kinematics route.

        This legacy hook is deliberately reserved for
        ``differentiable_step_mode = 'kinematics'``. It receives no detached
        solver control because FK-only environments do not invoke Newton
        solver dynamics.
        """
        raise NotImplementedError(
            "Kinematics subclasses of DifferentiableEmbodiedEnv must implement "
            "_apply_action_kernel(action_wp, tape)."
        )

    def _read_outputs(self, final_state: Any) -> dict:
        """Read the post-step observation and reward as torch tensors.

        Must return a dict with keys ``"obs"``, ``"reward"``,
        ``"terminated"``, ``"truncated"``, plus the ``_order`` and
        ``_grad_track`` metadata expected by
        :class:`NewtonStepFunc`. ``obs`` and ``reward`` should be torch
        tensors backed by ``wp.to_torch`` of grad-tracked Warp arrays.
        """
        raise NotImplementedError(
            "Subclasses of DifferentiableEmbodiedEnv must implement "
            "_read_outputs(final_state)."
        )

    def _make_kinematic_step_fn(self) -> Callable[[], Any]:
        """Return the explicitly selected FK-only stepping callback.

        Subclasses must override this hook only when they set
        :attr:`differentiable_step_mode` to ``"kinematics"``. This keeps
        kinematics distinct from the default solver-dynamics route.

        Raises:
            NotImplementedError: If kinematics mode has no named FK hook.
        """
        raise NotImplementedError(
            "DifferentiableEmbodiedEnv in kinematics mode requires "
            "_make_kinematic_step_fn()."
        )

    # -- gym surface ------------------------------------------------------ #

    def step(self, action: torch.Tensor):
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32)
        sim_state = self._build_sim_state_dict(action)
        outputs = NewtonStepFunc.apply(action, sim_state)
        obs, reward, terminated, truncated = outputs[:4]
        info = sim_state["last_info"]

        done_mask = terminated | truncated
        if done_mask.any():
            reset_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
            fresh_obs, _ = self.reset(options={"reset_ids": reset_ids})
            obs = torch.where(
                done_mask.unsqueeze(-1).expand_as(obs),
                fresh_obs.detach(),
                obs,
            )
        return obs, reward, terminated, truncated, info

    def _build_sim_state_dict(self, action: torch.Tensor) -> dict:
        mode = self.differentiable_step_mode
        if mode not in {"dynamics", "kinematics"}:
            raise ValueError(
                "differentiable_step_mode must be 'dynamics' or 'kinematics', "
                f"got {mode!r}."
            )

        action_kernel, tape_binder = self._action_kernel_for_mode(mode)
        sim_state = {
            "manager": self.sim,
            "step_mode": mode,
            "substeps": self.cfg.sim_steps_per_control,
            "action_to_control_kernel": action_kernel,
            "kernel_args": (),
            "obs_reward_fn": self._read_outputs,
            "last_info": {},
        }
        if tape_binder is not None:
            sim_state["_bind_dynamics_tape"] = tape_binder
        if mode == "kinematics":
            sim_state["step_fn"] = self._make_kinematic_step_fn()
        return sim_state

    def _action_kernel_for_mode(
        self,
        mode: str,
    ) -> tuple[Callable[..., None], Callable[[Any | None], None] | None]:
        """Build the mode-specific action callback consumed by NewtonStepFunc."""
        if mode == "dynamics":
            dynamics_hook = getattr(self, "_apply_dynamics_action_kernel", None)
            if (
                not callable(dynamics_hook)
                or getattr(dynamics_hook, "__func__", None)
                is DifferentiableEmbodiedEnv._apply_dynamics_action_kernel
            ):
                raise NotImplementedError(
                    "Dynamics environments using the legacy "
                    "_apply_action_kernel(action_wp, tape) must migrate to "
                    "_apply_dynamics_action_kernel(action_wp, control, tape)."
                )
            return self._wrap_dynamics_action_kernel(dynamics_hook)
        return self._wrap_kinematic_action_kernel(), None

    @staticmethod
    def _wrap_dynamics_action_kernel(
        dynamics_hook: Callable[..., None],
    ) -> tuple[Callable[..., None], Callable[[Any | None], None]]:
        """Expose a local-control hook with tape ownership scoped per step."""
        active_tape: list[Any | None] = [None]

        def _bind_tape(tape: Any | None) -> None:
            active_tape[0] = tape

        def _inner(action_wp: Any, control: Any, *_: Any) -> None:
            dynamics_hook(action_wp, control, tape=active_tape[0])

        return _inner, _bind_tape

    def _wrap_kinematic_action_kernel(self):
        """Expose the strict legacy action hook only for kinematics mode."""
        env = self

        def _inner(action_wp: Any, tape: Any, *_: Any) -> None:
            env._apply_action_kernel(action_wp, tape=tape)

        return _inner
