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
"""Newton-backed kinematic environment for analytic policy gradient.

Wraps a task-defined Newton kinematics callback in a Warp tape and bridges
autograd into PyTorch via :class:`embodichain.lab.sim.diff.NewtonStepFunc`.
The environment deliberately does not advance the configured Newton solver;
differentiable dynamics are outside the current public contract.

Usage:

    class MyTask(DifferentiableEnv):
        def _apply_action_kernel(self, action_wp, tape): ...
        def _make_kinematic_step_fn(self): ...
        def _read_outputs(self, final_state) -> dict: ...
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.sim.cfg import NewtonPhysicsCfg
from embodichain.lab.sim.diff import NewtonStepFunc
from embodichain.utils import logger

__all__ = ["DifferentiableEnv"]


class DifferentiableEnv(EmbodiedEnv):
    """Newton-only environment with an APG-ready kinematic :meth:`step`.

    Subclasses implement :meth:`_apply_action_kernel`,
    :meth:`_make_kinematic_step_fn`, and :meth:`_read_outputs`. The action,
    kinematics, observation, and reward kernels execute inside one Warp tape.
    No Newton solver or collision step is invoked by this environment.
    """

    def __init__(
        self,
        cfg: EmbodiedEnvCfg,
        *args: Any,
        **kwargs: Any,
    ) -> None:
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
                "DifferentiableEnv requires NewtonPhysicsCfg, "
                f"got {type(physics_cfg).__name__}."
            )
        if not physics_cfg.requires_grad:
            logger.log_error(
                "DifferentiableEnv requires requires_grad=True on "
                "the NewtonPhysicsCfg."
            )

    # -- subclass contract ------------------------------------------------ #

    def _apply_action_kernel(self, action_wp: Any, tape: Any) -> None:
        """Write an action for the task-defined kinematics callback.

        The hook receives no solver control because :class:`DifferentiableEnv`
        never advances Newton dynamics.
        """
        raise NotImplementedError(
            "DifferentiableEnv subclasses must implement "
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
            "DifferentiableEnv subclasses must implement _read_outputs(final_state)."
        )

    def _make_kinematic_step_fn(self) -> Callable[[], Any]:
        """Return the task-defined kinematics callback.

        Raises:
            NotImplementedError: If the subclass has no kinematics hook.
        """
        raise NotImplementedError(
            "DifferentiableEnv requires _make_kinematic_step_fn()."
        )

    # -- gym surface ------------------------------------------------------ #

    def step(self, action: torch.Tensor):
        """Advance one differentiable control step.

        Terminal environments are auto-reset only when the call cannot retain
        a Warp tape for backward. A grad-tracked step returns terminal
        observations unchanged and records ``deferred_reset_ids`` in ``info``;
        callers must run backward before resetting those environments.
        """
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32)
        retains_tape_for_backward = bool(
            torch.is_grad_enabled() and action.requires_grad
        )
        sim_state = self._build_sim_state_dict(action)
        outputs = NewtonStepFunc.apply(action, sim_state)
        obs, reward, terminated, truncated = outputs[:4]
        info = sim_state["last_info"]

        done_mask = terminated | truncated
        if done_mask.any():
            reset_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
            if retains_tape_for_backward:
                info["requires_reset_after_backward"] = True
                info["deferred_reset_ids"] = reset_ids.detach().clone()
            else:
                fresh_obs, _ = self.reset(options={"reset_ids": reset_ids})
                obs = torch.where(
                    done_mask.unsqueeze(-1).expand_as(obs),
                    fresh_obs.detach(),
                    obs,
                )
        return obs, reward, terminated, truncated, info

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

        def _inner(action_wp: Any, tape: Any, *_: Any) -> None:
            env._apply_action_kernel(action_wp, tape=tape)

        return _inner
