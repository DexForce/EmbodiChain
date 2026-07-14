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
        def _apply_action_kernel(self, action_wp, tape): ...
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

    Subclasses must implement :meth:`_apply_action_kernel` and
    :meth:`_read_outputs`; the rest of the EmbodiedEnv contract (reset,
    observation managers, reward functors) carries over. The default
    ``dynamics`` route invokes the Newton solver through
    :class:`NewtonStepFunc`; subclasses that intentionally use FK-only
    stepping must explicitly select ``kinematics`` and implement
    :meth:`_make_kinematic_step_fn`.
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

    def _apply_action_kernel(self, action_wp: Any, tape: Any) -> None:
        """Inside the open Warp tape, write the action into Newton control.

        Implementations launch a Warp kernel that reads ``action_wp``
        (a ``wp.array(dtype=wp.float32, requires_grad=True)`` of shape
        ``[num_envs * action_dim]``) and writes into
        ``self.sim.physics.newton_manager._control`` so the next stepper
        call uses the new control.
        """
        raise NotImplementedError(
            "Subclasses of DifferentiableEmbodiedEnv must implement "
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

        sim_state = {
            "manager": self.sim,
            "substeps": self.cfg.sim_steps_per_control,
            "action_to_control_kernel": self._wrap_action_kernel(),
            "kernel_args": (),
            "obs_reward_fn": self._read_outputs,
            "last_info": {},
        }
        if mode == "kinematics":
            sim_state["step_fn"] = self._make_kinematic_step_fn()
        return sim_state

    def _wrap_action_kernel(self):
        env = self

        def _inner(action_wp, *_):
            env._apply_action_kernel(action_wp, tape=None)

        return _inner
