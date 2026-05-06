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

"""Differentiable environment base class for analytic-gradient training.

This module provides :class:`DiffEnv`, a base environment that inherits
from :class:`BaseEnv` but skips the dexsim simulation stack.  State and
kinematics are managed by the **Newton** physics engine (via Warp), making
the environment suitable for Analytic Policy Gradient (APG) training where
gradients must flow through a differentiable forward-kinematics pass.

Design decisions:

* **BaseEnv subclass** — inherits the IS-A relationship with ``gym.Env``
  (via ``BaseEnv``) so it is compatible with gymnasium vectorised wrappers,
  but overrides ``__init__`` to bypass ``SimulationManager`` completely.
* **Newton FK** — :meth:`_build_newton_model` constructs a Newton/Warp model
  from any robot URDF.  :meth:`compute_fk` runs non-differentiable FK for
  state bookkeeping; differentiable FK is handled by ``_NewtonStepFunc`` in
  each task subclass.
* **Test-friendly** — ``_build_newton_model`` and ``_init_newton_state`` are
  ordinary methods; tests can mock them (and override ``compute_fk`` on the
  instance) without importing Newton at all.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import torch

from embodichain.utils import configclass

from embodichain.lab.gym.envs.base_env import BaseEnv, EnvCfg

__all__ = [
    "DiffEnvCfg",
    "DiffEnv",
]


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------


@configclass
class DiffEnvCfg(EnvCfg):
    """Configuration for a differentiable environment.

    Inherits from :class:`EnvCfg` and overrides fields that are unused or
    have different defaults in the differentiable-env context.  Subclasses
    add task-specific fields.
    """

    device: str = "cpu"
    """Compute device for tensors and Newton model (e.g. 'cpu' or 'cuda:0')."""

    robot_path: str = MISSING
    """Path to the robot URDF/USD file to load into the Newton model."""

    end_link_name: str = MISSING
    """Name of the end-effector link in the robot model."""

    num_joints: int = MISSING
    default_joint_q: list[float] = MISSING
    joint_limits: list[tuple[float, float]] | None = None
    """Per-joint ``(lower, upper)`` limits in radians.  When ``None``, limits
    are read from the Newton model after it is built."""


# ---------------------------------------------------------------------------
# Differentiable environment base class
# ---------------------------------------------------------------------------


class DiffEnv(BaseEnv):
    """Differentiable environment base for APG training.

    Inherits from :class:`BaseEnv` (or ``gym.Env`` as fallback) but
    **does not call** ``super().__init__()`` so that ``SimulationManager``
    is never created.  All state tensors live on the configured device.

    Newton is used for simulation-world management (FK, model state).
    Differentiable gradients are obtained via ``_NewtonStepFunc`` (defined
    in each task subclass), not through this class directly.

    Subclasses must implement:

    * :meth:`_compute_obs` — build the observation tensor.
    * :meth:`_compute_reward` — compute the per-env reward.
    * :meth:`_compute_terminated` — check success termination.
    * :meth:`_apply_action` — update internal state from an action.
    * :meth:`_reset_envs` — re-initialise a subset of environments.
    * :meth:`_compute_info` — return the step info dictionary.

    Attributes:
        cfg: Environment configuration.
        model: Newton physics model (``None`` until Newton is ready).
        state_0: Newton state buffer (``None`` until Newton is ready).
        joint_q: Joint positions ``[num_envs, num_joints]``.
        eef_pos: End-effector position ``[num_envs, 3]``.
        eef_quat: End-effector quaternion ``[num_envs, 4]`` (x, y, z, w).
        last_action: Previous step's action ``[num_envs, num_joints]``.
        step_count: Per-env step counter ``[num_envs]`` (int32).
    """

    def __init__(self, cfg: DiffEnvCfg):
        # ------------------------------------------------------------------ #
        # DO NOT call super().__init__() — BaseEnv.__init__ creates           #
        # SimulationManager (dexsim), which we intentionally bypass.          #
        # ------------------------------------------------------------------ #
        self.cfg = cfg
        self._num_envs = cfg.num_envs
        self._device = torch.device(cfg.device)
        self._device_str = cfg.device
        self.max_episode_steps = cfg.max_episode_steps

        # Satisfy BaseEnv class-level attributes without instantiating sim.
        self.sim = None
        self.robot = None
        self.sensors = {}
        self.active_joint_ids: list[int] = []

        # ---- Newton model build ---- #
        self.model, _ee_idx, _bodies_per_env = self._build_newton_model()

        # Build the EE global body-index array as a Warp array immediately.
        import warp as wp

        wp.init()
        self._ee_global: wp.array = wp.array(
            [_ee_idx + i * _bodies_per_env for i in range(self._num_envs)],
            dtype=wp.int32,
            device=self._device_str,
        )

        # Newton state (populated in _init_newton_state; can be mocked).
        self._newton_ready: bool = False
        self.state_0 = None
        self._model_joints_per_env: int = len(self.model.joint_limit_lower)
        self._init_newton_state()

        # ---- Joint limits ---- #
        if cfg.joint_limits is not None:
            limits = torch.tensor(
                cfg.joint_limits, dtype=torch.float32, device=self._device
            )
            self.joint_limit_lower: torch.Tensor = limits[:, 0]
            self.joint_limit_upper: torch.Tensor = limits[:, 1]
        elif self._newton_ready:
            lowers = list(self.model.joint_limit_lower)[: cfg.num_joints]
            uppers = list(self.model.joint_limit_upper)[: cfg.num_joints]
            self.joint_limit_lower = torch.tensor(
                lowers, dtype=torch.float32, device=self._device
            )
            self.joint_limit_upper = torch.tensor(
                uppers, dtype=torch.float32, device=self._device
            )
        else:
            raise RuntimeError(
                "joint_limits must be provided in cfg when Newton is not available "
                "(e.g. in test environments)."
            )

        # ---- State tensors ---- #
        self.joint_q = torch.zeros(
            cfg.num_envs, cfg.num_joints, dtype=torch.float32, device=self._device
        )
        self.eef_pos = torch.zeros(
            cfg.num_envs, 3, dtype=torch.float32, device=self._device
        )
        self.eef_quat = torch.zeros(
            cfg.num_envs, 4, dtype=torch.float32, device=self._device
        )
        self.last_action = torch.zeros(
            cfg.num_envs, cfg.num_joints, dtype=torch.float32, device=self._device
        )
        self.step_count = torch.zeros(
            cfg.num_envs, dtype=torch.int32, device=self._device
        )

        # ---- Gym spaces (single; batched spaces via BaseEnv cached_property) ---- #
        self.single_observation_space: gym.spaces.Space | None = None
        self.single_action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(cfg.num_joints,), dtype=np.float32
        )

        # ---- Startup ---- #
        # (No startup event functors; subclasses may override.)
        pass

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """Return compute device (overrides BaseEnv's sim.device)."""
        return self._device

    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        return self._num_envs

    @property
    def num_joints(self) -> int:
        """Number of actuated joints."""
        return self.cfg.num_joints

    # ------------------------------------------------------------------
    # Newton model construction (robot-agnostic)
    # ------------------------------------------------------------------

    def _build_newton_model(self) -> tuple[Any, int, int]:
        """Build a Newton physics model from the configured robot asset.

        Supports URDF (``.urdf``) and USD (``.usd`` / ``.usda`` / ``.usdc``)
        source files.  The robot builder initialisation and file parsing are
        shared between single-env and multi-env (replicated) layouts.

        Returns:
            ``(model, ee_idx, bodies_per_env)`` where ``ee_idx`` is the
            zero-based body index of the end-effector within a single robot
            instance, and ``bodies_per_env`` is the total number of bodies per
            environment.  The caller uses these to assemble
            ``self._ee_global``.

        Raises:
            RuntimeError: If the end-effector body name is not found.
            ValueError: If the asset file extension is not supported.
            ImportError: If Newton or Warp is not installed.
        """
        import warp as wp
        import newton

        wp.init()

        cfg = self.cfg
        robot_path = str(cfg.robot_path)
        default_q = cfg.default_joint_q
        end_link_name = cfg.end_link_name
        num_envs = self._num_envs
        device = self._device_str

        def _find_ee_index(body_labels: list) -> int:
            for i, key in enumerate(body_labels):
                if end_link_name in str(key):
                    return i
            raise RuntimeError(
                f"End-effector body '{end_link_name}' not found in '{robot_path}'."
            )

        def _fill_default_q(builder) -> None:
            for i, q in enumerate(default_q):
                if i < len(builder.joint_q):
                    builder.joint_q[i] = q
                    builder.joint_target_pos[i] = q

        def _parse_robot_asset(builder: newton.ModelBuilder) -> None:
            """Add the robot asset to *builder* based on file extension."""
            ext = robot_path.rsplit(".", 1)[-1].lower()
            xform = wp.transform((0.0, 0.0, 0.0), wp.quat_identity())
            if ext == "urdf":
                builder.add_urdf(
                    robot_path,
                    xform=xform,
                    floating=False,
                    enable_self_collisions=False,
                )
            elif ext in ("usd", "usda", "usdc"):
                builder.add_usd(
                    robot_path,
                    xform=xform,
                    floating=False,
                    enable_self_collisions=False,
                )
            else:
                raise ValueError(
                    f"Unsupported robot asset format '.{ext}'. "
                    "Expected one of: .urdf, .usd, .usda, .usdc"
                )

        # --- Build a single-robot builder to parse the asset and find EE --- #
        robot_builder = newton.ModelBuilder()
        _parse_robot_asset(robot_builder)
        _fill_default_q(robot_builder)
        ee_idx = _find_ee_index(robot_builder.body_label)
        bodies_per_env = len(list(robot_builder.body_label))

        # --- Assemble the final (possibly replicated) world builder --- #
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        if num_envs == 1:
            builder.add_builder(robot_builder)
        else:
            builder.replicate(
                robot_builder, world_count=num_envs, spacing=(0.0, 0.0, 0.0)
            )
        model = builder.finalize(device=device, requires_grad=cfg.requires_grad)

        return model, ee_idx, bodies_per_env

    def _init_newton_state(self) -> None:
        """Initialise Newton state buffers after model build.

        Creates ``state_0`` and ``_model_joints_per_env``.
        This method is deliberately **overridable** so that test fixtures can
        replace it with a lightweight mock that avoids real Newton/Warp calls.
        """
        import newton

        self.state_0 = self.model.state()
        newton.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, self.state_0
        )
        # Number of joints per environment in the Newton model (includes fingers, etc.)
        self._model_joints_per_env = len(list(self.model.joint_limit_lower))
        self._newton_ready = True

    # ------------------------------------------------------------------
    # Forward kinematics (Newton)
    # ------------------------------------------------------------------

    def compute_fk(self, joint_q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Non-differentiable forward kinematics via Newton.

        Syncs ``joint_q`` into the Newton state and evaluates FK.
        The result is detached from the PyTorch computation graph.

        .. tip::
            In tests, override this on the **instance** to inject a
            pure-PyTorch FK without Newton::

                env.compute_fk = my_dh_fk_function

        Args:
            joint_q: Joint positions ``[batch, num_joints]`` (arm joints only).

        Returns:
            ``(eef_pos, eef_quat)`` — both detached, shapes ``[batch, 3]``
            and ``[batch, 4]`` (x, y, z, w quaternion convention).
        """
        import warp as wp
        import newton

        batch = joint_q.shape[0]

        # Sync PyTorch joint_q → Warp state_0.joint_q (shared memory).
        state_q_t = wp.to_torch(self.state_0.joint_q).view(
            self._num_envs, self._model_joints_per_env
        )
        with torch.no_grad():
            state_q_t[:batch, : self.cfg.num_joints] = joint_q.to(self._device)

        # Evaluate FK in place (updates state_0.body_q).
        newton.eval_fk(
            self.model, self.state_0.joint_q, self.model.joint_qd, self.state_0
        )

        # Extract EEF transforms [pos(3) + quat(4)] in (x, y, z, w) order.
        body_q_t = wp.to_torch(self.state_0.body_q)
        ee_idx = wp.to_torch(self._ee_global).long()[:batch]
        eef_t = body_q_t[ee_idx]

        return eef_t[:, :3].detach().clone(), eef_t[:, 3:].detach().clone()

    def _compute_fk_all(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Run FK for **all** environments using ``self.joint_q``.

        Delegates to :meth:`compute_fk` so that test mocks applied to
        ``instance.compute_fk`` are transparently honoured here.

        Returns:
            ``(eef_pos, eef_quat)`` for all environments.
        """
        return self.compute_fk(self.joint_q)

    # ------------------------------------------------------------------
    # Lifecycle: reset / step
    # ------------------------------------------------------------------

    def reset(
        self,
        env_ids: torch.Tensor | None = None,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset environments and return initial observations.

        Args:
            env_ids: Subset of environments to reset (default: all).
            seed: Optional random seed.

        Returns:
            ``(obs, info)``
        """
        if seed is not None:
            torch.manual_seed(seed)

        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)

        self.step_count[env_ids] = 0
        self.last_action[env_ids] = 0.0

        self._reset_envs(env_ids)

        # Run FK for all envs (Newton FK is batched; updating all is safe
        # because self.joint_q is always up-to-date after _reset_envs).
        with torch.no_grad():
            all_eef_pos, all_eef_quat = self._compute_fk_all()
            self.eef_pos = all_eef_pos
            self.eef_quat = all_eef_quat

        obs = self._compute_obs()
        return obs, {}

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Step all environments with a pure-PyTorch (non-differentiable Newton) path.

        This default implementation is suitable for PPO rollout collection and
        for tests where :attr:`_newton_ready` is ``False``.  Subclasses that
        need a differentiable APG step (via ``_NewtonStepFunc``) should
        override :meth:`step` and call this as a fallback when Newton is
        unavailable.

        Args:
            action: Normalised actions ``[num_envs, num_joints]`` in ``[-1, 1]``.

        Returns:
            ``(obs, reward, terminated, truncated, info)``
        """
        self.step_count += 1

        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32, device=self._device)
        action = torch.clamp(action.to(self._device), -1.0, 1.0)

        # Subclass applies action → returns new (possibly differentiable) joint_q.
        new_jpos = self._apply_action(action)

        # FK: non-differentiable in production (Newton); differentiable in tests (DH mock).
        eef_pos, eef_quat = self.compute_fk(new_jpos)

        # Detach and persist state for next step.
        with torch.no_grad():
            self.joint_q = new_jpos.detach().clone()
            self.eef_pos = eef_pos.detach().clone()
            self.eef_quat = eef_quat.detach().clone()

        obs = self._compute_obs_from_state(new_jpos, eef_pos, eef_quat)
        info = self._compute_info(eef_pos.detach(), eef_quat.detach())
        reward = self._compute_reward(eef_pos, eef_quat, action, obs, info)

        self.last_action = action.detach().clone()

        truncated = self.step_count >= self.max_episode_steps
        terminated = self._compute_terminated(eef_pos.detach(), eef_quat.detach())
        done_mask = truncated | terminated

        info["success"] = terminated.detach()

        if done_mask.any():
            reset_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
            self.reset(reset_ids)
            fresh_obs = self._compute_obs()
            obs = torch.where(done_mask.unsqueeze(-1).expand_as(obs), fresh_obs, obs)

        return obs, reward, terminated, truncated, info

    def detach_state(self) -> None:
        """Detach all stateful tensors from the computation graph.

        Called by APG at segment boundaries to truncate the gradient chain.
        """
        self.joint_q = self.joint_q.detach()
        self.last_action = self.last_action.detach()
        self.eef_pos = self.eef_pos.detach()
        self.eef_quat = self.eef_quat.detach()

    def close(self) -> None:
        """Release resources (no-op; override if a viewer is attached)."""
        pass

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement
    # ------------------------------------------------------------------

    def _compute_obs(self) -> torch.Tensor:
        """Build observation from the *current* (detached) state."""
        raise NotImplementedError

    def _compute_obs_from_state(
        self,
        joint_q: torch.Tensor,
        eef_pos: torch.Tensor,
        eef_quat: torch.Tensor,
    ) -> torch.Tensor:
        """Build observation with possibly-differentiable FK outputs.

        Default delegates to :meth:`_compute_obs`.  Override for tasks
        that need gradient flow through observations.
        """
        return self._compute_obs()

    def _compute_reward(
        self,
        eef_pos: torch.Tensor,
        eef_quat: torch.Tensor,
        action: torch.Tensor,
        obs: torch.Tensor,
        info: dict[str, Any],
    ) -> torch.Tensor:
        """Compute per-env reward ``[num_envs]``."""
        raise NotImplementedError

    def _compute_terminated(
        self,
        eef_pos: torch.Tensor,
        eef_quat: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-env success flags ``[num_envs]``."""
        raise NotImplementedError

    def _apply_action(self, action: torch.Tensor) -> torch.Tensor:
        """Apply action and return new (possibly differentiable) joint positions.

        Args:
            action: Normalised action ``[num_envs, num_joints]``.

        Returns:
            New joint positions ``[num_envs, num_joints]``.
        """
        raise NotImplementedError

    def _reset_envs(self, env_ids: torch.Tensor) -> None:
        """Reset internal state for the given environment indices.

        Must at minimum set ``self.joint_q[env_ids]`` to valid values.
        """
        raise NotImplementedError

    def _compute_info(
        self,
        eef_pos: torch.Tensor,
        eef_quat: torch.Tensor,
    ) -> dict[str, Any]:
        """Return the step info dictionary."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Quaternion helper
# ---------------------------------------------------------------------------


def _rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to quaternions (x, y, z, w).

    Uses Shepperd's method with conditional branches for numerical stability.
    Fully differentiable through ``torch.where``.

    Args:
        R: Rotation matrices ``[batch, 3, 3]``.

    Returns:
        Quaternions ``[batch, 4]`` in (x, y, z, w).
    """
    batch = R.shape[0]
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    qw = torch.zeros(batch, device=R.device, dtype=R.dtype)
    qx = torch.zeros(batch, device=R.device, dtype=R.dtype)
    qy = torch.zeros(batch, device=R.device, dtype=R.dtype)
    qz = torch.zeros(batch, device=R.device, dtype=R.dtype)

    s = torch.sqrt(torch.clamp(trace + 1.0, min=1e-10)) * 2
    mask1 = trace > 0
    qw = torch.where(mask1, 0.25 * s, qw)
    qx = torch.where(mask1, (R[:, 2, 1] - R[:, 1, 2]) / s, qx)
    qy = torch.where(mask1, (R[:, 0, 2] - R[:, 2, 0]) / s, qy)
    qz = torch.where(mask1, (R[:, 1, 0] - R[:, 0, 1]) / s, qz)

    s2 = (
        torch.sqrt(torch.clamp(1.0 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2], min=1e-10))
        * 2
    )
    mask2 = (~mask1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    qw = torch.where(mask2, (R[:, 2, 1] - R[:, 1, 2]) / s2, qw)
    qx = torch.where(mask2, 0.25 * s2, qx)
    qy = torch.where(mask2, (R[:, 0, 1] + R[:, 1, 0]) / s2, qy)
    qz = torch.where(mask2, (R[:, 0, 2] + R[:, 2, 0]) / s2, qz)

    s3 = (
        torch.sqrt(torch.clamp(1.0 + R[:, 1, 1] - R[:, 0, 0] - R[:, 2, 2], min=1e-10))
        * 2
    )
    mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    qw = torch.where(mask3, (R[:, 0, 2] - R[:, 2, 0]) / s3, qw)
    qx = torch.where(mask3, (R[:, 0, 1] + R[:, 1, 0]) / s3, qx)
    qy = torch.where(mask3, 0.25 * s3, qy)
    qz = torch.where(mask3, (R[:, 1, 2] + R[:, 2, 1]) / s3, qz)

    s4 = (
        torch.sqrt(torch.clamp(1.0 + R[:, 2, 2] - R[:, 0, 0] - R[:, 1, 1], min=1e-10))
        * 2
    )
    mask4 = (~mask1) & (~mask2) & (~mask3)
    qw = torch.where(mask4, (R[:, 1, 0] - R[:, 0, 1]) / s4, qw)
    qx = torch.where(mask4, (R[:, 0, 2] + R[:, 2, 0]) / s4, qx)
    qy = torch.where(mask4, (R[:, 1, 2] + R[:, 2, 1]) / s4, qy)
    qz = torch.where(mask4, 0.25 * s4, qz)

    q = torch.stack([qx, qy, qz, qw], dim=-1)
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-10)
    return q
