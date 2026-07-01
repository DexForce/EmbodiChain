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
"""Franka FR3 reach task with differentiable Newton physics (APG).

Built on :class:`DifferentiableEmbodiedEnv`. The Warp-tape bridge
produces ``action.grad`` that flows back through a differentiable
forward-kinematics path (``newton.eval_fk``). The semi_implicit
solver does not propagate grad through ``joint_target_pos`` to
``body_q`` (the grad path is zero), so we bypass the dynamics
solver and run FK directly, matching the reference APG
implementation in
``/root/sources/analytic_policy_gradients/envs/franka_reach_env.py``.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
import warp as wp
import newton
import newton.utils

from embodichain.lab.gym.envs.differentiable_env import DifferentiableEmbodiedEnv
from embodichain.lab.gym.envs.embodied_env import EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim.cfg import (
    NewtonPhysicsCfg,
    RobotCfg,
    URDFCfg,
)
from embodichain.lab.sim.sim_manager import SimulationManagerCfg

__all__ = ["FrankaReachApgEnv"]

# Franka FR3 arm has 7 actuated arm joints; the URDF also has 2 finger
# joints (9 dof total). We only control the 7 arm joints.
FRANKA_NUM_ARM_JOINTS = 7
FRANKA_EE_BODY = "fr3_hand_tcp"
DEFAULT_ACTION_SCALE = 0.2
DEFAULT_MAX_EPISODE_STEPS = 30
TARGET_POS_RANGE = {
    "x": (0.05, 0.70),
    "y": (-0.45, 0.45),
    "z": (0.20, 0.95),
}


@wp.kernel
def _set_joint_targets_kernel(
    action: wp.array(dtype=wp.float32),
    current_q: wp.array(dtype=wp.float32),
    target_q: wp.array(dtype=wp.float32),
    limit_lo: wp.array(dtype=wp.float32),
    limit_hi: wp.array(dtype=wp.float32),
    action_scale: wp.float32,
    n_joints_per_env: wp.int32,
    n_arm: wp.int32,
    total: wp.int32,
):
    """Compute new joint q: target = clamp(current + action * scale, lo, hi)."""
    tid = wp.tid()
    if tid < total:
        env_idx = tid / n_arm
        j = tid % n_arm
        off = env_idx * n_joints_per_env + j
        new_q = current_q[off] + action[tid] * action_scale
        target_q[off] = wp.clamp(new_q, limit_lo[j], limit_hi[j])


@wp.kernel
def _reach_reward_kernel(
    body_q: wp.array(dtype=wp.transformf),
    ee_body_indices: wp.array(dtype=wp.int32),
    target_pos: wp.array(dtype=wp.vec3f),
    reward_out: wp.array(dtype=wp.float32),
):
    """Position-only reach reward (smoke task): -0.2*dist + 0.1*exp(-dist^2/0.02)."""
    env_idx = wp.tid()
    ee_transform = body_q[ee_body_indices[env_idx]]
    eef_pos = wp.transform_get_translation(ee_transform)
    diff = eef_pos - target_pos[env_idx]
    pos_dist = wp.sqrt(wp.dot(diff, diff) + wp.float32(1e-8))
    reward_out[env_idx] = wp.float32(-0.2) * pos_dist + wp.float32(0.1) * wp.exp(
        -pos_dist * pos_dist / wp.float32(0.02)
    )


@register_env("FrankaReachApg-v0")
class FrankaReachApgEnv(DifferentiableEmbodiedEnv):
    """Differentiable Franka FR3 reach task for analytic policy gradients.

    The environment resolves the Franka FR3 URDF via
    ``newton.utils.download_asset("franka_emika_panda")`` (network-dependent)
    or an explicit ``urdf_path`` kwarg override. The robot is added through
    the standard EmbodiChain ``sim.add_robot(cfg.robot)`` flow driven by
    :class:`EmbodiedEnv`/``BaseEnv.__init__``.

    The differentiable path is:

        action -> new_joint_q (action kernel) -> eval_fk -> body_q
                -> reward kernel -> reward_wp -> tape.backward -> action.grad

    The dynamics solver (semi_implicit) is bypassed because it does not
    propagate gradient through ``joint_target_pos`` to ``body_q`` (the
    stiffness-driven grad path evaluates to zero in practice). This
    matches the reference APG env's workaround.
    """

    metadata = {"render_modes": ["human"], "default_num_envs": 4}

    def __init__(
        self,
        cfg: EmbodiedEnvCfg | None = None,
        *,
        num_envs: int = 4,
        urdf_path: str | None = None,
        action_scale: float = DEFAULT_ACTION_SCALE,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        device: str = "cuda:0",
    ) -> None:
        self._urdf_path = urdf_path
        self._action_scale = float(action_scale)
        self._max_episode_steps = int(max_episode_steps)
        self._device_str = device

        if cfg is None:
            urdf = urdf_path or self._resolve_default_urdf()
            robot_cfg = RobotCfg(
                uid="franka",
                urdf_cfg=URDFCfg().set_urdf(urdf),
                fix_base=True,
            )
            cfg = EmbodiedEnvCfg(
                sim_cfg=SimulationManagerCfg(
                    physics_cfg=NewtonPhysicsCfg(
                        device=device,
                        requires_grad=True,
                        solver_cfg={"solver_type": "semi_implicit"},
                        use_cuda_graph=False,
                    ),
                    num_envs=num_envs,
                    headless=True,
                ),
                robot=robot_cfg,
                num_envs=num_envs,
                max_episode_steps=max_episode_steps,
            )
        # Bug 1 fix: cfg.robot is set BEFORE super().__init__() so that
        # EmbodiedEnv._init_sim_state -> BaseEnv._setup_scene ->
        # _setup_robot -> sim.add_robot(cfg.robot) has a valid robot to
        # add. BaseEnv.__init__ also calls finalize_newton_physics() once
        # the scene is built, so we do NOT re-finalize here.
        super().__init__(cfg)
        # EmbodiedEnv has added the robot and BaseEnv has finalized the
        # Newton model. Cache joint-limit Warp arrays and EE body indices.
        self._cache_franka_buffers()
        self._init_targets()

    # -- scene setup ----------------------------------------------------- #

    def _resolve_default_urdf(self) -> str:
        """Resolve the Franka URDF via Newton's asset cache.

        Raises:
            FileNotFoundError: If the URDF cannot be downloaded or
                located.
        """
        try:
            urdf = newton.utils.download_asset("franka_emika_panda") / (
                "urdf/fr3_franka_hand.urdf"
            )
            if urdf.exists():
                return str(urdf)
        except Exception:
            pass
        raise FileNotFoundError("Franka URDF not available; pass urdf_path explicitly.")

    def _cache_franka_buffers(self) -> None:
        """Cache joint-limit Warp arrays, EE body indices, and FK state."""
        nm = self.sim.physics.newton_manager
        model = nm._model
        # Warp's ``wp.zeros`` / ``wp.launch`` reject ``torch.device``
        # directly (``Invalid device identifier: cuda:0``), so cache the
        # Warp-compatible device string up-front.
        self._wp_device = model.device
        # ``model.joint_limit_lower`` is a ``wp.array``; convert via
        # ``.numpy()`` before slicing (``np.asarray`` on a wp.array slice
        # raises "Item indexing is not supported on wp.array objects").
        lo = model.joint_limit_lower.numpy()[:FRANKA_NUM_ARM_JOINTS].astype(np.float32)
        hi = model.joint_limit_upper.numpy()[:FRANKA_NUM_ARM_JOINTS].astype(np.float32)
        self._limit_lo_t = torch.from_numpy(lo).to(self.device)
        self._limit_hi_t = torch.from_numpy(hi).to(self.device)
        self._limit_lo_wp = wp.array(lo, dtype=wp.float32, device=self._wp_device)
        self._limit_hi_wp = wp.array(hi, dtype=wp.float32, device=self._wp_device)
        self._n_joints_per_env = int(len(model.joint_q) // self.sim.num_envs)
        # Fresh FK state; reused across step calls (eval_fk overwrites it).
        self._fk_state = model.state()
        self._new_joint_q: wp.array | None = None
        # Per-env global EE body indices into the flat body_q array.
        self._ee_global_idx = self._compute_ee_body_indices()
        self._ee_idx_wp = wp.array(
            np.asarray(self._ee_global_idx, dtype=np.int32),
            dtype=wp.int32,
            device=self._wp_device,
        )
        self._ee_idx_t = torch.tensor(
            self._ee_global_idx, dtype=torch.long, device=self.device
        )

    def _compute_ee_body_indices(self) -> list[int]:
        """Scan model.body_label for the EE body per env.

        Each cloned arena produces a full set of Franka bodies in the
        shared Newton model. We pick the ``FRANKA_EE_BODY`` body for
        each env block (one global index per env).
        """
        nm = self.sim.physics.newton_manager
        model = nm._model
        n_envs = self.sim.num_envs
        n_per_env = len(model.body_label) // n_envs
        idx_per_env: list[int] = []
        for i in range(n_envs):
            for j in range(n_per_env):
                global_idx = i * n_per_env + j
                if FRANKA_EE_BODY in str(model.body_label[global_idx]):
                    idx_per_env.append(global_idx)
                    break
        if len(idx_per_env) != n_envs:
            raise RuntimeError(
                f"Expected {n_envs} '{FRANKA_EE_BODY}' bodies, "
                f"found {len(idx_per_env)}."
            )
        return idx_per_env

    def _init_targets(self) -> None:
        n = self.sim.num_envs
        device = self.device
        self.target_pos = torch.zeros(n, 3, device=device)
        self.target_quat = torch.zeros(n, 4, device=device)
        self.last_action = torch.zeros(n, FRANKA_NUM_ARM_JOINTS, device=device)
        self.step_count = torch.zeros(n, dtype=torch.int32, device=device)
        self._sample_new_targets(torch.arange(n, device=device))

    def _sample_new_targets(self, env_ids: torch.Tensor) -> None:
        n = env_ids.numel()
        d = self.device
        self.target_pos[env_ids, 0] = TARGET_POS_RANGE["x"][0] + torch.rand(
            n, device=d
        ) * (TARGET_POS_RANGE["x"][1] - TARGET_POS_RANGE["x"][0])
        self.target_pos[env_ids, 1] = TARGET_POS_RANGE["y"][0] + torch.rand(
            n, device=d
        ) * (TARGET_POS_RANGE["y"][1] - TARGET_POS_RANGE["y"][0])
        self.target_pos[env_ids, 2] = TARGET_POS_RANGE["z"][0] + torch.rand(
            n, device=d
        ) * (TARGET_POS_RANGE["z"][1] - TARGET_POS_RANGE["z"][0])
        # Identity orientation: the smoke task uses position-only reward.
        self.target_quat[env_ids] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=d).expand(
            n, -1
        )

    # -- DifferentiableEmbodiedEnv contract ------------------------------ #

    def _make_step_fn(self) -> Callable[[], Any]:
        """FK bypass: compute body_q from new_joint_q via newton.eval_fk.

        The semi_implicit solver does not propagate grad through
        ``joint_target_pos`` to ``body_q`` (the grad path is zero), so
        we bypass the dynamics solver and run forward kinematics
        directly inside the tape. ``self._new_joint_q`` is populated by
        :meth:`_apply_action_kernel` before this callable runs.
        """
        env = self
        model = env.sim.physics.newton_manager._model

        def _step():
            newton.eval_fk(
                model,
                env._new_joint_q,
                env._fk_state.joint_qd,
                env._fk_state,
            )
            return env._fk_state

        return _step

    def _apply_action_kernel(self, action_wp: Any, tape: Any) -> None:
        """Launch the action-to-control kernel inside the open tape.

        Writes ``new_joint_q = clamp(current_q + action * scale, lo, hi)``
        into a freshly allocated ``self._new_joint_q`` Warp array. The
        FK step function then consumes this array via ``newton.eval_fk``.
        """
        nm = self.sim.physics.newton_manager
        n = self.sim.num_envs
        total = n * FRANKA_NUM_ARM_JOINTS
        # Allocate a fresh new_joint_q each call so each forward pass
        # has its own grad graph (the tape records the kernel writes).
        self._new_joint_q = wp.zeros(
            n * self._n_joints_per_env,
            dtype=wp.float32,
            device=self._wp_device,
            requires_grad=True,
        )
        wp.launch(
            _set_joint_targets_kernel,
            dim=total,
            inputs=[
                action_wp,
                nm._state_0.joint_q,
                self._new_joint_q,
                self._limit_lo_wp,
                self._limit_hi_wp,
                wp.float32(self._action_scale),
                wp.int32(self._n_joints_per_env),
                wp.int32(FRANKA_NUM_ARM_JOINTS),
                wp.int32(total),
            ],
            device=self._wp_device,
        )

    def _read_outputs(self, final_state: Any) -> dict:
        """Launch the reward kernel and build obs INSIDE the open tape.

        Reward is written into a grad-tracked ``reward_wp`` Warp array,
        then exposed as a torch tensor via ``wp.to_torch`` (zero-copy).
        The obs is built from ``wp.to_torch(final_state.joint_q)`` and
        ``wp.to_torch(final_state.body_q)`` (also tape-tracked).
        """
        n = self.sim.num_envs
        device = self._wp_device

        # Grad-tracked reward output array. The kernel launches inside
        # the open tape so reward_wp carries gradient back through the
        # reward kernel -> body_q -> FK -> new_joint_q -> action_wp.
        reward_wp = wp.zeros(n, dtype=wp.float32, device=device, requires_grad=True)
        target_pos_wp = wp.from_torch(
            self.target_pos.detach().clone().contiguous(), dtype=wp.vec3
        )
        wp.launch(
            _reach_reward_kernel,
            dim=n,
            inputs=[final_state.body_q, self._ee_idx_wp, target_pos_wp],
            outputs=[reward_wp],
            device=device,
        )

        joint_q_t = wp.to_torch(final_state.joint_q).view(n, -1)
        body_q_flat = wp.to_torch(final_state.body_q).view(-1, 7)
        ee_pose = body_q_flat[self._ee_idx_t]
        obs = torch.cat(
            [
                joint_q_t[:, :FRANKA_NUM_ARM_JOINTS],
                ee_pose,
                self.target_pos,
                self.target_quat,
                self.last_action,
            ],
            dim=-1,
        )

        reward_t = wp.to_torch(reward_wp)
        pos_dist = (ee_pose[:, :3] - self.target_pos).norm(dim=-1).detach()
        terminated = pos_dist < 0.01
        truncated = self.step_count >= self._max_episode_steps

        return {
            "_order": ("obs", "reward", "terminated", "truncated"),
            "_grad_track": {
                "obs": None,
                "reward": reward_wp,
                "terminated": None,
                "truncated": None,
            },
            "obs": obs,
            "reward": reward_t,
            "terminated": terminated,
            "truncated": truncated,
        }

    # -- gym overrides --------------------------------------------------- #

    def step(self, action: torch.Tensor):
        """Step the env, then advance the cached joint_q for the next call.

        The parent :meth:`DifferentiableEmbodiedEnv.step` runs the
        differentiable bridge. After it returns, we update
        ``nm._state_0.joint_q`` (detached) for envs that were not
        auto-reset so the next step starts from the new configuration.
        """
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32)
        clamped_action = torch.clamp(action.to(self.device), -1.0, 1.0)
        # Advance step_count BEFORE the bridge runs so _read_outputs
        # computes truncated against the post-step value.
        self.step_count += 1
        result = super().step(clamped_action)
        obs, reward, terminated, truncated, info = result
        done_mask = terminated | truncated
        live = (~done_mask).nonzero(as_tuple=False).squeeze(-1)
        if live.numel() > 0:
            with torch.no_grad():
                nm = self.sim.physics.newton_manager
                joint_q_t = wp.to_torch(nm._state_0.joint_q).view(self.sim.num_envs, -1)
                cur = joint_q_t[live, :FRANKA_NUM_ARM_JOINTS]
                delta = clamped_action[live].detach() * self._action_scale
                lo = self._limit_lo_t.unsqueeze(0).expand_as(cur)
                hi = self._limit_hi_t.unsqueeze(0).expand_as(cur)
                joint_q_t[live, :FRANKA_NUM_ARM_JOINTS] = torch.clamp(
                    cur + delta, lo, hi
                )
        self.last_action = clamped_action.detach().clone()
        return obs, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        """Reset joint_q, targets, and step_count for the touched envs.

        Args:
            seed: Optional RNG seed for deterministic resets.
            options: Optional dict; supports ``{"reset_ids": <tensor>}``
                for partial resets (used by auto-reset in
                :meth:`DifferentiableEmbodiedEnv.step`).

        Returns:
            Tuple of ``(obs, info)``.
        """
        if seed is not None:
            torch.manual_seed(seed)
        if options is None:
            options = {}
        reset_ids = options.get("reset_ids")
        if reset_ids is None:
            env_ids = torch.arange(self.sim.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(reset_ids, dtype=torch.long, device=self.device)
        with torch.no_grad():
            self.step_count[env_ids] = 0
            self.last_action[env_ids] = 0.0
            self._sample_new_targets(env_ids)
            nm = self.sim.physics.newton_manager
            joint_q_t = wp.to_torch(nm._state_0.joint_q).view(self.sim.num_envs, -1)
            joint_q_t[env_ids] = 0.0
            newton.eval_fk(
                nm._model,
                nm._state_0.joint_q,
                nm._state_0.joint_qd,
                nm._state_0,
            )
        obs = self._initial_obs()
        return obs, {}

    def _initial_obs(self) -> torch.Tensor:
        """Compute the initial obs from state_0 (no grad, no side effects)."""
        with torch.no_grad():
            nm = self.sim.physics.newton_manager
            state = nm._state_0
            n = self.sim.num_envs
            joint_q_t = wp.to_torch(state.joint_q).view(n, -1)
            body_q_flat = wp.to_torch(state.body_q).view(-1, 7)
            ee_pose = body_q_flat[self._ee_idx_t]
            obs = torch.cat(
                [
                    joint_q_t[:, :FRANKA_NUM_ARM_JOINTS],
                    ee_pose,
                    self.target_pos,
                    self.target_quat,
                    self.last_action,
                ],
                dim=-1,
            )
            return obs.detach()

    def close(self) -> None:
        """Close the environment and release resources."""
        self.sim.destroy()
