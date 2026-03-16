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

from __future__ import annotations

from typing import Callable

import torch
from tensordict import TensorDict

from embodichain.agents.rl.utils import dict_to_tensordict, flatten_dict_observation
from .base import BaseCollector

__all__ = ["SyncCollector"]


class SyncCollector(BaseCollector):
    """Synchronously collect rollouts from a vectorized environment."""

    def __init__(
        self,
        env,
        policy,
        device: torch.device,
        reset_every_rollout: bool = False,
    ) -> None:
        self.env = env
        self.policy = policy
        self.device = device
        self.reset_every_rollout = reset_every_rollout
        self._supports_shared_rollout = hasattr(self.env, "set_rollout_buffer")
        self.obs_td = self._reset_env()

    @torch.no_grad()
    def collect(
        self,
        num_steps: int,
        rollout: TensorDict | None = None,
        on_step_callback: Callable[[TensorDict, dict], None] | None = None,
    ) -> TensorDict:
        self.policy.train()
        if self.reset_every_rollout:
            self.obs_td = self._reset_env()

        if rollout is None:
            raise ValueError(
                "SyncCollector.collect() requires a preallocated rollout TensorDict."
            )
        if tuple(rollout.batch_size) != (self.env.num_envs, num_steps + 1):
            raise ValueError(
                "Preallocated rollout batch size mismatch: "
                f"expected ({self.env.num_envs}, {num_steps + 1}), got {tuple(rollout.batch_size)}."
            )
        self._validate_rollout(rollout, num_steps)
        if self._supports_shared_rollout:
            self.env.set_rollout_buffer(rollout)

        use_raw_obs = getattr(self.policy, "use_raw_obs", False)
        raw_obs_list = getattr(rollout, "raw_obs", None) if use_raw_obs else None

        if use_raw_obs:
            if raw_obs_list is None:
                raise ValueError(
                    "Policy requires raw observations, "
                    "but the provided rollout TensorDict has no 'raw_obs' buffer. "
                    "Create the rollout via RolloutBuffer or "
                    "start_rollout so that 'raw_obs' is allocated."
                )
            try:
                raw_obs_len = len(raw_obs_list)
            except TypeError:
                raise ValueError(
                    "Rollout field 'raw_obs' must be an indexable sequence of length "
                    f"{num_steps + 1} when policy.use_raw_obs=True."
                )
            expected_len = num_steps + 1
            if raw_obs_len != expected_len:
                raise ValueError(
                    "Rollout 'raw_obs' length mismatch: "
                    f"expected {expected_len} (num_steps + 1), got {raw_obs_len}. "
                    "Ensure the rollout was created with use_raw_obs=True and "
                    "its time dimension matches the requested num_steps."
                )

        action_chunk_size = getattr(self.policy, "action_chunk_size", 0)
        use_action_chunk = (
            getattr(self.policy, "use_action_chunk", False) and action_chunk_size > 0
        )
        cached_chunk = None

        if use_action_chunk:
            rollout.chunk_step = torch.zeros(
                self.env.num_envs,
                num_steps,
                dtype=torch.long,
                device=self.device,
            )

        if use_raw_obs and raw_obs_list is not None:
            raw_obs_list[0] = self.obs_td
            rollout["obs"][:, 0] = flatten_dict_observation(self.obs_td)
        else:
            rollout["obs"][:, 0] = flatten_dict_observation(self.obs_td)

        for step_idx in range(num_steps):
            step_in_chunk = step_idx % action_chunk_size if use_action_chunk else 0

            # At chunk boundary, or cached invalidated by env reset, we need a new chunk
            need_new_chunk = use_action_chunk and (
                step_in_chunk == 0 or cached_chunk is None
            )

            if need_new_chunk:
                if use_raw_obs and raw_obs_list is not None:
                    step_td = TensorDict(
                        {"obs": raw_obs_list[step_idx]},
                        batch_size=[rollout.batch_size[0]],
                        device=self.device,
                    )
                else:
                    step_td = TensorDict(
                        {"obs": rollout["obs"][:, step_idx]},
                        batch_size=[rollout.batch_size[0]],
                        device=self.device,
                    )
                step_td = self.policy.get_action(step_td)
                cached_chunk = step_td["action_chunk"]
                action = step_td["action"]
                effective_step_in_chunk = 0
            elif use_action_chunk and cached_chunk is not None:
                action = cached_chunk[:, step_in_chunk]
                effective_step_in_chunk = step_in_chunk
                step_td = TensorDict(
                    {
                        "action": action,
                        "sample_log_prob": torch.zeros(
                            action.shape[0], device=self.device, dtype=torch.float32
                        ),
                        "value": torch.zeros(
                            action.shape[0], device=self.device, dtype=torch.float32
                        ),
                    },
                    batch_size=[rollout.batch_size[0]],
                    device=self.device,
                )
            else:
                if use_raw_obs and raw_obs_list is not None:
                    step_td = TensorDict(
                        {"obs": raw_obs_list[step_idx]},
                        batch_size=[rollout.batch_size[0]],
                        device=self.device,
                    )
                else:
                    step_td = TensorDict(
                        {"obs": rollout["obs"][:, step_idx]},
                        batch_size=[rollout.batch_size[0]],
                        device=self.device,
                    )
                step_td = self.policy.get_action(step_td)
                action = step_td["action"]

            next_obs, reward, terminated, truncated, env_info = self.env.step(
                self._to_action_dict(action)
            )
            next_obs_td = dict_to_tensordict(next_obs, self.device)
            if use_action_chunk:
                rollout.chunk_step[:, step_idx] = effective_step_in_chunk
                # Invalidate cached_chunk on any env reset to avoid using old chunk for new episode
                if (terminated | truncated).any():
                    cached_chunk = None
            self._write_step(
                rollout=rollout,
                step_idx=step_idx,
                step_td=step_td,
            )
            if not self._supports_shared_rollout:
                self._write_env_step(
                    rollout=rollout,
                    step_idx=step_idx,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                )
            if use_raw_obs and raw_obs_list is not None:
                raw_obs_list[step_idx + 1] = next_obs_td
                rollout["obs"][:, step_idx + 1] = flatten_dict_observation(next_obs_td)
            else:
                rollout["obs"][:, step_idx + 1] = flatten_dict_observation(next_obs_td)

            if on_step_callback is not None:
                on_step_callback(rollout[:, step_idx], env_info)

            self.obs_td = next_obs_td

        self._attach_final_value(rollout)
        return rollout

    def _attach_final_value(self, rollout: TensorDict) -> None:
        """Populate the bootstrap value for the final observed state."""
        use_raw_obs = getattr(self.policy, "use_raw_obs", False)
        raw_obs_list = getattr(rollout, "raw_obs", None) if use_raw_obs else None
        if use_raw_obs and raw_obs_list is not None:
            final_obs = raw_obs_list[-1]
        else:
            final_obs = rollout["obs"][:, -1]
        last_next_td = TensorDict(
            {"obs": final_obs},
            batch_size=[rollout.batch_size[0]],
            device=self.device,
        )
        self.policy.get_value(last_next_td)
        rollout["value"][:, -1] = last_next_td["value"]

    def _reset_env(self) -> TensorDict:
        obs, _ = self.env.reset()
        return dict_to_tensordict(obs, self.device)

    def _to_action_dict(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        am = getattr(self.env, "action_manager", None)
        action_type = (
            am.action_type if am else getattr(self.env, "action_type", "delta_qpos")
        )
        return {action_type: action}

    def _write_step(
        self,
        rollout: TensorDict,
        step_idx: int,
        step_td: TensorDict,
    ) -> None:
        """Write policy-side fields for one transition into the shared rollout TensorDict."""
        rollout["action"][:, step_idx] = step_td["action"]
        rollout["sample_log_prob"][:, step_idx] = step_td["sample_log_prob"]
        rollout["value"][:, step_idx] = step_td["value"]

    def _write_env_step(
        self,
        rollout: TensorDict,
        step_idx: int,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        """Populate transition-side fields when the environment does not own the rollout."""
        done = terminated | truncated
        rollout["reward"][:, step_idx] = reward.to(self.device)
        rollout["done"][:, step_idx] = done.to(self.device)
        rollout["terminated"][:, step_idx] = terminated.to(self.device)
        rollout["truncated"][:, step_idx] = truncated.to(self.device)

    def _validate_rollout(self, rollout: TensorDict, num_steps: int) -> None:
        """Validate rollout layout expected by the collector."""
        obs_dim = rollout["obs"].shape[-1]
        expected_shapes = {
            "obs": (self.env.num_envs, num_steps + 1, obs_dim),
            "action": (self.env.num_envs, num_steps + 1, self.policy.action_dim),
            "sample_log_prob": (self.env.num_envs, num_steps + 1),
            "value": (self.env.num_envs, num_steps + 1),
            "reward": (self.env.num_envs, num_steps + 1),
            "done": (self.env.num_envs, num_steps + 1),
            "terminated": (self.env.num_envs, num_steps + 1),
            "truncated": (self.env.num_envs, num_steps + 1),
        }
        for key, expected_shape in expected_shapes.items():
            actual_shape = tuple(rollout[key].shape)
            if actual_shape != expected_shape:
                raise ValueError(
                    f"Preallocated rollout field '{key}' shape mismatch: "
                    f"expected {expected_shape}, got {actual_shape}."
                )
