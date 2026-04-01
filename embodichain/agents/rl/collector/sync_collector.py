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
from embodichain.utils import logger
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
        execute_full_chunk = bool(getattr(self.policy, "execute_full_chunk", False))
        self._supports_shared_rollout = (
            hasattr(self.env, "set_rollout_buffer") and not execute_full_chunk
        )
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
            logger.log_error(
                "SyncCollector.collect() requires a preallocated rollout TensorDict.",
                ValueError,
            )
        if tuple(rollout.batch_size) != (self.env.num_envs, num_steps + 1):
            logger.log_error(
                "Preallocated rollout batch size mismatch: "
                f"expected ({self.env.num_envs}, {num_steps + 1}), got {tuple(rollout.batch_size)}.",
                ValueError,
            )
        self._validate_rollout(rollout, num_steps)
        if self._supports_shared_rollout:
            self.env.set_rollout_buffer(rollout)

        use_raw_obs = getattr(self.policy, "use_raw_obs", False)
        raw_obs_list = getattr(rollout, "raw_obs", None) if use_raw_obs else None

        if use_raw_obs:
            if raw_obs_list is None:
                logger.log_error(
                    "Policy requires raw observations, "
                    "but the provided rollout TensorDict has no 'raw_obs' buffer. "
                    "Create the rollout via RolloutBuffer or "
                    "start_rollout so that 'raw_obs' is allocated.",
                    ValueError,
                )
            try:
                raw_obs_len = len(raw_obs_list)
            except TypeError:
                logger.log_error(
                    "Rollout field 'raw_obs' must be an indexable sequence of length "
                    f"{num_steps + 1} when policy.use_raw_obs=True.",
                    ValueError,
                )
            expected_len = num_steps + 1
            if raw_obs_len != expected_len:
                logger.log_error(
                    "Rollout 'raw_obs' length mismatch: "
                    f"expected {expected_len} (num_steps + 1), got {raw_obs_len}. "
                    "Ensure the rollout was created with use_raw_obs=True and "
                    "its time dimension matches the requested num_steps.",
                    ValueError,
                )

        action_chunk_size = getattr(self.policy, "action_chunk_size", 0)
        use_action_chunk = (
            getattr(self.policy, "use_action_chunk", False) and action_chunk_size > 0
        )
        # Execute a full predicted action chunk inside one logical rollout step.
        execute_full_chunk = bool(getattr(self.policy, "execute_full_chunk", False))
        cached_chunk = None
        chunk_cursor = 0

        if use_action_chunk:
            rollout.chunk_step = torch.zeros(
                self.env.num_envs,
                num_steps,
                dtype=torch.long,
                device=self.device,
            )
        rollout["step_repeat"] = torch.ones(
            self.env.num_envs,
            num_steps + 1,
            dtype=torch.float32,
            device=self.device,
        )
        rollout["step_repeat"][:, -1] = 0.0

        if use_raw_obs and raw_obs_list is not None:
            raw_obs_list[0] = self.obs_td
            rollout["obs"][:, 0] = flatten_dict_observation(self.obs_td)
        else:
            rollout["obs"][:, 0] = flatten_dict_observation(self.obs_td)

        for step_idx in range(num_steps):
            if execute_full_chunk and use_action_chunk:
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
                chunk = step_td.get("action_chunk")
                if chunk is None:
                    logger.log_error(
                        "execute_full_chunk=True requires policy to provide 'action_chunk'.",
                        ValueError,
                    )

                reward_sum = torch.zeros(
                    self.env.num_envs, dtype=torch.float32, device=self.device
                )
                terminated = torch.zeros(
                    self.env.num_envs, dtype=torch.bool, device=self.device
                )
                truncated = torch.zeros(
                    self.env.num_envs, dtype=torch.bool, device=self.device
                )
                env_info = {}
                next_obs_td = None

                executed_substeps = 0
                # Execute the whole chunk sequentially
                for sub_idx in range(action_chunk_size):
                    sub_action = chunk[:, sub_idx]
                    next_obs, reward, term_i, trunc_i, env_info = self.env.step(
                        self._to_action_dict(sub_action)
                    )
                    executed_substeps += 1
                    next_obs_td = dict_to_tensordict(next_obs, self.device)
                    reward_sum += reward.to(self.device).float()
                    terminated |= term_i.to(self.device)
                    truncated |= trunc_i.to(self.device)

                    # Stop chunk execution when any env reaches terminal/truncated.
                    if (term_i | trunc_i).any():
                        break

                if next_obs_td is None:
                    logger.log_error(
                        "Chunk execution produced no environment transition.",
                        RuntimeError,
                    )

                if use_action_chunk:
                    rollout.chunk_step[:, step_idx] = 0
                rollout["step_repeat"][:, step_idx] = float(executed_substeps)

                self._write_step(
                    rollout=rollout,
                    step_idx=step_idx,
                    step_td=step_td,
                )
                if not self._supports_shared_rollout:
                    self._write_env_step(
                        rollout=rollout,
                        step_idx=step_idx,
                        reward=reward_sum,
                        terminated=terminated,
                        truncated=truncated,
                    )
                if use_raw_obs and raw_obs_list is not None:
                    raw_obs_list[step_idx + 1] = next_obs_td
                    rollout["obs"][:, step_idx + 1] = flatten_dict_observation(
                        next_obs_td
                    )
                else:
                    rollout["obs"][:, step_idx + 1] = flatten_dict_observation(
                        next_obs_td
                    )

                if on_step_callback is not None:
                    on_step_callback(rollout[:, step_idx], env_info)

                self.obs_td = next_obs_td
                continue

            # Execute a predicted chunk sequentially
            need_new_chunk = use_action_chunk and (
                cached_chunk is None or chunk_cursor >= action_chunk_size
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
                chunk_cursor = 1
            elif use_action_chunk and cached_chunk is not None:
                action = cached_chunk[:, chunk_cursor]
                effective_step_in_chunk = chunk_cursor
                step_td = TensorDict(
                    {
                        "action": action,
                        "action_chunk": cached_chunk,
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
                chunk_cursor += 1
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
            rollout["step_repeat"][:, step_idx] = 1.0
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

    def _to_action_dict(self, action: torch.Tensor) -> TensorDict | torch.Tensor:
        am = getattr(self.env, "action_manager", None)
        if am is None:
            return action
        else:
            return am.convert_policy_action_to_env_action(action)

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
        if "action_chunk" in rollout.keys() and "action_chunk" in step_td.keys():
            rollout["action_chunk"][:, step_idx] = step_td["action_chunk"]

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
        num_envs = self.env.num_envs
        time_plus_one = num_steps + 1
        policy_obs_dim = int(getattr(self.policy, "obs_dim", 0) or 0)
        obs_shape = tuple(rollout["obs"].shape)
        if policy_obs_dim > 0:
            expected_obs = (num_envs, time_plus_one, policy_obs_dim)
            if obs_shape != expected_obs:
                logger.log_error(
                    f"Preallocated rollout field 'obs' shape mismatch: "
                    f"expected {expected_obs}, got {obs_shape}.",
                    ValueError,
                )
        else:
            if (
                len(obs_shape) != 3
                or obs_shape[0] != num_envs
                or obs_shape[1] != time_plus_one
            ):
                logger.log_error(
                    f"Preallocated rollout field 'obs' shape mismatch: "
                    f"expected ({num_envs}, {time_plus_one}, *), got {obs_shape}.",
                    ValueError,
                )

        expected_shapes = {
            "action": (num_envs, time_plus_one, self.policy.action_dim),
            "sample_log_prob": (num_envs, time_plus_one),
            "value": (num_envs, time_plus_one),
            "reward": (num_envs, time_plus_one),
            "done": (num_envs, time_plus_one),
            "terminated": (num_envs, time_plus_one),
            "truncated": (num_envs, time_plus_one),
        }
        for key, expected_shape in expected_shapes.items():
            actual_shape = tuple(rollout[key].shape)
            if actual_shape != expected_shape:
                logger.log_error(
                    f"Preallocated rollout field '{key}' shape mismatch: "
                    f"expected {expected_shape}, got {actual_shape}.",
                    ValueError,
                )
