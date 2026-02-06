# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

from typing import Dict, Any, Tuple, Callable
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import wandb
from tensordict import TensorDict

from embodichain.lab.gym.envs.managers.event_manager import EventManager
from .helper import dict_to_tensordict, mean_scalar, pack_log_dict


class Trainer:
    """Algorithm-agnostic trainer that coordinates training loop, logging, and evaluation."""

    def __init__(
        self,
        policy,
        env,
        algorithm,
        buffer_size: int,
        batch_size: int,
        writer: SummaryWriter | None,
        eval_freq: int,
        save_freq: int,
        checkpoint_dir: str,
        exp_name: str,
        use_wandb: bool = True,
        eval_env=None,
        event_cfg=None,
        eval_event_cfg=None,
        num_eval_episodes: int = 5,
        # Model type: "standard" (default PPO) or "vla"
        model_type: str = "standard",
    ):
        self.policy = policy
        self.env = env
        self.eval_env = eval_env
        self.algorithm = algorithm
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.writer = writer
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.checkpoint_dir = checkpoint_dir
        self.exp_name = exp_name
        self.use_wandb = use_wandb
        self.num_eval_episodes = num_eval_episodes

        # Buffer setup (depends on model_type)
        self.model_type = model_type
        device = (
            algorithm.device
            if hasattr(algorithm, "device")
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        if model_type == "vla":
            # VLA model: accumulate multiple rollouts with FIFO buffer
            from embodichain.agents.rl.buffer import VLABuffer

            self.buffer = VLABuffer(buffer_size=buffer_size, device=device)
        elif model_type == "standard":
            # Standard PPO model: single rollout, use and discard
            from embodichain.agents.rl.buffer import RolloutBuffer

            self.buffer = RolloutBuffer(buffer_size=buffer_size, device=device)
        else:
            raise ValueError(
                f"Unknown model_type: {model_type}. Use 'standard' or 'vla'."
            )

        if event_cfg is not None:
            self.event_manager = EventManager(event_cfg, env=self.env)
        if eval_event_cfg is not None:
            self.eval_event_manager = EventManager(eval_event_cfg, env=self.eval_env)

        # Get device from algorithm
        self.device = self.algorithm.device
        self.global_step = 0
        self.start_time = time.time()
        self.ret_window = deque(maxlen=100)
        self.len_window = deque(maxlen=100)

        # Get initial observation from env (dict) and convert to TensorDict at boundary
        obs, _ = self.env.reset()
        self.obs_tensordict = dict_to_tensordict(obs, self.device)
        num_envs = self.obs_tensordict.batch_size[0]

        # episode stats tracked on device to avoid repeated CPU round-trips
        self.curr_ret = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.curr_len = torch.zeros(num_envs, dtype=torch.int32, device=self.device)

    # ---- lightweight helpers for dense logging ----
    def _log_scalar_dict(self, prefix: str, data: dict):
        if not self.writer or not isinstance(data, dict):
            return
        for k, v in data.items():
            try:
                self.writer.add_scalar(
                    f"{prefix}/{k}", mean_scalar(v), self.global_step
                )
            except Exception:
                continue

    def train(self, total_timesteps: int):
        print(f"Start training, total steps: {total_timesteps}")
        print(f"Model type: {self.model_type}")

        if self.model_type == "vla":
            # VLA mode: Use async collector
            self._train_async(total_timesteps)
        else:
            # Standard mode: Use synchronous collection
            self._train_sync(total_timesteps)

    def _train_sync(self, total_timesteps: int):
        """Synchronous training loop (standard PPO)."""
        while self.global_step < total_timesteps:
            rollout = self._collect_rollout()

            # Add rollout to buffer
            self.buffer.add(rollout)

            # Train when buffer is full
            if self.buffer.is_full():
                data = self.buffer.get(flatten=True)
                losses = self.algorithm.update(data)
                self._log_train(losses)

            # Evaluation
            if (
                self.eval_freq > 0
                and self.eval_env is not None
                and self.global_step % self.eval_freq == 0
            ):
                self._eval_once(num_episodes=self.num_eval_episodes)

            # Checkpoint
            if self.global_step % self.save_freq == 0:
                self.save_checkpoint()

    def _train_async(self, total_timesteps: int):
        """Asynchronous training loop (VLA mode)."""
        from .async_collector import AsyncCollector, AsyncCollectorStats

        # Create statistics tracker
        num_envs = self.obs_tensordict.batch_size[0]
        async_stats = AsyncCollectorStats(num_envs, self.device)

        # Create callback for async collector
        def on_async_step(tensordict: TensorDict, env_info: dict):
            """Callback for async collection statistics."""
            # Extract reward and done
            reward = tensordict["next"]["reward"]
            done = tensordict["next"]["done"]

            # Update statistics
            async_stats.update(reward, done)

            # Update global step
            num_envs = tensordict.batch_size[0]
            self.global_step += num_envs

            # Log environment metrics
            if isinstance(env_info, dict):
                rewards_dict = env_info.get("rewards")
                metrics_dict = env_info.get("metrics")
                self._log_scalar_dict("rewards", rewards_dict)
                self._log_scalar_dict("metrics", metrics_dict)
                log_dict = {}
                log_dict.update(pack_log_dict("rewards", rewards_dict))
                log_dict.update(pack_log_dict("metrics", metrics_dict))
                if log_dict and self.use_wandb:
                    wandb.log(log_dict, step=self.global_step)

        # Create and start async collector
        collector = AsyncCollector(
            env=self.env,
            policy=self.policy,
            buffer=self.buffer,
            device=self.device,
            on_step_callback=on_async_step,
        )

        print("[Trainer] Starting async collector...")
        collector.start()

        # Training loop: wait for buffer to fill, then train
        last_eval_step = 0
        last_save_step = 0
        update_count = 0

        try:
            while self.global_step < total_timesteps:
                # Wait for buffer to fill
                while not self.buffer.is_full():
                    time.sleep(0.1)  # Check every 100ms
                    if not collector.is_running():
                        raise RuntimeError("Async collector stopped unexpectedly")

                # Get data and train
                data = self.buffer.get(flatten=True)
                losses = self.algorithm.update(data)

                # Update episode statistics from async tracker
                avg_ret, avg_len = async_stats.get_avg_stats()
                if not np.isnan(avg_ret):
                    self.ret_window.append(avg_ret)
                if not np.isnan(avg_len):
                    self.len_window.append(avg_len)

                # Log training
                self._log_train(losses)
                update_count += 1

                # Clear buffer for next collection (optional, depends on policy staleness tolerance)
                # For VLA, we might keep some data for stability
                # self.buffer.clear()

                # Evaluation
                if (
                    self.eval_freq > 0
                    and self.eval_env is not None
                    and self.global_step - last_eval_step >= self.eval_freq
                ):
                    # Temporarily pause collection during eval
                    print("[Trainer] Pausing collection for evaluation...")
                    collector.stop()
                    self._eval_once(num_episodes=self.num_eval_episodes)
                    collector.start()
                    print("[Trainer] Resuming collection...")
                    last_eval_step = self.global_step

                # Checkpoint
                if self.global_step - last_save_step >= self.save_freq:
                    self.save_checkpoint()
                    last_save_step = self.global_step

                    # Log buffer and collector stats
                    buffer_stats = self.buffer.get_stats()
                    collector_stats = collector.get_stats()
                    print(f"[Trainer] Buffer: {buffer_stats}")
                    print(f"[Trainer] Collector: {collector_stats}")

        finally:
            # Always stop collector when training ends
            print("[Trainer] Stopping async collector...")
            collector.stop()
            print(f"[Trainer] Training completed ({update_count} updates)")

    @torch.no_grad()
    def _collect_rollout(self) -> TensorDict:
        """Collect a rollout. Algorithm controls the data collection process.

        Returns:
            TensorDict with batch_size=[T, N] containing full rollout
        """

        # Callback function for statistics and logging (uses TensorDict)
        def on_step(tensordict: TensorDict, env_info: dict):
            """Callback called at each step during rollout collection.

            Args:
                tensordict: Complete transition TensorDict with "next" key
                env_info: Environment info dict
            """
            # Extract reward and done from next subdictionary
            reward = tensordict["next"]["reward"]
            done = tensordict["next"]["done"]

            # Squeeze if needed
            if reward.dim() > 1:
                reward = reward.squeeze(-1)
            if done.dim() > 1:
                done = done.squeeze(-1)

            # Episode stats (stay on device; convert only when episode ends)
            self.curr_ret += reward
            self.curr_len += 1
            done_idx = torch.nonzero(done, as_tuple=False).squeeze(-1)
            if done_idx.numel() > 0:
                finished_ret = self.curr_ret[done_idx].detach().cpu().tolist()
                finished_len = self.curr_len[done_idx].detach().cpu().tolist()
                self.ret_window.extend(finished_ret)
                self.len_window.extend(finished_len)
                self.curr_ret[done_idx] = 0
                self.curr_len[done_idx] = 0

            # Update global step and observation TensorDict (already TensorDict from PPO)
            next_obs = tensordict["next"]["observation"]
            num_envs = next_obs.batch_size[0]

            # Prepare next tensordict
            self.obs_tensordict = TensorDict(
                {"observation": next_obs},
                batch_size=torch.Size([num_envs]),
                device=self.device,
            )
            self.global_step += num_envs

            if isinstance(env_info, dict):
                rewards_dict = env_info.get("rewards")
                metrics_dict = env_info.get("metrics")
                self._log_scalar_dict("rewards", rewards_dict)
                self._log_scalar_dict("metrics", metrics_dict)
                log_dict = {}
                log_dict.update(pack_log_dict("rewards", rewards_dict))
                log_dict.update(pack_log_dict("metrics", metrics_dict))
                if log_dict and self.use_wandb:
                    wandb.log(log_dict, step=self.global_step)

        # Algorithm controls data collection and returns TensorDict rollout
        rollout = self.algorithm.collect_rollout(
            env=self.env,
            policy=self.policy,
            tensordict=self.obs_tensordict,
            buffer_size=self.buffer_size,
            on_step_callback=on_step,
        )

        return rollout

    def _log_train(self, losses: Dict[str, float]):
        if self.writer:
            for k, v in losses.items():
                self.writer.add_scalar(f"train/{k}", v, self.global_step)
            elapsed = max(1e-6, time.time() - self.start_time)
            sps = self.global_step / elapsed
            self.writer.add_scalar("charts/SPS", sps, self.global_step)
            if len(self.ret_window) > 0:
                self.writer.add_scalar(
                    "charts/episode_reward_avg_100",
                    float(np.mean(self.ret_window)),
                    self.global_step,
                )
            if len(self.len_window) > 0:
                self.writer.add_scalar(
                    "charts/episode_length_avg_100",
                    float(np.mean(self.len_window)),
                    self.global_step,
                )
        # console
        sps = self.global_step / max(1e-6, time.time() - self.start_time)
        avgR = np.mean(self.ret_window) if len(self.ret_window) > 0 else float("nan")
        avgL = np.mean(self.len_window) if len(self.len_window) > 0 else float("nan")
        print(
            f"[train] step={self.global_step} sps={sps:.0f} avgReward(100)={avgR:.3f} avgLength(100)={avgL:.1f}"
        )

        # wandb (mirror TB logs)
        if self.use_wandb:
            log_dict = {f"train/{k}": v for k, v in losses.items()}
            log_dict["charts/SPS"] = sps
            if not np.isnan(avgR):
                log_dict["charts/episode_reward_avg_100"] = float(avgR)
            if not np.isnan(avgL):
                log_dict["charts/episode_length_avg_100"] = float(avgL)
            wandb.log(log_dict, step=self.global_step)

    @torch.no_grad()
    def _eval_once(self, num_episodes: int = 5):
        """Run evaluation for specified number of episodes.

        Each episode runs all parallel environments until completion, allowing
        environments to finish at different times.

        Args:
            num_episodes: Number of episodes to evaluate
        """
        self.policy.eval()
        episode_returns = []
        episode_lengths = []

        for _ in range(num_episodes):
            # Reset and initialize episode tracking - env returns dict, convert at boundary
            obs, _ = self.eval_env.reset()
            obs = dict_to_tensordict(obs, self.device)
            num_envs = obs.batch_size[0]

            done_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
            cumulative_reward = torch.zeros(
                num_envs, dtype=torch.float32, device=self.device
            )
            step_count = torch.zeros(num_envs, dtype=torch.int32, device=self.device)

            # Run episode until all environments complete
            while not done_mask.all():
                # Get deterministic actions from policy (policy.forward modifies in-place)
                # For deterministic eval, we can set a flag or use mean directly
                # For now, use forward and extract action
                obs_copy = obs.clone()
                self.policy.forward(obs_copy)
                actions = obs_copy["action"]

                action_type = getattr(self.eval_env, "action_type", "delta_qpos")
                action_dict = {action_type: actions}

                # Environment step - env returns dict, convert to TensorDict at boundary
                next_obs, reward, terminated, truncated, info = self.eval_env.step(
                    action_dict
                )
                next_obs = dict_to_tensordict(next_obs, self.device)
                obs = next_obs

                # Update statistics only for still-running environments
                done = terminated | truncated
                still_running = ~done_mask
                cumulative_reward[still_running] += reward[still_running].float()
                step_count[still_running] += 1
                done_mask |= done

                # Trigger evaluation events (e.g., video recording)
                if hasattr(self, "eval_event_manager"):
                    if "interval" in self.eval_event_manager.available_modes:
                        self.eval_event_manager.apply(mode="interval")

            # Collect episode statistics
            episode_returns.extend(cumulative_reward.cpu().tolist())
            episode_lengths.extend(step_count.cpu().tolist())

        # Finalize evaluation functors (e.g., video recording)
        if hasattr(self, "eval_event_manager"):
            for functor_cfg in self.eval_event_manager._mode_functor_cfgs.get(
                "interval", []
            ):
                functor = functor_cfg.func
                save_path = functor_cfg.params.get("save_path", "./outputs/videos/eval")

                if hasattr(functor, "flush"):
                    functor.flush(save_path)
                if hasattr(functor, "finalize"):
                    functor.finalize(save_path)

        # Log evaluation metrics
        if self.writer and episode_returns:
            self.writer.add_scalar(
                "eval/avg_reward", float(np.mean(episode_returns)), self.global_step
            )
            self.writer.add_scalar(
                "eval/avg_length", float(np.mean(episode_lengths)), self.global_step
            )

    def save_checkpoint(self):
        # minimal model-only checkpoint; trainer/algorithm states can be added
        path = f"{self.checkpoint_dir}/{self.exp_name}_step_{self.global_step}.pt"
        torch.save(
            {
                "global_step": self.global_step,
                "policy": self.policy.state_dict(),
            },
            path,
        )
        print(f"Checkpoint saved: {path}")
