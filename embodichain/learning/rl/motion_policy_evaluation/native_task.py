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

"""Connect an EmbodiChain RL task to DexSim Motion Policy Evaluator."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from dexsim.kit.motion_policy import (
    EvaluationFrame,
    PolicyContext,
    PolicyOutput,
    RunOptions,
    create_motion_policy_evaluator,
)
from dexsim.kit.motion_policy.types import EnvironmentStep

from embodichain.learning.rl.evaluation import (
    convert_policy_action_for_env,
    infer_policy_action,
)
from embodichain.learning.rl.runtime import PolicyRuntime

__all__ = [
    "EmbodiChainTaskEnvironment",
    "EmbodiChainTaskPolicyAdapter",
    "NativeTaskEvaluationResult",
    "evaluate_native_task",
]

_MISSING = object()


@dataclass(frozen=True)
class NativeTaskEvaluationResult:
    """Result of evaluating one Policy in its original EmbodiChain task."""

    task_id: str
    reason: str
    simulation_time: float
    simulation_steps: int
    control_steps: int
    physics_backend: str
    effective_duration: float
    requested_duration: float | None
    episodes: tuple[Mapping[str, float | int | bool | str], ...]
    metrics: Mapping[str, float]
    viewer: bool


class EmbodiChainTaskPolicyAdapter:
    """Run an EmbodiChain Policy from the task observation in each frame."""

    def __init__(self, policy: torch.nn.Module, device: torch.device, num_envs: int):
        self.policy = policy
        self.device = device
        self.num_envs = num_envs
        self._previous_training = policy.training

    def setup(self, context: PolicyContext) -> None:
        """Select deterministic inference for this evaluation."""
        del context
        self.policy.eval()

    def reset(self, frame: EvaluationFrame) -> None:
        """Validate that the Environment supplied the next observation."""
        if frame.observation is None:
            raise RuntimeError("EmbodiChain task frame has no observation")

    @torch.no_grad()
    def infer(self, frame: EvaluationFrame) -> PolicyOutput:
        """Run the same observation and deterministic Policy path as RL evaluation."""
        if frame.observation is None:
            raise RuntimeError("EmbodiChain task frame has no observation")
        action = infer_policy_action(
            self.policy,
            frame.observation,
            device=self.device,
            num_envs=self.num_envs,
        )
        return PolicyOutput(action=action)

    def metrics(self) -> dict[str, float]:
        """Return Policy-side metrics."""
        return {}

    def close(self) -> None:
        """Restore the Policy mode used before evaluation."""
        self.policy.train(self._previous_training)


class EmbodiChainTaskEnvironment:
    """Expose one original EmbodiChain RL Environment to the Evaluator."""

    def __init__(
        self,
        env: Any,
        *,
        seed: int,
        viewer: bool,
    ) -> None:
        if int(env.num_envs) != 1:
            raise ValueError("Visual task evaluation requires num_envs=1")
        self.env = env
        self._base_env = getattr(env, "unwrapped", env)
        self._viewer = viewer
        if viewer:
            world = self._world()
            if world is None or not world.is_window_initialized():
                raise ValueError(
                    "Viewer evaluation requires an EmbodiChain simulator task "
                    "with an initialized window"
                )
        self._seed = seed
        self._first_reset = True
        self._control_step = 0
        self._frame: EvaluationFrame | None = None
        self._episode_return = 0.0
        self._episode_length = 0
        self._episodes: list[dict[str, float | int | bool | str]] = []
        self._closed = False
        self._previous_no_auto_reset = getattr(
            self._base_env,
            "_demo_no_auto_reset",
            _MISSING,
        )
        self._base_env._demo_no_auto_reset = True
        self._policy_context = _policy_context_from_env(self._base_env)

    @property
    def policy_context(self) -> PolicyContext:
        """Return the timing used by the original task Environment."""
        return self._policy_context

    @property
    def physics_backend(self) -> str:
        """Return the backend selected by the original task Environment."""
        return "default"

    @property
    def viewer_is_open(self) -> bool:
        """Return whether the original task Viewer remains open."""
        if not self._viewer:
            return False
        world = self._world()
        return bool(world is not None and world.is_window_initialized())

    @property
    def current_frame(self) -> EvaluationFrame:
        """Return the latest observation and task state."""
        if self._frame is None:
            raise RuntimeError("Environment has not been reset")
        return self._frame

    @property
    def episodes(self) -> tuple[Mapping[str, float | int | bool | str], ...]:
        """Return completed episode summaries."""
        return tuple(dict(episode) for episode in self._episodes)

    def open_viewer(self, title: str) -> None:
        """Apply the evaluation title to the task Viewer."""
        if not self._viewer:
            return
        world = self._world()
        if world is not None and world.is_window_initialized():
            world.get_windows().set_window_title(title)

    def reset(self) -> EvaluationFrame:
        """Run the task's original reset and return its observation."""
        kwargs = {"seed": self._seed} if self._first_reset else {}
        observation, info = self.env.reset(**kwargs)
        self._first_reset = False
        self._control_step = 0
        self._episode_return = 0.0
        self._episode_length = 0
        self._frame = self._make_frame(observation, {"info": info})
        return self._frame

    def poll(self) -> str | None:
        """Report when the native Viewer is closed or Escape is pressed."""
        if not self._viewer:
            return None
        world = self._world()
        if world is None or not world.is_window_initialized():
            return "viewer closed"
        from dexsim.types import InputKey

        native = world.get_windows().native()
        if native.key_state(InputKey.SCANCODE_ESCAPE):
            return "viewer closed"
        return None

    def step(self, action: object) -> EnvironmentStep:
        """Apply one raw Policy action through the task's original action path."""
        if not isinstance(action, torch.Tensor):
            raise TypeError("EmbodiChain Policy action must be a torch.Tensor")
        started = time.perf_counter()
        env_action = convert_policy_action_for_env(self.env, action)
        observation, reward, terminated, truncated, info = self.env.step(env_action)
        reward_value = _single_float(reward, "reward")
        terminated_value = _single_bool(terminated, "terminated")
        truncated_value = _single_bool(truncated, "truncated")
        self._control_step += 1
        self._episode_return += reward_value
        self._episode_length += 1
        task_state = {
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        }
        self._frame = self._make_frame(observation, task_state)
        reason = _termination_reason(info, terminated_value, truncated_value)
        metrics = _step_metrics(info, reward_value)
        if reason is not None:
            success = _info_bool(info, "success")
            self._episodes.append(
                {
                    "index": len(self._episodes),
                    "reason": reason,
                    "reward": self._episode_return,
                    "length": self._episode_length,
                    "success": success,
                }
            )
        if self._viewer:
            remaining = self._policy_context.policy_dt - (time.perf_counter() - started)
            if remaining > 0.0:
                time.sleep(remaining)
        return EnvironmentStep(
            frame=self._frame,
            termination_reason=reason,
            metrics=metrics,
        )

    def metrics(self) -> dict[str, float]:
        """Aggregate completed task episodes."""
        if not self._episodes:
            return {}
        return {
            "eval/avg_reward": float(
                np.mean([float(episode["reward"]) for episode in self._episodes])
            ),
            "eval/avg_length": float(
                np.mean([float(episode["length"]) for episode in self._episodes])
            ),
            "eval/success_rate": float(
                np.mean([bool(episode["success"]) for episode in self._episodes])
            ),
        }

    def wait_for_reset_or_close(self) -> str:
        """Keep a paused Viewer responsive until it is closed."""
        while self.viewer_is_open:
            event = self.poll()
            if event is not None:
                return event
            world = self._world()
            if world is not None:
                world.update(0.0)
            time.sleep(0.01)
        return "viewer closed"

    def close(self) -> None:
        """Close the original task Environment."""
        if self._closed:
            return
        if self._previous_no_auto_reset is _MISSING:
            delattr(self._base_env, "_demo_no_auto_reset")
        else:
            self._base_env._demo_no_auto_reset = self._previous_no_auto_reset
        if getattr(self._base_env, "sim", None) is not None:
            self._base_env.close(exit_process=False)
        else:
            self.env.close()
        self._closed = True

    def _make_frame(
        self,
        observation: object,
        task_state: Mapping[str, object],
    ) -> EvaluationFrame:
        simulation_step = (
            self._control_step * self._policy_context.sim_steps_per_control
        )
        return EvaluationFrame(
            control_step=self._control_step,
            policy_time=self._control_step * self._policy_context.policy_dt,
            simulation_step=simulation_step,
            simulation_time=simulation_step * self._policy_context.physics_dt,
            observation=observation,
            task_state=task_state,
        )

    def _world(self) -> Any | None:
        sim = getattr(self._base_env, "sim", None)
        return None if sim is None else sim.get_world()


def evaluate_native_task(
    runtime: PolicyRuntime,
    *,
    seed: int,
    viewer: bool,
    episodes: int | None,
    control_steps: int | None,
    duration: float | None,
    termination_behavior: str = "auto_reset",
) -> NativeTaskEvaluationResult:
    """Evaluate an EmbodiChain Policy in the task used for training."""
    if episodes is not None and episodes <= 0:
        raise ValueError("episodes must be positive")
    if control_steps is not None and control_steps <= 0:
        raise ValueError("control_steps must be positive")
    if duration is not None and (duration <= 0.0 or not math.isfinite(duration)):
        raise ValueError("duration must be finite and positive")
    if control_steps is not None and duration is not None:
        raise ValueError("control_steps and duration are mutually exclusive")
    if termination_behavior == "continue":
        raise ValueError("Native task evaluation supports pause or auto_reset")

    try:
        environment = EmbodiChainTaskEnvironment(
            runtime.env,
            seed=seed,
            viewer=viewer,
        )
    except Exception:
        runtime.env.close()
        raise
    adapter = EmbodiChainTaskPolicyAdapter(runtime.policy, runtime.device, num_envs=1)
    if duration is not None:
        control_steps = math.ceil(
            duration / environment.policy_context.policy_dt - 1e-12
        )
    total_steps = 0
    reason = "viewer closed" if viewer else "episode target reached"
    options = RunOptions(
        headless=not viewer,
        termination_behavior=(
            "continue" if termination_behavior == "auto_reset" else "pause"
        ),
    )
    with create_motion_policy_evaluator(
        options=options,
        adapter=adapter,
        environment=environment,
        title=f"{runtime.env_id} - EmbodiChain",
    ) as evaluator:
        evaluator.reset()
        while True:
            if control_steps is not None and total_steps >= control_steps:
                reason = "control steps reached"
                break
            if episodes is not None and len(environment.episodes) >= episodes:
                reason = "episode target reached"
                break
            completed_before = len(environment.episodes)
            result = evaluator.step()
            if result.advanced:
                total_steps += 1
            if len(environment.episodes) > completed_before:
                if episodes is not None and len(environment.episodes) >= episodes:
                    reason = "episode target reached"
                    break
                if termination_behavior == "auto_reset":
                    evaluator.reset()
                    continue
            if result.reason is not None and not result.reset_performed:
                reason = result.reason
                break
        episode_results = environment.episodes
        metrics = environment.metrics()
        context = environment.policy_context
        backend = environment.physics_backend

    simulation_steps = total_steps * context.sim_steps_per_control
    return NativeTaskEvaluationResult(
        task_id=runtime.env_id,
        reason=reason,
        simulation_time=simulation_steps * context.physics_dt,
        simulation_steps=simulation_steps,
        control_steps=total_steps,
        physics_backend=backend,
        effective_duration=total_steps * context.policy_dt,
        requested_duration=duration,
        episodes=episode_results,
        metrics=metrics,
        viewer=viewer,
    )


def _single_float(value: object, name: str) -> float:
    tensor = torch.as_tensor(value).reshape(-1)
    if tensor.numel() != 1:
        raise ValueError(f"Native task {name} must contain one value")
    return float(tensor.item())


def _single_bool(value: object, name: str) -> bool:
    tensor = torch.as_tensor(value, dtype=torch.bool).reshape(-1)
    if tensor.numel() != 1:
        raise ValueError(f"Native task {name} must contain one value")
    return bool(tensor.item())


def _info_bool(info: object, name: str) -> bool:
    if not isinstance(info, Mapping) or name not in info:
        return False
    return _single_bool(info[name], f"info.{name}")


def _termination_reason(
    info: object,
    terminated: bool,
    truncated: bool,
) -> str | None:
    if _info_bool(info, "success"):
        return "success"
    if _info_bool(info, "fail"):
        return "failure"
    if truncated:
        return "time limit"
    if terminated:
        return "terminated"
    return None


def _step_metrics(info: object, reward: float) -> dict[str, float]:
    result = {"reward": reward}
    if not isinstance(info, Mapping):
        return result
    metrics = info.get("metrics")
    if not isinstance(metrics, Mapping):
        return result
    for name, value in metrics.items():
        tensor = torch.as_tensor(value).reshape(-1)
        if tensor.numel() == 1:
            result[str(name)] = float(tensor.item())
    return result


def _policy_context_from_env(env: Any) -> PolicyContext:
    """Read timing from a simulator task or lightweight learning task."""
    if hasattr(env, "physics_dt") and hasattr(env, "step_dt"):
        return PolicyContext(
            robot=None,
            physics_dt=float(env.physics_dt),
            sim_steps_per_control=int(env.cfg.sim_steps_per_control),
            policy_dt=float(env.step_dt),
        )
    if not hasattr(env, "dt"):
        raise ValueError("Lightweight task evaluation requires an env.dt value")
    policy_dt = float(env.dt)
    return PolicyContext(
        robot=None,
        physics_dt=policy_dt,
        sim_steps_per_control=1,
        policy_dt=policy_dt,
    )
