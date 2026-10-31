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

"""Common simulator host for configuration-defined manipulation RL tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
from typing import Any

import torch

from embodichain.utils import configclass
from .embodied_env import EmbodiedEnv, EmbodiedEnvCfg
from .managers.cfg import RLTaskCfg
from .managers.outcome_manager import OutcomeManager
from embodichain.lab.sim.types import EnvAction, EnvObs

__all__ = ["ConfiguredRLEnv", "ConfiguredRLEnvCfg"]


@configclass
class ConfiguredRLEnvCfg(EmbodiedEnvCfg):
    """An ordinary simulator configuration with explicit RL task conditions."""

    rl_task: RLTaskCfg | None = None


def _row(value: Any, env_id: int) -> Any:
    if isinstance(value, torch.Tensor):
        return _row(value[env_id].detach().cpu().tolist(), env_id)
    if isinstance(value, Mapping):
        return {key: _row(child, env_id) for key, child in value.items()}
    if isinstance(value, list):
        return [_row(child, env_id) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class ConfiguredRLEnv(EmbodiedEnv):
    """Run configured managers and measured conditions through the ordinary loop.

    Args:
        cfg: Configuration with an ``rl_task`` declaration and no Task Program.
        **kwargs: Ordinary simulator environment construction options.
    """

    def __init__(self, cfg: ConfiguredRLEnvCfg, **kwargs: Any) -> None:
        if not isinstance(cfg.rl_task, RLTaskCfg):
            raise ValueError("ConfiguredRLEnv requires an RLTaskCfg.")
        if cfg.task_program is not None:
            raise ValueError("rl_task and task_program cannot share one deployment.")
        self.outcome_manager: OutcomeManager | None = None
        self._task_initial_states: list[dict[str, Any]] = []
        super().__init__(cfg, **kwargs)

    def _init_sim_state(self, **kwargs: Any) -> None:
        super()._init_sim_state(**kwargs)
        self.outcome_manager = OutcomeManager(self.cfg.rl_task, self)
        self._last_terminated = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._last_truncated = torch.zeros_like(self._last_terminated)
        self._task_initial_states = [{} for _ in range(self.num_envs)]

    def _update_sim_state(self, **kwargs: Any) -> None:
        super()._update_sim_state(**kwargs)
        self.outcome_manager.update()
        # BaseEnv increments the counter after reward computation. Resolve the
        # imminent time limit now so reward, Gym flags and recorded acceptance
        # use the same outcome, including success on the last allowed step.
        self.outcome_manager.finalize(self._elapsed_steps + 1 >= self.max_episode_steps)

    def _target_keys(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    cfg.params["target_pose_key"]
                    for group in ("success", "failure", "truncation")
                    for cfg in getattr(self.cfg.rl_task, group).values()
                    if "target_pose_key" in cfg.params
                }
            )
        )

    def _reset_physical_objective(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        super()._reset_physical_objective(env_ids)
        self.initialize_task_outcome(env_ids)

    def initialize_task_outcome(
        self, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> None:
        """Initialize selected task contexts after reset events or replay restore.

        Args:
            env_ids: Rows with their final initial physical state restored.
        """
        ids = self.episode_context._ids(env_ids)
        targets = {}
        for key in self._target_keys():
            value = getattr(self, key, None)
            if not isinstance(value, torch.Tensor) or value.shape not in (
                (self.num_envs, 3),
                (self.num_envs, 4, 4),
            ):
                raise ValueError(f"Missing or malformed reset target {key!r}.")
            if not value.is_floating_point() or not bool(
                torch.isfinite(value[ids]).all()
            ):
                raise ValueError(
                    f"Reset target {key!r} must contain finite floating poses."
                )
            targets[key] = value
        self.outcome_manager.reset(ids)
        self._last_terminated[ids] = False
        self._last_truncated[ids] = False
        for env_id in ids.cpu().tolist():
            self._task_initial_states[env_id] = {
                "schema_version": 1,
                "targets": {
                    key: value[env_id].detach().cpu().tolist()
                    for key, value in targets.items()
                },
            }

    def get_task_initial_state(self, env_id: int = 0) -> dict[str, Any]:
        """Return an owned initial virtual-target context for recording.

        Args:
            env_id: Environment row to snapshot.

        Returns:
            Versioned JSON-compatible reset target values.
        """
        return deepcopy(self._task_initial_states[env_id])

    def restore_task_initial_state(self, state: Mapping[str, Any]) -> None:
        """Restore a recorded single-environment virtual-target context.

        Args:
            state: Versioned initial state from episode trajectory metadata.
        """
        if self.num_envs != 1:
            raise ValueError("Task-outcome replay requires a single environment.")
        if not isinstance(state, Mapping) or state.get("schema_version") != 1:
            raise ValueError(
                "Dynamic task replay requires task_initial_state metadata."
            )
        targets = state.get("targets")
        if not isinstance(targets, Mapping) or set(targets) != set(self._target_keys()):
            raise ValueError("Replay virtual targets do not match the configured task.")
        prepared = {}
        for key, value in targets.items():
            current = getattr(self, key, None)
            restored = torch.as_tensor(
                value, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            if (
                current is None
                or restored.shape != current.shape
                or not bool(torch.isfinite(restored).all())
            ):
                raise ValueError(f"Invalid replay target {key!r}.")
            prepared[key] = restored.to(dtype=current.dtype)
        for key, value in prepared.items():
            getattr(self, key).copy_(value)
        self.initialize_task_outcome()

    def compute_task_state(
        self, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Return cached condition acceptance and diagnostics."""
        return self.outcome_manager.task_state()

    def get_info(self, **kwargs: Any) -> dict[str, Any]:
        """Return ordinary Gym info plus an owned physical outcome snapshot."""
        info = super().get_info(**kwargs)
        info["task_outcome"] = self.outcome_manager.snapshot()
        return info

    def check_truncated(self, obs: EnvObs, info: dict[str, Any]) -> torch.Tensor:
        """Return custom truncation; the base loop adds its episode time limit."""
        return self.outcome_manager.truncation()

    def is_task_success(self, **kwargs: Any) -> torch.Tensor:
        """Return finalized configured acceptance instead of legacy default true."""
        return self.outcome_manager.task_state()[0]

    def _hook_after_sim_step(
        self,
        obs: EnvObs,
        action: EnvAction,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        info: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        self.outcome_manager.finalize(kwargs["truncateds"])
        self._last_terminated.copy_(kwargs["terminateds"])
        self._last_truncated.copy_(kwargs["truncateds"])
        info["success"], info["fail"], info["metrics"] = (
            self.outcome_manager.task_state()
        )
        info["task_outcome"] = self.outcome_manager.snapshot()
        super()._hook_after_sim_step(obs, action, rewards, dones, info, **kwargs)

    def get_demo_episode_metadata(self, env_id: int) -> dict[str, Any]:
        """Add measured outcome and the original target context to episode metadata.

        Args:
            env_id: Recorded environment row.

        Returns:
            Owned JSON-compatible episode metadata.
        """
        metadata = super().get_demo_episode_metadata(env_id)
        outcome = _row(self.outcome_manager.snapshot(), env_id)
        metadata["task_initial_state"] = self.get_task_initial_state(env_id)
        metadata["task_outcome"] = outcome
        # The parent synthesizes a legacy span using reset-time _task_success.
        # Explicit exports during the next episode must use this episode's
        # configured authority rather than that previous terminal snapshot.
        if not self._demo_episode_metadata[env_id].get("segments"):
            accepted = bool(outcome["accepted"])
            reason = (
                "truncated"
                if outcome["truncation"]
                else (
                    "failure"
                    if outcome["failure"]
                    else "success" if accepted else "task_incomplete"
                )
            )
            metadata.update(
                success=accepted,
                completed=accepted,
                terminal_reason=reason,
                terminated=bool(self._last_terminated[env_id]),
                truncated=bool(self._last_truncated[env_id]),
            )
            for segment in metadata.get("segments", []):
                segment.update(
                    success=accepted,
                    outcome_kind=(
                        "succeeded"
                        if accepted
                        else (
                            "truncated"
                            if outcome["truncation"]
                            else "validation_failed"
                        )
                    ),
                    failure_reason=None if accepted else reason,
                )
        return metadata
