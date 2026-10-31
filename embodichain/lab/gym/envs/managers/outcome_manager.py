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

"""Once-per-transition evaluation and final acceptance of configured RL tasks."""

from __future__ import annotations

from collections.abc import Sequence
import inspect
from typing import TYPE_CHECKING, Any

import torch

from .cfg import ConditionCfg, RLTaskCfg
from .manager_base import Functor, ManagerBase
from ..objectives.conditions import ConditionResult

if TYPE_CHECKING:
    from ..embodied_env import EmbodiedEnv

__all__ = ["OutcomeManager"]


class OutcomeManager(ManagerBase):
    """Aggregate named conditions without advancing them during queries.

    Args:
        cfg: Success, failure and custom truncation condition groups.
        env: Owning vector environment.
    """

    def __init__(self, cfg: RLTaskCfg, env: EmbodiedEnv) -> None:
        self._terms: dict[str, dict[str, ConditionCfg]] = {}
        super().__init__(cfg, env)
        self._initialized = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        self._success = torch.zeros_like(self._initialized)
        self._failure = torch.zeros_like(self._initialized)
        self._truncation = torch.zeros_like(self._initialized)
        self._valid = torch.zeros_like(self._initialized)
        self._accepted = torch.zeros_like(self._initialized)
        self._results: dict[str, ConditionResult] = {}
        self._steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    @property
    def active_functors(self) -> dict[str, list[str]]:
        """Return condition names grouped by their outcome role."""
        return {group: list(terms) for group, terms in self._terms.items()}

    def _prepare_functors(self) -> None:
        for group in ("success", "failure", "truncation"):
            terms = getattr(self.cfg, group)
            self._terms[group] = {}
            for name, cfg in terms.items():
                if not isinstance(cfg, ConditionCfg):
                    raise TypeError(f"{group}.{name} must be a ConditionCfg.")
                self._resolve_common_functor_cfg(f"{group}.{name}", cfg, min_argc=1)
                inspect.signature(cfg.func).bind(self._env, **cfg.params)
                from .conditions import object_near_target, object_below_height
                from ..objectives.conditions import (
                    evaluate_near_target,
                    evaluate_below_height,
                )

                if cfg.func in (object_near_target, object_below_height):
                    entity = self._env.sim.get_rigid_object(
                        cfg.params["entity_cfg"].uid
                    )
                    if entity is None:
                        raise ValueError(f"{group}.{name} requires a rigid object.")
                    sample = torch.zeros(1, 3)
                    if cfg.func is object_near_target:
                        key = cfg.params["target_pose_key"]
                        if not isinstance(key, str) or not key or key != key.strip():
                            raise ValueError(
                                "target_pose_key must be a nonempty string."
                            )
                        evaluate_near_target(
                            sample,
                            sample,
                            cfg.params["tolerance"],
                            axes=cfg.params.get("axes", (0, 1, 2)),
                            inclusive=cfg.params.get("inclusive", False),
                        )
                    else:
                        evaluate_below_height(sample, cfg.params["height"])
                self._terms[group][name] = cfg

    def _ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def reset(
        self, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> dict[str, float]:
        """Clear selected rows and reset stateful condition functors.

        Args:
            env_ids: Rows being reset, or every row.

        Returns:
            An empty statistics mapping.
        """
        ids = self._ids(env_ids)
        for terms in self._terms.values():
            for cfg in terms.values():
                if isinstance(cfg.func, Functor):
                    cfg.func.reset(env_ids=ids)
        for value in (
            self._initialized,
            self._success,
            self._failure,
            self._truncation,
            self._valid,
            self._accepted,
            self._steps,
        ):
            value[ids] = 0
        for result in self._results.values():
            result.satisfied[ids] = False
            result.valid[ids] = False
            for metric in result.metrics.values():
                metric[ids] = 0
        return {}

    def __str__(self) -> str:
        return f"OutcomeManager({sum(len(terms) for terms in self._terms.values())} conditions)"

    def update(self) -> None:
        """Sample each condition exactly once for the current control transition.

        Stateful functors advance only here. Reset and snapshot operations never
        call them, so partial resets cannot advance another row's history.
        """
        self._valid.fill_(True)
        self._success.fill_(bool(self._terms["success"]))
        self._failure.zero_()
        self._truncation.zero_()
        masks = {
            "success": self._success,
            "failure": self._failure,
            "truncation": self._truncation,
        }
        for group, terms in self._terms.items():
            for name, cfg in terms.items():
                result = self._call_functor(f"{group}.{name}", cfg, self._env)
                if not isinstance(result, ConditionResult):
                    raise TypeError(f"{group}.{name} must return ConditionResult.")
                if result.satisfied.shape != (
                    self.num_envs,
                ) or result.satisfied.device != torch.device(self.device):
                    raise ValueError(
                        f"{group}.{name} must match environment batch and device."
                    )
                self._valid &= result.valid
                if group == "success":
                    masks[group] &= result.satisfied
                else:
                    masks[group] |= result.satisfied
                key = f"{group}.{name}"
                if key not in self._results:
                    self._results[key] = ConditionResult(
                        torch.zeros_like(self._valid),
                        torch.zeros_like(self._valid),
                        {
                            metric: torch.zeros_like(value)
                            for metric, value in result.metrics.items()
                        },
                    )
                owned = self._results[key]
                if set(owned.metrics) != set(result.metrics) or any(
                    owned.metrics[metric].shape != value.shape
                    or owned.metrics[metric].dtype != value.dtype
                    for metric, value in result.metrics.items()
                ):
                    raise ValueError(
                        f"{group}.{name} changed its metric schema during execution."
                    )
                owned.satisfied.copy_(result.satisfied)
                owned.valid.copy_(result.valid)
                for metric, value in result.metrics.items():
                    owned.metrics[metric].copy_(value)
        self._initialized.fill_(True)
        self._failure |= ~self._valid
        self._accepted.copy_(
            self._success & self._valid & ~self._failure & ~self._truncation
        )
        self._steps += 1

    def finalize(self, truncated: torch.Tensor) -> None:
        """Apply actual Gym time limits before recording or publication.

        Args:
            truncated: Actual per-row truncation mask from the host step.
        """
        self._truncation |= truncated
        self._accepted &= ~self._failure & ~self._truncation

    def snapshot(self) -> dict[str, Any]:
        """Return owned result tensors; reading does not sample conditions."""
        return {
            "initialized": self._initialized.clone(),
            "valid": self._valid.clone(),
            "physical_success": self._success.clone(),
            "invalid_measurement": (self._initialized & ~self._valid).clone(),
            "failure": self._failure.clone(),
            "truncation": self._truncation.clone(),
            "accepted": self._accepted.clone(),
            "steps": self._steps.clone(),
            "conditions": {
                name: {
                    "satisfied": value.satisfied.clone(),
                    "valid": value.valid.clone(),
                    "metrics": {
                        key: metric.clone() for key, metric in value.metrics.items()
                    },
                }
                for name, value in self._results.items()
            },
        }

    def task_state(self) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Return the cached Gym success/failure and named scalar metrics."""
        metrics = {
            f"{name}.{key}": value.clone()
            for name, result in self._results.items()
            for key, value in result.metrics.items()
        }
        return self._accepted.clone(), self._failure.clone(), metrics

    def truncation(self) -> torch.Tensor:
        """Return cached custom or finalized time-limit truncation."""
        return self._truncation.clone()
