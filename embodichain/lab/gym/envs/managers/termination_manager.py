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

"""Once-per-transition condition evaluation and cached RL termination flags."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import inspect
from typing import TYPE_CHECKING, Any

import torch

from embodichain.lab.evaluation import EvaluationStatus, RuleResult
from embodichain.lab.evaluation.observations import _rule_tensor_report

from .cfg import ConditionCfg, RLTaskCfg
from .manager_base import Functor, ManagerBase
from ..objectives.conditions import ConditionResult

if TYPE_CHECKING:
    from ..embodied_env import EmbodiedEnv

__all__ = ["TerminationManager"]


class TerminationManager(ManagerBase):
    """Evaluate conditions and own one authoritative termination snapshot.

    Success conditions use AND; failure and truncation conditions use OR.
    Invalid measurements fail the affected rows. Failure and truncation both
    exclude accepted success, while raw physical success remains available.
    The host supplies its time-limit mask when calling :meth:`compute`.

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
        self._terminated = torch.zeros_like(self._initialized)
        self._results: dict[str, ConditionResult] = {}
        self._rule_results: dict[str, Mapping[str, RuleResult]] = {}
        self._rule_identities: dict[str, str | None] = {}
        self._steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    @property
    def active_functors(self) -> dict[str, list[str]]:
        """Return condition names grouped by their outcome role."""
        return {group: list(terms) for group, terms in self._terms.items()}

    @property
    def terminated(self) -> torch.Tensor:
        """Return owned Gym termination flags from the latest computation."""
        return self._terminated.clone()

    @property
    def truncated(self) -> torch.Tensor:
        """Return owned custom and time-limit truncation flags."""
        return self._truncation.clone()

    def _prepare_functors(self) -> None:
        for group in ("success", "failure", "truncation"):
            terms = getattr(self.cfg, group)
            self._terms[group] = {}
            for name, cfg in terms.items():
                if not isinstance(cfg, ConditionCfg):
                    raise TypeError(f"{group}.{name} must be a ConditionCfg.")
                self._resolve_common_functor_cfg(f"{group}.{name}", cfg, min_argc=1)
                inspect.signature(cfg.func).bind(self._env, **cfg.params)
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
            self._terminated,
            self._steps,
        ):
            value[ids] = 0
        for result in self._results.values():
            result.satisfied[ids] = False
            result.valid[ids] = False
            for metric in result.metrics.values():
                metric[ids] = 0
        for rules in self._rule_results.values():
            for rule in rules.values():
                rule.status[ids] = EvaluationStatus.NOT_RUN
                for metric in rule.metrics.values():
                    metric[ids] = float("nan") if metric.is_floating_point() else 0
                for boundary in (
                    rule.witness_steps,
                    rule.witness_revisions,
                    rule.violation_steps,
                    rule.violation_revisions,
                ):
                    if boundary is not None:
                        boundary[ids] = -1
        return {}

    def __str__(self) -> str:
        return f"TerminationManager({sum(len(terms) for terms in self._terms.values())} conditions)"

    def compute(
        self,
        time_out: torch.Tensor | None = None,
        *,
        ignore_terminations: bool = False,
    ) -> None:
        """Compute conditions, acceptance and Gym flags once per transition.

        Stateful functors advance only here. Reset and snapshot operations never
        call them, so partial resets cannot advance another row's history.

        Args:
            time_out: Host-owned time-limit mask for this transition, or no
                external time limit. Must be a boolean vector on the env device.
            ignore_terminations: Suppress Gym termination flags while retaining
                measured task acceptance and failure, matching the host option.
        """
        if time_out is not None and (
            time_out.shape != (self.num_envs,)
            or time_out.dtype != torch.bool
            or time_out.device != torch.device(self.device)
        ):
            raise ValueError("time_out must match the environment batch and device.")
        self._valid.fill_(True)
        self._success.fill_(bool(self._terms["success"]))
        self._failure.zero_()
        self._truncation.zero_()
        if time_out is not None:
            self._truncation.copy_(time_out)
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
                if result.rule_results:
                    self._rule_results[key] = result.rule_results
                    self._rule_identities[key] = result.evaluation_identity
        self._initialized.fill_(True)
        progress = getattr(self._env, "task_program_progress", None)
        if progress is not None:
            final, process_failed, gate_valid = progress.gates()
            self._success &= final
            self._failure |= process_failed
            self._valid &= gate_valid
        self._failure |= ~self._valid
        self._terminated.copy_(self._accepted_success() | self._failure)
        if ignore_terminations:
            self._terminated.zero_()
        self._steps += 1

    def _accepted_success(self) -> torch.Tensor:
        return self._success & self._valid & ~self._failure & ~self._truncation

    def snapshot(self) -> dict[str, Any]:
        """Return owned result tensors; reading does not sample conditions."""
        return {
            "initialized": self._initialized.clone(),
            "valid": self._valid.clone(),
            "physical_success": self._success.clone(),
            "invalid_measurement": (self._initialized & ~self._valid).clone(),
            "failure": self._failure.clone(),
            "truncation": self._truncation.clone(),
            "accepted": self._accepted_success(),
            "terminated": self._terminated.clone(),
            "steps": self._steps.clone(),
            "conditions": {
                name: {
                    "satisfied": value.satisfied.clone(),
                    "valid": value.valid.clone(),
                    "metrics": {
                        key: metric.clone() for key, metric in value.metrics.items()
                    },
                    **(
                        {
                            "evaluation_identity": self._rule_identities[name],
                            "rules": {
                                rule_id: _rule_tensor_report(rule)
                                for rule_id, rule in self._rule_results[name].items()
                            },
                        }
                        if name in self._rule_results
                        else {}
                    ),
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
        return self._accepted_success(), self._failure.clone(), metrics
