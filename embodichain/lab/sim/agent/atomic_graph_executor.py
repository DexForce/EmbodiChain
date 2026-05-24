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

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from embodichain.lab.sim.agent.action_plan import ActionPlan
from embodichain.lab.sim.agent.atomic_action_adapter import _select_arm
from embodichain.lab.sim.agent.atomic_engine_planner import (
    AtomicEnginePlanner,
    _controlled_joint_ids,
    _flatten_action_specs,
    _robot_name_from_action_spec,
)
from embodichain.utils.logger import log_warning

__all__ = ["AtomicGraphAction"]


@dataclass
class AtomicGraphAction:
    """Executable wrapper for compiled atomic-action graph payloads."""

    spec: Mapping[str, Any]
    fallback_action: Callable[..., Any] | None = None
    func: Callable[..., Any] | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.spec = deepcopy(dict(self.spec))
        self.func = getattr(self.fallback_action, "func", self.fallback_action)

    def __call__(self, env=None, **kwargs):
        plan = self.plan(env=env, **kwargs)
        return plan.trajectory[0].detach().cpu().numpy().astype(np.float32, copy=False)

    def plan(self, env=None, **kwargs) -> ActionPlan:
        if not kwargs.get("use_atomic_action_graph", True) or not kwargs.get(
            "use_public_atomic_actions", True
        ):
            return self._run_fallback_plan(env=env, kwargs=kwargs)

        try:
            return self._run_atomic_graph(env=env, kwargs=kwargs)
        except Exception as exc:
            if kwargs.get("require_atomic_action_graph", False) or not kwargs.get(
                "allow_legacy_atomic_action_fallback", False
            ):
                raise RuntimeError(
                    f"Atomic graph action '{self.action_name}' failed: {exc}"
                ) from exc
            if self.fallback_action is None:
                raise
            log_warning(
                f"Atomic graph action '{self.action_name}' failed with "
                f"{type(exc).__name__}: {exc}. Falling back to legacy action."
            )
            return self._run_fallback_plan(env=env, kwargs=kwargs)

    @property
    def action_name(self) -> str:
        return str(self.spec.get("name", self.spec.get("kind", "atomic_graph_action")))

    @property
    def legacy_action_name(self) -> str | None:
        func = getattr(self.fallback_action, "func", self.fallback_action)
        return getattr(func, "__name__", None)

    def _run_fallback(self, *, env, kwargs: dict[str, Any]):
        if self.fallback_action is None:
            raise RuntimeError(
                f"Atomic graph action '{self.action_name}' has no fallback action."
            )
        return self.fallback_action(env=env, **_fallback_kwargs(kwargs))

    def _run_fallback_plan(self, *, env, kwargs: dict[str, Any]) -> ActionPlan:
        legacy_action = self._run_fallback(env=env, kwargs=kwargs)
        action_specs = _flatten_action_specs(self.spec)
        robot_name = None
        for action in action_specs:
            robot_name = _robot_name_from_action_spec(action)
            if robot_name is not None:
                break
        if robot_name is None:
            raise RuntimeError(
                f"Atomic graph action '{self.action_name}' fallback has no robot_name."
            )
        trajectory = torch.as_tensor(
            legacy_action,
            dtype=torch.float32,
            device=getattr(env.robot, "device", None),
        )
        if trajectory.ndim == 2:
            trajectory = trajectory.unsqueeze(0)
        joint_ids = _controlled_joint_ids(env, action_specs, robot_name)
        if trajectory.shape[-1] != len(joint_ids):
            is_left, _, _ = _select_arm(robot_name)
            joint_ids = list(env.left_arm_joints if is_left else env.right_arm_joints)
            joint_ids += list(env.left_eef_joints if is_left else env.right_eef_joints)
        return ActionPlan(
            is_success=True,
            trajectory=trajectory,
            joint_ids=joint_ids,
            action_name=self.action_name,
        )

    def _run_atomic_graph(self, *, env, kwargs: dict[str, Any]) -> ActionPlan:
        return AtomicEnginePlanner().plan(self.spec, env=env, **kwargs)


def _fallback_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    filtered = dict(kwargs)
    filtered.pop("use_atomic_action_graph", None)
    filtered.pop("require_atomic_action_graph", None)
    return filtered
