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

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from embodichain.lab.sim.agent.action_plan import ActionPlan
from embodichain.lab.sim.agent.atomic_engine_planner import (
    AtomicEnginePlanner,
)

__all__ = ["AtomicGraphAction"]


@dataclass
class AtomicGraphAction:
    """Executable wrapper for compiled atomic-action graph payloads."""

    spec: Mapping[str, Any]

    def __post_init__(self) -> None:
        self.spec = deepcopy(dict(self.spec))

    def __call__(self, env=None, **kwargs):
        plan = self.plan(env=env, **kwargs)
        return plan.trajectory[0].detach().cpu().numpy().astype(np.float32, copy=False)

    def plan(self, env=None, **kwargs) -> ActionPlan:
        return self._run_atomic_graph(env=env, kwargs=kwargs)

    @property
    def action_name(self) -> str:
        return str(self.spec.get("name", self.spec.get("kind", "atomic_graph_action")))

    def _run_atomic_graph(self, *, env, kwargs: dict[str, Any]) -> ActionPlan:
        return AtomicEnginePlanner().plan(self.spec, env=env, **kwargs)
