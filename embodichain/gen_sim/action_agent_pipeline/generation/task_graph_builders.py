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

"""Compatibility facade for executable Seed Graph v2 builders.

Config generation no longer produces a grounded task graph. Historical builder
names remain importable inside the package, but now return the immutable Seed
v2 topology that the runtime grounds per environment.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.generation.builder_protocols import (
    ArrangementSpecLike,
    RelativeSpecLike,
    StackingSpecLike,
)
from embodichain.gen_sim.action_agent_pipeline.generation.seed_task_graph import (
    make_arrangement_seed_task_graph,
    make_relative_seed_task_graph,
    make_stacking_seed_task_graph,
    validate_seed_task_graph,
)

__all__ = [
    "compile_arrangement_task_graph",
    "compile_relative_task_graph",
    "compile_stacking_task_graph",
    "make_arrangement_task_graph",
    "make_relative_task_graph",
    "make_stacking_task_graph",
]


def make_arrangement_task_graph(
    task_name: str,
    spec: ArrangementSpecLike,
) -> dict[str, Any]:
    return make_arrangement_seed_task_graph(task_name, spec)


def compile_arrangement_task_graph(
    task_name: str,
    seed_graph: Mapping[str, Any],
    spec: ArrangementSpecLike,
) -> dict[str, Any]:
    return _validate_compilation_input(
        task_name,
        seed_graph,
        make_arrangement_seed_task_graph(task_name, spec),
        route="arrangement_line",
    )


def make_stacking_task_graph(
    task_name: str,
    spec: StackingSpecLike,
) -> dict[str, Any]:
    return make_stacking_seed_task_graph(task_name, spec)


def compile_stacking_task_graph(
    task_name: str,
    seed_graph: Mapping[str, Any],
    spec: StackingSpecLike,
) -> dict[str, Any]:
    return _validate_compilation_input(
        task_name,
        seed_graph,
        make_stacking_seed_task_graph(task_name, spec),
        route="stacking",
    )


def make_relative_task_graph(
    task_name: str,
    spec: RelativeSpecLike,
) -> dict[str, Any]:
    return make_relative_seed_task_graph(task_name, spec)


def compile_relative_task_graph(
    task_name: str,
    seed_graph: Mapping[str, Any],
    spec: RelativeSpecLike,
) -> dict[str, Any]:
    return _validate_compilation_input(
        task_name,
        seed_graph,
        make_relative_seed_task_graph(task_name, spec),
        route="object_manipulation",
    )


def _validate_compilation_input(
    task_name: str,
    seed_graph: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    route: str,
) -> dict[str, Any]:
    validate_seed_task_graph(seed_graph, task_name=task_name, route=route)
    if dict(seed_graph) != dict(expected):
        raise ValueError(
            f"{route} Seed v2 was not derived from the supplied semantic plan."
        )
    return deepcopy(dict(seed_graph))
