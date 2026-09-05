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

"""Configuration owned by Task Engine orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from importlib.resources import files
from pathlib import Path
from typing import Any, Final

import yaml

from embodichain.utils import configclass

__all__ = [
    "TASK_ENGINE_DEFAULTS_SCHEMA",
    "TaskEngineExecutionCfg",
    "TaskEnginePlanningCfg",
    "TaskEngineWorkflowCfg",
    "load_task_engine_config",
]

TASK_ENGINE_DEFAULTS_SCHEMA: Final = "embodichain.task-engine-defaults/v1"
_IK_SOLVER_MODES: Final = ("auto", "ur", "pytorch")


@configclass
class TaskEngineExecutionCfg:
    """Success policy for vectorized simulator execution."""

    num_envs: int = 1
    success_policy: str = "any"
    min_successful_envs: int = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_envs, bool)
            or not isinstance(self.num_envs, int)
            or self.num_envs < 1
        ):
            raise ValueError("num_envs must be a positive integer.")
        if self.success_policy not in {"any", "all", "at_least"}:
            raise ValueError("success_policy must be any, all, or at_least.")
        if (
            isinstance(self.min_successful_envs, bool)
            or not isinstance(self.min_successful_envs, int)
            or not 1 <= self.min_successful_envs <= self.num_envs
        ):
            raise ValueError("min_successful_envs must be in [1, num_envs].")
        if self.success_policy == "any" and self.min_successful_envs != 1:
            raise ValueError("success_policy=any requires min_successful_envs=1.")
        if self.success_policy == "all" and self.min_successful_envs != self.num_envs:
            raise ValueError(
                "success_policy=all requires min_successful_envs=num_envs."
            )

    @property
    def required_successes(self) -> int:
        """Return the number of successful replicas required for acceptance."""
        if self.success_policy == "all":
            return self.num_envs
        if self.success_policy == "any":
            return 1
        return self.min_successful_envs


@configclass
class TaskEngineWorkflowCfg:
    """Conservative first-version orchestration limits.

    The packaged YAML owns retry limits so deployment testing can tune them
    without changing the orchestration implementation.
    """

    max_parallel_workers: int = 2
    max_scene_attempts: int = 2
    max_action_attempts: int = 3

    def __post_init__(self) -> None:
        for field_name in (
            "max_parallel_workers",
            "max_scene_attempts",
            "max_action_attempts",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer.")


@configclass
class TaskEnginePlanningCfg:
    """Task interpretation and Action bundle generation defaults."""

    candidate_count: int = 3
    planning_mode: str = "offline"
    gripper_model: str = "pgi"
    ik_solver: str = "auto"
    max_episodes: int = 1
    max_episode_steps: int = 6000
    planner: dict[str, Any] = {}

    def __post_init__(self) -> None:
        for field_name in (
            "candidate_count",
            "max_episodes",
            "max_episode_steps",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer.")
        if self.planning_mode not in {"offline", "ab"}:
            raise ValueError("planning_mode must be offline or ab.")
        if self.gripper_model not in {"pgi", "robotiq"}:
            raise ValueError(
                f"Unsupported gripper model {self.gripper_model!r}; expected one "
                "of: pgi, robotiq."
            )
        if self.ik_solver not in _IK_SOLVER_MODES:
            raise ValueError(
                f"Unsupported IK solver mode {self.ik_solver!r}; expected one of: "
                "auto, ur, pytorch."
            )
        if not isinstance(self.planner, Mapping):
            raise TypeError("planner must be a mapping.")
        from embodichain.gen_sim.action_engine.config.runtime_policy import (
            _resolve_planner_policy,
        )

        _resolve_planner_policy(self.planner)
        self.planner = deepcopy(dict(self.planner))


def load_task_engine_config(
    path: str | Path | None = None,
) -> tuple[
    TaskEngineWorkflowCfg,
    TaskEnginePlanningCfg,
    TaskEngineExecutionCfg,
]:
    """Load strict Task Engine defaults from YAML.

    Args:
        path: Optional override YAML. The packaged defaults are used when omitted.

    Returns:
        Validated workflow, planning, and execution configurations.

    Raises:
        TypeError: If a configuration section is not a mapping.
        ValueError: If the YAML schema or fields are invalid.
    """
    content = (
        Path(path).expanduser().resolve().read_text(encoding="utf-8")
        if path is not None
        else files(__package__).joinpath("defaults.yaml").read_text(encoding="utf-8")
    )
    raw = yaml.safe_load(content)
    if not isinstance(raw, Mapping):
        raise TypeError("Task Engine configuration must be a mapping.")
    expected = {"schema_version", "workflow", "planning", "execution"}
    if set(raw) != expected:
        raise ValueError("Task Engine configuration fields are invalid.")
    if raw.get("schema_version") != TASK_ENGINE_DEFAULTS_SCHEMA:
        raise ValueError("Task Engine configuration schema_version is invalid.")
    workflow = _mapping(raw.get("workflow"), "workflow")
    planning = _mapping(raw.get("planning"), "planning")
    execution = _mapping(raw.get("execution"), "execution")
    if set(workflow) != {
        "max_parallel_workers",
        "max_scene_attempts",
        "max_action_attempts",
    }:
        raise ValueError("Task Engine workflow configuration fields are invalid.")
    required_planning = {
        "candidate_count",
        "planning_mode",
        "gripper_model",
        "ik_solver",
        "max_episodes",
        "max_episode_steps",
    }
    if set(planning) not in {
        frozenset(required_planning),
        frozenset((*required_planning, "planner")),
    }:
        raise ValueError("Task Engine planning configuration fields are invalid.")
    if "planner" in planning:
        planning["planner"] = _mapping(planning["planner"], "planning.planner")
    if set(execution) != {
        "num_envs",
        "success_policy",
        "min_successful_envs",
    }:
        raise ValueError("Task Engine execution configuration fields are invalid.")
    return (
        TaskEngineWorkflowCfg(**workflow),
        TaskEnginePlanningCfg(**planning),
        TaskEngineExecutionCfg(**execution),
    )


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Task Engine {field_name} configuration must be a mapping.")
    return dict(value)
