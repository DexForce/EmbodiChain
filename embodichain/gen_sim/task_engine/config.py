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


@configclass
class TaskEngineExecutionCfg:
    """Success policy for vectorized simulator execution."""

    num_envs: int = 1
    success_policy: str = "any"
    min_successful_envs: int = 1
    acceptance_mode: str = "strict"

    def __post_init__(self) -> None:
        if self.acceptance_mode not in {"strict", "experiment"}:
            raise ValueError("acceptance_mode must be strict or experiment.")
        if self.acceptance_mode == "experiment" and self.num_envs != 1:
            raise ValueError(
                "Experimental program acceptance currently requires one environment."
            )
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
    """Task interpretation and semantic graph generation defaults."""

    candidate_count: int = 3
    planning_mode: str = "offline"
    max_episodes: int = 1
    max_episode_steps: int = 10000
    fit_grasp_assets: bool = False
    twist_mass_source: bool = False
    physical_reference: str | None = None
    runtime_profile: str = "default"
    experimental_open_release: bool | None = None

    def __post_init__(self) -> None:
        from .experiment_policy import validate_profile

        validate_profile(self.runtime_profile)
        if (
            self.experimental_open_release is not None
            and type(self.experimental_open_release) is not bool
        ):
            raise TypeError("experimental_open_release must be a boolean or None.")
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
        if type(self.fit_grasp_assets) is not bool:
            raise TypeError("fit_grasp_assets must be a boolean.")
        if type(self.twist_mass_source) is not bool:
            raise TypeError("twist_mass_source must be a boolean.")
        if self.physical_reference is not None:
            if not isinstance(self.physical_reference, str):
                raise TypeError("physical_reference must be a bundle path or None.")
            if not self.physical_reference.strip():
                raise ValueError("physical_reference must be a non-empty bundle path.")
            if self.runtime_profile != "default":
                raise ValueError("physical_reference requires runtime_profile=default.")
            if self.fit_grasp_assets or self.twist_mass_source:
                raise ValueError(
                    "physical_reference cannot be combined with physical adaptation."
                )


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
        "max_episodes",
        "max_episode_steps",
    }
    if (
        not required_planning
        <= set(planning)
        <= required_planning
        | {
            "fit_grasp_assets",
            "twist_mass_source",
            "physical_reference",
            "runtime_profile",
            "experimental_open_release",
        }
    ):
        raise ValueError("Task Engine planning configuration fields are invalid.")
    if (
        not {"num_envs", "success_policy", "min_successful_envs"}
        <= set(execution)
        <= {
            "num_envs",
            "success_policy",
            "min_successful_envs",
            "acceptance_mode",
        }
    ):
        raise ValueError("Task Engine execution configuration fields are invalid.")
    if planning.get("physical_reference") is not None:
        reference = planning["physical_reference"]
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError("physical_reference must be a non-empty bundle path.")
        reference_path = Path(reference).expanduser()
        if not reference_path.is_absolute():
            base = Path(path).expanduser().resolve().parent if path else Path.cwd()
            reference_path = base / reference_path
        planning["physical_reference"] = str(reference_path.resolve())
    return (
        TaskEngineWorkflowCfg(**workflow),
        TaskEnginePlanningCfg(**planning),
        TaskEngineExecutionCfg(**execution),
    )


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Task Engine {field_name} configuration must be a mapping.")
    return dict(value)
