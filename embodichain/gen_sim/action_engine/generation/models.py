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

"""Small value objects used by Action Engine config generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["GeneratedConfigPaths", "PreparedScene"]


@dataclass(frozen=True)
class GeneratedConfigPaths:
    """Paths written by one successful generation transaction."""

    gym_config: Path
    agent_config: Path
    task_agent: Path
    execution_program: Path
    seed_task_graph_png: Path

    @property
    def seed_task_graph(self) -> Path:
        """Expose the user-facing filename without creating a second artifact."""
        return self.execution_program


@dataclass(frozen=True)
class PreparedScene:
    """A source scene normalized for both planning and simulator loading."""

    source_config_path: Path
    scene_dir: Path
    planner_objects: tuple[dict[str, Any], ...]
    background: tuple[dict[str, Any], ...]
    rigid_objects: tuple[dict[str, Any], ...]
    articulations: tuple[dict[str, Any], ...]
    uid_map: dict[str, str]
    table_top_z: float | None
    z_rotation_degrees: float
    body_scale_policy: str
    body_scale: tuple[float, float, float]
    asset_hashes: dict[str, str]
    asset_provenance: tuple[dict[str, Any], ...] = ()
