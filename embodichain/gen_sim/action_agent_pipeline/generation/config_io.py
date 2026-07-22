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
import json
from pathlib import Path
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    GeneratedActionAgentConfigPaths,
)

__all__ = [
    "read_json",
    "raise_if_generated_files_exist",
    "write_config_bundle",
    "write_json",
    "write_text",
]


def write_config_bundle(
    *,
    output_dir: Path,
    bundle: Mapping[str, Any],
    overwrite: bool,
) -> GeneratedActionAgentConfigPaths:
    paths = GeneratedActionAgentConfigPaths(
        output_dir=output_dir,
        gym_config=output_dir / "fast_gym_config.json",
        agent_config=output_dir / "agent_config.json",
        task_prompt=output_dir / "task_prompt.txt",
        task_graph=output_dir / "task_graph.json",
        basic_background=output_dir / "basic_background.txt",
        atom_actions=output_dir / "atom_actions.txt",
        summary=dict(bundle.get("summary", {})),
    )
    raise_if_generated_files_exist(output_dir, overwrite)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(paths.gym_config, bundle["gym_config"])
    write_json(paths.agent_config, bundle["agent_config"])
    write_text(paths.task_prompt, bundle["task_prompt"])
    write_json(paths.task_graph, bundle["task_graph"])
    write_text(paths.basic_background, bundle["basic_background"])
    write_text(paths.atom_actions, bundle["atom_actions"])
    return paths


def raise_if_generated_files_exist(output_dir: Path, overwrite: bool) -> None:
    if overwrite:
        return
    output_files = [
        output_dir / "fast_gym_config.json",
        output_dir / "agent_config.json",
        output_dir / "task_prompt.txt",
        output_dir / "task_graph.json",
        output_dir / "basic_background.txt",
        output_dir / "atom_actions.txt",
    ]
    existing = [path for path in output_files if path.exists()]
    if existing:
        existing_text = ", ".join(path.as_posix() for path in existing)
        raise FileExistsError(
            f"Generated file(s) already exist: {existing_text}. "
            "Pass overwrite=True or --overwrite to replace them."
        )


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
