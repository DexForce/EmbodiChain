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

"""Configuration for the application-owned GenSim execution workflow."""

from __future__ import annotations

import argparse
from dataclasses import MISSING, field
from pathlib import Path
from typing import Any

from embodichain.utils import configclass
from embodichain.utils.config_paths import resolve_config_path
from embodichain.utils.utility import load_config

__all__: list[str] = []


@configclass
class _RunCfg:
    image: Path = MISSING
    instruction: str = MISSING
    output: Path = MISSING
    runtime: Path = MISSING
    execution_policy: Path = MISSING
    task_id: str = "gensim_task"
    scene: Path | None = None
    device: str | None = None
    headless: bool = True
    binding: dict[str, Any] = field(default_factory=dict)
    scene_provider: dict[str, Any] = field(
        default_factory=lambda: {"id": "legacy_scene_engine", "profile": {}}
    )

    def __post_init__(self) -> None:
        for name in ("instruction", "task_id"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if self.device is not None and (
            type(self.device) is not str or not self.device.strip()
        ):
            raise ValueError("device must be a nonempty string or null")
        if type(self.headless) is not bool:
            raise TypeError("headless must be a boolean")
        if type(self.binding) is not dict or set(self.binding) - {
            "profile",
            "runtime_services",
            "scene_binding",
            "segments",
        }:
            raise ValueError(
                "binding supports profile, runtime_services, scene_binding and segments"
            )
        if (
            type(self.scene_provider) is not dict
            or set(self.scene_provider) != {"id", "profile"}
            or not isinstance(self.scene_provider["id"], str)
            or not self.scene_provider["id"].strip()
            or not isinstance(self.scene_provider["profile"], dict)
        ):
            raise ValueError("scene_provider requires id and profile mappings")


def _load_run_config(args: argparse.Namespace) -> _RunCfg:
    source = args.config.expanduser().resolve(strict=True)
    value = load_config(source)
    if type(value) is not dict:
        raise TypeError("GenSim run configuration must be a mapping")
    value = dict(value)
    if value.pop("schema_version", None) != "embodichain.gensim-run/v1":
        raise ValueError("schema_version must be embodichain.gensim-run/v1")
    allowed = set(_RunCfg.__annotations__)
    if set(value) - allowed:
        raise ValueError(
            f"Unsupported GenSim run fields: {sorted(set(value) - allowed)}"
        )
    path_fields = {"image", "output", "runtime", "execution_policy", "scene"}
    for name in allowed:
        override = getattr(args, name, None)
        if override is not None:
            value[name] = override
        if name in path_fields and name in value:
            raw = value[name]
            if raw is None and name == "scene":
                continue
            if not isinstance(raw, (str, Path)) or not str(raw).strip():
                raise ValueError(f"{name} must be a nonempty path")
            value[name] = resolve_config_path(
                str(raw), base_dir=Path.cwd() if override is not None else source.parent
            ).resolve()
    required = {"image", "instruction", "output", "runtime", "execution_policy"}
    if missing := required - set(value):
        raise ValueError(f"Missing GenSim run fields: {sorted(missing)}")
    cfg = _RunCfg(**value)
    cfg.validate()
    for path in (cfg.image, cfg.runtime, cfg.execution_policy):
        if not path.is_file():
            raise FileNotFoundError(path)
        if cfg.output == path or cfg.output in path.parents:
            raise ValueError("GenSim output must not contain an input file")
    if cfg.scene is not None:
        if not (cfg.scene / "scene_export/scene_config.json").is_file():
            raise ValueError(
                "scene must name a workspace containing scene_export/scene_config.json"
            )
        if (
            cfg.scene == cfg.output
            or cfg.scene in cfg.output.parents
            or cfg.output in cfg.scene.parents
        ):
            raise ValueError("GenSim output and source scene must not overlap")
    return cfg
