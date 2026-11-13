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
    task: Path | None = None
    device: str | None = None
    headless: bool = True
    planning_repairs: int = 1
    source_calibration: Path | None = None
    room: str | None = None
    recording: bool = False
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
        if type(self.planning_repairs) is not int or self.planning_repairs < 0:
            raise ValueError("planning_repairs must be a nonnegative integer")
        if self.room is not None and (
            type(self.room) is not str or not self.room.isidentifier()
        ):
            raise ValueError("room must name a scene asset class such as WoodLab")
        if type(self.recording) is not bool:
            raise TypeError("recording must be a boolean")
        if type(self.binding) is not dict or set(self.binding) - {
            "profile",
            "release_clearance",
            "resource_anchors",
            "runtime_services",
            "scene_binding",
            "segments",
        }:
            raise ValueError(
                "binding supports profile, release_clearance, resource_anchors, "
                "runtime_services, scene_binding and segments"
            )
        clearance = self.binding.get("release_clearance", 0.0)
        if (
            isinstance(clearance, bool)
            or not isinstance(clearance, (int, float))
            or not 0.0 <= clearance < 0.2
        ):
            raise ValueError("release_clearance must be a height in [0, 0.2) m")
        anchors = self.binding.get("resource_anchors", {})
        if type(anchors) is not dict or any(
            type(value) is not list
            or len(value) != 2
            or not all(isinstance(x, (int, float)) for x in value)
            for value in anchors.values()
        ):
            raise ValueError("resource_anchors maps resources to planar [x, y] anchors")
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
    path_fields = {
        "image",
        "output",
        "runtime",
        "execution_policy",
        "scene",
        "task",
        "source_calibration",
    }
    for name in allowed:
        override = getattr(args, name, None)
        if override is not None:
            value[name] = override
        if name in path_fields and name in value:
            raw = value[name]
            if raw is None and name in {"scene", "task", "source_calibration"}:
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
    inputs = (cfg.image, cfg.runtime, cfg.execution_policy, cfg.source_calibration)
    for path in (path for path in inputs if path is not None):
        if not path.is_file():
            raise FileNotFoundError(path)
        if cfg.output == path or cfg.output in path.parents:
            raise ValueError("GenSim output must not contain an input file")
    if cfg.task is not None and not (cfg.task / "task_plan_manifest.json").is_file():
        raise ValueError("task must name a published task plan directory")
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
