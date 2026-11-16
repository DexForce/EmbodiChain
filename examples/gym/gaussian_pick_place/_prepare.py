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

"""Resolve static Task Program deployments against the shared GS asset cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import yaml

__all__ = ["HERE", "SCENES", "prepare", "digest", "write_json"]

HERE = Path(__file__).resolve().parent
SCENES = {
    "discoverse": {"title": "DISCOVERSE lab3", "asset": "DiscoverseLab3GS"},
    "office": {"title": "RoboSplatter office", "asset": "RoboSplatterOfficeGS"},
    "robosimgs": {"title": "RoboSimGS tabletop", "asset": "RoboSimGSTabletopGS"},
}


def digest(path: Path) -> str:
    """Return the SHA256 of one locally owned artifact."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Atomically replace one example-owned JSON report."""
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def prepare(
    scene: str, output: Path, seed: int, spp: int, viewer: bool
) -> tuple[Any, dict[str, Any]]:
    """Snapshot and preflight one componentized Task Program before World creation.

    Args:
        scene: Name of one shipped deployment.
        output: Fresh run directory owned by this invocation.
        seed: Native episode seed.
        spp: Mesh-rendering sample budget for both RGB cameras.
        viewer: Whether to open the native preview window.

    Returns:
        The typed environment configuration and an owned run report.
    """
    from embodichain.data import get_data_path
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg

    recipe = SCENES[scene]
    resolved = output / "resolved"
    shutil.copytree(HERE / "configs", resolved)
    env_path = resolved / f"env.{scene}.yaml"
    environment = yaml.safe_load(env_path.read_text())
    environment["render_cfg"]["spp"] = spp
    env_path.write_text(yaml.safe_dump(environment, sort_keys=False))
    source = resolved / f"task.{scene}.yaml"
    cfg = config_to_cfg(yaml.safe_load(source.read_text()), source_path=source)
    cfg.record_trajectory = True
    cfg.trajectory_auto_save = False
    cfg.init_rollout_buffer = False
    cfg.seed = seed
    cfg.sim_cfg.headless = not viewer
    cfg.sim_cfg.startup_summary = "off"
    entry = Path(get_data_path(recipe["asset"] + "/asset.usda"))
    asset_document = json.loads((entry.parent / "asset.json").read_text())
    program = yaml.safe_load(
        (resolved / f"task_program/program.{scene}.yaml").read_text()
    )
    embodiment = yaml.safe_load((resolved / f"embodiment.{scene}.yaml").read_text())
    deployment = yaml.safe_load(source.read_text())
    write_json(
        resolved / "runtime_overrides.json",
        {
            "record_trajectory": True,
            "trajectory_auto_save": False,
            "init_rollout_buffer": False,
            "seed": seed,
            "headless": not viewer,
        },
    )
    code = resolved / "code"
    code.mkdir()
    for path in HERE.glob("*.py"):
        shutil.copyfile(path, code / path.name)
    report = {
        "scene": scene,
        "title": recipe["title"],
        "task_id": deployment["id"],
        "task_config": source.name,
        "seed": seed,
        "robot": {
            "model": embodiment["simulation"]["arm_model"],
            "base": deployment["embodiment"]["overrides"]["init_pos"],
            "same_initial_qpos_as_visual_preview": True,
        },
        "physics": {
            "backend": "default",
            "device": "cpu",
            "cube_body_type": "dynamic",
            "cube_mass_kg": 0.05,
            "effect_assurance": "verified",
            "target_center": program["targets"]["rest_pose"]["position"],
        },
        "asset": {
            "reference": recipe["asset"] + "/asset.usda",
            "asset_id": asset_document["asset_id"],
            "revision_id": asset_document["revision_id"],
            "asset_json_sha256": digest(entry.parent / "asset.json"),
        },
        "cameras": ["cam_third", "cam_wrist"],
        "python": sys.version,
        "configuration_hashes": {
            p.relative_to(resolved).as_posix(): digest(p)
            for p in resolved.rglob("*")
            if p.is_file() and "code" not in p.parts
        },
        "source_hashes": {p.name: digest(p) for p in code.iterdir()},
        "static_preflight": "passed configuration, scene binding, contract and Semantic Call lowering",
        "status": "prepared",
    }
    write_json(output / "run.json", report)
    return cfg, report
