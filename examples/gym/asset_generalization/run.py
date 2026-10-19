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

"""Prepare primitive cube variants and optionally collect measured UR5 Pick/Place."""

from __future__ import annotations

import argparse
from itertools import product
from tempfile import mkdtemp
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from embodichain.utils.config_paths import resolve_config_path
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.utils.registration import build_env
from embodichain.lab.sim.motion.expansion import ValidationCheck, ValidationResult
from embodichain.lab.sim.scene.expansion import SceneVariant
from embodichain.lab.task_program.integrations.scene_expansion import (
    execute_scene_variant,
)
from embodichain.lab.task_program.integrations.simulation.scene_expansion import (
    SimulationSceneExpansionHost,
)

CONFIGS = Path(__file__).resolve().parents[3] / "extensions/embodichain_tasks/configs"
TASK = CONFIGS / "tasks/manipulation/cube_platform_pick_place"


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _template(root: Path, *, side: float, seed: int, batch_index: int) -> Path:
    root.mkdir(parents=True)
    source = TASK / "task.ur5.yaml"
    task = yaml.safe_load(source.read_text())
    environment = yaml.safe_load(
        resolve_config_path(
            task["environment"]["component"], base_dir=source.parent
        ).read_text()
    )
    integration = yaml.safe_load(
        resolve_config_path(
            task["task_program"]["integration"], base_dir=source.parent
        ).read_text()
    )
    program = yaml.safe_load(
        resolve_config_path(
            task["task_program"]["program"], base_dir=source.parent
        ).read_text()
    )
    stem, version = task["id"].rsplit("-v", 1)
    task["id"] = f"{stem}-{int(side * 1000)}mm-{seed}-{batch_index}-v{version}"
    task["environment"] = {"component": "env.yaml"}
    task["embodiment"]["component"] = str(
        resolve_config_path(
            task["embodiment"]["component"], base_dir=source.parent
        ).resolve()
    )
    policy = resolve_config_path(
        task["task_program"]["execution_policy"], base_dir=source.parent
    )
    task["task_program"] = {
        "program": "program.yaml",
        "integration": "integration.yaml",
        "execution_policy": str(policy.resolve()),
    }
    cube = next(
        obj for obj in environment["simulation"]["rigid_object"] if obj["uid"] == "cube"
    )
    bottom = cube["init_pos"][2] - cube["shape"]["size"][2] / 2
    cube["shape"]["size"] = [side, side, side]
    cube["init_pos"][2] = bottom + side / 2
    platform = next(
        obj
        for obj in environment["simulation"]["background"]
        if obj["uid"] == "platform"
    )
    identity = f"primitive_{int(side * 1000)}mm_{seed}_{batch_index}"
    integration["integration_id"] += f"_{identity}"
    integration["scene_binding"]["registry_id"] += f"_{identity}"
    platform_binding = next(
        obj
        for obj in integration["scene_binding"]["rigid_objects"]
        if obj["entity_id"] == "platform"
    )
    support = next(
        item
        for item in platform_binding["affordances"]
        if item["entity_id"] == "platform_top"
    )
    # Parent-relative platform top + half the current primitive + release clearance.
    support["object_target_pose"][11] = (
        platform["shape"]["size"][2] / 2 + side / 2 + 0.002
    )
    for name, value in (
        ("task", task),
        ("env", environment),
        ("integration", integration),
        ("program", program),
    ):
        _write_yaml(root / f"{name}.yaml", value)
    path = root / "task.yaml"
    config_to_cfg(yaml.safe_load(path.read_text()), source_path=path)
    return path


def _world_bounds(env: Any, uid: str) -> np.ndarray:
    cfg = next(
        item for item in (*env.cfg.rigid_object, *env.cfg.background) if item.uid == uid
    )
    if cfg.shape.shape_type != "Cube":
        raise ValueError("Primitive baseline requires Cube shapes")
    corners = np.asarray(list(product((-0.5, 0.5), repeat=3))) * np.asarray(
        cfg.shape.size
    )
    pose = (
        env.sim.get_rigid_object(uid)
        .get_local_pose(to_matrix=True)[0]
        .detach()
        .cpu()
        .numpy()
    )
    # Transform all eight primitive corners so measured rotations remain visible.
    vertices = corners @ pose[:3, :3].T + pose[:3, 3]
    return np.stack((vertices.min(axis=0), vertices.max(axis=0)))


def _initial(env: Any) -> ValidationResult:
    cube, platform = _world_bounds(env, "cube"), _world_bounds(env, "platform")
    valid = (
        np.isfinite(cube).all()
        and np.isfinite(platform).all()
        and abs(cube[0, 2]) < 0.015
        and abs(platform[0, 2]) < 0.015
    )
    return ValidationResult(
        (
            ValidationCheck(
                "objects_supported_by_floor", "passed" if valid else "failed"
            ),
        )
    )


def _measured(env: Any) -> ValidationResult:
    cube, platform = _world_bounds(env, "cube"), _world_bounds(env, "platform")
    inset = np.minimum(cube[0, :2] - platform[0, :2], platform[1, :2] - cube[1, :2])
    gap = abs(float(cube[0, 2] - platform[1, 2]))
    finite = bool(np.isfinite(cube).all() and np.isfinite(platform).all())
    return ValidationResult(
        (
            ValidationCheck(
                "cube_footprint_on_platform",
                "passed" if finite and bool((inset >= 0).all()) else "failed",
            ),
            ValidationCheck(
                "cube_bottom_at_platform_top",
                "passed" if finite and gap < 0.012 else "failed",
                metrics={"vertical_gap_m": gap} if finite else {},
            ),
        )
    )


def _execute_batch(
    task_path: Path, *, seed: int, episodes: int, report: dict[str, Any]
) -> None:
    config = yaml.safe_load(task_path.read_text())
    env_cfg = config_to_cfg(config, source_path=task_path)
    env_cfg.seed = seed
    recordings = task_path.parent / "recordings"
    recordings.mkdir(exist_ok=True)
    run_root = Path(mkdtemp(prefix="run_", dir=recordings))
    env_cfg.record_trajectory = True
    env_cfg.trajectory_auto_save = True
    env_cfg.trajectory_save_dir = str(run_root / "trajectories")
    report["recording_root"] = str(run_root)
    env = build_env(config["id"], env_cfg)
    target = getattr(env, "unwrapped", env)
    execution_error: BaseException | None = None
    try:
        target.reset(seed=seed, options={"save_data": False})
        host = SimulationSceneExpansionHost(
            target, initial_validator=lambda: _initial(target), settle_steps=60
        )
        for episode_index in range(episodes):
            episode = {"episode_index": episode_index, "reset_submitted": False}
            report["episodes"].append(episode)
            variant = SceneVariant(
                parent_scene_id=config["id"],
                seed=seed,
                ordinal=episode_index,
                pose_changes=(),
                metadata=(("baseline", "primitive_cube_platform"),),
            )
            result = execute_scene_variant(
                env,
                variant,
                prepare_scene=host,
                measured_validator=lambda: _measured(target),
                episode_index=episode_index,
            )
            episode["accepted"] = result.accepted
            episode["validation"] = result.to_metadata()
            target._demo_episode_metadata[0]["primitive_variation"] = {
                "task_id": config["id"],
                "seed": seed,
                "cube_size_m": list(target.cfg.rigid_object[0].shape.size),
                "recording_root": str(run_root),
            }
            target.reset(
                options={
                    "save_data": result.accepted,
                    "commit_env_ids": [0] if result.accepted else [],
                }
            )
            episode["reset_submitted"] = True
    except BaseException as error:
        execution_error = error
        report["error_type"] = type(error).__name__
        report["detail"] = str(error)
        raise
    finally:
        try:
            target.close(exit_process=False)
            report["close_finalized"] = True
        except BaseException as close_error:
            report["close_error"] = str(close_error)
            if execution_error is None:
                raise
            execution_error.add_note(f"Environment close also failed: {close_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="New output directory"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run native simulation and save only accepted trajectories",
    )
    args = parser.parse_args()
    if args.batches < 1 or args.episodes < 1 or args.seed < 0:
        parser.error("batches/episodes must be positive and seed nonnegative")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    sides = (0.04, 0.05)
    deployments = [
        _template(
            output / f"batch_{index:04d}",
            side=sides[index % len(sides)],
            seed=args.seed + index,
            batch_index=index,
        )
        for index in range(args.batches)
    ]
    report: dict[str, Any] = {
        "baseline": "primitive_cube_platform",
        "status": "prepared",
        "deployments": [str(path) for path in deployments],
        "batches": [],
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"deployments": report["deployments"]}, indent=2))
    if args.execute:
        from embodichain.lab.sim import SimulationManager

        report["status"] = "executing"
        try:
            for index, task_path in enumerate(deployments):
                batch: dict[str, Any] = {
                    "batch_index": index,
                    "task_config": str(task_path),
                    "episodes": [],
                }
                report["batches"].append(batch)
                _execute_batch(
                    task_path,
                    seed=args.seed + index,
                    episodes=args.episodes,
                    report=batch,
                )
                # Drain only after the closed environment's helper frame unwinds.
                SimulationManager.flush_cleanup_queue()
            report["status"] = "completed"
        except BaseException as error:
            report["status"] = "failed"
            report["error_type"] = type(error).__name__
            report["detail"] = str(error)
            raise
        finally:
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not all(
            episode["accepted"]
            for batch in report["batches"]
            for episode in batch["episodes"]
        ):
            raise SystemExit(f"Primitive acceptance failed; inspect {report_path}")
    print(f"Primitive baseline report: {report_path}")


if __name__ == "__main__":
    main()
