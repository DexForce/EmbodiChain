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
from copy import deepcopy
from itertools import product
from tempfile import mkdtemp
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

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
TASK = CONFIGS / "tasks/manipulation/repeated_pick_place"


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _template(root: Path, *, side: float, seed: int, batch_index: int) -> Path:
    root.mkdir(parents=True)
    task = yaml.safe_load((TASK / "task.ur5.yaml").read_text())
    task.pop("expansion", None)
    task.update(
        id=f"PrimitiveCubePlatform-{int(side * 1000)}mm-{seed}-{batch_index}-v0",
        headless=True,
        renderer="no-render",
        enable_sensor=False,
    )
    task["environment"] = {"component": "env.yaml"}
    task["embodiment"]["component"] = str(
        CONFIGS / "components/embodiments/ur5_dh_pgi_140_80.yaml"
    )
    task["task_program"] = {
        "program": "program.yaml",
        "integration": "integration.yaml",
        "execution_policy": str(
            CONFIGS / "components/execution_policies/trajectory_open_loop.yaml"
        ),
    }
    environment = yaml.safe_load((TASK / "envs/default.yaml").read_text())
    environment["max_episode_steps"] = 2400
    environment["env"]["dataset"] = {}
    environment["env"].update(record_trajectory=True, trajectory_auto_save=True)
    cube = environment["simulation"]["rigid_object"][0]
    cube["shape"] = {"shape_type": "Cube", "size": [side, side, side]}
    cube["init_pos"][2] = side / 2
    platform = deepcopy(environment["simulation"]["rigid_object"][0])
    platform.update(uid="platform", body_type="static", init_pos=[-0.42, 0.30, 0.025])
    platform["shape"]["size"] = [0.16, 0.16, 0.05]
    environment["simulation"]["background"].append(platform)
    integration = yaml.safe_load((TASK / "task_program/integration.yaml").read_text())
    identity = f"primitive_{int(side * 1000)}mm_{seed}_{batch_index}"
    integration["integration_id"] += f"_{identity}"
    integration["scene_binding"]["registry_id"] += f"_{identity}"
    target_pose = np.eye(4)
    # Local platform top + half the source size + release clearance.
    target_pose[2, 3] = 0.05 / 2 + side / 2 + 0.002
    integration["scene_binding"]["rigid_objects"].append(
        {
            "entity_id": "platform",
            "simulation_uid": "platform",
            "dynamics": "static",
            "semantic_type": "platform",
            "affordances": [
                {
                    "kind": "support_surface",
                    "entity_id": "platform_top",
                    "native_name": "top",
                    "object_target_frame": "parent",
                    "object_target_pose": target_pose.reshape(-1).tolist(),
                }
            ],
        }
    )
    program = {
        "program_id": integration["program_id"],
        "instruction": "Pick up the cube and place it on the platform.",
        "targets": {},
        "program": {
            "kind": "sequence",
            "items": [
                {
                    "kind": "segment",
                    "name": "pick",
                    "steps": {
                        "kind": "invoke",
                        "call": {"kind": "pick", "object": "cube"},
                    },
                    "validators": [
                        {
                            "kind": "object_lifted",
                            "object": "cube",
                            "minimum_lift": 0.03,
                        }
                    ],
                },
                {
                    "kind": "segment",
                    "name": "place",
                    "steps": {
                        "kind": "invoke",
                        "call": {
                            "kind": "place",
                            "object": "cube",
                            "on": "platform_top",
                        },
                    },
                    "post": [{"kind": "wait_stable", "entity": "cube"}],
                    "validators": [{"kind": "object_upright", "object": "cube"}],
                },
            ],
        },
    }
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
    print(f"Primitive baseline report: {report_path}")


if __name__ == "__main__":
    main()
