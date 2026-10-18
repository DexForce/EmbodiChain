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

"""Prepare local cube variants and optionally collect measured UR5 Pick/Place."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import yaml

from embodichain.data.asset_catalog import (
    AssetDescriptor,
    AssetQuery,
    Bounds3D,
    DirectoryAssetProvider,
    ExplicitAssetProvider,
)
from embodichain.lab.gym.envs.asset_generalization import (
    AssetGeneralizationCfg,
    collect_asset_batches,
    prepare_asset_batches,
)
from embodichain.lab.sim.motion.expansion import ValidationCheck, ValidationResult
from embodichain.lab.task_program.semantics.asset_requirements import (
    AssetRelationRequirement,
    AssetRequirements,
    AssetRoleRequirement,
)

CONFIGS = Path(__file__).resolve().parents[3] / "extensions/embodichain_tasks/configs"
TASK = CONFIGS / "tasks/manipulation/repeated_pick_place"


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _asset(
    root: Path, name: str, size: tuple[float, float, float], *, support: bool = False
) -> Path:
    package = root / name
    package.mkdir(parents=True)
    mesh = trimesh.creation.box(extents=size)
    mesh.export(package / "model.obj")
    lower, upper = (tuple(float(v) for v in row) for row in mesh.bounds)
    descriptor = AssetDescriptor(
        asset_id=name,
        category="platform" if support else "cube",
        capabilities=("support_surface",) if support else ("antipodal_grasp",),
        bounds_m=Bounds3D(lower, upper),
        support_region=(
            Bounds3D((lower[0], lower[1], upper[2]), upper) if support else None
        ),
        normalized=True,
        provenance={
            "geometry": "trimesh box in meters",
            "capabilities": "declared by example",
        },
    )
    (package / "asset.catalog.json").write_text(
        json.dumps(
            {"descriptor": descriptor.to_dict(), "visual_path": "model.obj"}, indent=2
        ),
        encoding="utf-8",
    )
    return package


def _template(root: Path) -> Path:
    root.mkdir(parents=True)
    task = yaml.safe_load((TASK / "task.ur5.yaml").read_text())
    task.pop("expansion", None)
    task.update(
        id="AssetGeneralizationCubePlatform-v0",
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
    platform = deepcopy(environment["simulation"]["rigid_object"][0])
    platform.update(uid="platform", body_type="static", init_pos=[-0.42, 0.30, 0.025])
    platform["shape"]["size"] = [0.16, 0.16, 0.05]
    environment["simulation"]["background"].append(platform)
    integration = yaml.safe_load((TASK / "task_program/integration.yaml").read_text())
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
                }
            ],
        }
    )
    requirements = AssetRequirements(
        roles={
            "source": AssetRoleRequirement(
                "cube",
                AssetQuery(category="cube", capabilities=("antipodal_grasp",)),
                max_extent_m=0.06,
            ),
            "target": AssetRoleRequirement(
                "platform",
                AssetQuery(category="platform", capabilities=("support_surface",)),
            ),
        },
        relations=(
            AssetRelationRequirement(
                "supported_by", "source", "target", clearance_m=0.002
            ),
        ),
    )
    integration["asset_requirements"] = requirements.to_dict()
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
    return root / "task.yaml"


def _world_bounds(env: Any, uid: str) -> np.ndarray:
    cfg = next(
        item for item in (*env.cfg.rigid_object, *env.cfg.background) if item.uid == uid
    )
    mesh = trimesh.load(cfg.shape.fpath, force="mesh")
    pose = (
        env.sim.get_rigid_object(uid)
        .get_local_pose(to_matrix=True)[0]
        .detach()
        .cpu()
        .numpy()
    )
    # Transform every mesh vertex, not just an object center or two box corners.
    vertices = trimesh.transform_points(np.asarray(mesh.vertices), pose)
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
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    for side in (0.04, 0.045, 0.05):
        _asset(output / "catalog/cubes", f"cube-{side:.3f}", (side, side, side))
    platform = _asset(
        output / "catalog/platforms", "platform", (0.16, 0.16, 0.05), support=True
    )
    task = _template(output / "template")
    cfg = AssetGeneralizationCfg(
        seed=args.seed,
        max_batches=args.batches,
        episodes_per_batch=args.episodes,
        settle_steps=60,
        record_trajectory=True,
    )
    deployments = prepare_asset_batches(
        task,
        {
            "source": DirectoryAssetProvider(output / "catalog/cubes"),
            "target": ExplicitAssetProvider([platform]),
        },
        cfg=cfg,
        output_dir=output / "batches",
    )
    print(
        json.dumps(
            {"deployments": [str(item.deployment_path) for item in deployments]},
            indent=2,
        )
    )
    if args.execute:
        collect_asset_batches(
            deployments,
            cfg=cfg,
            initial_validator=_initial,
            measured_validator=_measured,
            report_path=output / "report.json",
        )
        print(f"Execution report: {output / 'report.json'}")


if __name__ == "__main__":
    main()
