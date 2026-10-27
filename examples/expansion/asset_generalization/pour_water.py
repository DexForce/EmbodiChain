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

"""Replace bottle/cup assets and measure a four-segment PourWater motion.

This example verifies rigid-body pouring geometry, not liquid transfer. Default
catalogs contain explicitly labelled size derivatives of the existing open-neck
PourWaterAssets models; --bottle-catalog/--cup-catalog select authored alternatives.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile, mkdtemp
from typing import Any

import gymnasium as gym
import numpy as np
import trimesh
import yaml

from embodichain.data import get_data_path
from embodichain.data.asset_catalog import (
    AssetDescriptor,
    Bounds3D,
    DirectoryAssetProvider,
)
from embodichain.lab.gym.envs.asset_generalization import (
    AssetGeneralizationCfg,
    collect_asset_batches,
    prepare_asset_batches,
)
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.utils.registration import build_env
from embodichain.lab.sim.motion.expansion import ValidationCheck, ValidationResult
from embodichain.lab.task_program.integrations.asset_materialization import (
    AssetAdaptationCfg,
    AssetPourAdaptationCfg,
    MaterializedAssetDeployment,
)
from embodichain.lab.task_program.integrations.simulation.asset_pouring import (
    PouringAcceptanceMonitor,
)
from embodichain.utils.config_paths import resolve_config_path

CONFIGS = resolve_config_path(
    "extensions/embodichain_tasks/configs", base_dir=Path(__file__).parent
)
TASK = Path(__file__).resolve().parent / "configs/pour_water"


def _yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _prepare_catalogs(output: Path) -> tuple[Path, Path]:
    source = Path(get_data_path("PourWaterAssets/bottle.glb")).parent
    integration = _yaml(TASK / "task_program/integration.yaml")
    grasp = np.asarray(
        integration["profile"]["action_options"]["pick"]["fixed_object_to_eef"]
    ).reshape(4, 4)
    axis = integration["scene_binding"]["rigid_objects"][1]["affordances"][0][
        "internal_axis"
    ]
    profile_id = _yaml(CONFIGS / "components/embodiments/cobotmagic.yaml")[
        "skill_profile"
    ]["profile_id"]
    rotation = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    for role, variants in (
        ("bottle", (("original", [1.0, 1.0, 1.0]), ("tall", [1.0, 1.0, 1.08]))),
        ("cup", (("original", [1.0, 1.0, 1.0]), ("wide", [1.1, 1.1, 0.95]))),
    ):
        for variant, scale in variants:
            package = output / role / f"{role}-{variant}"
            package.mkdir(parents=True)
            mesh = trimesh.load(source / f"{role}.glb", force="mesh", process=False)
            # PourWater GLBs are Y-up. Bake the same Y-to-Z boundary conversion
            # used by their runtime import into OBJ geometry exactly once.
            mesh.apply_transform(rotation)
            mesh.apply_scale(scale)
            mesh.export(package / "model.obj")
            lower, upper = (tuple(float(v) for v in row) for row in mesh.bounds)
            provenance: dict[str, Any] = {
                "source": str(source / f"{role}.glb"),
                "variant": variant,
                "baked_scale": scale,
                "coordinate_conversion": "Y-up GLB to Z-up OBJ, +90 degrees around X",
            }
            if role == "bottle":
                transformed_grasp = grasp.copy()
                transformed_grasp[:3, 3] *= np.asarray(scale)
                pouring = {
                    "mouth_center_m": [
                        0.0000537652522 * scale[0],
                        -0.0000407565385 * scale[1],
                        upper[2],
                    ],
                    "mouth_radius_m": 0.0194 * max(scale[:2]),
                    "tilt_axis": axis,
                    "object_to_eef": transformed_grasp.reshape(-1).tolist(),
                    "profile_id": profile_id,
                    "evidence": "Open rim measured from PourWaterAssets mesh; CobotMagic task grasp translated by the documented baked scale. Runtime acceptance separately validates the resulting grasp and pour.",
                }
                capabilities = ("antipodal_grasp", "pour_source")
                region = None
            else:
                radius = 0.0275 * min(scale[:2])
                pouring = {
                    "mouth_center_m": [0.0, 0.0, upper[2]],
                    "opening_radius_m": radius,
                    "evidence": "Conservative opening inscribed in the existing open-cup rim, with explicit baked scale; not inferred from exterior bounds.",
                }
                capabilities = ("container",)
                half = radius / np.sqrt(2)
                region = Bounds3D(
                    (-half, -half, lower[2] + 0.008 * scale[2]), (half, half, upper[2])
                )
            provenance["pouring"] = pouring
            descriptor = AssetDescriptor(
                asset_id=f"{role}-{variant}",
                category=role,
                attributes={"open_top": True},
                capabilities=capabilities,
                bounds_m=Bounds3D(lower, upper),
                container_region=region,
                normalized=True,
                provenance=provenance,
            )
            (package / "asset.catalog.json").write_text(
                json.dumps(
                    {"descriptor": descriptor.to_dict(), "visual_path": "model.obj"},
                    indent=2,
                ),
                encoding="utf-8",
            )
            for name in ("ATTRIBUTION.md", "CC-BY-4.0.txt"):
                shutil.copy2(source / name, package / name)
    return output / "bottle", output / "cup"


def _template(output: Path) -> Path:
    output.mkdir(parents=True)
    source = TASK / "task.cobotmagic.yaml"
    task = _yaml(source)
    env = _yaml(
        resolve_config_path(task["environment"]["component"], base_dir=source.parent)
    )
    integration = _yaml(
        resolve_config_path(task["task_program"]["integration"], base_dir=source.parent)
    )
    program = _yaml(
        resolve_config_path(task["task_program"]["program"], base_dir=source.parent)
    )
    task["environment"] = {"component": "env.yaml"}
    task["embodiment"]["component"] = str(
        resolve_config_path(
            task["embodiment"]["component"], base_dir=source.parent
        ).resolve()
    )
    policy = task["task_program"]["execution_policy"]
    if isinstance(policy, dict):
        policy["component"] = str(
            resolve_config_path(policy["component"], base_dir=source.parent).resolve()
        )
    else:
        policy = str(resolve_config_path(policy, base_dir=source.parent).resolve())
    task["task_program"] = {
        "program": "program.yaml",
        "integration": "integration.yaml",
        "execution_policy": policy,
    }
    for name, value in (
        ("task", task),
        ("env", env),
        ("integration", integration),
        ("program", program),
    ):
        _write_yaml(output / f"{name}.yaml", value)
    return output / "task.yaml"


def _table_top_m(background: list[dict[str, Any]]) -> float:
    table = next(obj for obj in background if obj["uid"] == "table")
    if (
        table["shape"]["shape_type"] != "Cube"
        or any(table.get("init_rot", [0, 0, 0]))
        or table.get("init_local_pose") is not None
        or any(value != 1 for value in table.get("body_scale", [1, 1, 1]))
    ):
        raise ValueError("Pour acceptance requires an upright, unscaled Cube table")
    return float(table["init_pos"][2] + table["shape"]["size"][2] / 2)


def _poses(env: Any) -> tuple[np.ndarray, np.ndarray]:
    return tuple(
        env.sim.get_rigid_object(uid)
        .get_local_pose(to_matrix=True)[0]
        .detach()
        .cpu()
        .numpy()
        for uid in ("bottle", "cup")
    )


class _ObservedPour(gym.Wrapper):
    def step(self, action: Any) -> Any:
        result = self.env.step(action)
        target = self.unwrapped
        bottle, cup = _poses(target)
        target._pour_monitor.observe(
            bottle, cup, in_pour_segment=int(target._demo_active_segment_ids[0]) == 2
        )
        return result


def _environment(
    deployment: MaterializedAssetDeployment, *, seed: int
) -> _ObservedPour:
    config = _yaml(deployment.deployment_path)
    cfg = config_to_cfg(config, source_path=deployment.deployment_path)
    cfg.seed = seed
    cfg.record_trajectory = True
    cfg.trajectory_auto_save = True
    parent = deployment.deployment_path.parent / "recordings"
    parent.mkdir(exist_ok=True)
    run_root = Path(mkdtemp(prefix="run_", dir=parent))
    cfg.trajectory_save_dir = str(run_root / "trajectories")
    table_top = _table_top_m(config["background"])
    env = build_env(config["id"], cfg)
    env._pour_table_top_m = table_top
    env._pour_initial_robot_id = id(env.robot)
    env._pour_initial_table_id = id(env.sim.get_rigid_object("table"))
    env._asset_generalization_recording_root = str(run_root)
    return _ObservedPour(env)


def _initial(env: Any) -> ValidationResult:
    descriptors = {
        role: asset["descriptor"]
        for role, asset in env._asset_generalization_manifest["resolved_assets"][
            "assets"
        ].items()
    }
    bottle_record, cup_record = (
        descriptors[role]["provenance"]["pouring"] for role in ("bottle", "cup")
    )
    env._pour_descriptors = descriptors
    env._pour_monitor = PouringAcceptanceMonitor(
        bottle_mouth_xyz=bottle_record["mouth_center_m"],
        cup_mouth_xyz=cup_record["mouth_center_m"],
        bottle_mouth_radius_m=bottle_record["mouth_radius_m"],
        cup_opening_radius_m=cup_record["opening_radius_m"],
    )
    bottle, cup = _poses(env)
    env._pour_monitor.reset(bottle, cup)
    checks = [
        ValidationCheck(
            "robot_instance_preserved",
            "passed" if id(env.robot) == env._pour_initial_robot_id else "failed",
        ),
        ValidationCheck(
            "table_instance_preserved",
            (
                "passed"
                if id(env.sim.get_rigid_object("table")) == env._pour_initial_table_id
                else "failed"
            ),
        ),
    ]
    for role, pose in (("bottle", bottle), ("cup", cup)):
        bottom = float(pose[2, 3] + env._pour_descriptors[role]["bounds_m"]["lower"][2])
        gap = abs(bottom - env._pour_table_top_m)
        checks.append(
            ValidationCheck(
                f"{role}_supported",
                "passed" if gap < 0.015 and pose[2, 2] > 0.98 else "failed",
                metrics={"bottom_gap_m": gap},
            )
        )
    arm_speed = max(
        float(env.robot.get_qvel(name=part).abs().max().item())
        for part in ("left_arm", "right_arm")
    )
    checks.append(
        ValidationCheck(
            "robot_ready_motion",
            "passed" if np.isfinite(arm_speed) and arm_speed < 0.5 else "failed",
            metrics=(
                {"max_arm_speed_rad_s": arm_speed} if np.isfinite(arm_speed) else {}
            ),
        )
    )
    return ValidationResult(tuple(checks))


def _measured(env: Any) -> ValidationResult:
    result = env._pour_monitor.validate_final(*_poses(env))
    env._demo_episode_metadata[0]["pour_acceptance"] = env._pour_monitor.to_metadata()
    env._demo_episode_metadata[0]["scene_instances"] = {
        "environment": id(env),
        "robot": id(env.robot),
        "table": id(env.sim.get_rigid_object("table")),
        "bottle": id(env.sim.get_rigid_object("bottle")),
        "cup": id(env.sim.get_rigid_object("cup")),
    }
    # Diagnostic evidence is kept separately from accepted trajectory data.
    report = {
        "monitor": env._pour_monitor.to_metadata(),
        "episode": env.get_demo_episode_metadata(0),
        "robot_qpos": env.robot.get_qpos().detach().cpu().tolist(),
        "robot_targets": env.robot.get_qpos(target=True).detach().cpu().tolist(),
    }
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="pour-diagnostics-",
        suffix=".json",
        dir=env._asset_generalization_recording_root,
        delete=False,
    ) as diagnostic:
        diagnostic.write(json.dumps(report, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bottle-catalog", type=Path)
    parser.add_argument("--cup-catalog", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if (args.bottle_catalog is None) != (args.cup_catalog is None):
        parser.error("Provide both --bottle-catalog and --cup-catalog, or neither.")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    bottle_dir, cup_dir = (
        (args.bottle_catalog, args.cup_catalog)
        if args.bottle_catalog is not None
        else _prepare_catalogs(output / "catalog")
    )
    cfg = AssetGeneralizationCfg(
        seed=args.seed,
        max_batches=args.batches,
        episodes_per_batch=args.episodes,
        record_trajectory=True,
    )
    template = _template(output / "template")
    table_top = _table_top_m(
        _yaml(template.parent / "env.yaml")["simulation"]["background"]
    )
    deployments = prepare_asset_batches(
        template,
        {
            "bottle": DirectoryAssetProvider(bottle_dir),
            "cup": DirectoryAssetProvider(cup_dir),
        },
        cfg=cfg,
        output_dir=output / "batches",
        adaptations=AssetAdaptationCfg(
            initial_bottom_z_m={
                "bottle": table_top + 0.001,
                "cup": table_top + 0.002,
            },
            pour=AssetPourAdaptationCfg(mouth_clearance_m=0.15),
        ),
    )
    print(
        json.dumps(
            {"deployments": [str(d.deployment_path) for d in deployments]}, indent=2
        )
    )
    if args.execute:
        outcomes = collect_asset_batches(
            deployments,
            cfg=cfg,
            initial_validator=_initial,
            measured_validator=_measured,
            environment_factory=_environment,
            report_path=output / "report.json",
        )
        print(f"Motion acceptance report: {output / 'report.json'}")
        if not all(
            episode.accepted for batch in outcomes for episode in batch.episodes
        ):
            raise SystemExit(
                "Pour acceptance failed; inspect report.json and pour-diagnostics-*.json"
            )


if __name__ == "__main__":
    main()
