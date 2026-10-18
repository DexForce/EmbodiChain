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

"""Production-loader coverage for asset-bound Task Program deployments."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest
import yaml
import trimesh

from embodichain.data.asset_catalog import (
    AssetDescriptor,
    AssetQuery,
    Bounds3D,
    ExplicitAssetProvider,
    PreparationContext,
)
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.utils._component_composition import _resolve_gym_components
from embodichain.lab.task_program.integrations.asset_materialization import (
    MaterializedAssetDeployment,
    load_asset_requirements_from_deployment,
    materialize_asset_deployment,
    validate_materialized_deployment,
)
from embodichain.lab.task_program.semantics.asset_matching import resolve_asset_set
from embodichain.lab.task_program.semantics.asset_requirements import (
    AssetRequirements,
    AssetRoleRequirement,
    AssetRelationRequirement,
)

ROOT = Path(__file__).resolve().parents[4]
CONFIGS = ROOT / "extensions/embodichain_tasks/configs"
TASK = CONFIGS / "tasks/manipulation/repeated_pick_place"


def _write(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


@pytest.fixture
def deployment(tmp_path: Path):
    template = tmp_path / "template"
    template.mkdir()
    task = yaml.safe_load((TASK / "task.ur5.yaml").read_text())
    task.pop("expansion", None)
    task["id"] = "AssetMaterializationTest-v0"
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
    env = yaml.safe_load((TASK / "envs/default.yaml").read_text())
    env["env"]["dataset"] = {}
    target = deepcopy(env["simulation"]["rigid_object"][0])
    target.update(uid="platform", body_type="static", init_pos=[0.3, 0, 0.05])
    target["shape"]["size"] = [0.3, 0.3, 0.1]
    env["simulation"]["background"].append(target)
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
    program = {
        "program_id": integration["program_id"],
        "instruction": "Place the cube on the platform.",
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
                },
            ],
        },
    }
    requirements = AssetRequirements(
        roles={
            "source": AssetRoleRequirement(
                "cube", AssetQuery(category="cube", capabilities=("antipodal_grasp",))
            ),
            "target": AssetRoleRequirement(
                "platform",
                AssetQuery(category="platform", capabilities=("support_surface",)),
            ),
        },
        relations=(AssetRelationRequirement("supported_by", "source", "target"),),
    )
    integration["asset_requirements"] = requirements.to_dict()
    for name, data in (
        ("task", task),
        ("env", env),
        ("integration", integration),
        ("program", program),
    ):
        _write(template / f"{name}.yaml", data)
    prepared = {}
    for index, (role, category, extents, capabilities) in enumerate(
        (
            ("source", "cube", (0.06, 0.06, 0.08), ("antipodal_grasp",)),
            ("target", "platform", (0.3, 0.3, 0.12), ("support_surface",)),
        )
    ):
        folder = tmp_path / role
        folder.mkdir()
        mesh = folder / "model.obj"
        trimesh.creation.box(extents=extents).export(mesh)
        lower = tuple(-size / 2 for size in extents)
        upper = tuple(size / 2 for size in extents)
        descriptor = AssetDescriptor(
            asset_id=category,
            category=category,
            capabilities=capabilities,
            bounds_m=Bounds3D(lower, upper),
            support_region=(
                Bounds3D((-0.15, -0.15, 0.06), (0.15, 0.15, 0.06))
                if role == "target"
                else None
            ),
            normalized=True,
        )
        (folder / "asset.catalog.json").write_text(
            json.dumps({"descriptor": descriptor.to_dict(), "visual_path": "model.obj"})
        )
        provider = ExplicitAssetProvider([folder])
        candidate = provider.candidates(AssetQuery(), limit=1)[0]
        prepared[role] = [
            provider.materialize(
                candidate, context=PreparationContext(str(tmp_path / "cache"))
            )
        ]
    resolved = resolve_asset_set(requirements, prepared)
    return template / "task.yaml", requirements, resolved


def test_materialized_deployment_uses_production_loader_and_fresh_geometry(
    deployment, tmp_path
):
    source, requirements, resolved = deployment
    original = source.read_bytes()
    result = materialize_asset_deployment(
        source, requirements, resolved, output_dir=tmp_path / "out"
    )
    config = yaml.safe_load(result.deployment_path.read_text())
    loaded = config_to_cfg(config, source_path=result.deployment_path)
    assert loaded is not None
    assert source.read_bytes() == original
    assert config["physics"] == "default"
    assert config["id"] != yaml.safe_load(source.read_text())["id"]
    assert config["rigid_object"][0]["init_pos"][2] == pytest.approx(0.04)
    assert config["rigid_object"][0]["shape"]["fpath"].startswith("./assets/")
    integration = yaml.safe_load(
        (result.deployment_path.parent / "integration.yaml").read_text()
    )
    target = integration["scene_binding"]["rigid_objects"][1]["affordances"][0]
    assert target["object_target_pose"][11] == pytest.approx(0.10)
    assert target["object_target_frame"] == "parent"
    assert (
        load_asset_requirements_from_deployment(result.deployment_path).to_dict()
        == requirements.to_dict()
    )
    validate_materialized_deployment(result)


def test_materialized_files_can_move_and_detect_tampering(deployment, tmp_path):
    source, requirements, resolved = deployment
    result = materialize_asset_deployment(
        source, requirements, resolved, output_dir=tmp_path / "original"
    )
    moved = tmp_path / "relocated"
    shutil.move(result.deployment_path.parent, moved)
    restored = MaterializedAssetDeployment(
        moved / "task.yaml",
        resolved.set_id,
        moved / "assets.manifest.json",
        result.manifest_sha256,
    )
    validate_materialized_deployment(restored)
    config_to_cfg(
        yaml.safe_load(restored.deployment_path.read_text()),
        source_path=restored.deployment_path,
    )
    config = _resolve_gym_components(
        yaml.safe_load(restored.deployment_path.read_text()), base_dir=moved
    ).config
    assert Path(config["rigid_object"][0]["shape"]["fpath"]).is_relative_to(moved)
    (moved / "program.yaml").write_text("changed")
    with pytest.raises(ValueError, match="Materialized file changed"):
        validate_materialized_deployment(restored)


@pytest.mark.parametrize(
    "invalid", ["fixed_grasp", "local_axis", "absolute_target", "ambiguous_support"]
)
def test_materialization_rejects_unadapted_calibration(deployment, tmp_path, invalid):
    source, requirements, resolved = deployment
    integration_path = source.parent / "integration.yaml"
    integration = yaml.safe_load(integration_path.read_text())
    if invalid == "fixed_grasp":
        integration["profile"]["action_options"]["pick"]["fixed_object_to_eef"] = [
            1.0
        ] * 16
    elif invalid == "local_axis":
        integration["scene_binding"]["rigid_objects"][0]["affordances"][0][
            "internal_axis"
        ] = [0, 0, 1]
    elif invalid == "ambiguous_support":
        integration["scene_binding"]["rigid_objects"][1]["affordances"].append(
            {
                "kind": "support_surface",
                "entity_id": "second_top",
                "native_name": "second",
            }
        )
    else:
        program_path = source.parent / "program.yaml"
        program = yaml.safe_load(program_path.read_text())
        call = program["program"]["items"][1]["steps"]["call"]
        call.pop("on")
        call["at"] = {
            "kind": "pose",
            "position": [0, 0, 0.1],
            "quaternion_xyzw": [0, 0, 0, 1],
        }
        _write(program_path, program)
    _write(integration_path, integration)
    with pytest.raises(ValueError):
        materialize_asset_deployment(
            source, requirements, resolved, output_dir=tmp_path / "out"
        )
    assert not (tmp_path / "out").exists()


def test_invalid_asset_requirements_rejected_by_production_composition(deployment):
    source, _, _ = deployment
    path = source.parent / "integration.yaml"
    integration = yaml.safe_load(path.read_text())
    integration["asset_requirements"]["unexpected"] = True
    _write(path, integration)
    with pytest.raises(ValueError, match="[Uu]nknown|unsupported"):
        config_to_cfg(yaml.safe_load(source.read_text()), source_path=source)


def test_explicit_local_mesh_paths_use_component_owner_and_preserve_data_ids(
    tmp_path, monkeypatch
):
    owner = tmp_path / "component"
    owner.mkdir()
    (owner / "mesh.obj").write_text("owner")
    shadow = tmp_path / "cwd"
    shadow.mkdir()
    (shadow / "mesh.obj").write_text("shadow")
    monkeypatch.chdir(shadow)
    _write(
        owner / "env.yaml",
        {
            "environment_id": "relative",
            "physics": "default",
            "env": {},
            "simulation": {
                "rigid_object": [
                    {
                        "uid": "local",
                        "shape": {"shape_type": "Mesh", "fpath": "./mesh.obj"},
                    },
                    {
                        "uid": "packaged",
                        "shape": {"shape_type": "Mesh", "fpath": "Dataset/mesh.obj"},
                    },
                ]
            },
        },
    )
    payload = {
        "id": "RelativeAssets-v0",
        "environment": {"component": "component/env.yaml"},
    }
    resolved = _resolve_gym_components(payload, base_dir=tmp_path).config
    assert resolved["rigid_object"][0]["shape"]["fpath"] == str(owner / "mesh.obj")
    assert resolved["rigid_object"][1]["shape"]["fpath"] == "Dataset/mesh.obj"


def test_nested_prepared_package_dependencies_are_bundled(deployment, tmp_path):
    source, requirements, resolved = deployment
    original = resolved.assets["source"]
    package = tmp_path / "nested"
    (package / "meshes").mkdir(parents=True)
    (package / "textures").mkdir()
    mesh = package / "meshes/model.obj"
    mesh.write_text(
        "mtllib ../textures/material.mtl\n" + Path(original.visual_path).read_text()
    )
    (package / "textures/material.mtl").write_text("newmtl paint\n")
    (package / "asset.catalog.json").write_text(
        json.dumps(
            {
                "descriptor": original.descriptor.to_dict(),
                "visual_path": "meshes/model.obj",
            }
        )
    )
    provider = ExplicitAssetProvider([package])
    asset = provider.materialize(
        provider.candidates(AssetQuery(), limit=1)[0],
        context=PreparationContext(str(tmp_path / "cache")),
    )
    replacement = resolve_asset_set(
        requirements,
        {"source": [asset], "target": [resolved.assets["target"]]},
    )
    result = materialize_asset_deployment(
        source, requirements, replacement, output_dir=tmp_path / "out"
    )
    root = result.deployment_path.parent / "assets" / asset.content_sha256
    assert (root / "meshes/model.obj").is_file()
    assert (root / "textures/material.mtl").is_file()
    validate_materialized_deployment(result)


def test_replacing_target_only_cannot_keep_stale_placement(deployment, tmp_path):
    source, requirements, resolved = deployment
    target_only = AssetRequirements({"target": requirements.roles["target"]})
    integration_path = source.parent / "integration.yaml"
    integration = yaml.safe_load(integration_path.read_text())
    integration["asset_requirements"] = target_only.to_dict()
    _write(integration_path, integration)
    selected = resolve_asset_set(target_only, {"target": [resolved.assets["target"]]})
    with pytest.raises(ValueError, match="source/target pair"):
        materialize_asset_deployment(
            source, target_only, selected, output_dir=tmp_path / "out"
        )


def test_placement_relation_cannot_bind_a_different_source(deployment, tmp_path):
    source, requirements, resolved = deployment
    program_path = source.parent / "program.yaml"
    program = yaml.safe_load(program_path.read_text())
    program["program"]["items"][1]["steps"]["call"]["object"] = "unmatched_source"
    _write(program_path, program)
    with pytest.raises(ValueError, match="source/target pair"):
        materialize_asset_deployment(
            source, requirements, resolved, output_dir=tmp_path / "out"
        )


def test_mutated_prepared_asset_rejected_before_configuration(deployment, tmp_path):
    source, requirements, resolved = deployment
    Path(resolved.assets["source"].visual_path).write_text("stale bytes")
    with pytest.raises(ValueError, match="files changed"):
        materialize_asset_deployment(
            source, requirements, resolved, output_dir=tmp_path / "out"
        )
    assert not (tmp_path / "out").exists()


def test_local_robot_resources_use_embodiment_owner(tmp_path, monkeypatch):
    owner = tmp_path / "robots"
    owner.mkdir()
    (owner / "robot.urdf").write_text("<robot name='local' />")
    _write(
        owner / "embodiment.yaml",
        {
            "embodiment_id": "local_robot",
            "simulation": {"uid": "robot", "urdf_cfg": {"fpath": "./robot.urdf"}},
            "sensor": [],
        },
    )
    monkeypatch.chdir(tmp_path)
    resolved = _resolve_gym_components(
        {
            "id": "LocalRobot-v0",
            "embodiment": {"component": "robots/embodiment.yaml"},
        },
        base_dir=tmp_path,
    ).config
    assert resolved["robot"]["urdf_cfg"]["fpath"] == str(owner / "robot.urdf")


def test_geometry_calibrated_expansion_requires_a_separate_template(
    deployment, tmp_path
):
    source, requirements, resolved = deployment
    raw = yaml.safe_load(source.read_text())
    raw["expansion"] = {"config": "old_geometry.yaml"}
    _write(source, raw)
    with pytest.raises(ValueError, match="expansion profiles"):
        materialize_asset_deployment(
            source, requirements, resolved, output_dir=tmp_path / "out"
        )


@pytest.mark.parametrize("target", ["platform", "platform_alias", "top_alias"])
def test_object_or_alias_placement_targets_use_adapted_affordance(
    deployment, tmp_path, target
):
    source, requirements, resolved = deployment
    integration_path = source.parent / "integration.yaml"
    integration = yaml.safe_load(integration_path.read_text())
    platform = integration["scene_binding"]["rigid_objects"][1]
    platform["aliases"] = ["platform_alias"]
    platform["affordances"][0]["aliases"] = ["top_alias"]
    _write(integration_path, integration)
    program_path = source.parent / "program.yaml"
    program = yaml.safe_load(program_path.read_text())
    program["program"]["items"][1]["steps"]["call"]["on"] = target
    _write(program_path, program)
    result = materialize_asset_deployment(
        source, requirements, resolved, output_dir=tmp_path / "out"
    )
    validate_materialized_deployment(result)


def test_object_target_only_replacement_requires_relation(deployment, tmp_path):
    source, requirements, resolved = deployment
    target_only = AssetRequirements({"target": requirements.roles["target"]})
    integration_path = source.parent / "integration.yaml"
    integration = yaml.safe_load(integration_path.read_text())
    integration["asset_requirements"] = target_only.to_dict()
    _write(integration_path, integration)
    program_path = source.parent / "program.yaml"
    program = yaml.safe_load(program_path.read_text())
    program["program"]["items"][1]["steps"]["call"]["on"] = "platform"
    _write(program_path, program)
    selected = resolve_asset_set(target_only, {"target": [resolved.assets["target"]]})
    with pytest.raises(ValueError, match="source/target pair"):
        materialize_asset_deployment(
            source, target_only, selected, output_dir=tmp_path / "out"
        )


def test_authoritative_local_pose_is_rejected(deployment, tmp_path):
    source, requirements, resolved = deployment
    env_path = source.parent / "env.yaml"
    env = yaml.safe_load(env_path.read_text())
    env["simulation"]["rigid_object"][0]["init_local_pose"] = [1.0] * 16
    _write(env_path, env)
    with pytest.raises(ValueError, match="init_local_pose"):
        materialize_asset_deployment(
            source, requirements, resolved, output_dir=tmp_path / "out"
        )


@pytest.mark.parametrize("change", ["empty_files", "bad_digest", "asset_metadata"])
def test_manifest_inventory_and_provenance_are_validated_without_optional_digest(
    deployment, tmp_path, change
):
    source, requirements, resolved = deployment
    result = materialize_asset_deployment(
        source, requirements, resolved, output_dir=tmp_path / "out"
    )
    manifest = json.loads(result.manifest_path.read_text())
    if change == "empty_files":
        manifest["files"] = {}
    elif change == "bad_digest":
        manifest["files"]["task.yaml"] = "not-a-hash"
    else:
        manifest["resolved_assets"]["assets"]["source"]["descriptor"][
            "category"
        ] = "changed"
    result.manifest_path.write_text(json.dumps(manifest))
    reconstructed = MaterializedAssetDeployment(
        result.deployment_path, result.asset_set_id, result.manifest_path
    )
    with pytest.raises(ValueError):
        validate_materialized_deployment(reconstructed)


def test_packaged_asset_id_is_not_shadowed_during_bundling(
    deployment, tmp_path, monkeypatch
):
    source, requirements, resolved = deployment
    shadow = source.parent / "Dataset"
    shadow.mkdir()
    (shadow / "mesh.obj").write_text("shadow")
    data_root = tmp_path / "dataset_cache"
    (data_root / "Dataset").mkdir(parents=True)
    shutil.copy2(resolved.assets["source"].visual_path, data_root / "Dataset/mesh.obj")
    monkeypatch.setattr(
        "embodichain.data.constants.EMBODICHAIN_DEFAULT_DATA_ROOT", str(data_root)
    )
    env_path = source.parent / "env.yaml"
    env = yaml.safe_load(env_path.read_text())
    env["simulation"]["background"].append(
        {
            "uid": "backdrop",
            "body_type": "static",
            "shape": {"shape_type": "Mesh", "fpath": "Dataset/mesh.obj"},
        }
    )
    _write(env_path, env)
    result = materialize_asset_deployment(
        source, requirements, resolved, output_dir=tmp_path / "out"
    )
    config = yaml.safe_load(result.deployment_path.read_text())
    assert config["background"][1]["shape"]["fpath"] == "Dataset/mesh.obj"
