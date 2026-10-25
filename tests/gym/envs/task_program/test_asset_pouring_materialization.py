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

"""Production-loader tests for selected-asset pouring calibration."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import trimesh
import yaml

from embodichain.data.asset_catalog import (
    AssetDescriptor,
    AssetQuery,
    Bounds3D,
    ExplicitAssetProvider,
    PreparationContext,
)
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.task_program.integrations.asset_materialization import (
    AssetAdaptationCfg,
    AssetPourAdaptationCfg,
    materialize_asset_deployment,
)
from embodichain.lab.task_program.semantics.asset_requirements import (
    AssetRequirements,
    AssetRoleRequirement,
    AssetRelationRequirement,
)
from embodichain.lab.task_program.semantics.asset_matching import resolve_asset_set

ROOT = Path(__file__).resolve().parents[4]
CONFIGS = ROOT / "extensions/embodichain_tasks/configs"
POUR = CONFIGS / "tasks/manipulation/tableware/pour_water"
PROFILE_ID = "ur5_dh_pgi_140_80"


def _dump(path: Path, document: dict) -> None:
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


@pytest.fixture
def pouring(tmp_path):
    source = tmp_path / "template"
    source.mkdir()
    integration = yaml.safe_load((POUR / "task_program/integration.yaml").read_text())
    program = yaml.safe_load((POUR / "task_program/program.yaml").read_text())
    requirements = AssetRequirements(
        roles={
            "bottle": AssetRoleRequirement(
                "bottle",
                AssetQuery(
                    category="bottle", capabilities=("antipodal_grasp", "pour_source")
                ),
            ),
            "cup": AssetRoleRequirement(
                "cup", AssetQuery(category="cup", capabilities=("container",))
            ),
        },
        relations=(AssetRelationRequirement("pours_into", "bottle", "cup", 0.002),),
    )
    integration["asset_requirements"] = requirements.to_dict()
    body = {
        "uid": "bottle",
        "body_type": "dynamic",
        "init_pos": [0.75, -0.1, 0.94],
        "shape": {"shape_type": "Cube", "size": [0.06, 0.06, 0.14]},
    }
    cup = deepcopy(body)
    cup.update(uid="cup", init_pos=[0.75, 0.1, 0.92])
    cup["shape"] = {
        "shape_type": "Cube",
        "size": [0.09, 0.09, 0.1],
        "collision": {"approximation": "convex_decomposition", "max_hulls": 16},
    }
    table = {
        "uid": "table",
        "body_type": "static",
        "init_pos": [0.75, 0, 0.845],
        "shape": {"shape_type": "Cube", "size": [0.6, 0.6, 0.05]},
    }
    env = {
        "environment_id": "asset_pour_test",
        "physics": "default",
        "max_episode_steps": 2000,
        "simulation": {"rigid_object": [body, cup], "background": [table]},
        "env": {"sim_steps_per_control": 4, "events": {}, "dataset": {}},
    }
    task = {
        "id": "AssetPourTest-v0",
        "environment": {"component": "env.yaml"},
        "embodiment": {
            "component": str(CONFIGS / "components/embodiments/ur5_dh_pgi_140_80.yaml")
        },
        "task_program": {
            "program": "program.yaml",
            "integration": "integration.yaml",
            "execution_policy": str(
                CONFIGS / "components/execution_policies/trajectory_open_loop.yaml"
            ),
        },
    }
    for name, document in (
        ("task", task),
        ("env", env),
        ("integration", integration),
        ("program", program),
    ):
        _dump(source / f"{name}.yaml", document)
    return source / "task.yaml", requirements


def _prepared(tmp_path, *, bottle_height=0.14, cup_height=0.1, change=None):
    result = {}
    for role in ("bottle", "cup"):
        folder = tmp_path / role
        folder.mkdir(parents=True)
        height = bottle_height if role == "bottle" else cup_height
        radius = 0.03 if role == "bottle" else 0.045
        mesh = (
            trimesh.creation.cylinder(radius=radius, height=height, sections=32)
            if role == "bottle"
            else trimesh.creation.annulus(
                r_min=0.03, r_max=radius, height=height, sections=32
            )
        )
        mesh.export(folder / "model.obj")
        calibration = {
            "mouth_center_m": [0.0, 0.0, height / 2],
            "evidence": "synthetic test asset calibration",
        }
        if role == "bottle":
            calibration.update(
                mouth_radius_m=0.015,
                tilt_axis=[1.0, 0.0, 0.0],
                object_to_eef=np.eye(4).reshape(-1).tolist(),
                profile_id=PROFILE_ID,
            )
        else:
            calibration["opening_radius_m"] = 0.03
        if change is not None:
            change(role, calibration)
        descriptor = AssetDescriptor(
            asset_id=role,
            category=role,
            capabilities=(
                ("antipodal_grasp", "pour_source")
                if role == "bottle"
                else ("container",)
            ),
            bounds_m=Bounds3D(
                (-radius, -radius, -height / 2), (radius, radius, height / 2)
            ),
            container_region=(
                Bounds3D((-0.03, -0.03, -height / 2 + 0.01), (0.03, 0.03, height / 2))
                if role == "cup"
                else None
            ),
            normalized=True,
            provenance={"pouring": calibration},
        )
        (folder / "asset.catalog.json").write_text(
            json.dumps({"descriptor": descriptor.to_dict(), "visual_path": "model.obj"})
        )
        provider = ExplicitAssetProvider([folder])
        result[role] = [
            provider.materialize(
                provider.candidates(AssetQuery(), limit=1)[0],
                context=PreparationContext(str(tmp_path / "cache")),
            )
        ]
    return result


@pytest.mark.parametrize(
    "bottle_height,cup_height,options_key",
    [
        (0.14, 0.1, "skill_options"),
        (0.18, 0.12, "skill_options"),
        (0.14, 0.1, "action_options"),
    ],
)
def test_pouring_rebinds_selected_bottle_and_cup_geometry(
    pouring, tmp_path, bottle_height, cup_height, options_key
):
    source, requirements = pouring
    integration_path = source.parent / "integration.yaml"
    if options_key == "action_options":
        integration = yaml.safe_load(integration_path.read_text())
        integration["profile"][options_key] = integration["profile"].pop(
            "skill_options"
        )
        _dump(integration_path, integration)
    source_integration = integration_path.read_bytes()
    selected = resolve_asset_set(
        requirements,
        _prepared(
            tmp_path / "assets", bottle_height=bottle_height, cup_height=cup_height
        ),
    )
    result = materialize_asset_deployment(
        source,
        requirements,
        selected,
        output_dir=tmp_path / "out",
        adaptations=AssetAdaptationCfg(pour=AssetPourAdaptationCfg()),
    )
    config = yaml.safe_load(result.deployment_path.read_text())
    config_to_cfg(config, source_path=result.deployment_path)
    integration = yaml.safe_load(
        (result.deployment_path.parent / "integration.yaml").read_text()
    )
    program = yaml.safe_load(
        (result.deployment_path.parent / "program.yaml").read_text()
    )
    assert integration_path.read_bytes() == source_integration
    assert "action_options" not in integration["profile"]
    options = integration["profile"]["skill_options"]
    assert options["pick"]["fixed_object_to_eef"] == np.eye(4).reshape(-1).tolist()
    bottle = next(
        node
        for node in integration["scene_binding"]["rigid_objects"]
        if node["entity_id"] == "bottle"
    )
    assert bottle["affordances"][0]["internal_axis"] == [1.0, 0.0, 0.0]
    move = integration["runtime_services"]["registered_semantic_lowerers"][0]
    pose = np.array(move["relative_pose"]).reshape(4, 4)
    mouth = np.array([0.0, 0.0, bottle_height / 2])
    angle = options["simulation.pour"]["rotate_angle"]
    rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(angle), -math.sin(angle)],
            [0.0, math.sin(angle), math.cos(angle)],
        ]
    )
    peak = pose[:3, 3] + rotation @ mouth
    assert peak[:2] == pytest.approx([0.0, 0.0])
    assert peak[2] == pytest.approx(cup_height / 2 + 0.17 + 0.015)
    return_call = program["program"]["items"][3]["steps"]["call"]
    assert return_call["on"] == "bottle_asset_return_support"
    assert "at" not in return_call
    validators = program["program"]["items"][3]["validators"]
    assert all(item["kind"] == "object_relative_position" for item in validators)
    assert all(item["reference_frame"] == "parent" for item in validators)
    assert (
        program["program"]["items"][1]["validators"][0]["reference_frame"] == "parent"
    )
    assert all(item["reference"] == "bottle_asset_return_table" for item in validators)
    assert validators[0]["position_tolerance"] == pytest.approx(
        [0.05 / math.sqrt(3)] * 3
    )
    assert config["rigid_object"][0]["init_pos"][2] == pytest.approx(
        0.87 + bottle_height / 2
    )
    assert config["rigid_object"][1]["init_pos"][2] == pytest.approx(
        0.87 + cup_height / 2
    )


@pytest.mark.parametrize(
    "invalid", ["profile", "axis", "transform", "mouth", "missing_evidence"]
)
def test_pouring_rejects_incompatible_selected_calibration(pouring, tmp_path, invalid):
    source, requirements = pouring

    def corrupt(role, record):
        if role != "bottle":
            return
        if invalid == "profile":
            record["profile_id"] = "other_robot_tcp"
        elif invalid == "axis":
            record["tilt_axis"] = [0.0, 0.0, 1.0]
        elif invalid == "transform":
            record["object_to_eef"] = [0.0] * 16
        elif invalid == "mouth":
            record["mouth_center_m"] = [0.0, 0.0, 100.0]
        else:
            record.pop("evidence")

    with pytest.raises(ValueError):
        selected = resolve_asset_set(
            requirements, _prepared(tmp_path / "assets", change=corrupt)
        )
        materialize_asset_deployment(
            source,
            requirements,
            selected,
            output_dir=tmp_path / "out",
            adaptations=AssetAdaptationCfg(pour=AssetPourAdaptationCfg()),
        )
    assert not (tmp_path / "out").exists()


def test_pouring_does_not_bypass_calibration_guards_without_explicit_adapter(
    pouring, tmp_path
):
    source, requirements = pouring
    selected = resolve_asset_set(requirements, _prepared(tmp_path / "assets"))
    with pytest.raises(ValueError, match="fixed grasp transforms|internal_axis"):
        materialize_asset_deployment(
            source, requirements, selected, output_dir=tmp_path / "out"
        )


def test_same_assets_with_distinct_pour_or_initial_geometry_get_distinct_ids(
    pouring, tmp_path
):
    source, requirements = pouring
    selected = resolve_asset_set(requirements, _prepared(tmp_path / "assets"))
    adaptations = [
        AssetAdaptationCfg(pour=AssetPourAdaptationCfg(mouth_clearance_m=0.15)),
        AssetAdaptationCfg(pour=AssetPourAdaptationCfg(mouth_clearance_m=0.17)),
        AssetAdaptationCfg(
            pour=AssetPourAdaptationCfg(mouth_clearance_m=0.15),
            initial_bottom_z_m={"bottle": 0.88, "cup": 0.88},
        ),
    ]
    identities = []
    for index, adaptation in enumerate(adaptations):
        artifact = materialize_asset_deployment(
            source,
            requirements,
            selected,
            output_dir=tmp_path / f"variant_{index}",
            adaptations=adaptation,
        )
        task = yaml.safe_load(artifact.deployment_path.read_text())
        config_to_cfg(task, source_path=artifact.deployment_path)
        integration = yaml.safe_load(
            (artifact.deployment_path.parent / "integration.yaml").read_text()
        )
        manifest = json.loads(artifact.manifest_path.read_text())
        assert artifact.asset_set_id == selected.set_id
        assert manifest["logical_task_id"] == "AssetPourTest-v0"
        identities.append(
            (
                task["id"],
                integration["integration_id"],
                integration["scene_binding"]["registry_id"],
            )
        )
    for index in range(3):
        assert len({identity[index] for identity in identities}) == 3
    repeated = materialize_asset_deployment(
        source,
        requirements,
        selected,
        output_dir=tmp_path / "same_configuration_other_directory",
        adaptations=adaptations[0],
    )
    assert (
        yaml.safe_load(repeated.deployment_path.read_text())["id"] == identities[0][0]
    )
    environment_path = source.parent / "env.yaml"
    environment = yaml.safe_load(environment_path.read_text())
    environment["simulation"]["rigid_object"][0]["init_pos"][2] += 0.01
    _dump(environment_path, environment)
    changed_source = materialize_asset_deployment(
        source,
        requirements,
        selected,
        output_dir=tmp_path / "changed_source_initial_pose",
        adaptations=adaptations[0],
    )
    assert yaml.safe_load(changed_source.deployment_path.read_text())["id"] not in {
        identity[0] for identity in identities
    }


def test_pouring_preserves_schema_default_position_tolerance(pouring, tmp_path):
    from embodichain.lab.task_program.language.schema import (
        ObjectNearTargetValidatorCfg,
    )

    source, requirements = pouring
    program_path = source.parent / "program.yaml"
    program = yaml.safe_load(program_path.read_text())
    for segment in program["program"]["items"]:
        for validator in segment.get("validators", []):
            if validator["kind"] == "object_near_target":
                validator.pop("position_tolerance", None)
    _dump(program_path, program)
    resolved = resolve_asset_set(requirements, _prepared(tmp_path / "assets"))
    result = materialize_asset_deployment(
        source,
        requirements,
        resolved,
        output_dir=tmp_path / "out",
        adaptations=AssetAdaptationCfg(pour=AssetPourAdaptationCfg()),
    )
    adapted = yaml.safe_load(
        (result.deployment_path.parent / "program.yaml").read_text()
    )
    validators = [
        item
        for segment in adapted["program"]["items"]
        for item in segment.get("validators", [])
        if item["kind"] == "object_relative_position"
    ]
    assert validators
    for validator in validators:
        assert validator["position_tolerance"] == pytest.approx(
            [
                ObjectNearTargetValidatorCfg(
                    object="bottle", target="return"
                ).position_tolerance
                / math.sqrt(3)
            ]
            * 3
        )


def test_pouring_rejects_mixed_skill_option_keys(pouring, tmp_path):
    source, requirements = pouring
    integration_path = source.parent / "integration.yaml"
    integration = yaml.safe_load(integration_path.read_text())
    integration["profile"]["action_options"] = deepcopy(
        integration["profile"]["skill_options"]
    )
    _dump(integration_path, integration)
    selected = resolve_asset_set(requirements, _prepared(tmp_path / "assets"))
    with pytest.raises(ValueError, match="cannot specify both"):
        materialize_asset_deployment(
            source,
            requirements,
            selected,
            output_dir=tmp_path / "out",
            adaptations=AssetAdaptationCfg(pour=AssetPourAdaptationCfg()),
        )
