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

"""Materialize task-bound rigid asset selections into configured deployments.

Preparation is an explicit offline operation. Normal config loading never invokes
asset providers. The supported geometry contract is metre-scale, upright +Z rigid
meshes with one annotated region per placement capability.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

import yaml

from ._asset_pouring import (
    AssetPourAdaptationCfg,
    prepare_pour_adaptation,
    finish_pour_adaptation,
)

from embodichain.utils import configclass
from embodichain.utils.config_paths import resolve_config_path
from embodichain.utils.utility import load_config

if TYPE_CHECKING:
    from embodichain.lab.task_program.semantics.asset_requirements import (
        AssetRequirements,
    )
    from embodichain.lab.task_program.semantics.asset_matching import ResolvedAssetSet

__all__ = [
    "AssetAdaptationCfg",
    "AssetPourAdaptationCfg",
    "MaterializedAssetDeployment",
    "load_asset_requirements_from_deployment",
    "materialize_asset_deployment",
    "validate_materialized_deployment",
]


@configclass
class AssetAdaptationCfg:
    """Explicit geometry adaptation for source templates without primitive bounds."""

    pour: AssetPourAdaptationCfg | None = None
    """Explicit selected-asset calibration for the four-segment PourWater task."""

    initial_bottom_z_m: dict[str, float] = {}
    """World height of each role's initial bottom; upright poses only."""


@dataclass(frozen=True)
class MaterializedAssetDeployment:
    """A preflighted deployment and its immutable asset manifest."""

    deployment_path: Path
    asset_set_id: str
    manifest_path: Path
    manifest_sha256: str = ""


def _read(path: Path) -> dict:
    value = load_config(path)
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must contain a mapping")
    return deepcopy(dict(value))


def _path(value: str, owner: Path) -> Path:
    result = resolve_config_path(value, base_dir=owner.parent).resolve()
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def load_asset_requirements_from_deployment(
    task_config_path: str | Path,
) -> AssetRequirements:
    """Read the strict optional role contract from the selected integration."""
    from embodichain.lab.task_program.semantics.asset_requirements import (
        AssetRequirements,
    )

    source = Path(task_config_path).resolve()
    task = _read(source)
    integration = _read(_path(task["task_program"]["integration"], source))
    if "asset_requirements" not in integration:
        raise ValueError("Task integration does not declare asset_requirements")
    return AssetRequirements.from_dict(integration["asset_requirements"])


def _walk(value: object):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resource_digest(inventory: Mapping[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(dict(inventory), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _resource_files(root: Path, model: Path) -> set[Path]:
    from embodichain.data.asset_catalog.local import _check_dependencies, _dependency

    files: set[Path] = set()
    pending = [model.resolve()]
    while pending:
        path = pending.pop()
        if path in files:
            continue
        files.update(_check_dependencies(root, path))
        if path.suffix.lower() not in {".urdf", ".xml"}:
            continue
        document = ET.parse(path)
        compiler = document.find("compiler")
        directories = {} if compiler is None else compiler.attrib
        for element in document.iter():
            locator = element.get("filename", element.get("file"))
            if locator is None:
                continue
            directory = ""
            if "file" in element.attrib and element.tag in {
                "mesh",
                "texture",
                "hfield",
                "skin",
            }:
                key = "meshdir" if element.tag in {"mesh", "skin"} else "texturedir"
                directory = directories.get(key, directories.get("assetdir", ""))
            dependency = _dependency(root, path.parent / directory, locator)
            pending.append(dependency)
    return files


def _dump(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _preflight(path: Path) -> None:
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg

    config_to_cfg(_read(path), source_path=path)


def validate_materialized_deployment(deployment: MaterializedAssetDeployment) -> None:
    """Verify file digests and asset identity before environment construction."""
    from embodichain.data.asset_catalog import PreparedAsset, verify_prepared_asset
    from embodichain.data.asset_catalog.local import _inventory
    from embodichain.lab.task_program.semantics.asset_matching import ResolvedAssetSet

    if (
        deployment.manifest_sha256
        and _hash(deployment.manifest_path) != deployment.manifest_sha256
    ):
        raise ValueError("Materialized manifest changed")
    manifest = json.loads(deployment.manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    required_files = {
        "task.yaml",
        "program.yaml",
        "integration.yaml",
        "embodiment.yaml",
        "policy.yaml",
    }
    if not isinstance(files, dict) or not required_files.issubset(files):
        raise ValueError("Materialized manifest is missing required files")
    if any(
        not isinstance(key, str)
        or not isinstance(value, str)
        or re.fullmatch("[0-9a-f]{64}", value) is None
        for key, value in files.items()
    ):
        raise ValueError("Materialized manifest contains malformed file digests")
    snapshot = manifest["resolved_assets"]
    reconstructed = ResolvedAssetSet(
        assets={
            role: PreparedAsset.from_dict(asset)
            for role, asset in snapshot["assets"].items()
        },
        seed=snapshot["seed"],
        requirements_sha256=snapshot["requirements_sha256"],
    )
    if (
        reconstructed.set_id != snapshot["set_id"]
        or reconstructed.set_id != deployment.asset_set_id
    ):
        raise ValueError("Materialized resolved asset metadata changed")
    if manifest["asset_set_id"] != deployment.asset_set_id:
        raise ValueError("Materialized asset set identity mismatch")
    root = deployment.manifest_path.parent.resolve()
    if deployment.deployment_path.resolve() != root / "task.yaml":
        raise ValueError("Materialized deployment path mismatch")
    for relative, expected in manifest["files"].items():
        path = (root / relative).resolve()
        if (
            not path.is_relative_to(root)
            or not path.is_file()
            or _hash(path) != expected
        ):
            raise ValueError(f"Materialized file changed: {relative}")
    asset_roots = []
    for asset in reconstructed.assets.values():
        package = root / "assets" / asset.content_sha256
        provenance = dict(asset.provenance)
        original_root = Path(provenance["package_root"])
        visual = package / Path(asset.visual_path).relative_to(original_root)
        collision = package / Path(asset.collision_path).relative_to(original_root)
        provenance["package_root"] = str(package)
        copied = PreparedAsset(
            asset.descriptor,
            str(visual),
            str(collision),
            asset.content_sha256,
            provenance,
        )
        verify_prepared_asset(copied)
        for relative, expected in provenance["files"].items():
            bundled = (package / relative).relative_to(root).as_posix()
            if files.get(bundled) != expected:
                raise ValueError(
                    f"Materialized manifest is missing asset inventory: {bundled}"
                )
        asset_roots.append(package)
    checked_resources = set()
    for filename in required_files:
        document = _read(root / filename)
        if filename == "task.yaml" and "objective" in document:
            required = document["objective"]["component"]
            if required not in files:
                raise ValueError("Materialized manifest is missing the objective")
        for node in _walk(document):
            for key in ("fpath", "urdf_path", "xml_path"):
                value = node.get(key)
                if not isinstance(value, str) or not value.startswith(("./", "../")):
                    continue
                path = (root / value).resolve()
                if (
                    not path.is_relative_to(root)
                    or path.relative_to(root).as_posix() not in files
                ):
                    raise ValueError(
                        f"Materialized manifest is missing local resource: {value}"
                    )
                if any(path.is_relative_to(package) for package in asset_roots):
                    continue
                relative = path.relative_to(root)
                if len(relative.parts) < 3 or relative.parts[0] != "resources":
                    raise ValueError(
                        f"Local resource is outside a bundled package: {value}"
                    )
                package = root / "resources" / relative.parts[1]
                if package in checked_resources:
                    continue
                inventory = _inventory(package)
                if _resource_digest(inventory) != package.name or any(
                    files.get((package / name).relative_to(root).as_posix()) != digest
                    for name, digest in inventory.items()
                ):
                    raise ValueError(
                        f"Materialized resource inventory changed: {value}"
                    )
                checked_resources.add(package)


def _bundle_local_paths(payload: dict, owner: Path, output: Path) -> None:
    # Packaged data IDs stay package-resolved. Explicit local resources are copied
    # with sibling sidecars so OBJ materials/textures retain their relative paths.
    for node in _walk(payload):
        for key in ("fpath", "urdf_path", "xml_path"):
            value = node.get(key)
            if not isinstance(value, str):
                continue
            path = Path(value).expanduser()
            if not path.is_absolute():
                if not value.startswith(("./", "../")):
                    continue
                path = owner.parent / path
            if not path.is_file():
                continue
            if path.resolve().is_relative_to(output):
                node[key] = "./" + str(path.resolve().relative_to(output))
                continue
            package_root = path.parent.resolve()
            if output.is_relative_to(package_root):
                raise ValueError("Output must not be nested inside a resource package")
            inventory = {
                item.relative_to(package_root).as_posix(): _hash(item)
                for item in _resource_files(package_root, path)
            }
            destination = output / "resources" / _resource_digest(inventory)
            for relative, expected in inventory.items():
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(package_root / relative, target)
                if _hash(target) != expected:
                    raise ValueError(
                        f"Local resource changed during bundling: {relative}"
                    )
            node[key] = "./" + str((destination / path.name).relative_to(output))


def materialize_asset_deployment(
    task_config_path: str | Path,
    requirements: AssetRequirements,
    resolved_assets: ResolvedAssetSet,
    *,
    output_dir: str | Path,
    adaptations: AssetAdaptationCfg | None = None,
) -> MaterializedAssetDeployment:
    """Build and preflight a new deployment without modifying its source files.

    Place calls must use a semantic support/container covered by a role relation.
    Calibrated grasps, rotated initial poses and absolute Place targets fail closed.
    Prepared assets must contain the verifiable package inventory contract.
    Resources are bundled locally; packaged robot/data identifiers retain normal
    SDK resolution. Output directories must be absent or empty.
    """
    from embodichain.data.asset_catalog import verify_prepared_asset
    from embodichain.lab.gym.utils._component_composition import _resolve_gym_components
    from embodichain.lab.task_program.semantics.asset_matching import resolve_asset_set

    source = Path(task_config_path).resolve()
    raw = _read(source)
    if "expansion" in raw:
        raise ValueError(
            "Asset materialization requires a template without geometry-calibrated expansion profiles"
        )
    physical = _resolve_gym_components(raw, base_dir=source.parent)
    if physical.embodiment_skill_profile is None:
        raise ValueError("Asset materialization requires a configured embodiment")
    if set(requirements.roles) != set(resolved_assets.assets):
        raise ValueError("Resolved assets must cover exactly the required roles")
    canonical = json.dumps(
        requirements.to_dict(), sort_keys=True, separators=(",", ":")
    )
    if (
        hashlib.sha256(canonical.encode()).hexdigest()
        != resolved_assets.requirements_sha256
    ):
        raise ValueError("Resolved assets were selected for different requirements")
    resolve_asset_set(
        requirements,
        {role: [asset] for role, asset in resolved_assets.assets.items()},
        seed=resolved_assets.seed,
    )
    adaptations = adaptations or AssetAdaptationCfg()
    adaptations.validate()
    unknown = set(adaptations.initial_bottom_z_m) - set(requirements.roles)
    if unknown:
        raise ValueError(f"Unknown initial-bottom roles: {sorted(unknown)}")
    task_program = raw["task_program"]
    integration_path = _path(task_program["integration"], source)
    program_path = _path(task_program["program"], source)
    policy_path = _path(task_program["execution_policy"], source)
    embodiment_path = _path(raw["embodiment"]["component"], source)
    integration = _read(integration_path)
    if (
        "asset_requirements" in integration
        and type(requirements).from_dict(integration["asset_requirements"]).to_dict()
        != requirements.to_dict()
    ):
        raise ValueError(
            "Supplied requirements differ from the task integration contract"
        )
    program = _read(program_path)
    policy = _read(policy_path)
    embodiment = _read(embodiment_path)
    prepared_pour = None
    if adaptations.pour is not None:
        if not isinstance(adaptations.pour, AssetPourAdaptationCfg):
            raise TypeError("pour adaptation must be AssetPourAdaptationCfg")
        prepared_pour = prepare_pour_adaptation(
            adaptations.pour,
            requirements,
            resolved_assets,
            physical.embodiment_skill_profile,
            integration,
            program,
        )
    for document in (integration, program, embodiment):
        for node in _walk(document):
            if "fixed_object_to_eef" in node or "internal_axis" in node:
                raise ValueError(
                    "Asset replacement cannot inherit fixed grasp transforms or internal_axis"
                )
    scene = integration["scene_binding"]
    entities = {
        entity["entity_id"]: entity for entity in scene.get("rigid_objects", [])
    }
    config = physical.config
    for group in ("rigid_object", "background"):
        for obj in config.get(group, []):
            default_body_type = "dynamic" if group == "rigid_object" else "static"
            dynamic = obj.get("body_type", default_body_type) == "dynamic"
            if dynamic != (group == "rigid_object"):
                raise ValueError(
                    "Dynamic objects belong in rigid_object; static/kinematic objects belong in background"
                )
    objects = {
        obj["uid"]: obj
        for group in ("rigid_object", "background")
        for obj in config.get(group, [])
    }
    replaced_ids = {role.entity_id for role in requirements.roles.values()}
    for node in _walk(program):
        if (
            node.get("kind") == "place"
            and node.get("object") in replaced_ids
            and "at" in node
        ):
            raise ValueError(
                "Asset replacement requires semantic Place targets; absolute at targets need an adaptation rule"
            )
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Materialization output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    try:
        for name, role in requirements.roles.items():
            asset = resolved_assets.assets[name]
            verify_prepared_asset(asset)
            descriptor = asset.descriptor
            if not descriptor.normalized or descriptor.bounds_m is None:
                raise ValueError(
                    f"Role {name} requires normalized metre/+Z asset bounds"
                )
            if (
                Path(asset.visual_path).resolve()
                != Path(asset.collision_path).resolve()
            ):
                raise ValueError(
                    "Rigid MeshCfg cannot consume a distinct collision_path"
                )
            entity = entities.get(role.entity_id)
            if entity is None or entity["simulation_uid"] not in objects:
                raise ValueError(
                    f"Role {name} must map to a declared physical rigid object"
                )
            obj = objects[entity["simulation_uid"]]
            if obj.get("init_local_pose") is not None:
                raise ValueError(f"Role {name} cannot inherit init_local_pose")
            if any(float(v) != 0.0 for v in obj.get("init_rot", [0, 0, 0])):
                raise ValueError(
                    f"Role {name} requires an upright unrotated initial pose"
                )
            if any(float(v) != 1.0 for v in obj.get("body_scale", [1, 1, 1])):
                raise ValueError(f"Role {name} cannot inherit body_scale")
            position = list(obj.get("init_pos", [0, 0, 0]))
            bottom = adaptations.initial_bottom_z_m.get(name)
            if bottom is None:
                shape = obj.get("shape", {})
                if shape.get("shape_type") != "Cube":
                    raise ValueError(
                        f"Role {name} requires explicit initial_bottom_z_m"
                    )
                bottom = position[2] - shape["size"][2] / 2
            if not math.isfinite(bottom):
                raise ValueError(f"Role {name} initial bottom must be finite")
            position[2] = bottom - descriptor.bounds_m.lower[2]
            obj["init_pos"] = position
            visual = Path(asset.visual_path).resolve()
            package_root = Path(
                asset.provenance.get("package_root", str(visual.parent))
            ).resolve()
            if not visual.is_relative_to(package_root):
                raise ValueError(
                    "Visual mesh must be contained by its prepared package_root"
                )
            destination = output / "assets" / asset.content_sha256
            if output.is_relative_to(package_root):
                raise ValueError("Output must not be nested inside an asset package")
            collision = obj.get("shape", {}).get("collision")
            if "container" in descriptor.capabilities and (
                not collision
                or collision.get("approximation", "convex_hull") == "convex_hull"
            ):
                raise ValueError(
                    "Container replacement requires an explicit concavity-preserving collision policy"
                )
            shutil.copytree(package_root, destination, dirs_exist_ok=True)
            copied_path = destination / visual.relative_to(package_root)
            copied_provenance = dict(asset.provenance)
            copied_provenance["package_root"] = str(destination)
            verify_prepared_asset(
                type(asset)(
                    asset.descriptor,
                    str(copied_path),
                    str(copied_path),
                    asset.content_sha256,
                    copied_provenance,
                )
            )
            obj["shape"] = {
                "shape_type": "Mesh",
                # Retain the output owner until bundling makes paths relative.
                "fpath": str(copied_path),
            }
            if collision is not None:
                obj["shape"]["collision"] = deepcopy(collision)
            obj["asset_physics_mode"] = "overlay"
            entity["semantic_type"] = descriptor.category
            for affordance in entity.get("affordances", []):
                kind = affordance["kind"]
                if kind not in descriptor.capabilities:
                    raise ValueError(f"Role {name} lacks scene capability {kind}")
                if kind == "antipodal_grasp":
                    if "relative_pose" in affordance:
                        raise ValueError(
                            "Asset replacement cannot inherit a calibrated grasp relative_pose"
                        )
                    affordance["revision"] = asset.content_sha256
        adapted_affordances = {}
        adapted_pairs = set()
        relative_offsets = {}
        if prepared_pour is not None:
            pour_pairs, pour_offsets = finish_pour_adaptation(
                prepared_pour, integration, entities, objects
            )
            adapted_pairs.update(pour_pairs)
            relative_offsets.update(pour_offsets)
        for relation in requirements.relations:
            if relation.kind == "pours_into":
                if prepared_pour is None:
                    raise ValueError(
                        "pours_into requires an explicit PourWater adaptation"
                    )
                continue
            source_asset = resolved_assets.assets[relation.object_role].descriptor
            target_asset = resolved_assets.assets[relation.target_role].descriptor
            kind = "container" if relation.kind == "fits_inside" else "support_surface"
            region = (
                target_asset.container_region
                if kind == "container"
                else target_asset.support_region
            )
            target_id = requirements.roles[relation.target_role].entity_id
            matches = [
                item
                for item in entities[target_id].get("affordances", [])
                if item["kind"] == kind
            ]
            if region is None or len(matches) != 1:
                raise ValueError(
                    f"Role {relation.target_role} needs exactly one annotated {kind} affordance"
                )
            affordance = matches[0]
            x = (
                region.lower[0]
                + region.upper[0]
                - source_asset.bounds_m.lower[0]
                - source_asset.bounds_m.upper[0]
            ) / 2
            y = (
                region.lower[1]
                + region.upper[1]
                - source_asset.bounds_m.lower[1]
                - source_asset.bounds_m.upper[1]
            ) / 2
            floor = region.lower[2] if kind == "container" else region.upper[2]
            z = floor - source_asset.bounds_m.lower[2] + relation.clearance_m
            pose = [
                1.0,
                0.0,
                0.0,
                x,
                0.0,
                1.0,
                0.0,
                y,
                0.0,
                0.0,
                1.0,
                z,
                0.0,
                0.0,
                0.0,
                1.0,
            ]
            if (
                affordance["entity_id"] in adapted_affordances
                and adapted_affordances[affordance["entity_id"]] != pose
            ):
                raise ValueError(
                    "One placement affordance cannot encode different source-object sizes"
                )
            affordance["object_target_frame"] = "parent"
            affordance["object_target_pose"] = pose
            adapted_affordances[affordance["entity_id"]] = pose
            adapted_pairs.add(
                (
                    requirements.roles[relation.object_role].entity_id,
                    affordance["entity_id"],
                )
            )
            relative_offsets[
                (requirements.roles[relation.object_role].entity_id, target_id)
            ] = [x, y, z]
        references = {}
        placement_owners = {}
        for entity_id, entity in entities.items():
            for reference in (entity_id, *entity.get("aliases", [])):
                if reference in references and references[reference] != entity_id:
                    raise ValueError(f"Ambiguous scene reference {reference}")
                references[reference] = entity_id
            for affordance in entity.get("affordances", []):
                affordance_id = affordance["entity_id"]
                for reference in (affordance_id, *affordance.get("aliases", [])):
                    if (
                        reference in references
                        and references[reference] != affordance_id
                    ):
                        raise ValueError(f"Ambiguous scene reference {reference}")
                    references[reference] = affordance_id
                if affordance["kind"] in {"support_surface", "container"}:
                    placement_owners[affordance_id] = entity_id
        for node in _walk(program):
            source_id = references.get(node.get("object"), node.get("object"))
            reference_id = references.get(node.get("reference"), node.get("reference"))
            if node.get("kind") == "place":
                raw_target = node.get("on", node.get("inside"))
                target = references.get(raw_target, raw_target)
                target_owner = (
                    target if target in entities else placement_owners.get(target)
                )
                if source_id in replaced_ids or target_owner in replaced_ids:
                    if target in entities:
                        kind = "support_surface" if "on" in node else "container"
                        matches = [
                            a
                            for a in entities[target].get("affordances", [])
                            if a["kind"] == kind
                        ]
                        if len(matches) != 1:
                            raise ValueError(
                                "Replacement Place target needs exactly one matching affordance"
                            )
                        target = matches[0]["entity_id"]
                    if (source_id, target) not in adapted_pairs:
                        raise ValueError(
                            "Each replacement Place source/target pair requires an annotated role relation"
                        )
            if node.get("kind") == "object_relative_position" and (
                source_id in replaced_ids or reference_id in replaced_ids
            ):
                key = (source_id, reference_id)
                if key not in relative_offsets:
                    raise ValueError(
                        "Relative-position validator requires a supported placement relation"
                    )
                node["position_offset"] = relative_offsets[key]
        # Flatten only physical ownership, then restore the original embodiment
        # selector so skill_profile remains available to production composition.
        config.pop("robot", None)
        config.pop("sensor", None)
        config["embodiment"] = deepcopy(raw["embodiment"])
        config["embodiment"]["component"] = "embodiment.yaml"
        config["task_program"] = {
            "program": "program.yaml",
            "integration": "integration.yaml",
            "execution_policy": "policy.yaml",
        }
        integration["asset_requirements"] = requirements.to_dict()
        for document, owner in (
            (config, source),
            (embodiment, embodiment_path),
            (policy, policy_path),
        ):
            _bundle_local_paths(document, owner, output)
        objective = None
        if "objective" in config:
            objective_path = _path(config["objective"]["component"], source)
            objective = _read(objective_path)
            shutil.copy2(objective_path, output / "objective.yaml")
            config["objective"]["component"] = "objective.yaml"
        # Include all source configuration and adaptation content in identity;
        # the same assets used by a changed task cannot reuse registry IDs.
        identity = hashlib.sha256(
            json.dumps(
                {
                    "task": config,
                    "integration": integration,
                    "program": program,
                    "policy": policy,
                    "embodiment": embodiment,
                    "assets": resolved_assets.set_id,
                    "objective": objective,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()[:16]
        logical_id = raw["id"]
        match = re.match(r"^(.*?)(-v[0-9]+)$", logical_id)
        config["id"] = (
            f"{match[1]}-Assets-{identity}{match[2]}"
            if match
            else f"{logical_id}-Assets-{identity}-v0"
        )
        integration["integration_id"] += f"_assets_{identity}"
        scene["registry_id"] += f"_assets_{identity}"
        for name, document in (
            ("task", config),
            ("integration", integration),
            ("program", program),
            ("embodiment", embodiment),
            ("policy", policy),
        ):
            _dump(output / f"{name}.yaml", document)
        _preflight(output / "task.yaml")
        files = {
            str(p.relative_to(output)): _hash(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        }
        manifest = {
            "asset_set_id": resolved_assets.set_id,
            "logical_task_id": logical_id,
            "deployment_id": config["id"],
            "resolved_assets": resolved_assets.to_dict(),
            "files": files,
        }
        manifest_path = output / "assets.manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
        )
        return MaterializedAssetDeployment(
            output / "task.yaml",
            resolved_assets.set_id,
            manifest_path,
            _hash(manifest_path),
        )
    except BaseException:
        shutil.rmtree(output)
        raise
