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

"""Admission and static inventories for configured whole-scene imports."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from embodichain.lab.sim.cfg import RobotCfg, UsdImportCfg

__all__: list[str] = []


def _file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        digest = sha256()
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def _open_usd_scene_source(
    recipe: UsdImportCfg,
) -> Iterator[UsdImportCfg]:
    """Keep admitted package dependencies alive for the caller's import."""
    resolved = deepcopy(recipe)
    path = Path(recipe.source.file_path).expanduser().resolve(strict=True)
    if path.suffix.lower() != ".zip":
        resolved.source.file_path = str(path)
        resolved.validate_import()
        yield resolved
        return

    from dexsim.kit.simready import open_asset_package, validate_asset_revision

    # A ZIP is a declared AssetRevision, never an arbitrary vendor archive.
    with TemporaryDirectory(prefix="embodichain-usd-scene-") as workspace:
        with open_asset_package(path, workspace=workspace) as root:
            if validate_asset_revision(root) is None:
                raise ValueError("USD scene ZIP must declare a standard AssetRevision.")
            metadata = json.loads((root / "asset.json").read_text())
            entries = {
                item["entrypoint"]
                for item in metadata["representations"]
                if item.get("entrypoint")
                and Path(item["entrypoint"]).suffix.lower()
                in {".usd", ".usda", ".usdc", ".usdz"}
            }
            if len(entries) != 1:
                raise ValueError(
                    "USD scene package requires one declared USD entrypoint."
                )
            entrypoint = (root / entries.pop()).resolve(strict=True)
            if not entrypoint.is_relative_to(root.resolve()):
                raise ValueError("USD scene entrypoint must remain inside its package.")
            resolved.source.file_path = str(entrypoint)
            resolved.validate_import()
            yield resolved


def _validate_environment_usd(recipe: UsdImportCfg) -> None:
    if recipe.robot_cfgs or any(
        isinstance(value, RobotCfg) for value in recipe.asset_cfgs.values()
    ):
        raise ValueError(
            "usd_scene robots must come from embodiment; remove USD robot bindings."
        )


def _inspect_usd_scene(
    recipe: UsdImportCfg,
    *,
    reserved_uids: Sequence[str] = (),
    physics_backend: str = "default",
    device: str | None = None,
    gravity: Sequence[float] = (0.0, 0.0, -9.81),
) -> dict[str, Any]:
    """Use the runtime compiler without allocating a World or object facades."""
    from pxr import Ar, Usd, UsdGeom, UsdPhysics, UsdUtils
    from .usd_import import _compile_usd_import

    recipe.validate_import()
    _validate_environment_usd(recipe)
    explicit_uids = {
        value.get("uid") if isinstance(value, Mapping) else value.uid
        for value in recipe.asset_cfgs.values()
    } - {None}
    if explicit_uids & set(reserved_uids):
        raise ValueError(
            f"USD asset UID already exists: {sorted(explicit_uids & set(reserved_uids))}"
        )
    original = Path(recipe.source.file_path).expanduser().resolve(strict=True)
    dependencies = {str(original): _file_digest(original)}
    with _open_usd_scene_source(recipe) as resolved:
        source = resolved.source.file_path
        layers, assets, missing = UsdUtils.ComputeAllDependencies(source)
        if missing:
            raise ValueError(f"Unresolved USD scene dependencies: {sorted(missing)}")
        if original.suffix.lower() not in {".zip", ".usdz"}:
            paths = {str(layer.realPath) for layer in layers if layer.realPath}
            paths.update(str(path) for path in assets)
            for path in paths:
                while Ar.IsPackageRelativePath(path):
                    path, _ = Ar.SplitPackageRelativePathOuter(path)
                dependencies[str(Path(path).resolve())] = _file_digest(Path(path))
        stage = Usd.Stage.Open(source)
        if stage is None:
            raise ValueError(f"Cannot read USD scene: {original}")
        length = UsdGeom.GetStageMetersPerUnit(stage)
        up = str(UsdGeom.GetStageUpAxis(stage))
        expected_gravity = np.asarray(gravity, dtype=np.float64)
        if recipe.pose is not None:
            rotation = np.asarray(recipe.pose, dtype=np.float64)[:3, :3]
        else:
            rotation = np.eye(3)
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.Scene):
                continue
            scene = UsdPhysics.Scene(prim)
            direction = scene.GetGravityDirectionAttr()
            magnitude = scene.GetGravityMagnitudeAttr()
            if not (direction.HasAuthoredValue() or magnitude.HasAuthoredValue()):
                continue
            vector = np.asarray(direction.Get(), dtype=np.float64)
            norm = np.linalg.norm(vector)
            value = float(magnitude.Get())
            if np.isnan(value) or value == np.inf or not np.isfinite(norm):
                raise ValueError(
                    f"Invalid authored USD scene gravity at {prim.GetPath()}."
                )
            # OpenUSD's zero direction requests -upAxis; a negative magnitude
            # (including the -inf fallback) requests earth gravity in SI.
            # https://openusd.org/release/api/class_usd_physics_scene.html
            if norm == 0:
                vector = np.array([0.0, -1.0, 0.0] if up == "Y" else [0.0, 0.0, -1.0])
            else:
                vector = vector / norm
            vector *= 9.81 if value < 0 else value * length
            if up == "Y":
                vector = np.array([vector[0], -vector[2], vector[1]])
            vector = rotation @ vector
            if not np.allclose(vector, expected_gravity, atol=1e-6, rtol=1e-6):
                raise ValueError(
                    f"USD scene gravity {vector.tolist()} conflicts with environment gravity {expected_gravity.tolist()}."
                )
        plan = _compile_usd_import(
            None,
            resolved,
            physics_backend=physics_backend,
            device_type="cuda" if device is None else str(device).split(":")[0],
        )
        duplicates = {
            uid
            for uid, count in Counter(plan.resource_uids.values()).items()
            if count > 1
        }
        conflicts = set(plan.resource_uids.values()) & set(reserved_uids)
        if duplicates or conflicts:
            raise ValueError(
                f"USD scene UID conflicts: {sorted(duplicates | conflicts)}"
            )
        joints = {
            item.usd.prim_path: [joint.name for joint in item.joints]
            for item in plan.source.articulations
        }
        inventory = dict(
            source=str(original),
            source_sha256=dependencies[str(original)],
            dependencies=dependencies,
            resource_uids=plan.resource_uids,
            resource_types=plan.resource_types,
            resource_source_paths=plan.resource_source_paths,
            constraint_uids=plan.constraint_uids,
            joints=joints,
            links={
                item.usd.prim_path: [link.name for link in item.links]
                for item in plan.source.articulations
            },
            explicit_uids=sorted(explicit_uids),
        )
    _assert_usd_scene_current(inventory)
    return inventory


def _assert_usd_scene_current(inventory: Mapping[str, Any]) -> None:
    for name, digest in inventory["dependencies"].items():
        if _file_digest(Path(name)) != digest:
            raise ValueError(f"USD scene source changed after preflight: {name}")
