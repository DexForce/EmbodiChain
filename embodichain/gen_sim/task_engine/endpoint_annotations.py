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

"""Audited offline migration of explicit articulation endpoint semantics."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping
import uuid

__all__: list[str] = []

MANIFEST_SCHEMA = "gen_sim.articulation-endpoints/v1"
REPORT_SCHEMA = "gen_sim.articulation-endpoint-migration/v1"
_CLOSED_ATTRIBUTE = "gen_sim:closedPosition"
_USD_SUFFIXES = {".usd", ".usda", ".usdc"}


def migrate_endpoint_manifest(
    *,
    asset_root: str | Path,
    manifest: Mapping[str, Any],
    apply: bool = False,
    backup_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate or atomically apply one reviewed endpoint manifest.

    The manifest supplies decisions; this function never infers endpoint meaning.
    Dry-run is the default. Apply mode requires a fresh backup root and rolls back
    replaced assets if publication fails.
    """
    root = Path(asset_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Endpoint asset root is not a directory: {root}")
    entries = _decode_manifest(manifest)
    backup = None
    if apply:
        if backup_root is None:
            raise ValueError("Endpoint apply mode requires backup_root.")
        backup = Path(backup_root).expanduser().resolve()
        if backup.exists():
            raise FileExistsError(f"Endpoint backup root already exists: {backup}")

    plans = [_plan_asset(root, entry) for entry in entries]
    report_assets = [plan["report"] for plan in plans]
    if not apply:
        return {
            "schema_version": REPORT_SCHEMA,
            "mode": "dry_run",
            "asset_root": root.as_posix(),
            "assets": report_assets,
        }

    assert backup is not None
    changing = [plan for plan in plans if plan["changes"]]
    temporary: list[tuple[dict[str, Any], Path]] = []
    replaced: list[dict[str, Any]] = []
    try:
        for plan in changing:
            source = plan["source"]
            suffix = source.suffix.lower()
            temp = source.with_name(
                f".{source.stem}.endpoint-{uuid.uuid4().hex}.part{suffix}"
            )
            shutil.copy2(source, temp)
            _write_endpoints(temp, plan["values"])
            _verify_only_endpoint_changes(source, temp, plan["values"])
            temporary.append((plan, temp))

        backup.mkdir(parents=True)
        for plan, _ in temporary:
            destination = backup / plan["relative"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(plan["source"], destination)

        for plan, temp in temporary:
            os.replace(temp, plan["source"])
            replaced.append(plan)

        for plan in plans:
            source = plan["source"]
            after_sha256 = _sha256(source)
            report = plan["report"]
            report["after_sha256"] = after_sha256
            report["status"] = "annotated" if plan["changes"] else "already_annotated"
            if plan["changes"]:
                report["backup"] = (backup / plan["relative"]).as_posix()
        return {
            "schema_version": REPORT_SCHEMA,
            "mode": "apply",
            "asset_root": root.as_posix(),
            "backup_root": backup.as_posix(),
            "assets": report_assets,
        }
    except BaseException:
        for plan in reversed(replaced):
            saved = backup / plan["relative"]
            if saved.is_file():
                shutil.copy2(saved, plan["source"])
        raise
    finally:
        for _, temp in temporary:
            temp.unlink(missing_ok=True)


def load_endpoint_manifest(path: str | Path) -> dict[str, Any]:
    """Load a JSON manifest without accepting implicit YAML coercions."""
    source = Path(path).expanduser().resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Endpoint manifest root must be an object.")
    return value


def write_endpoint_report(path: str | Path, report: Mapping[str, Any]) -> Path:
    """Write one machine-readable audit report."""
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Endpoint report already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return destination


def _decode_manifest(value: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    if set(value) != {"schema_version", "assets"}:
        raise ValueError("Endpoint manifest fields must be schema_version and assets.")
    if value["schema_version"] != MANIFEST_SCHEMA:
        raise ValueError("Endpoint manifest schema_version is unsupported.")
    assets = value["assets"]
    if not isinstance(assets, list) or not assets:
        raise ValueError("Endpoint manifest assets must be a non-empty list.")
    decoded = []
    paths = set()
    for index, asset in enumerate(assets):
        if not isinstance(asset, dict) or set(asset) != {
            "path",
            "expected_sha256",
            "joints",
        }:
            raise ValueError(f"Endpoint asset {index} has invalid fields.")
        relative = asset["path"]
        digest = asset["expected_sha256"]
        if (
            not isinstance(relative, str)
            or not relative.strip()
            or Path(relative).is_absolute()
            or "\\" in relative
        ):
            raise ValueError(
                f"Endpoint asset {index} path must be relative POSIX text."
            )
        if relative in paths:
            raise ValueError(f"Endpoint asset path is duplicated: {relative}")
        paths.add(relative)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"Endpoint asset {relative} has invalid SHA256.")
        joints = asset["joints"]
        if not isinstance(joints, list) or not joints:
            raise ValueError(f"Endpoint asset {relative} requires joint decisions.")
        decisions = []
        names = set()
        for joint in joints:
            if not isinstance(joint, dict) or set(joint) != {
                "joint",
                "closed_position",
                "evidence",
            }:
                raise ValueError(f"Endpoint asset {relative} joint fields are invalid.")
            name = joint["joint"]
            closed = joint["closed_position"]
            evidence = joint["evidence"]
            if not isinstance(name, str) or not name.strip() or name != name.strip():
                raise ValueError(
                    f"Endpoint asset {relative} has an invalid joint name."
                )
            if name in names:
                raise ValueError(f"Endpoint joint is duplicated: {relative}:{name}")
            names.add(name)
            if (
                isinstance(closed, bool)
                or not isinstance(closed, (int, float))
                or not math.isfinite(closed)
            ):
                raise ValueError(f"Endpoint {relative}:{name} must be finite and real.")
            if (
                not isinstance(evidence, list)
                or not evidence
                or any(
                    not isinstance(item, str)
                    or not item.strip()
                    or item != item.strip()
                    for item in evidence
                )
            ):
                raise ValueError(f"Endpoint {relative}:{name} requires evidence text.")
            decisions.append(
                {
                    "joint": name,
                    "closed_position": float(closed),
                    "evidence": tuple(evidence),
                }
            )
        decoded.append(
            {
                "path": relative,
                "expected_sha256": digest,
                "joints": tuple(decisions),
            }
        )
    return tuple(decoded)


def _plan_asset(root: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    relative = Path(str(entry["path"]))
    source = (root / relative).resolve()
    if root != source and root not in source.parents:
        raise ValueError(f"Endpoint asset escapes asset_root: {relative.as_posix()}")
    if source.suffix.lower() not in _USD_SUFFIXES or not source.is_file():
        raise ValueError(f"Endpoint asset must be an existing USD file: {relative}")
    before_sha256 = _sha256(source)
    if before_sha256 != entry["expected_sha256"]:
        raise ValueError(
            f"Endpoint asset hash changed for {relative}: "
            f"expected={entry['expected_sha256']}, actual={before_sha256}"
        )

    joints = _prismatic_joints(source)
    declared = {decision["joint"]: decision for decision in entry["joints"]}
    if set(joints) != set(declared):
        raise ValueError(
            f"Endpoint manifest must cover every prismatic joint in {relative}: "
            f"asset={sorted(joints)}, manifest={sorted(declared)}"
        )
    changes = []
    joint_reports = []
    values = {}
    for name, joint in joints.items():
        decision = declared[name]
        closed = decision["closed_position"]
        limits = joint["limits"]
        if not any(math.isclose(closed, value, abs_tol=1.0e-6) for value in limits):
            raise ValueError(
                f"Endpoint {relative}:{name}={closed} is not a joint-limit endpoint "
                f"from {limits}."
            )
        authored = joint["closed_position"]
        if authored is not None and not math.isclose(authored, closed, abs_tol=1.0e-6):
            raise ValueError(
                f"Endpoint {relative}:{name} conflicts with authored value {authored}."
            )
        if authored is None:
            changes.append(name)
        values[name] = closed
        joint_reports.append(
            {
                "joint": name,
                "limits": list(limits),
                "closed_position": closed,
                "evidence": list(decision["evidence"]),
                "status": "would_annotate" if authored is None else "already_annotated",
            }
        )
    return {
        "source": source,
        "relative": relative,
        "values": values,
        "changes": tuple(changes),
        "report": {
            "path": relative.as_posix(),
            "before_sha256": before_sha256,
            "status": "would_annotate" if changes else "already_annotated",
            "joints": joint_reports,
        },
    }


def _prismatic_joints(path: Path) -> dict[str, dict[str, Any]]:
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"Cannot open endpoint asset: {path}")
    result = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.PrismaticJoint):
            continue
        name = prim.GetName()
        if name in result:
            raise ValueError(f"Prismatic joint name is duplicated in {path}: {name}")
        joint = UsdPhysics.PrismaticJoint(prim)
        lower, upper = joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()
        if (
            lower is None
            or upper is None
            or not math.isfinite(lower)
            or not math.isfinite(upper)
            or lower >= upper
        ):
            raise ValueError(f"Prismatic joint has invalid limits: {path}:{name}")
        attribute = prim.GetAttribute(_CLOSED_ATTRIBUTE)
        authored = attribute.Get() if attribute else None
        result[name] = {
            "path": str(prim.GetPath()),
            "limits": (float(lower), float(upper)),
            "closed_position": None if authored is None else float(authored),
        }
    if not result:
        raise ValueError(f"Endpoint asset has no prismatic joints: {path}")
    return result


def _write_endpoints(path: Path, values: Mapping[str, float]) -> None:
    from pxr import Sdf, Usd, UsdPhysics

    stage = Usd.Stage.Open(str(path))
    joints = {
        prim.GetName(): prim
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.PrismaticJoint)
    }
    for name, closed in values.items():
        joints[name].CreateAttribute(_CLOSED_ATTRIBUTE, Sdf.ValueTypeNames.Double).Set(
            closed
        )
    stage.GetRootLayer().Save()


def _verify_only_endpoint_changes(
    before_path: Path, after_path: Path, values: Mapping[str, float]
) -> None:
    from pxr import Usd

    before = Usd.Stage.Open(str(before_path))
    after = Usd.Stage.Open(str(after_path))
    if _stage_snapshot(before) != _stage_snapshot(after):
        raise ValueError(
            f"Endpoint migration changed non-endpoint USD data: {before_path}"
        )
    joints = _prismatic_joints(after_path)
    for name, expected in values.items():
        if not math.isclose(joints[name]["closed_position"], expected, abs_tol=1.0e-6):
            raise ValueError(
                f"Endpoint migration failed to author {before_path}:{name}"
            )


def _stage_snapshot(stage: Any) -> tuple[Any, ...]:
    from pxr import UsdGeom

    default = stage.GetDefaultPrim()
    prims = []
    for prim in stage.Traverse():
        attributes = []
        for attribute in prim.GetAttributes():
            if attribute.GetName() == _CLOSED_ATTRIBUTE:
                continue
            times = attribute.GetTimeSamples()
            attributes.append(
                (
                    attribute.GetName(),
                    str(attribute.GetTypeName()),
                    repr(attribute.Get()),
                    tuple((time, repr(attribute.Get(time))) for time in times),
                )
            )
        relationships = tuple(
            sorted(
                (relationship.GetName(), tuple(map(str, relationship.GetTargets())))
                for relationship in prim.GetRelationships()
            )
        )
        prims.append(
            (
                str(prim.GetPath()),
                prim.GetTypeName(),
                prim.IsActive(),
                prim.IsInstanceable(),
                tuple(sorted(attributes)),
                relationships,
            )
        )
    return (
        str(default.GetPath()) if default else None,
        UsdGeom.GetStageUpAxis(stage),
        UsdGeom.GetStageMetersPerUnit(stage),
        tuple(prims),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
