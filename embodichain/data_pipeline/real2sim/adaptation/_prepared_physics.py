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
"""Materialize explicit physical-asset revisions of portable prepared inputs."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Any

from ._robot_geometry import required_collision_links, validate_robot_collisions

__all__: list[str] = []


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def _contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError(
            f"Prepared resource must remain inside the package: {relative}"
        )
    return path


def verify_prepared_checksums(
    root: Path, *, new_files_root: Path | None = None
) -> dict[str, str]:
    """Verify every declared package member before deriving a new asset revision."""
    root = root.resolve()
    expected: dict[str, str] = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        checksum, relative = line.split("  ", 1)
        if relative in expected or len(checksum) != 64:
            raise ValueError(f"Invalid or duplicate prepared checksum: {relative}")
        path = _contained(root, relative)
        if not path.is_file() or _digest(path) != checksum:
            raise ValueError(f"Prepared input checksum mismatch: {relative}")
        expected[relative] = checksum
    members = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    extra = members - set(expected) - {"SHA256SUMS"}
    allowed_root = new_files_root.resolve() if new_files_root is not None else None
    if allowed_root is not None and not allowed_root.is_relative_to(root):
        raise ValueError("New native evidence must remain inside the package")
    if extra and (
        allowed_root is None
        or any(
            not (root / name).resolve().is_relative_to(allowed_root) for name in extra
        )
    ):
        raise ValueError("Prepared package contains missing or unchecked files")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("Portable prepared packages must contain files, not symlinks")
    return expected


def seal_prepared_checksums(root: Path) -> None:
    """Publish package checksums last, including new provenance and native evidence."""
    root = root.resolve()
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("Portable prepared packages must contain files, not symlinks")
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != root / "SHA256SUMS":
            entries.append(f"{_digest(path)}  {path.relative_to(root).as_posix()}\n")
    temporary = root / ".SHA256SUMS.tmp"
    temporary.write_text("".join(entries))
    temporary.replace(root / "SHA256SUMS")


def prepare_physical_inputs(
    source: Path,
    output: Path,
    *,
    sample_id: str,
    prepare_robot: Callable[[Path, Path], tuple[Path, dict[str, Any]]],
) -> dict[str, Any]:
    """Copy a verified package and explicitly revise one sample's physical robot.

    The model-owned callback supplies a provenance-bearing derived URDF. Source
    observations, calibration and the original robot directory remain intact.
    Structural preflight never claims native physical or task qualification.
    """
    source, output = source.resolve(), output.resolve()
    if output.exists() or output.is_relative_to(source):
        raise ValueError(
            "Physical inputs require a new output outside the source package"
        )
    hashes = verify_prepared_checksums(source)
    manifest = json.loads((source / "manifest.json").read_text())
    matches = [entry for entry in manifest["samples"] if entry["id"] == sample_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one prepared sample: {sample_id}")
    relative = matches[0]["prepared_source"]
    _contained(source, relative)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as staging:
        root = Path(staging) / "package"
        shutil.copytree(source, root)
        sample = _contained(root, relative)
        metadata = json.loads((sample / "robot.json").read_text())
        original_urdf = _contained(sample, metadata["urdf"])
        old_urdf_sha256 = _digest(original_urdf)
        old_input = json.loads((sample / "input.json").read_text())
        old_geometry = old_input["prepared_source"]["robot_geometry_sha256"]
        urdf, provenance = prepare_robot(original_urdf, sample / "robot_physical")
        urdf = urdf.resolve()
        if not urdf.is_relative_to(sample / "robot_physical"):
            raise ValueError(
                "Physical robot preparer returned an asset outside its output"
            )
        metadata["urdf"] = urdf.relative_to(sample).as_posix()
        metadata["required_collision_links"] = list(provenance["collision_links"])
        validation = validate_robot_collisions(urdf, required_collision_links(metadata))
        validation["source"] = urdf.relative_to(root).as_posix()
        validation["collision_mesh_sha256"] = {
            Path(path).relative_to(root).as_posix(): checksum
            for path, checksum in validation["collision_mesh_sha256"].items()
        }
        metadata["physical_robot_provenance"] = (
            "robot_physical/collision_provenance.json"
        )
        _write(sample / "robot.json", metadata)
        old_input["prepared_source"]["robot_geometry_sha256"] = validation[
            "robot_geometry_sha256"
        ]
        old_input["prepared_source"]["physical_robot_provenance"] = metadata[
            "physical_robot_provenance"
        ]
        _write(sample / "input.json", old_input)
        receipt = {
            "schema": "prepared-physical-inputs/v1",
            "sample_id": sample_id,
            "source_checksums_sha256": _digest(source / "SHA256SUMS"),
            "source_urdf_sha256": old_urdf_sha256,
            "source_robot_geometry_sha256": old_geometry,
            "robot_geometry_sha256": validation["robot_geometry_sha256"],
            "collision_count": validation["collision_count"],
            "source_observations_unchanged": True,
            "native_qualification": "not_run",
            "task_success_verified": False,
        }
        reports = sample / "physical_validation"
        reports.mkdir()
        _write(reports / "preflight.json", validation)
        _write(reports / "preparation.json", receipt)
        matches[0]["physical_robot"] = {
            "urdf": f"{relative}/{metadata['urdf']}",
            "robot_geometry_sha256": validation["robot_geometry_sha256"],
            "preparation": f"{relative}/physical_validation/preparation.json",
        }
        _write(root / "manifest.json", manifest)
        audit_path = root / "input_validation.json"
        audit = json.loads(audit_path.read_text())
        if "relocation_check" in audit:
            audit["source_package_relocation_check"] = audit.pop("relocation_check")
        audit["relocation_check"] = {
            "status": "not_run",
            "scope": "Derived physical package; source check is retained separately",
        }
        for entry in audit["cases"]:
            if entry["id"] == sample_id:
                entry.update(
                    robot_geometry_unchanged=False,
                    physical_execution="not_run; native qualification is a separate check",
                    physical_robot_preflight="passed",
                    source_observations_unchanged=True,
                    robot_geometry_sha256=validation["robot_geometry_sha256"],
                    source_robot_geometry_sha256=old_geometry,
                )
        _write(audit_path, audit)
        readme = root / "README.md"
        if readme.is_file():
            historical = readme.read_text()
            readme.write_text(
                "# Derived physical input package\n\n"
                f"Sample `{sample_id}` now selects an explicitly prepared physical robot. "
                "See its `physical_validation/preparation.json` for current geometry identity "
                "and native qualification status, and `robot_physical/collision_provenance.json` "
                "for official sources and licensing. Original observations and the original "
                "robot directory are preserved. Task success has not been evaluated.\n\n"
                "The documentation below describes the original source revision; its "
                "unchanged-geometry and historical validation statements do not describe "
                "this derived physical revision.\n\n---\n\n" + historical
            )
        # Guard the callback's ownership: original URDF, videos, trajectories and
        # all calibration resources must remain byte-identical in the new copy.
        allowed = {
            "README.md",
            "manifest.json",
            "input_validation.json",
            f"{relative}/robot.json",
            f"{relative}/input.json",
        }
        for name, checksum in hashes.items():
            if name not in allowed and _digest(root / name) != checksum:
                raise ValueError(
                    f"Physical preparation changed a source observation: {name}"
                )
        seal_prepared_checksums(root)
        verify_prepared_checksums(root)
        root.rename(output)
    return receipt


def record_native_qualification(
    root: Path, sample_id: str, result: dict[str, Any]
) -> None:
    """Attach a geometry-matching native result and reseal the prepared package."""
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    sample = next(entry for entry in manifest["samples"] if entry["id"] == sample_id)
    receipt_path = _contained(root, sample["physical_robot"]["preparation"])
    receipt = json.loads(receipt_path.read_text())
    native = receipt_path.parent / "native"
    verify_prepared_checksums(root, new_files_root=native)
    if result["robot_geometry_sha256"] != receipt["robot_geometry_sha256"]:
        raise ValueError("Native qualification belongs to a different physical robot")
    report_path = native / "report.json"
    if not report_path.is_file() or json.loads(report_path.read_text()) != result:
        raise ValueError("Native qualification requires its matching persisted report")
    if result.get("schema") != "prepared-robot-native-qualification/v1" or result.get(
        "status"
    ) not in {"passed", "failed"}:
        raise ValueError("Invalid native qualification report")
    if result.get("task_success_verified") is not False:
        raise ValueError("Robot qualification cannot attest task success")
    trace = result.get("traces")
    if result["status"] == "passed" and not trace:
        raise ValueError("Passed native qualification requires its recorded trace")
    if trace:
        path = _contained(native, trace["path"])
        if not path.is_file() or _digest(path) != trace["sha256"]:
            raise ValueError("Native qualification trace checksum mismatch")
    receipt["native_qualification"] = result["status"]
    receipt["native_report"] = "native/report.json"
    _write(receipt_path, receipt)
    audit_path = root / "input_validation.json"
    audit = json.loads(audit_path.read_text())
    for entry in audit["cases"]:
        if entry["id"] == sample_id:
            entry["physical_execution"] = (
                f"native robot qualification {result['status']}; task success not evaluated"
            )
    _write(audit_path, audit)
    seal_prepared_checksums(root)
