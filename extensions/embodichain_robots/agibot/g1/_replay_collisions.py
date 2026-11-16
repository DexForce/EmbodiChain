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
"""Explicit collision repair for the pinned AgiBot observed-replay export.

This recipe transfers authored G1_120s collision geometry, never visual-to-
collision guesses. It preserves the prepared kinematic tree and calibration.
The result still requires native simulation qualification.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

__all__: list[str] = []

_PREPARED_SHA256 = "3afccc92f421d944672da01f7ec535082fd4f5f4dd1d5e4f26ec001218874e7e"
_URDF_REVISION = "37e760fdd15d27c45233baf11811059ecc65ee9d"
_URDF_SHA256 = "fdab90abab488633ebdc3a474d32c431c6c70a8504501f62dec32845173e6c91"
_URDF_FILE = "old-G1_120s.urdf"
_URDF_URL = (
    "https://huggingface.co/datasets/agibot-world/GenieSimAssets/resolve/"
    f"{_URDF_REVISION}/G1_120s/G1_120s.urdf"
)


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _source_bundle(bundle: Path | None) -> tuple[Path, dict[str, Any]]:
    from .assets import AgibotG1, _ensure_file, _FILES, _REVISION, _MODEL_SOURCE

    if bundle is None:
        bundle = Path(AgibotG1().extract_dir)
        _ensure_file(_URDF_URL, bundle / _URDF_FILE, _URDF_SHA256)
    bundle = bundle.resolve()
    expected = {**_FILES, _URDF_FILE: _URDF_SHA256}
    for relative, digest in expected.items():
        path = bundle / relative
        if not path.is_file() or _digest(path) != digest:
            raise ValueError(f"AgiBot collision source SHA256 mismatch: {relative}")
    return bundle, {
        "usd_revision": _REVISION,
        "usd_source": _MODEL_SOURCE,
        "urdf_revision": _URDF_REVISION,
        "urdf_source": _URDF_URL,
        "sha256": expected,
        "license": "CC-BY-NC-SA-4.0",
        "attribution": "AgiBot / agibot-world, GenieSimAssets G1_120s",
    }


def _meshes(stage: Any, link_name: str, subtree: str) -> list[dict[str, Any]]:
    from pxr import Usd, UsdGeom

    link = stage.GetPrimAtPath(f"/G1/{link_name}")
    parent = stage.GetPrimAtPath(f"/G1/{link_name}/{subtree}")
    if not link or not parent:
        raise ValueError(f"Missing authored {subtree} for {link_name}")
    cache = UsdGeom.XformCache()
    result = []
    for prim in Usd.PrimRange(parent, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        transform = np.asarray(cache.ComputeRelativeTransform(prim, link)[0])
        vertices = (np.c_[points, np.ones(len(points))] @ transform)[:, :3]
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        faces, offset = [], 0
        for count in counts:
            polygon = indices[offset : offset + count]
            faces.extend(
                (polygon[0], polygon[i], polygon[i + 1]) for i in range(1, count - 1)
            )
            offset += count
        faces = np.asarray(faces, dtype=int)
        if mesh.GetOrientationAttr().Get() == "leftHanded":
            faces = faces[:, ::-1]
        result.append(
            {"prim": str(prim.GetPath()), "vertices": vertices, "faces": faces}
        )
    if not result:
        raise ValueError(f"Missing authored meshes for {link_name}/{subtree}")
    return result


def _check_link_frame(stage: Any, link: ET.Element, source: Path) -> float:
    import trimesh
    from scipy.spatial import cKDTree

    name = link.get("name")
    visual = link.find("visual")
    if visual is None or visual.find("origin") is not None:
        raise ValueError(f"Expected the pinned link-local visual export: {name}")
    filename = visual.find("geometry/mesh").get("filename")
    prepared = trimesh.load(source.parent / filename, force="mesh", process=False)
    authored = np.concatenate(
        [part["vertices"] for part in _meshes(stage, name, "visuals")]
    )
    exported = np.asarray(prepared.vertices)
    error = max(
        cKDTree(authored).query(exported)[0].max(),
        cKDTree(exported).query(authored)[0].max(),
    )
    if not np.isfinite(error) or error > 1e-6:
        raise ValueError(
            f"Prepared and authored link frames disagree: {name} ({error:g} m)"
        )
    return float(error)


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# Authored AgiBot G1 collision geometry; CC-BY-NC-SA-4.0\n")
        for point in vertices:
            stream.write(
                "v " + " ".join(format(float(v), ".17g") for v in point) + "\n"
            )
        for face in faces:
            stream.write("f " + " ".join(str(int(v) + 1) for v in face) + "\n")


def prepare_replay_collisions(
    source_urdf: Path,
    output_dir: Path,
    *,
    source_bundle: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Transfer official collisions to the exact visual-only observed export.

    Args:
        source_urdf: Unmodified robot URDF from the #1212 pinned prepared package.
        output_dir: New standalone robot directory; original resources are copied.
        source_bundle: Optional offline official USD bundle, license and
            ``old-G1_120s.urdf``. All upstream bytes are checked against pinned
            hashes. Omit to resolve the normal AgiBot asset cache and download.

    Returns:
        Repaired URDF path and provenance report. Arm cylinders retain the
        official URDF declarations; hand meshes come from authored USD collision
        prims. Original joints, inertias, visuals and source resources survive.
        Preparation alone does not establish physical or task success.
    """
    source_urdf, output_dir = source_urdf.resolve(), output_dir.resolve()
    if _digest(source_urdf) != _PREPARED_SHA256:
        raise ValueError(
            "Collision recipe requires the pinned #1212 prepared AgiBot URDF"
        )
    if output_dir.exists():
        raise FileExistsError(output_dir)
    bundle, provenance = _source_bundle(source_bundle)
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(bundle / "robot.usda"))
    if stage is None or UsdGeom.GetStageMetersPerUnit(stage) != 1:
        raise ValueError("Expected the pinned metre-scale AgiBot USD stage")
    original = ET.parse(bundle / _URDF_FILE).getroot()
    tree = ET.parse(source_urdf)
    root = tree.getroot()
    if root.findall("./link/collision"):
        raise ValueError("The collision repair must not replace existing physics")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=".agibot-collisions-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary) / "robot"
        staging.mkdir()
        (staging / "collisions").mkdir()
        resources = {}
        for node in root.findall(".//mesh") + root.findall(".//texture"):
            relative = Path(node.get("filename", ""))
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError("Prepared robot resources must remain model-local")
            source = source_urdf.parent / relative
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            copyfile(source, destination)
            resources[relative.as_posix()] = _digest(destination)
        links = []
        for authored in original.findall("link"):
            collisions = authored.findall("collision")
            if not collisions:
                continue
            name = authored.get("name")
            link = root.find(f"link[@name='{name}']")
            if link is None:
                raise ValueError(
                    f"Prepared robot lacks authored collision link: {name}"
                )
            agreement = _check_link_frame(stage, link, source_urdf)
            record = {"link": name, "visual_frame_error_m": agreement, "collisions": []}
            for index, source_collision in enumerate(collisions):
                collision = deepcopy(source_collision)
                mesh_node = collision.find("geometry/mesh")
                if mesh_node is None:
                    record["collisions"].append(
                        {"kind": "cylinder", "source_link": name}
                    )
                else:
                    parts = _meshes(stage, name, "collisions")
                    if len(parts) != 1:
                        raise ValueError(
                            f"Expected one authored hand collision mesh: {name}"
                        )
                    part = parts[0]
                    relative = Path("collisions") / f"{name}_{index}.obj"
                    _write_obj(staging / relative, part["vertices"], part["faces"])
                    mesh_node.set("filename", relative.as_posix())
                    # Points already include the collision-to-link transform.
                    origin = collision.find("origin")
                    if origin is not None:
                        collision.remove(origin)
                    usd_collision = stage.GetPrimAtPath(f"/G1/{name}/collisions")
                    approximation = (
                        UsdPhysics.MeshCollisionAPI(usd_collision)
                        .GetApproximationAttr()
                        .Get()
                    )
                    record["collisions"].append(
                        {
                            "kind": "mesh",
                            "source_prim": part["prim"],
                            "authored_approximation": approximation,
                            "path": relative.as_posix(),
                            "sha256": _digest(staging / relative),
                        }
                    )
                link.append(collision)
            links.append(record)
        if len(root.findall("./link/collision")) != 36:
            raise ValueError("Expected all 14 arm and 22 hand collision definitions")
        target = staging / "robot.urdf"
        tree.write(target, encoding="unicode", xml_declaration=True)
        copyfile(bundle / "LICENSE", staging / "LICENSE.agibot")
        report = {
            "recipe": "agibot-g1-observed-authored-collisions-v1",
            "source_urdf_sha256": _PREPARED_SHA256,
            "output_urdf_sha256": _digest(target),
            "upstream": provenance,
            "original_resources_sha256": resources,
            "links": links,
            "collision_links": [record["link"] for record in links],
            "collision_count": 36,
            "arm_collision_count": 14,
            "hand_collision_count": 22,
            "kinematics_and_inertias_preserved": True,
            "derived_from_visuals": False,
            "native_physics_verified": False,
            "task_success_verified": False,
            "approximation_note": (
                "Authored URDF arm cylinders are enabled explicitly; the USD disables arm collisions. "
                "Hand collision meshes retain authored geometry; the native URDF importer owns cooking. "
                "Four support-link meshes use convexDecomposition in the source USD."
            ),
        }
        (staging / "collision_provenance.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        staging.rename(output_dir)
    return output_dir / "robot.urdf", report
