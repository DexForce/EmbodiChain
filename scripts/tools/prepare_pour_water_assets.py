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

"""Prepare the local pour-water asset bundle without publishing to the Hub."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
import zlib

import numpy as np
import open3d as o3d
from PIL import Image
import trimesh
from trimesh.visual.material import PBRMaterial

__all__ = ["main"]

_REVISION = "1b92c3d02ca4354984fec961357db0bff7b32166"
_OBJECT_URL = (
    "https://huggingface.co/datasets/robocasa/robocasa-assets/resolve/"
    f"{_REVISION}/objaverse.zip"
)
_OBJECT_ARCHIVE_SIZE = 2163884721
_OBJECTS = {
    "bottle": (
        "water_bottle/water_bottle_6",
        0.1557096168398857,
        -0.08598499000072479,
        0.058,
    ),
    "cup": ("cup/cup_4", 0.08775609731674194, -0.04387804865837097, 0.062),
}
_TEXTURE_ROOT = "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/wood_table_001"
_TEXTURES = {
    "diff": "56652a7339b16a76afad20d3585f66ea",
    "rough": "05b60bd20a1cc4355c5d73a99b245d44",
}
_ATTRIBUTION = """# PourWaterAssets

## Bottle and cup — CC BY 4.0

Adapted from RoboCasa's public Objaverse asset release, by the RoboCasa team
and the original contributing model creators. Upstream object identifiers:
`objaverse/water_bottle/water_bottle_6` and `objaverse/cup/cup_4`.

Source: https://huggingface.co/datasets/robocasa/robocasa-assets
Revision: 1b92c3d02ca4354984fec961357db0bff7b32166 (`objaverse.zip`).
Project: https://github.com/robocasa/robocasa
License: https://creativecommons.org/licenses/by/4.0/

Changes by DexForce: MJCF mesh scales and materials converted to embedded glTF
PBR materials; geometry normalized to metres, Z-up and the existing task's local
origins; dense solid-colour bottle parts simplified to 8000 triangles each.
The bottle's removable flip-cap assembly is omitted for the pouring scene.
The cup's UVs and texture are preserved. Original source files are in `source/`.
The visual opening is retained; convex collision cooking is owned by env.yaml.
This bundle contains no liquid simulation.

## Table texture — CC0 1.0

Wood Table 001, photographed by Dimitrios Savva, processed by Rico Cilliers,
published by Poly Haven: https://polyhaven.com/a/wood_table_001
License: https://creativecommons.org/publicdomain/zero/1.0/

Changes: 1K diffuse/roughness maps embedded into a glTF material; roughness
repacked into the green channel. The round tabletop and rim geometry were
created by DexForce and are licensed under Apache-2.0.

Retain this attribution, licenses and manifest when redistributing the bundle.
"""


class _RangeFile(io.RawIOBase):
    """Allow zipfile to read a pinned large archive using bounded HTTP ranges."""

    def __init__(self, url: str, size: int) -> None:
        self.url = url
        self.size = size
        self.position = 0

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        origin = {os.SEEK_SET: 0, os.SEEK_CUR: self.position, os.SEEK_END: self.size}[
            whence
        ]
        if origin + offset < 0:
            raise ValueError("Negative archive offset.")
        self.position = origin + offset
        return self.position

    def read(self, size: int = -1) -> bytes:
        length = max(0, self.size - self.position)
        if size >= 0:
            length = min(length, size)
        if not length:
            return b""
        end = self.position + length - 1
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.position}-{end}"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            expected_range = f"bytes {self.position}-{end}/{self.size}"
            if (
                response.status != 206
                or response.headers.get("Content-Range") != expected_range
            ):
                raise RuntimeError(
                    "Source server must honor HTTP ranges; refusing a full archive download."
                )
            data = response.read(length + 1)
        if len(data) != length:
            raise IOError("Incomplete source archive range.")
        self.position += length
        return data


def _fetch_sources(output: Path) -> Path:
    source = output / "source"
    prefixes = tuple(f"objaverse/{item[0]}/" for item in _OBJECTS.values())
    with zipfile.ZipFile(_RangeFile(_OBJECT_URL, _OBJECT_ARCHIVE_SIZE)) as archive:
        for info in archive.infolist():
            if info.is_dir() or not info.filename.startswith(prefixes):
                continue
            if "/visual/" not in info.filename and not info.filename.endswith(
                "/model.xml"
            ):
                continue
            target = source / info.filename
            if not target.resolve().is_relative_to(source.resolve()):
                raise ValueError("Unsafe upstream archive member.")
            if target.is_file() and zlib.crc32(target.read_bytes()) == info.CRC:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))
    for suffix, checksum in _TEXTURES.items():
        name = f"wood_table_001_{suffix}_1k.jpg"
        target = source / name
        if (
            not target.is_file()
            or hashlib.md5(target.read_bytes()).hexdigest() != checksum
        ):
            with urllib.request.urlopen(
                f"{_TEXTURE_ROOT}/{name}", timeout=60
            ) as response:
                data = response.read()
            if hashlib.md5(data).hexdigest() != checksum:
                raise ValueError(f"Texture checksum mismatch: {name}.")
            target.write_bytes(data)
    return source


def _export_scene(scene: trimesh.Scene, path: Path) -> None:
    # glTF is Y-up; DexSim converts that root frame back to simulation Z-up.
    gltf_scene = scene.copy()
    gltf_scene.apply_transform(
        trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    )
    gltf_scene.export(path)


def _build_object(source: Path, output: Path, kind: str) -> dict[str, object]:
    relative, height, bottom, diameter = _OBJECTS[kind]
    directory = source / "objaverse" / relative
    document = ET.parse(directory / "model.xml").getroot()
    meshes = {item.get("name"): item for item in document.findall("asset/mesh")}
    materials = {item.get("name"): item for item in document.findall("asset/material")}
    textures = {item.get("name"): item for item in document.findall("asset/texture")}
    parts = []
    original_faces = 0
    for geom in document.findall(".//geom"):
        if geom.get("group") != "1" or not geom.get("mesh"):
            continue
        if kind == "bottle" and geom.get("material") == "Masic_Matte":
            # Remove the closed flip-cap assembly before fitting the open bottle.
            continue
        asset = meshes[geom.get("mesh")]
        mesh = trimesh.load(directory / asset.get("file"), force="mesh", process=False)
        mesh.apply_scale(np.fromstring(asset.get("scale"), sep=" "))
        reference = np.fromstring(asset.get("refquat", "1 0 0 0"), sep=" ")
        mesh.apply_transform(trimesh.transformations.quaternion_matrix(reference).T)
        material = materials[geom.get("material")]
        rgba = np.fromstring(material.get("rgba", "1 1 1 1"), sep=" ")
        texture = textures.get(material.get("texture"))
        image = (
            Image.open(directory / texture.get("file")).convert("RGB")
            if texture is not None
            else None
        )
        uv = mesh.visual.uv if hasattr(mesh.visual, "uv") else None
        original_faces += len(mesh.faces)
        if kind == "bottle" and image is None and len(mesh.faces) > 8000:
            simplified = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(mesh.vertices),
                o3d.utility.Vector3iVector(mesh.faces),
            ).simplify_quadric_decimation(8000)
            mesh = trimesh.Trimesh(
                np.asarray(simplified.vertices),
                np.asarray(simplified.triangles),
                process=False,
            )
            uv = None
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=uv,
            material=PBRMaterial(
                name=material.get("name"),
                baseColorFactor=np.round(rgba * 255).astype(np.uint8),
                baseColorTexture=image,
                metallicFactor=0.0,
                roughnessFactor=0.42 if kind == "bottle" else 0.65,
            ),
        )
        parts.append((geom.get("mesh"), mesh))
    bounds = trimesh.util.concatenate([mesh for _, mesh in parts]).bounds
    scale = np.array(
        [diameter / max(bounds[1, :2] - bounds[0, :2])] * 2
        + [height / (bounds[1, 2] - bounds[0, 2])]
    )
    shift = np.r_[
        -bounds[:, :2].mean(axis=0) * scale[:2], bottom - bounds[0, 2] * scale[2]
    ]
    scene = trimesh.Scene()
    for name, mesh in parts:
        mesh.apply_scale(scale)
        mesh.apply_translation(shift)
        scene.add_geometry(mesh, geom_name=name, node_name=name)
    _export_scene(scene, output / f"{kind}.glb")
    return {
        "source": relative,
        "source_triangles": original_faces,
        "triangles": sum(len(mesh.faces) for _, mesh in parts),
        "scale": scale.tolist(),
        "translation": shift.tolist(),
        "bounds": scene.bounds.tolist(),
    }


def _build_table(source: Path, output: Path) -> dict[str, object]:
    cylinder = trimesh.creation.cylinder(radius=0.5, height=0.1, sections=96)
    top = cylinder.submesh(
        [np.where(cylinder.face_normals[:, 2] > 0.99)[0]], append=True
    )
    rim = cylinder.submesh(
        [np.where(cylinder.face_normals[:, 2] < 0.99)[0]], append=True
    )
    diffuse = Image.open(source / "wood_table_001_diff_1k.jpg").convert("RGB")
    roughness = Image.open(source / "wood_table_001_rough_1k.jpg").convert("L")
    orm = Image.merge(
        "RGB",
        (
            Image.new("L", roughness.size, 255),
            roughness,
            Image.new("L", roughness.size, 0),
        ),
    )
    top.visual = trimesh.visual.TextureVisuals(
        uv=top.vertices[:, :2] / 1.5 + 0.5,
        material=PBRMaterial(
            name="wood_table_001",
            baseColorTexture=diffuse,
            metallicRoughnessTexture=orm,
            metallicFactor=0.0,
            roughnessFactor=1.0,
        ),
    )
    rim.visual = trimesh.visual.TextureVisuals(
        material=PBRMaterial(
            name="walnut_rim",
            baseColorFactor=[44, 26, 15, 255],
            metallicFactor=0.0,
            roughnessFactor=0.6,
        )
    )
    scene = trimesh.Scene()
    scene.add_geometry(top, geom_name="wood_top")
    scene.add_geometry(rim, geom_name="rim")
    _export_scene(scene, output / "table.glb")
    return {
        "triangles": len(top.faces) + len(rim.faces),
        "bounds": scene.bounds.tolist(),
    }


def _write_archive(output: Path, archive_path: Path) -> None:
    """Write explicit directory entries required by Open3D's ZIP extractor."""
    if archive_path.is_relative_to(output):
        raise ValueError("Write the archive outside the asset directory.")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            archive.write(path, path.relative_to(output))


def main() -> None:
    """Download selected assets, convert them and optionally create an upload ZIP."""
    parser = argparse.ArgumentParser(description=__doc__)
    data_root = Path(
        os.environ.get("EMBODICHAIN_DATA_ROOT", "~/.cache/embodichain_data")
    ).expanduser()
    parser.add_argument(
        "--output-dir", type=Path, default=data_root / "PourWaterAssets"
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Optional ZIP path; its root contains the asset files.",
    )
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = _fetch_sources(output)
    assets = {kind: _build_object(source, output, kind) for kind in _OBJECTS}
    assets["table"] = _build_table(source, output)
    (output / "ATTRIBUTION.md").write_text(_ATTRIBUTION, encoding="utf-8")
    for name, url in {
        "CC-BY-4.0.txt": "https://creativecommons.org/licenses/by/4.0/legalcode.txt",
        "CC0-1.0.txt": "https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt",
    }.items():
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "EmbodiChain Asset Preparation (local research)"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            (output / name).write_bytes(response.read())
    (output / "Apache-2.0.txt").write_bytes(
        (Path(__file__).resolve().parents[2] / "LICENSE").read_bytes()
    )
    manifest = {
        "source_revision": _REVISION,
        "source_archive": _OBJECT_URL,
        "assets": assets,
        "sha256": {
            str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if args.archive:
        archive_path = args.archive.expanduser().resolve()
        _write_archive(output, archive_path)
        print(f"Upload bundle: {archive_path}")
    print(f"Local assets: {output}")


if __name__ == "__main__":
    main()
