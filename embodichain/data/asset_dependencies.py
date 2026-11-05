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
"""Source-asset dependency enumeration without simulation."""

from __future__ import annotations
from pathlib import Path
from typing import Any
from collections.abc import Mapping, Collection
import json
import shlex
import struct
from . import get_data_path
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET

__all__ = ["asset_dependency_files", "collect_declared_asset_files"]


def asset_dependency_files(
    asset_path: Path,
    *,
    reference_overrides: Mapping[str, str | Path] | None = None,
    urdf_collision_links: Collection[str] | None = None,
) -> tuple[Path, ...]:
    """Enumerate one source and its supported local declared dependencies.

    glTF/GLB buffers/images and OBJ/MTL material textures are included. Embedded
    data URIs remain covered by their container bytes. XML mesh, texture and
    include references are traversed transitively. USD
    sources additionally retain an adjacent ``.twist.json`` annotation. This
    reader checks file presence, not geometry or annotation validity; callers
    acquire content hashes and choose their own qualification boundary.

    Args:
        asset_path: Existing local source path. XML references may use relative,
            file or locally resolvable ROS package URIs.
        reference_overrides: Optional model-owner mapping from exact authored XML
            references to their existing absolute local files. These explicit
            bindings preserve registered archive repairs without path guessing.
        urdf_collision_links: Optional exact collision-link roster for the root
            URDF consumer. Visual-only and unselected-link mesh references are
            not read by that consumer; dependency formats remain transitive.

    Returns:
        Sorted unique resolved paths. Opaque non-XML input remains a direct
        dependency, including a file whose XML-looking suffix cannot be parsed.

    Raises:
        FileNotFoundError: A source or declared local dependency is missing.
        ValueError: A dependency uses a remote or unsupported URI scheme, or
            supported mesh metadata cannot be inventoried without guessing.
    """
    overrides = dict(reference_overrides or {})
    if any(
        type(key) is not str or not key or not Path(value).is_absolute()
        for key, value in overrides.items()
    ):
        raise ValueError("Asset overrides require exact references and absolute paths.")
    pending = [asset_path.resolve()]
    visited: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"Scene asset dependency does not exist: {path}")
        visited.add(path)
        pending.extend(_mesh_dependency_files(path))
        if path.suffix.lower() in {".usd", ".usda", ".usdc"}:
            semantics = path.with_suffix(".twist.json")
            if semantics.is_file():
                pending.append(semantics.resolve())
        if path.suffix.lower() not in {".urdf", ".xml", ".mjcf", ".xacro"}:
            continue
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            # Opaque articulation assets remain valid direct dependencies even
            # when their extension suggests XML.
            continue
        elements = root.iter()
        if path == asset_path.resolve() and urdf_collision_links is not None:
            selected = set(urdf_collision_links)
            links = {link.get("name"): link for link in root.findall("link")}
            if not selected or not selected <= links.keys():
                raise ValueError(
                    "URDF collision inventory requires exact existing links."
                )
            elements = (
                element
                for name in selected
                for collision in links[name].findall("collision")
                for element in collision.iter()
            )
        for element in elements:
            tag = element.tag.rsplit("}", maxsplit=1)[-1]
            if tag not in {"mesh", "texture", "include"}:
                continue
            for attribute in ("filename", "file", "url"):
                reference = element.attrib.get(attribute)
                if reference:
                    pending.append(
                        Path(overrides[reference]).resolve()
                        if reference in overrides
                        else _resolve_asset_reference(path, reference)
                    )
    return tuple(sorted(visited))


def _resolve_asset_reference(owner: Path, reference: str) -> Path:
    """Resolve a local filesystem or ROS package URI without global state."""
    parsed = urlparse(reference)
    if parsed.scheme in {"http", "https", "data"}:
        raise ValueError(
            f"Remote scene asset dependencies cannot be integrity-hashed: {reference}"
        )
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).expanduser().resolve()
    if parsed.scheme == "package":
        package_name = parsed.netloc
        relative = Path(unquote(parsed.path.lstrip("/")))
        candidates = [
            ancestor / package_name / relative
            for ancestor in (owner.parent, *owner.parents)
        ]
        candidates.append(owner.parent / relative)
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(
            f"Unable to resolve package asset {reference!r} from {owner}."
        )
    if parsed.scheme:
        raise ValueError(f"Unsupported scene asset URI scheme: {reference}")
    return (owner.parent / unquote(reference)).expanduser().resolve()


def collect_declared_asset_files(value: Any, *, base_dir: Path) -> set[Path]:
    """Resolve declared paths and enumerate their local dependencies and USD layers.

    Registered relative asset identifiers use the existing Data resolver; this
    does not generate substitute assets. Frozen callers verify recorded input
    files before resolving any additional declaration.

    Args:
        value: Plain declaration dictionaries, lists or tuples containing
            ``fpath``, ``urdf_path``, ``xml_path`` or ``mjcf_path`` string fields.
            Other metadata values are ignored.
        base_dir: Owner directory used before registry lookup for a relative
            top-level asset reference.

    Returns:
        Unique resolved source/dependency paths, including file-backed used USD
        layers. This is an inventory, not a sealed content identity or a native
        model-validation result. No simulation resource is created.

    Raises:
        FileNotFoundError: A supported source/dependency reference cannot resolve.
        ValueError: A URI is unsupported or a selected USD source cannot be read.
    """
    files: set[Path] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"fpath", "urdf_path", "xml_path", "mjcf_path"} and isinstance(
                item, str
            ):
                path = Path(item).expanduser()
                if not path.is_absolute():
                    local = base_dir / path
                    path = local if local.is_file() else Path(get_data_path(item))
                files.update(asset_dependency_files(path.resolve()))
                if path.suffix.lower() in {".usd", ".usda", ".usdc"}:
                    from pxr import Usd

                    stage = Usd.Stage.Open(str(path))
                    if stage is None:
                        raise ValueError(f"Frozen articulation cannot be read: {path}")
                    files.update(
                        Path(layer.realPath).resolve()
                        for layer in stage.GetUsedLayers()
                        if layer.realPath
                    )
            else:
                files.update(collect_declared_asset_files(item, base_dir=base_dir))
    elif isinstance(value, (list, tuple)):
        for item in value:
            files.update(collect_declared_asset_files(item, base_dir=base_dir))
    return files


def _mesh_dependency_files(owner: Path) -> tuple[Path, ...]:
    """Read asset references only; the existing URI resolver owns their paths."""
    suffix = owner.suffix.lower()
    if suffix in {".gltf", ".glb"}:
        if suffix == ".gltf":
            document = json.loads(owner.read_text(encoding="utf-8-sig"))
        else:
            with owner.open("rb") as stream:
                header = stream.read(20)
                if len(header) != 20:
                    raise ValueError(f"GLB dependency header is truncated: {owner}")
                magic, version, total, count, kind = struct.unpack("<4sIIII", header)
                if magic != b"glTF" or version != 2 or kind != 0x4E4F534A:
                    raise ValueError(f"Unsupported GLB dependency header: {owner}")
                if count > total - 20 or total != owner.stat().st_size:
                    raise ValueError(f"GLB dependency length is invalid: {owner}")
                data = stream.read(count)
                if len(data) != count:
                    raise ValueError(f"GLB dependency JSON is truncated: {owner}")
                document = json.loads(data.decode("utf-8"))
        if not isinstance(document, dict):
            raise ValueError(f"glTF dependencies require an object header: {owner}")
        references = []
        for group in ("buffers", "images"):
            values = document.get(group, [])
            if not isinstance(values, list) or any(
                not isinstance(v, dict) for v in values
            ):
                raise ValueError(
                    f"glTF dependency {group} must be object entries: {owner}"
                )
            for value in values:
                reference = value.get("uri")
                if reference is None:
                    continue
                if not isinstance(reference, str) or not reference:
                    raise ValueError(f"glTF dependency URI must be nonempty: {owner}")
                if urlparse(reference).scheme.lower() != "data":
                    references.append(_resolve_asset_reference(owner, reference))
        return tuple(references)
    if suffix not in {".obj", ".mtl"}:
        return ()
    references = []
    text = owner.read_text(encoding="utf-8-sig").replace("\\\n", "")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        command = parts[0].lower()
        remainder = parts[1] if len(parts) == 2 else ""
        if suffix == ".obj" and command == "mtllib":
            # Existing readers accept a whole filename containing spaces. The
            # standard multi-library form is used when no such file exists.
            if not remainder.strip():
                raise ValueError(f"OBJ material reference is missing: {owner}")
            literal = _resolve_asset_reference(owner, remainder.strip())
            if literal.is_file():
                references.append(literal)
            else:
                references.extend(
                    _resolve_asset_reference(owner, name)
                    for name in shlex.split(remainder)
                )
        elif suffix == ".mtl" and (
            command.startswith("map_")
            or command in {"bump", "disp", "decal", "refl", "norm"}
        ):
            tokens = shlex.split(remainder)
            index = 0
            fixed = {
                "-blendu": 1,
                "-blendv": 1,
                "-boost": 1,
                "-mm": 2,
                "-texres": 1,
                "-clamp": 1,
                "-bm": 1,
                "-imfchan": 1,
                "-type": 1,
                "-cc": 1,
            }
            while index < len(tokens) and tokens[index].startswith("-"):
                option = tokens[index].lower()
                index += 1
                if option in {"-o", "-s", "-t"}:
                    start = index
                    while index < len(tokens) and index - start < 3:
                        try:
                            float(tokens[index])
                        except ValueError:
                            break
                        index += 1
                    if index == start:
                        raise ValueError(
                            f"Invalid MTL texture option {option}: {owner}"
                        )
                elif option in fixed:
                    index += fixed[option]
                else:
                    raise ValueError(
                        f"Unsupported MTL dependency option {option}: {owner}"
                    )
            if index >= len(tokens):
                raise ValueError(f"MTL texture reference is missing: {owner}")
            references.append(_resolve_asset_reference(owner, " ".join(tokens[index:])))
    return tuple(references)
