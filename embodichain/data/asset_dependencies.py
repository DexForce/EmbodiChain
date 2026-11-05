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
from . import get_data_path
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET

__all__ = ["asset_dependency_files", "collect_declared_asset_files"]


def asset_dependency_files(asset_path: Path) -> tuple[Path, ...]:
    """Enumerate one source and its supported local declared dependencies.

    XML mesh, texture and include references are traversed transitively. USD
    sources additionally retain an adjacent ``.twist.json`` annotation. This
    reader checks file presence, not geometry or annotation validity; callers
    acquire content hashes and choose their own qualification boundary.

    Args:
        asset_path: Existing local source path. XML references may use relative,
            file or locally resolvable ROS package URIs.

    Returns:
        Sorted unique resolved paths. Opaque non-XML input remains a direct
        dependency, including a file whose XML-looking suffix cannot be parsed.

    Raises:
        FileNotFoundError: A source or declared local dependency is missing.
        ValueError: A dependency uses a remote or unsupported URI scheme.
    """
    pending = [asset_path.resolve()]
    visited: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"Scene asset dependency does not exist: {path}")
        visited.add(path)
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
        for element in root.iter():
            tag = element.tag.rsplit("}", maxsplit=1)[-1]
            if tag not in {"mesh", "texture", "include"}:
                continue
            for attribute in ("filename", "file", "url"):
                reference = element.attrib.get(attribute)
                if reference:
                    pending.append(_resolve_asset_reference(path, reference))
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
