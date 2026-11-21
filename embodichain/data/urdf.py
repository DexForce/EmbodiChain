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

"""Content-addressed URDF subtree derivatives without changing source assets."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import os
import xml.etree.ElementTree as ET

__all__ = ["materialize_urdf_subtree"]


def materialize_urdf_subtree(
    path: str | Path,
    root_link: str,
    *,
    mesh_paths: dict[str, str | Path] | None = None,
) -> Path:
    """Cache an exact authored subtree, retaining geometry, inertia and mimics.

    Args:
        path: Existing source URDF.
        root_link: Exact native root of the subtree to retain.
        mesh_paths: Explicit resource corrections for a known source archive.

    Returns:
        Derived URDF beside a source/derivative hash report in the data cache.

    Raises:
        ValueError: The selected root is absent or mimic ownership crosses roots.
        FileNotFoundError: A referenced source or mesh cannot be resolved.
    """
    source = Path(path).resolve(strict=True)
    content = source.read_bytes()
    document = ET.fromstring(content)
    links = {root_link}
    if document.find(f"./link[@name='{root_link}']") is None:
        raise ValueError(f"Missing hand root {root_link!r}.")
    joints = document.findall("joint")
    while True:
        children = {
            j.find("child").get("link")
            for j in joints
            if j.find("parent").get("link") in links
        }
        if children <= links:
            break
        links |= children
    selected = [j for j in joints if j.find("parent").get("link") in links]
    names = {j.get("name") for j in selected}
    if any(
        j.find("mimic") is not None and j.find("mimic").get("joint") not in names
        for j in selected
    ):
        raise ValueError(
            "A subtree cannot borrow mimic parents from another component."
        )
    result = ET.Element("robot", name=root_link)
    for element in document:
        if (
            element.tag == "material"
            or (element.tag == "link" and element.get("name") in links)
            or (element.tag == "joint" and element.get("name") in names)
        ):
            result.append(deepcopy(element))
    # The subtree root must be first: the existing component assembler uses it.
    root = result.find(f"./link[@name='{root_link}']")
    result.remove(root)
    result.insert(0, root)
    for mesh in result.findall(".//mesh"):
        filename = mesh.get("filename")
        target = (mesh_paths or {}).get(filename, source.parent / filename)
        mesh.set("filename", str(Path(target).resolve(strict=True)))
    output = ET.tostring(result, encoding="utf-8", xml_declaration=True)
    source_hash = hashlib.sha256(content).hexdigest()
    # The immutable report describes the complete source, even when an edit
    # outside this subtree leaves the derivative bytes unchanged.
    identity = json.dumps([str(source), source_hash], ensure_ascii=False).encode()
    digest = hashlib.sha256(b"relative-mesh-paths/v2" + identity + output).hexdigest()
    from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT

    folder = Path(EMBODICHAIN_DEFAULT_DATA_ROOT) / "derived" / "urdf" / digest
    folder.mkdir(parents=True, exist_ok=True)
    for mesh in result.findall(".//mesh"):
        mesh.set("filename", os.path.relpath(mesh.get("filename"), folder))
    output = ET.tostring(result, encoding="utf-8", xml_declaration=True)
    derived_hash = hashlib.sha256(output).hexdigest()
    report = json.dumps(
        {
            "source_sha256": source_hash,
            "derived_sha256": derived_hash,
            "source": str(source),
            "root_link": root_link,
            "links": sorted(links),
            "joints": sorted(names),
        },
        sort_keys=True,
    ).encode()
    for name, data in (("component.urdf", output), ("derivation.json", report)):
        target = folder / name
        if target.exists():
            if target.read_bytes() != data:
                raise ValueError(f"Cached URDF derivative was modified: {target}")
            continue
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=folder, delete=False) as stream:
                temporary = stream.name
                stream.write(data)
            os.replace(temporary, target)
        finally:
            if temporary and Path(temporary).exists():
                Path(temporary).unlink()
    return folder / "component.urdf"
