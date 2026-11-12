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
"""Prepare native collision inputs for visual-only recorded robot models."""

from __future__ import annotations
from copy import deepcopy
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

__all__: list[str] = []


def prepare_robot_collisions(
    source: Path, output: Path, required_links: tuple[str, ...]
) -> tuple[Path, dict]:
    """Derive missing physics only for wholly visual-only models, preserving source.

    Existing collision models must already define the required contact links.
    Visual-only models use the same mesh/origin/scale as their visual geometry;
    the native URDF importer owns collision construction and approximation.
    """
    source = source.resolve()
    tree = ET.parse(source)
    root = tree.getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    if not required_links or any(name not in links for name in required_links):
        raise ValueError("Explicit existing finger contact links are required")
    visual_only = not root.findall("./link/collision")
    generated = []
    if visual_only:
        for name, link in links.items():
            for visual in link.findall("visual"):
                geometry = visual.find("geometry")
                if geometry is None or not len(geometry):
                    raise ValueError(f"Missing visual geometry on {name}")
                collision = ET.SubElement(link, "collision")
                origin = visual.find("origin")
                if origin is not None:
                    collision.append(deepcopy(origin))
                collision.append(deepcopy(geometry))
                generated.append(name)
    for name in required_links:
        if not links[name].findall("collision/geometry"):
            raise ValueError(f"Finger contact link lacks collision geometry: {name}")
    report = dict(
        source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        derived_from_visuals=visual_only,
        generated_links=list(dict.fromkeys(generated)),
        required_contact_links=list(required_links),
        approximation="native URDF mesh collision construction",
    )
    if not visual_only:
        return source, report
    # Preserve model-relative meshes and their sidecar material files by linking
    # original resource directories; never edit or recopy recorded geometry.
    output.mkdir(parents=True, exist_ok=False)
    for mesh in root.findall("./link/*/geometry/mesh"):
        path = Path(mesh.get("filename", ""))
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("Derived robot collisions require model-local meshes")
        if not (source.parent / path).is_file():
            raise ValueError(f"Missing robot mesh: {path}")
        top = output / path.parts[0]
        if not top.exists():
            top.symlink_to((source.parent / path.parts[0]).resolve())
    target = output / "robot_with_collisions.urdf"
    tree.write(target, encoding="unicode", xml_declaration=True)
    report["prepared_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    return target, report
