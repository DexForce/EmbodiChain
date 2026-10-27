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
"""Robot-held assembly scenes composed through mjfit's public generators."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from typing import Any, Literal
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull

from embodichain.toolkits.assembly.adapter import (
    load_mjfit,
    build_assembly,
    import_mjcf,
)
from embodichain.toolkits.assembly.model import AssemblyAsset

__all__ = ["build_task_assembly"]


def _add_ring(
    model: ET.Element, body: ET.Element, xml: str, prefix: str, z: float
) -> None:
    """Merge a public peg generator's perforated plate into one gear body."""
    ring = ET.fromstring(xml)
    meshes = {}
    for mesh in ring.findall("./asset/mesh"):
        copied = deepcopy(mesh)
        meshes[copied.attrib["name"]] = prefix + copied.attrib["name"]
        copied.set("name", meshes[copied.attrib["name"]])
        model.find("asset").append(copied)
    for geom in ring.findall("./worldbody/body[@name='plate']/geom"):
        copied = deepcopy(geom)
        copied.set("name", prefix + copied.attrib.get("name", "piece"))
        copied.set("mesh", meshes[copied.attrib["mesh"]])
        copied.set("class", "mjfit_collision")
        copied.set("pos", f"0 0 {z:.17g}")
        body.append(copied)


def _gear_asset(config: dict[str, Any], seed: int) -> AssemblyAsset:
    mjfit, revision = load_mjfit()
    shaft_radius = float(config.pop("shaft_radius", 0.005))
    bore_clearance = float(config.pop("bore_clearance", 0.000875))
    shaft_tip = float(config.pop("shaft_tip", 0.0065))
    boss_radius = float(config.pop("boss_radius", 0.009))
    boss_height = float(config.pop("boss_height", 0.018))
    cfg = {
        "kind": "gear",
        "teeth": [20, 20],
        "module": 0.00275,
        "face_width": 0.0115,
        "pressure_angle": 22.5,
        "tooth_width": 0.46,
        "motor": False,
        "axles": False,
        **config,
    }
    source = build_assembly(cfg, seed=seed)
    model = ET.fromstring(source.source_mjcf)
    body = model.find("./worldbody/body[@name='gear_b']")
    if body is None:
        raise ValueError("Gear assembly requires exactly a generated gear pair.")
    root_mesh = model.find("./asset/mesh[@name='gear_b_col_0']")
    if root_mesh is None:
        raise ValueError("Pinned mjfit gear output is missing its convex root polygon.")
    vertices = np.fromstring(root_mesh.attrib["vertex"], sep=" ").reshape(-1, 3)
    xy = np.unique(vertices[:, :2], axis=0)
    outline = xy[ConvexHull(xy).vertices].tolist()
    width = float(cfg["face_width"])
    for geom in list(body.findall("geom")):
        if geom.attrib.get("name") in {"gear_b_col_0", "gear_b_visual", "gear_b_hub"}:
            body.remove(geom)
    body.remove(body.find("joint"))
    ET.SubElement(body, "freejoint", name="held_gear_free")
    for name, outer, thickness, z in (
        ("bore_", outline, width, 0.0),
        (
            "boss_",
            {"type": "circle", "radius": boss_radius, "n": 48},
            boss_height,
            (width + boss_height) / 2,
        ),
    ):
        ring = mjfit.peg(
            hole={"type": "circle", "radius": shaft_radius + bore_clearance, "n": 64},
            plate=outer,
            plate_thickness=thickness,
            plate_density=7800.0,
            include_peg=False,
            friction=cfg.get("friction", 0.5),
            condim=cfg.get("condim", 3),
            standalone=False,
        )
        _add_ring(model, body, ring, name, z)
    source_position = np.fromstring(body.attrib["pos"], sep=" ")
    collar_top = source_position[2] - width / 2
    shaft_height = width + shaft_tip
    shaft = ET.SubElement(
        model.find("worldbody"),
        "body",
        name="shaft",
        childclass="mjfit",
        pos=f"{source_position[0]} 0 {collar_top}",
    )
    ET.SubElement(
        shaft,
        "geom",
        name="shaft_collision",
        type="cylinder",
        size=f"{shaft_radius} {shaft_height/2}",
        pos=f"0 0 {shaft_height/2}",
        friction="1.0 0.005 0.0001",
        condim="3",
    )
    ET.SubElement(
        shaft,
        "geom",
        name="collar_collision",
        type="cylinder",
        size=f"{boss_radius*1.5} 0.0025",
        pos="0 0 -0.0025",
        friction="1.0 0.005 0.0001",
        condim="3",
    )
    asset = import_mjcf(
        ET.tostring(model, encoding="unicode"),
        config={
            **cfg,
            "shaft_radius": shaft_radius,
            "bore_clearance": bore_clearance,
            "shaft_tip": shaft_tip,
            "boss_radius": boss_radius,
            "boss_height": boss_height,
            "task_topology": "free held gear, passive hinged anchor, fixed shaft and collar",
        },
        seed=seed,
        generator_revision=revision,
    )
    asset.generator_version = source.generator_version
    return asset


def build_task_assembly(
    kind: Literal["peg", "gear", "nut"],
    config: Mapping[str, Any] | None = None,
    *,
    seed: int = 0,
) -> AssemblyAsset:
    """Build a held part and a genuine mating fixture using public generators.

    The gear scene adds a perforated hub, grip boss, shaft and collar to the
    public gear-pair example. Its neighbor keeps a passive hinge. The nut scene
    reverses the public example's attachment: the robot holds a free nut while
    the bolt stays fixed. Peg fixtures use mjfit's public 2 mm support floor;
    set ``bottom_thickness=0`` for an open bottom. These task changes
    are recorded in the manifest and leave raw :func:`build_assembly` unchanged.
    """
    parameters = dict(config or {})
    if kind == "gear":
        return _gear_asset(parameters, seed)
    if kind == "peg":
        cfg = {
            "kind": "peg",
            "hole": {"type": "stadium", "width": 0.052, "height": 0.032, "n": 32},
            "clearance": 0.00125,
            "plate_size": 0.14,
            "plate_thickness": 0.018,
            "bottom_thickness": 0.002,
            "peg_length": 0.09,
            "peg_mass": 0.5,
            **parameters,
        }
        asset = build_assembly(cfg, seed=seed)
        mjfit, _ = load_mjfit()
        hole = cfg["hole"]
        asset.config["mating_outline"] = (
            getattr(mjfit.shapes, hole["type"])(
                **{k: v for k, v in hole.items() if k != "type"}
            ).tolist()
            if isinstance(hole, dict)
            else np.asarray(hole).tolist()
        )
        return asset
    if kind == "nut":
        cfg = {
            "kind": "thread",
            "major_diameter": 0.036,
            "pitch": 0.004,
            "thread_length": 0.05,
            "clearance": 0.00015,
            "nut_shape": "hex",
            "nut_size": 0.055,
            "nut_height": 0.029,
            "bolt_shank_length": 0.0,
            "friction": 0.29,
            "condim": 3,
            **parameters,
        }
        source = build_assembly(cfg, seed=seed)
        model = ET.fromstring(source.source_mjcf)
        for name, free in (("bolt", False), ("nut", True)):
            body = model.find(f"./worldbody/body[@name='{name}']")
            if body is None:
                raise ValueError(f"Pinned mjfit thread output has no {name}.")
            for joint in list(body):
                if joint.tag in {"joint", "freejoint"}:
                    body.remove(joint)
            if free:
                ET.SubElement(body, "freejoint", name="held_nut_free")
        result = import_mjcf(
            ET.tostring(model, encoding="unicode"),
            config={**cfg, "task_topology": "free held nut, fixed bolt"},
            seed=seed,
            generator_revision=source.generator_revision,
        )
        result.generator_version = source.generator_version
        return result
    raise ValueError(f"Unknown held assembly kind: {kind!r}")
