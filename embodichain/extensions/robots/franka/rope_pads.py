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
"""Construct flat FR3 fingertip pads for direct floor cable pickup."""

from __future__ import annotations
import xml.etree.ElementTree as ET
from typing import Sequence

__all__ = ["apply_rope_pads"]


def apply_rope_pads(
    root: ET.Element,
    *,
    stem_size: Sequence[float] = (0.018, 0.008, 0.04),
    center: Sequence[float] = (0, 0.002, 0.04175),
    size: Sequence[float] = (0.028, 0.004, 0.0135),
) -> None:
    """Replace finger geometry, preserving the original inertias and joint frames.

    Args:
        root: FR3 URDF robot element containing both finger links.
        stem_size: XYZ dimensions of each finger stem in metres.
        center: XYZ position of each pad in its finger frame, in metres.
        size: XYZ dimensions of each pad in metres.
    """
    for name in ("fr3_leftfinger", "fr3_rightfinger"):
        link = root.find(f"link[@name='{name}']")
        if link is None:
            raise ValueError(f"FR3 model is missing its finger link {name!r}")
        for child in list(link):
            if child.tag in ("visual", "collision"):
                link.remove(child)
        _box(
            link,
            name + "_stem",
            (0, 0.004, 0.02),
            stem_size,
            ".08 .09 .11 1",
        )
        _box(link, name + "_tip", center, size, ".16 .18 .21 1")


def _box(
    link: ET.Element,
    name: str,
    xyz: Sequence[float],
    size: Sequence[float],
    color: str,
) -> None:
    for kind in ("visual", "collision"):
        element = ET.SubElement(link, kind, name=name + "_" + kind)
        ET.SubElement(element, "origin", xyz=" ".join(map(str, xyz)), rpy="0 0 0")
        ET.SubElement(
            ET.SubElement(element, "geometry"), "box", size=" ".join(map(str, size))
        )
        if kind == "visual":
            ET.SubElement(
                ET.SubElement(element, "material", name=name + "_material"),
                "color",
                rgba=color,
            )
