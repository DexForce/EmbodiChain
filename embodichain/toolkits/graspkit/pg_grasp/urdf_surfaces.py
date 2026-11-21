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

"""Deterministic link-local collision surfaces from explicit URDF selections."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
import trimesh

from ._urdf_geometry import _load_collision_geometry

__all__ = ["sample_urdf_collision_surfaces"]


def _link_surface(
    link: ET.Element,
    directory: Path,
    *,
    max_edge: float,
    max_subdivisions: int,
) -> torch.Tensor:
    clouds = []
    for collision in link.findall("collision"):
        vertices, faces, pose, _ = _load_collision_geometry(collision, directory)
        vertices, _ = trimesh.remesh.subdivide_to_size(
            vertices,
            faces,
            max_edge=max_edge,
            max_iter=max_subdivisions,
        )
        clouds.append(trimesh.transform_points(vertices, pose))
    if not clouds:
        return torch.empty((0, 3), dtype=torch.float32)
    return torch.as_tensor(
        np.unique(np.concatenate(clouds), axis=0), dtype=torch.float32
    )


def sample_urdf_collision_surfaces(
    urdf_path: str | Path,
    *,
    link_names: Sequence[str],
    max_edge_by_link: Mapping[str, float],
    max_subdivisions: int,
) -> dict[str, torch.Tensor]:
    """Sample selected collision links without robot state or simulator access.

    Args:
        urdf_path: URDF path; relative mesh filenames resolve beside this file.
        link_names: Unique, explicit link names in the required output order.
        max_edge_by_link: Positive subdivision edge target in metres for every
            selected link. Pad/body policies belong to the caller.
        max_subdivisions: Positive maximum subdivision depth per collision mesh.

    Returns:
        Owned CPU float32 XYZ tensors in each link's local frame. A selected
        link without collision geometry returns shape ``(0, 3)``. Authored
        scale and collision origins are applied exactly once; visual meshes
        are never substituted for missing collision surfaces.

    Raises:
        ValueError: A selection, sampling bound or collision geometry is invalid.
        FileNotFoundError: The URDF or an explicitly referenced mesh is absent.

    .. attention::
        Sampling uses deterministic triangle subdivision followed by duplicate
        removal. The subdivision bound can limit achieved edge resolution;
        surface samples are not a continuous collision certificate.
    """
    if isinstance(link_names, str):
        raise ValueError("link_names must be a sequence of explicit link names.")
    selected = tuple(link_names)
    if any(type(name) is not str or not name for name in selected) or len(
        set(selected)
    ) != len(selected):
        raise ValueError("Selected link names must be nonempty and unique.")
    if set(max_edge_by_link) != set(selected):
        raise ValueError(
            "Subdivision edge bounds must cover exactly the selected links."
        )
    if type(max_subdivisions) is not int or max_subdivisions < 1:
        raise ValueError("max_subdivisions must be a positive integer.")
    for edge in max_edge_by_link.values():
        if (
            isinstance(edge, bool)
            or not isinstance(edge, (int, float))
            or not (math.isfinite(edge) and edge > 0)
        ):
            raise ValueError("Subdivision edge bounds must be finite and positive.")
    path = Path(urdf_path).expanduser().resolve(strict=True)
    document = ET.parse(path).getroot()
    links = {link.attrib["name"]: link for link in document.findall("link")}
    unknown = set(selected) - links.keys()
    if unknown:
        raise ValueError(f"Selected URDF links do not exist: {sorted(unknown)}")
    return {
        name: _link_surface(
            links[name],
            path.parent,
            max_edge=float(max_edge_by_link[name]),
            max_subdivisions=max_subdivisions,
        )
        for name in selected
    }
