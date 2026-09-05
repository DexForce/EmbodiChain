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

"""Object-local axis analysis shared by GenSim grounding and grasp lowering."""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["LocalGeometryAxes", "analyze_local_geometry_axes"]


@dataclass(frozen=True, slots=True)
class LocalGeometryAxes:
    """Validated local AABB axes with a PCA alignment cross-check."""

    bounds_center: torch.Tensor
    extents: torch.Tensor
    ordered_axis_indices: tuple[int, int, int]
    long_axis_index: int
    short_axis_index: int
    long_axis: torch.Tensor
    long_half_extent: float
    elongation_ratio: float
    principal_alignment: float


def analyze_local_geometry_axes(
    vertices: torch.Tensor,
    *,
    minimum_elongation_ratio: float = 1.10,
    minimum_principal_alignment: float = 0.90,
) -> LocalGeometryAxes:
    """Resolve stable local long/short axes or fail on ambiguous geometry."""
    if (
        not isinstance(vertices, torch.Tensor)
        or not vertices.is_floating_point()
        or vertices.ndim != 2
        or vertices.shape[1] != 3
        or vertices.shape[0] < 3
        or not torch.isfinite(vertices).all()
    ):
        raise ValueError("vertices must be a finite floating tensor shaped (N, 3).")
    if minimum_elongation_ratio <= 1.0:
        raise ValueError("minimum_elongation_ratio must be greater than one.")
    if not 0.0 < minimum_principal_alignment <= 1.0:
        raise ValueError("minimum_principal_alignment must be in (0, 1].")

    lower = vertices.min(dim=0).values
    upper = vertices.max(dim=0).values
    extents = upper - lower
    if torch.any(extents <= 1.0e-8):
        raise ValueError("Object geometry must have non-zero extent on every axis.")
    ordered = torch.argsort(extents, descending=True)
    long_index = int(ordered[0].item())
    short_index = int(ordered[-1].item())
    elongation = float((extents[long_index] / extents[int(ordered[1])]).item())
    if elongation < minimum_elongation_ratio:
        raise ValueError(
            "Object long axis is ambiguous; provide an explicit upright_local_axis."
        )

    centered = vertices - vertices.mean(dim=0, keepdim=True)
    covariance = centered.transpose(0, 1) @ centered
    _, eigenvectors = torch.linalg.eigh(covariance)
    principal = eigenvectors[:, -1]
    principal_index = int(torch.argmax(torch.abs(principal)).item())
    alignment = float(torch.abs(principal[principal_index]).item())
    if principal_index != long_index or alignment < minimum_principal_alignment:
        raise ValueError(
            "Object principal axis is not aligned with its local AABB; provide an "
            "explicit local axis instead of inferring long_axis."
        )

    long_axis = torch.zeros(3, dtype=vertices.dtype, device=vertices.device)
    long_axis[long_index] = 1.0
    return LocalGeometryAxes(
        bounds_center=((lower + upper) * 0.5).clone(),
        extents=extents.clone(),
        ordered_axis_indices=tuple(int(index) for index in ordered.tolist()),
        long_axis_index=long_index,
        short_axis_index=short_index,
        long_axis=long_axis,
        long_half_extent=float((extents[long_index] * 0.5).item()),
        elongation_ratio=elongation,
        principal_alignment=alignment,
    )
