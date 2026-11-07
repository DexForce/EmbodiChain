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


"""Closed, measured acceptance for a deformable task deployment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from embodichain.lab.gym.utils._component_composition import (
    _component_path,
    _identifier,
    _load_yaml_component,
    _mapping,
)

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager

__all__: list[str] = []


@dataclass(frozen=True)
class _FinalAcceptance:
    deformable_uid: str
    pairs: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    max_distance_m: float
    mesh_sha256: str | None = None

    def measure(self, simulation: SimulationManager) -> torch.Tensor:
        deformable = simulation.get_deformable_object(self.deformable_uid)
        if deformable is None:
            raise ValueError(
                f"Acceptance deformable {self.deformable_uid!r} is missing."
            )
        nodes = deformable.data.nodal_pos_w
        accepted = torch.ones(nodes.shape[0], device=nodes.device, dtype=torch.bool)
        for source, target in self.pairs:
            if max(*source, *target) >= nodes.shape[1]:
                raise ValueError("Acceptance particle index exceeds the mesh size.")
            source_xy = nodes[:, source, :2].mean(dim=1)
            target_xy = nodes[:, target, :2].mean(dim=1)
            distance = torch.linalg.vector_norm(source_xy - target_xy, dim=-1)
            accepted &= torch.isfinite(distance) & (distance <= self.max_distance_m)
        return accepted


def _load_final_acceptance(
    value: object,
    *,
    base_dir: Path,
    component_sources: Mapping[str, Mapping[str, object]] | None = None,
) -> _FinalAcceptance:
    component = _mapping(value, path="acceptance", required=frozenset({"component"}))
    sources = (
        None
        if component_sources is None
        else {
            Path(os.path.abspath(base_dir / name)): data
            for name, data in component_sources.items()
        }
    )
    path = _component_path(
        component["component"],
        base_dir=base_dir,
        path="acceptance.component",
        sources=sources,
    )
    config = _mapping(
        _load_yaml_component(path, field_name="acceptance component", sources=sources),
        path="acceptance component",
        required=frozenset({"kind", "deformable_uid", "pairs", "max_distance_m"}),
    )
    if config["kind"] != "particle_centroid_xy":
        raise ValueError("Unsupported final acceptance kind.")
    tolerance = config["max_distance_m"]
    if (
        type(tolerance) not in (int, float)
        or not math.isfinite(tolerance)
        or tolerance <= 0
    ):
        raise ValueError("Acceptance max_distance_m must be finite and positive.")
    pairs = config["pairs"]
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Acceptance requires at least one particle centroid pair.")
    normalized = []
    for pair in pairs:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("Acceptance pairs must contain two particle groups.")
        for group in pair:
            if (
                not isinstance(group, list)
                or not group
                or any(type(index) is not int or index < 0 for index in group)
                or len(set(group)) != len(group)
            ):
                raise ValueError(
                    "Acceptance groups require unique nonnegative particle indices."
                )
        if set(pair[0]) & set(pair[1]):
            raise ValueError(
                "Acceptance centroid groups must be disjoint within each pair."
            )
        normalized.append(tuple(tuple(sorted(group)) for group in pair))
    return _FinalAcceptance(
        _identifier(config["deformable_uid"], path="acceptance.deformable_uid"),
        tuple(normalized),
        float(tolerance),
    )
