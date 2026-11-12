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


"""Closed deformable declarations translated into shared evaluation rules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import math
import os
from pathlib import Path


from embodichain.lab.gym.utils._component_composition import (
    _component_path,
    _identifier,
    _load_yaml_component,
    _mapping,
)

__all__: list[str] = []


@dataclass(frozen=True)
class _FinalAcceptance:
    deformable_uid: str
    pairs: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    max_distance_m: float
    mesh_sha256: str | None = None

    def require_registration(self, registration) -> None:
        """Reject declarations whose protected shared final rules were dropped."""
        from embodichain.lab.evaluation import EvaluationSpec, builtin_predicates

        payload = self.evaluation()
        graph = builtin_predicates().compile(EvaluationSpec.from_dict(payload["graph"]))
        references = tuple(payload["final_acceptance"])
        conditions = getattr(registration, "evaluation_conditions", {})
        if tuple(
            getattr(registration, "evaluation_final_refs", ())
        ) != references or any(
            ref not in conditions
            or conditions[ref].identity != graph.select(ref).identity
            for ref in references
        ):
            raise ValueError(
                "Deformable acceptance requires matching registered final acceptance rules."
            )

    def evaluation(self) -> dict:
        """Convert the legacy declaration into shared measured final rules."""
        from embodichain.lab.evaluation._deformable import _centroid_field

        if self.mesh_sha256 is None:
            raise ValueError("Deformable acceptance requires a verified replay mesh.")
        rules = {}
        for index, (source, target) in enumerate(self.pairs):
            rules[f"centroid_pair_{index}"] = {
                "kind": "near_target",
                "bindings": {
                    "position": _centroid_field(
                        self.deformable_uid, self.mesh_sha256, source
                    ),
                    "target": _centroid_field(
                        self.deformable_uid, self.mesh_sha256, target
                    ),
                },
                "params": {
                    "tolerance": self.max_distance_m,
                    "axes": [0, 1],
                    "inclusive": True,
                },
            }
        roots = list(rules)
        return {"graph": {"roots": roots, "rules": rules}, "final_acceptance": roots}


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


def _load_acceptance_evaluation(
    value, surface_value, *, base_dir, component_sources=None
):
    """Resolve immutable declarations before the shared integration is compiled."""
    from ._physical_replay import _load_replay_surface

    if surface_value is None:
        raise ValueError("Deformable acceptance requires its verified replay surface.")
    surface = _load_replay_surface(surface_value)
    declaration = replace(
        _load_final_acceptance(
            value, base_dir=base_dir, component_sources=component_sources
        ),
        mesh_sha256=surface.mesh_sha256,
    )
    if declaration.deformable_uid != surface.object_config.get("uid"):
        raise ValueError("Final acceptance must measure the registered replay mesh.")
    return declaration, surface, declaration.evaluation()
