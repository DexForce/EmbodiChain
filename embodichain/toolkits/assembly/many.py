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
"""The public mjfit many-pegs grid as a portable assembly manifest."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from pathlib import Path
from typing import Any

import numpy as np

from .adapter import load_mjfit, import_mjcf
from .model import AssemblyAsset

__all__ = ["build_many_pegs"]


def build_many_pegs(
    count: int = 25,
    *,
    seed: int = 0,
    family: str = "primitive",
    cache_dir: str | Path | None = None,
    geometry_parameters: Mapping[str, Any] | None = None,
    **sampling_parameters: Any,
) -> AssemblyAsset:
    """Prepare the official many-pegs example as one portable physical scene.

    Generation uses public ``sample_pegs``, ``build(standalone=False)`` and
    ``grid`` calls from the pinned mjfit revision. The official example's
    spacing is retained: 0.3 m for primitive pegs and 0.2 m for convex pegs.
    Each grid cell has its own fixed plate and free peg; cells are not arenas
    or hidden copies of a shared object library. Generation and MJCF parsing
    happen here, before simulation starts.

    Args:
        count: Positive number of independently sampled peg assemblies.
        seed: Seed shared by the public sampler's indexed random streams.
        family: Public sampler family, ``primitive`` or ``convex``.
        cache_dir: Optional directory for the checksummed portable manifest.
        geometry_parameters: Optional outline settings applied after sampling.
            Currently ``{"max_error": 2e-6}`` sets this public shape parameter
            on every parametric circle, stadium and Bezier outline in each
            part, including its islands. Explicit side counts are removed so
            the requested error controls tessellation. Baked vertex arrays
            remain unchanged; use the convex sampler's ``max_error`` argument
            to control its sampling. Original and actual configurations are
            recorded separately. With no overrides, official geometry settings
            are retained.
        **sampling_parameters: Keyword arguments forwarded to ``sample_pegs``.

    Returns:
        Asset with source-prefixed bodies and a ``config["assemblies"]`` list
        recording each sampled configuration, body names and grid placement.
        Loading a saved manifest does not require mjfit or MuJoCo.

    Raises:
        ValueError: If count or family is invalid, or outline overrides are
            unsupported or have a nonpositive/nonfinite error bound.
        RuntimeError: If the installed generator revision is not pinned.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("The many-pegs grid requires a positive integer count.")
    if family not in {"primitive", "convex"}:
        raise ValueError("The many-pegs family must be primitive or convex.")
    geometry_parameters = dict(geometry_parameters or {})
    if geometry_parameters:
        if set(geometry_parameters) != {"max_error"}:
            raise ValueError("Many-pegs geometry_parameters supports only max_error.")
        max_error = float(geometry_parameters["max_error"])
        if not math.isfinite(max_error) or max_error <= 0:
            raise ValueError("The outline max_error must be finite and positive.")
        geometry_parameters["max_error"] = max_error
    mjfit, revision = load_mjfit()
    sampled_configs = mjfit.sample_pegs(
        count, seed=seed, family=family, **sampling_parameters
    )
    configs = [deepcopy(config) for config in sampled_configs]
    if geometry_parameters:
        for config in configs:
            for key in ("hole", "peg", "handle", "plate"):
                _configure_outline(config.get(key), geometry_parameters)
            for outline in config.get("islands", []):
                _configure_outline(outline, geometry_parameters)
    spacing = 0.3 if family == "primitive" else 0.2
    spec = mjfit.grid(
        [mjfit.build(config, standalone=False) for config in configs],
        spacing=spacing,
    )
    # Match the source example: compile before serializing the attached specs.
    # The resulting self-contained XML is also the native reference's input.
    spec.compile()
    asset = import_mjcf(spec.to_xml(), seed=seed, generator_revision=revision)
    bodies = {body.name: body for body in asset.bodies}
    assemblies = []
    for index, config in enumerate(configs):
        plate_name, peg_name = f"p{index}_plate", f"p{index}_peg"
        if (
            plate_name not in bodies
            or peg_name not in bodies
            or bodies[plate_name].joint.kind != "fixed"
            or bodies[peg_name].joint.kind != "free"
        ):
            raise RuntimeError("The pinned many-pegs grid changed body ownership.")
        assemblies.append(
            {
                "index": index,
                "plate_body": plate_name,
                "peg_body": peg_name,
                "position": [*bodies[plate_name].position[:2], 0.0],
                "config": config,
            }
        )
        if geometry_parameters:
            assemblies[-1]["sampled_config"] = sampled_configs[index]
    if len(bodies) != 2 * count:
        raise RuntimeError("The many-pegs grid contains unexpected physical bodies.")
    asset.config = {
        "kind": "many_pegs",
        "count": count,
        "family": family,
        "spacing": spacing,
        "sampling_parameters": sampling_parameters,
        "assemblies": assemblies,
    }
    if geometry_parameters:
        asset.config["geometry_parameters"] = geometry_parameters
    asset.generator_version = mjfit.__version__
    if cache_dir is not None:
        asset.save(Path(cache_dir) / f"{asset.content_hash}.json")
    return asset


def _configure_outline(outline: Any, parameters: Mapping[str, Any]) -> None:
    """Set public curved-shape tessellation parameters on copied mappings."""
    if isinstance(outline, dict) and outline.get("type") in {
        "circle",
        "stadium",
        "bezier",
    }:
        outline.update(parameters)
        outline.pop("n", None)
