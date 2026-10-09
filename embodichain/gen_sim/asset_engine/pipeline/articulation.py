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

"""Asset Engine orchestration for articulated SimReady asset generation."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from embodichain.utils.logger import log_info

from ..clients.articulated_generation import ArticulatedGenerationClient
from ..utils.articulated_usdc_utils import (
    _canonicalize_articulated_usdc_bottom_center,
)

__all__ = ["generate_articulated_usdcs"]


def generate_articulated_usdcs(
    *,
    asset_objects: Sequence[Any],
    output_root: str | Path,
    coarse_scales_y_up_by_id: Mapping[str, list[float]],
    articulated_generation_client: ArticulatedGenerationClient | None,
    canonicalize_fn: Callable[
        [str | Path], Path
    ] = _canonicalize_articulated_usdc_bottom_center,
) -> None:
    """Generate and normalize USDC assets for articulated asset records.

    The records are intentionally duck-typed so Asset Engine does not import
    Scene Engine's scene model. They must expose ``id``, ``name``,
    ``category``, ``description``, ``is_articulated``, ``visible_rgba_path``,
    ``articulated_usdc_path`` and ``articulated_usdc_scale`` attributes.

    Args:
        asset_objects: Asset records selected for articulated generation.
        output_root: Directory receiving generated USDC files.
        coarse_scales_y_up_by_id: Positive coarse layout scales keyed by asset ID.
        articulated_generation_client: Asset Engine service client.
        canonicalize_fn: Optional asset normalization hook, primarily for
            compatibility tests and controlled adapters.

    Raises:
        ValueError: If an articulated record lacks required observations or
            scale metadata.
    """
    articulated_objects = []
    for asset in asset_objects:
        if _is_rubiks_cube(asset):
            asset.is_articulated = False
            asset.articulated_usdc_path = None
            asset.articulated_usdc_scale = None
            continue
        if asset.is_articulated:
            articulated_objects.append(asset)
    if not articulated_objects:
        return
    if articulated_generation_client is None:
        raise ValueError(
            "Articulated asset records require an Asset Engine generation client."
        )

    resolved_output_root = Path(output_root).expanduser().resolve()
    resolved_output_root.mkdir(parents=True, exist_ok=True)
    for asset in articulated_objects:
        if asset.visible_rgba_path is None:
            raise ValueError(
                f"Articulated asset {asset.id!r} has no visible RGBA observation."
            )
        coarse_scale_y_up = coarse_scales_y_up_by_id.get(asset.id)
        if (
            not isinstance(coarse_scale_y_up, list)
            or len(coarse_scale_y_up) != 3
            or any(
                not isinstance(value, (int, float)) or not math.isfinite(float(value))
                for value in coarse_scale_y_up
            )
            or any(value <= 0.0 for value in coarse_scale_y_up)
        ):
            raise ValueError(
                f"Articulated asset {asset.id!r} has no valid coarse-layout scale."
            )
        generated_usdc_path = articulated_generation_client.generate_articulated_usdc(
            prompt=_articulation_prompt(asset),
            image_path=asset.visible_rgba_path,
            output_path=resolved_output_root / f"{asset.id}.usdc",
        )
        asset.articulated_usdc_path = str(canonicalize_fn(generated_usdc_path))
        asset.articulated_usdc_scale = list(coarse_scale_y_up)
        log_info(f"Created articulated asset USDC: {asset.id!r}.")


def _articulation_prompt(asset: Any) -> str:
    return (
        f"Object: {asset.name} ({asset.category}). {asset.description}\n"
        "Reconstruct every functional movable part visible in the reference, "
        "including switches, buttons, knobs, doors, drawers, plungers, and "
        "handles. Use real revolute or prismatic joints with physically "
        "meaningful axes and motion limits, connected to valid rigid-body "
        "links. Do not fuse a movable part into the base or add an unrelated "
        "token joint. For every prismatic joint, pass "
        "closed_position=<endpoint> to model.joint(...), equal to the "
        "joint-limit endpoint at which the drawer or slider is physically "
        "closed; do not assume zero. This requirement also applies to every "
        "push button and plunger: use its physically fully depressed endpoint, "
        "not its released/rest coordinate. The value must equal that joint's "
        "lower or upper limit. The official exporter writes "
        "gen_sim:closedPosition from this model field. Do not patch USD files "
        "or private exporter functions; a separate output file is not the "
        "compiler's final artifact. Repair missing model metadata rather than "
        "deleting functional joints. Match the reference door hinge "
        "orientation and initial opening: a side-hinged door must rotate about "
        "a vertical hinge axis, not fold down about a horizontal axis. Deliver "
        "a self-contained USDC with an articulation root, meshes, and enabled "
        "non-fixed joints; a rigid GLB proxy is not enough."
    )


def _is_rubiks_cube(asset: Any) -> bool:
    text = " ".join(
        (str(asset.category), str(asset.name), str(asset.description))
    ).lower()
    return any(
        token in text
        for token in ("rubik", "rubik's", "rubiks", "puzzle_cube", "puzzle cube")
    )
