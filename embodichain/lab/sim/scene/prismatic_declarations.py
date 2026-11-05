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

"""Explicit source prismatic mass/drive declarations without runtime mutation."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any
import re
from .articulation_inputs import discover_prismatic_parts

__all__ = ["source_prismatic_physics_declarations"]

PASSIVE_FRICTION = 0.01
PRISMATIC_LINK_MASS = 1.0


def source_prismatic_physics_declarations(
    articulations: Sequence[Mapping[str, Any]],
    *,
    bound_uids: Sequence[str],
    source_mass_owned_uids: Sequence[str] = (),
) -> tuple[dict[str, Any], ...]:
    """Apply original passive source calibration to explicitly selected assets.

    Args:
        articulations: Canonical scene articulation declarations.
        bound_uids: Exact native identities selected by the caller's prismatic routes.
        source_mass_owned_uids: Explicit assets retaining another authoritative
            mass source. Their moving-link masses are not overwritten.

    Returns:
        Owned declarations preserving existing drive modes/friction and matching
        all discovered moving parts. No simulation, actuator or asset is changed.
    """
    owned = tuple(deepcopy(dict(value)) for value in articulations)
    selected = set(bound_uids)
    mass_owned = set(source_mass_owned_uids)
    if not selected <= {value["uid"] for value in owned}:
        raise ValueError("Prismatic calibration refers to an undeclared articulation.")
    for articulation in owned:
        if articulation["uid"] not in selected:
            continue
        articulation.setdefault("asset_physics_mode", "overlay")
        parts = discover_prismatic_parts(articulation)
        _merge_prismatic_link_mass(
            articulation,
            [part.link for part in parts],
            preserve_source_mass=articulation["uid"] in mass_owned,
        )
        drive = articulation.setdefault("joint_drive_props", {})
        _merge_passive_prismatic_drive(drive, [part.joint for part in parts])
    return owned


def _merge_passive_prismatic_drive(
    drive: dict[str, Any], joints: Sequence[str]
) -> None:
    """Compose generated passive drawer properties with existing drive rules."""
    if drive.get("drive_type", "none") == "none":
        modes = drive.get("target_mode")
        if modes is None:
            drive.setdefault("drive_type", "none")
        elif isinstance(modes, dict) and "drive_type" not in drive:
            # A global NONE would disable calibrated buttons in the same asset.
            for joint in joints:
                if not any(re.fullmatch(pattern, joint) for pattern in modes):
                    modes[re.escape(joint)] = "none"
        if "friction" not in drive:
            drive["friction"] = {joint: PASSIVE_FRICTION for joint in joints}


def _merge_prismatic_link_mass(
    articulation: dict[str, Any],
    links: Sequence[str],
    *,
    preserve_source_mass: bool = False,
) -> None:
    """Compose the E6 moving-link calibration with an existing mass owner."""
    if preserve_source_mass:
        return
    link_attrs = articulation.setdefault("link_attrs", {})
    group = "gen_sim_prismatic_link_mass"
    if group in link_attrs:
        raise ValueError(f"Generated E6 articulation reserves {group!r}.")
    link_attrs[group] = {
        "link_names_expr": sorted(set(links)),
        "attrs": {"mass_props": {"mass": PRISMATIC_LINK_MASS}},
    }
