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

"""Owned source button drive/mass declarations separate from semantic acceptance."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from embodichain.compute.geometry.press_calibration import source_press_release_bias
from .control_mass_source import qualify_mass_source_copy
from .press_inputs import PressBinding

__all__ = ["PressSourceCalibrationRequest", "calibrate_press_source_declarations"]


@dataclass(frozen=True, slots=True)
class PressSourceCalibrationRequest:
    """Original and adapted authored source observations plus explicit bias mode."""

    binding: PressBinding
    source_binding: PressBinding
    source_moving_mass: float
    moving_mass: float
    variant: Literal["original", "limits_only", "calibrated", "approach_only"] = (
        "calibrated"
    )
    mass_lineage_sha256: str | None = None


def calibrate_press_source_declarations(
    articulations: Sequence[Mapping[str, Any]],
    requests: Sequence[PressSourceCalibrationRequest],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Compose original source button limit/drive calibration without live objects.

    Args:
        articulations: Ordered canonical physical articulation declarations.
        requests: Explicit original/adapted bindings and source masses.

    Returns:
        Independent physical declarations and numeric audit records. Authored
        drive/limit overrides reject. A pinned mass lineage retains its separate
        fixed inertia owner; no source file or simulation object is modified.
    """
    owned = tuple(deepcopy(dict(value)) for value in articulations)
    records = []
    calibrated = set()
    for request in requests:
        b, source = request.binding, request.source_binding
        if b.object_id in calibrated:
            continue
        calibrated.add(b.object_id)
        if source.object_id != b.object_id:
            raise ValueError("Press calibration source owner differs.")
        if request.variant not in {
            "original",
            "limits_only",
            "calibrated",
            "approach_only",
        }:
            raise ValueError("Unknown source Press calibration variant.")
        mass_copy = None
        if request.mass_lineage_sha256 is not None:
            mass_copy = qualify_mass_source_copy(
                Path(b.source_path), manifest_sha256=request.mass_lineage_sha256
            )
            if (
                mass_copy.source_sha256 != source.source_sha256
                or mass_copy.runtime_sha256 != b.source_sha256
                or mass_copy.scale != b.scale
            ):
                raise ValueError("E9 source changed during unit-scale deployment.")
        art = next(value for value in owned if value["uid"] == b.object_id)
        if art.get("joint_drive_props") or art.get("qpos_limits"):
            raise ValueError(
                "E9 calibration cannot replace authored runtime drive/limit overrides."
            )
        mass = (
            request.source_moving_mass
            if mass_copy is not None
            else request.source_moving_mass * source.scale**3
        )
        stiffness, damping, effort = source_press_release_bias(mass, source.stroke)
        art["asset_physics_mode"] = "overlay"
        record = dict(object_id=b.object_id, source_edited=False)
        if request.variant == "original":
            records.append(
                dict(record, model_origin="source_physics_with_exported_scale")
            )
            continue
        art["qpos_limits"] = {b.joint: list(b.limits)}
        if request.variant == "limits_only":
            records.append(dict(record, model_origin="scaled_limits_only"))
            continue
        art["joint_drive_props"] = dict(
            target_mode={b.joint: "position_velocity"},
            stiffness={b.joint: stiffness},
            damping={b.joint: damping},
            max_effort={b.joint: effort},
        )
        if mass_copy is None:
            art.setdefault("link_attrs", {})["gen_sim_press_density"] = dict(
                link_names_expr=[b.link],
                attrs=dict(mass_props=dict(mass=mass, recompute_inertia=True)),
            )
        records.append(
            dict(
                record,
                schema="gen_sim.press_calibration/v1",
                model_origin="generated_release_bias_not_source_mechanism",
                success_criterion="contact_press_0.05mm_not_activation",
                mass_policy=(
                    "qualified_fixed_mass_source"
                    if mass_copy is not None
                    else "preserve_pre_resize_calibration"
                ),
                calibration_scale=source.scale,
                source_mass=request.moving_mass,
                runtime_mass=mass,
                stiffness=stiffness,
                damping=damping,
                runtime_limits=list(b.limits),
                contact_policy="native_contact_envelope",
                **(
                    dict(
                        mass_lineage_sha256=mass_copy.manifest_sha256,
                        origin_source_sha256=mass_copy.source_sha256,
                    )
                    if mass_copy is not None
                    else {}
                ),
            )
        )
    return owned, tuple(records)
