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

"""Bounded source control-grasp dimensions independent of simulation/Task Engine."""

from __future__ import annotations
from collections.abc import Mapping
import math
from typing import Any

__all__ = ["TwistGripDimensionError", "select_control_grasp_scale"]
PREFERRED_GRIP_DEPTH = 0.010


class TwistGripDimensionError(ValueError):
    """An explicit dimensional candidate violates unchanged gripper bounds."""


def select_control_grasp_scale(
    *,
    source_scale: float,
    source_grip_depth: float,
    source_grip_width: float,
    finger_thickness: float,
    minimum_opening_width: float,
    maximum_opening_width: float,
    opening_margin: float,
    calibration: Mapping[str, Any] | None = None,
    grip_depth_candidate: float | None = None,
) -> tuple[float | None, dict[str, Any]]:
    """Retain the source dimensional scale candidate without assets or robots.

    Args:
        source_scale: Current positive uniform asset scale.
        source_grip_depth: Measured axial engagement length.
        source_grip_width: Measured radial jaw span.
        finger_thickness: Explicit nominal hand thickness.
        minimum_opening_width: Model-owned minimum opening.
        maximum_opening_width: Model-owned maximum opening.
        opening_margin: Explicit source nominal jaw reserve.
        calibration: Optional model-owned physical depth/opening calibration.
        grip_depth_candidate: Explicit requested engagement-depth candidate.

    Returns:
        Original bounded scale or None when source dimensions remain compatible,
        plus a numeric audit. Semantic identities/provenance are added by the
        host; this result qualifies dimensions only, not native task success.
    """
    values = dict(
        source_scale=source_scale,
        source_grip_depth=source_grip_depth,
        source_grip_width=source_grip_width,
        finger_thickness=finger_thickness,
        minimum_opening_width=minimum_opening_width,
        maximum_opening_width=maximum_opening_width,
        opening_margin=opening_margin,
    )
    if any(
        type(value) not in (int, float) or not math.isfinite(value)
        for value in values.values()
    ) or any(
        values[key] <= 0
        for key in values
        if key not in {"opening_margin", "minimum_opening_width"}
    ):
        raise ValueError("E8 scale selection requires measured positive dimensions.")
    if min(values["opening_margin"], values["minimum_opening_width"]) < 0:
        raise ValueError("E8 opening margin and minimum opening cannot be negative.")
    if calibration is not None:
        values["maximum_opening_width"] = min(
            values["maximum_opening_width"], calibration["maximum_opening_width"]
        )
    # Generic finger_thickness is along the opening axis, not axial engagement.
    # The canonical hand's smaller E8 threshold is backed by physical n3 evidence.
    minimum_depth = (
        values["finger_thickness"]
        if calibration is None
        else calibration["minimum_knob_grip_depth"]
    )
    usable_width = values["maximum_opening_width"] - values["opening_margin"]
    if usable_width <= values["minimum_opening_width"]:
        raise ValueError("E8 gripper has no usable opening interval.")
    depth = values["source_grip_depth"]
    width = values["source_grip_width"]
    reasons = []
    if depth < minimum_depth - 1e-8:
        reasons.append("insufficient_axial_grip_depth")
    if not values["minimum_opening_width"] <= width <= usable_width:
        reasons.append("outside_usable_gripper_opening")
    if grip_depth_candidate is not None and (
        type(grip_depth_candidate) not in (int, float)
        or not math.isfinite(grip_depth_candidate)
        or grip_depth_candidate <= 0
    ):
        raise ValueError(
            "E8 explicit grip-depth candidate must be finite and positive."
        )
    factor = (
        float(grip_depth_candidate) / depth
        if grip_depth_candidate is not None
        else PREFERRED_GRIP_DEPTH / depth if reasons else 1.0
    )
    selection_audit = {}
    if grip_depth_candidate is None and reasons:
        lower = max(minimum_depth / depth, values["minimum_opening_width"] / width)
        upper = usable_width / width
        # Keep the representable interval inside the measured dimensional bounds.
        if (
            depth * lower < minimum_depth
            or width * lower < values["minimum_opening_width"]
        ):
            lower = math.nextafter(lower, math.inf)
        if width * upper > usable_width:
            upper = math.nextafter(upper, 0.0)
        if lower > upper:
            raise ValueError(
                "E8 grip-depth candidate cannot satisfy the selected gripper's "
                "depth and opening bounds."
            )
        preferred_factor = factor
        factor = min(max(factor, lower), upper)
        if factor != preferred_factor:
            selection_audit = {
                "candidate_selection": "preferred_depth_clamped_to_feasible_scale_interval",
                "feasible_uniform_scale_factor_interval": [lower, upper],
                "preferred_candidate_limit": (
                    "maximum_opening_width"
                    if preferred_factor > upper
                    else (
                        "minimum_grip_depth"
                        if minimum_depth / depth
                        >= values["minimum_opening_width"] / width
                        else "minimum_opening_width"
                    )
                ),
            }
    candidate_depth, candidate_width = depth * factor, width * factor
    if grip_depth_candidate is not None and 0 < candidate_width - usable_width <= 1e-8:
        factor = math.nextafter(usable_width / width, 0.0)
        candidate_depth, candidate_width = depth * factor, width * factor
    if (
        candidate_depth < minimum_depth - 1e-8
        or not values["minimum_opening_width"] <= candidate_width <= usable_width
    ):
        error = ValueError if grip_depth_candidate is None else TwistGripDimensionError
        raise error(
            "E8 grip-depth candidate cannot satisfy the selected gripper's "
            "depth and opening bounds."
        )
    selected_scale = values["source_scale"] * factor
    audit = {
        "policy": "adapt_only_dimensional_mismatch_prefer_10mm_grip_depth",
        "status": "candidate_scaled" if reasons else "source_dimensions_preserved",
        "reasons": reasons,
        **values,
        "minimum_knob_grip_depth": minimum_depth,
        "depth_screening_basis": (
            "conservative_finger_metadata_proxy"
            if calibration is None
            else calibration["basis"]
        ),
        "pad_collision_extents": (
            None if calibration is None else list(calibration["pad_collision_extents"])
        ),
        "usable_maximum_opening_width": usable_width,
        "preferred_grip_depth": PREFERRED_GRIP_DEPTH,
        "candidate_grip_depth": candidate_depth,
        "candidate_grip_width": candidate_width,
        "candidate_body_scale": selected_scale,
        "assembly_uniform_scale_factor": factor,
        "qualification": "dimensional_candidate_only_runtime_checks_required",
        "source_edited": False,
        **selection_audit,
    }
    if grip_depth_candidate is not None:
        audit.update(
            policy="bounded_runtime_uniform_scale_candidate",
            status="candidate_scaled",
            requested_grip_depth=float(grip_depth_candidate),
        )
    return (
        selected_scale if reasons or grip_depth_candidate is not None else None
    ), audit
