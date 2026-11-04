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
"""Endpoint measurements for independent cells of a many-pegs grid."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import numpy as np

from embodichain.toolkits.assembly import AssemblyAsset

__all__: list[str] = []


def _score_many_pegs(
    asset: AssemblyAsset,
    initial: Mapping[str, np.ndarray],
    final: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Measure each cell using its own dimensions and translated plate origin.

    This is a coarse, endpoint seating check. It does not establish clearance,
    tilt, contact onset, trajectory stability or the absence of cross-cell
    contact. Those require independent probes and backend health accounting.
    Input arrays may have an environment batch axis before the XYZ axis.
    """
    assemblies = asset.config.get("assemblies", [])
    if asset.config.get("kind") != "many_pegs" or not assemblies:
        raise ValueError("Many-pegs scoring requires a nonempty grid manifest.")
    rows = []
    all_successes = []
    all_falls = []
    all_insertions = []
    finite = all(np.isfinite(value).all() for value in final.values())
    for assembly in assemblies:
        config = assembly["config"]
        peg_name, plate_name = assembly["peg_body"], assembly["plate_body"]
        scale = float(config.get("scale", 1.0))
        thickness = float(config.get("plate_thickness", 0.01)) * scale
        length = float(config.get("peg_length", 0.03)) * scale
        clearance = float(config.get("clearance", 1e-4)) * scale
        tolerance = min(5e-6 * scale, 0.01 * thickness)
        relative = (
            np.asarray(final[peg_name])[..., :3]
            - np.asarray(final[plate_name])[..., :3]
        )
        insertion = 0.5 * (thickness + length) - relative[..., 2]
        lateral = np.max(np.abs(relative[..., :2]), axis=-1)
        fell = (
            np.asarray(initial[peg_name])[..., 2] - np.asarray(final[peg_name])[..., 2]
        )
        seated = (
            (np.abs(insertion - thickness) <= tolerance)
            & (lateral <= clearance + tolerance)
            & np.isfinite(relative).all(axis=-1)
        )
        all_successes.append(np.ravel(seated))
        all_falls.append(np.ravel(fell))
        all_insertions.append(np.ravel(insertion))
        rows.append(
            {
                "index": assembly["index"],
                "peg_body": peg_name,
                "plate_body": plate_name,
                "clearance_m": clearance,
                "required_insertion_depth_m": thickness,
                "seating_tolerance_m": tolerance,
                "fall_m": float(np.mean(fell)),
                "insertion_depth_m": float(np.mean(insertion)),
                "seating_error_m": float(np.max(np.abs(insertion - thickness))),
                "lateral_error_m": float(np.max(lateral)),
                "success_rate": float(np.mean(seated)),
            }
        )
    return {
        "assembly_count": len(rows),
        "fall_m": float(np.mean(np.concatenate(all_falls))),
        "insertion_depth_m": float(np.mean(np.concatenate(all_insertions))),
        "max_seating_error_m": max(row["seating_error_m"] for row in rows),
        "max_lateral_error_m": max(row["lateral_error_m"] for row in rows),
        "success_rate": float(np.mean(np.concatenate(all_successes))),
        "finite": bool(finite),
        "per_assembly": rows,
        "measurement_scope": "Endpoint seating; independent clearance probes still required",
    }
