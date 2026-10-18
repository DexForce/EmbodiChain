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

"""Opt-in upright-container closing quality, separate from contact safety."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

__all__: list[str] = []


@dataclass(frozen=True, slots=True)
class UprightClosurePolicy:
    """Limit the named pad-center axis relative to the Z-up horizontal plane."""

    max_tilt_degrees: float
    up_world: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        angle = self.max_tilt_degrees
        if (
            type(angle) not in (int, float)
            or not math.isfinite(angle)
            or not 0 <= angle < 90
        ):
            raise ValueError("Closure tilt must be finite and in [0, 90) degrees.")
        up = np.asarray(self.up_world, dtype=np.float64)
        if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) <= 1e-9:
            raise ValueError("Closure quality requires a finite nonzero up axis.")
        object.__setattr__(
            self, "up_world", tuple(float(x) for x in up / np.linalg.norm(up))
        )


def declarations(value: object) -> dict[str, UprightClosurePolicy]:
    """Decode only explicitly declared, per-segment upright-container policies."""
    if type(value) is not dict or any(type(k) is not str or not k for k in value):
        raise ValueError(
            "upright_container_picks must map Pick segments to tilt degrees."
        )
    return {name: UprightClosurePolicy(angle) for name, angle in value.items()}


def bind_policies(value: object, program: Any, purposes: dict[str, str]) -> tuple:
    """Keep undeclared calls unchanged; never infer policy from an object name."""
    remaining = declarations(value)
    if not remaining:
        return ()
    from embodichain.lab.task_program.semantics import Pick

    result = []
    for segment in program.iter_segments():
        if segment.name not in remaining:
            continue
        if len(segment.calls) != 1 or type(segment.calls[0].call) is not Pick:
            raise ValueError("Upright closure quality needs one exact Pick segment.")
        identifier = f"{program.program_id}/{segment.segment_id}:{segment.calls[0].segment_call_index}"
        if purposes.get(identifier) != "pour":
            raise ValueError(
                "Upright container quality requires an explicit pour Pick purpose."
            )
        result.append((identifier, remaining.pop(segment.name)))
    if remaining:
        raise ValueError("Upright closure quality references a missing Pick segment.")
    return tuple(result)


def pad_closing_axes(parts: tuple) -> np.ndarray:
    """Derive closure from named native pad geometry/FK, not arbitrary TCP X."""
    pads = [p for p in parts if p.pad]
    if len(pads) != 2:
        raise ValueError("Closure quality requires exactly two named pads.")
    centers = []
    for part in pads:
        vertices = np.concatenate([np.asarray(m.vertices) for m in part.meshes])
        center = (vertices.min(0) + vertices.max(0)) * 0.5
        centers.append(part.tcp_poses[:, :3, :3] @ center + part.tcp_poses[:, :3, 3])
    axis = centers[1] - centers[0]
    norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    if not np.isfinite(axis).all() or (norm <= 1e-9).any():
        raise ValueError("Native pad closing axis is unavailable or degenerate.")
    return axis / norm


def inspect_tilt(
    pose: np.ndarray, axes: np.ndarray, policy: UprightClosurePolicy
) -> dict:
    """Check every nominal closing sample up to first bilateral pad rest."""
    world_axes = axes @ pose[:3, :3].T
    vertical = np.abs(world_axes @ np.asarray(policy.up_world))
    maximum = float(np.max(vertical))
    return {
        "valid": bool(
            np.isfinite(maximum)
            and maximum <= math.sin(math.radians(policy.max_tilt_degrees)) + 1e-7
        ),
        "maximum_closure_tilt_degrees": math.degrees(math.asin(min(1.0, maximum))),
    }
