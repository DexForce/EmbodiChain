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

"""Deterministic pinhole views of an unchanged approved rendering scene."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from scripts.benchmark.rendering.scenes import validate_scene

__all__ = ["fixed_views"]


def fixed_views(scene: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the twenty fixed R-01 views without changing the scene.

    The view grid covers five azimuths and four elevations around the first
    external camera's target and distance. It stays on the approved camera's
    side of the scene, avoiding views behind the workshop's back wall. Each
    view carries the same resolution-independent pinhole model; use
    :func:`scripts.benchmark.rendering.scenes.camera_intrinsics` to rasterize it.

    Args:
        scene: Validated frozen scene, with an ``external_front`` camera.

    Returns:
        Twenty independent look-at records with stable IDs and pinhole models.
    """
    validate_scene(scene)
    camera = next(
        (view for view in scene["cameras"] if view["name"] == "external_front"),
        None,
    )
    if camera is None:
        raise ValueError("Fixed views require the approved external_front camera")
    model = scene["camera_model"]
    if not (
        0 < model["horizontal_fov_deg"] < 180
        and 0 < model["near_m"] < model["far_m"]
        and all(math.isfinite(value) for value in model.values())
    ):
        raise ValueError("Invalid portable pinhole camera model")
    offset = [eye - target for eye, target in zip(camera["eye"], camera["target"])]
    distance = math.sqrt(sum(value * value for value in offset))
    azimuth = math.atan2(offset[1], offset[0])
    elevation = math.asin(offset[2] / distance)
    views = []
    for elevation_delta in (-12, -4, 4, 12):
        angle_z = elevation + math.radians(elevation_delta)
        if not 0 < angle_z < math.pi / 2:
            raise ValueError("Fixed-view elevations must remain above the scene")
        for azimuth_delta in (-36, -18, 0, 18, 36):
            angle_xy = azimuth + math.radians(azimuth_delta)
            radial = distance * math.cos(angle_z)
            offset = [
                radial * math.cos(angle_xy),
                radial * math.sin(angle_xy),
                distance * math.sin(angle_z),
            ]
            index = len(views)
            views.append(
                {
                    "name": f"r01_view_{index:02d}",
                    "view_id": index,
                    "eye": [
                        value + origin
                        for value, origin in zip(offset, camera["target"])
                    ],
                    "target": deepcopy(camera["target"]),
                    "up": [0.0, 0.0, 1.0],
                    "camera_model": deepcopy(model),
                }
            )
    return views
