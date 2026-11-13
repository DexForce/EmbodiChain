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

"""Fit a generated scene export into a calibrated source camera and tabletop.

A single photograph fixes object directions but not metric scale. When the
source camera and tabletop are known, each observed object is rescaled and
placed on the tabletop so that its projection reproduces its validated image
mask. The generated table is resized to the calibrated tabletop.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

__all__: list[str] = []

_SCHEMA = "embodichain.gensim-source-calibration/v1"


def load_source_calibration(path: Path) -> dict[str, Any]:
    """Read and validate a source camera/tabletop calibration file.

    Args:
        path: JSON with ``image_size``, ``camera`` (``position``, ``forward``,
            ``left``, ``fovy_deg``), ``table`` (``center_xy``, ``size_xy``,
            ``surface_z``, ``thickness``) and ``scene_yaw_deg``.

    Returns:
        The validated calibration mapping.

    Raises:
        ValueError: The file is not a complete calibration.
    """
    value = json.loads(Path(path).read_text())
    if value.get("schema_version") != _SCHEMA:
        raise ValueError(f"source calibration must declare {_SCHEMA}")
    camera, table = value.get("camera", {}), value.get("table", {})
    required = (
        len(value.get("image_size", [])) == 2
        and all(len(camera.get(k, [])) == 3 for k in ("position", "forward", "left"))
        and float(camera.get("fovy_deg", 0)) > 0
        and len(table.get("center_xy", [])) == 2
        and len(table.get("size_xy", [])) == 2
        and float(table.get("thickness", 0)) > 0
        and "surface_z" in table
        and "scene_yaw_deg" in value
    )
    if not required:
        raise ValueError("source calibration is missing camera, table or scene yaw")
    return value


class _Camera:
    """Pinhole camera with RoboTwin-style forward/left/up axes."""

    def __init__(self, calibration: dict[str, Any]) -> None:
        import numpy as np

        camera = calibration["camera"]
        self.position = np.asarray(camera["position"], dtype=float)
        self.forward = np.asarray(camera["forward"], dtype=float)
        self.forward /= np.linalg.norm(self.forward)
        self.left = np.asarray(camera["left"], dtype=float)
        self.left /= np.linalg.norm(self.left)
        self.up = np.cross(self.forward, self.left)
        width, height = calibration["image_size"]
        self.focal = (height / 2) / math.tan(math.radians(camera["fovy_deg"]) / 2)
        self.center = (width / 2, height / 2)

    def project(self, points: Any) -> tuple[Any, Any]:
        """Return pixel coordinates and depths of world points."""
        import numpy as np

        offset = np.asarray(points, dtype=float) - self.position
        depth = offset @ self.forward
        u = self.center[0] - self.focal * (offset @ self.left) / depth
        v = self.center[1] - self.focal * (offset @ self.up) / depth
        return np.stack([u, v], axis=-1), depth

    def plane_hit(self, u: float, v: float, z: float) -> Any:
        """Intersect the pixel ray with the horizontal plane at height ``z``."""
        direction = (
            self.forward
            + (self.center[0] - u) / self.focal * self.left
            + (self.center[1] - v) / self.focal * self.up
        )
        return self.position + direction * (z - self.position[2]) / direction[2]


def _mask_bbox(path: Path, image_size: tuple[int, int]) -> list[float]:
    """Return the mask bounding box in calibrated image pixels."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        mask = np.asarray(image.convert("L")) > 127
        width, height = image.size
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        raise ValueError(f"Empty object mask: {path}")
    sx, sy = image_size[0] / width, image_size[1] / height
    return [
        cols.min() * sx,
        rows.min() * sy,
        (cols.max() + 1) * sx,
        (rows.max() + 1) * sy,
    ]


def _world_vertices(entry: dict[str, Any], export: Path, yaw: Any) -> Any:
    """Body-frame mesh vertices rotated into the calibrated world."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    from ._run_binding import _EULER, _z_up_mesh

    vertices = np.asarray(_z_up_mesh(entry, export).vertices)
    rotation = Rotation.from_euler(
        _EULER, entry.get("init_rot", [0, 0, 0]), degrees=True
    ).as_matrix()
    if len(vertices) > 4000:
        vertices = vertices[np.linspace(0, len(vertices) - 1, 4000).astype(int)]
    return vertices @ (yaw @ rotation).T


def _fit_object(
    vertices: Any,
    bbox: list[float],
    camera: _Camera,
    surface_z: float,
    clearance: float,
) -> dict[str, Any]:
    """Fit uniform scale and tabletop XY so the projection matches ``bbox``."""
    import numpy as np
    from scipy.optimize import least_squares

    target = np.array(
        [
            (bbox[0] + bbox[2]) / 2,
            (bbox[1] + bbox[3]) / 2,
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
        ]
    )

    def place(params: Any) -> Any:
        scale = math.exp(params[0])
        points = vertices * scale
        offset = np.array(
            [params[1], params[2], surface_z + clearance - points[:, 2].min()]
        )
        return points + offset, offset, scale

    def residual(params: Any) -> Any:
        points, _, _ = place(params)
        pixels, depth = camera.project(points)
        if np.any(depth <= 0):
            return np.full(4, 1e3)
        low, high = pixels.min(axis=0), pixels.max(axis=0)
        return np.array([*(low + high) / 2, *(high - low)]) - target

    bottom = camera.plane_hit((bbox[0] + bbox[2]) / 2, bbox[3], surface_z)
    start = np.array([0.0, bottom[0], bottom[1]])
    pixels, _ = camera.project(place(start)[0])
    extent = np.ptp(pixels, axis=0)
    start[0] = math.log(
        max(1e-3, float(np.mean(target[2:] / np.maximum(extent, 1e-6))))
    )
    start[1:] = camera.plane_hit(*target[:2], surface_z)[:2]
    solution = least_squares(residual, start, x_scale=[0.3, 0.05, 0.05])
    error = residual(solution.x)
    _, offset, scale = place(solution.x)
    return {
        "position": offset.tolist(),
        "scale": float(scale),
        "center_error_px": float(np.hypot(*error[:2])),
        "relative_extent_error": float(np.max(np.abs(error[2:]) / target[2:])),
        "reference_bbox_px": [float(x) for x in bbox],
    }


def calibrate_scene(
    data: dict[str, Any],
    export: Path,
    masks: Path,
    calibration: dict[str, Any],
    *,
    clearance: float = 0.001,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a calibrated scene export and its per-object fit report.

    Args:
        data: Scene export (``embodichain.scene-export/v1``).
        export: Directory the export's mesh paths are relative to.
        masks: Scene-understanding mask directory (``<uid>_mask.png``).
        calibration: Result of :func:`load_source_calibration`.
        clearance: Release gap above the tabletop in metres.

    Returns:
        The calibrated export and a ``reference_layout`` report.

    Raises:
        ValueError: An object lacks a mask or the scene has no table.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation
    from ._run_binding import _EULER

    camera = _Camera(calibration)
    table_cal = calibration["table"]
    surface_z = float(table_cal["surface_z"])
    yaw = Rotation.from_euler(
        "z", calibration["scene_yaw_deg"], degrees=True
    ).as_matrix()
    result = deepcopy(data)
    report: dict[str, Any] = {
        "schema_version": "embodichain.gensim-reference-layout/v1",
        "calibration": calibration,
        "objects": [],
    }

    def oriented(entry: dict[str, Any]) -> list[float]:
        rotation = Rotation.from_euler(
            _EULER, entry.get("init_rot", [0, 0, 0]), degrees=True
        ).as_matrix()
        return (
            Rotation.from_matrix(yaw @ rotation).as_euler(_EULER, degrees=True).tolist()
        )

    tables = [e for e in result.get("background", []) if e.get("uid") == "table"]
    if len(tables) != 1:
        raise ValueError("Source calibration requires exactly one generated table")
    table = tables[0]
    from ._run_binding import _z_up_mesh

    body = np.asarray(_z_up_mesh(table, export).vertices)
    world_rotation = (
        yaw
        @ Rotation.from_euler(
            _EULER, table.get("init_rot", [0, 0, 0]), degrees=True
        ).as_matrix()
    )
    want = np.array([*table_cal["size_xy"], table_cal["thickness"]], dtype=float)
    # Generated tables are level and axis-aligned; map the world size to body axes.
    factor = (np.abs(world_rotation.T) @ want) / np.maximum(np.ptp(body, axis=0), 1e-9)
    scale = np.asarray(table.get("body_scale", [1, 1, 1]), dtype=float)
    # body_scale is authored in the export's Y-up axes: (x, height, depth).
    table["body_scale"] = (scale * factor[[0, 2, 1]]).tolist()
    vertices = (body * factor) @ world_rotation.T
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    table["init_pos"] = [
        float(table_cal["center_xy"][0] - (low[0] + high[0]) / 2),
        float(table_cal["center_xy"][1] - (low[1] + high[1]) / 2),
        float(surface_z - high[2]),
    ]
    table["init_rot"] = oriented(table)
    for category in ("rigid_object", "articulation"):
        for entry in result.get(category, []):
            mask = masks / f"{entry['uid']}_mask.png"
            if not mask.is_file():
                raise ValueError(
                    f"No image mask for calibrated object {entry['uid']!r}"
                )
            bbox = _mask_bbox(mask, tuple(calibration["image_size"]))
            fit = _fit_object(
                _world_vertices(entry, export, yaw), bbox, camera, surface_z, clearance
            )
            entry["init_rot"] = oriented(entry)
            entry["init_pos"] = fit["position"]
            entry["body_scale"] = (
                np.asarray(entry.get("body_scale", [1, 1, 1])) * fit["scale"]
            ).tolist()
            report["objects"].append({"uid": entry["uid"], **fit})
    return result, report
