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

from __future__ import annotations

import numpy as np

from embodichain.gen_sim.scene_engine.core.scene_object import SceneObject
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_generation_utils import (
    layout_object_to_transform_matrix,
    load_glb_mesh,
    transform_matrix_to_layout_object,
)


def update_scene_object_y_up_pose_from_z_up_support(
    *,
    scene_object: SceneObject,
    support_region_z: float,
    center_xy: list[float],
    clearance_m: float = 0.02,
) -> None:
    """Place a SimReady asset on a horizontal z-up support region."""
    if (
        not np.isfinite(support_region_z)
        or clearance_m < 0.0
        or not np.isfinite(clearance_m)
    ):
        raise ValueError("support_region_z and clearance_m must be finite and valid.")
    target_xy = two_floats(center_xy, field_name="center_xy")
    rotation_y_up = three_floats_or_default(
        scene_object.rot, field_name="rot", default=[0.0, 0.0, 0.0]
    )
    mesh = load_scene_object_z_up_mesh(
        scene_object=scene_object, rotation_y_up=rotation_y_up
    )
    target_position_z_up = np.array(
        [
            target_xy[0] - float(mesh.bounds[:, 0].mean()),
            target_xy[1] - float(mesh.bounds[:, 1].mean()),
            float(support_region_z) + clearance_m - float(mesh.bounds[0, 2]),
        ]
    )
    scene_object.pos = (
        np.linalg.inv(y_up_to_z_up_matrix())[:3, :3] @ target_position_z_up
    ).tolist()
    scene_object.rot = rotation_y_up
    scene_object.center_xy = target_xy


def translate_scene_object_y_up_by_z_up_delta(
    *, scene_object: SceneObject, delta_xy: list[float]
) -> None:
    """Translate an existing y-up pose by a solved z-up XY delta."""
    dx, dy = two_floats(delta_xy, field_name="delta_xy")
    position = three_floats_or_default(scene_object.pos, field_name="pos", default=None)
    scene_object.pos = [position[0] + dx, position[1], position[2] - dy]
    if scene_object.center_xy is not None:
        scene_object.center_xy = [
            scene_object.center_xy[0] + dx,
            scene_object.center_xy[1] + dy,
        ]


def measure_scene_object_z_up_world_aabb(
    *, scene_object: SceneObject
) -> list[list[float]]:
    """Measure one current SceneObject pose in z-up world coordinates."""
    position_y_up = three_floats_or_default(
        scene_object.pos, field_name="pos", default=None
    )
    mesh = load_scene_object_z_up_mesh(scene_object=scene_object)
    mesh.apply_translation(
        y_up_to_z_up_matrix()[:3, :3] @ np.asarray(position_y_up, dtype=float)
    )
    return mesh.bounds.tolist()


def load_scene_object_z_up_mesh(
    *, scene_object: SceneObject, rotation_y_up: list[float] | None = None
):
    """Load a SimReady mesh in z-up with orientation and scale but no translation."""
    if scene_object.simready_glb_path is None:
        raise ValueError(f"Asset {scene_object.id!r} has no SimReady GLB path.")
    y_up_layout = {
        "id": scene_object.id,
        "rot": (
            rotation_y_up
            if rotation_y_up is not None
            else three_floats_or_default(
                scene_object.rot, field_name="rot", default=[0.0, 0.0, 0.0]
            )
        ),
        "pos": [0.0, 0.0, 0.0],
        "scale": three_floats_or_default(
            scene_object.scale, field_name="scale", default=[1.0, 1.0, 1.0]
        ),
    }
    y_up_to_z_up = y_up_to_z_up_matrix()
    z_up_layout = transform_matrix_to_layout_object(
        scene_object.id,
        y_up_to_z_up
        @ layout_object_to_transform_matrix(y_up_layout)
        @ np.linalg.inv(y_up_to_z_up),
    )
    mesh = load_glb_mesh(scene_object.simready_glb_path)
    mesh.apply_transform(y_up_to_z_up)
    mesh.apply_transform(layout_object_to_transform_matrix(z_up_layout))
    return mesh


def y_up_to_z_up_matrix() -> np.ndarray:
    """Return the coordinate transform used by layout and export stages."""
    matrix = np.eye(4)
    matrix[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    return matrix


def scene_object_y_up_layout(scene_object: SceneObject) -> dict[str, object]:
    """Adapt one persisted SceneObject pose to a complete y-up layout object."""
    return {
        "id": scene_object.id,
        "rot": scene_object.rot,
        "pos": scene_object.pos,
        "scale": scene_object.scale,
    }


def two_floats(value: object, *, field_name: str) -> list[float]:
    """Validate and return one finite two-value vector."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field_name} must contain two values.")
    result = [float(component) for component in value]
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{field_name} must contain finite values.")
    return result


def three_floats_or_default(
    value: object, *, field_name: str, default: list[float] | None
) -> list[float]:
    """Validate three finite values, or return a canonical default."""
    if value is None:
        if default is None:
            raise ValueError(f"{field_name} must contain three values.")
        return list(default)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field_name} must contain three values.")
    result = [float(component) for component in value]
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{field_name} must contain finite values.")
    return result
