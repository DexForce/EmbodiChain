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

import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.spatial.transform import Rotation

from embodichain.gen_sim.scene_engine.core.asset import Asset
from embodichain.gen_sim.scene_engine.core.scene import Scene
from embodichain.gen_sim.scene_engine.core.table import Table

_DEFAULT_MAX_CONVEX_HULL_NUM = 16
_TABLE_PHYSICS_ATTRS = {
    "mass": 10.0,
    "static_friction": 0.95,
    "dynamic_friction": 0.9,
    "restitution": 0.01,
}
_ASSET_PHYSICS_ATTRS = {
    "mass": 0.01,
    "contact_offset": 0.003,
    "rest_offset": 0.001,
    "restitution": 0.01,
    "max_depenetration_velocity": 10.0,
    "min_position_iters": 32,
    "min_velocity_iters": 8,
}
_Y_UP_TO_Z_UP_ROTATION = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)


def export_scene_to_gym(
    *,
    scene: Scene,
    output_root: str | Path,
    table_max_convex_hull_num: int = _DEFAULT_MAX_CONVEX_HULL_NUM,
    asset_max_convex_hull_num: int = _DEFAULT_MAX_CONVEX_HULL_NUM,
) -> Path:
    """Write the Gym config and copy SimReady GLBs into ``mesh_assets``.

    Scene layouts are y-up. The simulator automatically converts each y-up GLB
    to z-up, so this exporter copies each GLB unchanged and converts only its
    world position and rotation for ``init_pos`` and ``init_rot``. ``body_scale``
    remains the original y-up scale associated with the GLB.
    """
    if scene.table is None:
        raise ValueError("Cannot export a gym scene without a table.")
    table_max_convex_hull_num = _positive_int(
        table_max_convex_hull_num,
        field_name="table_max_convex_hull_num",
    )
    asset_max_convex_hull_num = _positive_int(
        asset_max_convex_hull_num,
        field_name="asset_max_convex_hull_num",
    )

    export_root = Path(output_root).expanduser().resolve() / "gym_export"
    mesh_assets_root = export_root / "mesh_assets"
    mesh_assets_root.mkdir(parents=True, exist_ok=True)

    scene_objects = [scene.table, *scene.assets]
    object_ids = [scene_object.id for scene_object in scene_objects]
    if len(set(object_ids)) != len(object_ids):
        raise ValueError("Gym export requires unique table and asset ids.")

    exported_entries = {
        scene_object.id: _copy_scene_object_to_gym_assets(
            scene_object=scene_object,
            mesh_assets_root=mesh_assets_root,
        )
        for scene_object in scene_objects
    }
    gym_config = {
        "id": f"Prompt2Scene-{int(time.time() * 1000)}-v0",
        "max_episodes": 10,
        "max_episode_steps": 300,
        "env": {"events": {}, "observations": {}, "dataset": {}},
        "robot": {},
        "sensor": [],
        "light": {},
        "background": [
            _gym_object_config(
                scene_object=scene.table,
                asset_relative_path=exported_entries[scene.table.id],
                body_type="kinematic",
                attrs=_TABLE_PHYSICS_ATTRS,
                max_convex_hull_num=table_max_convex_hull_num,
            )
        ],
        "rigid_object": [
            _gym_object_config(
                scene_object=asset,
                asset_relative_path=exported_entries[asset.id],
                body_type="dynamic",
                attrs=_ASSET_PHYSICS_ATTRS,
                max_convex_hull_num=asset_max_convex_hull_num,
            )
            for asset in scene.assets
        ],
    }
    gym_config_path = export_root / "gym_config.json"
    gym_config_path.write_text(
        json.dumps(gym_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return gym_config_path


def _copy_scene_object_to_gym_assets(
    *,
    scene_object: Table | Asset,
    mesh_assets_root: Path,
) -> str:
    """Copy one referenced SimReady GLB and return its config-relative path."""
    object_id = scene_object.id
    if Path(object_id).name != object_id or object_id in {"", ".", ".."}:
        raise ValueError(
            f"Scene object id is not safe for a GLB filename: {object_id!r}"
        )
    if scene_object.simready_glb_path is None:
        raise ValueError(f"Scene object {object_id!r} has no SimReady GLB path.")

    source_glb_path = Path(scene_object.simready_glb_path).expanduser().resolve()
    if not source_glb_path.is_file():
        raise FileNotFoundError(
            f"SimReady GLB for scene object {object_id!r} not found: {source_glb_path}"
        )
    destination_glb_path = mesh_assets_root / object_id / f"{object_id}.glb"
    destination_glb_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_glb_path, destination_glb_path)
    return destination_glb_path.relative_to(mesh_assets_root.parent).as_posix()


def _gym_object_config(
    *,
    scene_object: Table | Asset,
    asset_relative_path: str,
    body_type: str,
    attrs: dict[str, float | int],
    max_convex_hull_num: int,
) -> dict[str, object]:
    """Build one z-up gym object config from a final y-up scene object."""
    pos_y_up = _scene_vector(scene_object, "pos")
    rot_y_up = _scene_vector(scene_object, "rot")
    scale_y_up = _scene_vector(scene_object, "scale")

    pos_z_up = _Y_UP_TO_Z_UP_ROTATION @ np.asarray(pos_y_up, dtype=float)
    rotation_y_up = Rotation.from_euler("xyz", rot_y_up, degrees=True).as_matrix()
    rotation_z_up = _Y_UP_TO_Z_UP_ROTATION @ rotation_y_up @ _Y_UP_TO_Z_UP_ROTATION.T
    rot_z_up = Rotation.from_matrix(rotation_z_up).as_euler(
        # RigidObjectCfg.init_rot is interpreted with uppercase XYZ.
        "XYZ",
        degrees=True,
    )

    return {
        "uid": scene_object.id,
        "description": scene_object.description,
        "shape": {
            "shape_type": "Mesh",
            "fpath": asset_relative_path,
            "compute_uv": False,
        },
        "attrs": attrs,
        "body_type": body_type,
        "init_pos": pos_z_up.tolist(),
        "init_rot": rot_z_up.tolist(),
        # Do not permute this scale: it belongs to the original y-up GLB, which
        # SimulationManager itself converts to z-up.
        "body_scale": scale_y_up,
        "max_convex_hull_num": max_convex_hull_num,
    }


def _scene_vector(scene_object: Table | Asset, field_name: str) -> list[float]:
    """Read one finite final y-up layout vector from a scene object."""
    values = getattr(scene_object, field_name)
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError(
            f"Scene object {scene_object.id!r} has no final {field_name!r} vector."
        )
    vector = [float(value) for value in values]
    if not np.all(np.isfinite(vector)):
        raise ValueError(
            f"Scene object {scene_object.id!r} has non-finite {field_name!r}."
        )
    return vector


def _positive_int(value: int, *, field_name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{field_name} must be positive.")
    return result
