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
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from embodichain.gen_sim.scene_engine.core.scene import Scene
from embodichain.gen_sim.scene_engine.core.scene_object import (
    ObjectPhysics,
    SceneObject,
)
from embodichain.utils.logger import log_info

_Y_UP_TO_Z_UP_ROTATION = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)
_Z_UP_TO_Y_UP_ROTATION = _Y_UP_TO_Z_UP_ROTATION.T


class SceneExportImporter:
    """Import an editable ``Scene`` from an exported Scene Engine directory."""

    def __init__(
        self,
        *,
        output_root: str | Path,
    ) -> None:
        self.output_root = Path(output_root).expanduser().resolve()
        self.scene_export_root = self.output_root / "scene_export"
        self.mesh_assets_root = self.scene_export_root / "mesh_assets"
        self.scene_config_path = self.scene_export_root / "scene_config.json"
        self.scene_json_path = self.scene_export_root / "scene.json"

    def import_scene(self) -> Scene:
        """Validate the scene export, write ``scene.json``, and return a ``Scene``."""
        # Editing only runs on an existing Scene Engine output directory.
        if not self.output_root.is_dir() or not any(self.output_root.iterdir()):
            raise ValueError(
                "Output root must exist and contain files when edit_prompt is provided."
            )

        # The editor consumes the portable scene export and its copied GLB assets.
        if not self.scene_export_root.is_dir():
            raise FileNotFoundError(
                f"Scene export directory not found: {self.scene_export_root}"
            )
        if not self.mesh_assets_root.is_dir():
            raise FileNotFoundError(
                f"Scene mesh assets directory not found: {self.mesh_assets_root}"
            )
        if not self.scene_config_path.is_file():
            raise FileNotFoundError(f"Scene config not found: {self.scene_config_path}")

        try:
            scene_config = json.loads(
                self.scene_config_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Scene config is not valid JSON: {self.scene_config_path}"
            ) from exc
        if not isinstance(scene_config, dict):
            raise ValueError("Scene config must be a JSON object.")

        scene = self._scene_from_config(scene_config)
        if self.scene_json_path.exists():
            self.scene_json_path.unlink()
        self.scene_json_path.write_text(
            json.dumps(scene.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        log_info(f"Imported scene JSON: {self.scene_json_path}")
        return scene

    def _scene_from_config(self, scene_config: dict[str, Any]) -> Scene:
        """Build a y-up ``Scene`` from the z-up scene-export config."""
        # The table is the required support object for scene-edit operations.
        background = scene_config.get("background", [])
        if not isinstance(background, list):
            raise ValueError("Scene config background must be a list.")
        table_entry = next(
            (
                scene_object
                for scene_object in background
                if isinstance(scene_object, dict) and scene_object.get("uid") == "table"
            ),
            None,
        )
        if table_entry is None:
            raise ValueError("Scene config background must contain a table entry.")

        rigid_object_entries = scene_config.get("rigid_object", [])
        if not isinstance(rigid_object_entries, list):
            raise ValueError("Scene config rigid_object must be a list.")

        return Scene(
            objects=[
                self._scene_object_from_export_entry(table_entry, kind="table"),
                *[
                    self._scene_object_from_export_entry(entry, kind="asset")
                    for entry in rigid_object_entries
                ],
            ]
        )

    def _scene_object_from_export_entry(
        self,
        entry: object,
        *,
        kind: str,
    ) -> SceneObject:
        """Convert one z-up scene-export entry back to a y-up ``SceneObject``."""
        if not isinstance(entry, dict):
            raise ValueError("Scene config entries must be objects.")
        uid = entry.get("uid")
        if not isinstance(uid, str) or not uid:
            raise ValueError("Scene config entries must contain a valid uid.")

        glb_path = self._resolve_export_glb_path(entry, uid=uid)
        pos_z_up = self._vector3(
            entry.get("init_pos", [0.0, 0.0, 0.0]),
            field_name=f"{uid}.init_pos",
        )
        rot_z_up = self._vector3(
            entry.get("init_rot", [0.0, 0.0, 0.0]),
            field_name=f"{uid}.init_rot",
        )
        scale = self._vector3(
            entry.get("body_scale", [1.0, 1.0, 1.0]),
            field_name=f"{uid}.body_scale",
        )

        pos_y_up = _Z_UP_TO_Y_UP_ROTATION @ np.asarray(pos_z_up, dtype=float)
        rotation_z_up = Rotation.from_euler("XYZ", rot_z_up, degrees=True).as_matrix()
        rotation_y_up = (
            _Z_UP_TO_Y_UP_ROTATION @ rotation_z_up @ _Z_UP_TO_Y_UP_ROTATION.T
        )
        rot_y_up = Rotation.from_matrix(rotation_y_up).as_euler("xyz", degrees=True)

        return SceneObject(
            id=uid,
            kind=kind,  # type: ignore[arg-type]
            category=uid,
            name=uid,
            description=str(entry.get("description") or uid),
            simready_glb_path=str(glb_path),
            rot=rot_y_up.tolist(),
            pos=pos_y_up.tolist(),
            scale=scale,
            physics=ObjectPhysics(
                body_type=str(entry.get("body_type", "dynamic")),  # type: ignore[arg-type]
                attrs=self._physics_attrs(entry.get("attrs", {"mass": 1.0})),
                max_convex_hull_num=max(1, int(entry.get("max_convex_hull_num", 32))),
            ),
        )

    def _resolve_export_glb_path(
        self,
        entry: dict[str, Any],
        *,
        uid: str,
    ) -> Path:
        """Validate one exported mesh reference and return its absolute GLB path."""
        shape = entry.get("shape")
        if not isinstance(shape, dict) or not isinstance(shape.get("fpath"), str):
            raise ValueError(f"Scene object {uid!r} must contain shape.fpath.")
        fpath = Path(shape["fpath"])
        if fpath.is_absolute():
            raise ValueError(f"Scene object {uid!r} shape.fpath must be relative.")
        if fpath.suffix.lower() != ".glb":
            raise ValueError(f"Scene object {uid!r} shape.fpath must point to a GLB.")
        glb_path = (self.scene_export_root / fpath).resolve()
        if self.scene_export_root.resolve() not in glb_path.parents:
            raise ValueError(
                f"Scene object {uid!r} shape.fpath must stay within "
                f"{self.scene_export_root.resolve()}."
            )
        if not glb_path.is_file():
            raise FileNotFoundError(f"Scene object {uid!r} GLB not found: {glb_path}")
        return glb_path

    @staticmethod
    def _vector3(value: object, *, field_name: str) -> list[float]:
        """Validate one length-3 numeric vector."""
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError(
                f"Scene config field {field_name!r} must be a length-3 list."
            )
        vector = [float(item) for item in value]
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"Scene config field {field_name!r} must be finite.")
        return vector

    @staticmethod
    def _physics_attrs(value: object) -> dict[str, float | int]:
        """Validate exported physics attributes."""
        if not isinstance(value, dict) or not value:
            raise ValueError("Scene object attrs must be a non-empty object.")
        attrs: dict[str, float | int] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not isinstance(item, (float, int)):
                raise ValueError("Scene object attrs must map strings to numbers.")
            attrs[key] = item
        return attrs


def import_scene_from_output_root(output_root: str | Path) -> Scene:
    """Import an editable ``Scene`` from ``scene_export/scene_config.json``."""
    return SceneExportImporter(output_root=output_root).import_scene()
