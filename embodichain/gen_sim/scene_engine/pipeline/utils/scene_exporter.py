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

from embodichain.gen_sim.scene_engine.core.scene import Scene
from embodichain.gen_sim.scene_engine.core.scene_graph import SceneGraph
from embodichain.gen_sim.scene_engine.core.scene_object import SceneObject
from embodichain.utils.logger import log_info

_Y_UP_TO_Z_UP_ROTATION = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)


class SceneExporter:
    """Write one generated scene and its SimReady meshes as a scene export.

    By default, the complete z-up scene is rotated 180 degrees around the
    table center before serialization. The input ``Scene`` is not mutated.
    """

    def __init__(
        self,
        *,
        scene: Scene,
        scene_graph: SceneGraph,
        output_root: str | Path,
        rotate_z_up_180: bool = True,
    ) -> None:
        self.scene = scene
        self.scene_graph = scene_graph
        self.output_root = Path(output_root).expanduser().resolve()
        self.rotate_z_up_180 = rotate_z_up_180  # Keep the legacy frame on request.
        self.export_root = self.output_root / "scene_export"
        self.scene_config_path: Path | None = None
        self.scene_graph_path: Path | None = None
        self.scene_json_path: Path | None = None

    def export(self) -> Path:
        """Write a scene-only config and copy SimReady GLBs into ``mesh_assets``.

        Scene layouts are y-up. The simulator automatically converts each y-up
        GLB to z-up, so this exporter copies each GLB unchanged and converts
        only its world position and rotation for ``init_pos`` and ``init_rot``.
        The default applies one additional 180-degree global z-up rotation about
        the table center to every object and XY layout metadata. ``body_scale``
        remains the original y-up scale associated with the GLB.
        This is not a complete ``EmbodiedEnv``/``run-env`` configuration because
        a generated scene does not determine a robot, its placement, or control.
        """
        if self.scene.table is None:
            raise ValueError("Cannot export a scene without a table.")

        mesh_assets_root = self.export_root / "mesh_assets"
        mesh_assets_root.mkdir(parents=True, exist_ok=True)
        scene_objects = self.scene.objects
        object_ids = [scene_object.id for scene_object in scene_objects]
        if len(set(object_ids)) != len(object_ids):
            raise ValueError("Scene export requires unique table and asset ids.")
        self.scene_graph.validate()
        if set(self.scene_graph.node_by_id()) != set(object_ids):
            raise ValueError("Scene graph nodes must match exported scene object ids.")

        z_up_rotation, z_up_pivot_xy = self._z_up_export_transform()

        exported_entries = {
            scene_object.id: self._copy_scene_object_to_assets(
                scene_object=scene_object,
                mesh_assets_root=mesh_assets_root,
            )
            for scene_object in scene_objects
        }
        scene_config = {
            "format": "embodichain.scene-export/v1",
            # This identifies the exported scene data only. It is deliberately not
            # a Gymnasium environment ID because scene exports do not register or
            # instantiate an EmbodiedEnv.
            "scene_id": f"scene-engine-{int(time.time() * 1000)}",
            "background": [
                self._scene_object_config(
                    scene_object=self.scene.table,
                    asset_relative_path=exported_entries[self.scene.table.id],
                    z_up_rotation=z_up_rotation,
                    z_up_pivot_xy=z_up_pivot_xy,
                )
            ],
            "rigid_object": [
                self._scene_object_config(
                    scene_object=asset,
                    asset_relative_path=exported_entries[asset.id],
                    z_up_rotation=z_up_rotation,
                    z_up_pivot_xy=z_up_pivot_xy,
                )
                for asset in self.scene.assets
            ],
        }
        self.scene_config_path = self.export_root / "scene_config.json"
        self.scene_config_path.write_text(
            json.dumps(scene_config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        log_info(f"Exported scene config: {self.scene_config_path}")
        self.scene_graph_path = self.export_root / "scene_graph.json"
        self.scene_graph_path.write_text(
            json.dumps(self.scene_graph.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        log_info(f"Exported scene graph: {self.scene_graph_path}")
        self.scene_json_path = self.export_root / "scene.json"
        self.scene_json_path.write_text(
            json.dumps(
                {
                    "objects": [
                        self._scene_object_y_up_dict(
                            scene_object=scene_object,
                            z_up_rotation=z_up_rotation,
                            z_up_pivot_xy=z_up_pivot_xy,
                        )
                        for scene_object in scene_objects
                    ]
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        log_info(f"Exported scene JSON: {self.scene_json_path}")
        # Remove only assets absent from the completed scene export.
        self._remove_stale_mesh_assets(
            mesh_assets_root=mesh_assets_root,
            object_ids=set(object_ids),
        )
        return self.scene_config_path

    @staticmethod
    def _copy_scene_object_to_assets(
        *,
        scene_object: SceneObject,
        mesh_assets_root: Path,
    ) -> str:
        """Copy one referenced SimReady GLB and return its config-relative path."""
        object_id = scene_object.id
        if (
            Path(object_id).name != object_id
            or "\\" in object_id
            or object_id in {"", ".", ".."}
        ):
            raise ValueError(
                f"Scene object id is not safe for a GLB filename: {object_id!r}"
            )
        if scene_object.simready_glb_path is None:
            raise ValueError(f"Scene object {object_id!r} has no SimReady GLB path.")

        source_glb_path = Path(scene_object.simready_glb_path).expanduser().resolve()
        if not source_glb_path.is_file():
            raise FileNotFoundError(
                "SimReady GLB for scene object "
                f"{object_id!r} not found: {source_glb_path}"
            )
        destination_glb_path = mesh_assets_root / object_id / f"{object_id}.glb"
        destination_glb_path.parent.mkdir(parents=True, exist_ok=True)
        # Imported assets already live at their export destination.
        if not destination_glb_path.is_file() or not source_glb_path.samefile(
            destination_glb_path
        ):
            shutil.copy2(source_glb_path, destination_glb_path)
        return destination_glb_path.relative_to(mesh_assets_root.parent).as_posix()

    @staticmethod
    def _remove_stale_mesh_assets(
        *,
        mesh_assets_root: Path,
        object_ids: set[str],
    ) -> None:
        """Remove copied asset directories that no longer belong to the scene."""
        for asset_root in mesh_assets_root.iterdir():
            if asset_root.name in object_ids:
                continue
            if asset_root.is_dir():
                shutil.rmtree(asset_root)
            else:
                asset_root.unlink()

    def _z_up_export_transform(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the optional global z-up rotation and its table-center pivot."""
        table = self.scene.table
        if table is None:
            raise ValueError("Cannot transform a scene export without a table.")
        table_pos_z_up, _ = self._final_z_up_pose(
            scene_object=table,
            z_up_rotation=np.eye(3),
            z_up_pivot_xy=np.zeros(2),
        )
        z_up_rotation = (
            Rotation.from_euler("z", 180.0, degrees=True).as_matrix()
            if self.rotate_z_up_180
            else np.eye(3)
        )
        return z_up_rotation, table_pos_z_up[:2]

    @staticmethod
    def _scene_object_config(
        *,
        scene_object: SceneObject,
        asset_relative_path: str,
        z_up_rotation: np.ndarray,
        z_up_pivot_xy: np.ndarray,
    ) -> dict[str, object]:
        """Build one z-up scene-only object config from a final y-up object."""
        scale_y_up = SceneExporter._scene_vector(scene_object, "scale")
        if scene_object.physics is None:
            raise ValueError(
                f"Scene object {scene_object.id!r} has no SimReady physics settings."
            )

        pos_z_up, rotation_z_up = SceneExporter._final_z_up_pose(
            scene_object=scene_object,
            z_up_rotation=z_up_rotation,
            z_up_pivot_xy=z_up_pivot_xy,
        )
        rot_z_up = Rotation.from_matrix(rotation_z_up).as_euler(
            # RigidObjectCfg.init_rot is interpreted with uppercase XYZ.
            "XYZ",
            degrees=True,
        )

        return {
            "uid": scene_object.id,
            "category": scene_object.category,
            "name": scene_object.name,
            "description": scene_object.description,
            "shape": {
                "shape_type": "Mesh",
                "fpath": asset_relative_path,
                "compute_uv": False,
            },
            "attrs": scene_object.physics.attrs,
            "body_type": scene_object.physics.body_type,
            "init_pos": pos_z_up.tolist(),
            "init_rot": rot_z_up.tolist(),
            # Do not permute this scale: it belongs to the original y-up GLB,
            # which SimulationManager itself converts to z-up.
            "body_scale": scale_y_up,
            "center_xy": SceneExporter._transformed_optional_xy(
                scene_object=scene_object,
                field_name="center_xy",
                z_up_rotation=z_up_rotation,
                z_up_pivot_xy=z_up_pivot_xy,
            ),
            "support_surface_z": scene_object.support_surface_z,
            "support_contour_xy": SceneExporter._transformed_optional_xy_points(
                scene_object=scene_object,
                field_name="support_contour_xy",
                z_up_rotation=z_up_rotation,
                z_up_pivot_xy=z_up_pivot_xy,
            ),
            "support_optimization_rect_xy": (
                SceneExporter._transformed_optional_xy_points(
                    scene_object=scene_object,
                    field_name="support_optimization_rect_xy",
                    z_up_rotation=z_up_rotation,
                    z_up_pivot_xy=z_up_pivot_xy,
                )
            ),
            "max_convex_hull_num": scene_object.physics.max_convex_hull_num,
        }

    @staticmethod
    def _final_z_up_pose(
        *,
        scene_object: SceneObject,
        z_up_rotation: np.ndarray,
        z_up_pivot_xy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert one y-up pose and apply the export's global z-up rotation."""
        pos_y_up = SceneExporter._scene_vector(scene_object, "pos")
        rot_y_up = SceneExporter._scene_vector(scene_object, "rot")
        pos_z_up = _Y_UP_TO_Z_UP_ROTATION @ np.asarray(pos_y_up, dtype=float)
        pos_z_up[:2] = z_up_pivot_xy + z_up_rotation[:2, :2] @ (
            pos_z_up[:2] - z_up_pivot_xy
        )
        rotation_y_up = Rotation.from_euler("xyz", rot_y_up, degrees=True).as_matrix()
        rotation_z_up = z_up_rotation @ (
            _Y_UP_TO_Z_UP_ROTATION @ rotation_y_up @ _Y_UP_TO_Z_UP_ROTATION.T
        )
        return pos_z_up, rotation_z_up

    @staticmethod
    def _scene_object_y_up_dict(
        *,
        scene_object: SceneObject,
        z_up_rotation: np.ndarray,
        z_up_pivot_xy: np.ndarray,
    ) -> dict[str, object]:
        """Serialize the globally rotated export without mutating the input scene."""
        pos_z_up, rotation_z_up = SceneExporter._final_z_up_pose(
            scene_object=scene_object,
            z_up_rotation=z_up_rotation,
            z_up_pivot_xy=z_up_pivot_xy,
        )
        result = scene_object.to_dict()
        result["pos"] = (_Y_UP_TO_Z_UP_ROTATION.T @ pos_z_up).tolist()
        result["rot"] = (
            Rotation.from_matrix(
                _Y_UP_TO_Z_UP_ROTATION.T @ rotation_z_up @ _Y_UP_TO_Z_UP_ROTATION
            )
            .as_euler("xyz", degrees=True)
            .tolist()
        )
        result["center_xy"] = SceneExporter._transformed_optional_xy(
            scene_object=scene_object,
            field_name="center_xy",
            z_up_rotation=z_up_rotation,
            z_up_pivot_xy=z_up_pivot_xy,
        )
        result["support_contour_xy"] = SceneExporter._transformed_optional_xy_points(
            scene_object=scene_object,
            field_name="support_contour_xy",
            z_up_rotation=z_up_rotation,
            z_up_pivot_xy=z_up_pivot_xy,
        )
        result["support_optimization_rect_xy"] = (
            SceneExporter._transformed_optional_xy_points(
                scene_object=scene_object,
                field_name="support_optimization_rect_xy",
                z_up_rotation=z_up_rotation,
                z_up_pivot_xy=z_up_pivot_xy,
            )
        )
        return result

    @staticmethod
    def _transformed_optional_xy(
        *,
        scene_object: SceneObject,
        field_name: str,
        z_up_rotation: np.ndarray,
        z_up_pivot_xy: np.ndarray,
    ) -> list[float] | None:
        """Rotate one optional z-up XY metadata point around the table center."""
        value = getattr(scene_object, field_name)
        if value is None:
            return None
        point = np.asarray(value, dtype=float)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError(
                f"Scene object {scene_object.id!r} has invalid {field_name!r} metadata."
            )
        return (
            z_up_pivot_xy + z_up_rotation[:2, :2] @ (point - z_up_pivot_xy)
        ).tolist()

    @staticmethod
    def _transformed_optional_xy_points(
        *,
        scene_object: SceneObject,
        field_name: str,
        z_up_rotation: np.ndarray,
        z_up_pivot_xy: np.ndarray,
    ) -> list[list[float]] | None:
        """Rotate optional z-up XY support geometry around the table center."""
        value = getattr(scene_object, field_name)
        if value is None:
            return None
        points = np.asarray(value, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
            raise ValueError(
                f"Scene object {scene_object.id!r} has invalid {field_name!r} metadata."
            )
        return (
            z_up_pivot_xy + (z_up_rotation[:2, :2] @ (points - z_up_pivot_xy).T).T
        ).tolist()

    @staticmethod
    def _scene_vector(scene_object: SceneObject, field_name: str) -> list[float]:
        """Read one finite final y-up layout vector from a scene object."""
        values = getattr(scene_object, field_name)
        if not isinstance(values, list) or len(values) != 3:
            raise ValueError(
                f"Scene object {scene_object.id!r} has no final "
                f"{field_name!r} vector."
            )
        vector = [float(value) for value in values]
        if not np.all(np.isfinite(vector)):
            raise ValueError(
                f"Scene object {scene_object.id!r} has non-finite " f"{field_name!r}."
            )
        return vector
