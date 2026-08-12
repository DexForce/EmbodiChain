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
    """Write one generated scene and its SimReady meshes as a scene export."""

    def __init__(
        self,
        *,
        scene: Scene,
        scene_graph: SceneGraph,
        output_root: str | Path,
    ) -> None:
        self.scene = scene
        self.scene_graph = scene_graph
        self.output_root = Path(output_root).expanduser().resolve()
        self.export_root = self.output_root / "scene_export"
        self.scene_config_path: Path | None = None
        self.scene_graph_path: Path | None = None

    def export(self) -> Path:
        """Write a scene-only config and copy SimReady GLBs into ``mesh_assets``.

        Scene layouts are y-up. The simulator automatically converts each y-up
        GLB to z-up, so this exporter copies each GLB unchanged and converts
        only its world position and rotation for ``init_pos`` and ``init_rot``.
        ``body_scale`` remains the original y-up scale associated with the GLB.
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
                )
            ],
            "rigid_object": [
                self._scene_object_config(
                    scene_object=asset,
                    asset_relative_path=exported_entries[asset.id],
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
        shutil.copy2(source_glb_path, destination_glb_path)
        return destination_glb_path.relative_to(mesh_assets_root.parent).as_posix()

    @staticmethod
    def _scene_object_config(
        *,
        scene_object: SceneObject,
        asset_relative_path: str,
    ) -> dict[str, object]:
        """Build one z-up scene-only object config from a final y-up object."""
        pos_y_up = SceneExporter._scene_vector(scene_object, "pos")
        rot_y_up = SceneExporter._scene_vector(scene_object, "rot")
        scale_y_up = SceneExporter._scene_vector(scene_object, "scale")
        if scene_object.physics is None:
            raise ValueError(
                f"Scene object {scene_object.id!r} has no SimReady physics settings."
            )

        pos_z_up = _Y_UP_TO_Z_UP_ROTATION @ np.asarray(pos_y_up, dtype=float)
        rotation_y_up = Rotation.from_euler("xyz", rot_y_up, degrees=True).as_matrix()
        rotation_z_up = (
            _Y_UP_TO_Z_UP_ROTATION @ rotation_y_up @ _Y_UP_TO_Z_UP_ROTATION.T
        )
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
            "attrs": scene_object.physics.attrs,
            "body_type": scene_object.physics.body_type,
            "init_pos": pos_z_up.tolist(),
            "init_rot": rot_z_up.tolist(),
            # Do not permute this scale: it belongs to the original y-up GLB,
            # which SimulationManager itself converts to z-up.
            "body_scale": scale_y_up,
            "center_xy": scene_object.center_xy,
            "max_convex_hull_num": scene_object.physics.max_convex_hull_num,
        }

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
