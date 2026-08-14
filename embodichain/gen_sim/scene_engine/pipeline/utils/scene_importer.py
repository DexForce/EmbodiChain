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
from embodichain.gen_sim.scene_engine.core.scene_graph import (
    SceneGraph,
    SceneGraphNode,
    SceneGraphRelation,
    TABLE_REGIONS,
)
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
        self.scene_graph_path = self.scene_export_root / "scene_graph.json"
        self.scene_json_path = self.scene_export_root / "scene.json"

    def import_scene(self) -> Scene:
        """Validate the scene export, write ``scene.json``, and return a ``Scene``."""
        scene = self._load_scene()
        self._write_scene_json(scene)
        return scene

    def import_scene_and_graph(self) -> tuple[Scene, SceneGraph]:
        """Import a scene and graph after validating the complete edit input."""
        scene = self._load_scene()
        scene_graph = self._load_scene_graph()
        if set(scene_graph.node_by_id()) != {
            scene_object.id for scene_object in scene.objects
        }:
            raise ValueError("Scene graph nodes must match imported scene object ids.")
        self._write_scene_json(scene)
        return scene, scene_graph

    def _load_scene(self) -> Scene:
        """Validate the exported scene files and restore the ``Scene`` data."""
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

        return self._scene_from_config(scene_config)

    def _load_scene_graph(self) -> SceneGraph:
        """Read and validate the exported scene graph."""
        if not self.scene_graph_path.is_file():
            raise FileNotFoundError(f"Scene graph not found: {self.scene_graph_path}")
        try:
            scene_graph_data = json.loads(
                self.scene_graph_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Scene graph is not valid JSON: {self.scene_graph_path}"
            ) from exc
        return self._scene_graph_from_data(scene_graph_data)

    def _write_scene_json(self, scene: Scene) -> None:
        """Write the restored scene debugging artifact after validation succeeds."""
        self.scene_json_path.write_text(
            json.dumps(scene.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        log_info(f"Imported scene JSON: {self.scene_json_path}")

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

    @staticmethod
    def _scene_graph_from_data(value: object) -> SceneGraph:
        """Build a validated ``SceneGraph`` from exported graph JSON."""
        if not isinstance(value, dict) or set(value) != {"nodes", "relations"}:
            raise ValueError("Scene graph must contain exactly nodes and relations.")
        nodes_value = value["nodes"]
        relations_value = value["relations"]
        if not isinstance(nodes_value, list) or not isinstance(relations_value, list):
            raise ValueError("Scene graph nodes and relations must be lists.")

        nodes = [
            SceneExportImporter._scene_graph_node_from_data(node)
            for node in nodes_value
        ]
        relations = [
            SceneExportImporter._scene_graph_relation_from_data(relation)
            for relation in relations_value
        ]
        return SceneGraph(nodes=nodes, relations=relations)

    @staticmethod
    def _scene_graph_node_from_data(value: object) -> SceneGraphNode:
        if not isinstance(value, dict) or set(value) != {
            "object_id",
            "parent_id",
            "parent_relation",
            "table_region",
        }:
            raise ValueError("Scene graph nodes must use the serialized node schema.")
        object_id = value["object_id"]
        parent_id = value["parent_id"]
        parent_relation = value["parent_relation"]
        table_region = value["table_region"]
        if not isinstance(object_id, str) or not isinstance(
            parent_id, (str, type(None))
        ):
            raise ValueError("Scene graph node ids must be strings or null.")
        if parent_relation not in {None, "on"}:
            raise ValueError("Scene graph parent_relation must be 'on' or null.")
        if table_region is not None and table_region not in TABLE_REGIONS:
            raise ValueError("Scene graph table_region is invalid.")
        return SceneGraphNode(
            object_id=object_id,
            parent_id=parent_id,
            parent_relation=parent_relation,
            table_region=table_region,
        )

    @staticmethod
    def _scene_graph_relation_from_data(value: object) -> SceneGraphRelation:
        if not isinstance(value, dict) or set(value) != {
            "source_id",
            "relation",
            "target_id",
        }:
            raise ValueError(
                "Scene graph relations must use the serialized relation schema."
            )
        source_id = value["source_id"]
        relation = value["relation"]
        target_id = value["target_id"]
        if not isinstance(source_id, str) or not isinstance(target_id, str):
            raise ValueError("Scene graph relation ids must be strings.")
        if relation not in {"left_of", "right_of", "in_front_of", "behind"}:
            raise ValueError("Scene graph relation is invalid.")
        return SceneGraphRelation(
            source_id=source_id,
            relation=relation,
            target_id=target_id,
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
        center_xy = entry.get("center_xy")
        if center_xy is not None:
            center_xy = self._vector2(center_xy, field_name=f"{uid}.center_xy")
        support_surface_z = entry.get("support_surface_z")
        if support_surface_z is not None:
            support_surface_z = float(support_surface_z)
        support_contour_xy = self._points2(
            entry.get("support_contour_xy"), field_name=f"{uid}.support_contour_xy"
        )
        support_optimization_rect_xy = self._points2(
            entry.get("support_optimization_rect_xy"),
            field_name=f"{uid}.support_optimization_rect_xy",
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
            category=self._semantic_text(
                entry.get("category"),
                field_name=f"{uid}.category",
                default=uid,
            ),
            name=self._semantic_text(
                entry.get("name"),
                field_name=f"{uid}.name",
                default=uid,
            ),
            description=str(entry.get("description") or uid),
            simready_glb_path=str(glb_path),
            rot=rot_y_up.tolist(),
            pos=pos_y_up.tolist(),
            scale=scale,
            center_xy=center_xy,
            support_surface_z=support_surface_z,
            support_contour_xy=support_contour_xy,
            support_optimization_rect_xy=support_optimization_rect_xy,
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
        expected_fpath = Path("mesh_assets") / uid / f"{uid}.glb"
        if fpath != expected_fpath:
            raise ValueError(
                f"Scene object {uid!r} shape.fpath must be {expected_fpath.as_posix()!r}."
            )
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
    def _semantic_text(
        value: object,
        *,
        field_name: str,
        default: str,
    ) -> str:
        """Read one non-empty semantic label with a legacy-export fallback."""
        if value is None:
            return default
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Scene config field {field_name!r} must be non-empty.")
        return value

    @staticmethod
    def _vector2(value: object, *, field_name: str) -> list[float]:
        """Validate one length-2 numeric vector."""
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(
                f"Scene config field {field_name!r} must be a length-2 list."
            )
        vector = [float(item) for item in value]
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"Scene config field {field_name!r} must be finite.")
        return vector

    @classmethod
    def _points2(cls, value: object, *, field_name: str) -> list[list[float]] | None:
        """Validate an optional list of XY points from the scene export."""
        if value is None:
            return None
        if not isinstance(value, list) or len(value) < 3:
            raise ValueError(
                f"Scene config field {field_name!r} must contain 3 points."
            )
        return [
            cls._vector2(point, field_name=f"{field_name}[{index}]")
            for index, point in enumerate(value)
        ]

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
