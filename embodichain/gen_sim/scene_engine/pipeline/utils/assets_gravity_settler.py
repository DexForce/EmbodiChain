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

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from embodichain.gen_sim.scene_engine.core.scene import Scene
from embodichain.gen_sim.scene_engine.core.scene_object import (
    ObjectPhysics,
    SceneObject,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_generation_utils import (
    layout_object_to_transform_matrix,
    load_glb_mesh,
    transform_matrix_to_layout_object,
)
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import RigidBodyAttributesCfg, RigidObjectCfg
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.utils.logger import log_info


@dataclass(frozen=True)
class AssetsGravitySettlerConfig:
    """Physics controls for table-top asset settling."""

    clearance_m: float = 0.02  # Initial gap between each asset and the table top.
    settle_steps: int = 300  # Fixed number of simulator steps to execute.
    physics_dt: float = 1.0 / 100.0  # Physics timestep in seconds.
    sim_device: str = "cpu"  # Simulation device requested from EmbodiChain Lab.


class AssetsGravitySettler:
    """Settle all assets together on one kinematic table in a z-up simulation."""

    def __init__(
        self,
        *,
        scene: Scene,
        table_layout: dict[str, object],
        assets_layout: list[dict[str, object]],
        geometry_root: str | Path,
        config: AssetsGravitySettlerConfig | None = None,
    ) -> None:
        self.scene = scene
        self.table_layout = table_layout
        self.assets_layout = assets_layout
        self.geometry_root = Path(geometry_root).expanduser().resolve()
        self.settled_assets_layout: list[dict[str, object]] | None = None
        self.config = config if config is not None else AssetsGravitySettlerConfig()
        # Check.
        if self.config.clearance_m < 0.0:
            raise ValueError("Gravity-settle clearance_m must be non-negative.")
        if self.config.settle_steps <= 0:
            raise ValueError("Gravity-settle settle_steps must be positive.")
        if self.config.physics_dt <= 0.0:
            raise ValueError("Gravity-settle physics_dt must be positive.")

    def settle(self) -> list[dict[str, object]]:
        """Run gravity settling and return the resulting y-up asset layouts."""
        self.settled_assets_layout = None
        if not self.assets_layout:
            self.settled_assets_layout = []
            log_info("Scene has no movable assets; skipping gravity settling.")
            return self.settled_assets_layout

        table_id = self._require_layout_id(self.table_layout, name="Table")
        table_object = self._require_scene_object(table_id, kind="table")
        asset_ids: set[str] = set()
        asset_objects_by_id: dict[str, SceneObject] = {}
        for asset_layout in self.assets_layout:
            asset_id = self._require_layout_id(asset_layout, name="Asset")
            if asset_id in asset_ids:
                raise ValueError(f"Asset layouts contain duplicate id {asset_id!r}.")
            asset_ids.add(asset_id)
            asset_objects_by_id[asset_id] = self._require_scene_object(
                asset_id, kind="asset"
            )
        expected_asset_ids = {asset.id for asset in self.scene.assets}
        if asset_ids != expected_asset_ids:
            raise ValueError(
                "Gravity-settle layouts must contain exactly the scene asset ids."
            )

        y_up_to_z_up_matrix = np.eye(4)
        y_up_to_z_up_matrix[:3, :3] = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
        )
        z_up_to_y_up_matrix = np.linalg.inv(y_up_to_z_up_matrix)
        table_info = self._prepare_sim_body(
            layout_object=self.table_layout,
            y_up_to_z_up_matrix=y_up_to_z_up_matrix,
        )
        table_world_mesh = self._mesh_to_z_up_world_for_aabb(
            y_up_mesh=table_info["mesh"],
            z_up_rigid_layout=table_info["rigid_layout"],
            z_up_scale=table_info["z_up_scale"],
            y_up_to_z_up_matrix=y_up_to_z_up_matrix,
        )
        table_top_z = float(table_world_mesh.bounds[1, 2])

        prepared_assets: dict[str, dict[str, object]] = {}
        for asset_layout in self.assets_layout:
            asset_id = str(asset_layout["id"])
            asset_info = self._prepare_sim_body(
                layout_object=asset_layout,
                y_up_to_z_up_matrix=y_up_to_z_up_matrix,
            )
            asset_world_mesh = self._mesh_to_z_up_world_for_aabb(
                y_up_mesh=asset_info["mesh"],
                z_up_rigid_layout=asset_info["rigid_layout"],
                z_up_scale=asset_info["z_up_scale"],
                y_up_to_z_up_matrix=y_up_to_z_up_matrix,
            )
            asset_bottom_z = float(asset_world_mesh.bounds[0, 2])
            asset_info["rigid_layout"]["pos"][2] += (
                table_top_z + self.config.clearance_m - asset_bottom_z
            )
            prepared_assets[asset_id] = asset_info

        log_info(
            "Gravity settling started: "
            f"assets={len(prepared_assets)}, steps={self.config.settle_steps}, "
            f"physics_dt={self.config.physics_dt:.4f} s."
        )
        sim = SimulationManager(
            SimulationManagerCfg(
                headless=True,
                physics_dt=self.config.physics_dt,
                sim_device=self.config.sim_device,
            )
        )
        try:
            # Add table.
            sim.add_rigid_object(
                RigidObjectCfg(
                    uid=table_id,
                    shape=MeshCfg(fpath=str(table_info["mesh_path"])),
                    init_pos=tuple(table_info["rigid_layout"]["pos"]),
                    init_rot=tuple(
                        self._simulation_euler_xyz_degrees(table_info["rigid_layout"])
                    ),
                    body_scale=tuple(table_info["y_up_scale"]),
                    attrs=self._rigid_body_attrs(table_object.physics),
                    body_type=table_object.physics.body_type,
                    max_convex_hull_num=table_object.physics.max_convex_hull_num,
                    acd_method="vhacd",
                )
            )
            # Add assets.
            simulated_assets: dict[str, object] = {}
            for asset_id, asset_info in prepared_assets.items():
                rigid_layout = asset_info["rigid_layout"]
                simulated_assets[asset_id] = sim.add_rigid_object(
                    RigidObjectCfg(
                        uid=asset_id,
                        shape=MeshCfg(fpath=str(asset_info["mesh_path"])),
                        init_pos=tuple(rigid_layout["pos"]),
                        init_rot=tuple(
                            self._simulation_euler_xyz_degrees(rigid_layout)
                        ),
                        body_scale=tuple(asset_info["y_up_scale"]),
                        attrs=self._rigid_body_attrs(
                            asset_objects_by_id[asset_id].physics
                        ),
                        body_type=asset_objects_by_id[asset_id].physics.body_type,
                        max_convex_hull_num=(
                            asset_objects_by_id[asset_id].physics.max_convex_hull_num
                        ),
                        acd_method="vhacd",
                    )
                )
            # Run simulation to settle all assets.
            sim.update(step=self.config.settle_steps)

            # Update the final layouts.
            settled_layout_by_id: dict[str, dict[str, object]] = {}
            for asset_id, simulated_asset in simulated_assets.items():
                final_rigid_pose_z_up = np.asarray(
                    simulated_asset.get_local_pose(to_matrix=True)[0]
                    .detach()
                    .cpu()
                    .numpy(),
                    dtype=float,
                )
                scale_matrix = np.eye(4)
                scale_matrix[:3, :3] = np.diag(prepared_assets[asset_id]["z_up_scale"])
                final_z_up_layout_matrix = final_rigid_pose_z_up @ scale_matrix
                settled_layout_by_id[asset_id] = transform_matrix_to_layout_object(
                    asset_id,
                    z_up_to_y_up_matrix
                    @ final_z_up_layout_matrix
                    @ y_up_to_z_up_matrix,
                )
        finally:
            # Release resources.
            sim.destroy(exit_process=False)
            SimulationManager.flush_cleanup_queue()

        self.settled_assets_layout = [
            settled_layout_by_id[str(asset_layout["id"])]
            for asset_layout in self.assets_layout
        ]
        log_info("Gravity settling completed for all assets.")
        return self.settled_assets_layout

    def _prepare_sim_body(
        self,
        *,
        layout_object: dict[str, object],
        y_up_to_z_up_matrix: np.ndarray,
    ) -> dict[str, object]:
        """Load one y-up GLB and prepare its z-up simulation pose."""
        object_id = self._require_layout_id(layout_object, name="Layout object")
        source_mesh_path = self.geometry_root / f"{object_id}.glb"
        source_mesh = load_glb_mesh(source_mesh_path)
        z_up_layout = self._convert_layout_coordinate_system(
            layout_object,
            source_to_target_matrix=y_up_to_z_up_matrix,
        )
        return {
            "mesh_path": source_mesh_path,
            "mesh": source_mesh,
            "rigid_layout": {
                "id": object_id,
                "rot": self._three_floats(z_up_layout.get("rot"), field_name="rot"),
                "pos": self._three_floats(z_up_layout.get("pos"), field_name="pos"),
                "scale": [1.0, 1.0, 1.0],
            },
            "y_up_scale": self._three_floats(
                layout_object.get("scale"), field_name="scale"
            ),
            "z_up_scale": self._three_floats(
                z_up_layout.get("scale"), field_name="scale"
            ),
        }

    def _require_scene_object(self, object_id: str, *, kind: str) -> SceneObject:
        """Return one physics-ready scene object with the expected semantic kind."""
        matching_objects = [
            scene_object
            for scene_object in self.scene.objects
            if scene_object.id == object_id
        ]
        if len(matching_objects) != 1:
            raise ValueError(
                f"Gravity settling requires exactly one scene object {object_id!r}."
            )
        scene_object = matching_objects[0]
        if scene_object.kind != kind:
            raise ValueError(
                f"Scene object {object_id!r} must have kind {kind!r} before "
                "gravity settling."
            )
        if scene_object.physics is None:
            raise ValueError(
                f"Scene object {object_id!r} has no SimReady physics settings."
            )
        return scene_object

    @staticmethod
    def _rigid_body_attrs(physics: ObjectPhysics | None) -> RigidBodyAttributesCfg:
        """Convert persisted SceneObject physics attributes into Lab config."""
        if physics is None:
            raise ValueError("Gravity settling requires SimReady physics settings.")
        return RigidBodyAttributesCfg(**physics.attrs)

    @staticmethod
    def _mesh_to_z_up_world_for_aabb(
        *,
        y_up_mesh: trimesh.Trimesh,
        z_up_rigid_layout: dict[str, object],
        z_up_scale: Sequence[float],
        y_up_to_z_up_matrix: np.ndarray,
    ) -> trimesh.Trimesh:
        """Transform a y-up mesh into its z-up world pose for AABB measurement."""
        mesh = y_up_mesh.copy()
        mesh.apply_transform(y_up_to_z_up_matrix)
        scale_matrix = np.eye(4)
        scale_matrix[:3, :3] = np.diag(z_up_scale)
        mesh.apply_transform(scale_matrix)
        mesh.apply_transform(layout_object_to_transform_matrix(z_up_rigid_layout))
        return mesh

    @staticmethod
    def _simulation_euler_xyz_degrees(layout_object: dict[str, object]) -> list[float]:
        """Convert lowercase-xyz layout rotation to SimulationManager's XYZ order."""
        layout_rotation = Rotation.from_euler(
            "xyz",
            AssetsGravitySettler._three_floats(
                layout_object.get("rot"), field_name="rot"
            ),
            degrees=True,
        )
        return layout_rotation.as_euler("XYZ", degrees=True).tolist()

    @staticmethod
    def _convert_layout_coordinate_system(
        layout_object: dict[str, object],
        *,
        source_to_target_matrix: np.ndarray,
    ) -> dict[str, object]:
        """Convert one layout object between coordinate frames through its matrix."""
        return transform_matrix_to_layout_object(
            str(layout_object["id"]),
            source_to_target_matrix
            @ layout_object_to_transform_matrix(layout_object)
            @ np.linalg.inv(source_to_target_matrix),
        )

    @staticmethod
    def _require_layout_id(layout_object: dict[str, object], *, name: str) -> str:
        """Check id."""
        object_id = layout_object.get("id")
        if not isinstance(object_id, str) or not object_id:
            raise ValueError(f"{name} layout must contain a non-empty string id.")
        return object_id

    @staticmethod
    def _three_floats(value: object, *, field_name: str) -> list[float]:
        """Check three values."""
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError(f"Layout field {field_name} must contain three values.")
        try:
            return [float(item) for item in value]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Layout field {field_name} must contain numeric values."
            ) from exc
