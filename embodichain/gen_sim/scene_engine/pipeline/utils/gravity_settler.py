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

import numpy as np
from scipy.spatial.transform import Rotation

from embodichain.gen_sim.scene_engine.core.scene_object import (
    ObjectPhysics,
    SceneObject,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_generation_utils import (
    layout_object_to_transform_matrix,
    transform_matrix_to_layout_object,
)
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import RigidBodyAttributesCfg, RigidObjectCfg
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.utils.logger import log_info


@dataclass(frozen=True)
class GravitySettlerConfig:
    """Physics controls for one caller-defined gravity-settlement pass."""

    settle_steps: int = 300
    physics_dt: float = 1.0 / 100.0
    sim_device: str = "cpu"

    def __post_init__(self) -> None:
        """Reject invalid numerical controls before starting a simulation."""
        if self.settle_steps <= 0:
            raise ValueError("Gravity-settle settle_steps must be positive.")
        if self.physics_dt <= 0.0:
            raise ValueError("Gravity-settle physics_dt must be positive.")


@dataclass(frozen=True)
class GravitySettleBody:
    """One scene object and its latest complete y-up pipeline layout."""

    scene_object: SceneObject
    y_up_layout: dict[str, object]


class GravitySettler:
    """Settle caller-selected dynamic assets against a mandatory table body.

    All supplied layouts use Scene Engine's y-up pipeline convention. The
    settler converts them to z-up only at the simulation boundary. The caller
    explicitly classifies every participant as dynamic or static; static
    participants and the table are kinematic collision bodies.
    """

    def __init__(
        self,
        *,
        table_body: GravitySettleBody,
        participant_bodies: list[GravitySettleBody],
        dynamic_asset_ids: set[str],
        static_asset_ids: set[str],
        config: GravitySettlerConfig | None = None,
    ) -> None:
        self.table_body = table_body
        self.participant_bodies = participant_bodies
        self.dynamic_asset_ids = set(dynamic_asset_ids)
        self.static_asset_ids = set(static_asset_ids)
        self.config = config if config is not None else GravitySettlerConfig()

    def settle(self) -> dict[str, dict[str, list[float]]]:
        """Return final y-up poses for dynamic participants only.

        Input layouts are used as-is.  Placement clearance and support-surface
        alignment remain the responsibility of the calling layout optimizer.
        Static participants and every object's scale are unchanged, so they are
        deliberately omitted from the result.
        """
        # Check table.
        table = self.table_body.scene_object
        if table.kind != "table":
            raise ValueError("Gravity settling requires a table body.")
        table_id = self._require_body_layout_id(self.table_body, name="Table")

        participant_bodies_by_id: dict[str, GravitySettleBody] = {}
        for participant_body in self.participant_bodies:
            asset_id = self._require_body_layout_id(
                participant_body, name="Participant asset"
            )
            if asset_id == table_id:
                raise ValueError(
                    "Gravity-settle participants cannot include the table."
                )
            if participant_body.scene_object.kind != "asset":
                raise ValueError(
                    f"Gravity-settle participant {asset_id!r} must be an asset body."
                )
            if asset_id in participant_bodies_by_id:
                raise ValueError(
                    f"Gravity-settle participant assets repeat id {asset_id!r}."
                )
            participant_bodies_by_id[asset_id] = participant_body

        participant_ids = set(participant_bodies_by_id)
        classified_ids = self.dynamic_asset_ids | self.static_asset_ids
        if self.dynamic_asset_ids & self.static_asset_ids:
            raise ValueError(
                "Gravity-settle dynamic and static asset IDs must not overlap."
            )
        if classified_ids != participant_ids:
            raise ValueError(
                "Gravity-settle dynamic and static asset IDs must exactly match "
                f"participants; participants={sorted(participant_ids)}, "
                f"classified={sorted(classified_ids)}."
            )
        if not self.dynamic_asset_ids:
            log_info("Gravity settle has no dynamic participants; skipping simulation.")
            return {}

        y_up_to_z_up_matrix = self._y_up_to_z_up_matrix()
        z_up_to_y_up_matrix = np.linalg.inv(y_up_to_z_up_matrix)
        table_info = self._prepare_sim_body(
            body=self.table_body,
            y_up_to_z_up_matrix=y_up_to_z_up_matrix,
        )
        participant_infos_by_id = {
            asset_id: self._prepare_sim_body(
                body=participant_body,
                y_up_to_z_up_matrix=y_up_to_z_up_matrix,
            )
            for asset_id, participant_body in participant_bodies_by_id.items()
        }

        log_info(
            "Gravity settling started: "
            f"dynamic_assets={len(self.dynamic_asset_ids)}, "
            f"kinematic_assets={len(participant_infos_by_id) - len(self.dynamic_asset_ids)}, "
            f"steps={self.config.settle_steps}, "
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
            self._add_sim_body(
                sim=sim,
                object_id=table_id,
                body_info=table_info,
                physics=table.physics,
                body_type="kinematic",
            )
            simulated_assets: dict[str, object] = {}
            for asset_id, asset_info in participant_infos_by_id.items():
                simulated_assets[asset_id] = self._add_sim_body(
                    sim=sim,
                    object_id=asset_id,
                    body_info=asset_info,
                    physics=participant_bodies_by_id[asset_id].scene_object.physics,
                    body_type=(
                        "dynamic" if asset_id in self.dynamic_asset_ids else "kinematic"
                    ),
                )
            sim.prepare()
            sim.update(step=self.config.settle_steps)

            settled_pose_by_id: dict[str, dict[str, list[float]]] = {}
            for asset_id in self.dynamic_asset_ids:
                simulated_asset = simulated_assets[asset_id]
                final_rigid_pose_z_up = np.asarray(
                    simulated_asset.get_local_pose(to_matrix=True)[0]
                    .detach()
                    .cpu()
                    .numpy(),
                    dtype=float,
                )
                scale_matrix = np.eye(4)
                scale_matrix[:3, :3] = np.diag(
                    participant_infos_by_id[asset_id]["z_up_scale"]
                )
                final_y_up_layout = transform_matrix_to_layout_object(
                    asset_id,
                    z_up_to_y_up_matrix
                    @ final_rigid_pose_z_up
                    @ scale_matrix
                    @ y_up_to_z_up_matrix,
                )
                settled_pose_by_id[asset_id] = {
                    "pos": self._three_floats(
                        final_y_up_layout.get("pos"), field_name="pos"
                    ),
                    "rot": self._three_floats(
                        final_y_up_layout.get("rot"), field_name="rot"
                    ),
                }
        finally:
            sim.destroy(exit_process=False)
            SimulationManager.flush_cleanup_queue()

        log_info("Gravity settling completed for participating assets.")
        return settled_pose_by_id

    def _prepare_sim_body(
        self,
        *,
        body: GravitySettleBody,
        y_up_to_z_up_matrix: np.ndarray,
    ) -> dict[str, object]:
        """Convert one supplied y-up layout into a simulator body pose."""
        scene_object = body.scene_object
        if scene_object.simready_glb_path is None:
            raise ValueError(
                f"Gravity-settle object {scene_object.id!r} has no SimReady GLB path."
            )
        mesh_path = Path(scene_object.simready_glb_path).expanduser().resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(
                f"Gravity-settle GLB for {scene_object.id!r} not found: {mesh_path}"
            )
        y_up_layout = body.y_up_layout
        z_up_layout = self._convert_layout_coordinate_system(
            y_up_layout,
            source_to_target_matrix=y_up_to_z_up_matrix,
        )
        return {
            "mesh_path": mesh_path,
            "rigid_layout": {
                "id": scene_object.id,
                "rot": self._three_floats(z_up_layout.get("rot"), field_name="rot"),
                "pos": self._three_floats(z_up_layout.get("pos"), field_name="pos"),
                "scale": [1.0, 1.0, 1.0],
            },
            "y_up_scale": self._three_floats(
                y_up_layout.get("scale"), field_name="scale"
            ),
            "z_up_scale": self._three_floats(
                z_up_layout.get("scale"), field_name="scale"
            ),
        }

    def _add_sim_body(
        self,
        *,
        sim: SimulationManager,
        object_id: str,
        body_info: dict[str, object],
        physics: ObjectPhysics | None,
        body_type: str,
    ) -> object:
        """Add one supplied body with a pass-specific dynamic or kinematic type."""
        rigid_layout = body_info["rigid_layout"]
        if not isinstance(rigid_layout, dict):
            raise ValueError("Gravity-settle body has invalid rigid layout.")
        return sim.add_rigid_object(
            RigidObjectCfg(
                uid=object_id,
                shape=MeshCfg(
                    fpath=str(body_info["mesh_path"]),
                    max_convex_hull_num=self._max_convex_hull_num(physics),
                    acd_method="vhacd",
                ),
                init_pos=tuple(
                    self._three_floats(rigid_layout.get("pos"), field_name="pos")
                ),
                init_rot=tuple(self._simulation_euler_xyz_degrees(rigid_layout)),
                body_scale=tuple(
                    self._three_floats(body_info["y_up_scale"], field_name="scale")
                ),
                attrs=self._rigid_body_attrs(physics),
                body_type=body_type,
            )
        )

    @staticmethod
    def _rigid_body_attrs(physics: ObjectPhysics | None) -> RigidBodyAttributesCfg:
        """Convert persisted collision material data into one Lab config."""
        if physics is None:
            raise ValueError("Gravity settling requires SimReady physics settings.")
        return RigidBodyAttributesCfg(**physics.attrs)

    @staticmethod
    def _max_convex_hull_num(physics: ObjectPhysics | None) -> int:
        """Read the persisted collision-hull budget after validating physics."""
        if physics is None:
            raise ValueError("Gravity settling requires SimReady physics settings.")
        return physics.max_convex_hull_num

    @staticmethod
    def _require_body_layout_id(body: GravitySettleBody, *, name: str) -> str:
        """Validate that a body layout belongs to its scene object."""
        object_id = body.y_up_layout.get("id")
        if not isinstance(object_id, str) or not object_id:
            raise ValueError(f"{name} layout requires a non-empty string id.")
        if object_id != body.scene_object.id:
            raise ValueError(
                f"{name} layout id {object_id!r} does not match its scene object."
            )
        return object_id

    @staticmethod
    def _three_floats(value: object, *, field_name: str) -> list[float]:
        """Return three finite layout values as Python floats."""
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"Gravity-settle {field_name} must contain three values.")
        result = [float(component) for component in value]
        if not np.all(np.isfinite(result)):
            raise ValueError(f"Gravity-settle {field_name} must contain finite values.")
        return result

    @staticmethod
    def _simulation_euler_xyz_degrees(layout_object: dict[str, object]) -> list[float]:
        """Convert lowercase-xyz layout rotation to SimulationManager's XYZ order."""
        layout_rotation = Rotation.from_euler(
            "xyz",
            GravitySettler._three_floats(layout_object.get("rot"), field_name="rot"),
            degrees=True,
        )
        return layout_rotation.as_euler("XYZ", degrees=True).tolist()

    @staticmethod
    def _convert_layout_coordinate_system(
        layout_object: dict[str, object],
        *,
        source_to_target_matrix: np.ndarray,
    ) -> dict[str, object]:
        """Convert one complete layout through the y-up/z-up basis change."""
        return transform_matrix_to_layout_object(
            str(layout_object["id"]),
            source_to_target_matrix
            @ layout_object_to_transform_matrix(layout_object)
            @ np.linalg.inv(source_to_target_matrix),
        )

    @staticmethod
    def _y_up_to_z_up_matrix() -> np.ndarray:
        """Return the coordinate conversion used by Scene Engine layouts."""
        matrix = np.eye(4)
        matrix[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        return matrix
