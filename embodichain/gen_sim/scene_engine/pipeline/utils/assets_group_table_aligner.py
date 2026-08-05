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

from embodichain.gen_sim.scene_engine.pipeline.utils.scene_generation_utils import (
    layout_object_to_transform_matrix,
    load_glb_mesh,
    transform_matrix_to_layout_object,
)
from embodichain.utils.logger import log_info


@dataclass(frozen=True)
class AssetsGroupTableAlignerConfig:
    """Controls for the initial vertical gap above the table."""

    clearance_m: float = 0.02  # Initial table-to-group gap in metres.


class AssetsGroupTableAligner:
    """Place every asset as one rigid vertical group above a table AABB top."""

    def __init__(
        self,
        *,
        table_layout: dict[str, object],
        assets_layout: list[dict[str, object]],
        geometry_root: str | Path,
        config: AssetsGroupTableAlignerConfig | None = None,
    ) -> None:
        self.table_layout = table_layout
        self.assets_layout = assets_layout
        self.geometry_root = Path(geometry_root).expanduser().resolve()
        self.aligned_table_layout: dict[str, object] | None = None
        self.aligned_assets_layout: list[dict[str, object]] | None = None
        self.config = config if config is not None else AssetsGroupTableAlignerConfig()
        # Check.
        if self.config.clearance_m < 0.0:
            raise ValueError("Table clearance_m must be non-negative.")

    def align(self) -> tuple[dict[str, object], list[dict[str, object]]]:
        """Return y-up layouts with the complete asset group above the table.

        Input and output layouts use y-up, matching the GLBs on disk. The group
        is temporarily measured in z-up coordinates and every asset receives the
        same vertical translation. This preserves all asset-to-asset relative
        poses.
        """
        self.aligned_table_layout = None
        self.aligned_assets_layout = None
        if not self.assets_layout:
            self.aligned_table_layout = self.table_layout
            self.aligned_assets_layout = []
            log_info("Scene has no movable assets; skipping vertical group alignment.")
            return self.aligned_table_layout, self.aligned_assets_layout

        y_up_to_z_up_matrix = np.eye(4)
        y_up_to_z_up_matrix[:3, :3] = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
        )
        z_up_to_y_up_matrix = np.linalg.inv(y_up_to_z_up_matrix)

        z_up_table_layout = self._convert_layout_coordinate_system(
            self.table_layout,
            source_to_target_matrix=y_up_to_z_up_matrix,
        )
        z_up_assets_layout = [
            self._convert_layout_coordinate_system(
                asset_layout,
                source_to_target_matrix=y_up_to_z_up_matrix,
            )
            for asset_layout in self.assets_layout
        ]

        table_id = self._require_layout_id(z_up_table_layout, name="Table")
        table_mesh = load_glb_mesh(self.geometry_root / f"{table_id}.glb")
        table_mesh.apply_transform(y_up_to_z_up_matrix)
        table_mesh.apply_transform(layout_object_to_transform_matrix(z_up_table_layout))
        target_group_bottom_z = table_mesh.bounds[1, 2] + self.config.clearance_m

        group_bottom_z = np.inf
        for asset_layout in z_up_assets_layout:
            asset_id = self._require_layout_id(asset_layout, name="Asset")
            asset_mesh = load_glb_mesh(self.geometry_root / f"{asset_id}.glb")
            asset_mesh.apply_transform(y_up_to_z_up_matrix)
            asset_mesh.apply_transform(layout_object_to_transform_matrix(asset_layout))
            # Find the lowest z among all the assets.
            group_bottom_z = min(group_bottom_z, float(asset_mesh.bounds[0, 2]))

        group_vertical_translation_z = target_group_bottom_z - group_bottom_z
        for asset_layout in z_up_assets_layout:
            asset_layout["pos"][2] += group_vertical_translation_z

        self.aligned_table_layout = self._convert_layout_coordinate_system(
            z_up_table_layout,
            source_to_target_matrix=z_up_to_y_up_matrix,
        )
        self.aligned_assets_layout = [
            self._convert_layout_coordinate_system(
                asset_layout,
                source_to_target_matrix=z_up_to_y_up_matrix,
            )
            for asset_layout in z_up_assets_layout
        ]
        log_info(
            "Aligned the asset group above the table with "
            f"delta_z={group_vertical_translation_z:.4f} m and "
            f"clearance={self.config.clearance_m:.4f} m."
        )
        return self.aligned_table_layout, self.aligned_assets_layout

    @staticmethod
    def _convert_layout_coordinate_system(
        layout_object: dict[str, object],
        *,
        source_to_target_matrix: np.ndarray,
    ) -> dict[str, object]:
        """Convert one layout object between coordinate systems through its matrix."""
        return transform_matrix_to_layout_object(
            str(layout_object["id"]),
            source_to_target_matrix
            @ layout_object_to_transform_matrix(layout_object)
            @ np.linalg.inv(source_to_target_matrix),
        )

    @staticmethod
    def _require_layout_id(layout_object: dict[str, object], *, name: str) -> str:
        object_id = layout_object.get("id")
        if not isinstance(object_id, str) or not object_id:
            raise ValueError(f"{name} layout must contain a non-empty string id.")
        return object_id
