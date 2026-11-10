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

from pathlib import Path

import matplotlib
import numpy as np
import trimesh

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from embodichain.lab.sim.scene._support_cpu import (
    SupportSurfaceConfig,
    DetectedSupportSurface as TableSupportRegion,
    SupportSurfaceDetector,
)

__all__ = ["SupportSurfaceConfig", "TableSupportRegion", "TableSupportSurfaceDetector"]


class TableSupportSurfaceDetector:
    """Detect one main upward table surface and render auditable diagnostics.

    The input mesh must be standing on x-y plane of the z-up world coordinates.
    This class deliberately returns only the largest outer 2D contour because
    the current layout stage models one main tabletop; the original support
    triangles remain available for 3D diagnostics.
    """

    def __init__(
        self,
        *,
        table_world_mesh: trimesh.Trimesh,
        debug_output_root: str | Path | None = None,
        config: SupportSurfaceConfig | None = None,
    ) -> None:
        # Init.
        self.table_world_mesh = table_world_mesh
        self.debug_output_root = (
            Path(debug_output_root).expanduser().resolve()
            if debug_output_root is not None
            else None
        )
        self.support_region: TableSupportRegion | None = None
        self.config = config if config is not None else SupportSurfaceConfig()
        # Check config values.
        if not 0.0 < self.config.normal_z_min <= 1.0:
            raise ValueError("normal_z_min must be in (0, 1].")
        if self.config.min_surface_area_m2 <= 0.0:
            raise ValueError("min_surface_area_m2 must be positive.")
        if self.config.max_face_height_span_m <= 0.0:
            raise ValueError("max_face_height_span_m must be positive.")
        if self.config.height_level_tolerance_m <= 0.0:
            raise ValueError("height_level_tolerance_m must be positive.")

    def detect(self) -> TableSupportRegion:
        """Detect support through the shared Z-up geometry implementation."""
        detector = SupportSurfaceDetector(
            self.table_world_mesh.vertices,
            self.table_world_mesh.faces,
            config=self.config,
        )
        self.support_region = detector.detect()
        return self.support_region

    def save_support_surface_debug_images(
        self,
        *,
        save_3d: bool = True,
        save_2d: bool = True,
        output_3d_path: str | Path | None = None,
        output_2d_path: str | Path | None = None,
    ) -> bool:
        """Optionally save standard 3D and 2D support-detection diagnostics."""
        if not save_3d and not save_2d:
            return True
        if self.support_region is None:
            self.detect()
        assert self.support_region is not None
        if save_3d:
            self._save_support_surface_3d_image(
                table_world_mesh=self.table_world_mesh,
                support_region=self.support_region,
                output_path=self._resolve_debug_output_path(
                    output_3d_path, "table_support_surface_3d.png"
                ),
            )
        if save_2d:
            self._save_support_region_2d_image(
                support_region=self.support_region,
                output_path=self._resolve_debug_output_path(
                    output_2d_path, "table_support_region_2d.png"
                ),
            )
        return True

    def _resolve_debug_output_path(
        self, output_path: str | Path | None, default_filename: str
    ) -> Path:
        if output_path is not None:
            return Path(output_path).expanduser().resolve()
        if self.debug_output_root is None:
            raise ValueError(
                "A debug_output_root or an explicit debug output path is required "
                "when saving support-surface debug images."
            )
        return self.debug_output_root / default_filename

    @classmethod
    def _save_support_surface_3d_image(
        cls,
        *,
        table_world_mesh: trimesh.Trimesh,
        support_region: TableSupportRegion,
        output_path: str | Path,
    ) -> Path:
        resolved_output_path = cls._resolve_png_output_path(output_path)
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        figure = plt.figure(figsize=(9, 8), dpi=160, constrained_layout=True)
        axis = figure.add_subplot(projection="3d")
        axis.add_collection3d(
            Poly3DCollection(
                table_world_mesh.vertices[table_world_mesh.faces],
                facecolor="steelblue",
                edgecolor="none",
                alpha=0.18,
            )
        )
        axis.add_collection3d(
            Poly3DCollection(
                support_region.vertices[support_region.faces],
                facecolor="darkorange",
                edgecolor="saddlebrown",
                linewidth=0.25,
                alpha=0.95,
            )
        )
        lower = table_world_mesh.bounds[0].copy()
        upper = table_world_mesh.bounds[1].copy()
        extent = upper - lower
        lower[extent <= 1e-9] -= 0.001
        upper[extent <= 1e-9] += 0.001
        axis.set(
            xlim=(lower[0], upper[0]),
            ylim=(lower[1], upper[1]),
            zlim=(lower[2], upper[2]),
        )
        axis.set_box_aspect(upper - lower)
        axis.view_init(elev=25.0, azim=-55.0)
        axis.set_xlabel("x (z-up world)")
        axis.set_ylabel("y (z-up world)")
        axis.set_zlabel("z (up)")
        axis.set_title(
            "Detected main table support surface\n"
            f"top z={support_region.top_z:.4f} m, faces={len(support_region.faces)}"
        )
        figure.savefig(resolved_output_path, bbox_inches="tight")
        plt.close(figure)
        return resolved_output_path

    @classmethod
    def _save_support_region_2d_image(
        cls,
        *,
        support_region: TableSupportRegion,
        output_path: str | Path,
    ) -> Path:
        resolved_output_path = cls._resolve_png_output_path(output_path)
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        boundary_xy = np.asarray(support_region.support_polygon.exterior.coords)
        figure, axis = plt.subplots(figsize=(8, 8), dpi=160, constrained_layout=True)
        axis.fill(
            boundary_xy[:, 0],
            boundary_xy[:, 1],
            facecolor="darkorange",
            edgecolor="none",
            alpha=0.82,
            label="detected support region",
        )
        axis.plot(
            boundary_xy[:, 0],
            boundary_xy[:, 1],
            color="saddlebrown",
            linewidth=2.0,
            label="outer support contour",
        )
        rectangle_xy = np.asarray(support_region.optimization_rectangle.exterior.coords)
        axis.plot(
            rectangle_xy[:, 0],
            rectangle_xy[:, 1],
            color="seagreen",
            linewidth=2.0,
            label="optimization rectangle",
        )
        axis.autoscale_view()
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x (z-up world)")
        axis.set_ylabel("y (z-up world)")
        axis.set_title(
            "Detected 2D table support region\n"
            f"z={support_region.top_z:.4f} m, contour vertices={len(boundary_xy) - 1}"
        )
        axis.legend(loc="best")
        figure.savefig(resolved_output_path, bbox_inches="tight")
        plt.close(figure)
        return resolved_output_path

    @staticmethod
    def _resolve_png_output_path(output_path: str | Path) -> Path:
        resolved_output_path = Path(output_path).expanduser().resolve()
        if resolved_output_path.suffix.lower() != ".png":
            return resolved_output_path.with_suffix(".png")
        return resolved_output_path
