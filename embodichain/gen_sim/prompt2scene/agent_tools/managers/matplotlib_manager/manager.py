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

"""Matplotlib manager for mesh visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from embodichain.gen_sim.prompt2scene.agent_tools.managers.matplotlib_manager.schemas import (
    RenderFootprintLayoutRequest,
    RenderFootprintLayoutResult,
    RenderImageComparisonRequest,
    RenderImageComparisonResult,
    RenderSupportRegionRequest,
    RenderSupportRegionResult,
    RenderXYComparisonRequest,
    RenderXYComparisonResult,
)

__all__ = ["MatplotlibManager"]


class MatplotlibManager:
    """Manager for mesh visualization via matplotlib.

    Wraps matplotlib rendering with typed request/response methods,
    following the same pattern as service clients.
    """

    def __init__(
        self,
        *,
        figsize: tuple[float, float] = (8, 8),
        dpi: int = 180,
    ) -> None:
        """Initialize the matplotlib manager.

        Args:
            figsize: Default figure size for rendered images.
            dpi: Output image resolution.
        """
        self._figsize = figsize
        self._dpi = dpi

    def render_footprint_layout(
        self,
        request: RenderFootprintLayoutRequest,
    ) -> RenderFootprintLayoutResult:
        """Render labeled XY footprints with full-length coordinate axes."""
        output_path = request.output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not request.object_ids:
            return RenderFootprintLayoutResult(output_path=output_path)

        centers = {
            object_id: np.asarray(request.centers[object_id], dtype=float)
            for object_id in request.object_ids
        }
        sizes = {
            object_id: np.asarray(request.xy_sizes[object_id], dtype=float)
            for object_id in request.object_ids
        }
        footprint_mins = np.vstack(
            [
                centers[object_id] - 0.5 * sizes[object_id]
                for object_id in request.object_ids
            ]
        )
        footprint_maxs = np.vstack(
            [
                centers[object_id] + 0.5 * sizes[object_id]
                for object_id in request.object_ids
            ]
        )
        data_min = footprint_mins.min(axis=0)
        data_max = footprint_maxs.max(axis=0)
        span = np.maximum(data_max - data_min, 1.0e-6)
        padding = max(float(span.max()) * 0.12, 1.0e-3)
        x_limits = (float(data_min[0] - padding), float(data_max[0] + padding))
        y_limits = (float(data_min[1] - padding), float(data_max[1] + padding))

        fig, ax = plt.subplots(figsize=self._figsize)
        for object_id in request.object_ids:
            center = centers[object_id]
            size = sizes[object_id]
            ax.add_patch(
                Rectangle(
                    (center[0] - 0.5 * size[0], center[1] - 0.5 * size[1]),
                    size[0],
                    size[1],
                    facecolor=(0.35, 0.60, 0.95, 0.30),
                    edgecolor=(0.08, 0.22, 0.60, 1.0),
                    linewidth=1.5,
                )
            )
            label = object_id.replace("interact_", "").removesuffix("_0")
            ax.text(
                center[0],
                center[1],
                label,
                ha="center",
                va="center",
                fontsize=9,
                color="black",
            )

        self._draw_full_xy_axes(ax, x_limits=x_limits, y_limits=y_limits)
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(request.title)
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.30)
        fig.tight_layout()
        fig.savefig(output_path, dpi=self._dpi)
        plt.close(fig)
        return RenderFootprintLayoutResult(output_path=output_path)

    def render_image_comparison(
        self,
        request: RenderImageComparisonRequest,
    ) -> RenderImageComparisonResult:
        """Render two images side by side with numbered labels."""
        output_path = request.output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        first_image = plt.imread(request.first_image_path.expanduser().resolve())
        second_image = plt.imread(request.second_image_path.expanduser().resolve())

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        for ax, image, label in (
            (axes[0], first_image, request.first_label),
            (axes[1], second_image, request.second_label),
        ):
            ax.imshow(image)
            ax.text(
                0.03,
                0.92,
                label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=16,
                color="white",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "black",
                    "edgecolor": "none",
                    "alpha": 0.55,
                },
            )
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=self._dpi, facecolor="white")
        plt.close(fig)
        return RenderImageComparisonResult(output_path=output_path)

    @staticmethod
    def _draw_full_xy_axes(
        ax: Any,
        *,
        x_limits: tuple[float, float],
        y_limits: tuple[float, float],
    ) -> None:
        """Draw axes across the full viewport, centered on the data bounds."""
        axis_color = "#303030"
        x_center = 0.5 * (x_limits[0] + x_limits[1])
        y_center = 0.5 * (y_limits[0] + y_limits[1])
        # Horizontal axis (X) — spans full width, positioned at vertical centre.
        ax.annotate(
            "",
            xy=(x_limits[1], y_center),
            xytext=(x_limits[0], y_center),
            arrowprops={"arrowstyle": "->", "color": axis_color, "lw": 1.8},
            zorder=8,
        )
        # Vertical axis (Y) — spans full height, positioned at horizontal centre.
        ax.annotate(
            "",
            xy=(x_center, y_limits[1]),
            xytext=(x_center, y_limits[0]),
            arrowprops={"arrowstyle": "->", "color": axis_color, "lw": 1.8},
            zorder=8,
        )
        x_span = x_limits[1] - x_limits[0]
        y_span = y_limits[1] - y_limits[0]
        ax.text(
            x_limits[1] - 0.03 * x_span,
            y_center + 0.02 * y_span,
            "+X",
            ha="right",
            va="bottom",
            color=axis_color,
            fontsize=11,
        )
        ax.text(
            x_center + 0.02 * x_span,
            y_limits[1] - 0.03 * y_span,
            "+Y",
            ha="left",
            va="top",
            color=axis_color,
            fontsize=11,
        )
        # Mark the origin at the centre.
        ax.plot(x_center, y_center, "o", color=axis_color, markersize=6, zorder=9)
        ax.text(
            x_center + 0.015 * x_span,
            y_center + 0.015 * y_span,
            "Origin",
            fontsize=8,
            color=axis_color,
            ha="left",
            va="bottom",
            zorder=9,
        )

    def render_selected_support_region(
        self, request: RenderSupportRegionRequest
    ) -> RenderSupportRegionResult:
        """Render a mesh with the selected support region highlighted."""
        output_path = request.output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        vertices = np.asarray(request.mesh.vertices, dtype=float)
        faces = np.asarray(request.mesh.faces, dtype=int)
        selected_faces = faces[np.asarray(request.face_indices, dtype=int)]

        fig = plt.figure(figsize=self._figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.add_collection3d(
            Poly3DCollection(
                vertices[faces],
                facecolors=(0.65, 0.68, 0.72, 0.16),
                edgecolors=(0.35, 0.37, 0.40, 0.08),
                linewidths=0.15,
            )
        )
        ax.add_collection3d(
            Poly3DCollection(
                vertices[selected_faces],
                facecolors=(1.0, 0.18, 0.05, 0.88),
                edgecolors=(0.55, 0.02, 0.0, 1.0),
                linewidths=0.8,
            )
        )
        self._set_equal_axes(ax, vertices)
        ax.view_init(elev=25.0, azim=-45.0)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Selected Support Region")
        fig.tight_layout()
        fig.savefig(output_path, dpi=self._dpi)
        plt.close(fig)
        return RenderSupportRegionResult(output_path=output_path)

    def render_xy_alignment_comparison(
        self, request: RenderXYComparisonRequest
    ) -> RenderXYComparisonResult:
        """Render before/after XY projections for PCA yaw alignment."""
        output_path = request.output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        before_polygons, before_xy = self._xy_polygons_and_vertices(request.before_mesh)
        after_polygons, after_xy = self._xy_polygons_and_vertices(request.after_mesh)
        center, view_half = self._xy_view_bounds(before_xy, after_xy)

        fig, axes = plt.subplots(1, 2, figsize=self._figsize)
        self._draw_xy_projection(
            axes[0],
            before_polygons,
            before_xy,
            "Before PCA yaw",
            center,
            view_half,
        )
        self._draw_xy_projection(
            axes[1],
            after_polygons,
            after_xy,
            f"After PCA yaw ({request.angle_degrees:.2f} deg)",
            center,
            view_half,
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=self._dpi)
        plt.close(fig)
        return RenderXYComparisonResult(output_path=output_path)

    @staticmethod
    def _xy_polygons_and_vertices(mesh: Any) -> tuple[Any, Any]:
        vertices = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=int)
        return vertices[faces][:, :, :2], vertices[:, :2]

    @staticmethod
    def _xy_view_bounds(before_xy: Any, after_xy: Any) -> tuple[Any, float]:
        values = np.concatenate([before_xy, after_xy], axis=0)
        bounds_min = values.min(axis=0)
        bounds_max = values.max(axis=0)
        center = 0.5 * (bounds_min + bounds_max)
        span = np.maximum(bounds_max - bounds_min, 1e-3)
        view_half = max(float(span.max()) * 0.65, 0.5)
        return center, view_half

    def _draw_xy_projection(
        self,
        ax: Any,
        polygons_xy: Any,
        vertices_xy: Any,
        title: str,
        center: Any,
        view_half: float,
    ) -> None:
        ax.add_collection(
            PolyCollection(
                polygons_xy,
                facecolors=(0.24, 0.50, 0.90, 0.28),
                edgecolors=(0.05, 0.16, 0.35, 0.20),
                linewidths=0.20,
            )
        )
        self._draw_xy_aabb(ax, vertices_xy)
        self._add_xy_axes(ax, view_half)
        ax.set_xlim(center[0] - view_half, center[0] + view_half)
        ax.set_ylim(center[1] - view_half, center[1] + view_half)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(title)
        ax.grid(True, which="major", linestyle="-", linewidth=0.7, alpha=0.35)
        ax.minorticks_on()
        ax.grid(True, which="minor", linestyle=":", linewidth=0.45, alpha=0.25)

    @staticmethod
    def _draw_xy_aabb(ax: Any, vertices_xy: Any) -> None:
        bounds_min = vertices_xy.min(axis=0)
        bounds_max = vertices_xy.max(axis=0)
        width, height = bounds_max - bounds_min
        ax.add_patch(
            Rectangle(
                (bounds_min[0], bounds_min[1]),
                width,
                height,
                fill=False,
                edgecolor="#d62828",
                linewidth=1.6,
                linestyle="-",
                alpha=0.95,
            )
        )

    @staticmethod
    def _add_xy_axes(ax: Any, view_half: float) -> None:
        arrow_len = max(view_half * 0.35, 0.2)
        ax.scatter([0.0], [0.0], color="black", s=22, zorder=8)
        ax.text(0.0, 0.0, " Origin", fontsize=9, ha="left", va="bottom")
        ax.arrow(
            0.0,
            0.0,
            arrow_len,
            0.0,
            width=arrow_len * 0.015,
            head_width=arrow_len * 0.06,
            head_length=arrow_len * 0.08,
            color="#d62828",
            length_includes_head=True,
            zorder=9,
        )
        ax.text(arrow_len * 1.08, 0.0, "+X", color="#d62828", fontsize=11)
        ax.arrow(
            0.0,
            0.0,
            0.0,
            arrow_len,
            width=arrow_len * 0.015,
            head_width=arrow_len * 0.06,
            head_length=arrow_len * 0.08,
            color="#2a9d8f",
            length_includes_head=True,
            zorder=9,
        )
        ax.text(0.0, arrow_len * 1.08, "+Y", color="#2a9d8f", fontsize=11)

    @staticmethod
    def _set_equal_axes(ax: Any, vertices: Any) -> None:
        mins = np.min(vertices, axis=0)
        maxs = np.max(vertices, axis=0)
        center = (mins + maxs) * 0.5
        radius = max(float(np.max(maxs - mins)) * 0.5, 1e-6)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
