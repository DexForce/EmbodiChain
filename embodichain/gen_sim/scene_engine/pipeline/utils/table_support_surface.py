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
from collections.abc import Collection
from pathlib import Path
from typing import Literal

import matplotlib
import numpy as np
import trimesh
from scipy.spatial import ConvexHull

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from embodichain.gen_sim.scene_engine.core.scene_object import SceneObject
from embodichain.gen_sim.scene_engine.pipeline.utils.simready_processor_utils import (
    _strip_json_code_fence,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.table_support_surface_preview import (
    render_table_support_candidates,
)
from embodichain.gen_sim.services.compat import VisionClient
from embodichain.lab.sim.scene._support_cpu import (
    SupportSurfaceConfig,
    DetectedSupportSurface as TableSupportRegion,
    SupportSurfaceDetector,
)
from embodichain.utils.logger import log_warning

__all__ = [
    "SupportSurfaceConfig",
    "TableSupportRegion",
    "TableSupportSurfaceDetector",
    "store_table_support",
]

_TABLETOP_SELECTION_PROMPT = """Select the main upward-facing tabletop of an isolated support asset.
The front-elevated view marks visible support candidates with colored patches and
numbers on the mesh. Use the right-hand sidebar to map a visible number to its
surface ID. A sidebar entry alone is not evidence of a usable support surface.
Choose the principal surface that would directly support objects placed on this table.
Exclude feet, braces, undersides, and raised rims or ledges around the usable surface.
{base_requirement}
If the supporting face cannot be verified from the visible mesh, return null for surface_id.

Return only JSON with exactly the keys surface_id and reason. Use a supplied
surface ID or null, and briefly describe the visible evidence in reason."""
_DEGRADED_BASE_ALLOWED = (
    "For a tabletop-only slab, including when table_state is degraded, "
    "its broad visible upper face can be the support."
)
_COMPLETE_BASE_REQUIRED = (
    "Confirm the table has visible legs or a base extending well below the tabletop. "
    "Reject a tabletop-only slab or an upside-down table."
)
_MIN_REFERENCE_TABLETOP_COVERAGE = 0.10


def store_table_support(table: SceneObject, region: TableSupportRegion) -> None:
    """Record the chosen tabletop in the table's canonical Z-up frame."""
    table.support_surface_z = region.top_z
    table.support_contour_xy = [
        [float(x), float(y)] for x, y in region.support_polygon.exterior.coords[:-1]
    ]
    table.support_optimization_rect_xy = [
        [float(x), float(y)]
        for x, y in region.optimization_rectangle.exterior.coords[:-1]
    ]


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

    def detect(self, *, degraded_reference: bool = False) -> TableSupportRegion:
        """Detect support, requiring a broad connected footprint for a reference slab."""
        detector = self._geometry_detector()
        region = detector.detect()
        if degraded_reference:
            # Disconnected strips can jointly span the slab without providing
            # one useful footprint to fit a complete replacement table.
            footprint_area = ConvexHull(region.vertices[:, :2]).volume
            minimum_area = _MIN_REFERENCE_TABLETOP_COVERAGE * footprint_area
            if region.support_polygon.area < minimum_area:
                replacement = max(
                    detector.detect_candidates(limit=3),
                    key=lambda candidate: candidate.support_polygon.area,
                )
                if replacement.support_polygon.area < minimum_area:
                    raise ValueError(
                        "Degraded tabletop has no broad connected reference surface."
                    )
                log_warning(
                    "The selected reference tabletop was fragmented; "
                    "using the broader connected candidate."
                )
                region = replacement
        self.support_region = region
        return region

    def _geometry_detector(self) -> SupportSurfaceDetector:
        return SupportSurfaceDetector(
            self.table_world_mesh.vertices,
            self.table_world_mesh.faces,
            config=self.config,
        )

    def select_with_vision(
        self,
        *,
        vlm_client: VisionClient,
        table_state: Literal["normal", "degraded"] | None,
        require_complete_base: bool = False,
    ) -> TableSupportRegion:
        """Select a measured tabletop using an isolated, numbered mesh preview."""
        self.support_region = None
        preview_path = self._resolve_debug_output_path(
            None, "table_support_candidates.png"
        )
        decision_path = preview_path.with_name("table_support_selection.json")
        candidates = self._geometry_detector().detect_candidates(limit=3)
        labeled_ids = render_table_support_candidates(
            table_mesh=self.table_world_mesh,
            candidates=candidates,
            output_path=preview_path,
        )
        by_id = {f"surface_{index}": candidates[index - 1] for index in labeled_ids}
        prompt_data = {"table_state": table_state, "surface_ids": list(by_id)}
        system_prompt = _TABLETOP_SELECTION_PROMPT.format(
            base_requirement=(
                _COMPLETE_BASE_REQUIRED
                if require_complete_base
                else _DEGRADED_BASE_ALLOWED
            )
        )
        user_prompt = json.dumps(prompt_data)
        audit = {
            **prompt_data,
            "require_complete_base": require_complete_base,
            "candidate_top_z_m": {
                f"surface_{index}": region.top_z
                for index, region in enumerate(candidates, start=1)
            },
        }
        responses: list[str] = []
        surface_id = None
        reason = "No candidate is visible in the tabletop preview."
        if by_id:
            for _ in range(3):
                response = vlm_client.complete(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    image_path=preview_path,
                )
                responses.append(response)
                try:
                    surface_id, reason = _parse_tabletop_selection(response, by_id)
                except ValueError:
                    continue
                break
            else:
                decision_path.write_text(
                    json.dumps(
                        {**audit, "status": "invalid_response", "responses": responses},
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                raise ValueError(
                    "VLM returned invalid tabletop support selections after 3 attempts."
                )

        decision_path.write_text(
            json.dumps(
                {
                    **audit,
                    "status": "matched" if surface_id is not None else "unresolved",
                    "surface_id": surface_id,
                    "reason": reason,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if surface_id is None:
            raise ValueError(f"Tabletop support selection was unresolved: {reason}")
        self.support_region = by_id[surface_id]
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


def _parse_tabletop_selection(
    response: str, surface_ids: Collection[str]
) -> tuple[str | None, str]:
    value = json.loads(_strip_json_code_fence(response))
    if not isinstance(value, dict) or set(value) != {"surface_id", "reason"}:
        raise ValueError("Expected surface_id and reason in tabletop selection.")
    surface_id = value["surface_id"]
    if surface_id is not None and (
        not isinstance(surface_id, str) or surface_id not in surface_ids
    ):
        raise ValueError("Tabletop selection must reference a candidate surface ID.")
    reason = value["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Tabletop selection requires a reason.")
    return surface_id, reason.strip()
