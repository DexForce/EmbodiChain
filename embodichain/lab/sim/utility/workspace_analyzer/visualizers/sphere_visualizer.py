# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

import numpy as np
import torch
from typing import Union, Any, Dict
from pathlib import Path

from embodichain.lab.sim.utility.workspace_analyzer.configs import (
    VisualizationType,
)

from embodichain.lab.sim.utility.workspace_analyzer.visualizers.base_visualizer import (
    BaseVisualizer,
    OPEN3D_AVAILABLE,
)

if OPEN3D_AVAILABLE:
    import open3d as o3d

from embodichain.utils import logger


__all__ = ["SphereVisualizer"]


class SphereVisualizer(BaseVisualizer):
    """Sphere-based visualizer using Open3D or matplotlib.

    Attributes:
        sphere_radius: Radius of each sphere.
        sphere_resolution: Resolution of sphere mesh (higher = smoother).
    """

    def __init__(
        self,
        backend: str = "sim_manager",
        sphere_radius: float = 0.005,
        sphere_resolution: int = 10,
        config: Dict[str, Any] | None = None,
        sim_manager: Any | None = None,
        control_part_name: str | None = None,
    ):
        """Initialize the sphere visualizer.

        Args:
            backend: Visualization backend ('sim_manager', 'open3d', 'matplotlib', or 'data').
                    Defaults to 'sim_manager'. 'data' backend returns sphere data without visualization.
            sphere_radius: Radius of each sphere. Defaults to 0.005.
            sphere_resolution: Sphere mesh resolution. Defaults to 10.
            config: Optional configuration dictionary. Defaults to None.
            sim_manager: SimulationManager instance for 'sim_manager' backend. Defaults to None.
            control_part_name: Control part name for naming. Defaults to None.
        """
        super().__init__(backend, config)
        self.sphere_radius = sphere_radius
        self.sphere_resolution = sphere_resolution
        self.sim_manager = sim_manager
        self.control_part_name = control_part_name

    def visualize(
        self,
        points: torch.Tensor | np.ndarray,
        colors: torch.Tensor | np.ndarray | None = None,
        **kwargs: Any,
    ) -> Any:
        """Visualize points as spheres.

        Args:
            points: Array of shape (N, 3) containing point positions.
            colors: Optional array of shape (N, 3) or (N, 4) containing colors.
            **kwargs: Additional visualization parameters:
                - sphere_radius: Override default sphere radius
                - sphere_resolution: Override sphere resolution
                - max_spheres: Maximum number of spheres to render (for performance)

        Returns:
            Open3D TriangleMesh or matplotlib figure.

        Examples:
            >>> visualizer = SphereVisualizer(sphere_radius=0.01)
            >>> points = np.random.rand(100, 3)
            >>> colors = np.random.rand(100, 3)
            >>> mesh = visualizer.visualize(points, colors)
            >>> visualizer.show()
        """
        # Convert to numpy
        points = self._to_numpy(points)
        self._validate_points(points)

        # Get visualization parameters
        sphere_radius = kwargs.get("sphere_radius", self.sphere_radius)
        sphere_resolution = kwargs.get("sphere_resolution", self.sphere_resolution)
        max_spheres = kwargs.get("max_spheres", None)

        # Limit number of spheres for performance
        if max_spheres is not None and len(points) > max_spheres:
            logger.log_warning(
                f"Limiting visualization to {max_spheres} spheres "
                f"(total points: {len(points)})"
            )
            indices = np.random.choice(len(points), max_spheres, replace=False)
            points = points[indices]
            if colors is not None:
                colors = (
                    colors[indices]
                    if isinstance(colors, np.ndarray)
                    else self._to_numpy(colors)[indices]
                )

        # Validate and prepare colors
        colors = self._validate_colors(colors, len(points))
        if colors is None:
            colors = self._get_default_colors(len(points))

        # Convert to RGB if RGBA
        if colors.shape[1] == 4:
            colors = colors[:, :3]

        if self.backend == "data":
            # Return sphere data
            data = {
                "centers": points,
                "colors": colors,
                "radius": sphere_radius,
                "resolution": sphere_resolution,
                "num_spheres": len(points),
                "type": "spheres",
            }
            self._last_visualization = {"data": data}
            logger.log_info(
                f"Created sphere data with {len(points)} spheres (radius={sphere_radius})"
            )
            return data
        elif self.backend == "sim_manager":
            spheres_handle = self._create_sim_manager_spheres(
                points, colors, sphere_radius
            )
            self._last_visualization = {
                "spheres_handle": spheres_handle,
                "radius": sphere_radius,
            }
            return spheres_handle
        elif self.backend == "open3d":
            mesh = self._create_open3d_spheres(
                points, colors, sphere_radius, sphere_resolution
            )
            self._last_visualization = {"mesh": mesh}
            return mesh
        elif self.backend == "matplotlib":
            fig = self._create_matplotlib_spheres(points, colors, sphere_radius)
            self._last_visualization = {"figure": fig}
            return fig
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _create_sim_manager_spheres(
        self, points: np.ndarray, colors: np.ndarray, sphere_radius: float
    ) -> Any:
        if self.sim_manager is None:
            raise ValueError("sim_manager is required for 'sim_manager' backend")

        # Get simulation env
        env = self.sim_manager.get_env()
        if env is None:
            raise RuntimeError("Simulation manager has no active env")

        sphere_handles = []
        for i, point in enumerate(points):
            sphere_handle = env.create_sphere(radius=sphere_radius, resolution=10)
            # Unpack array to individual x, y, z coordinates
            sphere_handle.set_location(
                float(point[0]), float(point[1]), float(point[2])
            )
            # TODO: Unsupported in current sim_manager API
            # sphere_handle.set_color(colors[i].tolist())
            sphere_handle.set_name(f"workspace_sphere_{i}")
            sphere_handles.append(sphere_handle)

        logger.log_info(f"Created {len(points)} spheres with radius={sphere_radius}")

        return sphere_handles

    def _create_open3d_spheres(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        sphere_radius: float,
        sphere_resolution: int,
    ) -> "o3d.geometry.TriangleMesh":
        # Create a template sphere
        sphere_template = o3d.geometry.TriangleMesh.create_sphere(
            radius=sphere_radius, resolution=sphere_resolution
        )

        # Combine all spheres into one mesh
        combined_mesh = o3d.geometry.TriangleMesh()

        for point, color in zip(points, colors):
            # Copy and translate sphere
            sphere = o3d.geometry.TriangleMesh(sphere_template)
            sphere.translate(point)
            sphere.paint_uniform_color(color)

            # Merge into combined mesh
            combined_mesh += sphere

        # Compute normals for proper lighting
        combined_mesh.compute_vertex_normals()

        logger.log_info(f"Created {len(points)} spheres with radius={sphere_radius}")

        return combined_mesh

    def _create_matplotlib_spheres(
        self, points: np.ndarray, colors: np.ndarray, sphere_radius: float
    ):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Scale marker size based on sphere radius
        marker_size = (sphere_radius * 1000) ** 2

        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=colors,
            s=marker_size,
            alpha=0.8,
            marker="o",
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Workspace Spheres (radius={sphere_radius:.4f})")

        return fig

    def _save_impl(self, filepath: Path, **kwargs: Any) -> None:
        if self.backend == "data":
            # Save sphere data
            data = self._last_visualization["data"]
            np.savez(filepath, **data)
        elif self.backend == "open3d":
            mesh = self._last_visualization["mesh"]

            # Determine file format from extension
            suffix = filepath.suffix.lower()
            if suffix in [".ply", ".obj", ".stl", ".gltf", ".glb"]:
                o3d.io.write_triangle_mesh(str(filepath), mesh)
            elif suffix in [".png", ".jpg", ".jpeg"]:
                # Render to image
                vis = o3d.visualization.Visualizer()
                vis.create_window(visible=False)
                vis.add_geometry(mesh)

                vis.update_geometry(mesh)
                vis.poll_events()
                vis.update_renderer()
                vis.capture_screen_image(str(filepath))
                vis.destroy_window()
            else:
                raise ValueError(
                    f"Unsupported file format: {suffix}. "
                    f"Use .ply, .obj, .stl, .gltf, .glb, .png, .jpg"
                )

        elif self.backend == "matplotlib":
            fig = self._last_visualization["figure"]
            fig.savefig(filepath, dpi=300, bbox_inches="tight")

    def _show_impl(self, **kwargs: Any) -> None:
        if self.backend == "data":
            logger.log_warning(
                "Cannot display visualization with 'data' backend. "
                "Use 'open3d' or 'matplotlib' backend for interactive display."
            )
            return
        elif self.backend == "open3d":
            geometries = [self._last_visualization["mesh"]]

            # Coordinate frame removed - implement separately if needed

            o3d.visualization.draw_geometries(geometries)

        elif self.backend == "matplotlib":
            import matplotlib.pyplot as plt

            plt.show()

    def get_type_name(self) -> str:
        return VisualizationType.SPHERE.value
