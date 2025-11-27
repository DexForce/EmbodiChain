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
from typing import Union, Optional, Any, Dict
from pathlib import Path

try:
    from .base_visualizer import BaseVisualizer, VisualizationType, OPEN3D_AVAILABLE
except ImportError:
    from base_visualizer import BaseVisualizer, VisualizationType, OPEN3D_AVAILABLE

if OPEN3D_AVAILABLE:
    import open3d as o3d

from embodichain.utils import logger


__all__ = ["SphereVisualizer"]


class SphereVisualizer(BaseVisualizer):
    """Sphere-based visualizer using Open3D or matplotlib.

    This visualizer renders workspace points as spheres,
    which provides a smooth, visually appealing representation
    with adjustable sphere radius to show reachability zones.

    Advantages:
        - Smooth visual appearance
        - Good for showing reachability regions
        - Intuitive spatial understanding
        - Can represent uncertainty/tolerance

    Disadvantages:
        - More computationally expensive than point clouds
        - Higher memory usage
        - Can be cluttered with many points

    Attributes:
        sphere_radius: Radius of each sphere.
        sphere_resolution: Resolution of sphere mesh (higher = smoother).
        show_coordinate_frame: Whether to show coordinate frame.
    """

    def __init__(
        self,
        backend: str = "open3d",
        sphere_radius: float = 0.005,
        sphere_resolution: int = 10,
        show_coordinate_frame: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the sphere visualizer.

        Args:
            backend: Visualization backend ('open3d', 'matplotlib', or 'data'). Defaults to 'open3d'.
                    'data' backend returns sphere data without visualization.
            sphere_radius: Radius of each sphere. Defaults to 0.005.
            sphere_resolution: Sphere mesh resolution. Defaults to 10.
            show_coordinate_frame: Whether to show coordinate frame. Defaults to True.
            config: Optional configuration dictionary. Defaults to None.
        """
        super().__init__(backend, config)
        self.sphere_radius = sphere_radius
        self.sphere_resolution = sphere_resolution
        self.show_coordinate_frame = show_coordinate_frame

    def visualize(
        self,
        points: Union[torch.Tensor, np.ndarray],
        colors: Optional[Union[torch.Tensor, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Any:
        """Visualize points as spheres.

        Args:
            points: Array of shape (N, 3) containing point positions.
            colors: Optional array of shape (N, 3) or (N, 4) containing colors.
            **kwargs: Additional visualization parameters:
                - sphere_radius: Override default sphere radius
                - sphere_resolution: Override sphere resolution
                - show_coordinate_frame: Override coordinate frame display
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
        show_frame = kwargs.get("show_coordinate_frame", self.show_coordinate_frame)
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
                "show_frame": show_frame,
                "num_spheres": len(points),
                "type": "spheres",
            }
            self._last_visualization = {"data": data}
            logger.log_info(
                f"Created sphere data with {len(points)} spheres (radius={sphere_radius})"
            )
            return data
        elif self.backend == "open3d":
            mesh = self._create_open3d_spheres(
                points, colors, sphere_radius, sphere_resolution
            )
            self._last_visualization = {"mesh": mesh, "show_frame": show_frame}
            return mesh
        elif self.backend == "matplotlib":
            fig = self._create_matplotlib_spheres(points, colors, sphere_radius)
            self._last_visualization = {"figure": fig}
            return fig
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _create_open3d_spheres(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        sphere_radius: float,
        sphere_resolution: int,
    ) -> "o3d.geometry.TriangleMesh":
        """Create Open3D sphere meshes.

        Args:
            points: Array of shape (N, 3) containing point positions.
            colors: Array of shape (N, 3) containing RGB colors.
            sphere_radius: Radius of each sphere.
            sphere_resolution: Sphere mesh resolution.

        Returns:
            Combined Open3D TriangleMesh object.
        """
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
        """Create matplotlib 3D scatter plot with sphere markers.

        Args:
            points: Array of shape (N, 3) containing point positions.
            colors: Array of shape (N, 3) containing RGB colors.
            sphere_radius: Radius of each sphere (affects marker size).

        Returns:
            Matplotlib figure.
        """
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
        """Save sphere visualization to file.

        Args:
            filepath: Path to save the visualization.
            **kwargs: Additional save parameters.
        """
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

                if self._last_visualization.get("show_frame", True):
                    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                    vis.add_geometry(frame)

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
        """Display sphere visualization interactively.

        Args:
            **kwargs: Display parameters.
        """
        if self.backend == "data":
            logger.log_warning(
                "Cannot display visualization with 'data' backend. "
                "Use 'open3d' or 'matplotlib' backend for interactive display."
            )
            return
        elif self.backend == "open3d":
            geometries = [self._last_visualization["mesh"]]

            if self._last_visualization.get("show_frame", True):
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                geometries.append(frame)

            o3d.visualization.draw_geometries(geometries)

        elif self.backend == "matplotlib":
            import matplotlib.pyplot as plt

            plt.show()

    def get_type_name(self) -> str:
        """Get the name of the visualization type.

        Returns:
            String identifier for the visualization type.
        """
        return VisualizationType.SPHERE.value
