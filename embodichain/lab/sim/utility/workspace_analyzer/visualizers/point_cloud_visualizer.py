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

__all__ = ["PointCloudVisualizer"]


class PointCloudVisualizer(BaseVisualizer):
    """Point cloud visualizer using Open3D or matplotlib.

    This visualizer renders workspace points as a point cloud,
    which is efficient for large numbers of points and provides
    a clear view of the spatial distribution.

    Advantages:
        - Fast rendering for large point sets
        - Memory efficient
        - Clear spatial representation
        - Interactive viewing with Open3D

    Attributes:
        point_size: Size of points in visualization.
        show_coordinate_frame: Whether to show coordinate frame.
    """

    def __init__(
        self,
        backend: str = "open3d",
        point_size: float = 2.0,
        show_coordinate_frame: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the point cloud visualizer.

        Args:
            backend: Visualization backend ('open3d', 'matplotlib', or 'data'). Defaults to 'open3d'.
                    'data' backend returns raw data without visualization.
            point_size: Size of points in visualization. Defaults to 2.0.
            show_coordinate_frame: Whether to show coordinate frame. Defaults to True.
            config: Optional configuration dictionary. Defaults to None.
        """
        super().__init__(backend, config)
        self.point_size = point_size
        self.show_coordinate_frame = show_coordinate_frame

    def visualize(
        self,
        points: Union[torch.Tensor, np.ndarray],
        colors: Optional[Union[torch.Tensor, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Any:
        """Visualize points as a point cloud.

        Args:
            points: Array of shape (N, 3) containing point positions.
            colors: Optional array of shape (N, 3) or (N, 4) containing colors.
            **kwargs: Additional visualization parameters:
                - point_size: Override default point size
                - show_coordinate_frame: Override coordinate frame display

        Returns:
            Open3D PointCloud geometry or matplotlib figure.

        Examples:
            >>> visualizer = PointCloudVisualizer()
            >>> points = np.random.rand(1000, 3)
            >>> colors = np.random.rand(1000, 3)
            >>> pcd = visualizer.visualize(points, colors)
            >>> visualizer.show()
        """
        # Convert to numpy
        points = self._to_numpy(points)
        self._validate_points(points)

        # Get visualization parameters
        point_size = kwargs.get("point_size", self.point_size)
        show_frame = kwargs.get("show_coordinate_frame", self.show_coordinate_frame)

        # Validate and prepare colors
        colors = self._validate_colors(colors, len(points))
        if colors is None:
            colors = self._get_default_colors(len(points))

        # Convert to RGB if RGBA
        if colors.shape[1] == 4:
            colors = colors[:, :3]

        if self.backend == "data":
            # Return raw data for user to handle
            data = {
                "points": points,
                "colors": colors,
                "point_size": point_size,
                "show_frame": show_frame,
                "type": "point_cloud",
            }
            self._last_visualization = {"data": data}
            return data
        elif self.backend == "open3d":
            pcd = self._create_open3d_point_cloud(points, colors)
            self._last_visualization = {
                "point_cloud": pcd,
                "point_size": point_size,
                "show_frame": show_frame,
            }
            return pcd
        elif self.backend == "matplotlib":
            fig = self._create_matplotlib_point_cloud(points, colors, point_size)
            self._last_visualization = {"figure": fig}
            return fig
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _create_open3d_point_cloud(
        self, points: np.ndarray, colors: np.ndarray
    ) -> "o3d.geometry.PointCloud":
        """Create Open3D point cloud geometry.

        Args:
            points: Array of shape (N, 3) containing point positions.
            colors: Array of shape (N, 3) containing RGB colors.

        Returns:
            Open3D PointCloud object.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        logger.log_info(f"Created point cloud with {len(points)} points")

        return pcd

    def _create_matplotlib_point_cloud(
        self, points: np.ndarray, colors: np.ndarray, point_size: float
    ):
        """Create matplotlib 3D scatter plot.

        Args:
            points: Array of shape (N, 3) containing point positions.
            colors: Array of shape (N, 3) containing RGB colors.
            point_size: Size of points.

        Returns:
            Matplotlib figure.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2], c=colors, s=point_size, alpha=0.6
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Workspace Point Cloud")

        return fig

    def _save_impl(self, filepath: Path, **kwargs: Any) -> None:
        """Save point cloud to file.

        Args:
            filepath: Path to save the visualization.
            **kwargs: Additional save parameters.
        """
        if self.backend == "data":
            # Save data as numpy file
            data = self._last_visualization["data"]
            np.savez(filepath, **data)
        elif self.backend == "open3d":
            pcd = self._last_visualization["point_cloud"]

            # Determine file format from extension
            suffix = filepath.suffix.lower()
            if suffix in [".pcd", ".ply", ".xyz", ".xyzrgb", ".pts"]:
                o3d.io.write_point_cloud(str(filepath), pcd)
            elif suffix in [".png", ".jpg", ".jpeg"]:
                # Render to image
                vis = o3d.visualization.Visualizer()
                vis.create_window(visible=False)
                vis.add_geometry(pcd)

                if self._last_visualization.get("show_frame", True):
                    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                    vis.add_geometry(frame)

                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                vis.capture_screen_image(str(filepath))
                vis.destroy_window()
            else:
                raise ValueError(
                    f"Unsupported file format: {suffix}. "
                    f"Use .pcd, .ply, .xyz, .xyzrgb, .pts, .png, .jpg"
                )

        elif self.backend == "matplotlib":
            fig = self._last_visualization["figure"]
            fig.savefig(filepath, dpi=300, bbox_inches="tight")

    def _show_impl(self, **kwargs: Any) -> None:
        """Display point cloud interactively.

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
            geometries = [self._last_visualization["point_cloud"]]

            if self._last_visualization.get("show_frame", True):
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                geometries.append(frame)

            # Set point size in visualization
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            for geom in geometries:
                vis.add_geometry(geom)

            render_option = vis.get_render_option()
            render_option.point_size = self._last_visualization.get("point_size", 2.0)

            vis.run()
            vis.destroy_window()

        elif self.backend == "matplotlib":
            import matplotlib.pyplot as plt

            plt.show()

    def get_type_name(self) -> str:
        """Get the name of the visualization type.

        Returns:
            String identifier for the visualization type.
        """
        return VisualizationType.POINT_CLOUD.value
