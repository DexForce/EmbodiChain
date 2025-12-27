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

__all__ = ["PointCloudVisualizer"]


class PointCloudVisualizer(BaseVisualizer):
    """Point cloud visualizer using Open3D or matplotlib.

    Attributes:
        point_size: Size of points in visualization.
    """

    def __init__(
        self,
        backend: str = "sim_manager",
        point_size: float = 2.0,
        config: Dict[str, Any] | None = None,
        sim_manager: Any | None = None,
        control_part_name: str | None = None,
    ):
        """Initialize the point cloud visualizer.

        Args:
            backend: Visualization backend ('sim_manager', 'open3d', 'matplotlib', or 'data').
                    Defaults to 'sim_manager'. 'data' backend returns raw data without visualization.
                    'sim_manager' backend uses simulation environment for visualization.
            point_size: Size of points in visualization. Defaults to 2.0.
            config: Optional configuration dictionary. Defaults to None.
                sim_manager: SimulationManager instance for 'sim_manager' backend. Defaults to None.
                control_part_name: Control part name for naming the point cloud. Defaults to None.
        """
        super().__init__(backend, config)
        self.point_size = point_size
        self.sim_manager = sim_manager
        self.control_part_name = control_part_name

    def visualize(
        self,
        points: Union[torch.Tensor, np.ndarray],
        colors: Union[torch.Tensor, np.ndarray] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Visualize points as a point cloud.

        Args:
            points: Array of shape (N, 3) containing point positions.
            colors: Optional array of shape (N, 3) or (N, 4) containing colors.
            **kwargs: Additional visualization parameters:
                - point_size: Override default point size

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
                "type": "point_cloud",
            }
            self._last_visualization = {"data": data}
            return data
        elif self.backend == "sim_manager":
            pcd_handle = self._create_sim_manager_point_cloud(
                points, colors, point_size
            )
            self._last_visualization = {
                "point_cloud_handle": pcd_handle,
                "point_size": point_size,
            }
            return pcd_handle
        elif self.backend == "open3d":
            pcd = self._create_open3d_point_cloud(points, colors)
            self._last_visualization = {
                "point_cloud": pcd,
                "point_size": point_size,
            }
            return pcd
        elif self.backend == "matplotlib":
            fig = self._create_matplotlib_point_cloud(points, colors, point_size)
            self._last_visualization = {"figure": fig}
            return fig
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _create_sim_manager_point_cloud(
        self, points: np.ndarray, colors: np.ndarray, point_size: float
    ) -> Any:
        if self.sim_manager is None:
            raise ValueError("sim_manager is required for 'sim_manager' backend")

        # Get simulation environment
        env = self.sim_manager.get_env()
        if env is None:
            raise RuntimeError("Simulation manager has no active simulation")

        # Create point cloud name
        pcd_name = f"workspace_pcd_{self.control_part_name or 'default'}"

        # Create point cloud in simulation
        pcd_handle = env.create_point_cloud(name=pcd_name)
        pcd_handle.add_points(points)
        pcd_handle.set_colors(colors)
        pcd_handle.set_point_size(point_size)

        logger.log_info(
            f"Created point cloud '{pcd_name}' with {len(points)} points "
            f"(point_size={point_size})"
        )

        return pcd_handle

    def _create_open3d_point_cloud(
        self, points: np.ndarray, colors: np.ndarray
    ) -> "o3d.geometry.PointCloud":
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        logger.log_info(f"Created point cloud with {len(points)} points")

        return pcd

    def _create_matplotlib_point_cloud(
        self, points: np.ndarray, colors: np.ndarray, point_size: float
    ):
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

                # Coordinate frame removed - implement separately if needed

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
        if self.backend == "data":
            logger.log_warning(
                "Cannot display visualization with 'data' backend. "
                "Use 'open3d' or 'matplotlib' backend for interactive display."
            )
            return
        elif self.backend == "open3d":
            geometries = [self._last_visualization["point_cloud"]]

            # Coordinate frame removed - implement separately if needed

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
        return VisualizationType.POINT_CLOUD.value
