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


__all__ = ["VoxelVisualizer"]


class VoxelVisualizer(BaseVisualizer):
    """Voxel grid visualizer using Open3D or matplotlib.

    Attributes:
        voxel_size: Size of each voxel cube.
    """

    def __init__(
        self,
        backend: str = "sim_manager",
        voxel_size: float = 0.01,
        config: Dict[str, Any] | None = None,
        sim_manager: Any | None = None,
        control_part_name: str | None = None,
    ):
        """Initialize the voxel visualizer.

        Args:
            backend: Visualization backend ('sim_manager', 'open3d', 'matplotlib', or 'data').
                    Defaults to 'sim_manager'. 'data' backend returns voxelized data without visualization.
            voxel_size: Size of each voxel. Defaults to 0.01.
            config: Optional configuration dictionary. Defaults to None.
            sim_manager: SimulationManager instance for 'sim_manager' backend. Defaults to None.
            control_part_name: Control part name for naming. Defaults to None.
        """
        super().__init__(backend, config)
        self.voxel_size = voxel_size
        self.sim_manager = sim_manager
        self.control_part_name = control_part_name

    def visualize(
        self,
        points: torch.Tensor | np.ndarray,
        colors: torch.Tensor | np.ndarray | None = None,
        **kwargs: Any,
    ) -> Any:
        """Visualize points as a voxel grid.

        Args:
            points: Array of shape (N, 3) containing point positions.
            colors: Optional array of shape (N, 3) or (N, 4) containing colors.
            **kwargs: Additional visualization parameters:
                - voxel_size: Override default voxel size
        Returns:
            Open3D VoxelGrid geometry or matplotlib figure.

        Examples:
            >>> visualizer = VoxelVisualizer(voxel_size=0.02)
            >>> points = np.random.rand(1000, 3)
            >>> voxel_grid = visualizer.visualize(points)
            >>> visualizer.show()
        """
        # Convert to numpy
        points = self._to_numpy(points)
        self._validate_points(points)

        # Get visualization parameters
        voxel_size = kwargs.get("voxel_size", self.voxel_size)

        # Validate and prepare colors
        colors = self._validate_colors(colors, len(points))
        if colors is None:
            colors = self._get_default_colors(len(points))

        # Convert to RGB if RGBA
        if colors.shape[1] == 4:
            colors = colors[:, :3]

        if self.backend == "data":
            # Return voxelized data
            voxel_data = self._create_voxel_data(points, colors, voxel_size)
            self._last_visualization = {"data": voxel_data}
            return voxel_data
        elif self.backend == "sim_manager":
            voxels_handle = self._create_sim_manager_voxels(points, colors, voxel_size)
            self._last_visualization = {
                "voxels_handle": voxels_handle,
                "voxel_size": voxel_size,
            }
            return voxels_handle
        elif self.backend == "open3d":
            voxel_grid = self._create_open3d_voxel_grid(points, colors, voxel_size)
            self._last_visualization = {
                "voxel_grid": voxel_grid,
                "voxel_size": voxel_size,
            }
            return voxel_grid
        elif self.backend == "matplotlib":
            fig = self._create_matplotlib_voxel_grid(points, colors, voxel_size)
            self._last_visualization = {"figure": fig}
            return fig
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _create_sim_manager_voxels(
        self, points: np.ndarray, colors: np.ndarray, voxel_size: float
    ) -> Any:
        if self.sim_manager is None:
            raise ValueError("sim_manager is required for 'sim_manager' backend")

        # Get simulation env
        env = self.sim_manager.get_env()
        if env is None:
            raise RuntimeError("Simulation manager has no active env")

        cube_handles = []
        for i, point in enumerate(points):
            cube_handle = env.create_cube(l=voxel_size, w=voxel_size, h=voxel_size)
            cube_handle.set_location(float(point[0]), float(point[1]), float(point[2]))
            # TODO: Unsupported in current sim_manager API
            # cube_handle.set_color(colors[i].tolist())
            cube_handle.set_name(f"workspace_cube_{i}")
            cube_handles.append(cube_handle)

        logger.log_info(f"Created {len(points)} cubes with size={voxel_size}")

        return cube_handles

    def _create_open3d_voxel_grid(
        self, points: np.ndarray, colors: np.ndarray, voxel_size: float
    ) -> "o3d.geometry.VoxelGrid":
        # First create a point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Convert to voxel grid
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            pcd, voxel_size=voxel_size
        )

        num_voxels = len(voxel_grid.get_voxels())
        logger.log_info(
            f"Created voxel grid with {num_voxels} voxels " f"(voxel_size={voxel_size})"
        )

        return voxel_grid

    def _create_voxel_data(
        self, points: np.ndarray, colors: np.ndarray, voxel_size: float
    ) -> Dict[str, Any]:
        # Discretize points to voxel grid
        min_bounds = points.min(axis=0)
        max_bounds = points.max(axis=0)

        # Create voxel indices
        voxel_indices = np.floor((points - min_bounds) / voxel_size).astype(int)

        # Create unique voxels with averaged colors
        unique_voxels = {}
        for idx, color in zip(voxel_indices, colors):
            key = tuple(idx)
            if key not in unique_voxels:
                unique_voxels[key] = []
            unique_voxels[key].append(color)

        # Average colors for each voxel
        voxel_positions = []
        voxel_colors = []
        voxel_indices_list = []
        for idx, color_list in unique_voxels.items():
            voxel_positions.append(np.array(idx) * voxel_size + min_bounds)
            voxel_colors.append(np.mean(color_list, axis=0))
            voxel_indices_list.append(idx)

        data = {
            "voxel_positions": np.array(voxel_positions),
            "voxel_colors": np.array(voxel_colors),
            "voxel_indices": np.array(voxel_indices_list),
            "voxel_size": voxel_size,
            "min_bounds": min_bounds,
            "max_bounds": max_bounds,
            "num_voxels": len(unique_voxels),
            "type": "voxel_grid",
        }

        logger.log_info(
            f"Created voxel data with {data['num_voxels']} voxels "
            f"(voxel_size={voxel_size})"
        )

        return data

    def _create_matplotlib_voxel_grid(
        self, points: np.ndarray, colors: np.ndarray, voxel_size: float
    ):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # Discretize points to voxel grid
        min_bounds = points.min(axis=0)

        # Create voxel indices
        voxel_indices = np.floor((points - min_bounds) / voxel_size).astype(int)

        # Create unique voxels with averaged colors
        unique_voxels = {}
        for idx, color in zip(voxel_indices, colors):
            key = tuple(idx)
            if key not in unique_voxels:
                unique_voxels[key] = []
            unique_voxels[key].append(color)

        # Average colors for each voxel
        voxel_positions = []
        voxel_colors = []
        for idx, color_list in unique_voxels.items():
            voxel_positions.append(np.array(idx) * voxel_size + min_bounds)
            voxel_colors.append(np.mean(color_list, axis=0))

        voxel_positions = np.array(voxel_positions)
        voxel_colors = np.array(voxel_colors)

        # Create figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Plot voxels as cubes
        ax.scatter(
            voxel_positions[:, 0],
            voxel_positions[:, 1],
            voxel_positions[:, 2],
            c=voxel_colors,
            s=100,
            marker="s",
            alpha=0.8,
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Workspace Voxel Grid (size={voxel_size:.4f})")

        return fig

    def _save_impl(self, filepath: Path, **kwargs: Any) -> None:
        if self.backend == "data":
            # Save voxel data
            data = self._last_visualization["data"]
            np.savez(filepath, **data)
        elif self.backend == "open3d":
            voxel_grid = self._last_visualization["voxel_grid"]

            # Determine file format from extension
            suffix = filepath.suffix.lower()
            if suffix in [".ply"]:
                # Extract voxel centers and save as point cloud
                voxels = voxel_grid.get_voxels()
                points = np.array(
                    [
                        voxel_grid.origin + voxel.grid_index * voxel_grid.voxel_size
                        for voxel in voxels
                    ]
                )
                colors = np.array([voxel.color for voxel in voxels])

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                o3d.io.write_point_cloud(str(filepath), pcd)
            elif suffix in [".png", ".jpg", ".jpeg"]:
                # Render to image
                vis = o3d.visualization.Visualizer()
                vis.create_window(visible=False)
                vis.add_geometry(voxel_grid)

                # Coordinate frame removed - implement separately if needed

                vis.update_geometry(voxel_grid)
                vis.poll_events()
                vis.update_renderer()
                vis.capture_screen_image(str(filepath))
                vis.destroy_window()
            else:
                raise ValueError(
                    f"Unsupported file format: {suffix}. " f"Use .ply, .png, .jpg"
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
            geometries = [self._last_visualization["voxel_grid"]]

            # Coordinate frame removed - implement separately if needed

            o3d.visualization.draw_geometries(geometries)

        elif self.backend == "matplotlib":
            import matplotlib.pyplot as plt

            plt.show()

    def get_type_name(self) -> str:
        return VisualizationType.VOXEL.value
