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
from typing import Union, Dict, Any
from pathlib import Path

try:
    import open3d as o3d

    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from embodichain.utils import logger
from embodichain.lab.sim.utility.workspace_analyzer.visualizers.base_visualizer import (
    BaseVisualizer,
)
from embodichain.lab.sim.utility.workspace_analyzer.configs.visualization_config import (
    VisualizationType,
)

__all__ = ["AxisVisualizer"]


class AxisVisualizer(BaseVisualizer):
    """Visualizer for coordinate axes/frames at specified poses.

    This visualizer creates coordinate axes (X, Y, Z) at given transformation matrices,
    useful for visualizing robot end-effector poses, workspace reference frames, etc.

    Supports multiple backends:
    - 'sim_manager': Uses SimulationManager.draw_marker() with MarkerCfg
    - 'open3d': Creates coordinate frames using Open3D
    - 'matplotlib': Draws axis lines in 3D matplotlib plot
    - 'data': Returns axis data without visualization
    """

    def __init__(
        self,
        backend: str = "sim_manager",
        axis_length: float = 0.15,
        axis_size: float = 0.005,
        config: Dict[str, Any] | None = None,
        sim_manager: Any | None = None,
        control_part_name: str | None = None,
        reference_pose: Union[np.ndarray, torch.Tensor] | None = None,
        arena_index: int = 0,
    ):
        """Initialize the axis visualizer.

        Args:
            backend: Visualization backend ('sim_manager', 'open3d', 'matplotlib', or 'data').
                    Defaults to 'sim_manager'.
            axis_length: Length of each axis. Defaults to 0.15.
            axis_size: Thickness/size of axes. Defaults to 0.005.
            config: Optional configuration dictionary. Defaults to None.
            sim_manager: SimulationManager instance for 'sim_manager' backend. Defaults to None.
            control_part_name: Control part name for naming (compatibility). Defaults to None.
            reference_pose: Reference pose (4x4 matrix) for orientation. Defaults to None.
            arena_index: Arena index for sim_manager markers. Defaults to 0.
        """
        super().__init__(backend, config)
        self.axis_length = axis_length
        self.axis_size = axis_size
        self.sim_manager = sim_manager
        self.control_part_name = control_part_name
        self.reference_pose = reference_pose
        self.arena_index = arena_index

    def visualize(
        self,
        poses: Union[torch.Tensor, np.ndarray],
        colors: Union[torch.Tensor, np.ndarray] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Visualize coordinate axes at specified poses or points.

        Args:
            poses: Either transformation matrices of shape (4, 4) or (N, 4, 4),
                  or point coordinates of shape (N, 3) which will be converted to poses.
            colors: Optional colors (not used for axes but kept for interface compatibility).
            **kwargs: Additional visualization parameters:
                - axis_length: Override default axis length
                - axis_size: Override default axis size
                - name_prefix: Prefix for axis names (default: "axis")
                - arena_index: Arena index for sim_manager (default: 0)

        Returns:
            Backend-specific axis representation.

        Examples:
            >>> import numpy as np
            >>> visualizer = AxisVisualizer(backend='sim_manager', sim_manager=sim)
            >>> # Using transformation matrix
            >>> pose = np.eye(4)
            >>> pose[:3, 3] = [1.0, 0.5, 1.2]  # Set position
            >>> result = visualizer.visualize(pose)
            >>>
            >>> # Using point coordinates
            >>> points = np.array([[1.0, 0.5, 1.2], [2.0, 1.0, 0.8]])
            >>> result = visualizer.visualize(points)
            >>> visualizer.show()
        """
        # Convert to numpy
        poses = self._to_numpy(poses)

        # Convert points to poses if needed
        poses = self._convert_points_to_poses(poses)

        # Validate pose dimensions
        self._validate_poses(poses)

        # Get parameters
        axis_length = kwargs.get("axis_length", self.axis_length)
        axis_size = kwargs.get("axis_size", self.axis_size)
        name_prefix = kwargs.get("name_prefix", "axis")
        arena_index = kwargs.get("arena_index", self.arena_index)

        # Dispatch to backend implementation
        if self.backend == "sim_manager":
            return self._visualize_sim_manager(
                poses, axis_length, axis_size, name_prefix, arena_index, **kwargs
            )
        elif self.backend == "open3d":
            return self._visualize_open3d(
                poses, axis_length, axis_size, name_prefix, **kwargs
            )
        elif self.backend == "matplotlib":
            return self._visualize_matplotlib(
                poses, axis_length, axis_size, name_prefix, **kwargs
            )
        elif self.backend == "data":
            return self._visualize_data(
                poses, axis_length, axis_size, name_prefix, **kwargs
            )
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _visualize_sim_manager(
        self,
        poses: np.ndarray,
        axis_length: float,
        axis_size: float,
        name_prefix: str,
        arena_index: int,
        **kwargs: Any,
    ) -> Any:
        """Visualize axes using sim_manager backend."""
        if self.sim_manager is None:
            raise ValueError("sim_manager is required for 'sim_manager' backend")

        # Import here to avoid circular imports
        from embodichain.lab.sim.cfg import MarkerCfg

        axis_markers = []

        # Handle single pose (4,4) or multiple poses (N,4,4)
        if poses.ndim == 2:
            poses = poses[np.newaxis, :, :]

        for i, pose_matrix in enumerate(poses):
            marker_name = f"{name_prefix}_{i}"

            try:
                # Create axis marker using MarkerCfg
                marker_cfg = MarkerCfg(
                    name=marker_name,
                    marker_type="axis",
                    axis_xpos=pose_matrix,  # 4x4 transformation matrix
                    axis_size=axis_size,
                    axis_len=axis_length,
                    arena_index=arena_index,
                )

                # Draw the marker
                self.sim_manager.draw_marker(cfg=marker_cfg)
                axis_markers.append(marker_name)

            except Exception as e:
                logger.log_warning(f"Failed to draw axis {marker_name}: {e}")

        logger.log_info(f"Created {len(axis_markers)} coordinate axes with sim_manager")
        self._last_visualization = axis_markers
        return axis_markers

    def _visualize_open3d(
        self,
        poses: np.ndarray,
        axis_length: float,
        axis_size: float,
        name_prefix: str,
        **kwargs: Any,
    ) -> Any:
        """Visualize axes using Open3D backend."""
        if not OPEN3D_AVAILABLE:
            raise RuntimeError(
                "Open3D is not available. Install with: pip install open3d"
            )

        # Create coordinate frame geometries
        frame_geometries = []

        if poses.ndim == 2:
            poses = poses[np.newaxis, :, :]

        for i, pose_matrix in enumerate(poses):
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=axis_length, origin=[0, 0, 0]
            )
            frame.transform(pose_matrix)
            frame_geometries.append(frame)

        logger.log_info(
            f"Created {len(frame_geometries)} coordinate frames with Open3D"
        )
        self._last_visualization = frame_geometries
        return frame_geometries

    def _visualize_matplotlib(
        self,
        poses: np.ndarray,
        axis_length: float,
        axis_size: float,
        name_prefix: str,
        **kwargs: Any,
    ) -> Any:
        """Visualize axes using matplotlib backend."""
        if not MATPLOTLIB_AVAILABLE:
            raise RuntimeError(
                "Matplotlib is not available. Install with: pip install matplotlib"
            )

        # Create figure and 3D axis
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        if poses.ndim == 2:
            poses = poses[np.newaxis, :, :]

        colors = ["red", "green", "blue"]  # X, Y, Z axes
        labels = ["X", "Y", "Z"]

        for i, pose_matrix in enumerate(poses):
            origin = pose_matrix[:3, 3]

            for j, (color, label) in enumerate(zip(colors, labels)):
                axis_direction = pose_matrix[:3, j]
                axis_end = origin + axis_direction * axis_length

                ax.plot3D(
                    [origin[0], axis_end[0]],
                    [origin[1], axis_end[1]],
                    [origin[2], axis_end[2]],
                    color=color,
                    linewidth=axis_size * 1000,
                    label=f"{label}_axis_{i}" if i == 0 else "",
                )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        ax.set_title(f"Coordinate Axes ({len(poses)} frames)")

        logger.log_info(f"Created matplotlib plot with {len(poses)} coordinate frames")
        self._last_visualization = fig
        return fig

    def _visualize_data(
        self,
        poses: np.ndarray,
        axis_length: float,
        axis_size: float,
        name_prefix: str,
        **kwargs: Any,
    ) -> Any:
        """Return axis data without visualization."""
        axis_data = {
            "poses": poses,
            "axis_length": axis_length,
            "axis_size": axis_size,
            "name_prefix": name_prefix,
            "type": "coordinate_axes",
            "num_frames": len(poses) if poses.ndim == 3 else 1,
        }

        logger.log_info(
            f"Generated axis data for {axis_data['num_frames']} coordinate frames"
        )
        self._last_visualization = axis_data
        return axis_data

    def _convert_points_to_poses(self, data: np.ndarray) -> np.ndarray:
        """Convert point coordinates to transformation matrices if needed.

        Args:
            data: Either points (N, 3) or poses (4, 4) or (N, 4, 4)

        Returns:
            Transformation matrices (4, 4) or (N, 4, 4)
        """
        # Check if input is points (N, 3)
        if data.ndim == 2 and data.shape[1] == 3:
            # Convert points to poses
            num_points = data.shape[0]

            # Use reference pose if available, otherwise identity rotation
            if self.reference_pose is not None:
                # Convert reference pose to numpy if needed
                if isinstance(self.reference_pose, torch.Tensor):
                    ref_pose = self.reference_pose.cpu().numpy()
                else:
                    ref_pose = self.reference_pose

                # Ensure 4x4 matrix
                if ref_pose.ndim == 3:
                    ref_pose = ref_pose[0]  # Take first pose if batch

                # Create poses with reference orientation and point positions
                poses = np.tile(ref_pose, (num_points, 1, 1))
                poses[:, :3, 3] = data  # Override translation with point positions

                logger.log_debug(
                    f"Using reference pose orientation for {num_points} coordinate axes"
                )
            else:
                # Fallback to identity rotation
                poses = np.tile(np.eye(4), (num_points, 1, 1))
                poses[:, :3, 3] = data  # Set translation

                logger.log_debug(
                    f"Using identity orientation for {num_points} coordinate axes"
                )

            return poses

        elif data.ndim == 1 and data.shape[0] == 3:
            # Single point case
            if self.reference_pose is not None:
                # Convert reference pose to numpy if needed
                if isinstance(self.reference_pose, torch.Tensor):
                    pose = self.reference_pose.cpu().numpy()
                else:
                    pose = self.reference_pose.copy()

                # Ensure 2D matrix
                if pose.ndim == 3:
                    pose = pose[0]

                # Override translation
                pose[:3, 3] = data
            else:
                # Fallback to identity
                pose = np.eye(4)
                pose[:3, 3] = data

            return pose
        else:
            # Already poses, return as is
            return data

    def _validate_poses(self, poses: np.ndarray) -> None:
        """Validate pose array dimensions and values.

        Args:
            poses: Array of transformation matrices

        Raises:
            ValueError: If poses have invalid shape or values
        """
        if poses.ndim not in [2, 3]:
            raise ValueError(
                f"Poses must be 2D (4,4) or 3D (N,4,4) array, got shape {poses.shape}"
            )

        if poses.ndim == 2:
            if poses.shape != (4, 4):
                raise ValueError(f"Single pose must be (4,4), got {poses.shape}")
        else:  # poses.ndim == 3
            if poses.shape[1:] != (4, 4):
                raise ValueError(f"Multiple poses must be (N,4,4), got {poses.shape}")

        # Check if last row is [0, 0, 0, 1] for valid transformation matrices
        if poses.ndim == 2:
            poses_to_check = [poses]
        else:
            poses_to_check = poses

        for i, pose in enumerate(poses_to_check):
            expected_bottom_row = np.array([0, 0, 0, 1])
            if not np.allclose(pose[3, :], expected_bottom_row, atol=1e-6):
                logger.log_warning(
                    f"Pose {i} bottom row {pose[3, :]} != [0,0,0,1]. May not be a valid transformation matrix."
                )

    def get_type_name(self) -> str:
        """Return the type name for this visualizer."""
        return VisualizationType.AXIS.value

    def _save_impl(self, filepath: Path, **kwargs: Any) -> None:
        """Save the visualization to file."""
        if self._last_visualization is None:
            raise RuntimeError("No visualization to save. Call visualize() first.")

        if self.backend == "open3d":
            if filepath.suffix.lower() in [".ply", ".obj", ".stl"]:
                # Save combined mesh
                combined_mesh = o3d.geometry.TriangleMesh()
                for frame in self._last_visualization:
                    combined_mesh += frame
                o3d.io.write_triangle_mesh(str(filepath), combined_mesh)
                logger.log_info(f"Saved Open3D coordinate frames to {filepath}")
            else:
                logger.log_warning(
                    f"Unsupported file format {filepath.suffix} for Open3D backend"
                )

        elif self.backend == "matplotlib":
            self._last_visualization.savefig(filepath, **kwargs)
            logger.log_info(f"Saved matplotlib plot to {filepath}")

        elif self.backend in ["sim_manager", "data"]:
            # Save as numpy file
            if self.backend == "sim_manager":
                data_to_save = {
                    "axis_names": self._last_visualization,
                    "type": "sim_manager_axes",
                }
            else:
                data_to_save = self._last_visualization

            np.save(filepath.with_suffix(".npy"), data_to_save, allow_pickle=True)
            logger.log_info(f"Saved axis data to {filepath.with_suffix('.npy')}")

    def _show_impl(self, **kwargs: Any) -> None:
        """Display the visualization."""
        if self._last_visualization is None:
            logger.log_warning("No visualization to show. Call visualize() first.")
            return

        if self.backend == "open3d":
            o3d.visualization.draw_geometries(self._last_visualization)

        elif self.backend == "matplotlib":
            plt.show()

        elif self.backend == "sim_manager":
            logger.log_info(
                f"Axes are displayed in simulation. Marker names: {self._last_visualization}"
            )

        elif self.backend == "data":
            logger.log_info(
                f"Data backend - no visual display. Use .save() to export data."
            )
