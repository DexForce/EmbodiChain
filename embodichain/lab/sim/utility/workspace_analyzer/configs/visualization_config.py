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

from enum import Enum
from dataclasses import dataclass
from typing import Union


class VisualizationType(Enum):
    POINT_CLOUD = "point_cloud"
    VOXEL = "voxel"
    SPHERE = "sphere"
    AXIS = "axis"
    # MESH = "mesh"
    # HEATMAP = "heatmap"


@dataclass
class VisualizationConfig:
    """Visualization configuration"""

    enabled: bool = True
    """Enable or disable visualization."""

    vis_type: Union[VisualizationType, str] = VisualizationType.POINT_CLOUD
    """Type of visualization to use. Can be VisualizationType enum or string.
    
    Available visualization types:
    - POINT_CLOUD: Fast point cloud rendering, good for large datasets
    - VOXEL: Volumetric voxel grid representation for occupancy mapping
    - SPHERE: Smooth sphere rendering for publication-quality figures
    - AXIS: Coordinate frame visualization for poses and transformations
    
    Examples:
        vis_type = VisualizationType.SPHERE
        vis_type = "point_cloud"
        vis_type = "axis"  # For coordinate frames
    """

    voxel_size: float = 0.05
    """Voxel size for downsampling."""

    nb_neighbors: int = 20
    """Number of neighbors for statistical outlier removal."""

    std_ratio: float = 2.0
    """Standard deviation ratio for statistical outlier removal."""

    is_voxel_down: bool = True
    """Enable voxel downsampling."""

    color_by_distance: bool = True
    """Color points by distance."""

    point_size: float = 4.0
    """Size of points in visualization."""

    alpha: float = 0.5
    """Transparency level of points."""

    sphere_radius: float = 0.005
    """Radius of spheres for sphere visualization."""

    sphere_resolution: int = 10
    """Sphere mesh resolution for sphere visualization."""

    axis_length: float = 0.05
    """Length of coordinate axes for axis visualization."""

    axis_size: float = 0.003
    """Thickness/size of coordinate axes for axis visualization."""

    show_unreachable_points: bool = True
    """Whether to show unreachable points in Cartesian space and Plane sampling modes.
    
    If True, shows both reachable (green, large) and unreachable (red, small) points.
    If False, only shows reachable points in Cartesian space and Plane sampling modes.
    Has no effect in Joint space mode (all points are always shown as reachable).
    
    Note: This parameter now supports all IK-based analysis modes including:
    - AnalysisMode.CARTESIAN_SPACE
    - AnalysisMode.PLANE_SAMPLING
    """
