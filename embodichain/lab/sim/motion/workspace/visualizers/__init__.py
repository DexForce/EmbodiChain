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

"""Workspace result visualizers deriving from ``BaseVisualizer``.

Built-in visualizers: point-cloud, voxel, sphere, axis, and manipulability, plus a ``VisualizerFactory`` and ``create_visualizer`` helper.

The manipulability path additionally exports its pure mapping helpers
(:func:`align_manipulability_scores`, :func:`normalize_manipulability`,
:func:`map_manipulability_colors`, :func:`select_inspection_indices`,
:func:`inspect_points`, :func:`translational_manipulability_ellipsoid`) so score
mapping can be reused and tested without a renderer.
"""

from embodichain.lab.sim.motion.workspace.visualizers.base_visualizer import (
    BaseVisualizer,
    IVisualizer,
)
from embodichain.lab.sim.motion.workspace.configs import (
    VisualizationType,
)

from embodichain.lab.sim.motion.workspace.visualizers.point_cloud_visualizer import (
    PointCloudVisualizer,
)

from embodichain.lab.sim.motion.workspace.visualizers.voxel_visualizer import (
    VoxelVisualizer,
)

from embodichain.lab.sim.motion.workspace.visualizers.sphere_visualizer import (
    SphereVisualizer,
)

from embodichain.lab.sim.motion.workspace.visualizers.axis_visualizer import (
    AxisVisualizer,
)

from embodichain.lab.sim.motion.workspace.visualizers.manipulability_visualizer import (
    InspectionSelection,
    ManipulabilityColorCfg,
    ManipulabilityColorMapping,
    ManipulabilityNormalization,
    ManipulabilityPointSet,
    ManipulabilityVisualizer,
    PointInspection,
    align_manipulability_scores,
    ellipsoid_surface,
    inspect_points,
    map_manipulability_colors,
    normalize_manipulability,
    select_inspection_indices,
    translational_manipulability_ellipsoid,
)

from embodichain.lab.sim.motion.workspace.visualizers.visualizer_factory import (
    VisualizerFactory,
    create_visualizer,
)

__all__ = [
    "BaseVisualizer",
    "IVisualizer",
    "VisualizationType",
    "PointCloudVisualizer",
    "VoxelVisualizer",
    "SphereVisualizer",
    "AxisVisualizer",
    "ManipulabilityVisualizer",
    "ManipulabilityColorCfg",
    "ManipulabilityColorMapping",
    "ManipulabilityNormalization",
    "ManipulabilityPointSet",
    "InspectionSelection",
    "PointInspection",
    "align_manipulability_scores",
    "normalize_manipulability",
    "map_manipulability_colors",
    "select_inspection_indices",
    "inspect_points",
    "translational_manipulability_ellipsoid",
    "ellipsoid_surface",
    "VisualizerFactory",
    "create_visualizer",
]
