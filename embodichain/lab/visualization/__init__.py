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

"""Browser-based visualization of simulation scenes via Viser.

``SceneExporter`` produces backend-neutral snapshots, ``VisualizationRuntime`` pushes them to a Viser backend, with gizmo overlays, camera preview, and CLI helpers.
"""

from __future__ import annotations

from .cli import add_viser_args_to_parser, visualization_cfg_from_args
from .cfg import VisualizationCfg, ViserServerCfg
from .protocol import (
    CameraImage,
    CameraImageFrame,
    CameraSpec,
    DynamicMeshUpdate,
    FrameOverlay,
    GizmoCommand,
    GizmoSpec,
    GizmoState,
    JointControlCommand,
    JointControlProvider,
    JointControlSpec,
    JointControlState,
    MeshGeometry,
    PickCommand,
    PointCloudOverlay,
    SceneFrame,
    SceneManifest,
    SceneNode,
    SceneOverlays,
    TargetOverlay,
    TrajectoryOverlay,
    pose_to_position_wxyz,
)
from .runtime import (
    GizmoCommandQueue,
    JointControlCommandQueue,
    LatestFrameQueue,
    RuntimeHealth,
    RuntimeStats,
    VisualizationRuntime,
)
from .scene_exporter import CameraImageCaptureResult, CaptureResult, SceneExporter

__all__ = [
    "CameraImage",
    "CameraImageCaptureResult",
    "CameraImageFrame",
    "CameraSpec",
    "CaptureResult",
    "DynamicMeshUpdate",
    "FrameOverlay",
    "GizmoCommand",
    "GizmoCommandQueue",
    "GizmoSpec",
    "GizmoState",
    "JointControlCommand",
    "JointControlCommandQueue",
    "JointControlProvider",
    "JointControlSpec",
    "JointControlState",
    "LatestFrameQueue",
    "MeshGeometry",
    "PickCommand",
    "PointCloudOverlay",
    "RuntimeHealth",
    "RuntimeStats",
    "SceneExporter",
    "SceneFrame",
    "SceneManifest",
    "SceneNode",
    "SceneOverlays",
    "TargetOverlay",
    "TrajectoryOverlay",
    "VisualizationCfg",
    "VisualizationRuntime",
    "ViserServerCfg",
    "add_viser_args_to_parser",
    "pose_to_position_wxyz",
    "visualization_cfg_from_args",
]
