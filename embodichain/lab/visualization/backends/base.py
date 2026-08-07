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

from abc import ABC, abstractmethod
from collections.abc import Callable

from ..protocol import (
    CameraImageFrame,
    GizmoCommand,
    JointControlCommand,
    SceneFrame,
    SceneManifest,
)

__all__ = ["VisualizationBackend"]


class VisualizationBackend(ABC):
    """Lifecycle and publishing contract for live visualization backends."""

    def set_gizmo_command_sink(
        self,
        sink: Callable[[GizmoCommand], None] | None,
    ) -> None:
        """Set the thread-safe sink used for browser Gizmo commands."""
        self._gizmo_command_sink = sink

    def set_joint_control_command_sink(
        self,
        sink: Callable[[JointControlCommand], None] | None,
    ) -> None:
        """Set the thread-safe sink used for browser joint commands."""
        self._joint_control_command_sink = sink

    @property
    @abstractmethod
    def endpoint(self) -> str | None:
        """Return the local HTTP endpoint after startup."""

    @property
    @abstractmethod
    def client_count(self) -> int:
        """Return the number of currently connected clients."""

    @abstractmethod
    def start(self) -> None:
        """Start backend resources on the visualization thread."""

    @abstractmethod
    def publish_manifest(self, manifest: SceneManifest) -> None:
        """Replace the current static scene."""

    @abstractmethod
    def publish_frame(self, frame: SceneFrame) -> bool:
        """Publish a dynamic frame, returning whether it was accepted."""

    @abstractmethod
    def publish_camera_images(self, frame: CameraImageFrame) -> bool:
        """Publish low-frequency camera images."""

    @abstractmethod
    def poll(self) -> None:
        """Apply queued backend-local events while no frame is available."""

    @abstractmethod
    def stop(self) -> None:
        """Flush and release backend resources on the visualization thread."""
