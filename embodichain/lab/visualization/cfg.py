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

from dataclasses import field

from embodichain.utils import configclass

__all__ = ["VisualizationCfg", "ViserServerCfg"]


@configclass
class ViserServerCfg:
    """Configure the local Viser HTTP/WebSocket server.

    Args:
        host: Interface on which Viser listens. Server deployments should inject
            this value instead of accepting it from an untrusted run request.
        port: TCP port allocated to the worker.
        label: Browser application label.
        verbose: Whether Viser should print server diagnostics.
    """

    host: str = "127.0.0.1"
    port: int = 8080
    label: str = "EmbodiChain"
    verbose: bool = False

    def __post_init__(self) -> None:
        """Validate server settings."""
        if not self.host:
            raise ValueError("host must not be empty.")
        if not 1 <= self.port <= 65_535:
            raise ValueError("port must be between 1 and 65535.")
        if not self.label:
            raise ValueError("label must not be empty.")


@configclass
class VisualizationCfg:
    """Configure live scene visualization.

    Args:
        backend: Visualization backend name. Supported values are ``"none"``
            and ``"viser"``.
        scene_fps: Maximum scene capture rate.
        env_ids: Environment indices exposed by the visualizer. ``None`` selects
            every simulation environment.
        max_visible_envs: Optional safety limit on the number of selected
            environments. ``None`` disables the limit.
        point_cloud_max_points: Maximum number of points retained per point cloud.
        sensor_image_fps: Maximum camera RGB preview update rate. ``None``
            captures once per visualization step instead of using wall-clock
            rate limiting.
        soft_body_fps: Maximum soft-body and cloth vertex update rate.
        allow_commands: Whether simulation-mutating browser commands are allowed.
            This enables Viser Gizmo dragging and registered articulation joint
            controls. Keep it disabled for untrusted or publicly reachable
            browser sessions.
        viser_server: Viser HTTP/WebSocket server binding settings.
    """

    backend: str = "none"
    scene_fps: float = 15.0
    env_ids: list[int] | None = [0]
    max_visible_envs: int | None = None
    point_cloud_max_points: int = 100_000
    sensor_image_fps: float | None = 2.0
    soft_body_fps: float = 5.0
    allow_commands: bool = False
    viser_server: ViserServerCfg = field(default_factory=ViserServerCfg)

    def __post_init__(self) -> None:
        """Validate visualization settings."""
        if self.backend not in {"none", "viser"}:
            raise ValueError(
                f"Unsupported visualization backend {self.backend!r}; expected 'none' or 'viser'."
            )
        if self.scene_fps <= 0.0:
            raise ValueError("scene_fps must be greater than zero.")
        if self.max_visible_envs is not None and self.max_visible_envs <= 0:
            raise ValueError("max_visible_envs must be greater than zero.")
        if self.point_cloud_max_points <= 0:
            raise ValueError("point_cloud_max_points must be greater than zero.")
        if self.sensor_image_fps is not None and self.sensor_image_fps <= 0.0:
            raise ValueError("sensor_image_fps must be greater than zero.")
        if self.soft_body_fps <= 0.0:
            raise ValueError("soft_body_fps must be greater than zero.")
        if self.env_ids is not None:
            if not self.env_ids:
                raise ValueError("env_ids must contain at least one environment index.")
            if len(set(self.env_ids)) != len(self.env_ids):
                raise ValueError("env_ids must not contain duplicates.")
            if any(env_id < 0 for env_id in self.env_ids):
                raise ValueError(
                    "env_ids must contain non-negative environment indices."
                )
            if (
                self.max_visible_envs is not None
                and len(self.env_ids) > self.max_visible_envs
            ):
                raise ValueError(
                    f"Selected {len(self.env_ids)} environments, exceeding "
                    f"max_visible_envs={self.max_visible_envs}."
                )
        if self.allow_commands and self.backend != "viser":
            raise ValueError("allow_commands is only supported by the Viser backend.")
