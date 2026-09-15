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

from dataclasses import MISSING, field

from embodichain.utils import configclass
from embodichain.cli._visualization import (
    _DEFAULT_HOST,
    _DEFAULT_PORT,
    _DEFAULT_SCENE_FPS,
    _DEFAULT_IMAGE_FPS,
    _DEFAULT_SOFT_BODY_FPS,
    _DEFAULT_ENV_IDS,
)

__all__ = ["PreviewGroupCfg", "VisualizationCfg", "ViserServerCfg"]


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

    host: str = _DEFAULT_HOST
    port: int = _DEFAULT_PORT
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
class PreviewGroupCfg:
    """Register one translucent preview copy of a simulated articulation.

    A preview group adds extra scene nodes that reuse the link meshes of an
    existing robot or articulation but keep independent poses supplied by the
    caller, one :class:`~embodichain.lab.visualization.protocol.PreviewNodeUpdate`
    per captured frame. Registering a group never reads or writes joint state,
    so a preview is free of simulation side effects.

    Args:
        group_id: Unique identifier of the preview group. Preview node IDs and
            scene paths are derived from it.
        articulation_uid: UID of the robot or articulation whose link meshes
            are reused.
        source_kind: Registry owning ``articulation_uid``; either ``"robot"``
            or ``"articulation"``.
        env_id: Environment instance the preview meshes are drawn in. It must
            be one of the visualized environments.
        link_names: Source links included in the preview. ``None`` selects
            every source link that has renderable geometry.
        opacity: Constant transparency of the preview meshes in ``(0, 1]``.
        color: RGB tint for the preview meshes. ``None`` reuses the source
            geometry entries verbatim, which shares one batched mesh (and its
            color) with the real links instead of uploading a tinted copy.
        visible: Whether preview nodes are visible before the first pose
            update arrives. Preview groups stay hidden by default.
    """

    group_id: str = MISSING
    articulation_uid: str = MISSING
    source_kind: str = "robot"
    env_id: int = 0
    link_names: list[str] | None = None
    opacity: float = 0.35
    color: tuple[int, int, int] | None = (120, 190, 255)
    visible: bool = False

    def __post_init__(self) -> None:
        """Validate the preview group selection and appearance."""
        if self.source_kind not in {"robot", "articulation"}:
            raise ValueError(
                "source_kind must be either 'robot' or 'articulation'; "
                f"received {self.source_kind!r}."
            )
        if self.env_id < 0:
            raise ValueError("env_id must be non-negative.")
        if not 0.0 < self.opacity <= 1.0:
            raise ValueError("opacity must be greater than zero and at most one.")
        if self.link_names is not None:
            if not self.link_names:
                raise ValueError("link_names must contain at least one link name.")
            if len(set(self.link_names)) != len(self.link_names):
                raise ValueError("link_names must not contain duplicates.")
        if self.color is not None:
            if len(self.color) != 3 or any(
                component < 0 or component > 255 for component in self.color
            ):
                raise ValueError("color must be an RGB triple with values in [0, 255].")


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
    scene_fps: float = _DEFAULT_SCENE_FPS
    env_ids: list[int] | None = list(_DEFAULT_ENV_IDS)
    max_visible_envs: int | None = None
    point_cloud_max_points: int = 100_000
    sensor_image_fps: float | None = _DEFAULT_IMAGE_FPS
    soft_body_fps: float = _DEFAULT_SOFT_BODY_FPS
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
