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
import math

from embodichain.utils import configclass
from embodichain.cli._visualization import (
    _DEFAULT_HOST,
    _DEFAULT_PORT,
    _DEFAULT_SCENE_FPS,
    _DEFAULT_IMAGE_FPS,
    _DEFAULT_SOFT_BODY_FPS,
    _DEFAULT_ENV_IDS,
)

__all__ = ["VisualizationCfg", "ViserServerCfg", "DexSimWebCfg"]


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
class DexSimWebCfg:
    """Configure the experimental native WebRTC window replacement.

    Requires a Linux/NVIDIA DexSim build with WebApplication and NVENC.
    The native listener binds all IPv4 interfaces and lives until World teardown.
    One viewer and one simulation environment are supported initially.

    Args:
        port: Native HTTP/signaling port on a trusted network.
        width: Main observation view width, even and between 32 and 4096.
        height: Main observation view height, even and between 32 and 4096.
        fps: Maximum application rendering rate, independent of physics time.
        eye: Initial observation camera location in world coordinates.
        target: Initial observation camera look-at target.
        ice: Optional DexSim ICE configuration for STUN/TURN connectivity.
    """

    port: int = 8765
    width: int = 1280
    height: int = 720
    fps: float = 15.0
    eye: tuple[float, float, float] = (2.5, -3.5, 2.0)
    target: tuple[float, float, float] = (0.0, 0.0, 0.6)
    ice: dict[str, object] | None = None

    def __post_init__(self) -> None:
        """Validate settings before allocating native resources."""
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("WebRTC port must be between 1 and 65535.")
        for name in ("width", "height"):
            value = getattr(self, name)
            if type(value) is not int or not 32 <= value <= 4096 or value % 2:
                raise ValueError(f"WebRTC {name} must be even and in [32, 4096].")
        if not math.isfinite(self.fps) or not 0 < self.fps <= 30:
            raise ValueError("WebRTC fps must be finite and in (0, 30].")
        for name in ("eye", "target"):
            value = getattr(self, name)
            if len(value) != 3 or not all(math.isfinite(v) for v in value):
                raise ValueError(
                    f"WebRTC {name} must contain three finite coordinates."
                )
        if tuple(self.eye) == tuple(self.target):
            raise ValueError("WebRTC eye and target must differ.")


@configclass
class VisualizationCfg:
    """Configure live scene visualization.

    Args:
        backend: Visualization backend name. Supported values are ``"none"``,
            ``"viser"`` and ``"dexsim-web"``.
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
            Enables Viser Gizmos/joint controls or native WebRTC entity editing.
            Read-only WebRTC clients can still navigate the observer camera.
            Keep disabled for untrusted or publicly reachable browser sessions.
        viser_server: Viser HTTP/WebSocket server binding settings.
        web: Native WebRTC observation-view settings.
    """

    web: DexSimWebCfg = field(default_factory=DexSimWebCfg)
    """Native WebRTC observation-view settings."""

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
        if isinstance(self.web, dict):
            self.web = DexSimWebCfg(**self.web)
        if self.backend not in {"none", "viser", "dexsim-web"}:
            raise ValueError(
                f"Unsupported visualization backend {self.backend!r}; expected 'none', 'viser' or 'dexsim-web'."
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
        if self.allow_commands and self.backend == "none":
            raise ValueError("allow_commands requires Viser or dexsim-web.")
        if self.backend == "dexsim-web" and self.env_ids != [0]:
            raise ValueError("dexsim-web currently requires env_ids=[0].")
