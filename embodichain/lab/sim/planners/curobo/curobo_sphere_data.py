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

"""Analytic sphere-obstacle storage for cuRobo V2's generic Warp checker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import warp as wp

from curobo._src.geom.data.helper_pose import (
    get_obs_idx,
    load_transform_from_inv_pose,
)
from curobo._src.util.logging import log_and_raise

if TYPE_CHECKING:
    from curobo._src.geom.types import SceneCfg, Sphere
    from curobo._src.types.device_cfg import DeviceCfg
    from curobo._src.types.pose import Pose

__all__ = [
    "SphereData",
    "SphereDataWarp",
    "compute_local_sdf",
    "compute_local_sdf_with_grad",
    "is_obs_enabled",
    "load_obstacle_transform",
]

_SDF_EPS = 1.0e-8


@wp.struct
class SphereDataWarp:
    """Warp view of batched analytic sphere obstacles."""

    radius: wp.array(dtype=wp.float32)
    inv_pose: wp.array2d(dtype=wp.float32)
    enable: wp.array(dtype=wp.uint8)
    n_per_env: wp.array(dtype=wp.int32)
    max_n: wp.int32
    num_envs: wp.int32


@dataclass
class SphereData:
    """GPU tensor storage for analytic sphere obstacles."""

    radius: torch.Tensor
    inv_pose: torch.Tensor
    enable: torch.Tensor
    count: torch.Tensor
    names: list[list[str | None]]
    max_n: int
    num_envs: int
    device_cfg: "DeviceCfg"

    @classmethod
    def create_cache(
        cls, max_n: int, num_envs: int, device_cfg: "DeviceCfg"
    ) -> "SphereData":
        """Create an empty fixed-capacity sphere cache."""
        radius = torch.zeros(
            (num_envs, max_n),
            dtype=device_cfg.dtype,
            device=device_cfg.device,
        )
        inv_pose = torch.zeros(
            (num_envs, max_n, 8),
            dtype=device_cfg.dtype,
            device=device_cfg.device,
        )
        inv_pose[..., 3] = 1.0
        enable = torch.zeros(
            (num_envs, max_n), dtype=torch.uint8, device=device_cfg.device
        )
        count = torch.zeros((num_envs,), dtype=torch.int32, device=device_cfg.device)
        return cls(
            radius=radius,
            inv_pose=inv_pose,
            enable=enable,
            count=count,
            names=[[None for _ in range(max_n)] for _ in range(num_envs)],
            max_n=max_n,
            num_envs=num_envs,
            device_cfg=device_cfg,
        )

    @classmethod
    def from_scene_cfg(
        cls,
        scene_cfg: "SceneCfg",
        device_cfg: "DeviceCfg",
        env_idx: int = 0,
        num_envs: int = 1,
        max_n: int | None = None,
    ) -> "SphereData":
        """Create storage from one cuRobo scene."""
        spheres = scene_cfg.sphere or []
        capacity = max_n if max_n is not None else max(len(spheres), 1)
        instance = cls.create_cache(capacity, num_envs, device_cfg)
        if spheres:
            instance.load_batch(spheres, env_idx)
        return instance

    @classmethod
    def from_batch_scene_cfg(
        cls,
        scene_cfg_list: list["SceneCfg"],
        device_cfg: "DeviceCfg",
        max_n: int | None = None,
    ) -> "SphereData":
        """Create storage from independent batched scenes."""
        num_envs = len(scene_cfg_list)
        counts = [len(scene.sphere or []) for scene in scene_cfg_list]
        capacity = max_n if max_n is not None else max(max(counts), 1)
        instance = cls.create_cache(capacity, num_envs, device_cfg)
        for env_idx, scene in enumerate(scene_cfg_list):
            if scene.sphere:
                instance.load_batch(scene.sphere, env_idx)
        return instance

    def load_batch(self, spheres: list["Sphere"], env_idx: int) -> None:
        """Replace one environment's sphere obstacles."""
        if len(spheres) > self.max_n:
            log_and_raise(
                f"Cannot load {len(spheres)} spheres, max cache size is {self.max_n}"
            )
        if not spheres:
            self.clear(env_idx)
            return
        num_spheres = len(spheres)
        centers = torch.as_tensor(
            [sphere.pose[:3] for sphere in spheres],
            dtype=self.device_cfg.dtype,
            device=self.device_cfg.device,
        )
        inverse_poses = torch.zeros(
            (num_spheres, 7),
            dtype=self.device_cfg.dtype,
            device=self.device_cfg.device,
        )
        # Sphere orientation is immaterial. Identity rotation plus translated
        # origin is the exact world-to-local transform and avoids cuRobo's
        # CUDA-only generic pose-inverse kernel.
        inverse_poses[:, :3] = -centers
        inverse_poses[:, 3] = 1.0
        self.radius[env_idx, :num_spheres] = torch.as_tensor(
            [sphere.radius for sphere in spheres],
            dtype=self.device_cfg.dtype,
            device=self.device_cfg.device,
        )
        self.inv_pose[env_idx, :num_spheres, :7] = inverse_poses
        self.enable[env_idx, :num_spheres] = 1
        self.enable[env_idx, num_spheres:] = 0
        self.names[env_idx][:num_spheres] = [sphere.name for sphere in spheres]
        self.names[env_idx][num_spheres:] = [None] * (self.max_n - num_spheres)
        self.count[env_idx] = num_spheres

    def update_pose(self, name: str, pose: Pose, env_idx: int = 0) -> None:
        """Update a named sphere pose."""
        idx = self.get_idx(name, env_idx)
        position = pose.position.reshape(-1, 3)[0].to(self.inv_pose)
        self.inv_pose[env_idx, idx, :3] = -position
        self.inv_pose[env_idx, idx, 3:7] = torch.as_tensor(
            [1.0, 0.0, 0.0, 0.0],
            dtype=self.device_cfg.dtype,
            device=self.device_cfg.device,
        )

    def set_enabled(self, name: str, enabled: bool, env_idx: int = 0) -> None:
        """Enable or disable a named sphere."""
        self.enable[env_idx, self.get_idx(name, env_idx)] = int(enabled)

    def get_idx(self, name: str, env_idx: int = 0) -> int:
        """Return a named sphere's local index."""
        try:
            return self.names[env_idx].index(name)
        except ValueError:
            log_and_raise(
                f"Sphere with name '{name}' not found in environment {env_idx}"
            )
        raise AssertionError("unreachable")

    def get_names(self, env_idx: int = 0) -> list[str]:
        """Return active sphere names."""
        return self.names[env_idx][: int(self.count[env_idx].item())]

    def clear(self, env_idx: int | None = None) -> None:
        """Disable all spheres in one or every environment."""
        if env_idx is None:
            self.enable.zero_()
            self.count.zero_()
            self.names = [
                [None for _ in range(self.max_n)] for _ in range(self.num_envs)
            ]
        else:
            self.enable[env_idx].zero_()
            self.count[env_idx] = 0
            self.names[env_idx] = [None for _ in range(self.max_n)]

    def to_warp(self) -> SphereDataWarp:
        """Return the Warp view consumed by cuRobo's generic kernels."""
        data = SphereDataWarp()
        data.radius = wp.from_torch(self.radius.view(-1), dtype=wp.float32)
        data.inv_pose = wp.from_torch(self.inv_pose.view(-1, 8), dtype=wp.float32)
        data.enable = wp.from_torch(self.enable.view(-1), dtype=wp.uint8)
        data.n_per_env = wp.from_torch(self.count.view(-1), dtype=wp.int32)
        data.max_n = self.max_n
        data.num_envs = self.num_envs
        return data


def is_obs_enabled(
    obs_set: SphereDataWarp, env_idx: wp.int32, local_idx: wp.int32
) -> wp.bool:
    """Return whether a sphere slot is active."""
    flat_idx = get_obs_idx(env_idx, local_idx, obs_set.max_n)
    return obs_set.enable[flat_idx] == wp.uint8(1)


def load_obstacle_transform(
    obs_set: SphereDataWarp, env_idx: wp.int32, local_idx: wp.int32
) -> wp.transform:
    """Load a sphere's world-to-local transform."""
    flat_idx = get_obs_idx(env_idx, local_idx, obs_set.max_n)
    return load_transform_from_inv_pose(obs_set.inv_pose, flat_idx)


def compute_local_sdf(
    obs_set: SphereDataWarp,
    env_idx: wp.int32,
    local_idx: wp.int32,
    local_pt: wp.vec3,
) -> wp.float32:
    """Return analytic signed distance to a sphere surface."""
    flat_idx = get_obs_idx(env_idx, local_idx, obs_set.max_n)
    return wp.length(local_pt) - obs_set.radius[flat_idx]


def compute_local_sdf_with_grad(
    obs_set: SphereDataWarp,
    env_idx: wp.int32,
    local_idx: wp.int32,
    local_pt: wp.vec3,
) -> wp.vec4:
    """Return analytic sphere SDF and negative spatial gradient."""
    flat_idx = get_obs_idx(env_idx, local_idx, obs_set.max_n)
    distance = wp.length(local_pt)
    gx = wp.float32(0.0)
    gy = wp.float32(0.0)
    gz = wp.float32(0.0)
    if distance > _SDF_EPS:
        inverse_distance = -1.0 / distance
        gx = local_pt[0] * inverse_distance
        gy = local_pt[1] * inverse_distance
        gz = local_pt[2] * inverse_distance
    return wp.vec4(distance - obs_set.radius[flat_idx], gx, gy, gz)
