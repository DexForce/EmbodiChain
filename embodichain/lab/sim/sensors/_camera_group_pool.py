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

"""Scene-owned camera groups shared by compatible startup sensors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import dexsim
from dexsim.types import Renderer

if TYPE_CHECKING:
    from dexsim.scene import Scene, SpawnedCameraGroup

    from .base_sensor import SensorCfg
    from .camera import CameraCfg

__all__: list[str] = []

_HYBRID_GROUP_CAPACITY = 32


@dataclass(frozen=True)
class _CameraGroupSlice:
    group: SpawnedCameraGroup
    slot_start: int
    count: int


@dataclass(frozen=True)
class _GroupKey:
    width: int
    height: int
    view_flags: int
    include_visual_overlays: bool
    enable_color: bool
    enable_depth: bool
    enable_mask: bool
    enable_normal: bool
    enable_position: bool
    enable_disparity: bool


@dataclass(frozen=True)
class _CameraGroupLease:
    bucket_id: int
    uid: str
    slices: tuple[_CameraGroupSlice, ...]

    @property
    def groups(self) -> tuple[SpawnedCameraGroup, ...]:
        """Return the groups touched by this sensor in view order."""
        return tuple(item.group for item in self.slices)

    @property
    def camera_groups(self) -> tuple[SpawnedCameraGroup, ...]:
        """Return one native group for each of this sensor's view rows."""
        return tuple(item.group for item in self.slices for _ in range(item.count))


@dataclass
class _GroupBucket:
    bucket_id: int
    key: _GroupKey
    total_views: int
    name: str
    groups: tuple[SpawnedCameraGroup, ...] = ()
    capacity: int = 0
    next_slot: int = 0
    active_uids: set[str] = field(default_factory=set)
    scene: Scene | None = None


class _CameraGroupPool:
    """Own fixed-capacity Scene groups and stable sensor view ranges."""

    def __init__(self, num_envs: int) -> None:
        self._num_envs = num_envs
        self._planned: dict[str, tuple[int, int]] = {}
        self._consumed_planned: set[str] = set()
        self._buckets: dict[int, _GroupBucket] = {}
        self._by_key: dict[_GroupKey, int] = {}
        self._next_id = 0

    @staticmethod
    def _key(cfg: CameraCfg) -> _GroupKey:
        return _GroupKey(
            width=cfg.width,
            height=cfg.height,
            view_flags=int(cfg.get_view_attrib()),
            include_visual_overlays=cfg.include_visual_overlays,
            enable_color=cfg.enable_color,
            enable_depth=cfg.enable_depth,
            enable_mask=cfg.enable_mask,
            enable_normal=cfg.enable_normal,
            enable_position=cfg.enable_position,
            enable_disparity=bool(getattr(cfg, "enable_disparity", False)),
        )

    def plan(self, configs: Sequence[SensorCfg]) -> None:
        """Reserve capacities for every known startup camera before creation."""
        from .camera import CameraCfg
        from .stereo import StereoCameraCfg

        if self._planned or self._buckets:
            raise RuntimeError("Camera groups were already planned or created.")
        for cfg in configs:
            if not isinstance(cfg, CameraCfg) or cfg.uid is None:
                continue
            if cfg.uid in self._planned:
                raise ValueError(f"Duplicate planned camera UID {cfg.uid!r}.")
            key = self._key(cfg)
            count = self._num_envs * (2 if isinstance(cfg, StereoCameraCfg) else 1)
            bucket_id = self._by_key.get(key)
            if bucket_id is None:
                bucket_id = self._next_id
                self._next_id += 1
                self._by_key[key] = bucket_id
                self._buckets[bucket_id] = _GroupBucket(
                    bucket_id, key, 0, f"__embodichain_camera_pool_{bucket_id}"
                )
            self._buckets[bucket_id].total_views += count
            self._planned[cfg.uid] = bucket_id, count

    def acquire(
        self, cfg: CameraCfg, view_count: int, scene: Scene
    ) -> _CameraGroupLease:
        """Reserve consecutive native slots for one sensor."""
        if cfg.uid is None:
            raise ValueError("Camera group ownership requires a sensor UID.")
        planned = (
            None if cfg.uid in self._consumed_planned else self._planned.get(cfg.uid)
        )
        if planned is None:
            # Camera additions after startup keep their independent targets.
            bucket_id = self._next_id
            self._next_id += 1
            bucket = _GroupBucket(
                bucket_id,
                self._key(cfg),
                view_count,
                f"__embodichain_camera_pool_{bucket_id}",
            )
            self._buckets[bucket_id] = bucket
        else:
            bucket_id, expected = planned
            bucket = self._buckets[bucket_id]
            if view_count != expected or self._key(cfg) != bucket.key:
                raise ValueError(f"Camera {cfg.uid!r} differs from its group plan.")
        if cfg.uid in bucket.active_uids:
            raise ValueError(f"Camera {cfg.uid!r} already owns group slots.")
        if not bucket.groups:
            self._create_groups(bucket, scene, cfg)
        if bucket.scene is not scene:
            raise RuntimeError("A camera group cannot move to another Scene.")
        start = bucket.next_slot
        end = start + view_count
        if end > bucket.total_views:
            raise RuntimeError("Camera group plan has no remaining view slots.")
        slices: list[_CameraGroupSlice] = []
        cursor = start
        while cursor < end:
            group_index, slot_start = divmod(cursor, bucket.capacity)
            count = min(bucket.capacity - slot_start, end - cursor)
            slices.append(
                _CameraGroupSlice(bucket.groups[group_index], slot_start, count)
            )
            cursor += count
        bucket.next_slot = end
        bucket.active_uids.add(cfg.uid)
        if planned is not None:
            self._consumed_planned.add(cfg.uid)
        return _CameraGroupLease(bucket_id, cfg.uid, tuple(slices))

    def _create_groups(
        self, bucket: _GroupBucket, scene: Scene, cfg: CameraCfg
    ) -> None:
        capacity = (
            _HYBRID_GROUP_CAPACITY
            if dexsim.get_world_config().renderer == Renderer.HYBRID
            else bucket.total_views
        )
        groups: list[SpawnedCameraGroup] = []
        try:
            for start in range(0, bucket.total_views, capacity):
                count = min(capacity, bucket.total_views - start)
                groups.append(
                    scene.create_camera_group(
                        f"{bucket.name}_{start}",
                        cfg.width,
                        cfg.height,
                        count,
                        view_flags=cfg.get_view_attrib(),
                    )
                )
        except Exception:
            if groups:
                scene.release_camera_groups(groups)
            raise
        bucket.groups = tuple(groups)
        bucket.capacity = capacity
        bucket.scene = scene

    def group_ids_for_bucket(self, lease: _CameraGroupLease) -> tuple[int, ...]:
        """Return the complete render dependency for direct sensor capture."""
        return tuple(
            group.get_group_id() for group in self._buckets[lease.bucket_id].groups
        )

    def is_shared(self, lease: _CameraGroupLease) -> bool:
        """Whether more than one live sensor uses this group bucket."""
        return len(self._buckets[lease.bucket_id].active_uids) > 1

    def release(self, lease: _CameraGroupLease) -> None:
        """Release group resources after the last camera removes its views."""
        bucket = self._buckets[lease.bucket_id]
        bucket.active_uids.discard(lease.uid)
        if bucket.active_uids or not bucket.groups:
            return
        assert bucket.scene is not None
        bucket.scene.release_camera_groups(bucket.groups)
        bucket.groups = ()
        bucket.scene = None
        bucket.next_slot = 0
        if lease.bucket_id not in self._by_key.values():
            self._buckets.pop(lease.bucket_id)
