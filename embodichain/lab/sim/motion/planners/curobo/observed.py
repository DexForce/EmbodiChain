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

"""Fresh static cuRobo worlds from explicit observation-bound planning inputs."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, runtime_checkable
import torch

from embodichain.lab.sim.objects.rigid_object import CollisionShapeDesc
from embodichain.lab.sim.scene._validation import _owned_b1_pose
from embodichain.lab.sim.scene.collision_sources import snapshot_collision_sources
from embodichain.utils.math import pose_inv
from .payload import CuroboPayload, CuroboQuerySnapshot, curobo_motion_generator
from .curobo_planner import CuroboPlanner, CuroboPlannerCfg

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.motion.motion_generator import MotionGenerator

__all__ = [
    "CollisionShapeReader",
    "CuroboWorldObservation",
    "ObservedCuroboQueryFactory",
]


@runtime_checkable
class CollisionShapeReader(Protocol):
    """Geometry-only provider for explicitly declared collision owners."""

    def capture(
        self, *, excluded_entity_ids: tuple[str, ...] = ()
    ) -> Mapping[str, Sequence[CollisionShapeDesc]]:
        """Read source shapes without observing poses or creating a planner.

        Args:
            excluded_entity_ids: Exact logical IDs excluded by the query.

        Returns:
            Ordered obstacle-local shape descriptors, borrowed read-only until
            the immediate snapshot boundary.
        """
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class CuroboWorldObservation:
    """Owned B1 obstacle poses plus their explicit common observation scope.

    Args:
        timestamp: Common observation time, in the host execution-clock domain.
        scene_version: Semantic scene version captured with those poses.
        native_revision: Actual native observation/source revision, not a renamed
            semantic version. The host must verify it during the capture.
        poses: Canonical collision-source IDs to owner-world poses. Extra
            non-collision entities are permitted; missing selected IDs reject.
    """

    timestamp: float
    scene_version: int
    native_revision: int
    poses: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        if (
            type(self.timestamp) not in (int, float)
            or not math.isfinite(self.timestamp)
            or self.timestamp < 0
        ):
            raise ValueError(
                "World observation timestamp must be finite and nonnegative."
            )
        if any(
            type(value) is not int or value < 0
            for value in (self.scene_version, self.native_revision)
        ):
            raise ValueError(
                "World observation revisions must be nonnegative integers."
            )
        if not isinstance(self.poses, Mapping):
            raise TypeError("World observation poses must be an explicit mapping.")
        if any(
            type(key) is not str or not key or key != key.strip() for key in self.poses
        ):
            raise ValueError("World observation IDs must be canonical.")
        object.__setattr__(
            self,
            "poses",
            MappingProxyType(
                {
                    key: _owned_b1_pose(pose, f"World pose {key}")
                    for key, pose in self.poses.items()
                }
            ),
        )


class ObservedCuroboQueryFactory:
    """Open a fresh standard motion facade without reading live robot/world poses.

    Args:
        robot: Exact robot planned by the host engine; native ownership remains
            with the host. Only static robot identities are consulted here.
        planner_cfg: Explicit single-world, no-graph, preserved-sample cuRobo
            configuration. Its world must not declare a second set of sources.
        shape_reader: Geometry-only source adapter for the declared world roster.

    The host owns observation coherence, native revision validation and explicit
    body exclusions. This service does not select phases, targets, sampling
    alternatives, RNG scope, acceleration overrides or success criteria.
    """

    def __init__(
        self,
        robot: Robot,
        planner_cfg: CuroboPlannerCfg,
        shape_reader: CollisionShapeReader,
    ) -> None:
        if type(planner_cfg) is not CuroboPlannerCfg or not isinstance(
            shape_reader, CollisionShapeReader
        ):
            raise TypeError(
                "Observed queries require a cuRobo config and typed shape reader."
            )
        cfg = deepcopy(planner_cfg)
        cfg.validate()
        if cfg.robot_uid != robot.uid or robot.num_instances != 1:
            raise ValueError("Observed queries require the exact configured B1 robot.")
        if cfg.world.rigid_objects is not None:
            raise ValueError(
                "Observed query geometry must have one shape-reader owner."
            )
        CuroboPlanner._validate_query_cfg(cfg)
        if cfg.sim_base_to_curobo_base is not None:
            transform = torch.as_tensor(
                cfg.sim_base_to_curobo_base, dtype=torch.float64
            )
            if transform.shape != (4, 4) or not torch.equal(
                transform, torch.eye(4, dtype=transform.dtype)
            ):
                raise ValueError(
                    "Observed root poses require the unshifted generated URDF-root convention."
                )
        self._robot, self._cfg, self._shape_reader = robot, cfg, shape_reader

    @property
    def robot(self) -> Robot:
        """Return the exact borrowed robot for service-owner validation."""
        return self._robot

    @property
    def planning_device(self) -> torch.device:
        """Return the device selected by the existing cuRobo backend resolver."""
        from .curobo_planner import _resolve_curobo_device

        return _resolve_curobo_device(
            self._cfg.cuda_device, torch.device(self._robot.device)
        )

    @contextmanager
    def open(
        self,
        *,
        control_part: str,
        full_qpos: torch.Tensor,
        root_pose: torch.Tensor,
        root_link_name: str,
        observed_at: float,
        world: CuroboWorldObservation,
        control_dt: float,
        payload: CuroboPayload | None = None,
        world_metadata: dict[str, object] | None = None,
    ) -> Iterator[MotionGenerator]:
        """Project one complete same-observation world and own its backend lifetime.

        Args:
            control_part: Explicit active control group.
            full_qpos: Full start/passive-joint planning state in native joint order.
                An owning Atomic phase may qualify explicit open-hand values;
                these remain planning inputs, never physical evidence.
            root_pose: Captured world pose of the generated URDF root, B1/4x4.
            root_link_name: Exact identity of that root, never a solver subchain.
            observed_at: Timestamp of the robot/root observation.
            world: Same-timestamp obstacle poses with semantic and native revisions.
            control_dt: Positive command cadence in seconds.
            world_metadata: Optional caller-owned diagnostics receiving the actually selected collision-source roster.
            payload: Optional explicitly selected body envelope/pose; projected
                held poses remain planning-only and the caller records their origin.

        Yields:
            A fresh standard MotionGenerator. Native samples and owned outputs
            survive close; borrowed robot and scene objects are never destroyed.
        """
        if type(world) is not CuroboWorldObservation:
            raise TypeError("world must be a CuroboWorldObservation.")
        if (
            type(observed_at) not in (int, float)
            or not math.isfinite(observed_at)
            or observed_at != world.timestamp
        ):
            raise ValueError("Robot and world must share the exact observation time.")
        if (
            type(control_dt) not in (int, float)
            or not math.isfinite(control_dt)
            or control_dt <= 0
        ):
            raise ValueError("control_dt must be finite and positive.")
        if root_link_name != self._robot.root_link_name:
            raise ValueError(
                "Observed query root must identify the whole robot URDF root."
            )
        if control_part not in self._robot.control_parts:
            raise ValueError(
                "Observed query control part is not declared by the robot."
            )
        if not isinstance(full_qpos, torch.Tensor) or full_qpos.shape != (
            1,
            len(self._robot.joint_names),
        ):
            raise ValueError(
                "Observed full qpos must match the complete B1 joint order."
            )
        if payload is not None and payload.control_part != control_part:
            raise ValueError("Payload and query control parts disagree.")
        snapshot = CuroboQuerySnapshot(
            full_qpos=full_qpos,
            world_to_base=pose_inv(_owned_b1_pose(root_pose, "Observed robot root")),
            payload=payload,
            base_link_name=root_link_name,
        )
        excluded = () if payload is None else (payload.object_id,)
        shapes = self._shape_reader.capture(excluded_entity_ids=excluded)
        missing = set(shapes) - set(world.poses)
        if missing:
            raise ValueError(
                f"Observed collision poses are missing declared sources: {sorted(missing)}"
            )
        sources = snapshot_collision_sources(
            shapes,
            {key: world.poses[key] for key in shapes},
            snapshot.world_to_base,
            provenance=getattr(self._shape_reader, "provenance", None),
        )
        if world_metadata is not None:
            world_metadata["world_ids"] = list(sources)
        cfg = deepcopy(self._cfg)
        cfg.world.rigid_objects = sources
        cfg.interpolation_dt = float(control_dt)
        with curobo_motion_generator(cfg, query_snapshot=snapshot) as generator:
            if generator.robot is not self._robot:
                raise ValueError(
                    "Observed query factory resolved a different native robot."
                )
            yield generator
