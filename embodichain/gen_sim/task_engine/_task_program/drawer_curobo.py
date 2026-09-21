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

"""Task-owned cuRobo transport with live link geometry and a carried payload.

The private planner world is a fresh snapshot per transport attempt, not a
second simulation scene. E6, pickup and final release keep their existing
motion generator. No attachment or collision object modifies simulator state.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Iterator

import torch
import trimesh

from embodichain.lab.sim.motion.motion_generator import (
    MotionGenCfg,
    MotionGenOptions,
    MotionGenerator,
)
from embodichain.lab.sim.motion.planners.utils import MoveType, PlanState
from embodichain.lab.sim.motion.planners.curobo.curobo_planner import (
    CuroboPlanner,
    CuroboPlannerCfg,
    CuroboWorldCfg,
    CuroboPlanOptions,
    _matrix_to_position_quaternion,
)
from embodichain.lab.sim.objects.rigid_object import CollisionShapeDesc, RigidBodyShape
from embodichain.utils import logger
from embodichain.utils.math import pose_inv
from .drawer_geometry import collision_meshes
from .motion import ApproachMotionGenerator

__all__: list[str] = []

PAYLOAD_LINK = "gen_sim_payload"
PAYLOAD_SPHERES = 27


@dataclass
class LinkCollisionSource:
    """Expose individual collision components in one live articulation-link frame."""

    uid: str
    articulation: Any
    link: str
    meshes: list[trimesh.Trimesh]

    def get_local_pose(self, *, to_matrix: bool = True) -> torch.Tensor:
        return self.articulation.get_link_pose(self.link, to_matrix=to_matrix)

    def get_collision_shapes(self, env_id: int = 0) -> list[CollisionShapeDesc]:
        del env_id
        shapes = []
        for mesh in self.meshes:
            for component in mesh.split(only_watertight=False):
                low, high = component.bounds
                pose = torch.eye(4)
                pose[:3, 3] = torch.as_tensor((low + high) / 2, dtype=torch.float32)
                # Keep panels separate. A convex hull of the whole drawer would
                # incorrectly fill its opening and prevent every insertion.
                shapes.append(
                    CollisionShapeDesc(
                        name=f"component_{len(shapes)}",
                        shape_type=RigidBodyShape.BOX,
                        local_pose=pose,
                        half_extents=torch.as_tensor(
                            (high - low) / 2, dtype=torch.float32
                        ).clamp_min(1e-5),
                    )
                )
        return shapes


@dataclass
class CollisionSnapshot:
    shapes: tuple[CollisionShapeDesc, ...]
    pose: torch.Tensor

    def get_collision_shapes(self, env_id: int = 0) -> list[CollisionShapeDesc]:
        del env_id
        return list(self.shapes)

    def get_local_pose(self, *, to_matrix: bool = True) -> torch.Tensor:
        return self.pose.clone()


def snapshot_in_base(
    sources: dict[str, Any], base_pose: torch.Tensor
) -> dict[str, CollisionSnapshot]:
    """Freeze a fresh world in the same URDF-root frame used for cuRobo goals."""
    inverse = pose_inv(base_pose)
    return {
        name: CollisionSnapshot(
            tuple(source.get_collision_shapes()),
            inverse @ source.get_local_pose(to_matrix=True).to(inverse),
        )
        for name, source in sources.items()
    }


def world_sources(simulation: Any, payload_id: str) -> dict[str, Any]:
    """Exclude only the carried object, not the cabinet or target drawer."""
    sources = {
        uid: simulation.get_rigid_object(uid)
        for uid in simulation.get_rigid_object_uid_list()
        if uid != payload_id
    }
    for uid in simulation.get_articulation_uid_list():
        art = simulation.get_articulation(uid)
        meshes = collision_meshes(
            {"fpath": art.cfg.fpath, "body_scale": art.cfg.body_scale}
        )
        by_link: dict[str, list[trimesh.Trimesh]] = {}
        for (link, _), mesh in meshes.items():
            by_link.setdefault(link, []).append(mesh)
        for link, components in by_link.items():
            key = f"{uid}__{link}"
            if key in sources:
                raise ValueError(f"Duplicate drawer collision source {key!r}.")
            sources[key] = LinkCollisionSource(key, art, link, components)
    return sources


def payload_spheres(vertices: torch.Tensor) -> torch.Tensor:
    """Cover the object-local bounding box with 3x3x3 circumscribed cell spheres."""
    points = torch.as_tensor(vertices, dtype=torch.float32)
    low, high = points.amin(0), points.amax(0)
    cell = (high - low) / 3
    index = torch.arange(3, dtype=points.dtype, device=points.device) + 0.5
    grid = torch.stack(
        torch.meshgrid(index, index, index, indexing="ij"), dim=-1
    ).reshape(-1, 3)
    centers = low + grid * cell
    radius = torch.linalg.vector_norm(cell) / 2
    return torch.cat((centers, radius.expand(len(centers), 1)), dim=1)


def payload_robot_config(
    config: dict[str, Any], observed_joints: dict[str, float]
) -> dict[str, Any]:
    """Reserve a separate fixed link; never overwrite the gripper's own spheres."""
    result = deepcopy(config)
    kinematics = result["robot_cfg"]["kinematics"]
    parent = kinematics["tool_frames"][0]
    kinematics["extra_links"] = dict(kinematics.get("extra_links") or {})
    kinematics["extra_links"][PAYLOAD_LINK] = {
        "link_name": PAYLOAD_LINK,
        "joint_name": f"{PAYLOAD_LINK}_joint",
        "joint_type": "FIXED",
        "parent_link_name": parent,
        "fixed_transform": [0, 0, 0, 1, 0, 0, 0],
    }
    kinematics["extra_collision_spheres"] = dict(
        kinematics.get("extra_collision_spheres") or {}
    )
    kinematics["extra_collision_spheres"][PAYLOAD_LINK] = PAYLOAD_SPHERES
    kinematics["collision_link_names"] = [
        *kinematics["collision_link_names"],
        PAYLOAD_LINK,
    ]
    kinematics["lock_joints"] = {
        name: observed_joints[name] for name in kinematics.get("lock_joints", {})
    }
    return result


def attachment_models(planner: Any) -> tuple[Any, ...]:
    """Find every distinct native sphere buffer used by IK, optimization and metrics."""
    models = [planner.kinematics]
    rollouts = []
    for solver in (planner.ik_solver, planner.trajopt_solver):
        rollouts.extend(solver.get_all_rollout_instances())
    rollouts.extend(planner.trajopt_solver.additional_metrics_rollouts.values())
    if planner.graph_planner is not None:
        rollouts.extend(planner.graph_planner.get_all_rollout_instances())
    for rollout in rollouts:
        models.extend(
            (
                rollout.transition_model.robot_model,
                rollout.metrics_transition_model.robot_model,
            )
        )
    distinct = {}
    for model in models:
        spheres = model.config.kinematics_config.link_spheres
        distinct[(str(spheres.device), spheres.data_ptr())] = model
    return tuple(distinct.values())


class _PayloadPlanner(CuroboPlanner):
    """Bind the native attachment API within one task-owned planning lifetime."""

    def __init__(
        self,
        cfg: CuroboPlannerCfg,
        *,
        observation: Any,
        motion: Any,
        qpos: torch.Tensor,
    ) -> None:
        self._observation, self._motion, self._qpos = observation, motion, qpos.clone()
        self._attachments: dict[int, tuple[Any, ...]] = {}
        super().__init__(cfg)

    def _load_runtime_robot_config(self, path: str) -> dict[str, Any]:
        config = super()._load_runtime_robot_config(path)
        joints = dict(
            zip(
                self.robot.joint_names,
                self._qpos[0].detach().cpu().tolist(),
                strict=True,
            )
        )
        return payload_robot_config(config, joints)

    def _materialize_profile(self, control_part: str) -> Any:
        profile = super()._materialize_profile(control_part)
        self._snapshot_base_pose = self.robot.get_link_pose(
            profile.sim_base_link_name, to_matrix=True
        ).clone()
        return profile

    def _auto_generate_world_scene(self, world_cfg: CuroboWorldCfg) -> Any:
        # The installed native updater does not route voxel obstacles by name.
        # This planner is scoped to one attempt, so bake its live base-frame snapshot.
        sources = snapshot_in_base(world_cfg.rigid_objects, self._snapshot_base_pose)
        return super()._auto_generate_world_scene(
            replace(world_cfg, rigid_objects=sources)
        )

    def _get_backend(
        self,
        control_part: str,
        batch_size: int,
        planning_mode: MoveType = MoveType.EEF_MOVE,
    ) -> Any:
        backend = super()._get_backend(control_part, batch_size, planning_mode)
        if id(backend) not in self._attachments:
            from curobo._src.collision.attachment_manager import AttachmentManager

            obj_pose = self._observation.obj.get_local_pose(to_matrix=True).to(
                self._curobo_device
            )
            base_pose = self._sim_world_to_curobo_base_pose(obj_pose, backend)
            position, quaternion = _matrix_to_position_quaternion(base_pose)
            native_pose = self._bindings.Pose(position=position, quaternion=quaternion)
            state = self._to_curobo_joint_state(
                self._qpos[:, list(self._motion.joint_ids)], backend
            )
            spheres = payload_spheres(torch.as_tensor(self._observation.vertices)).to(
                self._curobo_device
            )
            # cuRobo 0.8's planner property delegates to an absent solver property.
            # Use its existing component directly and cover all distinct rollouts.
            managers = tuple(
                AttachmentManager(
                    model,
                    device_cfg=self._bindings.DeviceCfg(device=self._curobo_device),
                )
                for model in attachment_models(backend.planner)
            )
            self._attachments[id(backend)] = managers
            for manager in managers:
                manager.update(
                    spheres,
                    state,
                    link_name=PAYLOAD_LINK,
                    world_objects_pose_offset=native_pose,
                )
            logger.log_info(
                f"GenSim cuRobo attached {self._observation.route.object_id!r}: {PAYLOAD_SPHERES} payload spheres on {PAYLOAD_LINK!r}."
            )
        return backend

    def close(self) -> None:
        try:
            for managers in self._attachments.values():
                for manager in managers:
                    manager.detach(link_name=PAYLOAD_LINK)
        finally:
            self._attachments.clear()
            super().close()


class DrawerMotionGenerator(ApproachMotionGenerator):
    """Use cuRobo only inside the explicit held-transport planning scope."""

    def __init__(self, cfg: MotionGenCfg, *, simulation: Any) -> None:
        super().__init__(cfg)
        self._simulation = simulation
        self._transport = None
        self._placement_path = None
        self.transport_metadata: dict[str, Any] = {}

    @contextmanager
    def placement_path(self, waypoints: torch.Tensor) -> Iterator[None]:
        """Preserve Place's phase/effect handling with a reverse entry retreat."""
        self._placement_path = torch.cat((waypoints, waypoints[:, :2].flip(1)), dim=1)
        try:
            yield
        finally:
            self._placement_path = None

    @contextmanager
    def transport(self, observation: Any, motion: Any, context: Any) -> Iterator[None]:
        if self._transport is not None:
            raise RuntimeError("Drawer transport planning scopes cannot overlap.")
        # Preparation is intentionally lazy: opening and pickup do not initialize CUDA planning.
        self._transport = (observation, motion, context)
        self.transport_metadata = {}
        try:
            yield
        finally:
            self._transport = None

    def generate(
        self, target_states: list[Any], options: MotionGenOptions | None = None
    ) -> Any:
        if self._transport is None:
            if self._placement_path is not None:
                target_states = [
                    PlanState(move_type=MoveType.EEF_MOVE, xpos=p)
                    for p in self._placement_path.unbind(1)
                ]
            return super().generate(target_states, options)
        observation, motion, context = self._transport
        sources = world_sources(self._simulation, observation.route.object_id)
        cfg = CuroboPlannerCfg(
            robot_uid=self.robot.uid,
            sim_instance_id=self.planner.cfg.sim_instance_id,
            world=CuroboWorldCfg(
                rigid_objects=sources,
                # The reference table needs 2.38M cells at the unchanged 1 cm resolution.
                max_voxel_count=4_000_000,
            ),
            use_cuda_graph=False,
            warmup_iterations=0,
            interpolation_dt=context.require_control_dt(),
            preserve_plan_samples=True,
        )
        generator = _PayloadMotionGenerator(
            cfg, observation=observation, motion=motion, qpos=context.robot.qpos
        )
        planner = generator.planner
        self.transport_metadata = {
            "backend": "curobo",
            "world_ids": list(sources),
            "payload": observation.route.object_id,
            "payload_spheres": PAYLOAD_SPHERES,
            "world_update": "live_snapshot_per_plan",
            "simulator_attachment": False,
            "articulation_geometry": "link_local_collision_component_boxes",
            "payload_geometry": "object_local_box_sphere_cover",
        }
        logger.log_info(f"GenSim cuRobo drawer transport: {self.transport_metadata}")
        try:
            native_options = deepcopy(options)
            native_options.strategy = "motion_gen"
            native_options.sample_count = None
            native_options.plan_opts = CuroboPlanOptions()
            result = generator.generate(target_states, native_options)
            self.transport_metadata["plan_success"] = (
                result.success.detach().cpu().tolist()
            )
            return result
        finally:
            planner.close()


class _PayloadMotionGenerator(MotionGenerator):
    def __init__(self, cfg: CuroboPlannerCfg, **payload: Any) -> None:
        self._payload = payload
        super().__init__(MotionGenCfg(planner_cfg=cfg))

    def _create_planner(self, planner_cfg: CuroboPlannerCfg) -> _PayloadPlanner:
        return _PayloadPlanner(planner_cfg, **self._payload)
