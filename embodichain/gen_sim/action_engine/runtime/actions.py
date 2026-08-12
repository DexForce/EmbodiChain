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

"""Adapt Action Engine requests to the shared typed atomic-action planner."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
import math
from typing import Any

import torch

from embodichain.gen_sim.action_engine.capabilities import (
    AtomicCapability,
    build_atomic_capability_registry,
)
from embodichain.gen_sim.action_engine.config import default_runtime_policy
from embodichain.lab.sim.atomic_actions import (
    ActionBinding,
    ActionInvocation,
    ActionPlan,
    AntipodalAffordance,
    AtomicActionEngine,
    ControlPartCommandProfile,
    DynamicCollisionMode,
    EntityState,
    MotionPolicy,
    ObjectSemantics,
    PlanningContext,
    RecoveryPolicy,
    RobotObservation,
    SceneSnapshot,
    StateDelta,
)
from embodichain.lab.sim.planners import (
    CuroboPlannerCfg,
    CuroboWorldCfg,
    MotionGenCfg,
    MotionGenerator,
    ToppraPlannerCfg,
)
from embodichain.toolkits.graspkit.pg_grasp import (
    AntipodalSamplerCfg,
    GraspGeneratorCfg,
    GripperCollisionCfg,
)
from embodichain.utils.logger import log_info

from .grasp_collision_cache import ensure_vhacd_grasp_collision_cache
from .models import ActionOutcome, GroundedAction
from .state import ExecutionState

__all__ = ["AtomicActionAdapter"]


_DEFAULT_PLANNER_POLICY: dict[str, Any] = {
    "backend": "curobo",
    "single_arm_strategy": "motion_gen",
    "coordinated_strategy": "ik_interp",
    "fallback_strategy": "ik_interp",
    "allow_fallback": True,
    "dynamic_collision": False,
    "static_obstacle_uids": [],
    "dynamic_obstacle_uids": [],
    "curobo": {
        "log_level": "error",
        "obstacle_representation": "cuboid",
        "multi_env": False,
        "use_cuda_graph": True,
        "preserve_plan_samples": False,
        "max_attempts": 5,
        "collision_activation_distance": 0.01,
    },
}


def _supported_kwargs(config_type: type, values: Mapping[str, Any]) -> dict[str, Any]:
    names: set[str] = set()
    for cls in reversed(config_type.__mro__):
        names.update(getattr(cls, "__annotations__", {}))
    return {key: value for key, value in values.items() if key in names}


def _as_hand_qpos(value: Any, dof: int, device: Any) -> torch.Tensor:
    if dof == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    result = torch.as_tensor(value, dtype=torch.float32, device=device).flatten()
    if result.numel() == 0:
        return torch.zeros(dof, dtype=torch.float32, device=device)
    if result.numel() == 1:
        return result.repeat(dof)
    if result.numel() >= dof:
        return result[:dof]
    repeats = (dof + result.numel() - 1) // result.numel()
    return result.repeat(repeats)[:dof]


def _diagonal_approach_direction(
    horizontal: torch.Tensor,
    *,
    vertical: float = -1.0,
) -> torch.Tensor:
    """Combine one normalized horizontal role direction with a vertical component."""
    horizontal = horizontal.to(dtype=torch.float32)
    norm = torch.linalg.vector_norm(horizontal)
    if float(norm) <= 1.0e-6:
        raise ValueError("Handover role direction must be non-zero.")
    horizontal = horizontal / norm
    direction = torch.stack(
        (horizontal[0], horizontal[1], horizontal.new_tensor(float(vertical)))
    )
    return direction / torch.linalg.vector_norm(direction)


class AtomicActionAdapter:
    """Own the shared atomic engine and preserve Action Engine runtime contracts."""

    def __init__(
        self,
        env: Any,
        *,
        grasp_policy: Mapping[str, Any] | None = None,
        planner_policy: Mapping[str, Any] | None = None,
        capability_registry: Any | None = None,
    ) -> None:
        self.env = env
        self.num_envs = int(env.num_envs)
        self.device = env.device
        if grasp_policy is None:
            profile = str(getattr(env, "agent_robot_profile", "dual_ur10"))
            grasp_policy = default_runtime_policy(profile).grasp
            grasp_policy = {
                **grasp_policy,
                **(getattr(env, "agent_grasp_runtime_defaults", {}) or {}),
            }
        self.grasp_policy = deepcopy(dict(grasp_policy))
        self.planner_policy = deepcopy(_DEFAULT_PLANNER_POLICY)
        if planner_policy is not None:
            self._merge_planner_policy(self.planner_policy, planner_policy)
        if not self.planner_policy.get("static_obstacle_uids"):
            configured = getattr(env, "agent_static_obstacle_uids", ()) or ()
            if configured:
                self.planner_policy["static_obstacle_uids"] = [
                    str(uid) for uid in configured
                ]
            else:
                get_rigid_object = getattr(env.sim, "get_rigid_object", None)
                if callable(get_rigid_object) and get_rigid_object("table") is not None:
                    self.planner_policy["static_obstacle_uids"] = ["table"]
        self.capabilities = capability_registry or build_atomic_capability_registry()
        self._motion_generator: MotionGenerator | None = None
        self._atomic_engine: AtomicActionEngine | None = None
        self._semantics: dict[str, ObjectSemantics] = {}
        self._scene_version = 0

    @staticmethod
    def _merge_planner_policy(
        target: dict[str, Any],
        update: Mapping[str, Any],
    ) -> None:
        for key, value in update.items():
            if isinstance(value, Mapping) and isinstance(target.get(key), dict):
                AtomicActionAdapter._merge_planner_policy(target[key], value)
            else:
                target[key] = deepcopy(value)

    def initial_state(self) -> ExecutionState:
        """Capture the initial full-robot planning seed."""
        return ExecutionState(last_qpos=self.env.robot.get_qpos().clone())

    def semantics(self, uid: str) -> ObjectSemantics:
        """Build object semantics once while retaining the live entity handle."""
        cached = self._semantics.get(uid)
        if cached is not None:
            return cached
        entity = self.env.sim.get_rigid_object(uid)
        if entity is None:
            raise ValueError(f"Unknown grasp target {uid!r}.")
        vertices = entity.get_vertices(env_ids=[0], scale=True)
        triangles = entity.get_triangles(env_ids=[0])
        if isinstance(vertices, (tuple, list)):
            vertices = vertices[0]
        if isinstance(triangles, (tuple, list)):
            triangles = triangles[0]
        vertices = torch.as_tensor(vertices, dtype=torch.float32)
        triangles = torch.as_tensor(triangles, dtype=torch.int64)
        if vertices.ndim == 3 and vertices.shape[0] == 1:
            vertices = vertices[0]
        if triangles.ndim == 3 and triangles.shape[0] == 1:
            triangles = triangles[0]
        if vertices.ndim != 2 or vertices.shape[-1] != 3 or vertices.numel() == 0:
            raise ValueError(f"Object {uid!r} has invalid mesh vertices.")
        if triangles.ndim != 2 or triangles.shape[-1] != 3 or triangles.numel() == 0:
            raise ValueError(f"Object {uid!r} has invalid mesh triangles.")

        grasp_options = self.grasp_policy
        sampler = AntipodalSamplerCfg(
            n_sample=int(grasp_options["antipodal_n_sample"]),
            max_angle=float(grasp_options["antipodal_max_angle"]),
            max_length=float(grasp_options["max_open_length"]),
            min_length=float(grasp_options["min_open_length"]),
        )
        generator = GraspGeneratorCfg(
            viser_port=int(grasp_options["viser_port"]),
            antipodal_sampler_cfg=sampler,
            max_deviation_angle=float(grasp_options["max_deviation_angle"]),
            n_deviated_approach_directions=1,
        )
        max_hulls = int(grasp_options["max_decomposition_hulls"])
        collision = GripperCollisionCfg(
            max_open_length=float(grasp_options["max_open_length"]),
            finger_length=float(grasp_options["finger_length"]),
            point_sample_dense=float(grasp_options["point_sample_dense"]),
            max_decomposition_hulls=max_hulls,
        )
        cache_result = ensure_vhacd_grasp_collision_cache(
            mesh_vertices=vertices,
            mesh_triangles=triangles,
            max_decomposition_hulls=max_hulls,
        )
        if cache_result.status != "hit":
            log_info(f"Prepared V-HACD grasp cache for {uid!r}: {cache_result.status}.")

        semantics = ObjectSemantics(
            label=uid,
            entity=entity,
            geometry={"mesh_vertices": vertices, "mesh_triangles": triangles},
            affordance=AntipodalAffordance(
                object_label=uid,
                mesh_vertices=vertices,
                mesh_triangles=triangles,
                generator_cfg=generator,
                gripper_collision_cfg=collision,
                force_reannotate=bool(grasp_options["force_grasp_reannotate"]),
            ),
        )
        self._semantics[uid] = semantics
        return semantics

    def plan(
        self,
        grounded: GroundedAction,
        state: ExecutionState | None = None,
    ) -> ActionOutcome:
        """Plan one grounded primitive through the mainline typed contract."""
        capability = self.capabilities.require_executable(grounded.action_class)
        state = state or self.initial_state()
        grounded = self._select_upright_transport_yaw(grounded, state)
        context = self._planning_context(state)
        invocation = self._invocation(grounded, capability)
        plan = self._engine().plan(invocation, context)
        selected_positions = self._positions_with_agent_holds(
            plan,
            grounded,
            capability,
        )
        combined_success = plan.plan_success.to(self.device)
        fallback_plan: ActionPlan | None = None
        use_fallback = torch.zeros_like(combined_success)

        fallback_strategy = self.planner_policy.get("fallback_strategy")
        if (
            bool(self.planner_policy.get("allow_fallback", True))
            and invocation.motion_policy.strategy == "motion_gen"
            and fallback_strategy in {"ik_interp"}
            and not bool(combined_success.all())
        ):
            fallback_policy = replace(
                invocation.motion_policy,
                strategy=str(fallback_strategy),
                dynamic_collision_mode=DynamicCollisionMode.OFF,
                plan_opts=None,
            )
            fallback_plan = self._engine().plan(
                replace(invocation, motion_policy=fallback_policy),
                context,
            )
            fallback_positions = self._positions_with_agent_holds(
                fallback_plan,
                grounded,
                capability,
            )
            use_fallback = ~combined_success & fallback_plan.plan_success.to(
                self.device
            )
            selected_positions = self._merge_plan_rows(
                selected_positions,
                fallback_positions,
                use_fallback,
                state.last_qpos,
            )
            combined_success |= fallback_plan.plan_success.to(self.device)

        options = invocation.skill_options
        if capability.config_materializer == "handover":
            combined_success &= self._handover_receiver_hold_mask(
                selected_positions,
                grounded,
                options,
                tolerance=float(
                    grounded.motion_policy.get(
                        "receiver_hold_joint_tolerance",
                        2.0e-3,
                    )
                ),
            )

        terminal_qpos = (
            selected_positions[:, -1]
            if selected_positions.shape[1]
            else state.last_qpos
        )
        primary_rows = combined_success & plan.plan_success.to(self.device)
        projected_task = plan.expected_effects.apply(
            context.task,
            primary_rows,
        )
        held_keys = set(plan.expected_effects.held_object_updates)
        coordinated_keys = set(plan.expected_effects.coordinated_held_object_updates)
        if fallback_plan is not None:
            fallback_rows = combined_success & use_fallback
            projected_task = fallback_plan.expected_effects.apply(
                projected_task,
                fallback_rows,
            )
            held_keys.update(fallback_plan.expected_effects.held_object_updates)
            coordinated_keys.update(
                fallback_plan.expected_effects.coordinated_held_object_updates
            )
        committed_effects = StateDelta(
            held_object_updates={
                key: projected_task.held_objects.get(key) for key in held_keys
            },
            coordinated_held_object_updates={
                key: projected_task.coordinated_held_objects.get(key)
                for key in coordinated_keys
            },
        )
        next_state = ExecutionState.from_task_state(
            projected_task,
            last_qpos=torch.where(
                combined_success[:, None], terminal_qpos, state.last_qpos
            ),
        )
        return ActionOutcome(
            trajectory=selected_positions,
            success=combined_success,
            next_state=next_state,
            grounded=grounded,
            prior_state=state,
            expected_effects=committed_effects,
        )

    def _select_upright_transport_yaw(
        self,
        grounded: GroundedAction,
        state: ExecutionState,
    ) -> GroundedAction:
        """Choose the closest IK-feasible yaw for an upright object target."""
        sample_count = int(grounded.cfg.get("upright_yaw_samples", 1))
        capability = self.capabilities.get(grounded.action_class)
        if (
            capability.target_materializer != "semantic_held_object"
            or sample_count <= 1
        ):
            return grounded
        target_pose = getattr(grounded.target, "object_target_pose", None)
        if not isinstance(target_pose, torch.Tensor):
            return grounded
        target_pose = target_pose.to(device=self.device, dtype=torch.float32)
        if target_pose.shape == (4, 4):
            target_pose = target_pose.unsqueeze(0).repeat(self.num_envs, 1, 1)
        if target_pose.shape != (self.num_envs, 4, 4):
            raise ValueError(
                "Upright transport target must have shape (4, 4) or (N, 4, 4)."
            )

        arm_part, _, _ = self._parts(grounded.arm)
        held = state.get_held_object(arm_part)
        if held is None:
            return grounded
        object_to_eef = held.object_to_eef.to(
            device=self.device,
            dtype=target_pose.dtype,
        )
        if object_to_eef.shape == (4, 4):
            object_to_eef = object_to_eef.unsqueeze(0).repeat(self.num_envs, 1, 1)
        variants = self._upright_yaw_variants(target_pose, sample_count)
        eef_variants = torch.matmul(variants, object_to_eef[:, None])
        joint_ids = list(self.env.robot.get_joint_ids(name=arm_part))
        start_qpos = state.last_qpos[:, joint_ids]
        seeds = start_qpos[:, None].expand(-1, sample_count, -1)
        success, qpos = self.env.robot.compute_batch_ik(
            pose=eef_variants,
            name=arm_part,
            joint_seed=seeds,
        )
        success = torch.as_tensor(
            success,
            dtype=torch.bool,
            device=self.device,
        ).reshape(self.num_envs, sample_count)
        qpos = torch.as_tensor(qpos, dtype=torch.float32, device=self.device)
        success &= torch.isfinite(qpos).all(dim=-1)
        distance = torch.linalg.vector_norm(qpos - seeds, dim=-1)
        distance = torch.where(
            success,
            distance,
            torch.full_like(distance, torch.inf),
        )
        best = distance.argmin(dim=1)
        env_ids = torch.arange(self.num_envs, device=self.device)
        selected = variants[env_ids, best]
        selected = torch.where(
            success.any(dim=1)[:, None, None],
            selected,
            target_pose,
        )
        return replace(
            grounded,
            target=replace(grounded.target, object_target_pose=selected),
            target_object_pose=selected,
        )

    @staticmethod
    def _upright_yaw_variants(
        target_pose: torch.Tensor,
        sample_count: int,
    ) -> torch.Tensor:
        signed_steps = [0]
        for step in range(1, (sample_count + 1) // 2):
            signed_steps.extend((step, -step))
        if sample_count % 2 == 0:
            signed_steps.append(sample_count // 2)
        angles = target_pose.new_tensor(signed_steps) * (2.0 * math.pi / sample_count)
        yaw = target_pose.new_zeros((sample_count, 3, 3))
        yaw[:, 0, 0] = torch.cos(angles)
        yaw[:, 0, 1] = -torch.sin(angles)
        yaw[:, 1, 0] = torch.sin(angles)
        yaw[:, 1, 1] = torch.cos(angles)
        yaw[:, 2, 2] = 1.0
        variants = target_pose[:, None].repeat(1, sample_count, 1, 1)
        variants[:, :, :3, :3] = torch.matmul(yaw[None], target_pose[:, None, :3, :3])
        return variants

    def _planning_context(self, state: ExecutionState) -> PlanningContext:
        qpos = state.last_qpos.to(device=self.device, dtype=torch.float32)
        get_qvel = getattr(self.env.robot, "get_qvel", None)
        qvel = get_qvel() if callable(get_qvel) else None
        if not isinstance(qvel, torch.Tensor) or qvel.shape != qpos.shape:
            qvel = torch.zeros_like(qpos)
        else:
            qvel = qvel.to(device=self.device, dtype=qpos.dtype)
        return PlanningContext(
            robot=RobotObservation(timestamp=0.0, qpos=qpos, qvel=qvel),
            task=state.to_task_state(),
            scene=self._scene_snapshot(),
            env_ids=torch.arange(
                self.num_envs,
                dtype=torch.long,
                device=self.device,
            ),
        )

    def _scene_snapshot(self) -> SceneSnapshot:
        dynamic_uids = tuple(
            str(uid) for uid in self.planner_policy.get("dynamic_obstacle_uids", ())
        )
        if not bool(self.planner_policy.get("dynamic_collision", False)):
            return SceneSnapshot.empty()
        entities: dict[str, EntityState] = {}
        for uid in dynamic_uids:
            entity = self.env.sim.get_rigid_object(uid)
            if entity is None:
                raise ValueError(f"Unknown cuRobo dynamic obstacle {uid!r}.")
            pose = torch.as_tensor(
                entity.get_local_pose(to_matrix=True),
                dtype=torch.float32,
                device=self.device,
            )
            entities[uid] = EntityState(pose=pose)
        self._scene_version += 1
        return SceneSnapshot(
            timestamp=0.0,
            version=self._scene_version,
            entities=entities,
            collision_world_revision=self._scene_version,
            collision_entity_ids=dynamic_uids,
        )

    def _invocation(
        self,
        grounded: GroundedAction,
        capability: AtomicCapability,
    ) -> ActionInvocation:
        if capability.resource_mode == "coordinated_object":
            strategy = str(self.planner_policy["coordinated_strategy"])
        elif grounded.control == "hand":
            strategy = "ik_interp"
        else:
            strategy = str(self.planner_policy["single_arm_strategy"])
        sample_count = max(2, int(grounded.cfg.get("sample_interval", 50)))
        control_dt = float(getattr(self.env, "step_dt", 1.0 / 60.0))
        dynamic_mode = (
            DynamicCollisionMode.AUTO
            if bool(self.planner_policy.get("dynamic_collision", False))
            and strategy == "motion_gen"
            else DynamicCollisionMode.OFF
        )
        return ActionInvocation(
            skill_id=str(capability.action_type.skill_id),
            goal=grounded.target,
            binding=self._binding(grounded, capability),
            motion_policy=MotionPolicy(
                planner=str(self.planner_policy["backend"]),
                strategy=strategy,
                sample_count=sample_count,
                control_dt=control_dt,
                velocity_limit=grounded.cfg.get("velocity_limit"),
                acceleration_limit=grounded.cfg.get("acceleration_limit"),
                dynamic_collision_mode=dynamic_mode,
            ),
            recovery_policy=RecoveryPolicy(),
            skill_options=self._build_config(grounded, capability),
        )

    def _binding(
        self,
        action: GroundedAction,
        capability: AtomicCapability,
    ) -> ActionBinding:
        if capability.config_materializer == "handover":
            transfer_side = str(action.cfg.get("transfer_arm", "left_arm"))
            receive_side = "right_arm" if transfer_side == "left_arm" else "left_arm"
            transfer_arm, transfer_hand, _ = self._parts(transfer_side)
            receive_arm, receive_hand, _ = self._parts(receive_side)
            if transfer_hand is None or receive_hand is None:
                raise ValueError("HandOver requires two configured end effectors.")
            return ActionBinding(
                manipulators={"source": transfer_arm, "destination": receive_arm},
                end_effectors={"source": transfer_hand, "destination": receive_hand},
            )
        if capability.config_materializer == "coordinated_pickment":
            left_arm, left_hand, _ = self._parts("left_arm")
            right_arm, right_hand, _ = self._parts("right_arm")
            if left_hand is None or right_hand is None:
                raise ValueError("Coordinated pickup requires two end effectors.")
            return ActionBinding(
                manipulators={"left": left_arm, "right": right_arm},
                end_effectors={"left": left_hand, "right": right_hand},
            )
        if capability.config_materializer == "coordinated_placement":
            placing_arm, placing_hand, _ = self._parts("left_arm")
            support_arm, support_hand, _ = self._parts("right_arm")
            if placing_hand is None or support_hand is None:
                raise ValueError("Coordinated placement requires two end effectors.")
            return ActionBinding(
                manipulators={"placing": placing_arm, "support": support_arm},
                end_effectors={"placing": placing_hand, "support": support_hand},
            )

        arm_part, hand_part, _ = self._parts(action.arm)
        control_part = hand_part if action.control == "hand" else arm_part
        if control_part is None:
            raise ValueError(f"{action.arm} has no configured {action.control} part.")
        end_effectors: dict[str, str] = {}
        if capability.action_type.end_effector_roles:
            if hand_part is None:
                raise ValueError(f"{capability.name} requires an end effector.")
            end_effectors["primary"] = hand_part
        return ActionBinding(
            manipulators={"primary": control_part},
            end_effectors=end_effectors,
        )

    def _build_config(
        self,
        action: GroundedAction,
        capability: AtomicCapability | type,
    ) -> Any:
        """Build the mainline immutable ``ActionOptions`` value.

        The method name is retained as a narrow compatibility hook for existing
        Action Engine tests and extensions; it no longer constructs legacy
        hardware-bound ``ActionCfg`` objects.
        """
        if isinstance(capability, type):
            registered = self.capabilities.require_executable(action.action_class)
            if registered.config_type is not capability:
                raise ValueError(
                    f"Options type {capability.__name__!r} does not match "
                    f"AtomicAction {action.action_class!r}."
                )
            capability = registered
        if capability.config_materializer_hook is not None:
            return capability.config_materializer_hook(
                adapter=self,
                action=action,
                capability=capability,
            )
        builder = getattr(
            self,
            f"_build_{capability.config_materializer}_config",
            self._build_single_arm_config,
        )
        return builder(action, capability)

    def _config_policy(self, action: GroundedAction) -> dict[str, Any]:
        policy = dict(action.cfg)
        for key in (
            "postcondition_tolerance",
            "relation_distance",
            "hover_height",
            "staging_lift_height",
            "transport_clearance",
            "surface_clearance",
            "receiver_hold_joint_tolerance",
            "post_hold_steps",
        ):
            policy.pop(key, None)
        return policy

    def _build_single_arm_config(
        self,
        action: GroundedAction,
        capability: AtomicCapability,
    ) -> Any:
        policy = self._config_policy(action)
        if (
            capability.target_materializer == "semantic_held_object"
            and int(action.cfg.get("upright_yaw_samples", 1)) > 1
        ):
            policy["allow_automatic_transport_rotation"] = False
        approach_mode = policy.pop("approach_direction_mode", None)
        if approach_mode == "handover_transfer":
            from .frames import robot_frame_axes

            _, lateral = robot_frame_axes(self.env)
            outward = lateral[0] if action.arm == "left_arm" else -lateral[0]
            policy["approach_direction"] = _diagonal_approach_direction(
                -outward.to(device=self.device)
            )
        elif approach_mode is not None:
            raise ValueError(f"Unknown approach_direction_mode {approach_mode!r}.")
        for name in ("approach_direction", "obj_upright_direction"):
            if name in policy and not isinstance(policy[name], torch.Tensor):
                policy[name] = torch.as_tensor(
                    policy[name], dtype=torch.float32, device=self.device
                )
        return capability.config_type(
            **_supported_kwargs(capability.config_type, policy)
        )

    def _build_coordinated_pickment_config(
        self,
        action: GroundedAction,
        capability: AtomicCapability,
    ) -> Any:
        return self._build_single_arm_config(action, capability)

    def _build_coordinated_placement_config(
        self,
        action: GroundedAction,
        capability: AtomicCapability,
    ) -> Any:
        return self._build_single_arm_config(action, capability)

    def _build_handover_config(
        self,
        action: GroundedAction,
        capability: AtomicCapability,
    ) -> Any:
        policy = self._config_policy(action)
        middle = action.cfg.get("middle_object_pose")
        final = action.cfg.get("final_object_pose")
        if middle is None or final is None:
            raise ValueError("HandOver grounding must provide middle and final poses.")
        transfer_side = str(action.cfg.get("transfer_arm", "left_arm"))
        from .frames import robot_frame_axes

        _, lateral = robot_frame_axes(self.env)
        transfer_outward = (
            lateral[0] if transfer_side == "left_arm" else -lateral[0]
        ).to(device=self.device)
        policy.update(
            {
                "middle_object_pose": middle,
                # Delivery is represented by a following MoveHeldObject node.
                # Keep the receiver fixed while the source retreats here.
                "final_object_pose": middle,
                "preserve_current_object_orientation": False,
                "receive_approach_direction": _diagonal_approach_direction(
                    transfer_outward
                ),
            }
        )
        return capability.config_type(
            **_supported_kwargs(capability.config_type, policy)
        )

    def _positions_with_agent_holds(
        self,
        plan: ActionPlan,
        grounded: GroundedAction,
        capability: AtomicCapability,
    ) -> torch.Tensor:
        positions = plan.trajectory.positions.to(
            device=self.device,
            dtype=torch.float32,
        )
        hold_steps = int(grounded.cfg.get("post_hold_steps", 0))
        if capability.state_effect != "release" or hold_steps <= 0:
            return positions
        release = next((item for item in plan.segments if item.name == "release"), None)
        if release is None or release.stop <= 0 or release.stop > positions.shape[1]:
            return positions
        hold = positions[:, release.stop - 1 : release.stop].repeat(1, hold_steps, 1)
        return torch.cat(
            (positions[:, : release.stop], hold, positions[:, release.stop :]),
            dim=1,
        )

    @staticmethod
    def _merge_plan_rows(
        primary: torch.Tensor,
        fallback: torch.Tensor,
        use_fallback: torch.Tensor,
        hold_qpos: torch.Tensor,
    ) -> torch.Tensor:
        steps = max(primary.shape[1], fallback.shape[1], 1)

        def padded(value: torch.Tensor) -> torch.Tensor:
            if value.shape[1] == 0:
                return hold_qpos[:, None].repeat(1, steps, 1)
            if value.shape[1] < steps:
                value = torch.cat(
                    (value, value[:, -1:].repeat(1, steps - value.shape[1], 1)),
                    dim=1,
                )
            return value

        primary = padded(primary)
        fallback = padded(fallback)
        return torch.where(use_fallback[:, None, None], fallback, primary)

    def _handover_receiver_hold_mask(
        self,
        trajectory: torch.Tensor,
        grounded: GroundedAction,
        options: Any,
        *,
        tolerance: float,
    ) -> torch.Tensor:
        if tolerance < 0.0:
            raise ValueError("receiver_hold_joint_tolerance must be non-negative.")
        retreat_steps = max(2, int(options.retreat_steps))
        if trajectory.shape[1] < retreat_steps:
            return torch.zeros(
                self.num_envs, dtype=torch.bool, device=trajectory.device
            )
        transfer_side = str(grounded.cfg.get("transfer_arm", "left_arm"))
        receive_side = "right_arm" if transfer_side == "left_arm" else "left_arm"
        receive_arm, _, _ = self._parts(receive_side)
        receiver_ids = self.env.robot.get_joint_ids(name=receive_arm)
        receiver = trajectory[:, -retreat_steps:, receiver_ids]
        drift = torch.amax(torch.abs(receiver - receiver[:, :1]), dim=(1, 2))
        return torch.isfinite(drift) & (drift <= tolerance)

    def execute_trajectory(
        self,
        trajectory: torch.Tensor,
        *,
        active: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Advance the environment while holding inactive vectorized rows."""
        if trajectory.ndim != 3 or trajectory.shape[0] != self.num_envs:
            raise ValueError("Execution trajectory must have shape (N, T, robot_dof).")
        active = active.to(device=trajectory.device, dtype=torch.bool)
        current = self.env.robot.get_qpos().to(
            device=trajectory.device,
            dtype=trajectory.dtype,
        )
        commands: list[torch.Tensor] = []
        for waypoint in trajectory.unbind(dim=1):
            command = torch.where(active[:, None], waypoint, current)
            self.env.step(command)
            update = getattr(self.env, "update_obj_info", None)
            if callable(update):
                update()
            commands.append(command.detach())
            current = command
        sync = getattr(self.env, "sync_agent_state_from_qpos", None)
        if callable(sync) and commands:
            sync(commands[-1])
        return commands

    def combine(
        self,
        outcomes: Mapping[str, ActionOutcome | None],
        masks: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Merge independently planned arm paths into one synchronized stream."""
        present = [item for item in outcomes.values() if item is not None]
        if not present:
            raise ValueError("At least one arm outcome is required.")
        steps = max(int(item.trajectory.shape[1]) for item in present)
        current = self.env.robot.get_qpos().to(self.device, dtype=torch.float32)
        merged = current[:, None, :].repeat(1, max(steps, 1), 1)
        success = torch.ones(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        for arm, outcome in outcomes.items():
            if outcome is None:
                continue
            mask = masks[arm].to(self.device, dtype=torch.bool)
            success &= ~mask | outcome.success
            trajectory = outcome.trajectory
            if trajectory.shape[1] == 0:
                continue
            if trajectory.shape[1] < steps:
                padding = trajectory[:, -1:].repeat(1, steps - trajectory.shape[1], 1)
                trajectory = torch.cat((trajectory, padding), dim=1)
            joint_ids = self.joint_ids(arm, include_hand=True)
            if not joint_ids:
                continue
            selected = merged[:, :, joint_ids]
            merged[:, :, joint_ids] = torch.where(
                mask[:, None, None], trajectory[:, :, joint_ids], selected
            )
        return merged, success

    def joint_ids(self, arm: str, *, include_hand: bool) -> list[int]:
        if arm == "coordinated":
            return list(range(int(self.env.robot.dof)))
        side = "left" if arm == "left_arm" else "right"
        result = list(getattr(self.env, f"{side}_arm_joints", ()))
        if include_hand:
            result.extend(getattr(self.env, f"{side}_eef_joints", ()))
        return result

    def _engine(self) -> AtomicActionEngine:
        if self._atomic_engine is None:
            self._atomic_engine = AtomicActionEngine(
                self._generator(),
                control_profiles=self._control_profiles(),
            )
        return self._atomic_engine

    def _generator(self) -> MotionGenerator:
        if self._motion_generator is None:
            backend = str(self.planner_policy.get("backend", "curobo"))
            if backend == "curobo":
                options = dict(self.planner_policy.get("curobo", {}))
                obstacle_uids = tuple(
                    dict.fromkeys(
                        [
                            *self.planner_policy.get("static_obstacle_uids", ()),
                            *self.planner_policy.get("dynamic_obstacle_uids", ()),
                        ]
                    )
                )
                rigid_objects = []
                for uid in obstacle_uids:
                    entity = self.env.sim.get_rigid_object(str(uid))
                    if entity is None:
                        raise ValueError(f"Unknown cuRobo obstacle {uid!r}.")
                    rigid_objects.append(entity)
                world = CuroboWorldCfg(
                    rigid_objects=rigid_objects or None,
                    obstacle_representation=str(
                        options.get("obstacle_representation", "cuboid")
                    ),
                    dynamic_obstacle_names=[
                        str(uid)
                        for uid in self.planner_policy.get("dynamic_obstacle_uids", ())
                    ],
                    multi_env=bool(options.get("multi_env", False)),
                )
                planner_cfg = CuroboPlannerCfg(
                    robot_uid=self.env.robot.uid,
                    log_level=str(options.get("log_level", "error")),
                    world=world,
                    use_cuda_graph=bool(options.get("use_cuda_graph", True)),
                    preserve_plan_samples=bool(
                        options.get("preserve_plan_samples", False)
                    ),
                    max_attempts=int(options.get("max_attempts", 5)),
                    collision_activation_distance=float(
                        options.get("collision_activation_distance", 0.01)
                    ),
                )
            elif backend == "toppra":
                planner_cfg = ToppraPlannerCfg(robot_uid=self.env.robot.uid)
            else:
                raise ValueError(
                    f"Unsupported Action Engine planner backend {backend!r}."
                )
            self._motion_generator = MotionGenerator(
                cfg=MotionGenCfg(planner_cfg=planner_cfg)
            )
        return self._motion_generator

    def _control_profiles(self) -> dict[str, ControlPartCommandProfile]:
        profiles: dict[str, ControlPartCommandProfile] = {}
        for side in ("left_arm", "right_arm"):
            try:
                _, hand_part, hand_dof = self._parts(side)
            except ValueError:
                continue
            if hand_part is None or hand_dof == 0 or hand_part in profiles:
                continue
            profiles[hand_part] = ControlPartCommandProfile.joint_positions(
                open=_as_hand_qpos(self.env.open_state, hand_dof, self.device),
                grasp=_as_hand_qpos(self.env.close_state, hand_dof, self.device),
            )
        return profiles

    def _parts(self, arm: str) -> tuple[str, str | None, int]:
        if arm not in {"left_arm", "right_arm"}:
            raise ValueError(f"Expected a physical arm, got {arm!r}.")
        is_left = arm == "left_arm"
        if hasattr(self.env, "get_agent_arm_control_part"):
            arm_part = self.env.get_agent_arm_control_part(is_left)
            hand_part = self.env.get_agent_eef_control_part(is_left)
        else:
            arm_part = arm
            hand_part = "left_eef" if is_left else "right_eef"
        hand_ids = (
            []
            if hand_part is None
            else list(self.env.robot.get_joint_ids(name=hand_part))
        )
        return (
            str(arm_part),
            None if hand_part is None else str(hand_part),
            len(hand_ids),
        )
