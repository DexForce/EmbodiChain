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

from collections.abc import Callable, Mapping
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
    AxisAlignGoal,
    ControlPartCommandProfile,
    CoordinatedPickGoal,
    DynamicCollisionMode,
    EndEffectorPoseGoal,
    EntityState,
    ExecutionSession,
    MotionPolicy,
    ObjectSemantics,
    PlanningContext,
    RecoveryPolicy,
    RobotObservation,
    RigidObjectSceneProvider,
    SceneProvider,
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
from embodichain.toolkits.graspkit import ParallelJawGripperModelCfg
from embodichain.toolkits.graspkit.pg_grasp import (
    AntipodalGraspPoseGenerator,
    AntipodalGraspPoseGeneratorCfg,
    GraspAnnotationCfg,
    ParallelJawGraspCollisionCfg,
)
from embodichain.utils.logger import log_info

from .body_grasp import AxisAlignBodyGraspAdapter
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

# Preserve cuRobo's fixed world shape while disabling intentional-contact objects.
_COLLISION_PARKING_Z_OFFSET = -100.0
_BODY_GRASP_CANDIDATE_LIMIT = 500
_BODY_GRASP_SEED = 17_392
_FREE_YAW_SAMPLE_COUNT = 8


def _collision_cache_for_world(
    representation: str, obstacle_count: int
) -> dict[str, int]:
    """Size cuRobo's fixed collision cache for the generated scene."""
    cache = {"cuboid": 8, "mesh": 2}
    if representation in cache:
        cache[representation] = max(cache[representation], obstacle_count)
    return cache


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
        scene_provider: SceneProvider | None = None,
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
        self._coordinated_engines: dict[bool, AtomicActionEngine] = {}
        self._semantics: dict[str, ObjectSemantics] = {}
        self._scene_time = 0.0
        if scene_provider is not None and not isinstance(scene_provider, SceneProvider):
            raise TypeError("scene_provider must implement SceneProvider.")
        self.scene_provider = scene_provider or self._build_scene_provider()

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

    def start_session(
        self,
        grounded: GroundedAction,
        state: ExecutionState | None = None,
    ) -> ExecutionSession:
        """Start one closed-loop AtomicAction session from live scene state.

        ProgramExecutor may continue using its compatibility scheduler for
        compound and per-arm merged trajectories. New callers can use this
        boundary to adopt feedback-driven execution without constructing
        private planning contexts.
        """
        capability = self.capabilities.require_executable(grounded.action_class)
        state = state or self.initial_state()
        grounded = self._select_transport_yaw(grounded, state)
        grounded = self._adapt_coordinated_pickment_grasps(
            grounded,
            capability,
        )[0]
        context = self._planning_context(state, grounded)
        engine = self._engine_for(grounded, capability)
        invocation = self._invocation(grounded, capability, engine=engine)
        return engine.start((invocation,), context)

    def _build_scene_provider(self) -> SceneProvider | None:
        """Create the shared live rigid-object provider when entities are available."""
        sim = getattr(self.env, "sim", None)
        if sim is None:
            return None
        dynamic_uids = tuple(
            str(uid) for uid in self.planner_policy.get("dynamic_obstacle_uids", ())
        )
        list_uids = getattr(sim, "get_rigid_object_uid_list", None)
        uids = tuple(str(uid) for uid in list_uids()) if callable(list_uids) else ()
        if not uids:
            uids = dynamic_uids
        get_rigid_object = getattr(sim, "get_rigid_object", None)
        if not callable(get_rigid_object):
            return None
        entities = {
            uid: entity for uid in uids if (entity := get_rigid_object(uid)) is not None
        }
        if not entities:
            return None
        collision_uids = (
            dynamic_uids
            if bool(self.planner_policy.get("dynamic_collision", False))
            else ()
        )
        return RigidObjectSceneProvider(
            entities,
            collision_entity_ids=collision_uids,
        )

    def semantics(self, uid: str) -> ObjectSemantics:
        """Build object semantics once while retaining the live entity handle."""
        cached = self._semantics.get(uid)
        if cached is not None:
            return cached
        entity = self.env.sim.get_rigid_object(uid)
        if entity is None:
            entity = getattr(
                self.env.sim,
                "get_articulation",
                lambda _uid: None,
            )(uid)
            if entity is None:
                raise ValueError(f"Unknown grasp target {uid!r}.")
            active_joint_ids = list(getattr(entity, "active_joint_ids", ()))
            if len(active_joint_ids) != 1:
                raise ValueError(
                    "Articulation semantics require exactly one active joint."
                )
            backend_entities = getattr(
                entity,
                "_entities",
                getattr(entity, "entities", ()),
            )
            if not backend_entities:
                raise ValueError("Articulation backend does not expose joint metadata.")
            joint_name = str(entity.joint_names[active_joint_ids[0]])
            joint_info = backend_entities[0].get_joint_info(joint_name)
            child_link = str(getattr(joint_info, "child_link_name", ""))
            vertices, triangles = entity.get_link_vert_face(child_link)
        else:
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
        max_hulls = int(grasp_options["max_decomposition_hulls"])
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
        grounded = self._select_transport_yaw(grounded, state)
        context = self._planning_context(state, grounded)
        coordinated_candidates = self._adapt_coordinated_pickment_grasps(
            grounded,
            capability,
        )
        grounded_candidates = tuple(
            candidate
            for coordinated in coordinated_candidates
            for candidate in self._adapt_axis_align_body_grasps(
                coordinated,
                context,
                capability,
            )
        )
        selected: (
            tuple[
                GroundedAction,
                ActionInvocation,
                ActionPlan,
                AtomicActionEngine,
            ]
            | None
        ) = None
        best_failure_count = self.num_envs + 1
        for candidate in grounded_candidates:
            candidate_engine = self._engine_for(candidate, capability)
            candidate_invocation = self._invocation(
                candidate,
                capability,
                engine=candidate_engine,
            )
            candidate_plan = candidate_engine.plan(candidate_invocation, context)
            failure_count = int((~candidate_plan.plan_success).sum().item())
            if selected is None or failure_count < best_failure_count:
                selected = (
                    candidate,
                    candidate_invocation,
                    candidate_plan,
                    candidate_engine,
                )
                best_failure_count = failure_count
            if failure_count == 0:
                break
        if selected is None:
            raise RuntimeError("Atomic action adaptation produced no plan candidate.")
        grounded, invocation, plan, selected_engine = selected
        selected_positions = self._positions_with_agent_holds(
            plan,
            grounded,
            capability,
        )
        primary_success = plan.plan_success.to(self.device)
        reachability_search = None
        if bool(grounded.motion_policy.get("retreat_reachability_search", False)):
            (
                grounded,
                selected_positions,
                primary_success,
                reachability_search,
            ) = self._search_reachable_retreat(
                grounded=grounded,
                capability=capability,
                state=state,
                context=context,
                invocation=invocation,
                initial_positions=selected_positions,
                initial_success=primary_success,
            )
            invocation = replace(invocation, goal=grounded.target)
        combined_success = primary_success.clone()
        fallback_plan: ActionPlan | None = None
        use_fallback = torch.zeros_like(combined_success)
        fallback_attempted = torch.zeros_like(combined_success)
        fallback_success = torch.zeros_like(combined_success)

        fallback_strategy = self.planner_policy.get("fallback_strategy")
        collision_safety = str(grounded.motion_policy.get("collision_safety", "auto"))
        fallback_allowed = bool(self.planner_policy.get("allow_fallback", True)) and (
            collision_safety != "required"
        )
        if (
            fallback_allowed
            and invocation.motion_policy.strategy == "motion_gen"
            and fallback_strategy in {"ik_interp"}
            and not bool(combined_success.all())
        ):
            fallback_attempted = ~primary_success
            fallback_policy = replace(
                invocation.motion_policy,
                strategy=str(fallback_strategy),
                dynamic_collision_mode=DynamicCollisionMode.OFF,
                plan_opts=None,
            )
            fallback_plan = selected_engine.plan(
                replace(invocation, motion_policy=fallback_policy),
                context,
            )
            fallback_positions = self._positions_with_agent_holds(
                fallback_plan,
                grounded,
                capability,
            )
            fallback_success = fallback_plan.plan_success.to(self.device)
            use_fallback = fallback_attempted & fallback_success
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
        primary_rows = combined_success & primary_success
        projected_task = plan.expected_effects.apply(
            context.task,
            primary_rows,
        )
        held_keys = set(plan.expected_effects.held_object_updates)
        if fallback_plan is not None:
            fallback_rows = combined_success & use_fallback
            projected_task = fallback_plan.expected_effects.apply(
                projected_task,
                fallback_rows,
            )
            held_keys.update(fallback_plan.expected_effects.held_object_updates)
        committed_effects = StateDelta(
            held_object_updates={
                key: projected_task.held_objects.get(key) for key in held_keys
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
            planner_trace={
                **self._planner_trace(
                    grounded=grounded,
                    invocation=invocation,
                    context=context,
                    state=state,
                    primary_success=primary_success,
                    fallback_allowed=fallback_allowed,
                    fallback_strategy=(
                        str(fallback_strategy)
                        if invocation.motion_policy.strategy == "motion_gen"
                        and fallback_strategy in {"ik_interp"}
                        else None
                    ),
                    fallback_attempted=fallback_attempted,
                    fallback_success=fallback_success,
                    fallback_used=use_fallback,
                    reachability_search=reachability_search,
                ),
                # Auditability takes precedence over compactness here: every
                # selected planner route retains its complete joint path.
                "planned_trajectory": selected_positions.detach().clone(),
                "primary_action_diagnostics": deepcopy(dict(plan.diagnostics.metadata)),
                "fallback_action_diagnostics": (
                    None
                    if fallback_plan is None
                    else deepcopy(dict(fallback_plan.diagnostics.metadata))
                ),
                "action_segments": {
                    segment.name: {
                        "start": int(segment.start),
                        "stop": int(segment.stop),
                    }
                    for segment in plan.segments
                },
            },
        )

    def _adapt_axis_align_body_grasps(
        self,
        grounded: GroundedAction,
        context: PlanningContext,
        capability: AtomicCapability,
    ) -> tuple[GroundedAction, ...]:
        if capability.target_materializer != "axis_align":
            return (grounded,)
        goal = grounded.target
        if not isinstance(goal, AxisAlignGoal) or goal.grasp_xpos is not None:
            return (grounded,)
        object_pose = grounded.object_pose
        if (
            not isinstance(object_pose, torch.Tensor)
            or object_pose.shape != (self.num_envs, 4, 4)
            or not torch.isfinite(object_pose).all()
        ):
            raise ValueError(
                "AxisAlign body grasp requires a finite grounded live object pose "
                f"with shape ({self.num_envs}, 4, 4)."
            )
        if goal.semantics.entity_id is not None:
            goal = replace(
                goal,
                semantics=replace(goal.semantics, entity_id=None),
            )
        _, hand_part, _ = self._parts(grounded.arm)
        if hand_part is None:
            raise ValueError("AxisAlign body grasp requires a configured hand part.")
        options = self._build_config(grounded, capability)
        selected_approach = options.approach_direction
        adaptation = AxisAlignBodyGraspAdapter().adapt(
            goal,
            object_pose=object_pose,
            grasp_generator=self._engine().grasp_pose_generators[hand_part],
            approach_direction=selected_approach,
            target_axis=options.target_axis,
            seed=_BODY_GRASP_SEED,
        )
        cfg = dict(grounded.cfg)
        cfg["approach_direction"] = selected_approach
        candidates: list[GroundedAction] = []
        for adaptation_index, candidate_goal in enumerate(adaptation.alternative_goals):
            rank = adaptation.alternative_rank_indices[adaptation_index]
            policy = dict(grounded.motion_policy)
            policy["body_grasp"] = {
                "long_axis_index": adaptation.axes.long_axis_index,
                "short_axis_index": adaptation.axes.short_axis_index,
                "elongation_ratio": adaptation.axes.elongation_ratio,
                "candidate_indices": (
                    adaptation.selection.ranked_candidate_indices[:, rank].tolist()
                ),
                "candidate_counts": (
                    adaptation.selection.body_candidate_counts.tolist()
                ),
                "candidate_rank": rank,
                "approach_direction": selected_approach.detach().cpu().tolist(),
            }
            candidates.append(
                replace(
                    grounded,
                    target=candidate_goal,
                    cfg=cfg,
                    motion_policy=policy,
                )
            )
        return tuple(candidates)

    def _adapt_coordinated_pickment_grasps(
        self,
        grounded: GroundedAction,
        capability: AtomicCapability,
    ) -> tuple[GroundedAction, ...]:
        """Build deterministic geometry-ranked E5 partition candidates.

        ``left_to_right_arm_direction`` remains the live base-to-base direction:
        it labels the two participant regions and is not the transport direction.
        Object geometry only adjusts how much of the projected middle is excluded.
        """
        if capability.config_materializer != "coordinated_pickment":
            return (grounded,)
        target = self._validate_coordinated_pickment_goal(grounded)
        live_pose = grounded.object_pose
        expected_shape = (self.num_envs, 4, 4)
        if (
            not isinstance(live_pose, torch.Tensor)
            or live_pose.shape != expected_shape
            or not bool(torch.isfinite(live_pose).all())
        ):
            raise ValueError(
                "CoordinatedPickment requires a finite grounded live object pose "
                f"with shape {expected_shape}."
            )
        live_pose = live_pose.to(device=self.device, dtype=torch.float32).clone()
        affordance = target.semantics.affordance
        assert isinstance(affordance, AntipodalAffordance)
        vertices = torch.as_tensor(
            affordance.mesh_vertices,
            dtype=torch.float32,
            device=self.device,
        )
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or vertices.shape[0] < 3
            or not bool(torch.isfinite(vertices).all())
        ):
            raise ValueError(
                "CoordinatedPickment mesh vertices must be finite with shape (N, 3)."
            )

        left_base, right_base = self._coordinated_arm_bases()
        arm_directions = right_base[:, :3, 3] - left_base[:, :3, 3]
        arm_norms = torch.linalg.vector_norm(arm_directions, dim=1, keepdim=True)
        if not bool(torch.isfinite(arm_directions).all()) or bool(
            (arm_norms <= 1.0e-6).any()
        ):
            raise ValueError(
                "Coordinated pickup requires distinct finite left/right arm bases."
            )
        arm_directions = arm_directions / arm_norms
        shared_direction = arm_directions[0]
        if bool(
            (torch.matmul(arm_directions, shared_direction).abs() < 1.0 - 1.0e-4).any()
        ):
            raise ValueError(
                "CoordinatedPickment requires one shared base-to-base direction "
                "across vectorized environments."
            )

        centered = vertices - vertices.mean(dim=0, keepdim=True)
        covariance = centered.transpose(0, 1) @ centered / float(vertices.shape[0])
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        principal_local = eigenvectors[:, -1]
        principal_world = torch.matmul(
            live_pose[:, :3, :3],
            principal_local,
        )
        principal_world = principal_world / torch.linalg.vector_norm(
            principal_world,
            dim=1,
            keepdim=True,
        ).clamp_min(1.0e-6)
        arm_alignment = torch.abs((principal_world * arm_directions).sum(dim=1))
        elongation_ratio = torch.sqrt(
            eigenvalues[-1].clamp_min(1.0e-12) / eigenvalues[-2].clamp_min(1.0e-12)
        )
        elongation_confidence = torch.clamp(
            (elongation_ratio - 1.0) / 1.5,
            min=0.0,
            max=1.0,
        )
        base_ratio = float(grounded.cfg.get("middle_empty_ratio", 0.4))
        if not math.isfinite(base_ratio) or not 0.0 <= base_ratio < 1.0:
            raise ValueError("middle_empty_ratio must be finite and in [0, 1).")
        geometric_ratio = 0.25 + 0.45 * float(arm_alignment.mean())
        confidence = float(elongation_confidence)
        preferred_ratio = (1.0 - confidence) * base_ratio + confidence * geometric_ratio
        raw_ratios = (
            preferred_ratio,
            base_ratio,
            preferred_ratio - 0.15,
            preferred_ratio + 0.15,
        )
        ratios: list[float] = []
        for raw_ratio in raw_ratios:
            ratio = min(0.90, max(0.05, float(raw_ratio)))
            if not any(abs(ratio - existing) <= 1.0e-6 for existing in ratios):
                ratios.append(ratio)

        approach = grounded.cfg.get("approach_direction", (0.0, 0.0, -1.0))
        approach = torch.as_tensor(
            approach,
            dtype=torch.float32,
            device=self.device,
        )
        if approach.shape != (3,) or not bool(torch.isfinite(approach).all()):
            raise ValueError("approach_direction must be a finite vector shaped (3,).")
        approach_norm = torch.linalg.vector_norm(approach)
        if float(approach_norm) <= 1.0e-6:
            raise ValueError("approach_direction must be non-zero.")
        approach = approach / approach_norm
        trace = {
            "strategy": "live_geometry_partition_search",
            "local_principal_axis": principal_local.detach().cpu().tolist(),
            "world_principal_axes": principal_world.detach().cpu().tolist(),
            "elongation_ratio": float(elongation_ratio),
            "elongation_confidence": confidence,
            "arm_axis_alignment": arm_alignment.detach().cpu().tolist(),
            "left_to_right_arm_direction": shared_direction.detach().cpu().tolist(),
            "approach_direction": approach.detach().cpu().tolist(),
            "candidate_middle_empty_ratios": list(ratios),
        }
        candidates: list[GroundedAction] = []
        for candidate_index, ratio in enumerate(ratios):
            cfg = {
                **grounded.cfg,
                "left_to_right_arm_direction": shared_direction.clone(),
                "approach_direction": approach.clone(),
                "middle_empty_ratio": ratio,
            }
            motion_policy = {
                **grounded.motion_policy,
                "coordinated_grasp": {
                    **trace,
                    "candidate_index": candidate_index,
                    "selected_middle_empty_ratio": ratio,
                },
            }
            candidates.append(
                replace(
                    grounded,
                    target=replace(target, object_initial_pose=live_pose.clone()),
                    cfg=cfg,
                    object_pose=live_pose.clone(),
                    motion_policy=motion_policy,
                )
            )
        return tuple(candidates)

    def _coordinated_arm_bases(self) -> tuple[torch.Tensor, torch.Tensor]:
        from .frames import arm_base_poses

        return arm_base_poses(self.env)

    def _search_reachable_retreat(
        self,
        *,
        grounded: GroundedAction,
        capability: AtomicCapability,
        state: ExecutionState,
        context: PlanningContext,
        invocation: ActionInvocation,
        initial_positions: torch.Tensor,
        initial_success: torch.Tensor,
    ) -> tuple[GroundedAction, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Select the highest row-local retreat accepted by the live planner."""
        candidates = self._retreat_search_targets(grounded)
        target = getattr(grounded.target, "xpos", None)
        if not isinstance(target, torch.Tensor) or len(candidates) <= 1:
            return (
                grounded,
                initial_positions,
                initial_success,
                {
                    "strategy": "bounded_motion_planner",
                    "attempts": [],
                    "selected_target_z": (
                        None
                        if not isinstance(target, torch.Tensor)
                        else target[:, 2, 3]
                    ),
                },
            )

        selected_target = candidates[0][1].clone()
        selected_positions = initial_positions
        success = initial_success.clone()
        attempts: list[dict[str, Any]] = [
            {
                "candidate": candidates[0][0],
                "target_z": candidates[0][1][:, 2, 3].detach().clone(),
                "success": initial_success.detach().clone(),
            }
        ]
        for label, candidate_target in candidates[1:]:
            unresolved = ~success
            if not bool(unresolved.any()):
                break
            row_target = torch.where(
                unresolved[:, None, None],
                candidate_target,
                selected_target,
            )
            candidate_grounded = replace(
                grounded,
                target=EndEffectorPoseGoal(xpos=row_target),
            )
            candidate_invocation = replace(
                invocation,
                goal=candidate_grounded.target,
            )
            candidate_plan = self._engine().plan(candidate_invocation, context)
            candidate_positions = self._positions_with_agent_holds(
                candidate_plan,
                candidate_grounded,
                capability,
            )
            candidate_success = candidate_plan.plan_success.to(self.device)
            selected_rows = unresolved & candidate_success
            selected_positions = self._merge_plan_rows(
                selected_positions,
                candidate_positions,
                selected_rows,
                state.last_qpos,
            )
            selected_target = torch.where(
                selected_rows[:, None, None],
                candidate_target,
                selected_target,
            )
            success |= candidate_success
            attempts.append(
                {
                    "candidate": label,
                    "target_z": candidate_target[:, 2, 3].detach().clone(),
                    "success": candidate_success.detach().clone(),
                }
            )

        metadata = {
            "retreat_selected_target_z": selected_target[:, 2, 3].detach().clone(),
            "retreat_reachability_found": success.detach().clone(),
        }
        selected_grounded = replace(
            grounded,
            target=EndEffectorPoseGoal(xpos=selected_target),
            cfg={**grounded.cfg, **metadata},
            motion_policy={**grounded.motion_policy, **metadata},
        )
        return (
            selected_grounded,
            selected_positions,
            success,
            {
                "strategy": "bounded_motion_planner",
                "attempts": attempts,
                "selected_target_z": selected_target[:, 2, 3].detach().clone(),
            },
        )

    def _retreat_search_targets(
        self,
        grounded: GroundedAction,
    ) -> list[tuple[str, torch.Tensor]]:
        """Build bounded height and baseward retreat candidates from live poses."""
        target = getattr(grounded.target, "xpos", None)
        reference = grounded.motion_policy.get("retreat_reference_pose")
        if not isinstance(target, torch.Tensor) or not isinstance(
            reference, torch.Tensor
        ):
            return []
        target = target.to(device=self.device, dtype=torch.float32)
        reference = reference.to(device=self.device, dtype=torch.float32)
        if target.shape == (4, 4):
            target = target.unsqueeze(0).repeat(self.num_envs, 1, 1)
        if reference.shape == (4, 4):
            reference = reference.unsqueeze(0).repeat(self.num_envs, 1, 1)
        expected = (self.num_envs, 4, 4)
        if target.shape != expected or reference.shape != expected:
            return []

        sample_count = int(grounded.cfg.get("retreat_search_samples", 6))
        if not 2 <= sample_count <= 16:
            raise ValueError("retreat_search_samples must be in [2, 16].")
        minimum_height = float(grounded.cfg.get("minimum_retreat_height", 0.05))
        if not math.isfinite(minimum_height) or minimum_height < 0.0:
            raise ValueError("minimum_retreat_height must be finite and non-negative.")
        desired_height = torch.clamp(
            target[:, 2, 3] - reference[:, 2, 3],
            min=0.0,
        )
        minimum = torch.minimum(
            desired_height,
            torch.full_like(desired_height, minimum_height),
        )
        fractions = torch.linspace(
            1.0,
            0.0,
            sample_count,
            dtype=target.dtype,
            device=target.device,
        )
        heights = (
            minimum[:, None] + (desired_height - minimum)[:, None] * fractions[None]
        )
        candidates: list[tuple[str, torch.Tensor]] = [("requested", target.clone())]
        for index in range(1, sample_count):
            candidate = target.clone()
            candidate[:, 2, 3] = reference[:, 2, 3] + heights[:, index]
            candidates.append((f"height_{index}", candidate))

        from .frames import arm_base_poses

        left_base, right_base = arm_base_poses(self.env)
        base = left_base if grounded.arm == "left_arm" else right_base
        direction = base[:, :2, 3] - reference[:, :2, 3]
        norm = torch.linalg.vector_norm(direction, dim=1, keepdim=True)
        direction = torch.where(
            norm > 1.0e-6,
            direction / torch.clamp(norm, min=1.0e-6),
            torch.zeros_like(direction),
        )
        distance = float(grounded.cfg.get("retreat_distance", 0.10))
        if not math.isfinite(distance) or distance < 0.0:
            raise ValueError("retreat_distance must be finite and non-negative.")
        for index in range(sample_count):
            candidate = target.clone()
            candidate[:, :2, 3] = reference[:, :2, 3] + direction * distance
            candidate[:, 2, 3] = reference[:, 2, 3] + heights[:, index]
            candidates.append((f"baseward_{index}", candidate))
        return candidates

    def _planner_trace(
        self,
        *,
        grounded: GroundedAction,
        invocation: ActionInvocation,
        context: PlanningContext,
        state: ExecutionState,
        primary_success: torch.Tensor,
        fallback_allowed: bool,
        fallback_strategy: str | None,
        fallback_attempted: torch.Tensor,
        fallback_success: torch.Tensor,
        fallback_used: torch.Tensor,
        reachability_search: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build compact per-row evidence for the planner route actually used."""
        exclusions = self._collision_exclusion_masks(grounded, state)
        obstacle_positions = {
            uid: context.scene.entities[uid].pose[:, :3, 3].detach().clone()
            for uid in context.scene.collision_entity_ids
        }
        revisions = torch.as_tensor(
            context.scene.collision_world_revisions(self.num_envs),
            dtype=torch.int64,
            device=self.device,
        )
        trace = {
            "action_class": grounded.action_class,
            "arm": grounded.arm,
            "planner": str(self.planner_policy["backend"]),
            "primary_strategy": invocation.motion_policy.strategy,
            "dynamic_collision_mode": invocation.motion_policy.dynamic_collision_mode.value,
            "primary_success": primary_success.detach().clone(),
            "fallback_allowed": fallback_allowed,
            "fallback_strategy": fallback_strategy,
            "fallback_attempted": fallback_attempted.detach().clone(),
            "fallback_success": fallback_success.detach().clone(),
            "fallback_used": fallback_used.detach().clone(),
            "search_budget": {
                "primary_max_attempts": int(
                    self.planner_policy.get("curobo", {}).get("max_attempts", 1)
                ),
                "fallback_enabled": bool(fallback_allowed),
            },
            "collision_world_revision": revisions,
            "collision_obstacle_positions": obstacle_positions,
            "collision_exclusions": {
                uid: mask.detach().clone() for uid, mask in exclusions.items()
            },
        }
        if reachability_search is not None:
            trace["reachability_search"] = deepcopy(dict(reachability_search))
        options = invocation.skill_options
        object_part = getattr(options, "pick_object_part", None)
        approach_direction = getattr(options, "approach_direction", None)
        if object_part is None:
            object_part = getattr(options, "receive_pick_object_part", None)
            approach_direction = getattr(
                options,
                "receive_approach_direction",
                approach_direction,
            )
        if object_part is not None:
            grasp_policy: dict[str, Any] = {"object_part": str(object_part)}
            if isinstance(approach_direction, torch.Tensor):
                direction = approach_direction.to(dtype=torch.float32)
                norm = torch.linalg.vector_norm(direction)
                if bool(torch.isfinite(norm)) and float(norm) > 0.0:
                    grasp_policy["approach_direction"] = (
                        (direction / norm).detach().cpu().tolist()
                    )
            trace["grasp_policy"] = grasp_policy
        body_grasp = grounded.motion_policy.get("body_grasp")
        if isinstance(body_grasp, Mapping):
            trace["body_grasp"] = deepcopy(dict(body_grasp))
        coordinated_grasp = grounded.motion_policy.get("coordinated_grasp")
        if isinstance(coordinated_grasp, Mapping):
            trace["coordinated_grasp"] = deepcopy(dict(coordinated_grasp))
        return trace

    def _select_transport_yaw(
        self,
        grounded: GroundedAction,
        state: ExecutionState,
    ) -> GroundedAction:
        """Choose the closest IK-feasible yaw when task semantics leave it free."""
        sample_count = _FREE_YAW_SAMPLE_COUNT if grounded.allow_yaw_search else 1
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
            raise ValueError("Transport target must have shape (4, 4) or (N, 4, 4).")

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
        variants = self._yaw_variants(target_pose, sample_count)
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
        yaw_offsets = torch.matmul(
            variants[:, :, :3, :3],
            target_pose[:, None, :3, :3].transpose(-1, -2),
        )
        yaw_distance = torch.atan2(
            yaw_offsets[:, :, 1, 0],
            yaw_offsets[:, :, 0, 0],
        ).abs()
        minimum_yaw = torch.where(
            success,
            yaw_distance,
            torch.full_like(yaw_distance, torch.inf),
        ).amin(dim=1)
        minimum_rotation = success & torch.isclose(
            yaw_distance,
            minimum_yaw[:, None],
            atol=1.0e-6,
            rtol=0.0,
        )
        best = torch.where(
            minimum_rotation,
            distance,
            torch.full_like(distance, torch.inf),
        ).argmin(dim=1)
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
    def _yaw_variants(
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

    def _planning_context(
        self,
        state: ExecutionState,
        grounded: GroundedAction,
    ) -> PlanningContext:
        qpos = state.last_qpos.to(device=self.device, dtype=torch.float32)
        get_qvel = getattr(self.env.robot, "get_qvel", None)
        qvel = get_qvel() if callable(get_qvel) else None
        if not isinstance(qvel, torch.Tensor) or qvel.shape != qpos.shape:
            qvel = torch.zeros_like(qpos)
        else:
            qvel = qvel.to(device=self.device, dtype=qpos.dtype)
        return PlanningContext(
            robot=RobotObservation(timestamp=self._scene_time, qpos=qpos, qvel=qvel),
            task=state.to_task_state(),
            scene=self._scene_snapshot(grounded, state),
            env_ids=torch.arange(
                self.num_envs,
                dtype=torch.long,
                device=self.device,
            ),
            control_dt=float(getattr(self.env, "step_dt", 1.0 / 60.0)),
        )

    def _scene_snapshot(
        self,
        grounded: GroundedAction,
        state: ExecutionState,
    ) -> SceneSnapshot:
        dynamic_uids = tuple(
            str(uid) for uid in self.planner_policy.get("dynamic_obstacle_uids", ())
        )
        env_ids = torch.arange(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        if self.scene_provider is None:
            base = SceneSnapshot(timestamp=self._scene_time, version=0)
        else:
            base = self.scene_provider.snapshot(
                timestamp=self._scene_time,
                env_ids=env_ids,
            )
        if not bool(self.planner_policy.get("dynamic_collision", False)):
            return base
        exclusion_masks = self._collision_exclusion_masks(grounded, state)
        entities = dict(base.entities)
        for uid in dynamic_uids:
            entity_state = entities.get(uid)
            if entity_state is None:
                raise ValueError(
                    f"SceneProvider omitted cuRobo dynamic obstacle {uid!r}."
                )
            pose = entity_state.pose.to(dtype=torch.float32, device=self.device)
            if pose.shape == (4, 4):
                pose = pose.unsqueeze(0).repeat(self.num_envs, 1, 1)
            if pose.shape != (self.num_envs, 4, 4):
                raise ValueError(
                    f"Dynamic obstacle {uid!r} pose must have shape (4, 4) or "
                    f"({self.num_envs}, 4, 4), got {tuple(pose.shape)}."
                )
            excluded = exclusion_masks.get(uid)
            if excluded is not None and bool(excluded.any()):
                pose = pose.clone()
                pose[excluded, 2, 3] += _COLLISION_PARKING_Z_OFFSET
            entities[uid] = EntityState(
                pose=pose,
                confidence=entity_state.confidence,
            )
        return SceneSnapshot(
            timestamp=base.timestamp,
            version=base.version,
            entities=entities,
            collision_world_revision=base.collision_world_revision,
            collision_entity_ids=dynamic_uids,
        )

    def _collision_exclusion_masks(
        self,
        grounded: GroundedAction,
        state: ExecutionState,
    ) -> dict[str, torch.Tensor]:
        """Return per-environment masks for obstacles intentionally in contact."""
        dynamic_uids = {
            str(uid) for uid in self.planner_policy.get("dynamic_obstacle_uids", ())
        }
        masks: dict[str, torch.Tensor] = {}

        def include(uid: str | None, env_mask: torch.Tensor | None = None) -> None:
            if uid is None or uid not in dynamic_uids:
                return
            mask = (
                torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
                if env_mask is None
                else torch.as_tensor(
                    env_mask,
                    dtype=torch.bool,
                    device=self.device,
                ).reshape(-1)
            )
            if mask.shape != (self.num_envs,):
                raise ValueError(
                    f"Collision exclusion mask for {uid!r} must have shape "
                    f"({self.num_envs},), got {tuple(mask.shape)}."
                )
            masks[uid] = masks.get(uid, torch.zeros_like(mask)) | mask

        if self.capabilities.get(grounded.action_class).allows_target_contact:
            target_uid = grounded.object_uid
            if target_uid is None:
                target_uid = getattr(
                    getattr(grounded.target, "semantics", None),
                    "label",
                    None,
                )
            include(target_uid)

        for held in state.held_objects.values():
            include(held.semantics.label, held.env_mask)
        collision_exclusion_uids = grounded.motion_policy.get(
            "collision_exclusion_uids", ()
        )
        if isinstance(collision_exclusion_uids, str):
            collision_exclusion_uids = (collision_exclusion_uids,)
        for uid in collision_exclusion_uids:
            include(str(uid))
        return masks

    def _invocation(
        self,
        grounded: GroundedAction,
        capability: AtomicCapability,
        *,
        engine: AtomicActionEngine | None = None,
    ) -> ActionInvocation:
        if capability.resource_mode == "coordinated_object":
            strategy = str(self.planner_policy["coordinated_strategy"])
        elif grounded.control == "hand":
            strategy = "ik_interp"
        else:
            strategy = str(self.planner_policy["single_arm_strategy"])
        sample_count = max(2, int(grounded.cfg.get("sample_interval", 50)))
        dynamic_collision = bool(self.planner_policy.get("dynamic_collision", False))
        collision_required = (
            grounded.motion_policy.get("collision_safety") == "required"
        )
        if dynamic_collision and strategy == "motion_gen":
            dynamic_mode = (
                DynamicCollisionMode.REQUIRED
                if collision_required
                else DynamicCollisionMode.AUTO
            )
        else:
            dynamic_mode = DynamicCollisionMode.OFF
        goal = grounded.target
        if capability.config_materializer == "coordinated_pickment":
            self._validate_coordinated_pickment_goal(grounded)
        return ActionInvocation(
            skill_id=str(capability.action_type.skill_id),
            goal=goal,
            binding=self._binding(grounded, capability, engine=engine),
            motion_policy=MotionPolicy(
                strategy=strategy,
                sample_count=sample_count,
                dynamic_collision_mode=dynamic_mode,
            ),
            recovery_policy=RecoveryPolicy(),
            skill_options=self._build_config(grounded, capability),
        )

    @staticmethod
    def _validate_coordinated_pickment_goal(
        grounded: GroundedAction,
    ) -> CoordinatedPickGoal:
        """Validate the coordinated goal against the engine-scoped generator."""
        target = grounded.target
        if not isinstance(target, CoordinatedPickGoal):
            raise TypeError("CoordinatedPickment requires a CoordinatedPickGoal.")
        requested = grounded.cfg.get("is_filter_ground_collision")
        if requested is not None and not isinstance(requested, bool):
            raise TypeError("is_filter_ground_collision must be a boolean.")
        if not isinstance(target.semantics.affordance, AntipodalAffordance):
            raise TypeError(
                "CoordinatedPickment requires an AntipodalAffordance for GenSim "
                "grasp filtering."
            )
        return target

    def _binding(
        self,
        action: GroundedAction,
        capability: AtomicCapability,
        *,
        engine: AtomicActionEngine | None = None,
    ) -> ActionBinding:
        engine = self._engine() if engine is None else engine
        contract = getattr(capability.action_type, "binding_contract", None)
        if contract is None:
            return ActionBinding(owner_id=engine.binding_owner_id)

        slot_parts: dict[str, tuple[str, str | None]] = {}
        if capability.config_materializer == "handover":
            transfer_side = str(action.cfg.get("transfer_arm", "left_arm"))
            receive_side = "right_arm" if transfer_side == "left_arm" else "left_arm"
            transfer_arm, transfer_hand, _ = self._parts(transfer_side)
            receive_arm, receive_hand, _ = self._parts(receive_side)
            if transfer_hand is None or receive_hand is None:
                raise ValueError("HandOver requires two configured end effectors.")
            slot_parts = {
                "source": (transfer_arm, transfer_hand),
                "destination": (receive_arm, receive_hand),
            }
        elif capability.config_materializer == "coordinated_pickment":
            left_arm, left_hand, _ = self._parts("left_arm")
            right_arm, right_hand, _ = self._parts("right_arm")
            if left_hand is None or right_hand is None:
                raise ValueError("Coordinated pickup requires two end effectors.")
            slot_parts = {
                "left": (left_arm, left_hand),
                "right": (right_arm, right_hand),
            }
        elif capability.config_materializer == "coordinated_placement":
            placing_arm, placing_hand, _ = self._parts("left_arm")
            support_arm, support_hand, _ = self._parts("right_arm")
            if placing_hand is None or support_hand is None:
                raise ValueError("Coordinated placement requires two end effectors.")
            slot_parts = {
                "placing": (placing_arm, placing_hand),
                "support": (support_arm, support_hand),
            }
        else:
            arm_part, hand_part, _ = self._parts(action.arm)
            motion_part = hand_part if action.control == "hand" else arm_part
            if motion_part is None:
                raise ValueError(
                    f"{action.arm} has no configured {action.control} part."
                )
            slot_parts = {"primary": (motion_part, hand_part)}

        endpoints: dict[str, dict[str, str]] = {}
        for slot in contract.slots:
            try:
                motion_part, hand_part = slot_parts[slot.slot_id]
            except KeyError as exc:
                raise ValueError(
                    f"No GenSim binding is available for slot {slot.slot_id!r}."
                ) from exc
            selected: dict[str, str] = {}
            for requirement in slot.endpoints:
                if requirement.endpoint_id == "motion":
                    selected["motion"] = motion_part
                elif requirement.endpoint_id == "grasp":
                    if hand_part is None:
                        raise ValueError(
                            f"{capability.name} requires a grasp endpoint for "
                            f"slot {slot.slot_id!r}."
                        )
                    selected["grasp"] = hand_part
                else:
                    raise ValueError(
                        f"Unsupported GenSim endpoint {slot.slot_id}."
                        f"{requirement.endpoint_id}."
                    )
            endpoints[slot.slot_id] = selected
        return engine.bind_control_parts(
            str(capability.action_type.skill_id),
            endpoints,
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
        config_type = capability.config_type
        if capability.target_materializer == "semantic_held_object":
            from .atomic_compat import ExactTargetMoveHeldObjectOptions

            config_type = ExactTargetMoveHeldObjectOptions
        if capability.target_materializer == "press":
            press_depth = policy.pop("press_depth", None)
            if press_depth is not None and "press_distance" not in policy:
                policy["press_distance"] = press_depth
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
        return config_type(**_supported_kwargs(config_type, policy))

    def _build_coordinated_pickment_config(
        self,
        action: GroundedAction,
        capability: AtomicCapability,
    ) -> Any:
        policy = self._config_policy(action)
        left_base, right_base = self._coordinated_arm_bases()
        direction = right_base[0, :3, 3] - left_base[0, :3, 3]
        norm = torch.linalg.vector_norm(direction)
        if not torch.isfinite(direction).all() or norm <= 1.0e-6:
            raise ValueError(
                "Coordinated pickup requires distinct finite left/right arm bases."
            )
        policy.setdefault("left_to_right_arm_direction", direction / norm)
        for name in ("approach_direction", "left_to_right_arm_direction"):
            if name in policy and not isinstance(policy[name], torch.Tensor):
                policy[name] = torch.as_tensor(
                    policy[name], dtype=torch.float32, device=self.device
                )
        return capability.config_type(
            **_supported_kwargs(capability.config_type, policy)
        )

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
        receive_side = "right_arm" if transfer_side == "left_arm" else "left_arm"
        from .frames import robot_frame_axes

        _, lateral = robot_frame_axes(self.env)
        receiver_outward = (
            lateral[0] if receive_side == "left_arm" else -lateral[0]
        ).to(device=self.device)
        receiver_inward_approach = -receiver_outward
        policy.update(
            {
                "middle_object_pose": middle,
                # Delivery is represented by a following MoveHeldObject node.
                # Keep the receiver fixed while the source retreats here.
                "final_object_pose": middle,
                "receive_approach_direction": _diagonal_approach_direction(
                    receiver_inward_approach
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
        trajectory = plan.joint_trajectory
        if trajectory is None:
            raise ValueError(
                f"AtomicAction {plan.skill_id!r} did not retain a joint trajectory."
            )
        positions = trajectory.positions.to(
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
        waypoint_observer: Callable[[int], None] | None = None,
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
        for waypoint_index, waypoint in enumerate(trajectory.unbind(dim=1)):
            command = torch.where(active[:, None], waypoint, current)
            self.env.step(command)
            self._scene_time += self._scene_step_duration()
            update = getattr(self.env, "update_obj_info", None)
            if callable(update):
                update()
            if waypoint_observer is not None:
                waypoint_observer(waypoint_index)
            commands.append(command.detach())
            current = command
        sync = getattr(self.env, "sync_agent_state_from_qpos", None)
        if callable(sync) and commands:
            sync(commands[-1])
        return commands

    def _scene_step_duration(self) -> float:
        """Return one positive logical waypoint duration for scene timestamps."""
        sim_config = getattr(getattr(self.env, "sim", None), "sim_config", None)
        candidates = (
            getattr(self.env, "physics_dt", None),
            getattr(sim_config, "physics_dt", None),
        )
        for value in candidates:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                duration = float(value)
                if math.isfinite(duration) and duration > 0.0:
                    return duration
        return 1.0

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
            self._atomic_engine = self._new_engine(
                self._generator(),
                filter_ground_collision=True,
            )
        return self._atomic_engine

    def _engine_for(
        self,
        grounded: GroundedAction,
        capability: AtomicCapability,
    ) -> AtomicActionEngine:
        if capability.config_materializer != "coordinated_pickment":
            return self._engine()
        filter_ground_collision = grounded.cfg.get(
            "is_filter_ground_collision",
            True,
        )
        if not isinstance(filter_ground_collision, bool):
            raise TypeError("is_filter_ground_collision must be a boolean.")
        if filter_ground_collision:
            return self._engine()
        cached = self._coordinated_engines.get(filter_ground_collision)
        if cached is None:
            cached = self._new_engine(
                MotionGenerator(cfg=self._motion_generator_cfg()),
                filter_ground_collision=filter_ground_collision,
            )
            self._coordinated_engines[filter_ground_collision] = cached
        return cached

    def _new_engine(
        self,
        motion_generator: MotionGenerator,
        *,
        filter_ground_collision: bool,
    ) -> AtomicActionEngine:
        from embodichain.gen_sim.action_engine.capabilities import HeldObjectHandOver

        from .atomic_compat import ExactTargetMoveHeldObject

        engine = AtomicActionEngine(
            motion_generator,
            control_profiles=self._control_profiles(),
            grasp_pose_generators=self._grasp_pose_generators(
                filter_ground_collision=filter_ground_collision,
            ),
        )
        engine.register(ExactTargetMoveHeldObject(), replace=True)
        engine.register(HeldObjectHandOver(), replace=True)
        return engine

    def _generator(self) -> MotionGenerator:
        if self._motion_generator is None:
            self._motion_generator = MotionGenerator(cfg=self._motion_generator_cfg())
        return self._motion_generator

    def _motion_generator_cfg(self) -> MotionGenCfg:
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
            rigid_objects: dict[str, Any] = {}
            for uid in obstacle_uids:
                obstacle_uid = str(uid)
                entity = self.env.sim.get_rigid_object(obstacle_uid)
                if entity is None:
                    raise ValueError(f"Unknown cuRobo obstacle {uid!r}.")
                rigid_objects[obstacle_uid] = entity
            obstacle_representation = str(
                options.get("obstacle_representation", "cuboid")
            )
            world = CuroboWorldCfg(
                rigid_objects=rigid_objects or None,
                obstacle_representation=obstacle_representation,
                collision_cache=_collision_cache_for_world(
                    obstacle_representation,
                    len(rigid_objects),
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
                preserve_plan_samples=bool(options.get("preserve_plan_samples", False)),
                max_attempts=int(options.get("max_attempts", 5)),
                collision_activation_distance=float(
                    options.get("collision_activation_distance", 0.01)
                ),
            )
        elif backend == "toppra":
            planner_cfg = ToppraPlannerCfg(robot_uid=self.env.robot.uid)
        else:
            raise ValueError(f"Unsupported Action Engine planner backend {backend!r}.")
        return MotionGenCfg(planner_cfg=planner_cfg)

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

    def _grasp_pose_generators(
        self,
        *,
        filter_ground_collision: bool = True,
    ) -> dict[str, AntipodalGraspPoseGenerator]:
        """Build one mainline grasp service for each runtime hand endpoint."""
        if not isinstance(filter_ground_collision, bool):
            raise TypeError("filter_ground_collision must be a boolean.")
        options = self.grasp_policy
        model = ParallelJawGripperModelCfg(
            model_id="gen_sim_parallel_jaw",
            min_opening_width=float(options["min_open_length"]),
            max_opening_width=float(options["max_open_length"]),
            finger_length=float(options["finger_length"]),
            finger_width=0.03,
            finger_thickness=0.01,
            palm_depth=0.08,
        )
        algorithm = AntipodalGraspPoseGeneratorCfg(
            sample_count=int(options["antipodal_n_sample"]),
            ray_deviation_angle=float(options["antipodal_max_angle"]),
            approach_deviation_angle=float(options["max_deviation_angle"]),
            approach_direction_samples=int(options["n_deviated_approach_directions"]),
            max_candidates=_BODY_GRASP_CANDIDATE_LIMIT,
        )
        collision = ParallelJawGraspCollisionCfg(
            point_sample_density=float(options["point_sample_dense"]),
            max_decomposition_hulls=int(options["max_decomposition_hulls"]),
            opening_margin=0.01,
            filter_ground_collision=filter_ground_collision,
        )
        annotation = GraspAnnotationCfg(
            selection_mode="whole_mesh",
            viser_port=int(options["viser_port"]),
            force_refresh=bool(options["force_grasp_reannotate"]),
        )
        shared_generator = AntipodalGraspPoseGenerator(
            model,
            algorithm_cfg=algorithm,
            collision_cfg=collision,
            annotation_cfg=annotation,
        )
        generators: dict[str, AntipodalGraspPoseGenerator] = {}
        for arm in ("left_arm", "right_arm"):
            try:
                _, hand_part, _ = self._parts(arm)
            except ValueError:
                continue
            if hand_part is None or hand_part in generators:
                continue
            generators[hand_part] = shared_generator
        return generators

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
