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

"""Adapt typed execution requests to the shared atomic-action primitives."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import torch

from embodichain.lab.sim.atomic_actions import (
    AntipodalAffordance,
    CoordinatedPickment,
    CoordinatedPickmentCfg,
    CoordinatedPlacement,
    CoordinatedPlacementCfg,
    MoveEndEffector,
    MoveEndEffectorCfg,
    MoveHeldObject,
    MoveHeldObjectCfg,
    MoveJoints,
    MoveJointsCfg,
    ObjectSemantics,
    PickUp,
    PickUpCfg,
    Place,
    PlaceCfg,
    Press,
    PressCfg,
    WorldState,
)
from embodichain.gen_sim.action_engine.config import default_runtime_policy
from embodichain.lab.sim.planners import MotionGenerator, MotionGenCfg, ToppraPlannerCfg
from embodichain.toolkits.graspkit.pg_grasp import (
    AntipodalSamplerCfg,
    GraspGeneratorCfg,
    GripperCollisionCfg,
)
from embodichain.utils.logger import log_info

from .grasp_collision_cache import ensure_vhacd_grasp_collision_cache
from .models import ActionOutcome, GroundedAction, success_mask

__all__ = ["AtomicActionAdapter"]

_ACTION_TYPES: dict[str, tuple[type, type]] = {
    "PickUp": (PickUp, PickUpCfg),
    "MoveEndEffector": (MoveEndEffector, MoveEndEffectorCfg),
    "MoveJoints": (MoveJoints, MoveJointsCfg),
    "MoveHeldObject": (MoveHeldObject, MoveHeldObjectCfg),
    "Place": (Place, PlaceCfg),
    "Press": (Press, PressCfg),
    "CoordinatedPickment": (CoordinatedPickment, CoordinatedPickmentCfg),
    "CoordinatedPlacement": (CoordinatedPlacement, CoordinatedPlacementCfg),
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


class AtomicActionAdapter:
    """Own primitive construction, grasp semantics, and trajectory execution."""

    def __init__(
        self,
        env: Any,
        *,
        grasp_policy: Mapping[str, Any] | None = None,
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
        self._motion_generator: MotionGenerator | None = None
        self._semantics: dict[str, ObjectSemantics] = {}

    def initial_state(self) -> WorldState:
        return WorldState(last_qpos=self.env.robot.get_qpos().clone())

    def semantics(self, uid: str) -> ObjectSemantics:
        """Build object semantics once; poses remain live through the entity."""
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
        # The shared checker otherwise creates the same cache key with CoACD.
        # Publish a labelled V-HACD payload before the lazy affordance builds it.
        cache_result = ensure_vhacd_grasp_collision_cache(
            mesh_vertices=vertices,
            mesh_triangles=triangles,
            max_decomposition_hulls=max_hulls,
        )
        if cache_result.status != "hit":
            log_info(
                f"Prepared V-HACD grasp cache for {uid!r}: " f"{cache_result.status}."
            )
        semantics = ObjectSemantics(
            label=uid,
            entity=entity,
            geometry={
                "mesh_vertices": vertices,
                "mesh_triangles": triangles,
            },
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
        state: WorldState | None = None,
    ) -> ActionOutcome:
        """Plan one primitive without advancing the simulator."""
        if grounded.action_class not in _ACTION_TYPES:
            raise ValueError(f"Unsupported atomic action {grounded.action_class!r}.")
        state = state or self.initial_state()
        action_type, config_type = _ACTION_TYPES[grounded.action_class]
        config = self._build_config(grounded, config_type)
        primitive = action_type(self._generator(), config)
        result = primitive.execute(grounded.target, state)
        trajectory = torch.as_tensor(
            result.trajectory,
            dtype=torch.float32,
            device=self.device,
        )
        if trajectory.ndim != 3 or trajectory.shape[0] != self.num_envs:
            raise ValueError(
                f"{grounded.action_class} returned invalid trajectory shape "
                f"{tuple(trajectory.shape)}."
            )
        success = success_mask(result.success, self.num_envs, self.device)
        return ActionOutcome(
            trajectory=trajectory,
            success=success,
            next_state=result.next_state,
            grounded=grounded,
        )

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
                mask[:, None, None],
                trajectory[:, :, joint_ids],
                selected,
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

    def _generator(self) -> MotionGenerator:
        if self._motion_generator is None:
            self._motion_generator = MotionGenerator(
                cfg=MotionGenCfg(
                    planner_cfg=ToppraPlannerCfg(robot_uid=self.env.robot.uid)
                )
            )
        return self._motion_generator

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

    def _dual_part(self, left: str, right: str) -> str:
        parts = getattr(self.env.robot, "control_parts", {}) or {}
        expected = list(self.env.robot.get_joint_ids(name=left)) + list(
            self.env.robot.get_joint_ids(name=right)
        )
        for name in ("dual_arm", "both_arms"):
            if name in parts:
                return name
        for name in parts:
            if list(self.env.robot.get_joint_ids(name=name)) == expected:
                return str(name)
        if isinstance(parts, dict):
            parts["dual_arm"] = list(parts[left]) + list(parts[right])
            cache = getattr(self.env.robot, "_joint_ids", None)
            if isinstance(cache, dict):
                cache["dual_arm"] = expected
            return "dual_arm"
        raise ValueError("Coordinated actions require a dual-arm control part.")

    def _build_config(self, action: GroundedAction, config_type: type) -> Any:
        policy = dict(action.cfg)
        policy.pop("postcondition_tolerance", None)
        policy.pop("relation_distance", None)
        policy.pop("hover_height", None)
        policy.pop("staging_lift_height", None)
        policy.pop("transport_clearance", None)
        policy.pop("surface_clearance", None)
        if action.action_class in {"CoordinatedPickment", "CoordinatedPlacement"}:
            left_arm, left_hand, left_dof = self._parts("left_arm")
            right_arm, right_hand, right_dof = self._parts("right_arm")
            if left_hand is None or right_hand is None:
                raise ValueError("Coordinated actions require two configured hands.")
            common = {
                "control_part": self._dual_part(left_arm, right_arm),
            }
            if action.action_class == "CoordinatedPickment":
                common.update(
                    {
                        "left_arm_control_part": left_arm,
                        "right_arm_control_part": right_arm,
                        "left_hand_control_part": left_hand,
                        "right_hand_control_part": right_hand,
                        "left_hand_open_qpos": _as_hand_qpos(
                            self.env.open_state, left_dof, self.device
                        ),
                        "left_hand_close_qpos": _as_hand_qpos(
                            self.env.close_state, left_dof, self.device
                        ),
                        "right_hand_open_qpos": _as_hand_qpos(
                            self.env.open_state, right_dof, self.device
                        ),
                        "right_hand_close_qpos": _as_hand_qpos(
                            self.env.close_state, right_dof, self.device
                        ),
                    }
                )
            else:
                # The v1 coordinated-place contract assigns the logical left
                # arm to the placing object and the logical right arm to the
                # support object. Resolve those slots through the environment
                # instead of relying on the primitive's left_hand/right_hand
                # defaults, which do not match all generated robot profiles.
                common.update(
                    {
                        "placing_arm_control_part": left_arm,
                        "support_arm_control_part": right_arm,
                        "placing_hand_control_part": left_hand,
                        "support_hand_control_part": right_hand,
                        "placing_hand_open_qpos": _as_hand_qpos(
                            self.env.open_state,
                            left_dof,
                            self.device,
                        ),
                        "placing_hand_close_qpos": _as_hand_qpos(
                            self.env.close_state,
                            left_dof,
                            self.device,
                        ),
                        "support_hand_close_qpos": _as_hand_qpos(
                            self.env.close_state,
                            right_dof,
                            self.device,
                        ),
                    }
                )
            return config_type(
                **common,
                **_supported_kwargs(config_type, policy),
            )

        arm_part, hand_part, hand_dof = self._parts(action.arm)
        values: dict[str, Any] = {"control_part": arm_part}
        if action.control == "hand":
            if hand_part is None:
                raise ValueError(f"{action.arm} has no configured hand.")
            values["control_part"] = hand_part
        if action.action_class in {"PickUp", "MoveHeldObject", "Place", "Press"}:
            if hand_part is None:
                raise ValueError(f"{action.action_class} requires a hand control part.")
            values["hand_control_part"] = hand_part
        if action.action_class in {"PickUp", "Place"}:
            values["hand_open_qpos"] = _as_hand_qpos(
                self.env.open_state, hand_dof, self.device
            )
        if action.action_class in {"PickUp", "MoveHeldObject", "Place", "Press"}:
            key = (
                "hand_close_qpos"
                if action.action_class != "Press"
                else "hand_close_qpos"
            )
            values[key] = _as_hand_qpos(self.env.close_state, hand_dof, self.device)
        values.update(_supported_kwargs(config_type, policy))
        return config_type(**values)
