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

"""Measured skill objectives shared by standalone and Task Program consumers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
import math
import hashlib
import json
from typing import Any, TYPE_CHECKING

import torch

from embodichain.compute.geometry.skill_measurements import manipulation_contacts
from embodichain.lab.sim.atomic_skills.reward_profiles import (
    default_skill_reward_profile,
    skill_reward_profiles,
)
from embodichain.utils.math import matrix_from_quat
from .cfg import ObservationCfg
from .manager_base import Functor
from .observation_types import (
    ObservationDescriptor,
    ObservationSample,
    ObservationFrame,
)

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
    from ..objectives.conditions import ConditionResult

__all__ = ["SkillObjectiveObservation", "validate_skill_objective"]


def validate_skill_objective(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and own an explicit measured goal without acquiring simulation.

    Args:
        value: Skill ID, binding, goal and optional acceptance declarations.
            Bindings use explicit runtime object and control-part identities.
            Hosts may derive omitted hand links from a calibrated grasp endpoint.

    Returns:
        Owned declaration with finite numeric parameters and closed fields.
    """
    if (
        not isinstance(value, Mapping)
        or set(value) - {"skill_id", "binding", "goal", "acceptance", "reward_profile"}
        or not {"skill_id", "binding", "goal"} <= set(value)
    ):
        raise ValueError(
            "Skill objective requires skill_id, binding, goal and optional acceptance"
        )
    result = deepcopy(dict(value))
    skill = result["skill_id"]
    default_skill_reward_profile(skill)
    selected_profile = result.get("reward_profile", "default")
    profiles = skill_reward_profiles(skill)
    if not isinstance(selected_profile, str) or selected_profile not in profiles:
        raise ValueError("Unknown measured skill reward profile")
    if not set(profiles[selected_profile].roles) <= set(
        default_skill_reward_profile(skill).roles
    ):
        raise ValueError(
            "A measured skill profile can only use its declared physical score roles"
        )
    allowed_binding = {"robot", "motion_part"}
    required_binding = {"robot", "motion_part"}
    if skill in ("pick_up", "place", "move_held_object"):
        allowed_binding |= {"object", "pad_links", "hand_links", "support"}
        required_binding |= {"object", "support"}
    binding = result["binding"]
    if (
        not isinstance(binding, Mapping)
        or set(binding) - allowed_binding
        or required_binding - set(binding)
    ):
        raise ValueError(f"{skill} binding requires {sorted(required_binding)}")
    for key, item in binding.items():
        if key in ("pad_links", "hand_links"):
            if (
                not isinstance(item, (list, tuple))
                or not item
                or len(set(item)) != len(item)
                or any(not isinstance(v, str) or not v for v in item)
            ):
                raise ValueError("Hand links must be explicit unique native link names")
        elif not isinstance(item, str) or not item:
            raise ValueError("Skill binding values must be explicit nonempty names")
    if "pad_links" in binding and (
        len(binding["pad_links"]) != 2
        or (
            "hand_links" in binding
            and not set(binding["pad_links"]) <= set(binding["hand_links"])
        )
    ):
        raise ValueError(
            "Both opposing pads must be included in the complete hand link selection"
        )
    goal = result["goal"]
    allowed_goal = (
        {"positions"}
        if skill == "move_joints"
        else {
            "position",
            "quaternion_xyzw",
            "reference",
            "frame",
            "preserve_orientation",
        }
    )
    required_goal = {"positions"} if skill == "move_joints" else {"position"}
    if (
        not isinstance(goal, Mapping)
        or set(goal) - allowed_goal
        or required_goal - set(goal)
    ):
        raise ValueError(f"Invalid {skill} goal declaration")
    for key in ("positions", "position", "quaternion_xyzw"):
        if key not in goal:
            continue
        raw = goal[key]
        if not isinstance(raw, (list, tuple)) or any(
            type(v) not in (float, int) or not math.isfinite(v) for v in raw
        ):
            raise ValueError("Skill targets must contain finite numbers")
        if not raw or (
            key != "positions" and len(raw) != (3 if key == "position" else 4)
        ):
            raise ValueError("Skill target has an invalid dimension")
    if "quaternion_xyzw" in goal and not math.isclose(
        sum(v * v for v in goal["quaternion_xyzw"]), 1.0, abs_tol=1e-5
    ):
        raise ValueError("Skill goal quaternion must be unit XYZW")
    if goal.get("frame", "parent") not in ("parent", "world_axes"):
        raise ValueError("Goal frame must be parent or world_axes")
    if "preserve_orientation" in goal and (
        type(goal["preserve_orientation"]) is not bool
        or "quaternion_xyzw" in goal
        or skill not in ("place", "move_held_object")
    ):
        raise ValueError(
            "Preserved orientation must be explicit and cannot also declare a quaternion"
        )
    if "reference" in goal and (
        not isinstance(goal["reference"], str) or not goal["reference"]
    ):
        raise ValueError("Relative target reference must be explicit")
    defaults = {
        "position_tolerance": 0.02,
        "orientation_tolerance": 0.1,
        "joint_tolerance": 0.02,
        "position_scale": 0.1,
        "orientation_scale": 0.5,
        "joint_scale": 0.1,
        "hold_duration": 0.1,
        "stable_duration": 0.2,
        "linear_speed": 0.02,
        "angular_speed": 0.1,
        "relative_speed": 0.05,
        "separation": 0.002,
        "lift_height": 0.05,
    }
    acceptance = result.get("acceptance", {})
    if not isinstance(acceptance, Mapping) or set(acceptance) - set(defaults):
        raise ValueError("Unknown skill acceptance parameters")
    defaults.update(acceptance)
    if any(
        type(v) not in (float, int) or not math.isfinite(v) or v <= 0
        for v in defaults.values()
    ):
        raise ValueError(
            "Skill acceptance scales and thresholds must be finite and positive"
        )
    result["acceptance"] = defaults
    return result


def _resolve_hand_binding(
    objective: dict[str, Any], robot_cfg: Any, *, grasp_part: str | None = None
) -> str | None:
    """Resolve physical links from the selected calibrated hand, without a Robot."""
    if objective["skill_id"] not in ("pick_up", "place", "move_held_object"):
        return None
    binding = objective["binding"]
    if binding["robot"] != robot_cfg.uid:
        raise ValueError("Measured objective must bind the selected robot")
    hands = [
        hand
        for hand in (getattr(robot_cfg, "gripper_bindings", {}) or {}).values()
        if hand["declaration"]["motion_part"] == binding["motion_part"]
        and (grasp_part is None or hand["declaration"]["control_part"] == grasp_part)
    ]
    if len(hands) != 1:
        raise ValueError(
            "Measured manipulation requires one calibrated selected grasp endpoint"
        )
    hand = hands[0]
    pads = [hand["link_names"][name] for name in hand["defaults"]["pad_links"]]
    links = list(hand["link_names"].values())
    if "pad_links" in binding and set(binding["pad_links"]) != set(pads):
        raise ValueError("Measured pads must match the selected grasp endpoint")
    if "hand_links" in binding and set(binding["hand_links"]) != set(links):
        raise ValueError(
            "Measured collision links must match the selected grasp endpoint"
        )
    binding["pad_links"] = pads
    binding["hand_links"] = links
    return hand["declaration"]["control_part"]


def _shape_corners(shapes: Sequence[Any], device: torch.device) -> torch.Tensor:
    points = []
    signs = torch.tensor(
        [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
        device=device,
    )
    for shape in shapes:
        if shape.vertices is not None:
            vertex = shape.vertices.to(device=device, dtype=torch.float32)
        elif shape.half_extents is not None:
            vertex = signs * shape.half_extents.to(device=device, dtype=torch.float32)
        elif shape.radius is not None:
            half = signs.new_tensor(
                [shape.radius, shape.radius, shape.radius + (shape.half_height or 0.0)]
            )
            vertex = signs * half
        else:
            raise ValueError("Physical collision shape cannot establish separation")
        pose = shape.local_pose.to(device=device, dtype=torch.float32)
        points.append(vertex @ pose[:3, :3].T + pose[:3, 3])
    if not points:
        raise ValueError(
            "Complete collision geometry is required to establish separation"
        )
    vertices = torch.cat(points)
    low, high = vertices.amin(0), vertices.amax(0)
    return torch.where(signs > 0, high, low)


def _world_bounds(corners: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    points = corners @ pose[:, :3, :3].transpose(-1, -2) + pose[:, None, :3, 3]
    return torch.stack((points.amin(1), points.amax(1)), dim=1)


@dataclass(frozen=True)
class _ObjectiveProposal:
    sample: ObservationSample
    held_time: torch.Tensor
    stable_time: torch.Tensor
    relative: torch.Tensor
    step: int
    index: torch.Tensor
    height: torch.Tensor
    baseline: torch.Tensor
    rotation: torch.Tensor
    support: tuple | None
    held_now: torch.Tensor
    stable_now: torch.Tensor
    motion: torch.Tensor | None
    pose: torch.Tensor | None
    reference_pose: torch.Tensor | None
    hand_poses: torch.Tensor | None
    target_reference: torch.Tensor | None


class _ObjectiveInterval:
    """Bounded physical-substep evidence; canonical observations commit windows."""

    def __init__(self, objective):
        from embodichain.lab.sim.sensors.support_observation import (
            _BatchedSupportWitness,
        )

        self.objective = objective
        self.witness = _BatchedSupportWitness(objective.num_envs, objective.device)
        self.valid = torch.zeros(
            objective.num_envs, dtype=torch.bool, device=objective.device
        )
        self.held_time = torch.zeros(objective.num_envs, device=objective.device)
        self.stable_time = torch.zeros_like(self.held_time)
        self.held_previous = torch.zeros_like(self.valid)
        self.stable_previous = torch.zeros_like(self.valid)
        self.count = 0
        self.elapsed = 0.0
        self.last_step = -1
        self.last = None
        self.entry = None

    def begin_control_step(self):
        owner = self.objective
        owner._require_scene_bindings()
        self.valid.fill_(True)
        progress = getattr(owner._env, "task_program_progress", None)
        index = (
            torch.zeros_like(owner._invocation_index)
            if progress is None
            else progress.ordered_progress()[0].clamp_max(
                progress.ordered_progress()[1] - 1
            )
        )
        restart = (index != owner._invocation_index) | (
            owner._last_step != owner._env.sim.physics_step_index
        )
        self.held_time.copy_(torch.where(restart, 0.0, owner._held_time))
        self.stable_time.copy_(torch.where(restart, 0.0, owner._stable_time))
        self.held_previous.copy_(owner._held_boundary & ~restart)
        self.stable_previous.copy_(owner._stable_boundary & ~restart)
        self.count = 0
        self.elapsed = 0.0
        self.last_step = owner._env.sim.physics_step_index
        self.last = None
        self.relative = owner._previous_relative.clone()
        self.witness.copy_from(owner.support_witness)
        pose = owner.object.get_local_pose(to_matrix=True)
        self.entry = (pose[:, 2, 3].clone(), pose[:, :3, :3].clone())
        if owner._freeze_reference:
            self.entry = (
                *self.entry,
                owner.reference.get_local_pose(to_matrix=True).clone(),
            )

    def before_sample(self):
        return self.objective._env.sim.physics_step_index

    def after_sample(self, dt, token):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Objective physics sampling duration must be positive")
        owner = self.objective
        if token != self.last_step + 1:
            self.valid.zero_()
        proposal = owner._observe(
            owner._env,
            refresh_contacts=False,
            measurement_dt=dt,
            previous_relative=self.relative,
            sampling=True,
            witness=self.witness,
            entry=self.entry,
        )
        self.valid &= proposal.sample.valid
        self.held_time.copy_(
            torch.where(
                proposal.held_now & proposal.sample.valid & self.held_previous,
                self.held_time + dt,
                0.0,
            )
        )
        self.stable_time.copy_(
            torch.where(
                proposal.stable_now & proposal.sample.valid & self.stable_previous,
                self.stable_time + dt,
                0.0,
            )
        )
        self.held_previous.copy_(proposal.held_now & proposal.sample.valid)
        self.stable_previous.copy_(proposal.stable_now & proposal.sample.valid)
        self.relative = proposal.relative
        self.witness.advance(*proposal.support)
        self.last_step = token
        self.count += 1
        self.elapsed += dt
        self.last = proposal

    def coverage(self):
        owner = self.objective
        complete = (
            self.count == owner._env.cfg.sim_steps_per_control
            and math.isclose(
                self.elapsed, owner._env.step_dt, rel_tol=0.0, abs_tol=1e-9
            )
            and self.last_step == owner._env.sim.physics_step_index
        )
        return self.valid & complete

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.valid[ids] = False
        self.held_time[ids] = 0
        self.stable_time[ids] = 0
        self.held_previous[ids] = False
        self.stable_previous[ids] = False
        self.entry = None
        self.witness.reset(env_ids)


class SkillObjectiveObservation(Functor):
    """Observe one skill goal; temporal state advances only at host boundaries.

    The returned raw vector contains a potential and measured success. Native
    contacts, geometry, and velocity thresholds remain independent of reward
    weights and of the selected planner, real-motion source or learned policy.
    """

    def __init__(self, cfg: ObservationCfg, env: EmbodiedEnv) -> None:
        super().__init__(cfg, env)
        self.objective = validate_skill_objective(cfg.params["objective"])
        self.stage = cfg.params.get("stage")
        self._freeze_reference = cfg.params.get("freeze_reference", False)
        if type(self._freeze_reference) is not bool:
            raise ValueError("freeze_reference must be a bool")
        self._target_reference = (
            torch.eye(4, device=env.device).expand(env.num_envs, 4, 4).clone()
            if self._freeze_reference
            else None
        )
        self.profile = skill_reward_profiles(self.objective["skill_id"])[
            self.objective.get("reward_profile", "default")
        ]
        self.robot = env.sim.get_robot(self.objective["binding"]["robot"])
        if self.robot is None:
            raise ValueError("Skill objective robot is not present")
        self.binding = self.objective["binding"]
        if self.objective["skill_id"] in ("pick_up", "place", "move_held_object"):
            robot_cfg = getattr(self.robot, "cfg", None)
            if robot_cfg is not None:
                _resolve_hand_binding(
                    self.objective, robot_cfg, grasp_part=cfg.params.get("grasp_part")
                )
            if not {"pad_links", "hand_links"} <= set(self.binding):
                raise ValueError("Measured manipulation requires calibrated hand links")
        self.part = self.binding["motion_part"]
        self.robot.get_joint_ids(self.part)
        self.a = self.objective["acceptance"]
        self._held_time = torch.zeros(env.num_envs, device=env.device)
        self._stable_time = torch.zeros_like(self._held_time)
        self._held_boundary = torch.zeros_like(self._held_time, dtype=torch.bool)
        self._stable_boundary = torch.zeros_like(self._held_boundary)
        self._initialized = torch.zeros_like(self._held_time, dtype=torch.bool)
        self._last_step = torch.full_like(self._held_time, -1, dtype=torch.long)
        self._previous_relative = torch.zeros(env.num_envs, 3, device=env.device)
        self._last_sample = None
        self._candidate = None
        self._initial_height = torch.zeros_like(self._held_time)
        self._initial_rotation = (
            torch.eye(3, device=env.device).expand(env.num_envs, 3, 3).clone()
        )
        self._invocation_index = torch.full_like(self._last_step, -1)
        self.sensor = None
        self._interval = None
        self.object = None
        self.reference = None
        reference = self.objective["goal"].get("reference")
        if reference:
            self.reference = env.sim.get_rigid_object(reference)
            if self.reference is None:
                raise ValueError("Skill target reference is not a rigid scene object")
        if self._freeze_reference and self.reference is None:
            raise ValueError(
                "Frozen program targets require a bound rigid-object reference"
            )
        if "object" in self.binding:
            self.object = env.sim.get_rigid_object(self.binding["object"])
            if self.object is None:
                raise ValueError("Skill target object is not present")
            from embodichain.lab.sim.sensors.contact.sensor import (
                ContactSensorCfg,
                ArticulationContactFilterCfg,
            )

            self.sensor = env.sim.add_sensor(
                ContactSensorCfg(
                    uid=f"skill_contact_{cfg.name.replace(chr(47), chr(95))}",
                    rigid_uid_list=[self.binding["object"], self.binding["support"]],
                    articulation_cfg_list=[
                        ArticulationContactFilterCfg(
                            articulation_uid=self.robot.uid,
                            link_name_list=list(self.binding["hand_links"]),
                        )
                    ],
                    filter_need_both_actor=True,
                    max_contacts_per_env=128,
                )
            )
            if not self.sensor.contact_capabilities.impulse:
                raise ValueError(
                    "Measured manipulation objectives require native impulse evidence"
                )
            env.sensors[self.sensor.uid] = self.sensor
            from embodichain.lab.sim.sensors.support_observation import (
                _BatchedSupportWitness,
            )

            self.support_witness = _BatchedSupportWitness(env.num_envs, env.device)
            self.support = env.sim.get_rigid_object(self.binding["support"])
            if self.support is None:
                raise ValueError("Physical support must bind a rigid scene object")
            self.object_ids = self.sensor.get_actor_ids(self.binding["object"])[:, 0]
            self.pad_ids = self.sensor.get_actor_ids(
                self.robot.uid, self.binding["pad_links"]
            )
            self.support_ids = self.sensor.get_actor_ids(self.binding["support"])[:, 0]
            self.hand_ids = self.sensor.get_actor_ids(
                self.robot.uid, self.binding["hand_links"]
            )
            self.object_corners = torch.stack(
                [
                    _shape_corners(
                        self.object.get_collision_shapes(i, allow_heterogeneous=True),
                        env.device,
                    )
                    for i in range(env.num_envs)
                ]
            )
            self.hand_corners = {
                link: torch.stack(
                    [
                        _shape_corners(
                            self.robot.get_link_collision_shapes(link, i), env.device
                        )
                        for i in range(env.num_envs)
                    ]
                )
                for link in self.binding["hand_links"]
            }
            self._interval = _ObjectiveInterval(self)
            self.sensor.add_sampling_observer("skill_objective", self._interval)
        if self.profile.skill_id == "move_joints":
            positions = self.robot.get_qpos(self.part)
            if len(self.objective["goal"]["positions"]) != positions.shape[1]:
                raise ValueError(
                    "Skill target must match the explicitly bound control-part joint order"
                )

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Clear temporal state for selected episode rows."""
        ids = slice(None) if env_ids is None else env_ids
        self._initialized[ids] = False
        self._held_time[ids] = 0
        self._stable_time[ids] = 0
        self._held_boundary[ids] = False
        self._stable_boundary[ids] = False
        self._last_step[ids] = -1
        self._invocation_index[ids] = -1
        self._last_sample = None
        self._candidate = None
        if self._target_reference is not None:
            self._target_reference[ids] = torch.eye(4, device=self.device)
        if self.sensor is not None:
            self.support_witness.reset(env_ids)
            self._interval.reset(env_ids)

    def initialize_observation(
        self, frame: ObservationFrame, env_ids: Sequence[int] | torch.Tensor
    ) -> None:
        """Start selected windows after the reset acquisition, without counting it."""
        self._initialized[env_ids] = True
        self._last_step[env_ids] = self._env.sim.physics_step_index
        if self._candidate is not None:
            proposal = self._candidate
            self._previous_relative[env_ids] = proposal.relative[env_ids]
            self._held_boundary[env_ids] = (proposal.held_now & proposal.sample.valid)[
                env_ids
            ]
            self._stable_boundary[env_ids] = (
                proposal.stable_now & proposal.sample.valid
            )[env_ids]
            if self.object is not None:
                self._initial_height[env_ids] = proposal.height[env_ids]
                self._initial_rotation[env_ids] = proposal.rotation[env_ids]
            self._invocation_index[env_ids] = proposal.index[env_ids]
            if proposal.target_reference is not None:
                self._target_reference[env_ids] = proposal.target_reference[env_ids]

    def advance_observation(self) -> None:
        """Commit only the canonical control-boundary proposal."""
        proposal = self._candidate
        if proposal is None:
            return
        self._held_time.copy_(
            torch.where(proposal.sample.valid, proposal.held_time, 0.0)
        )
        self._stable_time.copy_(
            torch.where(proposal.sample.valid, proposal.stable_time, 0.0)
        )
        self._held_boundary.copy_(proposal.held_now & proposal.sample.valid)
        self._stable_boundary.copy_(proposal.stable_now & proposal.sample.valid)
        self._previous_relative.copy_(proposal.relative)
        self._last_step.fill_(proposal.step)
        self._invocation_index.copy_(proposal.index)
        if proposal.target_reference is not None:
            self._target_reference.copy_(proposal.target_reference)
        if self.object is not None:
            if self._interval.last is not None:
                self.support_witness.copy_from(self._interval.witness)
            self.support_witness.advance(*proposal.support)
            self.support_witness.known &= proposal.sample.valid
            self._initial_height.copy_(proposal.baseline)
            self._initial_rotation.copy_(proposal.rotation)

    def _pose_goal(self, current, preserved_rotation=None, reference_pose=None):
        goal = self.objective["goal"]
        position = current.new_tensor(goal["position"]).expand(self.num_envs, 3)
        rotation = None
        if goal.get("preserve_orientation", False):
            rotation = preserved_rotation
        if "quaternion_xyzw" in goal:
            rotation = matrix_from_quat(
                current.new_tensor(goal["quaternion_xyzw"]).expand(self.num_envs, 4)
            )
        if self.reference is not None:
            reference = (
                self.reference.get_local_pose(to_matrix=True)
                if reference_pose is None
                else reference_pose
            )
            position = (
                (reference[:, :3, :3] @ position[..., None]).squeeze(-1)
                if goal.get("frame", "parent") == "parent"
                else position
            ) + reference[:, :3, 3]
            if (
                rotation is not None
                and not goal.get("preserve_orientation", False)
                and goal.get("frame", "parent") == "parent"
            ):
                rotation = reference[:, :3, :3] @ rotation
        distance = torch.linalg.vector_norm(current[:, :3, 3] - position, dim=-1)
        angle = torch.zeros_like(distance)
        if rotation is not None:
            cosine = (
                (current[:, :3, :3].transpose(-1, -2) @ rotation)
                .diagonal(dim1=-2, dim2=-1)
                .sum(-1)
                - 1
            ) / 2
            angle = torch.acos(cosine.clamp(-1, 1))
        pos_score = 1 - torch.tanh(distance / self.a["position_scale"])
        rot_score = 1 - torch.tanh(angle / self.a["orientation_scale"])
        return (
            pos_score,
            rot_score,
            (distance <= self.a["position_tolerance"])
            & (angle <= self.a["orientation_tolerance"]),
        )

    def _require_scene_bindings(self) -> None:
        """Reject topology changes before touching a removed native handle."""
        for uid, handle in (
            (self.binding.get("object"), self.object),
            (self.binding.get("support"), getattr(self, "support", None)),
            (self.objective["goal"].get("reference"), self.reference),
        ):
            if (
                uid is not None
                and handle is not None
                and self._env.sim.get_rigid_object(uid) is not handle
            ):
                raise RuntimeError(
                    "A measured skill object was replaced or removed; rebuild the environment for new geometry"
                )

    def _observe(
        self,
        env,
        *,
        refresh_contacts=True,
        measurement_dt=None,
        previous_relative=None,
        sampling=False,
        witness=None,
        entry=None,
    ):
        """Evaluate physical facts without changing committed observation history."""
        skill = self.profile.skill_id
        motion = None
        valid = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        if skill != "move_joints":
            motion = self.robot.compute_fk(
                self.robot.get_qpos(self.part), name=self.part, to_matrix=True
            )
            valid &= torch.isfinite(motion).all(dim=(1, 2))
        step = env.sim.physics_step_index
        fresh = self._initialized & (self._last_step != step)
        active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        progress = getattr(env, "task_program_progress", None)
        if self.stage is not None and progress is not None:
            completed, total = progress.ordered_progress()
            indices = (
                progress._indices(self.stage)
                if isinstance(self.stage, tuple)
                else progress.indices(self.stage)
            )
            active = torch.stack([completed == i for i in indices]).any(0)
            if total - 1 in indices:
                active |= completed == total
        invocation_index = (
            torch.zeros_like(self._invocation_index)
            if progress is None
            else progress.ordered_progress()[0].clamp_max(
                progress.ordered_progress()[1] - 1
            )
        )
        entering = active & (invocation_index != self._invocation_index)
        target_reference = None
        if self._freeze_reference:
            live_reference = self.reference.get_local_pose(to_matrix=True)
            entry_reference = (
                entry[2] if entry is not None and len(entry) > 2 else live_reference
            )
            observed_reference = torch.where(
                active[:, None, None], entry_reference, live_reference
            )
            target_reference = torch.where(
                (entering | ~self._initialized | ~active)[:, None, None],
                observed_reference,
                self._target_reference,
            )
            valid &= torch.isfinite(target_reference).all(dim=(1, 2))
        candidate_index = invocation_index.clone()
        candidate_height = self._initial_height
        candidate_baseline = self._initial_height
        candidate_rotation = self._initial_rotation
        candidate_support = None
        held_now = torch.zeros_like(active)
        stable_now = torch.zeros_like(active)
        pose = reference_pose = hand_poses = None
        dt = env.step_dt if measurement_dt is None else measurement_dt
        previous_relative = (
            self._previous_relative if previous_relative is None else previous_relative
        )
        relative = self._previous_relative
        held_time, stable_time = self._held_time, self._stable_time
        if skill == "move_joints":
            positions = self.robot.get_qpos(self.part)
            error = positions - positions.new_tensor(
                self.objective["goal"]["positions"]
            )
            scores = {
                "joints": 1
                - torch.tanh(
                    torch.linalg.vector_norm(error, dim=-1) / self.a["joint_scale"]
                )
            }
            success = error.abs().amax(-1) <= self.a["joint_tolerance"]
            valid &= torch.isfinite(positions).all(-1)
        elif skill == "move_end_effector":
            position, orientation, success = self._pose_goal(
                motion, reference_pose=target_reference
            )
            scores = {"position": position, "orientation": orientation}
        else:
            pose = self.object.get_local_pose(to_matrix=True)
            hand_poses = torch.stack(
                [
                    self.robot.get_link_pose(link, to_matrix=True)
                    for link in self.binding["hand_links"]
                ],
                dim=1,
            )
            valid &= torch.isfinite(hand_poses).all(dim=(1, 2, 3))
            candidate_height = pose[:, 2, 3].clone()
            entry_height, entry_rotation = (
                (pose[:, 2, 3], pose[:, :3, :3]) if entry is None else entry[:2]
            )
            candidate_baseline = torch.where(
                entering, entry_height, self._initial_height
            )
            candidate_rotation = torch.where(
                (entering | ~self._initialized | ~active)[:, None, None],
                torch.where(active[:, None, None], entry_rotation, pose[:, :3, :3]),
                self._initial_rotation,
            )
            relative = (
                motion[:, :3, :3].transpose(-1, -2)
                @ (pose[:, :3, 3] - motion[:, :3, 3])[..., None]
            ).squeeze(-1)
            relative_speed = (
                torch.linalg.vector_norm(relative - previous_relative, dim=-1) / dt
            )
            if refresh_contacts:
                self.sensor.update()
            data = self.sensor.get_data()
            grip, hand_contact, supported, contact_valid = manipulation_contacts(
                data["user_ids"],
                data["normal"],
                data["impulse"],
                data["is_valid"],
                data["impulse_valid"],
                self.object_ids,
                self.pad_ids,
                self.support_ids,
            )
            all_hand = (
                (data["user_ids"][..., 0, None] == self.hand_ids[:, None, :])
                | (data["user_ids"][..., 1, None] == self.hand_ids[:, None, :])
            ).any(-1)
            target_contact = (data["user_ids"] == self.object_ids[:, None, None]).any(
                -1
            )
            hand_contact |= (
                all_hand & target_contact & data["is_valid"] & (data["impulse"] > 0)
            ).any(-1)
            contact_valid &= ~self.sensor.dropped_contacts_mask
            reference_pose = self.support.get_local_pose(to_matrix=True)
            candidate_support = (
                supported.clone(),
                contact_valid.clone(),
                pose.clone(),
                reference_pose.clone(),
            )
            supported = (self.support_witness if witness is None else witness).evaluate(
                supported, contact_valid, pose, reference_pose
            )
            valid &= torch.isfinite(pose).all(dim=(1, 2)) & (
                ~self._initialized | contact_valid
            )
            held_now = (
                grip
                & (relative_speed <= self.a["relative_speed"])
                & contact_valid
                & active
            )
            coverage = torch.ones_like(valid) if sampling else self._interval.coverage()
            valid &= ~self._initialized | coverage
            held_time = torch.where(
                fresh,
                (
                    torch.zeros_like(self._held_time)
                    if sampling
                    else torch.where(coverage, self._interval.held_time, 0.0)
                ),
                self._held_time,
            )
            held = ((held_time >= self.a["hold_duration"] - 1e-7) & held_now).float()
            pos, rot, near = self._pose_goal(pose, candidate_rotation, target_reference)
            pose_score = (pos + rot) / 2
            if skill == "pick_up":
                reach_distance = torch.linalg.vector_norm(
                    motion[:, :3, 3] - pose[:, :3, 3], dim=-1
                )
                scores = {
                    "reach": 1 - torch.tanh(reach_distance / self.a["position_scale"]),
                    "held": held,
                    "lifted": torch.zeros_like(pos),
                }
                lifted = (
                    (pose[:, 2, 3] - candidate_baseline) / self.a["lift_height"]
                ).clamp(0, 1)
                scores["lifted"] = lifted
                success = near & held.bool() & (lifted >= 1 - 1e-6)
            elif skill == "move_held_object":
                scores = {"pose": pose_score, "held": held}
                success = near & held.bool()
            else:
                object_bounds = _world_bounds(self.object_corners, pose)
                separation = torch.ones_like(valid)
                for i, corners in enumerate(self.hand_corners.values()):
                    link_pose = hand_poses[:, i]
                    bounds = _world_bounds(corners, link_pose)
                    gap = torch.maximum(
                        bounds[:, 0] - object_bounds[:, 1],
                        object_bounds[:, 0] - bounds[:, 1],
                    ).clamp_min(0)
                    separation &= (
                        torch.linalg.vector_norm(gap, dim=-1) >= self.a["separation"]
                    )
                    valid &= torch.isfinite(link_pose).all(dim=(1, 2))
                released = separation & ~hand_contact & contact_valid
                linear = self.object.body_data.lin_vel
                angular = self.object.body_data.ang_vel
                valid &= torch.isfinite(linear).all(-1) & torch.isfinite(angular).all(
                    -1
                )
                stable_now = (
                    released
                    & supported
                    & near
                    & (
                        torch.linalg.vector_norm(linear, dim=-1)
                        <= self.a["linear_speed"]
                    )
                    & (
                        torch.linalg.vector_norm(angular, dim=-1)
                        <= self.a["angular_speed"]
                    )
                    & valid
                    & active
                )
                stable_time = torch.where(
                    fresh,
                    (
                        torch.zeros_like(self._stable_time)
                        if sampling
                        else torch.where(coverage, self._interval.stable_time, 0.0)
                    ),
                    self._stable_time,
                )
                stable = torch.where(
                    stable_now,
                    (stable_time / self.a["stable_duration"]).clamp(0, 1),
                    0.0,
                )
                scores = {
                    "pose": pose_score,
                    "released": released.float(),
                    "stable": stable,
                }
                success = stable_now & (stable_time >= self.a["stable_duration"] - 1e-7)
        potential, scores_valid = self.profile.potential(
            {role: scores[role] for role in self.profile.roles}
        )
        valid &= (
            scores_valid
            & torch.isfinite(potential)
            & (potential >= 0)
            & (potential <= 1)
        )
        success &= valid & self._initialized & active
        values = torch.stack((potential, success.float()), dim=-1)
        descriptor = ObservationDescriptor(
            self.cfg.name,
            "skill_objective",
            "arena",
            "potential_success",
            ("unitless",),
            (2,),
            values.dtype,
        )
        sample = ObservationSample(descriptor, values, valid, (None,) * self.num_envs)
        return _ObjectiveProposal(
            sample,
            held_time,
            stable_time,
            relative,
            step,
            candidate_index,
            candidate_height,
            candidate_baseline,
            candidate_rotation,
            candidate_support,
            held_now,
            stable_now,
            None if motion is None else motion.clone(),
            None if pose is None else pose.clone(),
            None if reference_pose is None else reference_pose.clone(),
            None if hand_poses is None else hand_poses.clone(),
            None if target_reference is None else target_reference.clone(),
        )

    def __call__(self, env: EmbodiedEnv, obs: Any, **kwargs: Any) -> ObservationSample:
        """Propose the canonical result using a complete physical sampling interval."""
        self._require_scene_bindings()
        if self.sensor is not None:
            self.sensor.update()
        interval = self._interval
        proposal = self._observe(
            env,
            refresh_contacts=False,
            witness=None if interval is None else interval.witness,
            entry=None if interval is None else interval.entry,
        )
        if interval is not None and interval.last is not None:
            matching = torch.ones_like(proposal.sample.valid)
            for name in ("motion", "pose", "reference_pose", "hand_poses"):
                matching &= (
                    torch.isclose(
                        getattr(proposal, name),
                        getattr(interval.last, name),
                        atol=1e-6,
                        rtol=0.0,
                    )
                    .flatten(start_dim=1)
                    .all(-1)
                )
            valid = proposal.sample.valid & (~self._initialized | matching)
            values = proposal.sample.values.clone()
            values[:, 1] *= valid
            proposal = replace(
                proposal, sample=replace(proposal.sample, values=values, valid=valid)
            )
        self._candidate = proposal
        self._last_sample = proposal.sample
        return self._last_sample


def _objective_functors(env):
    manager = getattr(env, "observation_manager", None)
    if manager is not None:
        for configs in manager._mode_class_functor_cfgs.values():
            for cfg in configs:
                if isinstance(cfg.func, SkillObjectiveObservation):
                    yield cfg.func


def _initialize_objectives(env, frame, ids):
    for term in _objective_functors(env):
        term.initialize_observation(frame, ids)


def _advance_objectives(env):
    for term in _objective_functors(env):
        term.advance_observation()


def skill_objective_condition(env: EmbodiedEnv, objective: str) -> ConditionResult:
    """Read a skill's current physical result from its canonical observation.

    Args:
        env: Environment with the bound objective observation functor.
        objective: Exact ObservationManager configuration name.

    Returns:
        Measured success, validity and potential without acquiring new evidence.
    """
    from ..objectives.conditions import ConditionResult

    term = env.observation_manager.get_functor_cfg(objective).func
    if not isinstance(term, SkillObjectiveObservation) or term._last_sample is None:
        raise ValueError("Skill condition requires its canonical measured observation")
    sample = term._last_sample
    return ConditionResult(
        sample.values[:, 1].bool(), sample.valid, {"potential": sample.values[:, 0]}
    )


def _success_spec(field, objective):
    from embodichain.lab.evaluation import (
        PredicateDefinition,
        PredicateMeasurement,
        PredicateRegistry,
        EvaluationSpec,
        PredicateSpec,
    )

    def measure(arguments, params):
        value = arguments["state"]
        if value.ndim != 2 or value.shape[1] != 2:
            raise ValueError("Skill objective state must contain potential and success")
        return PredicateMeasurement(
            value[:, 1] == 1, torch.isfinite(value).all(-1), {"potential": value[:, 0]}
        )

    registry = PredicateRegistry()
    contract = hashlib.sha256(
        json.dumps(objective, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    registry.register(
        "skill_objective",
        PredicateDefinition(1, ("state",), lambda params: dict(params), measure),
    )
    return registry.compile(
        EvaluationSpec(
            {
                "success": PredicateSpec(
                    "skill_objective", {"state": field}, {"contract": contract}
                )
            },
            ("success",),
        )
    )


def _program_objective_conditions(cfg):
    result = {}
    for name, term in vars(cfg.observations).items():
        if (
            isinstance(term, ObservationCfg)
            and term.func is SkillObjectiveObservation
            and term.params.get("stage") is not None
        ):
            result[term.params["stage"]] = _success_spec(
                term.name, term.params["objective"]
            )
    return result


def _bind_objective(cfg, name, objective, *, stage=None):
    from .cfg import ConditionCfg

    objective = validate_skill_objective(objective)
    alias = f"skill_objective_{name}"
    field = f"skill_objectives/{name}/state"
    if hasattr(cfg.observations, alias):
        raise ValueError(f"Duplicate objective observation {alias!r}")
    setattr(
        cfg.observations,
        alias,
        ObservationCfg(
            func=SkillObjectiveObservation,
            mode="add",
            name=field,
            params={"objective": objective, "stage": stage},
            consumers=("recording",),
        ),
    )
    if stage is None:
        if "skill_goal" in cfg.rl_task.success:
            raise ValueError(
                "Standalone skill conflicts with explicit success.skill_goal"
            )
        cfg.rl_task.success["skill_goal"] = ConditionCfg(
            func=skill_objective_condition, params={"objective": alias}
        )
    return {
        "skill_id": objective["skill_id"],
        "metrics": {"potential": {"field": field, "index": 0}},
        "profile": "measured_potential",
        **(
            {"source_profile": objective["reward_profile"]}
            if objective.get("reward_profile", "default") != "default"
            else {}
        ),
        **(
            ({"call_index": stage[1]} if isinstance(stage, tuple) else {"stage": stage})
            if stage is not None
            else {}
        ),
    }


def _validate_program_objective_bindings(cfg, registration, stages):
    from embodichain.lab.task_program.integrations.progress import _stage_selections
    from embodichain.lab.task_program.semantics.calls import (
        Place,
        MoveHeldObjectCall,
        SemanticPose,
        SemanticRelativePose,
    )
    from embodichain.lab.task_program.semantics.integration import (
        SemanticIntegrationManifest,
    )
    from embodichain.utils.math import quat_from_matrix

    manifest = SemanticIntegrationManifest(
        scene=registration.catalog.scene,
        robot_profile=registration.catalog.robot_profile,
        call_catalog=registration.catalog.call_catalog,
        runtime_preset=cfg.task_program.integration.runtime_preset,
    )

    def orientation(goal, expected):
        expected = torch.as_tensor(expected, dtype=torch.float64)
        if goal.get("preserve_orientation", False):
            raise ValueError(
                "A fixed semantic orientation cannot be replaced by preserved orientation"
            )
        if "quaternion_xyzw" in goal:
            actual = torch.tensor(goal["quaternion_xyzw"], dtype=torch.float64)
            if not torch.isclose(
                (actual * expected).sum().abs(),
                torch.tensor(1.0, dtype=torch.float64),
                atol=1e-6,
                rtol=0,
            ):
                raise ValueError(
                    "Measured goal quaternion contradicts the semantic orientation"
                )
        else:
            goal["quaternion_xyzw"] = expected.tolist()

    objects = {
        binding.entity_id: binding.simulation_uid
        for binding in registration.scene_binding.rigid_objects
    }
    selections = _stage_selections(stages)
    for term in vars(cfg.observations).values():
        if (
            not isinstance(term, ObservationCfg)
            or term.func is not SkillObjectiveObservation
            or term.params.get("stage") is None
        ):
            continue
        objective = term.params["objective"]
        for index in selections.get(term.params["stage"], ()):
            stage = stages[index]
            call = stage.calls[0].call
            binding = objective["binding"]
            if hasattr(call, "object") and binding.get("object") != objects.get(
                call.object.entity_id
            ):
                raise ValueError(
                    "Measured objective must bind the Task Program call's exact object"
                )
            resources = dict(
                registration.robot_profile_binding.defaults.get(
                    objective["skill_id"], {}
                )
            )
            resources.update(call.resources)
            primary = resources.get("primary")
            if primary is None:
                raise ValueError(
                    "Measured skill objectives require an explicit primary resource selection"
                )
            resource = next(
                r
                for r in registration.robot_profile_binding.resources
                if r.resource_id == primary
            )
            motion = next(e for e in resource.endpoints if e.endpoint_id == "motion")
            if binding["motion_part"] != motion.control_part:
                raise ValueError(
                    "Measured objective must bind its call's selected motion endpoint"
                )
            if objective["skill_id"] in ("pick_up", "place", "move_held_object"):
                grasp = next(
                    (e for e in resource.endpoints if e.endpoint_id == "grasp"), None
                )
                if grasp is None:
                    raise ValueError(
                        "Measured manipulation requires a selected grasp endpoint"
                    )
                term.params["grasp_part"] = _resolve_hand_binding(
                    objective, cfg.robot, grasp_part=grasp.control_part
                )
            goal = objective["goal"]
            if isinstance(call, (Place, MoveHeldObjectCall)):
                target = call.at
                preserve_orientation = False
                if isinstance(call, Place):
                    linked = manifest.link_call(call)
                    preset = registration.catalog.robot_profile.presets[
                        linked.preset_id
                    ]
                    options = preset.skill_option_templates.get("place")
                    preserve_orientation = getattr(
                        options, "preserve_current_object_orientation", False
                    )
                    if preserve_orientation:
                        if (
                            "quaternion_xyzw" in goal
                            or goal.get("preserve_orientation", True) is False
                        ):
                            raise ValueError(
                                "Preserved semantic orientation cannot be replaced by a fixed goal"
                            )
                        goal["preserve_orientation"] = True
                if isinstance(call, Place) and call.on is not None:
                    surface = next(
                        (
                            s
                            for s in registration.scene_binding.support_surfaces
                            if s.entity_id
                            == linked.affordances["destination"].entity_id
                        ),
                        None,
                    )
                    if surface is None:
                        raise ValueError(
                            "Measured Place(on) requires an explicit bound support surface"
                        )
                    if not preserve_orientation:
                        orientation(
                            goal,
                            quat_from_matrix(
                                torch.tensor(surface.object_target_pose).reshape(4, 4)[
                                    :3, :3
                                ]
                            ),
                        )
                    expected_ref = objects.get(surface.parent_id)
                    expected_position = [
                        surface.object_target_pose[i] for i in (3, 7, 11)
                    ]
                    if (
                        binding["support"] != expected_ref
                        or goal.get("reference") != expected_ref
                        or goal.get("frame", "parent") != surface.object_target_frame
                        or any(
                            abs(a - b) > 1e-6
                            for a, b in zip(
                                goal["position"], expected_position, strict=True
                            )
                        )
                    ):
                        raise ValueError(
                            "Measured placement goal must match the semantic support frame"
                        )
                elif isinstance(target, SemanticPose):
                    if not preserve_orientation:
                        orientation(goal, target.quaternion_xyzw)
                    if "reference" in goal or not torch.allclose(
                        target.position.new_tensor(goal["position"]),
                        target.position,
                        atol=1e-6,
                        rtol=0,
                    ):
                        raise ValueError(
                            "Measured goal must match the semantic world target"
                        )
                elif isinstance(target, SemanticRelativePose):
                    expected_reference = objects.get(target.entity_id)
                    if expected_reference is None:
                        raise ValueError(
                            "Program scene-relative goals require a bound rigid-object reference"
                        )
                    if not preserve_orientation:
                        orientation(goal, target.quaternion_xyzw)
                    if (
                        goal.get("reference") != expected_reference
                        or goal.get("frame", "parent") != target.frame
                        or any(
                            abs(a - b) > 1e-6
                            for a, b in zip(
                                goal["position"], target.position, strict=True
                            )
                        )
                    ):
                        raise ValueError(
                            "Measured goal must match its semantic relative target"
                        )
                    term.params["freeze_reference"] = True
                else:
                    raise ValueError(
                        "Measured skill reward requires an explicit supported semantic target"
                    )
