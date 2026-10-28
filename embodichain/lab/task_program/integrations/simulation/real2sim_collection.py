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

"""Configure one B=1 tabletop Real2Sim owner for the run-task lifecycle."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from embodichain.data_pipeline.real2sim import load_prototype
from embodichain.lab.gym.envs.managers.cfg import DatasetFunctorCfg
from embodichain.lab.gym.envs.managers.datasets import LeRobotRecorder
from embodichain.lab.gym.envs.managers.real2sim_collection import (
    Real2SimCaptureRequest,
    Real2SimRunTaskOwner,
)
from embodichain.lab.gym.envs.managers.real2sim_sink import (
    Real2SimLeRobotEpisodeSink,
    build_real2sim_execution_trace,
)
from embodichain.lab.sim.atomic_actions import JointPositionTarget
from embodichain.lab.sim.motion.expansion import (
    CandidateIdentity,
    ValidationCheck,
    ValidationResult,
)
from embodichain.lab.sim.motion.expansion.real2sim_collision import (
    BoxCollisionBody,
    PlaneCollisionBody,
    SampledRobotCollisionChecker,
)
from embodichain.lab.sim.scene.expansion import ScenePoseChange, SceneVariant
from embodichain.lab.task_program.integrations.expansion import (
    TaskProgramMappedRetimeFactory,
    TaskProgramSameGridExpansionFactory,
)
from embodichain.lab.task_program.integrations.simulation.real2sim_collision import (
    TabletopCollisionGuardFactory,
    TabletopPrototypeCollisionValidator,
    _support_errors,
)
from embodichain.lab.task_program.integrations.simulation.scene_expansion import (
    SimulationSceneExpansionHost,
)
from embodichain.utils import configclass
from embodichain.utils.utility import load_config

__all__ = [
    "Real2SimTabletopCollectionCfg",
    "load_real2sim_tabletop_profile",
    "install_real2sim_tabletop_owner",
]


@configclass
class Real2SimTabletopCollectionCfg:
    """Explicit scene, collision policy and deterministic episode recipes.

    Each layout supplies ``id``, ``object_xyyaw`` and ``target_xyyaw``; offsets
    are metres and radians. Recipes are selected in listed order for every
    layout. Acceleration budgets must declare their provenance; a synthetic
    assumption does not establish real-device capability.
    """

    kind: str = "tabletop_box_v1"
    scene_revision: str = ""
    object_uid: str = ""
    target_uid: str = ""
    object_entity_id: str = ""
    target_entity_id: str = ""
    base_link: str = ""
    finger_links: tuple[str, ...] = ()
    finger_joints: tuple[str, ...] = ()
    closing_directions: tuple[int, ...] = ()
    contact_depth_m: float = 0.002
    base_ground_depth_m: float = 0.0035
    velocity_limits: tuple[float, ...] = ()
    velocity_limit_origin: str = ""
    acceleration_limits: tuple[float, ...] = ()
    acceleration_limit_origin: str = ""
    motion_headroom_fraction: float | None = None
    layouts: tuple[dict[str, object], ...] = ()
    recipes: tuple[str, ...] = ("nominal",)
    seed: int = 0
    placement_xy_tolerance_m: float = 0.025
    placement_height_tolerance_m: float = 0.01
    minimum_lift_m: float = 0.03
    maximum_drift_m: float = 0.003
    held_translation_tolerance_m: float = 0.015
    held_rotation_tolerance_deg: float = 8.0

    def __post_init__(self) -> None:
        if self.kind != "tabletop_box_v1":
            raise ValueError("unsupported Real2Sim collection profile")
        for name in (
            "scene_revision",
            "object_uid",
            "target_uid",
            "object_entity_id",
            "target_entity_id",
            "base_link",
            "velocity_limit_origin",
            "acceleration_limit_origin",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value.strip()
                or value != value.strip()
            ):
                raise ValueError(f"{name} must be a nonempty stripped string")
        if self.object_uid == self.target_uid:
            raise ValueError("object and target must be different bodies")
        if self.object_entity_id == self.target_entity_id:
            raise ValueError("object and target semantic entities must differ")
        if "ground" in {self.object_entity_id, self.target_entity_id}:
            raise ValueError("ground is reserved for the fixed support plane")
        self.finger_links = tuple(self.finger_links)
        self.finger_joints = tuple(self.finger_joints)
        self.closing_directions = tuple(self.closing_directions)
        if (
            not self.finger_links
            or len(self.finger_links) != len(self.finger_joints)
            or len(self.finger_links) != len(self.closing_directions)
        ):
            raise ValueError("finger links, joints and directions must align")
        if len(set(self.finger_links)) != len(self.finger_links) or len(
            set(self.finger_joints)
        ) != len(self.finger_joints):
            raise ValueError("finger link and joint declarations must be unique")
        if any(
            not isinstance(item, str) or not item
            for item in (*self.finger_links, *self.finger_joints)
        ) or any(
            type(item) is not int or item not in (-1, 1)
            for item in self.closing_directions
        ):
            raise ValueError("finger declarations are invalid")
        for name in ("velocity_limits", "acceleration_limits"):
            values = getattr(self, name)
            if any(type(value) not in (int, float) for value in values):
                raise ValueError(f"{name} must be numeric and cannot be booleans")
            converted = tuple(float(value) for value in values)
            if not converted or any(
                not math.isfinite(value) or value <= 0 for value in converted
            ):
                raise ValueError(f"{name} must be positive finite values")
            setattr(self, name, converted)
        for name in (
            "contact_depth_m",
            "base_ground_depth_m",
            "placement_xy_tolerance_m",
            "placement_height_tolerance_m",
            "minimum_lift_m",
            "maximum_drift_m",
            "held_translation_tolerance_m",
            "held_rotation_tolerance_deg",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.motion_headroom_fraction is not None and (
            isinstance(self.motion_headroom_fraction, bool)
            or not isinstance(self.motion_headroom_fraction, (int, float))
            or not math.isfinite(self.motion_headroom_fraction)
            or not 0 < self.motion_headroom_fraction < 1
        ):
            raise ValueError("motion_headroom_fraction must be between zero and one")
        allowed = {
            "nominal",
            "joint_7",
            "joint_11",
            "task_space",
            "time_115",
            "time_130",
        }
        self.recipes = tuple(self.recipes)
        if not self.recipes or any(type(recipe) is not str for recipe in self.recipes):
            raise ValueError("recipes must contain named profile values")
        if len(set(self.recipes)) != len(self.recipes) or set(self.recipes) - allowed:
            raise ValueError("recipes contain an unsupported or duplicate value")
        layouts = []
        for raw in self.layouts:
            if not isinstance(raw, dict) or set(raw) != {
                "id",
                "object_xyyaw",
                "target_xyyaw",
            }:
                raise ValueError(
                    "each layout requires id, object_xyyaw and target_xyyaw"
                )
            name = raw["id"]
            if not isinstance(name, str) or not name.strip():
                raise ValueError("layout id must be nonempty")
            item = {"id": name}
            for key in ("object_xyyaw", "target_xyyaw"):
                values = np.asarray(raw[key], dtype=np.float64)
                if values.shape != (3,) or not np.isfinite(values).all():
                    raise ValueError(f"layout {key} needs three finite values")
                item[key] = tuple(float(value) for value in values)
            layouts.append(item)
        if not layouts or len({item["id"] for item in layouts}) != len(layouts):
            raise ValueError("layouts must have unique IDs")
        self.layouts = tuple(layouts)


def load_real2sim_tabletop_profile(path: str | Path) -> Real2SimTabletopCollectionCfg:
    """Load and strictly validate an inert YAML/JSON collection profile."""
    payload = load_config(Path(path))
    if not isinstance(payload, dict):
        raise ValueError("Real2Sim profile must be a mapping")
    try:
        result = Real2SimTabletopCollectionCfg(**payload)
    except TypeError as error:
        raise ValueError("unknown Real2Sim profile field") from error
    result.validate()
    return result


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _offset(pose: torch.Tensor, delta: tuple[float, float, float]) -> torch.Tensor:
    result = pose.clone()
    x, y, yaw = delta
    result[0, 3] += x
    result[1, 3] += y
    cosine, sine = math.cos(yaw), math.sin(yaw)
    rotation = result.new_tensor([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
    result[:3, :3] = rotation @ result[:3, :3]
    return result


def _observed_program_phases(
    skills: Mapping[int, str],
    segments: Mapping[int, tuple[tuple[str, int, int], ...]],
    calls: np.ndarray,
    samples: np.ndarray,
) -> tuple[tuple[int, int, str, str], ...]:
    """Map each validated call phase to its first consumed action in O(T)."""
    if (
        not skills
        or set(skills) != set(segments)
        or calls.ndim != 1
        or samples.shape != calls.shape
        or calls.dtype.kind != "i"
        or samples.dtype.kind != "i"
    ):
        raise ValueError("observed trace and declared call occurrences do not match")
    starts: dict[int, tuple[int, ...]] = {}
    for index, phases in segments.items():
        if (
            not phases
            or phases[0][1] != 0
            or any(
                start >= stop or (pos and phases[pos - 1][2] != start)
                for pos, (_, start, stop) in enumerate(phases)
            )
        ):
            raise ValueError("validated phases must cover each plan contiguously")
        starts[index] = tuple(start for _, start, _ in phases)
    first: dict[tuple[int, int], int] = {}
    previous = -1
    for step, (call, sample) in enumerate(zip(calls.tolist(), samples.tolist())):
        if call == -1 and sample == -1:
            continue
        if call not in starts or call < previous or sample < -1:
            raise ValueError("observed trace has an unknown or out-of-order call")
        previous = call
        if sample == -1:
            continue
        phase = bisect_right(starts[call], sample) - 1
        if phase < 0 or sample >= segments[call][phase][2]:
            raise ValueError("observed trace refers outside a validated plan")
        first.setdefault((call, phase), step)
    events = []
    for index in sorted(skills):
        for phase, (name, _, _) in enumerate(segments[index]):
            if (index, phase) not in first:
                raise ValueError("observed trace is missing a declared phase")
            events.append((first[index, phase], index, skills[index], name))
    if any(events[pos][0] >= events[pos + 1][0] for pos in range(len(events) - 1)):
        raise ValueError("observed phases are not strictly sequential")
    return tuple(events)


def _held_pose_errors(objects: np.ndarray, tcp: np.ndarray) -> tuple[float, float]:
    """Measure maximum drift of the observed object-to-TCP relation."""
    if (
        objects.ndim != 2
        or objects.shape[1] != 7
        or tcp.shape != (len(objects), 4, 4)
        or not len(objects)
        or not np.isfinite(objects).all()
        or not np.isfinite(tcp).all()
    ):
        raise ValueError("held-state evidence must contain aligned finite poses")
    rotation = Rotation.from_quat(objects[:, 3:7]).as_matrix()
    relative_translation = np.einsum(
        "nij,nj->ni", rotation.transpose(0, 2, 1), tcp[:, :3, 3] - objects[:, :3]
    )
    relative_rotation = rotation.transpose(0, 2, 1) @ tcp[:, :3, :3]
    translation = float(
        np.linalg.norm(relative_translation - relative_translation[0], axis=1).max()
    )
    angle = float(
        np.degrees(
            Rotation.from_matrix(relative_rotation[0].T @ relative_rotation).magnitude()
        ).max()
    )
    return translation, angle


def install_real2sim_tabletop_owner(
    env: object, profile: Real2SimTabletopCollectionCfg, *, dataset_dir: str | Path
) -> Real2SimRunTaskOwner:
    """Install a configured source, collision and recorder owner on Gym.

    The source bindings come from the integration's configured runtime
    services, never from the Task Program. Physical geometry is obtained from
    the current scene and must match each prototype's captured identity.
    """
    target = getattr(env, "unwrapped", env)
    if not isinstance(profile, Real2SimTabletopCollectionCfg):
        raise TypeError("profile must be a Real2SimTabletopCollectionCfg")
    if (
        target.num_envs != 1
        or target.has_sensors
        or target.rollout_buffer is None
        or target._traj_buffer is None
    ):
        raise ValueError(
            "Real2Sim tabletop collection needs B=1, no sensors and both recording buffers"
        )
    if target.dataset_manager is not None:
        raise ValueError(
            "tabletop owner currently requires an exclusively owned LeRobot recorder"
        )
    adapter = target._task_program_adapter
    declarations = tuple(
        sorted(
            getattr(adapter, "_configured_nominal_sources", ()),
            key=lambda item: item.call_index,
        )
    )
    call_indices = tuple(item.call_index for item in declarations)
    if not call_indices or len(set(call_indices)) != len(call_indices):
        raise ValueError(
            "tabletop collection requires unique configured call occurrences"
        )
    registration = getattr(adapter, "_registration", None)
    if registration is None:
        raise ValueError("Real2Sim collection requires configured simulation bindings")
    scene_bindings = {
        item.entity_id: item.simulation_uid
        for item in registration.scene_binding.rigid_objects
    }
    if (
        scene_bindings.get(profile.object_entity_id) != profile.object_uid
        or scene_bindings.get(profile.target_entity_id) != profile.target_uid
    ):
        raise ValueError(
            "Real2Sim semantic-to-simulation mapping disagrees with integration"
        )
    if any(
        item.object_entity_id != profile.object_entity_id for item in declarations
    ) or any(
        item.target_entity_id != profile.target_entity_id
        for item in declarations
        if item.target_entity_id is not None
    ):
        raise ValueError("source and scene entity IDs must match this tabletop profile")
    if (
        set(target.sim.get_rigid_object_uid_list())
        != {profile.object_uid, profile.target_uid}
        or len(target.sim.get_robot_uid_list()) != 1
        or target.sim.get_articulation_uid_list()
        or target.sim.get_deformable_object_uid_list()
        or target.sim.get_rigid_object_group_uid_list()
    ):
        raise ValueError("tabletop profile cannot omit physical collision bodies")
    if len(profile.acceleration_limits) != len(target.robot.joint_names) or len(
        profile.velocity_limits
    ) != len(target.robot.joint_names):
        raise ValueError(
            "velocity and acceleration limits must cover the complete robot layout"
        )
    object_body = replace(
        BoxCollisionBody.from_rigid_object(
            target.sim.get_rigid_object(profile.object_uid)
        ),
        uid=profile.object_entity_id,
    )
    target_body = replace(
        BoxCollisionBody.from_rigid_object(
            target.sim.get_rigid_object(profile.target_uid)
        ),
        uid=profile.target_entity_id,
    )
    checker = SampledRobotCollisionChecker(
        target.robot.cfg.fpath,
        joint_names=target.robot.joint_names,
        bodies=(object_body, target_body, PlaneCollisionBody("ground", np.eye(4))),
        world_from_base=target.robot.get_local_pose(to_matrix=True)[0]
        .detach()
        .cpu()
        .numpy(),
    )
    prototypes = {
        item.call_index: load_prototype(item.prototype_manifest)
        for item in declarations
    }
    unique_prototypes = {}
    identities = {}
    for item in declarations:
        prototype = prototypes[item.call_index]
        source = prototype.unit.source
        key = (source.source_id, source.source_revision, source.unit_id)
        if key in identities and identities[key] != item.prototype_sha256:
            raise ValueError("a source unit revision cannot name different prototypes")
        identities[key] = item.prototype_sha256
        unique_prototypes[key] = prototype
    recorder = LeRobotRecorder(
        DatasetFunctorCfg(
            func=LeRobotRecorder,
            params={
                "save_path": str(Path(dataset_dir)),
                "use_videos": False,
                "robot_meta": {"robot_type": str(target.robot.uid)},
                "extra": {
                    "scene_type": "tabletop",
                    "task_description": "real2sim",
                    "data_type": "sim",
                },
            },
        ),
        target,
    )
    observations = {name: ("robot", name) for name in ("qpos", "qvel", "qf")}
    sink = Real2SimLeRobotEpisodeSink(
        recorder,
        observation_fields=observations,
        action_representation="joint_position",
        required_check_ids=(
            "geometry",
            "motion_limits",
            "phase_mapping",
            "plan_collision",
            "observed_collision",
            "task",
        ),
    )
    state: dict[str, Any] = {}

    def prepare(environment, episode_index, attempt_id):
        if episode_index >= len(profile.layouts) * len(profile.recipes):
            raise ValueError(
                "collection target exceeds the declared layout/recipe schedule"
            )
        layout_index, recipe_index = divmod(episode_index, len(profile.recipes))
        layout, recipe = profile.layouts[layout_index], profile.recipes[recipe_index]
        program = environment.compile_task_program(environment.cfg.task_program)
        compiled_calls = tuple(
            call.call_index
            for segment in program.iter_segments()
            for call in segment.calls
        )
        if call_indices != compiled_calls:
            raise ValueError(
                "tabletop collection requires one configured source for every "
                "sequential program call"
            )
        cube = environment.sim.get_rigid_object(profile.object_uid)
        platform = environment.sim.get_rigid_object(profile.target_uid)
        current = {
            profile.object_uid: cube.get_local_pose(to_matrix=True)[0].clone(),
            profile.target_uid: platform.get_local_pose(to_matrix=True)[0].clone(),
        }
        expected = {
            profile.object_uid: _offset(
                current[profile.object_uid], layout["object_xyyaw"]
            ),
            profile.target_uid: _offset(
                current[profile.target_uid], layout["target_xyyaw"]
            ),
        }
        changes = tuple(
            ScenePoseChange(uid, value.cpu().tolist())
            for uid, value in expected.items()
            if not torch.equal(value, current[uid])
        )
        variant = SceneVariant(
            profile.scene_revision, profile.seed, layout_index, changes
        )

        def initial():
            errors = []
            for uid, pose in expected.items():
                observed = environment.sim.get_rigid_object(uid).get_local_pose(
                    to_matrix=True
                )[0]
                errors.append(float(torch.max(torch.abs(observed - pose))))
            maximum = max(errors)
            return ValidationResult(
                (
                    ValidationCheck(
                        "initial.layout",
                        "passed" if maximum <= 0.003 else "failed",
                        metrics={"maximum_matrix_error": maximum},
                    ),
                )
            )

        preparation = SimulationSceneExpansionHost(
            environment, initial_validator=initial
        )(variant)
        if not preparation.validation.accepted:
            raise ValueError("scene layout failed initial validation")
        validator = TabletopPrototypeCollisionValidator(
            robot=environment.robot,
            checker=checker,
            prototypes=tuple(unique_prototypes.values()),
            object_uid=profile.object_entity_id,
            target_uid=profile.target_entity_id,
            ground_uid="ground",
            finger_links=profile.finger_links,
            finger_joints=profile.finger_joints,
            closing_directions=profile.closing_directions,
            base_link=profile.base_link,
            contact_depth_m=profile.contact_depth_m,
            base_ground_depth_m=profile.base_ground_depth_m,
        )
        delegate = None
        common = {
            "velocity_limits": profile.velocity_limits,
            "acceleration_limits": profile.acceleration_limits,
            "call_indices": call_indices,
            "diagnostic_unchecked": True,
        }
        if recipe.startswith("time_"):
            delegate = TaskProgramMappedRetimeFactory(
                duration_scale=1.15 if recipe == "time_115" else 1.30, **common
            )
        elif recipe != "nominal":

            def kinematics_factory(resolved, context, template):
                endpoint = resolved.binding.endpoint(
                    "primary", "motion"
                ).require_target(JointPositionTarget)
                if tuple(template.controlled_joint_indices) != tuple(
                    endpoint.joint_ids
                ):
                    raise ValueError(
                        "task-space profile requires the complete active arm"
                    )
                solver = environment.robot.get_solver(endpoint.control_part)
                base = environment.robot.get_link_pose(
                    link_name=solver.root_link_name, env_ids=[0], to_matrix=True
                )[0]
                tcp = torch.as_tensor(
                    solver.tcp_xpos, dtype=base.dtype, device=base.device
                )

                def kinematics(q):
                    q_device = q.to(base.device)
                    pose = base[None] @ solver.get_fk(q_device)
                    jac = solver.get_jacobian(q_device, locations=tcp[:3, 3])
                    jac = torch.cat(
                        (base[:3, :3] @ jac[:, :3], base[:3, :3] @ jac[:, 3:]), dim=1
                    )
                    return pose.to(q), jac.to(q)

                return kinematics

            delegate = TaskProgramSameGridExpansionFactory(
                operator=(
                    "task_space_residual"
                    if recipe == "task_space"
                    else "joint_residual"
                ),
                seed=11 if recipe == "joint_11" else 7,
                motion_headroom_fraction=(
                    profile.motion_headroom_fraction if recipe != "task_space" else None
                ),
                geometry_family_id=variant.variant_id,
                initial_state_id=preparation.initial_state_id,
                kinematics_factory=(
                    kinematics_factory if recipe == "task_space" else None
                ),
                **common,
            )
        guard = TabletopCollisionGuardFactory(
            validator,
            delegate,
            velocity_limits=common["velocity_limits"],
            acceleration_limits=common["acceleration_limits"],
            ik_smoothing_trigger_fraction=(
                profile.motion_headroom_fraction if recipe == "task_space" else None
            ),
        )
        tracer = adapter.create_traced_configured_sources(program, guard)
        segments = environment.create_demo_segments(
            task_program=program,
            nominal_source_factory=tracer,
            plan_transform_factory=guard,
        )
        environment._active_task_program_bridge.enable_execution_trace()
        state.clear()
        state.update(
            variant=variant,
            preparation=preparation,
            validator=validator,
            guard=guard,
            tracer=tracer,
            delegate=delegate,
            recipe=recipe,
            attempt_id=attempt_id,
            layout=layout["id"],
            layout_parameters=dict(layout),
        )
        return segments

    def capture(environment, result):
        length = int(environment._traj_steps[0].item())
        if length != result.length or environment._traj_buffer.shape[1] <= length:
            raise ValueError("realized state stream is incomplete")
        states = environment._traj_buffer["states"]
        qpos = states["robot", "qpos"][0, : length + 1]
        objects = (
            states["rigid_objects", profile.object_uid, "pose"][0, : length + 1]
            .detach()
            .cpu()
            .numpy()
        )
        target_pose = (
            environment.sim.get_rigid_object(profile.target_uid)
            .get_local_pose(to_matrix=True)[0]
            .detach()
            .cpu()
            .numpy()
        )
        tracer = state["tracer"]
        delegate = state["delegate"]
        guard = state["guard"]
        if set(tracer.motions) != set(call_indices):
            raise ValueError("not every configured source occurrence was executed")
        retime_maps = (
            {}
            if not isinstance(delegate, TaskProgramMappedRetimeFactory)
            else {item.call_index: item.result.time_map for item in delegate.records}
        )
        arrays = build_real2sim_execution_trace(
            environment._active_task_program_bridge.execution_trace,
            tracer.motions,
            retime_maps=retime_maps,
        )
        if len(arrays["step_call_index"]) != length:
            raise ValueError("observed action trace does not cover the rollout")
        skills = {
            index: prototypes[index].unit.source.skill_id for index in call_indices
        }
        events = _observed_program_phases(
            skills,
            guard.segments,
            arrays["step_call_index"],
            arrays["step_plan_index"],
        )
        phase_starts = {(index, name): start for start, index, _, name in events}
        call_starts = {
            index: next(start for start, call, _, _ in events if call == index)
            for index in call_indices
        }
        matrices = np.repeat(np.eye(4)[None], length + 1, axis=0)
        matrices[:, :3, :3] = Rotation.from_quat(objects[:, 3:7]).as_matrix()
        matrices[:, :3, 3] = objects[:, :3]
        outcomes = []
        held_intervals = []
        held_start = None
        held_endpoint = None
        held_call = None
        for position, index in enumerate(call_indices):
            skill = skills[index]
            start = call_starts[index]
            stop = (
                call_starts[call_indices[position + 1]]
                if position + 1 < len(call_indices)
                else length
            )
            endpoint = guard.motion_targets[index]
            outcome = {
                "call_index": index,
                "skill_id": skill,
                "start_step": start,
                "end_step": stop,
                "passed": True,
            }
            if skill == "pick_up":
                if held_start is not None:
                    raise ValueError("a Pick cannot begin while the object is held")
                lift = float(objects[stop, 2] - objects[start, 2])
                outcome.update(lift_m=lift, passed=lift >= profile.minimum_lift_m)
                held_start = phase_starts[index, "lift"]
                held_endpoint, held_call = endpoint, index
            else:
                if held_start is None:
                    if position:
                        raise ValueError(
                            "the call has no preceding held-object relation"
                        )
                    # Initial held state must still be verified by the live plan guard.
                    held_start, held_endpoint, held_call = start, endpoint, index
                if endpoint != held_endpoint:
                    raise ValueError("a held interval cannot switch motion endpoints")
                if skill == "place":
                    release = phase_starts[index, "release"]
                    held_intervals.append(
                        (held_start, release, held_endpoint, held_call, index)
                    )
                    release_target = guard.release_targets.get(index)
                    if release_target is None:
                        raise ValueError("Place is missing its verified release target")
                    xy_error = float(
                        np.linalg.norm(objects[stop, :2] - release_target[:2, 3])
                    )
                    height_error, overhang = _support_errors(
                        matrices[stop], target_pose, object_body, target_body
                    )
                    stable_start = max(phase_starts[index, "retract"], stop - 19)
                    drift = float(
                        np.linalg.norm(objects[stop, :3] - objects[stable_start, :3])
                    )
                    outcome.update(
                        expected_object_xy=release_target[:2, 3].tolist(),
                        placement_xy_error_m=xy_error,
                        height_error_m=height_error,
                        support_overhang_m=overhang,
                        stability_start_step=stable_start,
                        drift_m=drift,
                        passed=(
                            xy_error <= profile.placement_xy_tolerance_m
                            and height_error <= profile.placement_height_tolerance_m
                            and overhang <= profile.contact_depth_m
                            and drift <= profile.maximum_drift_m
                        ),
                    )
                    held_start = held_endpoint = held_call = None
                elif skill != "move_held_object":
                    raise ValueError("unsupported source-backed tabletop skill")
            outcomes.append(outcome)
        if held_start is not None:
            held_intervals.append(
                (held_start, length, held_endpoint, held_call, call_indices[-1])
            )
        held_checks = []
        for start, stop, endpoint, first_call, last_call in held_intervals:
            if not 0 <= start < stop <= length:
                raise ValueError("observed held-object interval is invalid")
            control_part, joint_ids = endpoint
            tcp = (
                environment.robot.compute_batch_fk(
                    qpos=qpos[None, start : stop + 1, list(joint_ids)],
                    name=control_part,
                    to_matrix=True,
                )[0]
                .detach()
                .cpu()
                .numpy()
            )
            translation, rotation = _held_pose_errors(objects[start : stop + 1], tcp)
            held_checks.append(
                {
                    "first_call_index": first_call,
                    "last_call_index": last_call,
                    "start_step": start,
                    "end_step": stop,
                    "held_translation_drift_m": translation,
                    "held_rotation_drift_deg": rotation,
                    "passed": (
                        translation <= profile.held_translation_tolerance_m
                        and rotation <= profile.held_rotation_tolerance_deg
                    ),
                }
            )
        metrics = {}
        for name, aggregate in (
            ("lift_m", min),
            ("placement_xy_error_m", max),
            ("height_error_m", max),
            ("support_overhang_m", max),
            ("drift_m", max),
        ):
            values = [item[name] for item in outcomes if name in item]
            if values:
                metrics[name] = aggregate(values)
        for name in ("held_translation_drift_m", "held_rotation_drift_deg"):
            if held_checks:
                metrics[name] = max(item[name] for item in held_checks)
        task_ok = all(item["passed"] for item in (*outcomes, *held_checks))
        validator = state["validator"]
        phases = tuple(
            validator.phase_policy(
                skill,
                phase,
                0 if i == 0 else start,
                events[i + 1][0] if i + 1 < len(events) else length + 1,
                object_supported_by_target=(
                    skill == "pick_up"
                    and all(
                        value <= profile.contact_depth_m
                        for value in _support_errors(
                            matrices[call_starts[index]],
                            target_pose,
                            object_body,
                            target_body,
                        )
                    )
                ),
            )
            for i, (start, index, skill, phase) in enumerate(events)
        )
        observed = validator.validate_observed(
            qpos, object_poses=matrices, target_pose=target_pose, phases=phases
        )
        expected_reports = len(call_indices) * (1 if delegate is None else 2)
        reports = validator.reports
        collision_ok = len(reports) == expected_reports and all(
            report.accepted for _, report in reports
        )
        motion_metrics = state["guard"].motion_metrics
        motion_ok = set(motion_metrics) == set(call_indices) and all(
            max(item.values(), default=float("inf")) <= 1.0 + 1e-6
            for item in motion_metrics.values()
        )
        phase_validation = []
        is_retime = isinstance(state["delegate"], TaskProgramMappedRetimeFactory)
        duration_scale = 1.15 if state["recipe"] == "time_115" else 1.30
        for index, samples in state["guard"].phase_samples.items():
            permissions = {
                item.name: item.allowed_operators
                for item in prototypes[index].unit.phases
            }
            for name, source_count, final_count in samples:
                retimed = is_retime and "retime" in permissions.get(name, ())
                expected_count = (
                    math.ceil((source_count - 1) * duration_scale - 1e-9) + 1
                    if retimed
                    else source_count
                )
                phase_validation.append(
                    {
                        "call_index": index,
                        "phase": name,
                        "source_samples": source_count,
                        "final_samples": final_count,
                        "expected_samples": expected_count,
                        "retimed": retimed,
                        "passed": final_count == expected_count,
                    }
                )
        phases_ok = set(state["guard"].phase_samples) == set(call_indices) and all(
            item["passed"] for item in phase_validation
        )
        validation = ValidationResult(
            (
                ValidationCheck(
                    "geometry",
                    "passed",
                    "prototype and physical geometry identities match",
                ),
                ValidationCheck(
                    "motion_limits",
                    "passed" if motion_ok else "failed",
                    "sampled full-joint velocity and acceleration",
                ),
                ValidationCheck(
                    "phase_mapping",
                    "passed" if phases_ok else "failed",
                    "semantic phase durations and sample mapping",
                ),
                ValidationCheck(
                    "plan_collision",
                    "passed" if collision_ok else "failed",
                    "sampled plans with gripper contact projection",
                ),
                ValidationCheck(
                    "observed_collision",
                    "passed" if observed.accepted else "failed",
                    str(observed.reason or "sampled actual states"),
                ),
                ValidationCheck(
                    "task",
                    "passed" if task_ok else "failed",
                    metrics=metrics,
                ),
            )
        )
        if not validation.accepted:
            raise ValueError(
                "Real2Sim measured acceptance failed: "
                + str(
                    [
                        (item.check_id, item.status, item.detail)
                        for item in validation.checks
                    ]
                )
            )
        variant = state["variant"]
        payload = {
            "program": tracer.program_fingerprint,
            "layout": variant.variant_id,
            "recipe": state["recipe"],
            "seed": profile.seed,
            "prototypes": [item.prototype_sha256 for item in declarations],
        }
        if any(item.refinement is not None for item in declarations):
            payload["refinement"] = [
                tracer.motions[item.call_index].refinement_report.get("baseline_id")
                for item in declarations
            ]
        candidate = (
            "candidate:"
            + hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        )
        trace_dir = (
            Path(recorder.dataset_full_path) / "meta" / "real2sim" / "host-traces"
        )
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace = trace_dir / (
            hashlib.sha256(
                (candidate + ":" + str(state["attempt_id"])).encode()
            ).hexdigest()
            + ".npz"
        )
        with trace.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        observed_path = trace.with_name(trace.stem + "-observed.npz")
        with observed_path.open("xb") as stream:
            np.savez_compressed(
                stream,
                object_pose=objects,
                target_pose=target_pose,
                timestamps=np.arange(length + 1, dtype=np.float64)
                * environment.step_dt,
            )
            stream.flush()
            os.fsync(stream.fileno())
        bundles, calls = {}, []
        variants = (
            {}
            if delegate is None
            else {item.call_index: item for item in delegate.records}
        )
        for item in declarations:
            index, root = item.call_index, Path(item.prototype_manifest).parent
            motion = tracer.motions[index]
            if motion.authority in {"eef_pose", "hybrid"} and (
                motion.fk_position_error_m is None
                or motion.fk_rotation_error_rad is None
            ):
                raise ValueError(
                    "Cartesian prototype is missing numerical FK residuals"
                )
            bundles[str(index)] = {
                "prototype_manifest": _artifact(root / "prototype.json"),
                "normalized_arrays": _artifact(root / "normalized.npz"),
                "source_manifest": _artifact(root / "source" / "trajectory.json"),
                "source_arrays": _artifact(root / "source" / "trajectory.npz"),
            }
            call = {
                "call_index": index,
                "source_bundle": str(index),
                "geometry_family_id": variant.variant_id,
                "source_sample_offset": motion.source_sample_offset,
                "operator": "nominal",
                "prototype_sha256": item.prototype_sha256,
                "source_id": motion.source_id,
                "source_revision": motion.source_revision,
                "source_unit_id": motion.unit_id,
                "source_origin": motion.origin,
                "authority": motion.authority,
                "pose_origin": motion.pose_origin,
                "object_supported_regrasp": motion.object_supported_regrasp,
                "semantic_anchor_position_correction_m": motion.semantic_anchor_position_correction_m,
                "semantic_anchor_rotation_correction_rad": motion.semantic_anchor_rotation_correction_rad,
                "ik_smoothing_iterations": motion.ik_smoothing_iterations,
                "ik_hold_smoothing_iterations": motion.ik_hold_smoothing_iterations,
                "ik_smoothing_trigger_fraction": guard.ik_smoothing_trigger_fraction,
                "fk_position_error_m": motion.fk_position_error_m,
                "fk_rotation_error_rad": motion.fk_rotation_error_rad,
            }
            if motion.refinement_report:
                call["refinement"] = dict(motion.refinement_report)
            if isinstance(delegate, TaskProgramSameGridExpansionFactory):
                record = variants[index]
                call.update(
                    operator=record.operator,
                    seed=record.seed,
                    candidate_id=record.candidate_id,
                    variant_ordinal=record.variant_ordinal,
                    attempted=record.attempted,
                    rejected=dict(record.rejected),
                    locked_joint_indices=record.locked_joint_indices,
                    locked_joint_names=[
                        environment.robot.joint_names[index]
                        for index in record.locked_joint_indices
                    ],
                    motion_headroom_fraction=record.motion_headroom_fraction,
                )
            elif isinstance(delegate, TaskProgramMappedRetimeFactory):
                record = variants[index]
                call.update(operator="retime", duration_scale=record.duration_scale)
            calls.append(call)
        first = prototypes[call_indices[0]].unit.source
        origins = {prototype.unit.source.origin for prototype in prototypes.values()}
        identity = CandidateIdentity(
            variant.variant_id,
            state["preparation"].initial_state_id,
            candidate,
            variant.variant_id,
            first.source_id,
            first.source_revision,
            "real2sim_program_episode",
            attempt_id=state["attempt_id"],
        )
        return Real2SimCaptureRequest(
            identity,
            validation,
            f"episode:{candidate}:{state['attempt_id']}",
            f"commit:{candidate}:{state['attempt_id']}",
            observations,
            "joint_position",
            {
                "origin": first.origin if len(origins) == 1 else "mixed",
                "acceleration_limit_origin": profile.acceleration_limit_origin,
                "acceleration_limits": profile.acceleration_limits,
                "velocity_limit_origin": profile.velocity_limit_origin,
                "velocity_limits": profile.velocity_limits,
                "sampled_motion_limits": {
                    str(index): dict(value) for index, value in motion_metrics.items()
                },
                "phase_duration_validation": phase_validation,
                "task_call_validation": outcomes,
                "held_interval_validation": held_checks,
                "layout": state["layout"],
                "layout_parameters": state["layout_parameters"],
                "scene_variant": {
                    "variant_id": variant.variant_id,
                    "parent_scene_id": variant.parent_scene_id,
                    "seed": variant.seed,
                    "ordinal": variant.ordinal,
                    "pose_changes": [
                        {"entity_uid": item.entity_uid, "arena_pose": item.arena_pose}
                        for item in variant.pose_changes
                    ],
                },
                "scene_initial_state": state["preparation"].metadata,
                "recipe": state["recipe"],
                "sampled_plan_collisions": [
                    {"unit_id": unit, **asdict(check)} for unit, check in reports
                ],
                "sampled_observed_collision": asdict(observed),
                "real2sim_observed_geometry": _artifact(observed_path),
                "real2sim_source_artifacts": bundles,
                "real2sim_lineage": {
                    "scene_revision": profile.scene_revision,
                    "program_fingerprint": tracer.program_fingerprint,
                    "trace": _artifact(trace),
                    "calls": calls,
                },
            },
        )

    def record_failed_attempt(environment, result, error):
        directory = (
            Path(recorder.dataset_full_path) / "meta" / "real2sim" / "failed-attempts"
        )
        directory.mkdir(parents=True, exist_ok=True)
        path = (
            directory
            / f"episode-{result.episode_index}-attempt-{result.attempt_id}.json"
        )
        evidence = {
            "layout": state.get("layout"),
            "recipe": state.get("recipe"),
            "error": error,
            "execution": result.to_metadata(),
            "sampled_plan_collisions": [
                {"unit_id": unit, **asdict(check)}
                for unit, check in state["validator"].reports
            ],
        }
        with path.open("x", encoding="utf-8") as stream:
            json.dump(evidence, stream, indent=2, allow_nan=False)

    owner = Real2SimRunTaskOwner(
        sink,
        prepare_attempt=prepare,
        capture_request=capture,
        record_failed_attempt=record_failed_attempt,
    )
    target._real2sim_collection_owner = owner
    return owner
