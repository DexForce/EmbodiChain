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

"""Press, slide, and twist benchmark cases."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

from embodichain.compute.geometry import nearest_half_turn_pose

from embodichain.lab.sim.atomic_actions import (
    ActionInvocation,
    ObjectSemantics,
    PressAffordance,
    PressGoal,
    PressOptions,
    SlideAffordance,
    SlideGoal,
    SlideOptions,
    TwistAffordance,
    TwistGoal,
    TwistOptions,
)
from embodichain.lab.sim.atomic_actions.plans import CompiledTrajectory

from ..atomic_skill import AtomicSkillCaseProvider, register_atomic_skill_provider
from ..contracts import ReplayObservationWindow
from ..config import SuiteCfg, TrackCfg
from ..models import BenchmarkCase
from .atomic_evaluation import (
    PressEvaluator,
    SlideEvaluator,
    TwistEvaluator,
)
from .atomic_case_utils import (
    atomic_case_id,
    case_name,
    case_pose,
    difficulty,
    randomization_parameters,
    slide_direction,
    snapshot_case_qpos,
    twist_arc_poses,
)
from .atomic_policy import benchmark_motion_policy

if TYPE_CHECKING:
    from ..planners.base import PlannerAdapter
    from .atomic_skill import AtomicSkillScenario

__all__ = ["PressCases", "SlideCases", "TwistCases"]


class PressCases(AtomicSkillCaseProvider):
    """Close the gripper, reach a contact pose, and retract."""

    skill_id = "press"
    requires_gripper = True

    def generate_case(
        self,
        scenario: "AtomicSkillScenario",
        suite: SuiteCfg,
        track: TrackCfg,
        config: Mapping[str, object],
        *,
        seed: int,
        batch_size: int,
    ) -> BenchmarkCase:
        articulation_id = config.get("articulation")
        target_link = config.get("target_link")
        target_joint = config.get("target_joint")
        if not isinstance(articulation_id, str) or not articulation_id:
            raise ValueError("Press cases must reference an articulation id.")
        if not isinstance(target_link, str) or not target_link:
            raise ValueError("Press cases must define a target_link.")
        if not isinstance(target_joint, str) or not target_joint:
            raise ValueError("Press cases must define a target_joint.")
        handle = scenario.activate_articulation(articulation_id)
        scenario.validate_articulation_target(
            handle,
            target_link=target_link,
            target_joint=target_joint,
            joint_type="prismatic",
        )
        scenario.restore_base_robot()
        articulation_initial_pose = scenario.randomize_articulation_pose(
            handle,
            config,
            seed=seed,
            stream=32,
        )
        scenario.randomize_robot_start(config, seed=seed, stream=31)
        start_qpos, full_start_qpos = snapshot_case_qpos(
            scenario.robot, scenario.control_part
        )
        start_pose = scenario.robot.compute_fk(
            start_qpos, name=scenario.control_part, to_matrix=True
        )
        articulation_initial_qpos = handle.entity.get_qpos().clone()
        joint_index = handle.entity.joint_names.index(target_joint)
        joint_limits = handle.entity.get_qpos_limits(joint_ids=[joint_index])[0, 0]
        articulation_joint_stroke = float((joint_limits[1] - joint_limits[0]).item())
        if (
            not math.isfinite(articulation_joint_stroke)
            or articulation_joint_stroke <= 0
        ):
            raise ValueError("Press target joint must expose a finite positive stroke.")
        target_pose = handle.link_pose(target_link).to(device=start_pose.device)
        geometry = scenario.sample_articulation_geometry(
            handle,
            target_link,
            seed=seed,
            config=config,
        )
        affordance = PressAffordance()
        ObjectSemantics(
            affordance=affordance,
            geometry=geometry,
            entity_id=handle.entity.uid,
            label="microwave_start_button",
        )
        if affordance.press_position is None:
            raise RuntimeError(
                "Press articulation geometry did not resolve a contact point."
            )
        press_axis = affordance.press_axis.detach().cpu().tolist()
        press_position = list(affordance.press_position)
        options = PressOptions(
            hand_interp_steps=int(config.get("hand_interp_steps", 8)),
            approach_distance=float(config.get("approach_distance_m", 0.1)),
            press_distance=float(config.get("press_distance_m", 0.05)),
            press_position=None,
        )
        contact_pose = affordance.get_press_pose(
            target_pose,
            press_position=options.press_position,
        )
        contact_pose = nearest_half_turn_pose(contact_pose, start_pose)
        approach_pose = contact_pose.clone()
        approach_pose[:, :3, 3] -= contact_pose[:, :3, 2] * options.approach_distance
        pressed_pose = contact_pose.clone()
        pressed_pose[:, :3, 3] += contact_pose[:, :3, 2] * options.press_distance
        targets = torch.stack(
            [approach_pose, contact_pose, pressed_pose, approach_pose],
            dim=1,
        )
        references = scenario.solve_reference_qpos(start_qpos, targets)
        name = case_name(config)
        return BenchmarkCase(
            suite_version=suite.suite_version,
            track=track.id,
            scenario_id=self.skill_id,
            case_id=atomic_case_id(
                track.id,
                self.skill_id,
                name,
                batch_size=batch_size,
                seed=seed,
            ),
            seed=seed,
            batch_size=batch_size,
            num_waypoints=4,
            path_shape="approach_contact_press_retract",
            start_state_bin="pre_action",
            start_qpos=start_qpos,
            target_waypoints=targets,
            reference_qpos=references,
            robot_id=suite.robot.id,
            skill_id=self.skill_id,
            object_id=articulation_id,
            task_difficulty=difficulty(config),
            primary_success="physical_skill_success",
            full_start_qpos=full_start_qpos,
            case_parameters={
                "sample_count": int(config.get("sample_count", 80)),
                "press_target_pose": target_pose.detach().cpu().tolist(),
                "press_axis": press_axis,
                "press_position": press_position,
                "hand_interp_steps": options.hand_interp_steps,
                "approach_distance_m": options.approach_distance,
                "press_distance_m": options.press_distance,
                "articulation_initial_pose": articulation_initial_pose.detach()
                .cpu()
                .tolist(),
                "articulation_translation_offsets_m": (
                    articulation_initial_pose[:, :3, 3]
                    - handle.initial_pose[:, :3, 3].to(
                        device=articulation_initial_pose.device,
                        dtype=articulation_initial_pose.dtype,
                    )
                )
                .detach()
                .cpu()
                .tolist(),
                "articulation_initial_qpos": articulation_initial_qpos.detach()
                .cpu()
                .tolist(),
                "target_link": target_link,
                "target_joint": target_joint,
                "target_entity_id": f"{handle.entity.uid}:{target_link}",
                "effect_joint_sign": float(config.get("effect_joint_sign", 1.0)),
                "articulation_joint_stroke": articulation_joint_stroke,
                "contact_minimum_joint_delta": float(
                    config.get("contact_minimum_joint_delta", 1.0e-5)
                ),
                "waypoint_rotation_symmetry": "half_turn_about_z",
                "randomization": randomization_parameters(config, seed=seed),
                "difficulty_factors": dict(config.get("difficulty_factors", {})),
            },
        )

    def build_invocation(
        self,
        scenario: "AtomicSkillScenario",
        case: BenchmarkCase,
        adapter: "PlannerAdapter",
    ) -> ActionInvocation:
        del adapter
        press_axis = torch.tensor(
            case.case_parameters["press_axis"],
            dtype=torch.float32,
            device=scenario.robot.device,
        )
        press_position = tuple(case.case_parameters["press_position"])
        semantics = ObjectSemantics(
            affordance=PressAffordance(
                press_axis=press_axis,
                press_position=press_position,
            ),
            geometry={},
            entity_id=str(case.case_parameters["target_entity_id"]),
            label="microwave_start_button",
        )
        return scenario.require_engine().make_invocation(
            self.skill_id,
            PressGoal(
                semantics,
                case_pose(
                    case,
                    "press_target_pose",
                    device=scenario.robot.device,
                ),
            ),
            control_parts={
                "primary": {
                    "motion": scenario.control_part,
                    "grasp": scenario.end_effector_part,
                }
            },
            motion_policy=benchmark_motion_policy(
                self.skill_id, int(case.case_parameters["sample_count"])
            ),
            skill_options=PressOptions(
                hand_interp_steps=int(case.case_parameters["hand_interp_steps"]),
                approach_distance=float(case.case_parameters["approach_distance_m"]),
                press_distance=float(case.case_parameters["press_distance_m"]),
                press_position=None,
            ),
        )

    def articulation_effect_segment(self, case: BenchmarkCase) -> str:
        """Start button-displacement diagnostics at the pressing segment."""
        del case
        return "press"

    def replay_observation_windows(
        self, compiled: CompiledTrajectory, case: BenchmarkCase
    ) -> tuple[ReplayObservationWindow, ...]:
        """Collect the target-joint trace only during commanded pressing."""
        del case
        segment = compiled.segment(0, "press")
        return (
            ReplayObservationWindow(
                "press_actuation",
                segment.start,
                segment.stop,
                frozenset({"articulation_joint"}),
            ),
        )

    def physical_evaluator(self) -> PressEvaluator:
        """Return the v1 press-stage evaluator."""
        return PressEvaluator()


class SlideCases(AtomicSkillCaseProvider):
    """Approach, grasp, and physically translate an articulation link."""

    skill_id = "slide"
    requires_gripper = True

    def generate_case(
        self,
        scenario: "AtomicSkillScenario",
        suite: SuiteCfg,
        track: TrackCfg,
        config: Mapping[str, object],
        *,
        seed: int,
        batch_size: int,
    ) -> BenchmarkCase:
        articulation_id = config.get("articulation")
        target_link = config.get("target_link")
        target_joint = config.get("target_joint")
        if not isinstance(articulation_id, str) or not articulation_id:
            raise ValueError("Slide cases must reference an articulation id.")
        if not isinstance(target_link, str) or not target_link:
            raise ValueError("Slide cases must define a target_link.")
        if not isinstance(target_joint, str) or not target_joint:
            raise ValueError("Slide cases must define a target_joint.")
        handle = scenario.activate_articulation(articulation_id)
        scenario.validate_articulation_target(
            handle,
            target_link=target_link,
            target_joint=target_joint,
            joint_type="prismatic",
        )
        scenario.restore_base_robot()
        articulation_initial_pose = scenario.randomize_articulation_pose(
            handle,
            config,
            seed=seed,
            stream=42,
        )
        scenario.randomize_robot_start(config, seed=seed, stream=41)
        start_qpos, full_start_qpos = snapshot_case_qpos(
            scenario.robot, scenario.control_part
        )
        articulation_initial_qpos = handle.entity.get_qpos().clone()
        target_pose = handle.link_pose(target_link).to(device=start_qpos.device)
        mesh_vertices, mesh_triangles = handle.link_mesh(target_link)
        mesh_vertices = mesh_vertices.to(device=target_pose.device, dtype=torch.float32)
        mesh_triangles = mesh_triangles.to(device=target_pose.device, dtype=torch.long)
        geometry = scenario.sample_articulation_geometry(
            handle,
            target_link,
            seed=seed,
            config=config,
        )
        affordance = SlideAffordance(
            mesh_vertices=mesh_vertices,
            mesh_triangles=mesh_triangles,
        )
        ObjectSemantics(
            affordance=affordance,
            geometry=geometry,
            entity_id=handle.entity.uid,
            label="drawer_large_handle",
        )
        axis = affordance.translation_axis.to(
            device=target_pose.device, dtype=torch.float32
        )
        axis = axis / torch.linalg.vector_norm(axis)
        translation_axis = axis.detach().cpu().tolist()
        world_axis = torch.matmul(target_pose[:, :3, :3], axis)
        grasp_pose = scenario.resolve_articulation_grasp(
            mesh_vertices=mesh_vertices,
            mesh_triangles=mesh_triangles,
            target_pose=target_pose,
            approach_direction=world_axis,
            start_qpos=start_qpos,
            direction=slide_direction(config.get("direction", "pull")),
            approach_distance=float(config.get("approach_distance_m", 0.1)),
            translation_distance=float(config.get("translation_distance_m", 0.18)),
            seed=seed,
            n_sample=int(config.get("grasp_sample_count", 10000)),
            max_candidates=int(config.get("grasp_max_candidates", 128)),
            alignment_max_angle_deg=float(
                config.get("grasp_alignment_max_angle_deg", 10.0)
            ),
        )
        options = SlideOptions(
            direction=slide_direction(config.get("direction", "pull")),
            hand_interp_steps=int(config.get("hand_interp_steps", 12)),
            approach_distance=float(config.get("approach_distance_m", 0.1)),
            translation_distance=float(config.get("translation_distance_m", 0.18)),
        )
        approach_pose = grasp_pose.clone()
        approach_pose[:, :3, 3] -= world_axis * options.approach_distance
        translation_sign = -1.0 if options.direction == "pull" else 1.0
        translated_pose = grasp_pose.clone()
        translated_pose[:, :3, 3] += (
            world_axis * translation_sign * options.translation_distance
        )
        target_values = [approach_pose, grasp_pose, translated_pose]
        if options.direction == "push":
            target_values.append(approach_pose)
        targets = torch.stack(target_values, dim=1)
        references = scenario.solve_reference_qpos(start_qpos, targets)
        name = case_name(config)
        return BenchmarkCase(
            suite_version=suite.suite_version,
            track=track.id,
            scenario_id=self.skill_id,
            case_id=atomic_case_id(
                track.id,
                self.skill_id,
                name,
                batch_size=batch_size,
                seed=seed,
            ),
            seed=seed,
            batch_size=batch_size,
            num_waypoints=targets.shape[1],
            path_shape=f"approach_grasp_{options.direction}",
            start_state_bin="pre_action",
            start_qpos=start_qpos,
            target_waypoints=targets,
            reference_qpos=references,
            robot_id=suite.robot.id,
            skill_id=self.skill_id,
            object_id=articulation_id,
            task_difficulty=difficulty(config),
            primary_success="physical_skill_success",
            full_start_qpos=full_start_qpos,
            case_parameters={
                "sample_count": int(config.get("sample_count", 140)),
                "slide_target_pose": target_pose.detach().cpu().tolist(),
                "translation_axis": translation_axis,
                "mesh_vertices": mesh_vertices.detach().cpu().tolist(),
                "mesh_triangles": mesh_triangles.detach().cpu().tolist(),
                "grasp_pose": grasp_pose.detach().cpu().tolist(),
                "grasp_source": str(config.get("grasp_source", "antipodal")),
                "grasp_sample_count": int(config.get("grasp_sample_count", 10000)),
                "grasp_max_candidates": int(config.get("grasp_max_candidates", 128)),
                "grasp_alignment_max_angle_deg": float(
                    config.get("grasp_alignment_max_angle_deg", 10.0)
                ),
                "direction": options.direction,
                "hand_interp_steps": options.hand_interp_steps,
                "approach_distance_m": options.approach_distance,
                "translation_distance_m": options.translation_distance,
                "articulation_initial_pose": articulation_initial_pose.detach()
                .cpu()
                .tolist(),
                "articulation_translation_offsets_m": (
                    articulation_initial_pose[:, :3, 3]
                    - handle.initial_pose[:, :3, 3].to(
                        device=articulation_initial_pose.device,
                        dtype=articulation_initial_pose.dtype,
                    )
                )
                .detach()
                .cpu()
                .tolist(),
                "articulation_initial_qpos": articulation_initial_qpos.detach()
                .cpu()
                .tolist(),
                "target_link": target_link,
                "target_joint": target_joint,
                "target_entity_id": f"{handle.entity.uid}:{target_link}",
                "effect_joint_sign": float(config.get("effect_joint_sign", 1.0)),
                "randomization": randomization_parameters(config, seed=seed),
                "difficulty_factors": dict(config.get("difficulty_factors", {})),
            },
        )

    def build_invocation(
        self,
        scenario: "AtomicSkillScenario",
        case: BenchmarkCase,
        adapter: "PlannerAdapter",
    ) -> ActionInvocation:
        del adapter
        if scenario.end_effector_part is None:
            raise RuntimeError("Slide requires a configured end-effector control part.")
        mesh_vertices = torch.tensor(
            case.case_parameters["mesh_vertices"],
            dtype=torch.float32,
            device=scenario.robot.device,
        )
        mesh_triangles = torch.tensor(
            case.case_parameters["mesh_triangles"],
            dtype=torch.long,
            device=scenario.robot.device,
        )
        semantics = ObjectSemantics(
            affordance=SlideAffordance(
                mesh_vertices=mesh_vertices,
                mesh_triangles=mesh_triangles,
                translation_axis=torch.tensor(
                    case.case_parameters["translation_axis"],
                    dtype=torch.float32,
                    device=scenario.robot.device,
                ),
            ),
            geometry={},
            entity_id=str(case.case_parameters["target_entity_id"]),
            label="drawer_large_handle",
        )

        return scenario.require_engine().make_invocation(
            self.skill_id,
            SlideGoal(
                semantics,
                case_pose(
                    case,
                    "slide_target_pose",
                    device=scenario.robot.device,
                ),
            ),
            control_parts={
                "primary": {
                    "motion": scenario.control_part,
                    "grasp": scenario.end_effector_part,
                }
            },
            motion_policy=benchmark_motion_policy(
                self.skill_id, int(case.case_parameters["sample_count"])
            ),
            skill_options=SlideOptions(
                direction=slide_direction(case.case_parameters["direction"]),
                hand_interp_steps=int(case.case_parameters["hand_interp_steps"]),
                approach_distance=float(case.case_parameters["approach_distance_m"]),
                translation_distance=float(
                    case.case_parameters["translation_distance_m"]
                ),
            ),
        )

    def articulation_effect_segment(self, case: BenchmarkCase) -> str:
        """Start drawer-displacement diagnostics at the pull or push segment."""
        return slide_direction(case.case_parameters["direction"])

    def replay_observation_windows(
        self, compiled: CompiledTrajectory, case: BenchmarkCase
    ) -> tuple[ReplayObservationWindow, ...]:
        """Collect the commanded slide and release windows."""
        actuation = compiled.segment(
            0, slide_direction(case.case_parameters["direction"])
        )
        release = compiled.segment(0, "open")
        return (
            ReplayObservationWindow(
                "slide_actuation",
                actuation.start,
                actuation.stop,
                frozenset({"articulation_joint"}),
            ),
            ReplayObservationWindow(
                "slide_release",
                release.start,
                release.stop,
                frozenset({"gripper_qpos"}),
            ),
        )

    def physical_evaluator(self) -> SlideEvaluator:
        """Return the v1 slide-stage evaluator."""
        return SlideEvaluator()


class TwistCases(AtomicSkillCaseProvider):
    """Approach, grasp, and physically rotate an articulation link."""

    skill_id = "twist"
    requires_gripper = True

    def generate_case(
        self,
        scenario: "AtomicSkillScenario",
        suite: SuiteCfg,
        track: TrackCfg,
        config: Mapping[str, object],
        *,
        seed: int,
        batch_size: int,
    ) -> BenchmarkCase:
        articulation_id = config.get("articulation")
        target_link = config.get("target_link")
        target_joint = config.get("target_joint")
        if not isinstance(articulation_id, str) or not articulation_id:
            raise ValueError("Twist cases must reference an articulation id.")
        if not isinstance(target_link, str) or not target_link:
            raise ValueError("Twist cases must define a target_link.")
        if not isinstance(target_joint, str) or not target_joint:
            raise ValueError("Twist cases must define a target_joint.")
        handle = scenario.activate_articulation(articulation_id)
        scenario.validate_articulation_target(
            handle,
            target_link=target_link,
            target_joint=target_joint,
            joint_type="revolute",
        )
        scenario.restore_base_robot()
        articulation_initial_pose = scenario.randomize_articulation_pose(
            handle,
            config,
            seed=seed,
            stream=52,
        )
        scenario.randomize_robot_start(config, seed=seed, stream=51)
        start_qpos, full_start_qpos = snapshot_case_qpos(
            scenario.robot, scenario.control_part
        )
        start_pose = scenario.robot.compute_fk(
            start_qpos, name=scenario.control_part, to_matrix=True
        )
        articulation_initial_qpos = handle.entity.get_qpos().clone()
        target_pose = handle.link_pose(target_link).to(device=start_pose.device)
        mesh_vertices, _ = handle.link_mesh(target_link)
        grasp_position_tensor = mesh_vertices.to(
            device=target_pose.device, dtype=torch.float32
        ).mean(dim=0)
        geometry = scenario.sample_articulation_geometry(
            handle,
            target_link,
            seed=seed,
            config=config,
        )
        affordance = TwistAffordance(
            grasp_position=tuple(float(value) for value in grasp_position_tensor),
        )
        ObjectSemantics(
            affordance=affordance,
            geometry=geometry,
            entity_id=handle.entity.uid,
            label="microwave_power_knob",
        )
        grasp_position = list(affordance.grasp_position)
        axis_origin = list(affordance.require_axis_origin())
        twist_axis = affordance.twist_axis.detach().cpu().tolist()
        options = TwistOptions(
            hand_interp_steps=int(config.get("hand_interp_steps", 12)),
            twist_waypoint_count=int(config.get("twist_waypoint_count", 8)),
            pre_grasp_distance=float(config.get("pre_grasp_distance_m", 0.12)),
            twist_angle=float(config.get("twist_angle_rad", -0.7853981634)),
        )
        grasp_pose = nearest_half_turn_pose(
            affordance.get_grasp_pose(target_pose),
            start_pose,
        )
        pre_grasp_pose = grasp_pose.clone()
        pre_grasp_pose[:, :3, 3] -= grasp_pose[:, :3, 2] * options.pre_grasp_distance
        twist_poses = twist_arc_poses(
            target_pose,
            grasp_pose,
            affordance.twist_axis,
            affordance.require_axis_origin(),
            options.twist_angle,
            options.twist_waypoint_count,
        )
        targets = torch.cat(
            (
                pre_grasp_pose[:, None],
                grasp_pose[:, None],
                twist_poses,
                pre_grasp_pose[:, None],
            ),
            dim=1,
        )
        references = scenario.solve_reference_qpos(start_qpos, targets)
        name = case_name(config)
        return BenchmarkCase(
            suite_version=suite.suite_version,
            track=track.id,
            scenario_id=self.skill_id,
            case_id=atomic_case_id(
                track.id,
                self.skill_id,
                name,
                batch_size=batch_size,
                seed=seed,
            ),
            seed=seed,
            batch_size=batch_size,
            num_waypoints=targets.shape[1],
            path_shape="approach_grasp_twist_retract",
            start_state_bin="pre_action",
            start_qpos=start_qpos,
            target_waypoints=targets,
            reference_qpos=references,
            robot_id=suite.robot.id,
            skill_id=self.skill_id,
            object_id=articulation_id,
            task_difficulty=difficulty(config),
            primary_success="physical_skill_success",
            full_start_qpos=full_start_qpos,
            case_parameters={
                "sample_count": int(config.get("sample_count", 140)),
                "twist_target_pose": target_pose.detach().cpu().tolist(),
                "grasp_position": grasp_position,
                "axis_origin": axis_origin,
                "twist_axis": twist_axis,
                "grasp_pose": grasp_pose.detach().cpu().tolist(),
                "hand_interp_steps": options.hand_interp_steps,
                "twist_waypoint_count": options.twist_waypoint_count,
                "pre_grasp_distance_m": options.pre_grasp_distance,
                "twist_angle_rad": options.twist_angle,
                "articulation_initial_pose": articulation_initial_pose.detach()
                .cpu()
                .tolist(),
                "articulation_translation_offsets_m": (
                    articulation_initial_pose[:, :3, 3]
                    - handle.initial_pose[:, :3, 3].to(
                        device=articulation_initial_pose.device,
                        dtype=articulation_initial_pose.dtype,
                    )
                )
                .detach()
                .cpu()
                .tolist(),
                "articulation_initial_qpos": articulation_initial_qpos.detach()
                .cpu()
                .tolist(),
                "target_link": target_link,
                "target_joint": target_joint,
                "target_entity_id": f"{handle.entity.uid}:{target_link}",
                "effect_joint_sign": float(config.get("effect_joint_sign", 1.0)),
                "waypoint_rotation_symmetry": "half_turn_about_z",
                "randomization": randomization_parameters(config, seed=seed),
                "difficulty_factors": dict(config.get("difficulty_factors", {})),
            },
        )

    def build_invocation(
        self,
        scenario: "AtomicSkillScenario",
        case: BenchmarkCase,
        adapter: "PlannerAdapter",
    ) -> ActionInvocation:
        del adapter
        if scenario.end_effector_part is None:
            raise RuntimeError("Twist requires a configured end-effector control part.")
        semantics = ObjectSemantics(
            affordance=TwistAffordance(
                grasp_position=tuple(case.case_parameters["grasp_position"]),
                axis_origin=tuple(case.case_parameters["axis_origin"]),
                twist_axis=torch.tensor(
                    case.case_parameters["twist_axis"],
                    dtype=torch.float32,
                    device=scenario.robot.device,
                ),
            ),
            geometry={},
            entity_id=str(case.case_parameters["target_entity_id"]),
            label="microwave_power_knob",
        )

        return scenario.require_engine().make_invocation(
            self.skill_id,
            TwistGoal(
                semantics,
                case_pose(
                    case,
                    "twist_target_pose",
                    device=scenario.robot.device,
                ),
            ),
            control_parts={
                "primary": {
                    "motion": scenario.control_part,
                    "grasp": scenario.end_effector_part,
                }
            },
            motion_policy=benchmark_motion_policy(
                self.skill_id, int(case.case_parameters["sample_count"])
            ),
            skill_options=TwistOptions(
                hand_interp_steps=int(case.case_parameters["hand_interp_steps"]),
                twist_waypoint_count=int(case.case_parameters["twist_waypoint_count"]),
                pre_grasp_distance=float(case.case_parameters["pre_grasp_distance_m"]),
                twist_angle=float(case.case_parameters["twist_angle_rad"]),
            ),
        )

    def articulation_effect_segment(self, case: BenchmarkCase) -> str:
        """Start knob-displacement diagnostics at the rotational segment."""
        del case
        return "twist"

    def replay_observation_windows(
        self, compiled: CompiledTrajectory, case: BenchmarkCase
    ) -> tuple[ReplayObservationWindow, ...]:
        """Collect executed TCP/joint traces and the release boundary."""
        del case
        actuation = compiled.segment(0, "twist")
        release = compiled.segment(0, "open")
        return (
            ReplayObservationWindow(
                "twist_actuation",
                actuation.start,
                actuation.stop,
                frozenset({"tcp_pose", "articulation_joint"}),
            ),
            ReplayObservationWindow(
                "twist_release",
                release.start,
                release.stop,
                frozenset({"gripper_qpos"}),
            ),
        )

    def physical_evaluator(self) -> TwistEvaluator:
        """Return the v1 twist-stage evaluator."""
        return TwistEvaluator()


register_atomic_skill_provider("press", PressCases)


register_atomic_skill_provider("slide", SlideCases)


register_atomic_skill_provider("twist", TwistCases)
