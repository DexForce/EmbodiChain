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

"""Physics-backed Atomic Skill scenario orchestration."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from inspect import signature
from pathlib import Path
from typing import Literal, TYPE_CHECKING

import torch

from embodichain.lab.sim.atomic_actions import (
    ActionInvocation,
    Affordance,
    AtomicActionEngine,
    ControlPartCommandProfile,
    GraspGoal,
    MotionPolicy,
    ObjectSemantics,
    PickUpOptions,
    TaskState,
    create_simulation_atomic_action_engine,
    sample_initial_articulation_geometry,
)
from embodichain.lab.sim.atomic_actions.plans import CompiledTrajectory
from embodichain.lab.sim.motion.planners.utils import PlanResult
from embodichain.toolkits.graspkit import GraspPoseGenerator
from embodichain.utils.math import pose_inv

from ..atomic_skill import AtomicSkillCaseProvider, create_atomic_skill_provider
from ..contracts import validate_physical_evaluations
from ..config import (
    SuiteCfg,
    TrackCfg,
    resolve_atomic_batch_sizes,
    resolve_atomic_pose_randomization,
)
from ..metrics.trajectory import compute_case_outcomes
from ..models import BenchmarkCase, CaseOutcome, with_physical_evaluation
from ..registry import register_scenario_provider
from ..video import VideoRecordCfg, build_video_path, record_with_window
from .atomic_objects import (
    AtomicArticulationHandle,
    AtomicObjectHandle,
    create_atomic_articulation,
    create_atomic_object,
)
from .base import ScenarioEvaluation, ScenarioProvider
from .atomic_replay import AtomicReplayRuntime
from .atomic_grasps import (
    _AntipodalGraspSearchError,
    ActiveCaseGraspPoseGenerator,
    CaseSampleCountGraspPoseGenerator,
    FrozenArticulationGraspPoseGenerator,
    grasp_sample_count,
    parallel_jaw_sampler,
)
from ._atomic_pose_sampling import _pose_settings, randomized_scene_pose
from ._atomic_reference import _recover_reference_ik

from .atomic_case_utils import (
    canonical_case_qpos,
    case_generation_seed,
    case_pose,
    float_vector,
    seeded_jitter,
)
from .atomic_policy import (
    benchmark_motion_attribution,
    benchmark_motion_policy_strategy,
    policy_motion_budget,
    with_motion_budget,
    with_plan_options,
)
from .atomic_evaluation import executed_final_pose_errors, motion_valid_mask

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.objects import Robot
    from ..planners.base import PlannerAdapter

__all__ = ["AtomicSkillScenario"]


class AtomicSkillScenario(ScenarioProvider):
    """Run fixed Atomic Actions through an adapter-owned MotionGenerator."""

    required_capabilities = frozenset(
        {"eef_waypoint", "joint_waypoint", "atomic_action"}
    )

    def __init__(self) -> None:
        self.simulation: SimulationManager | None = None
        self.robot: Robot | None = None
        self.suite: SuiteCfg | None = None
        self.track: TrackCfg | None = None
        self.control_part = "arm"
        self.end_effector_part: str | None = None
        self._objects: dict[str, AtomicObjectHandle] = {}
        self._articulations: dict[str, AtomicArticulationHandle] = {}
        self._articulation_geometry: dict[
            tuple[str, str, int, int, int], dict[str, torch.Tensor]
        ] = {}
        self._case_providers: dict[str, AtomicSkillCaseProvider] = {}
        self._cases_by_id: dict[str, BenchmarkCase] = {}
        self._base_full_qpos: torch.Tensor | None = None
        self._gripper_open: torch.Tensor | None = None
        self._gripper_grasp: torch.Tensor | None = None
        self._gripper_states: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._fixtures: dict[str, AtomicObjectHandle] = {}
        self._engine: AtomicActionEngine | None = None
        self._case_grasp_generators: dict[str, ActiveCaseGraspPoseGenerator] = {}
        self._case_grasp_sampler: CaseSampleCountGraspPoseGenerator | None = None
        self._reference_motion_generator: object | None = None
        self._runtime_entities_created = False

    def batch_sizes(self, suite: SuiteCfg, track: TrackCfg) -> list[int]:
        """Return the explicitly configured physical-execution batch sizes."""
        del suite
        return resolve_atomic_batch_sizes(track)

    def trial_metadata(self, case: BenchmarkCase) -> dict[str, object]:
        """Expose planner attribution before planning or evaluation can fail."""
        attribution = benchmark_motion_attribution(case.skill_id)
        values = {"timing_scope": "atomic_action_compile", **attribution}
        if self.track is not None:
            physics = dict(self.track.config.get("physics", {}))
            values["replay_clock"] = physics.get("clock", "fixed_samples")
            if self.simulation is not None:
                values["replay_control_period_s"] = float(
                    self.simulation.sim_config.physics_dt
                ) * int(physics.get("steps_per_waypoint", 4))
        if not attribution["selected_planner_backend_used"]:
            # These skills solve their own constrained IK interpolation. Their
            # clock must not inherit an unused adapter's nominal timing flag.
            values.setdefault("timing_is_solved", True)
            if self.suite is not None:
                values["policy_motion_limits"] = {
                    "velocity_limit": None,
                    "acceleration_limit": MotionPolicy().acceleration_limit,
                    **policy_motion_budget(self.suite.protocol),
                    "per_joint_reduction": "minimum",
                    "velocity_cap_source": "effective_robot_limits",
                }
        return values

    def create_runtime_entities(
        self,
        simulation: SimulationManager,
        suite: SuiteCfg,
        track: TrackCfg,
    ) -> None:
        """Add every rigid/articulated asset before GPU physics initialization."""
        del suite
        if self._runtime_entities_created:
            raise RuntimeError("Atomic Skill runtime entities are already configured.")
        self.simulation = simulation
        articulation_values = track.config.get("articulations", [])
        if not isinstance(articulation_values, list):
            raise TypeError("atomic-skill articulations must be a list of mappings.")
        for value in articulation_values:
            if not isinstance(value, Mapping):
                raise TypeError("Every atomic-skill articulation must be a mapping.")
            handle = create_atomic_articulation(
                simulation,
                value,
                initialize=False,
            )
            if handle.object_id in self._articulations:
                raise ValueError(
                    f"Duplicate atomic articulation id {handle.object_id!r}."
                )
            self._articulations[handle.object_id] = handle

        fixture_values = track.config.get("fixtures", [])
        if not isinstance(fixture_values, list):
            raise TypeError("atomic-skill fixtures must be a list of mappings.")
        for value in fixture_values:
            if not isinstance(value, Mapping):
                raise TypeError("Every atomic-skill fixture must be a mapping.")
            # Fixtures such as tables and support slabs stay at their declared
            # pose for every case; they are never parked or reset.
            handle = create_atomic_object(
                simulation,
                {"body_type": "static", **dict(value)},
                initialize=False,
            )
            if handle.object_id in self._fixtures:
                raise ValueError(f"Duplicate atomic fixture id {handle.object_id!r}.")
            self._fixtures[handle.object_id] = handle

        object_values = track.config.get("objects", [])
        if not isinstance(object_values, list):
            raise TypeError("atomic-skill objects must be a list of mappings.")
        for value in object_values:
            if not isinstance(value, Mapping):
                raise TypeError("Every atomic-skill object must be a mapping.")
            handle = create_atomic_object(simulation, value, initialize=False)
            if (
                handle.object_id in self._objects
                or handle.object_id in self._articulations
                or handle.object_id in self._fixtures
            ):
                raise ValueError(f"Duplicate atomic object id {handle.object_id!r}.")
            self._objects[handle.object_id] = handle
        self._runtime_entities_created = True

    def configure_runtime(
        self,
        simulation: SimulationManager,
        robot: Robot,
        suite: SuiteCfg,
        track: TrackCfg,
        control_part: str,
    ) -> None:
        """Finalize the pre-created object pool and cache robot command states."""
        if not self._runtime_entities_created or self.simulation is not simulation:
            raise RuntimeError(
                "Atomic Skill entities must be created before the first physics update."
            )
        self.simulation = simulation
        self.robot = robot
        self.suite = suite
        self.track = track
        self.control_part = control_part
        self._base_full_qpos = robot.get_qpos().clone()

        for index, handle in enumerate(self._objects.values()):
            handle.park(index)
        for index, handle in enumerate(self._articulations.values()):
            handle.park(len(self._objects) + index)
        simulation.update(step=1)

        if any(
            isinstance(value, Mapping)
            and create_atomic_skill_provider(str(value.get("id", ""))).requires_gripper
            for value in track.config.get("skills", [])
        ):
            gripper_values = track.config.get("grippers")
            if gripper_values is None:
                gripper_value = track.config.get("gripper")
                if not isinstance(gripper_value, Mapping):
                    raise ValueError(
                        "Atomic gripper-action tracks must define a gripper mapping."
                    )
                gripper_values = [gripper_value]
            elif track.config.get("gripper") is not None:
                raise ValueError("Define either gripper or grippers, not both.")
            if not isinstance(gripper_values, list) or not gripper_values:
                raise ValueError("grippers must be a non-empty list of mappings.")
            for gripper_value in gripper_values:
                part, open_state, grasp_state = self._resolve_gripper(
                    robot, gripper_value
                )
                if part in self._gripper_states:
                    raise ValueError(f"Gripper {part!r} is configured twice.")
                self._gripper_states[part] = (open_state, grasp_state)
            self.end_effector_part = next(iter(self._gripper_states))
            self._gripper_open, self._gripper_grasp = self._gripper_states[
                self.end_effector_part
            ]

    @staticmethod
    def _resolve_gripper(
        robot: Robot, gripper_value: object
    ) -> tuple[str, torch.Tensor, torch.Tensor]:
        """Validate one gripper control part and its open/grasp commands."""
        if not isinstance(gripper_value, Mapping):
            raise ValueError("Every gripper entry must be a mapping.")
        end_effector_part = gripper_value.get("control_part")
        if not isinstance(end_effector_part, str) or not end_effector_part:
            raise ValueError("gripper.control_part must be a non-empty string.")
        limits = robot.get_qpos_limits(name=end_effector_part)[0].to(
            device=robot.device, dtype=torch.float32
        )
        dofs = limits.shape[0]
        open_qpos = float_vector(
            gripper_value.get("open_qpos"),
            name="gripper.open_qpos",
            length=dofs,
            default=limits[:, 0].detach().cpu().tolist(),
        )
        grasp_qpos = float_vector(
            gripper_value.get("grasp_qpos"),
            name="gripper.grasp_qpos",
            length=dofs,
        )
        open_state = torch.tensor(open_qpos, dtype=limits.dtype, device=limits.device)
        grasp_state = torch.tensor(grasp_qpos, dtype=limits.dtype, device=limits.device)
        if bool(
            (
                (open_state < limits[:, 0])
                | (open_state > limits[:, 1])
                | (grasp_state < limits[:, 0])
                | (grasp_state > limits[:, 1])
            )
            .any()
            .item()
        ):
            raise ValueError("gripper open/grasp qpos must lie within joint limits.")
        return end_effector_part, open_state, grasp_state

    def generate_cases(
        self,
        suite: SuiteCfg,
        track: TrackCfg,
        robot: Robot,
        control_part: str,
        batch_size: int,
    ) -> list[BenchmarkCase]:
        """Generate the algorithm-independent Atomic Skill case manifest."""
        if robot is not self.robot or control_part != self.control_part:
            raise RuntimeError(
                "AtomicSkillScenario must be configured before generation."
            )
        skill_values = track.config.get("skills", [])
        if not isinstance(skill_values, list) or not skill_values:
            raise ValueError("atomic-skill skills must be a non-empty list.")
        seeds = [int(value) for value in track.config.get("seeds", [11])]
        if not seeds:
            raise ValueError("atomic-skill seeds must not be empty.")

        cases: list[BenchmarkCase] = []
        for skill_index, skill_value in enumerate(skill_values):
            if not isinstance(skill_value, Mapping):
                raise TypeError("Every atomic-skill skill entry must be a mapping.")
            skill_id = skill_value.get("id")
            if not isinstance(skill_id, str) or not skill_id:
                raise ValueError("Every atomic-skill skill entry must define an id.")
            raw_cases = skill_value.get("cases", [])
            if not isinstance(raw_cases, list) or not raw_cases:
                raise ValueError(f"Atomic skill {skill_id!r} needs at least one case.")
            defaults = {
                key: value
                for key, value in skill_value.items()
                if key not in {"id", "cases"}
            }
            for case_index, raw_case in enumerate(raw_cases):
                if not isinstance(raw_case, Mapping):
                    raise TypeError("Every atomic skill case must be a mapping.")
                config = {**defaults, **dict(raw_case)}
                config = self._prepare_pose_batch_config(
                    config,
                    track,
                    batch_size=batch_size,
                )
                for seed in seeds:
                    provider = create_atomic_skill_provider(skill_id)
                    rng_devices = (
                        [robot.device]
                        if torch.device(robot.device).type == "cuda"
                        else []
                    )
                    with torch.random.fork_rng(devices=rng_devices):
                        torch.manual_seed(
                            case_generation_seed(
                                seed,
                                skill_index=skill_index,
                                case_index=case_index,
                            )
                        )
                        case = provider.generate_case(
                            self,
                            suite,
                            track,
                            config,
                            seed=seed,
                            batch_size=batch_size,
                        )
                    if case.case_id in self._case_providers:
                        raise ValueError(
                            f"Duplicate Atomic Skill case {case.case_id!r}."
                        )
                    self._case_providers[case.case_id] = provider
                    self._cases_by_id[case.case_id] = case
                    cases.append(case)
        self.restore_base_robot()
        for index, handle in enumerate(self._objects.values()):
            handle.park(index)
        for index, handle in enumerate(self._articulations.values()):
            handle.park(len(self._objects) + index)
        return cases

    def bind_cases(self, cases: list[BenchmarkCase]) -> None:
        """Restore skill providers and scene bindings without resampling inputs.

        Args:
            cases: Generated or replayed cases for this configured runtime.
        """
        self._case_providers = {
            case.case_id: self._case_providers.get(case.case_id)
            or create_atomic_skill_provider(case.skill_id)
            for case in cases
        }
        self._cases_by_id = {case.case_id: case for case in cases}
        self.restore_base_robot()
        for index, handle in enumerate(self._objects.values()):
            handle.park(index)
        for index, handle in enumerate(self._articulations.values()):
            handle.park(len(self._objects) + index)

    @staticmethod
    def _prepare_pose_batch_config(
        config: dict[str, object],
        track: TrackCfg,
        *,
        batch_size: int,
    ) -> dict[str, object]:
        """Inject shared pose-batch controls into one skill configuration.

        Skill-specific jitter keys remain authoritative.  The track-level
        translation and yaw ranges are only fallbacks, allowing a compact
        YAML block to drive object and target pose variants.  No randomization
        is applied when the track does not opt in.
        """
        control_parts = track.config.get("control_parts")
        if control_parts is not None:
            if not isinstance(control_parts, Mapping):
                raise TypeError("atomic-skill control_parts must be a mapping.")
            config["control_parts"] = {
                str(slot): dict(parts)
                for slot, parts in control_parts.items()
                if isinstance(parts, Mapping)
            }
            if len(config["control_parts"]) != len(control_parts):
                raise TypeError("atomic-skill control_parts entries must be mappings.")

        randomization = resolve_atomic_pose_randomization(track)
        if randomization.enabled:
            if batch_size != randomization.pose_batch_size:
                raise ValueError(
                    "Atomic pose-batch size must match the simulator batch: "
                    f"{randomization.pose_batch_size} != {batch_size}."
                )
            config["_pose_batch"] = randomization
        return config

    def prepare_planner(
        self, adapter: PlannerAdapter, first_case: BenchmarkCase
    ) -> None:
        """Bind one AtomicActionEngine to the adapter-owned MotionGenerator."""
        del first_case
        control_profiles = None
        grippers = self._configured_grippers()
        if any(provider.requires_gripper for provider in self._case_providers.values()):
            if not grippers:
                raise RuntimeError("Configured gripper command states are unavailable.")
            control_profiles = {
                part: ControlPartCommandProfile.joint_positions(
                    open=open_state,
                    grasp=grasp_state,
                )
                for part, (open_state, grasp_state) in grippers.items()
            }
        motion_generator = adapter.require_motion_generator()
        scene_entities = tuple(handle.entity for handle in self._objects.values())
        grasp_pose_generators: dict[str, GraspPoseGenerator] | None = None
        self._case_grasp_sampler = None
        sampling_counts = {
            grasp_sample_count(self._cases_by_id[case_id])
            for case_id, provider in self._case_providers.items()
            if provider.samples_grasps
        }
        if sampling_counts:
            self._case_grasp_sampler = CaseSampleCountGraspPoseGenerator(
                {
                    count: parallel_jaw_sampler(count)
                    for count in sorted(sampling_counts)
                },
                default_count=max(sampling_counts),
            )
            grasp_pose_generators = {
                part: self._case_grasp_sampler for part in grippers
            }
        slide_cases = [
            self._cases_by_id[case_id]
            for case_id, provider in self._case_providers.items()
            if provider.skill_id == "slide"
        ]
        if slide_cases:
            if self.end_effector_part is None:
                raise RuntimeError("Configured Slide grasp endpoint is unavailable.")
            grasp_pose_generators = {
                **(grasp_pose_generators or {}),
                self.end_effector_part: FrozenArticulationGraspPoseGenerator(
                    [
                        (
                            case_pose(
                                case,
                                "slide_target_pose",
                                device=torch.device("cpu"),
                            ),
                            case_pose(
                                case,
                                "grasp_pose",
                                device=torch.device("cpu"),
                            ),
                        )
                        for case in slide_cases
                    ]
                ),
            }
        frozen_parts = {
            part
            for case_id, provider in self._case_providers.items()
            for part in provider.frozen_grasps(self, self._cases_by_id[case_id])
        }
        self._case_grasp_generators = {
            part: ActiveCaseGraspPoseGenerator(
                None
                if grasp_pose_generators is None
                else grasp_pose_generators.get(part)
            )
            for part in sorted(frozen_parts)
        }
        if self._case_grasp_generators:
            grasp_pose_generators = {
                **(grasp_pose_generators or {}),
                **self._case_grasp_generators,
            }
        if scene_entities:
            self._engine = create_simulation_atomic_action_engine(
                motion_generator=motion_generator,
                scene_entities=scene_entities,
                control_profiles=control_profiles,
                grasp_pose_generators=grasp_pose_generators,
            )
        else:
            self._engine = AtomicActionEngine(
                motion_generator=motion_generator,
                control_profiles=control_profiles,
                grasp_pose_generators=grasp_pose_generators,
            )

    def _configured_grippers(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Return every configured gripper's open and grasp commands."""
        if self._gripper_states:
            return dict(self._gripper_states)
        if (
            self.end_effector_part is None
            or self._gripper_open is None
            or self._gripper_grasp is None
        ):
            return {}
        return {self.end_effector_part: (self._gripper_open, self._gripper_grasp)}

    def close_planner(self, adapter: PlannerAdapter) -> None:
        """Drop the engine before its adapter closes the shared generator."""
        del adapter
        self._engine = None
        self._case_grasp_generators = {}
        self._case_grasp_sampler = None

    def require_engine(self) -> AtomicActionEngine:
        """Return the prepared Atomic Action engine."""
        if self._engine is None:
            raise RuntimeError("Atomic Skill planner resources were not prepared.")
        return self._engine

    def reset_case(
        self,
        simulation: SimulationManager,
        robot: Robot,
        case: BenchmarkCase,
        control_part: str,
    ) -> None:
        """Restore full robot and object state before every planner call."""
        del control_part
        if case.full_start_qpos is None:
            raise ValueError("Atomic Skill cases require full_start_qpos.")
        for target in (False, True):
            robot.set_qpos(case.full_start_qpos, target=target)
        robot.clear_dynamics()
        active_id = case.object_id
        reset_steps = 2
        active_object_pose: torch.Tensor | None = None
        active_articulation_state: (
            tuple[
                AtomicArticulationHandle,
                torch.Tensor | None,
                torch.Tensor | None,
            ]
            | None
        ) = None
        scene_object_poses = {
            str(object_id): torch.tensor(pose, dtype=torch.float32, device=robot.device)
            for object_id, pose in dict(
                case.case_parameters.get("scene_object_poses", {})
            ).items()
        }
        for index, handle in enumerate(self._objects.values()):
            if handle.object_id in scene_object_poses and handle.object_id != active_id:
                handle.entity.set_local_pose(scene_object_poses[handle.object_id])
                handle.entity.clear_dynamics()
            elif handle.object_id == active_id:
                reset_steps = int(
                    case.case_parameters.get("pre_action_settle_steps", reset_steps)
                )
                initial_pose = case.case_parameters.get("object_initial_pose")
                if initial_pose is None:
                    handle.reset()
                else:
                    active_object_pose = torch.tensor(
                        initial_pose,
                        dtype=torch.float32,
                        device=robot.device,
                    )
                    handle.entity.set_local_pose(active_object_pose)
                    handle.entity.clear_dynamics()
            else:
                handle.park(index)
        for index, handle in enumerate(self._articulations.values()):
            if handle.object_id == active_id:
                reset_steps = int(handle.config.get("reset_settle_steps", 10))
                pose_value = case.case_parameters.get("articulation_initial_pose")
                qpos_value = case.case_parameters.get("articulation_initial_qpos")
                pose = (
                    None
                    if pose_value is None
                    else torch.tensor(
                        pose_value,
                        dtype=torch.float32,
                        device=robot.device,
                    )
                )
                qpos = (
                    None
                    if qpos_value is None
                    else torch.tensor(
                        qpos_value,
                        dtype=torch.float32,
                        device=robot.device,
                    )
                )
                handle.reset(pose=pose, qpos=qpos)
                active_articulation_state = (handle, pose, qpos)
            else:
                handle.park(len(self._objects) + index)
        simulation.update(step=reset_steps)
        if active_object_pose is not None:
            # The case manifest freezes the pose used to derive its grasp and
            # target transforms.  Re-apply it after the optional settle window
            # so every planner/replay attempt starts from exactly that pose,
            # rather than a slightly different gravity-integrated state.
            active_handle = self._objects.get(active_id or "")
            if active_handle is not None:
                active_handle.entity.set_local_pose(active_object_pose)
                active_handle.entity.clear_dynamics()
        for object_id, pose in scene_object_poses.items():
            if object_id != active_id:
                self.object_handle(object_id).entity.set_local_pose(pose)
                self.object_handle(object_id).entity.clear_dynamics()
        if active_articulation_state is not None:
            # Settling may advance a weakly driven button/knob from the exact
            # state used to derive the frozen link target.  Restore the
            # manifest state once more so planning and replay begin at Q1,
            # rather than integrating it into a different Q2 state.
            handle, pose, qpos = active_articulation_state
            handle.reset(pose=pose, qpos=qpos)
        # Physics settling also moves driven robot joints. Restore the same
        # frozen state used by case validation before compiling any backend;
        # otherwise a correct plan starts at a different state than its case.
        for target in (False, True):
            robot.set_qpos(case.full_start_qpos, target=target)
        robot.clear_dynamics()

    def plan_case(self, adapter: PlannerAdapter, case: BenchmarkCase) -> object:
        """Compile one Atomic Action with an explicitly pinned motion backend."""
        engine = self.require_engine()
        if self.simulation is None:
            raise RuntimeError("Atomic Skill simulation is not configured.")
        provider = self._case_providers[case.case_id]
        invocation = provider.build_invocation(self, case, adapter)
        policy = invocation.motion_policy
        expected_strategy = benchmark_motion_policy_strategy(case.skill_id)
        if policy.strategy != expected_strategy:
            raise RuntimeError(
                f"Atomic Skill skill {case.skill_id!r} requires strategy "
                f"{expected_strategy!r}, got {policy.strategy!r}."
            )
        if engine.motion_generator is not adapter.require_motion_generator():
            raise RuntimeError(
                "Atomic Skill engine must own the selected adapter's MotionGenerator."
            )
        task = provider.initial_task_state(self, case)
        invocations: tuple[ActionInvocation, ...] = (invocation,)
        if case.skill_id in {"move_held_object", "place"}:
            if self.end_effector_part is None:
                raise RuntimeError("Place requires a configured end-effector part.")
            handle = self.object_handle(case.object_id)
            semantics = ObjectSemantics(
                affordance=Affordance(),
                geometry={},
                properties={"benchmark_object_id": handle.object_id},
                label=handle.object_id,
                entity_id=handle.entity.uid,
            )
            pick_invocation = engine.make_invocation(
                "pick_up",
                GraspGoal(
                    semantics=semantics,
                    grasp_xpos=case_pose(case, "grasp_pose", device=self.robot.device),
                    object_pose=case_pose(
                        case,
                        "object_initial_pose",
                        device=self.robot.device,
                    ),
                ),
                control_parts={
                    "primary": {
                        "motion": self.control_part,
                        "grasp": self.end_effector_part,
                    }
                },
                motion_policy=MotionPolicy(
                    strategy="motion_gen",
                    sample_count=int(case.case_parameters["pick_sample_count"]),
                ),
                skill_options=PickUpOptions(
                    approach_direction=torch.tensor(
                        case.case_parameters.get("approach_direction"),
                        dtype=torch.float32,
                        device=self.robot.device,
                    ),
                    pre_grasp_distance=float(
                        case.case_parameters["pre_grasp_distance_m"]
                    ),
                    lift_height=float(case.case_parameters["lift_height_m"]),
                    hand_interp_steps=int(case.case_parameters["hand_interp_steps"]),
                ),
            )
            invocations = (pick_invocation, invocation)
            task = TaskState(batch_size=case.batch_size, device=self.robot.device)
        else:
            prefix = provider.prefix_invocations(self, case)
            if prefix:
                invocations = (*prefix, invocation)
        if self.suite is not None:
            invocations = tuple(
                with_motion_budget(item, self.suite.protocol) for item in invocations
            )
        physics = (
            {} if self.track is None else dict(self.track.config.get("physics", {}))
        )
        control_dt = float(self.simulation.sim_config.physics_dt)
        if physics.get("clock", "fixed_samples") == "compiled":
            control_dt *= int(physics.get("steps_per_waypoint", 4))
        options_hook = adapter.atomic_plan_options
        try:
            signature(options_hook).bind(control_dt=control_dt)
        except TypeError:
            plan_opts = options_hook()
        else:
            plan_opts = options_hook(control_dt=control_dt)
        if plan_opts is not None:
            invocations = tuple(
                with_plan_options(item, plan_opts) for item in invocations
            )
        context = engine.initial_context(
            task=task,
            scene=provider.initial_scene(self, case),
            control_dt=control_dt,
        )
        frozen = provider.frozen_grasps(self, case)
        missing = sorted(set(frozen) - set(self._case_grasp_generators))
        if missing:
            raise RuntimeError(f"No prepared frozen-grasp generator for {missing}.")
        for part, (target_pose, grasp_pose) in frozen.items():
            self._case_grasp_generators[part].activate(target_pose, grasp_pose)
        sampler = self._case_grasp_sampler if provider.samples_grasps else None
        if sampler is not None:
            sampler.activate(grasp_sample_count(case))
        try:
            return engine.compile(invocations, context=context)
        finally:
            for part in frozen:
                self._case_grasp_generators[part].deactivate()
            if sampler is not None:
                sampler.deactivate()

    def plan_contract_error(self, result: object) -> str | None:
        """Accept compiled Atomic Action trajectories instead of raw plans."""
        if isinstance(result, CompiledTrajectory):
            return None
        return f"Expected CompiledTrajectory, got {type(result).__name__}."

    def failure_outcomes(
        self, case: BenchmarkCase, failure_code: str
    ) -> tuple[CaseOutcome, ...]:
        """Mark execution/task stages false after runner-level failures."""
        return tuple(
            replace(
                outcome,
                execution_success=False,
                task_success=False,
                replan_count=0,
            )
            for outcome in super().failure_outcomes(case, failure_code)
        )

    def evaluate_case(
        self,
        result: object,
        case: BenchmarkCase,
        robot: Robot,
        control_part: str,
        suite: SuiteCfg,
        *,
        planning_time_ms: float,
        timing_is_solved: bool = True,
    ) -> ScenarioEvaluation:
        """Validate motion, replay physics, and evaluate physical task success."""
        if not isinstance(result, CompiledTrajectory):
            raise TypeError(self.plan_contract_error(result))
        timing_is_solved = bool(
            self.trial_metadata(case).get("timing_is_solved", timing_is_solved)
        )
        arm_joint_ids = list(robot.get_joint_ids(name=control_part))
        arm_positions = result.trajectory.positions[:, :, arm_joint_ids]
        motion_outcomes = compute_case_outcomes(
            PlanResult(
                success=result.plan_success,
                positions=arm_positions,
                dt=result.trajectory.dt,
            ),
            case,
            robot,
            control_part,
            validation_samples=suite.protocol.validation_samples,
            position_threshold_m=suite.protocol.position_threshold_m,
            rotation_threshold_rad=suite.protocol.rotation_threshold_rad,
            joint_limit_tolerance_rad=suite.protocol.joint_limit_tolerance_rad,
            start_state_tolerance_rad=suite.protocol.start_state_tolerance_rad,
            joint_velocity_limit_rad_s=suite.protocol.joint_velocity_limit_rad_s,
            joint_acceleration_limit_rad_s2=(
                suite.protocol.joint_acceleration_limit_rad_s2
            ),
            joint_jerk_limit_rad_s3=suite.protocol.joint_jerk_limit_rad_s3,
            dynamic_limit_tolerance=suite.protocol.dynamic_limit_tolerance,
            timing_is_solved=timing_is_solved,
        )
        motion_outcomes = self._with_other_arm_validity(
            result,
            case,
            robot,
            suite,
            motion_outcomes,
            timing_is_solved=timing_is_solved,
        )
        provider = self._case_providers[case.case_id]
        observation = self._replay_runtime().execute(result, case, provider)
        physical_results = ()
        if observation is None:
            execution_success = torch.zeros(
                case.batch_size, dtype=torch.bool, device=robot.device
            )
            task_success = execution_success.clone()
            execution_time_ms = None
            tracking = torch.full((case.batch_size,), torch.nan, device=robot.device)
            object_lift = None
            articulation_initial = None
            articulation_final = None
            articulation_delta = None
            articulation_peak = None
            articulation_effect_end = None
            object_rotation_peak = None
            task_failure_code = "task_goal_miss"
        else:
            execution_success = observation.execution_success
            evaluator = provider.physical_evaluator()
            if evaluator is None:
                task_success, task_failure_code = provider.task_result(
                    self, case, result, observation, motion_outcomes
                )
            else:
                physical_results = validate_physical_evaluations(
                    evaluator.evaluate(case, observation, motion_outcomes),
                    batch_size=case.batch_size,
                )
                task_success = (
                    torch.tensor(
                        [row.task_success for row in physical_results],
                        dtype=torch.bool,
                        device=execution_success.device,
                    )
                    & execution_success
                    & motion_valid_mask(
                        motion_outcomes,
                        device=execution_success.device,
                    )
                )
                task_failure_code = "physical_stages_incomplete"
            execution_time_ms = observation.execution_time_ms
            tracking = observation.joint_tracking_rmse_rad
            object_lift = observation.object_lift_delta_m
            articulation_initial = observation.articulation_joint_initial
            articulation_final = observation.articulation_joint_final
            articulation_delta = (
                None
                if articulation_initial is None or articulation_final is None
                else articulation_final - articulation_initial
            )
            articulation_peak = observation.articulation_joint_peak_signed_delta
            articulation_effect_end = observation.articulation_joint_effect_end
            object_rotation_peak = observation.object_rotation_peak_signed_rad

        executed_translation_mm: torch.Tensor | None = None
        executed_rotation_deg: torch.Tensor | None = None
        if observation is not None:
            executed_translation, executed_rotation = executed_final_pose_errors(
                case,
                observation.final_tcp_pose,
            )
            executed_translation_mm = executed_translation * 1000.0
            executed_rotation_deg = executed_rotation * 180.0 / math.pi

        durations = result.trajectory.duration.detach().to("cpu")
        outcomes: list[CaseOutcome] = []
        for index, outcome in enumerate(motion_outcomes):
            executed = bool(execution_success[index].item())
            task_done = bool(task_success[index].item())
            if outcome.failure_code is not None:
                failure_code = outcome.failure_code
            elif not executed:
                failure_code = "controller_tracking_failure"
            elif not task_done:
                failure_code = (
                    physical_results[index].failure_code
                    if physical_results
                    else task_failure_code
                )
            else:
                failure_code = None
            tracking_value = float(tracking[index].item())
            outcomes.append(
                replace(
                    outcome,
                    execution_success=executed,
                    task_success=task_done,
                    stages=physical_results[index].stages if physical_results else (),
                    failure_stage=(
                        physical_results[index].failure_stage
                        if physical_results
                        else None
                    ),
                    task_completion_time_s=(
                        observation.task_completion_time_s
                        if observation is not None and task_done
                        else None
                    ),
                    joint_tracking_rmse_rad=(
                        tracking_value if math.isfinite(tracking_value) else None
                    ),
                    object_lift_delta_m=(
                        None
                        if object_lift is None
                        else float(object_lift[index].item())
                    ),
                    articulation_joint_initial=(
                        None
                        if articulation_initial is None
                        else float(articulation_initial[index].item())
                    ),
                    articulation_joint_final=(
                        None
                        if articulation_final is None
                        else float(articulation_final[index].item())
                    ),
                    articulation_joint_delta=(
                        None
                        if articulation_delta is None
                        else float(articulation_delta[index].item())
                    ),
                    articulation_joint_peak_signed_delta=(
                        None
                        if articulation_peak is None
                        else float(articulation_peak[index].item())
                    ),
                    articulation_joint_effect_end=(
                        None
                        if articulation_effect_end is None
                        else float(articulation_effect_end[index].item())
                    ),
                    object_rotation_peak_signed_rad=(
                        None
                        if object_rotation_peak is None
                        else float(object_rotation_peak[index].item())
                    ),
                    replan_count=0,
                    failure_code=failure_code,
                    executed_final_translation_err_mm=(
                        None
                        if executed_translation_mm is None
                        else float(executed_translation_mm[index].item())
                    ),
                    executed_final_rotation_err_deg=(
                        None
                        if executed_rotation_deg is None
                        else float(executed_rotation_deg[index].item())
                    ),
                )
            )
        if physical_results:
            outcomes = [
                with_physical_evaluation(outcome, physical, skill_id=case.skill_id)
                for outcome, physical in zip(outcomes, physical_results)
            ]
        trajectory_duration = (
            float(durations.mean().item()) if durations.numel() else None
        )
        return ScenarioEvaluation(
            outcomes=tuple(outcomes),
            execution_time_ms=execution_time_ms,
            end_to_end_time_ms=(
                planning_time_ms + execution_time_ms
                if execution_time_ms is not None
                else None
            ),
            trajectory_duration_s=trajectory_duration,
            trajectory_waypoints=result.trajectory.waypoint_count,
            metadata={
                "timing_scope": "atomic_action_compile",
                **benchmark_motion_attribution(case.skill_id),
                "physics_validation": "common_joint_target_replay",
                "constraint_information": (
                    "exact Cartesian samples through sequential IK; selected "
                    "planner backend bypassed"
                    if benchmark_motion_policy_strategy(case.skill_id) == "ik_interp"
                    else "empty external world; manipulated target excluded from "
                    "planner collision obstacles"
                ),
            },
        )

    @staticmethod
    def _with_other_arm_validity(
        result: CompiledTrajectory,
        case: BenchmarkCase,
        robot: Robot,
        suite: SuiteCfg,
        motion_outcomes: tuple[CaseOutcome, ...],
        *,
        timing_is_solved: bool = True,
    ) -> tuple[CaseOutcome, ...]:
        """Also require every other declared arm to pass its ordered waypoints.

        Coordinated skills move two arms.  ``case.target_waypoints`` belongs to
        the robot's primary control part; ``arm_waypoints`` lists the frozen
        poses of each other arm.  A row is motion-valid only when every arm
        passes its own waypoints in order within the joint limits.
        """
        arm_waypoints = case.case_parameters.get("arm_waypoints")
        if not arm_waypoints:
            return motion_outcomes
        if case.full_start_qpos is None:
            raise ValueError("Multi-arm cases require full_start_qpos.")

        def all_applicable(*values: bool | None) -> bool | None:
            if False in values:
                return False
            return True if all(value is True for value in values) else None

        def any_applicable(*values: bool | None) -> bool | None:
            if True in values:
                return True
            return False if all(value is False for value in values) else None

        merged = list(motion_outcomes)
        for part, waypoints in dict(arm_waypoints).items():
            joint_ids = list(robot.get_joint_ids(name=part))
            targets = torch.tensor(waypoints, dtype=torch.float32, device=robot.device)
            part_case = replace(
                case,
                num_waypoints=targets.shape[1],
                target_waypoints=targets,
                start_qpos=case.full_start_qpos[:, joint_ids].to(robot.device),
                reference_qpos=torch.zeros(
                    case.batch_size,
                    targets.shape[1],
                    len(joint_ids),
                    device=robot.device,
                ),
                case_parameters={
                    **case.case_parameters,
                    "motion_validity": "ordered_cartesian_waypoints",
                    "waypoint_rotation_symmetry": None,
                },
            )
            part_outcomes = compute_case_outcomes(
                PlanResult(
                    success=result.plan_success,
                    positions=result.trajectory.positions[:, :, joint_ids],
                    dt=result.trajectory.dt,
                ),
                part_case,
                robot,
                part,
                validation_samples=suite.protocol.validation_samples,
                position_threshold_m=suite.protocol.position_threshold_m,
                rotation_threshold_rad=suite.protocol.rotation_threshold_rad,
                joint_limit_tolerance_rad=suite.protocol.joint_limit_tolerance_rad,
                start_state_tolerance_rad=suite.protocol.start_state_tolerance_rad,
                joint_velocity_limit_rad_s=suite.protocol.joint_velocity_limit_rad_s,
                joint_acceleration_limit_rad_s2=suite.protocol.joint_acceleration_limit_rad_s2,
                joint_jerk_limit_rad_s3=suite.protocol.joint_jerk_limit_rad_s3,
                dynamic_limit_tolerance=suite.protocol.dynamic_limit_tolerance,
                timing_is_solved=timing_is_solved,
            )
            for index, (outcome, other) in enumerate(
                zip(merged, part_outcomes, strict=True)
            ):
                dynamic_fields = {}
                for name in (
                    "max_joint_velocity_rad_s",
                    "velocity_utilization",
                    "max_joint_acceleration_rad_s2",
                    "acceleration_utilization",
                    "max_joint_jerk_rad_s3",
                    "jerk_utilization",
                ):
                    values = [
                        value
                        for value in (getattr(outcome, name), getattr(other, name))
                        if value is not None
                    ]
                    dynamic_fields[name] = max(values) if values else None
                for name in (
                    "velocity_limit_violation",
                    "acceleration_limit_violation",
                    "jerk_limit_violation",
                ):
                    dynamic_fields[name] = any_applicable(
                        getattr(outcome, name), getattr(other, name)
                    )
                statuses = {
                    outcome.dynamic_measurement_status,
                    other.dynamic_measurement_status,
                }
                dynamic_fields["dynamic_measurement_status"] = (
                    "unbounded"
                    if "unbounded" in statuses
                    else "finite" if "finite" in statuses else "unavailable"
                )
                merged[index] = replace(
                    outcome,
                    finite=outcome.finite and other.finite,
                    start_state_valid=outcome.start_state_valid
                    and other.start_state_valid,
                    start_joint_error_rad=max(
                        outcome.start_joint_error_rad or 0.0,
                        other.start_joint_error_rad or 0.0,
                    ),
                    ordered_waypoints_reached=(
                        outcome.ordered_waypoints_reached
                        and other.ordered_waypoints_reached
                    ),
                    motion_valid=outcome.motion_valid and other.motion_valid,
                    timed_motion_valid=all_applicable(
                        outcome.timed_motion_valid, other.timed_motion_valid
                    ),
                    dynamic_limits_satisfied=all_applicable(
                        outcome.dynamic_limits_satisfied, other.dynamic_limits_satisfied
                    ),
                    joint_limit_violation=(
                        outcome.joint_limit_violation or other.joint_limit_violation
                    ),
                    failure_code=outcome.failure_code or other.failure_code,
                    **dynamic_fields,
                )
        return tuple(merged)

    def _replay_runtime(self) -> AtomicReplayRuntime:
        """Bind physical replay to the scenario's current runtime state."""
        if self.simulation is None or self.robot is None or self.track is None:
            raise RuntimeError("Atomic Skill runtime is not configured.")
        return AtomicReplayRuntime(
            simulation=self.simulation,
            robot=self.robot,
            control_part=self.control_part,
            track_config=self.track.config,
            objects=self._objects,
            articulations=self._articulations,
            gripper_states=self._configured_grippers(),
        )

    def record_replay(
        self,
        result: object,
        case: BenchmarkCase,
        evaluation: ScenarioEvaluation | None,
        *,
        output_dir: Path,
        algorithm_id: str,
        video: VideoRecordCfg,
    ) -> Path | None:
        """Reset the case and record a second untimed physics replay."""
        del evaluation
        if self.simulation is None or self.robot is None:
            return None
        self.reset_case(self.simulation, self.robot, case, self.control_part)
        compiled = result if isinstance(result, CompiledTrajectory) else None
        replay_runtime = self._replay_runtime()
        replayable = compiled is not None and replay_runtime.is_recordable(compiled)
        provider = self._case_providers.get(case.case_id)
        video_path = build_video_path(
            output_dir, algorithm_id, case.skill_id, case.case_id
        )

        def _replay() -> None:
            if replayable and compiled is not None:
                replay_runtime.replay_physics(
                    compiled, case, provider, collect_metrics=False
                )
            else:
                replay_runtime.hold_static()

        return record_with_window(self.simulation, video, video_path, _replay)

    def solve_reference_qpos(
        self, start_qpos: torch.Tensor, target_waypoints: torch.Tensor
    ) -> torch.Tensor:
        """Build independent sequential-IK validity evidence for a case.

        Failed exact IK may recover a joint witness inside the suite's arrival
        tolerances. FK always checks the original request; target poses and
        accepted exact solutions remain unchanged. This does not certify a path
        or a physical task outcome.
        """
        if self.robot is None:
            raise RuntimeError("Atomic Skill runtime is not configured.")
        seed = start_qpos
        references: list[torch.Tensor] = []
        for index in range(target_waypoints.shape[1]):
            target = target_waypoints[:, index]
            success, solved = self.robot.compute_ik(
                pose=target,
                joint_seed=seed,
                name=self.control_part,
            )
            if (
                not bool(torch.as_tensor(success).all().item())
                and self.suite is not None
            ):
                success, solved = _recover_reference_ik(
                    self.robot,
                    target,
                    seed,
                    success,
                    solved,
                    control_part=self.control_part,
                    protocol=self.suite.protocol,
                )
            if not bool(torch.as_tensor(success).all().item()):
                raise RuntimeError(
                    f"Independent IK rejected atomic target waypoint {index}."
                )
            seed = canonical_case_qpos(solved)
            references.append(seed.clone())
        return torch.stack(references, dim=1)

    def resolve_antipodal_grasp(
        self,
        handle: AtomicObjectHandle,
        object_pose: torch.Tensor,
        approach_direction: torch.Tensor,
        *,
        seed: int,
        start_qpos: torch.Tensor,
        pre_grasp_distance: float,
        lift_height: float,
        n_sample: int,
        max_candidates: int,
        alignment_max_angle_deg: float,
    ) -> torch.Tensor:
        """Freeze one geometry-aware, independently reachable PGI grasp pose.

        Grasp sampling and sequential IK screening happen once while the case
        manifest is built, before any planner adapter is evaluated.  Every
        planner therefore receives the same explicit grasp pose and planning
        latency excludes grasp generation.
        """

        def pick_up_poses(grasp: torch.Tensor) -> tuple[torch.Tensor, ...]:
            pre_grasp = grasp.clone()
            pre_grasp[:, :3, 3] -= approach_direction * pre_grasp_distance
            lift = grasp.clone()
            lift[:, 2, 3] += lift_height
            return pre_grasp, grasp, lift

        return self.screen_antipodal_grasp(
            handle,
            object_pose,
            approach_direction,
            seed=seed,
            start_qpos=start_qpos,
            n_sample=n_sample,
            max_candidates=max_candidates,
            alignment_max_angle_deg=alignment_max_angle_deg,
            waypoint_poses=pick_up_poses,
        )

    def screen_antipodal_grasp(
        self,
        handle: AtomicObjectHandle,
        object_pose: torch.Tensor,
        approach_direction: torch.Tensor,
        *,
        seed: int,
        start_qpos: torch.Tensor,
        n_sample: int,
        max_candidates: int,
        alignment_max_angle_deg: float,
        waypoint_poses: Callable[[torch.Tensor], Sequence[torch.Tensor]],
        rank: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        row_rank: (
            Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor] | None
        ) = None,
        env_ids: Sequence[int] | None = None,
        filter_alignment_before_ranking: bool = False,
    ) -> torch.Tensor:
        """Freeze each object's first grasp whose skill poses are reachable.

        Candidates are sampled once with ``seed`` for the full object batch.
        Each row independently orders its own world-frame pool, filters approach
        alignment, and tries each candidate before its half-turn variant. A row
        freezes its first variant whose full waypoint chain passes sequential IK
        from its own start state. Every selected environment must find a grasp.

        Args:
            handle: Object whose mesh is sampled.
            object_pose: Frozen object pose batch.
            approach_direction: World approach axis the grasp must follow.
            seed: Sampling seed.
            start_qpos: Arm state the IK chain starts from.
            n_sample: Generator surface-sample count.
            max_candidates: Number of ranked candidates to screen.
            alignment_max_angle_deg: Maximum grasp-axis deviation from the
                approach axis.
            waypoint_poses: Skill poses that a grasp variant must reach, in
                order, given that variant with shape ``(B, 4, 4)``.
            rank: Optional ordering of each row's finite candidates from their
                poses and costs; returns candidate indices, best first.
            row_rank: Optional row-aware ordering, receiving the input row index,
                its finite candidates and costs. Mutually exclusive with rank.
            filter_alignment_before_ranking: Apply the hard TCP-Z angle before
                grasp NMS and truncation. False preserves the legacy pool.
            env_ids: Robot environments matching the input rows, or all
                environments when omitted. A single-row screen in a batched
                robot must identify that environment explicitly.

        Returns:
            The selected grasp batch with shape ``(B, 4, 4)``.

        Raises:
            RuntimeError: No candidate variant reaches every skill pose.
        """
        if self.robot is None:
            raise RuntimeError("Atomic Skill runtime is not configured.")
        if object_pose.dim() != 3 or object_pose.shape[1:] != (4, 4):
            raise ValueError("Object grasp poses must have shape (B, 4, 4).")
        if start_qpos.shape[0] != object_pose.shape[0]:
            raise ValueError("Object grasp start states and poses must share a batch.")
        if env_ids is not None and len(env_ids) != object_pose.shape[0]:
            raise ValueError("Object grasp rows must match the selected environments.")
        if rank is not None and row_rank is not None:
            raise ValueError("Specify either rank or row_rank, not both.")
        if n_sample < 1 or max_candidates < 1:
            raise ValueError(
                "Antipodal grasp sample/candidate counts must be positive."
            )
        if not 0.0 < alignment_max_angle_deg <= 90.0:
            raise ValueError("grasp_alignment_max_angle_deg must be in (0, 90].")
        if type(filter_alignment_before_ranking) is not bool:
            raise TypeError("filter_alignment_before_ranking must be boolean.")
        ranked_pools = self._ranked_antipodal_grasp_pools(
            handle,
            object_pose,
            approach_direction,
            seed=seed,
            n_sample=n_sample,
            max_candidates=max_candidates,
            alignment_max_angle_deg=(
                alignment_max_angle_deg if filter_alignment_before_ranking else None
            ),
            rank=rank,
            row_rank=row_rank,
        )
        selected, found = self._screen_antipodal_grasp_rows(
            ranked_pools,
            approach_direction,
            start_qpos,
            alignment_max_angle_deg=alignment_max_angle_deg,
            waypoint_poses=waypoint_poses,
            env_ids=env_ids,
        )
        if bool(found.all().item()):
            return selected
        failed_rows = torch.nonzero(~found, as_tuple=False).flatten().cpu().tolist()
        raise _AntipodalGraspSearchError(
            f"No independently reachable antipodal grasp remained for "
            f"{handle.object_id!r} in environment rows {failed_rows} after "
            f"screening up to {[len(pool) for pool in ranked_pools]} candidates per row."
        )

    def _ranked_antipodal_grasp_pools(
        self,
        handle: AtomicObjectHandle,
        object_pose: torch.Tensor,
        approach_direction: torch.Tensor,
        *,
        seed: int,
        n_sample: int,
        max_candidates: int,
        alignment_max_angle_deg: float | None = None,
        allow_empty: bool = False,
        rank: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        row_rank: (
            Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor] | None
        ) = None,
    ) -> list[torch.Tensor]:
        """Sample/rank one frozen pool per row for reuse across goal variants."""
        assert self.robot is not None
        from .._atomic_fixtures import (
            create_antipodal_semantics,
            create_parallel_jaw_grasp_pose_generator,
        )

        fork_devices = (
            []
            if self.robot.device.type != "cuda"
            else [self.robot.device.index or torch.cuda.current_device()]
        )
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(seed)
            semantics = create_antipodal_semantics(
                handle.entity,
                label=handle.object_id,
            )
            affordance = semantics.affordance
            generator = create_parallel_jaw_grasp_pose_generator(
                n_sample=n_sample,
                force_refresh=False,
                max_candidates=max_candidates,
                **(
                    {"approach_alignment_angle": math.radians(alignment_max_angle_deg)}
                    if alignment_max_angle_deg is not None
                    else {}
                ),
            )
            candidate_pools = generator.get_valid_grasp_poses(
                mesh_vertices=affordance.mesh_vertices,
                mesh_triangles=affordance.mesh_triangles,
                obj_poses=object_pose,
                approach_direction=approach_direction,
            )
        batch_size = object_pose.shape[0]
        if len(candidate_pools) != batch_size:
            raise RuntimeError(
                "The grasp generator must return one pool per object row."
            )
        ranked_pools = []
        for row, (candidates, costs) in enumerate(candidate_pools):
            finite_indices = torch.nonzero(
                torch.isfinite(costs), as_tuple=False
            ).flatten()
            if finite_indices.numel() == 0:
                if allow_empty:
                    ranked_pools.append(candidates[:0].to(self.robot.device))
                    continue
                raise RuntimeError(
                    f"No valid antipodal grasp candidates were found for "
                    f"{handle.object_id!r} in environment row {row}."
                )
            if row_rank is not None:
                order = row_rank(row, candidates[finite_indices], costs[finite_indices])
            elif rank is not None:
                order = rank(candidates[finite_indices], costs[finite_indices])
            else:
                order = torch.argsort(costs[finite_indices])
            ranked = finite_indices[order.to(device=finite_indices.device)]
            ranked_pools.append(
                candidates[ranked[:max_candidates]].to(
                    device=self.robot.device, dtype=torch.float32
                )
            )
        return ranked_pools

    def _screen_antipodal_grasp_rows(
        self,
        ranked_pools: Sequence[torch.Tensor],
        approach_direction: torch.Tensor,
        start_qpos: torch.Tensor,
        *,
        alignment_max_angle_deg: float,
        waypoint_poses: Callable[[torch.Tensor], Sequence[torch.Tensor]],
        env_ids: Sequence[int] | None = None,
        initial_selection: torch.Tensor | None = None,
        initial_found: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return independently accepted rows without discarding unresolved ones.

        Existing selections are owned copies and remain frozen. Unresolved rows
        keep the declared candidate/half-turn order and complete IK-chain gate.
        The strict public screen raises when any row remains unresolved.
        """
        assert self.robot is not None
        batch_size = len(ranked_pools)
        selected = (
            torch.stack([pool[0] for pool in ranked_pools])
            if initial_selection is None
            else initial_selection.clone()
        )
        found = (
            torch.zeros(batch_size, dtype=torch.bool, device=self.robot.device)
            if initial_found is None
            else initial_found.clone()
        )
        minimum_alignment = math.cos(math.radians(alignment_max_angle_deg))
        for candidate_index in range(max(len(pool) for pool in ranked_pools)):
            grasp = selected.clone()
            available = torch.tensor(
                [candidate_index < len(pool) for pool in ranked_pools],
                dtype=torch.bool,
                device=self.robot.device,
            )
            for row, pool in enumerate(ranked_pools):
                if candidate_index < len(pool):
                    grasp[row] = pool[candidate_index]
            grasp = torch.where(found[:, None, None], selected, grasp)
            mirrored = grasp.clone()
            mirrored[:, :3, :2] = -mirrored[:, :3, :2]
            for variant in (grasp, mirrored):
                variant = torch.where(found[:, None, None], selected, variant)
                alignment = torch.sum(variant[:, :3, 2] * approach_direction, dim=-1)
                eligible = available & ~found & (alignment >= minimum_alignment)
                if not bool(eligible.any().item()):
                    continue
                feasible = self._ik_chain_feasible_rows(
                    start_qpos,
                    waypoint_poses(variant),
                    eligible=eligible,
                    env_ids=env_ids,
                )
                selected[feasible] = variant[feasible]
                found |= feasible
                if bool(found.all().item()):
                    return selected, found
        return selected, found

    def _ik_chain_feasible_rows(
        self,
        start_qpos: torch.Tensor,
        poses: Sequence[torch.Tensor],
        *,
        eligible: torch.Tensor,
        env_ids: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Keep each eligible row's chain result without truncating other rows.

        Full-batch calls retain Robot's environment/pose contract. Each row uses
        its own previous solution; already rejected rows cannot become feasible
        again, and accepted rows are frozen by the caller between candidates.
        """
        assert self.robot is not None
        feasible = eligible.clone()
        seed = start_qpos
        for pose in poses:
            success, solved = self.robot.compute_ik(
                pose=pose,
                joint_seed=seed,
                name=self.control_part,
                **({"env_ids": env_ids} if env_ids is not None else {}),
            )
            feasible &= torch.as_tensor(
                success, device=feasible.device, dtype=torch.bool
            ) & torch.isfinite(solved).all(dim=-1)
            if not bool(feasible.any().item()):
                break
            seed = torch.where(feasible[:, None], canonical_case_qpos(solved), seed)
        return feasible

    def _ik_chain_feasible(
        self,
        start_qpos: torch.Tensor,
        poses: Sequence[torch.Tensor],
        *,
        env_ids: Sequence[int] | None = None,
    ) -> bool:
        """Return whether every pose is reachable by IK seeded from the last."""
        assert self.robot is not None
        seed = start_qpos
        for pose in poses:
            success, seed = self.robot.compute_ik(
                pose=pose,
                joint_seed=seed,
                name=self.control_part,
                **({"env_ids": env_ids} if env_ids is not None else {}),
            )
            if not bool(torch.as_tensor(success).all().item()):
                return False
            seed = canonical_case_qpos(seed)
        return True

    def resolve_articulation_grasp(
        self,
        *,
        mesh_vertices: torch.Tensor,
        mesh_triangles: torch.Tensor,
        target_pose: torch.Tensor,
        approach_direction: torch.Tensor,
        start_qpos: torch.Tensor,
        direction: str,
        approach_distance: float,
        translation_distance: float,
        seed: int,
        n_sample: int,
        max_candidates: int,
        alignment_max_angle_deg: float,
    ) -> torch.Tensor:
        """Freeze a geometry-aware PGI grasp that can complete a Slide path."""
        translation_sign = -1.0 if direction == "pull" else 1.0

        def slide_poses(grasp: torch.Tensor) -> list[torch.Tensor]:
            translated = grasp.clone()
            translated[:, :3, 3] += (
                approach_direction * translation_sign * translation_distance
            )
            approach = grasp.clone()
            approach[:, :3, 3] -= approach_direction * approach_distance
            poses = [approach, grasp, translated]
            if direction == "push":
                poses.append(approach)
            return poses

        return self.screen_articulation_grasp(
            mesh_vertices=mesh_vertices,
            mesh_triangles=mesh_triangles,
            target_pose=target_pose,
            approach_direction=approach_direction,
            start_qpos=start_qpos,
            seed=seed,
            n_sample=n_sample,
            max_candidates=max_candidates,
            alignment_max_angle_deg=alignment_max_angle_deg,
            waypoint_poses=slide_poses,
        )

    def screen_articulation_grasp(
        self,
        *,
        mesh_vertices: torch.Tensor,
        mesh_triangles: torch.Tensor,
        target_pose: torch.Tensor,
        approach_direction: torch.Tensor,
        start_qpos: torch.Tensor,
        seed: int,
        n_sample: int,
        max_candidates: int,
        alignment_max_angle_deg: float,
        waypoint_poses: Callable[[torch.Tensor], Sequence[torch.Tensor]],
    ) -> torch.Tensor:
        """Freeze the first sampled link grasp whose skill poses are reachable.

        All rows share one target-link mesh and differ only by the articulation
        root pose, so candidates are sampled once and the selected target-local
        grasp is preserved across the batch.

        Args:
            mesh_vertices: Target-link vertices in the link frame.
            mesh_triangles: Target-link triangle indices.
            target_pose: Link pose batch with shape ``(B, 4, 4)``.
            approach_direction: World approach axes with shape ``(B, 3)``.
            start_qpos: Arm state batch the IK chain starts from.
            seed: Sampling seed.
            n_sample: Generator surface-sample count.
            max_candidates: Number of lowest-cost candidates to screen.
            alignment_max_angle_deg: Maximum grasp-axis deviation from the
                approach axis in every row.
            waypoint_poses: Skill poses that a batched grasp variant must reach,
                in order.

        Returns:
            The selected grasp batch with shape ``(B, 4, 4)``.

        Raises:
            RuntimeError: No candidate variant reaches every skill pose.
        """
        if self.robot is None:
            raise RuntimeError("Atomic Skill runtime is not configured.")
        if target_pose.dim() != 3 or target_pose.shape[1:] != (4, 4):
            raise ValueError(
                "Articulation grasp target poses must have shape (B, 4, 4)."
            )
        if approach_direction.shape != (target_pose.shape[0], 3):
            raise ValueError(
                "Articulation grasp approach directions must have shape (B, 3)."
            )
        if start_qpos.shape[0] != target_pose.shape[0]:
            raise ValueError(
                "Articulation grasp start states and target poses must share a batch."
            )
        if n_sample < 1 or max_candidates < 1:
            raise ValueError(
                "Articulation grasp sample/candidate counts must be positive."
            )
        if not 0.0 < alignment_max_angle_deg <= 90.0:
            raise ValueError("grasp_alignment_max_angle_deg must be in (0, 90].")
        from .._atomic_fixtures import (
            create_parallel_jaw_grasp_pose_generator,
        )

        fork_devices = (
            []
            if self.robot.device.type != "cuda"
            else [self.robot.device.index or torch.cuda.current_device()]
        )
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(seed)
            generator = create_parallel_jaw_grasp_pose_generator(
                n_sample=n_sample,
                force_refresh=False,
                max_candidates=max_candidates,
            )
            candidates, costs = generator.get_valid_grasp_poses(
                mesh_vertices=mesh_vertices,
                mesh_triangles=mesh_triangles,
                obj_poses=target_pose[:1],
                approach_direction=approach_direction[:1],
            )[0]
        finite_indices = torch.nonzero(torch.isfinite(costs), as_tuple=False).flatten()
        if finite_indices.numel() == 0:
            raise RuntimeError(
                "No valid antipodal articulation grasp candidates were found."
            )
        ranked = finite_indices[torch.argsort(costs[finite_indices])]
        ranked = ranked[:max_candidates].detach().to("cpu").tolist()
        minimum_alignment = math.cos(math.radians(alignment_max_angle_deg))
        for candidate_index in ranked:
            candidate = candidates[candidate_index].to(
                device=self.robot.device, dtype=torch.float32
            )
            target_to_grasp = torch.bmm(
                pose_inv(target_pose[:1]), candidate.unsqueeze(0)
            )
            grasp = torch.bmm(
                target_pose,
                target_to_grasp.expand(target_pose.shape[0], -1, -1),
            )
            mirrored = grasp.clone()
            mirrored[:, :3, 0] = -mirrored[:, :3, 0]
            mirrored[:, :3, 1] = -mirrored[:, :3, 1]
            for variant in (grasp, mirrored):
                alignment = torch.sum(variant[:, :3, 2] * approach_direction, dim=-1)
                if not bool((alignment >= minimum_alignment).all().item()):
                    continue
                if self._ik_chain_feasible(start_qpos, waypoint_poses(variant)):
                    return variant
        raise RuntimeError(
            "No independently reachable antipodal articulation grasp remained after "
            f"screening {len(ranked)} candidates."
        )

    def restore_base_robot(self) -> None:
        """Restore the robot state captured before scenario case generation."""
        if self.robot is None or self._base_full_qpos is None:
            raise RuntimeError("Atomic Skill runtime is not configured.")
        for target in (False, True):
            self.robot.set_qpos(self._base_full_qpos, target=target)
        self.robot.clear_dynamics()

    def randomize_robot_start(
        self,
        config: Mapping[str, object],
        *,
        seed: int,
        stream: int,
    ) -> torch.Tensor:
        """Apply a deterministic, bounded perturbation to the arm start state."""
        if self.robot is None:
            raise RuntimeError("Atomic Skill runtime is not configured.")
        start = self.robot.get_qpos(name=self.control_part).clone()
        amplitude = float_vector(
            config.get("start_qpos_jitter_rad"),
            name="start_qpos_jitter_rad",
            length=start.shape[-1],
            default=(0.0,) * start.shape[-1],
        )
        if any(value < 0.0 for value in amplitude):
            raise ValueError("start_qpos_jitter_rad values must be non-negative.")
        randomized = start + seeded_jitter(
            amplitude,
            seed=seed,
            stream=stream,
            dtype=start.dtype,
            device=start.device,
            batch_size=start.shape[0],
            seed_stride=int(_pose_settings(config).seed_stride),
        )
        limits = self.robot.get_qpos_limits(name=self.control_part)[0]
        margin = float(config.get("start_joint_limit_margin_rad", 0.05))
        if not math.isfinite(margin) or margin < 0.0:
            raise ValueError("start_joint_limit_margin_rad must be non-negative.")
        lower = limits[:, 0] + margin
        upper = limits[:, 1] - margin
        if bool(((randomized < lower) | (randomized > upper)).any().item()):
            raise RuntimeError(
                "Randomized arm start violates the configured joint-limit margin."
            )
        self.set_robot_start(
            randomized,
            open_gripper=False,
            grasp_gripper=False,
        )
        return randomized

    def randomize_articulation_pose(
        self,
        handle: AtomicArticulationHandle,
        config: Mapping[str, object],
        *,
        seed: int,
        stream: int,
    ) -> torch.Tensor:
        """Translate and yaw an articulation root in every environment.

        The articulation joint state stays frozen.  Skill-local position and
        yaw ranges override the pose-batch track's object-pose ranges.
        """
        pose = randomized_scene_pose(
            handle.initial_pose, config, kind="articulation", seed=seed, stream=stream
        )
        handle.reset(pose=pose, qpos=handle.initial_qpos)

        update_steps = handle.config.get("reset_settle_steps", 10)
        if type(update_steps) is not int or update_steps < 0:
            raise ValueError(
                "articulation reset_settle_steps must be a non-negative integer."
            )
        if self.simulation is not None and update_steps:
            self.simulation.update(step=update_steps)
        handle.entity.clear_dynamics()
        return handle.entity.get_local_pose(to_matrix=True).clone()

    def randomize_object_pose(
        self,
        handle: AtomicObjectHandle,
        config: Mapping[str, object],
        *,
        seed: int,
        stream: int,
        settle_steps: int | None = None,
    ) -> torch.Tensor:
        """Install one deterministic position/yaw perturbation for an object.

        When ``settle_steps`` is supplied, physics is advanced for exactly
        that many frames and the live post-settle pose is returned.  This is
        used by PickUp case generation so its frozen grasp target matches a
        cube that has actually fallen onto the table.  Omitting the argument
        preserves the legacy two-frame write/restore behavior used by the
        other case providers.
        """
        if settle_steps is not None and (
            type(settle_steps) is not int or settle_steps < 0
        ):
            raise ValueError("settle_steps must be a non-negative integer or None.")
        pose = randomized_scene_pose(
            handle.initial_pose, config, kind="object", seed=seed, stream=stream
        )

        handle.entity.set_local_pose(pose)
        handle.entity.clear_dynamics()
        if self.simulation is not None:
            if settle_steps is None:
                self.simulation.update(step=2)
                handle.entity.set_local_pose(pose)
                handle.entity.clear_dynamics()
            elif settle_steps:
                self.simulation.update(step=settle_steps)
                pose = handle.entity.get_local_pose(to_matrix=True).clone()
                handle.entity.clear_dynamics()
        return pose

    def set_robot_start(
        self,
        manipulator_qpos: torch.Tensor,
        *,
        open_gripper: bool,
        grasp_gripper: bool = False,
    ) -> None:
        """Install a manipulator start and optional gripper command."""
        if self.robot is None:
            raise RuntimeError("Atomic Skill runtime is not configured.")
        if open_gripper and grasp_gripper:
            raise ValueError("The gripper cannot start both open and grasping.")
        for target in (False, True):
            self.robot.set_qpos(manipulator_qpos, name=self.control_part, target=target)
            if open_gripper or grasp_gripper:
                gripper_qpos = (
                    self._gripper_open if open_gripper else self._gripper_grasp
                )
                if self.end_effector_part is None or gripper_qpos is None:
                    raise RuntimeError("Requested gripper state is unavailable.")
                command = gripper_qpos.unsqueeze(0).expand(
                    manipulator_qpos.shape[0], -1
                )
                self.robot.set_qpos(command, name=self.end_effector_part, target=target)
        self.robot.clear_dynamics()

    def activate_object(self, object_id: str) -> AtomicObjectHandle:
        """Reset one case object and park every other configured object."""
        handle = self.object_handle(object_id)
        for index, candidate in enumerate(self._objects.values()):
            if candidate is handle:
                candidate.reset()
            else:
                candidate.park(index)
        for index, candidate in enumerate(self._articulations.values()):
            candidate.park(len(self._objects) + index)
        if self.simulation is not None:
            self.simulation.update(step=2)
        return handle

    def activate_objects(self, object_ids: Sequence[str]) -> list[AtomicObjectHandle]:
        """Reset several case objects together and park every other one."""
        handles = [self.object_handle(object_id) for object_id in object_ids]
        active = {handle.object_id for handle in handles}
        for index, candidate in enumerate(self._objects.values()):
            if candidate.object_id in active:
                candidate.reset()
            else:
                candidate.park(index)
        for index, candidate in enumerate(self._articulations.values()):
            candidate.park(len(self._objects) + index)
        if self.simulation is not None:
            self.simulation.update(step=2)
        return handles

    def activate_articulation(self, object_id: str) -> AtomicArticulationHandle:
        """Reset one articulation and park all inactive scene objects."""
        handle = self.articulation_handle(object_id)
        for index, candidate in enumerate(self._objects.values()):
            candidate.park(index)
        for index, candidate in enumerate(self._articulations.values()):
            if candidate is handle:
                candidate.reset()
            else:
                candidate.park(len(self._objects) + index)
        if self.simulation is not None:
            self.simulation.update(
                step=int(handle.config.get("reset_settle_steps", 10))
            )
        return handle

    def object_handle(self, object_id: str | None) -> AtomicObjectHandle:
        """Resolve a configured object identifier with an actionable error."""
        if object_id is None:
            raise ValueError("This Atomic Skill case has no object id.")
        try:
            return self._objects[object_id]
        except KeyError as exc:
            raise ValueError(
                f"Unknown atomic object {object_id!r}; configured objects: "
                f"{sorted(self._objects)}."
            ) from exc

    def articulation_handle(self, object_id: str | None) -> AtomicArticulationHandle:
        """Resolve one configured articulation identifier."""
        if object_id is None:
            raise ValueError("This Atomic Skill case has no articulation id.")
        try:
            return self._articulations[object_id]
        except KeyError as exc:
            raise ValueError(
                f"Unknown atomic articulation {object_id!r}; configured "
                f"articulations: {sorted(self._articulations)}."
            ) from exc

    def sample_articulation_geometry(
        self,
        handle: AtomicArticulationHandle,
        target_link: str,
        *,
        seed: int,
        config: Mapping[str, object],
    ) -> dict[str, torch.Tensor]:
        """Resolve and cache deterministic tutorial-style link geometry."""
        articulation_points = int(config.get("articulation_point_count", 100_000))
        target_points = int(config.get("target_point_count", 5_000))
        geometry_seed = int(config.get("geometry_seed", seed))
        key = (
            handle.object_id,
            target_link,
            articulation_points,
            target_points,
            geometry_seed,
        )
        cached = self._articulation_geometry.get(key)
        if cached is None:
            import open3d as o3d

            o3d.utility.random.seed(geometry_seed % (2**31 - 1))
            cached = sample_initial_articulation_geometry(
                handle.entity,
                target_link,
                initial_qpos=handle.initial_qpos[0],
                initial_qpos_joint_names=handle.entity.joint_names,
                body_scale=handle.entity.cfg.body_scale,
                articulation_point_count=articulation_points,
                target_point_count=target_points,
            ).to_object_geometry()
            self._articulation_geometry[key] = {
                name: value.clone() for name, value in cached.items()
            }
        return {name: value.clone() for name, value in cached.items()}

    @staticmethod
    def validate_articulation_target(
        handle: AtomicArticulationHandle,
        *,
        target_link: str,
        target_joint: str,
        joint_type: Literal["prismatic", "revolute"],
    ) -> None:
        """Require the configured joint to be the link's nearest typed ancestor."""
        handle.joint_qpos(target_joint)
        resolved = next(
            (
                joint
                for joint in handle.entity.get_parent_joint_chain(target_link)
                if joint.joint_type == joint_type
            ),
            None,
        )
        if resolved is None:
            raise ValueError(
                f"Articulation link {target_link!r} has no {joint_type} ancestor."
            )
        if resolved.name != target_joint:
            raise ValueError(
                f"Articulation link {target_link!r} resolves to {joint_type} "
                f"joint {resolved.name!r}, not configured joint {target_joint!r}."
            )

    def reference_motion_generator(self) -> object:
        """Return the TOPPRA generator used to record a skill's own key poses.

        Coordinated skills compute their waypoints internally.  Providers
        compile them once with this generator while building the manifest and
        freeze the poses the skill requests; those poses do not depend on the
        planner later being benchmarked.
        """
        if self.robot is None:
            raise RuntimeError("Atomic Skill runtime is not configured.")
        if self._reference_motion_generator is None:
            from embodichain.lab.sim.motion.motion_generator import (
                MotionGenCfg,
                MotionGenerator,
            )
            from embodichain.lab.sim.motion.planners import ToppraPlannerCfg

            self._reference_motion_generator = MotionGenerator(
                MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=self.robot.uid))
            )
        return self._reference_motion_generator

    def open_grippers(self) -> None:
        """Command and set every configured gripper to its open state."""
        if self.robot is None:
            raise RuntimeError("Atomic Skill runtime is not configured.")
        batch_size = self.robot.get_qpos().shape[0]
        for part, (open_state, _) in self._configured_grippers().items():
            command = open_state.unsqueeze(0).expand(batch_size, -1)
            for target in (False, True):
                self.robot.set_qpos(command, name=part, target=target)
        self.robot.clear_dynamics()

    def close_runtime(self) -> None:
        """Release Python references before SimulationManager teardown."""
        reference = self._reference_motion_generator
        self._reference_motion_generator = None
        close = getattr(getattr(reference, "planner", None), "close", None)
        if close is not None:
            close()
        self._engine = None
        self._case_providers.clear()
        self._cases_by_id.clear()
        self._objects.clear()
        self._articulations.clear()
        self._articulation_geometry.clear()
        self._base_full_qpos = None
        self.end_effector_part = None
        self._gripper_open = None
        self._gripper_grasp = None
        self._gripper_states.clear()
        self._fixtures.clear()
        self.simulation = None
        self.robot = None
        self.suite = None
        self.track = None
        self._runtime_entities_created = False


register_scenario_provider("atomic_skill", AtomicSkillScenario)
