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

"""Native World lifecycle and Task Program evidence behind the common host port."""

from __future__ import annotations
from collections.abc import Mapping, Callable
from dataclasses import asdict
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from embodichain.data_pipeline.recording.media import MediaSpool, _atomic_json
from embodichain.lab.expansion.configured import ExpansionRunCfg
from embodichain.lab.expansion.contracts import EpisodeCandidate
from embodichain.lab.expansion.program import TaskProgramBatchExecutor
from embodichain.lab.sim.motion.expansion import (
    ExpansionSession,
    CandidateIdentity,
    SceneCase,
    ExpertEpisode,
    ValidationCheck,
    ValidationResult,
)

from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.utils.registration import REGISTERED_ENVS
from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.scene import SceneView, capture_state
from embodichain.lab.sim.scene.physics import (
    capture_physics_properties,
    capture_rigid_physics,
    apply_rigid_physics,
)
from embodichain.utils.utility import load_config
from .engine import PreparedEpisodeWorkItem, ExpansionWorkItem, ExpansionAssignment
from .recipes import (
    ExpansionRecipe,
    ExpansionContext,
    OperatorCapability,
    RecipeOperation,
    BatchCompatibility,
    UniformRangeOperator,
)
from .journal import RecipeAttempt, ExpansionJournal

__all__ = ["GymExpansionHost"]


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()
    ).hexdigest()


def _images(
    observation: object, row: int, prefix: tuple[str, ...] = ()
) -> dict[str, np.ndarray]:
    frames = {}
    if not hasattr(observation, "items"):
        return frames
    for key, value in observation.items():
        name = prefix + (
            (str(key),) if not isinstance(key, tuple) else tuple(map(str, key))
        )
        if isinstance(value, torch.Tensor):
            if value.ndim >= 3 and name[-1].split("_", 1)[0] in {
                "rgb",
                "depth",
                "color",
                "mask",
                "normal",
                "position",
                "disparity",
            }:
                frames["/".join(name)] = value[row].detach().cpu().numpy()
        else:
            frames.update(_images(value, row, name))
    return frames


def _observation_schema(space):
    children = getattr(space, "spaces", None)
    if children is not None:
        return {name: _observation_schema(value) for name, value in children.items()}
    if isinstance(space, Mapping):
        return {name: _observation_schema(value) for name, value in space.items()}
    return {
        "type": type(space).__name__,
        "shape": tuple(space.shape),
        "dtype": np.dtype(space.dtype).str,
    }


class GymExpansionHost:
    """Native batched host; scheduling, retries and commits belong to the engine.

    Args:
        cfg: Resolved deployment and collection configuration.
        report: Successful preflight report pinning recipe semantics.
        root: Job artifact directory owned by the caller.
        journal: Open journal under the caller's exclusive job lock.
        source_artifacts: Content-verified source references to retain in evidence.
    """

    def __init__(
        self,
        cfg: ExpansionRunCfg,
        report: Mapping[str, object],
        root: Path,
        journal: ExpansionJournal,
        source_artifacts: Mapping[str, object],
    ) -> None:
        self.cfg, self.report, self.root, self.journal, self.source_artifacts = (
            cfg,
            report,
            root,
            journal,
            source_artifacts,
        )
        self.capacity = cfg.batch_size
        self.env = self.collision = self.appearance = None
        try:
            self._open()
        except BaseException:
            self.close()
            raise

    def _open(self) -> None:
        cfg, report, root = self.cfg, self.report, self.root
        collision = None
        # Source artifacts and camera buffers belong to this run, never normal reset-time writers.
        path = Path(cfg.deployment)
        raw = load_config(path)
        environment_cfg = config_to_cfg(
            raw, source_path=path, task_program_path_override=cfg.program
        )
        environment_cfg.num_envs = environment_cfg.sim_cfg.num_envs = cfg.batch_size
        environment_cfg.seed = cfg.seed
        environment_cfg.dataset = None
        environment_cfg.init_rollout_buffer = False
        environment_cfg.record_trajectory = False
        environment_cfg.sim_cfg.headless = True
        if not cfg.capture_media:
            environment_cfg.enable_sensor = False
            environment_cfg.sim_cfg.render_cfg.renderer = "no-render"
        else:
            environment_cfg.enable_sensor = True
            if environment_cfg.sim_cfg.render_cfg.renderer == "no-render":
                environment_cfg.sim_cfg.render_cfg.renderer = "hybrid"
        from embodichain.lab.sim.profiler import ProfilerCfg

        environment_cfg.profiler = ProfilerCfg(
            warmup_steps=0, output_path=str(root / "profile.json")
        )
        appearance = None
        visual_registry = None
        if cfg.visual_profile_file is not None:
            from .visual import load_visual_profiles

            visual_registry = load_visual_profiles(cfg.visual_profile_file)
        started = time.monotonic()
        timings = {
            "prepare": 0.0,
            "execute": 0.0,
            "commit": 0.0,
            "render_variants": 0.0,
            "export": 0.0,
        }
        env = REGISTERED_ENVS[raw["id"]].make(cfg=environment_cfg)
        self.env = env
        timings["startup"] = time.monotonic() - started
        if getattr(
            env, "event_manager", None
        ) is not None and env.event_manager.active_functors.get("interval"):
            raise ValueError(
                "offline expansion requires fixed episode parameters; interval events are unsupported"
            )
        view = SceneView(env.sim, env_ids=tuple(range(cfg.batch_size)))
        managed = tuple(sorted(set(cfg.mass_scale) | set(cfg.friction)))
        baseline = capture_rigid_physics(view, managed) if managed else None
        auxiliary_baseline = (
            capture_physics_properties(view)
            if (
                cfg.center_of_mass
                or cfg.joint_stiffness_scale
                or cfg.joint_damping_scale
            )
            else None
        )
        if cfg.collision_profile is not None:
            from .collision import _MeasuredCollision

            collision = _MeasuredCollision(
                env, cfg.collision_profile, root / "collision"
            )
        self.collision = collision
        profile = None
        if cfg.source == "real2sim":
            from embodichain.lab.task_program.integrations.simulation.real2sim_collection import (
                load_real2sim_tabletop_profile,
            )

            profile = load_real2sim_tabletop_profile(cfg.source_profile)
        scene_source = None
        if cfg.scene_atlas is not None:
            from .scene import JointSceneSource

            scene_source = JointSceneSource(cfg.scene_atlas)
        motion_profile = None
        motion_recipes = ()
        if cfg.motion_profile is not None:
            from embodichain.lab.sim.motion.expansion.profile import (
                load_expansion_profile,
            )
            from embodichain.lab.sim.motion.expansion.combined import (
                enumerate_candidate_recipes,
            )

            motion_profile = load_expansion_profile(cfg.motion_profile)
            motion_recipes = enumerate_candidate_recipes(motion_profile)
        names = tuple(env.robot.joint_names[i] for i in env.active_joint_ids)
        limits = env.robot.get_qpos_limits()
        if limits.ndim == 3:
            limits = limits[0]
        limits = limits[list(env.active_joint_ids)].cpu()
        self.environment_cfg = environment_cfg
        self.env = env
        self.view = view
        self.managed = managed
        self.baseline = baseline
        self.auxiliary_baseline = auxiliary_baseline
        self.collision = collision
        self.profile = profile
        self.scene_source = scene_source
        self.motion_profile = motion_profile
        self.motion_recipes = motion_recipes
        self.names = names
        self.limits = limits
        self.visual_registry = visual_registry
        self.appearance = appearance
        self.timings = timings
        self.started = started

        self.compatibility = BatchCompatibility(
            robot=env.robot.uid,
            joint_names=tuple(env.robot.joint_names),
            topology_revision=report["semantic_id"],
            program_revision=_hash(environment_cfg.task_program.to_dict()),
            backend=env.sim.physics_backend,
            control_dt=env.step_dt,
            observation_revision=_hash(
                _observation_schema(env.single_observation_space)
            ),
            world_profile=_hash(
                visual_registry[cfg.visual_profile] if cfg.visual_profile else {}
            ),
            shared_resources=tuple(
                sorted(
                    (
                        visual_registry[cfg.visual_profile].get("materials", {})
                        if cfg.visual_profile
                        else {}
                    ).keys()
                )
            ),
        )
        self.supported_operations = {
            "physical_parameters": OperatorCapability(
                "physics", "prepare", "environment"
            ),
            "joint_scene": OperatorCapability(
                "scene", "prepare", "environment", sources=("task_program",)
            ),
            "visual": OperatorCapability("visual", "prepare", "world"),
            "motion": OperatorCapability("motion", "plan", "environment"),
        }

    def recipe(self, ordinal: int) -> ExpansionRecipe:
        """Sample an immutable logical recipe before assigning a physical slot."""
        cfg = self.cfg
        context = ExpansionContext(
            cfg.source, self.report["recipe_revision"], cfg.seed, ordinal
        )
        physical = {
            kind: dict(UniformRangeOperator(kind, getattr(cfg, kind)).sample(context))
            for kind in (
                "mass_scale",
                "friction",
                "joint_stiffness_scale",
                "joint_damping_scale",
            )
            if kind in {"mass_scale", "friction"} or getattr(cfg, kind)
        }
        if cfg.center_of_mass:
            physical["center_of_mass"] = {
                uid: [
                    context.random("center_of_mass", "1", f"{uid}/{axis}").uniform(
                        *bounds
                    )
                    for axis, bounds in enumerate(axes)
                ]
                for uid, axes in cfg.center_of_mass.items()
            }
        operations = [
            RecipeOperation(
                "physical_parameters",
                "1",
                self.supported_operations["physical_parameters"],
                json.dumps(physical),
            )
        ]
        if self.scene_source:
            candidate = self.scene_source.candidate(ordinal)
            operations.append(
                RecipeOperation(
                    "joint_scene",
                    "1",
                    self.supported_operations["joint_scene"],
                    json.dumps(
                        {"candidate_id": candidate.candidate_id, "ordinal": ordinal}
                    ),
                )
            )
        if cfg.visual_profile:
            operations.append(
                RecipeOperation(
                    "visual",
                    "1",
                    self.supported_operations["visual"],
                    json.dumps(self.visual_registry[cfg.visual_profile]),
                )
            )
        motion = {
            "recipe_index": (
                ordinal % len(self.motion_recipes) if self.motion_recipes else None
            ),
            "source_recipe": (
                self.profile.recipes[ordinal % len(self.profile.recipes)]
                if self.profile
                else None
            ),
        }
        operations.append(
            RecipeOperation(
                "motion", "1", self.supported_operations["motion"], json.dumps(motion)
            )
        )
        order = ("joint_scene", "physical_parameters", "visual", "motion")
        operations.sort(key=lambda operation: order.index(operation.name))
        return ExpansionRecipe(context, self.compatibility, tuple(operations))

    def preflight(self, items: tuple[ExpansionWorkItem, ...]) -> None:
        """Reject foreign prepared work and resource mismatches without stepping."""
        for item in items:
            if (
                not isinstance(item, PreparedEpisodeWorkItem)
                or item.compatibility_key != self.compatibility.key
            ):
                raise ValueError("native host requires its prepared compatible recipes")
            if (
                item.env_id not in self.prepared
                or self.prepared[item.env_id][0] != item.candidate.identity
            ):
                raise ValueError("prepared candidate does not belong to this cohort")

    def validate_plan(self, item: ExpansionWorkItem) -> ValidationResult:
        """Return preparation evidence; motion is generated lazily during execution."""
        return self.preparation_checks[item.env_id]

    def on_deferred(self, item: PreparedEpisodeWorkItem) -> None:
        """Release an unstarted row for a new attempt with the same recipe."""
        self.journal.update(
            self.prepared[item.env_id][3],
            status="interrupted",
            executed_steps_lower_bound=0,
        )

    def on_acceptance(
        self, assignment: ExpansionAssignment, episode: ExpertEpisode, accepted: bool
    ) -> None:
        """Record measured acceptance without granting committed episode credit."""
        self.journal.update(
            self.prepared[assignment.env_id][3],
            status=(
                "accepted"
                if accepted
                else (
                    "coverage_rejected" if episode.validation.accepted else "rejected"
                )
            ),
            measured_accepted=episode.validation.accepted,
        )

    def prepare_batch(
        self, requests: tuple[RecipeAttempt, ...], session: ExpansionSession
    ) -> tuple[PreparedEpisodeWorkItem, ...]:
        """Prepare one complete World cohort before the scheduler admits its rows."""
        cfg, report, root = self.cfg, self.report, self.root
        environment_cfg = self.environment_cfg
        env = self.env
        view = self.view
        managed = self.managed
        baseline = self.baseline
        auxiliary_baseline = self.auxiliary_baseline
        collision = self.collision
        scene_source = self.scene_source
        names = self.names
        limits = self.limits
        visual_registry = self.visual_registry
        appearance = self.appearance
        timings = self.timings

        phase_started = time.monotonic()
        for request in requests:
            request.recipe.validate_host(self.supported_operations, self.compatibility)
            if (
                request.recipe.recipe_id
                != self.recipe(request.recipe.context.ordinal).recipe_id
            ):
                raise ValueError(
                    "recipe differs from this host's pinned source and operator versions"
                )
        count = len(requests)
        ordinals = [request.recipe.context.ordinal for request in requests]
        physical = {
            row: json.loads(
                next(
                    op.payload_json
                    for op in request.recipe.operations
                    if op.name == "physical_parameters"
                )
            )
            for row, request in enumerate(requests)
        }
        if appearance is not None:
            appearance.close()
        appearance = ExitStack()
        self.appearance = appearance

        def prepare():
            if scene_source is not None:
                from .scene import apply_joint_scene_candidates

                apply_joint_scene_candidates(
                    SceneView(env.sim, env_ids=tuple(range(count))),
                    {
                        row: scene_source.candidate(ordinal)
                        for row, ordinal in enumerate(ordinals)
                    },
                )
            if baseline is not None:
                masses = {
                    uid: torch.tensor(
                        [
                            (
                                physical[row]["mass_scale"].get(uid, 1.0)
                                if row < count
                                else 1.0
                            )
                            for row in range(cfg.batch_size)
                        ]
                    )
                    for uid in cfg.mass_scale
                }
                friction = {
                    uid: torch.tensor(
                        [
                            (
                                physical[row]["friction"][uid]
                                if row < count
                                else baseline.bodies[managed.index(uid)][3][row]
                            )
                            for row in range(cfg.batch_size)
                        ]
                    )
                    for uid in cfg.friction
                }
                apply_rigid_physics(
                    view, baseline, mass_scale=masses, friction=friction
                )

            if auxiliary_baseline is not None:
                from .parameters import _apply_auxiliary_physics

                _apply_auxiliary_physics(
                    view,
                    auxiliary_baseline,
                    robot_uid=env.robot.uid,
                    com_offsets={
                        uid: torch.tensor(
                            [
                                (
                                    physical[row]["center_of_mass"][uid]
                                    if row < count
                                    else [0.0, 0.0, 0.0]
                                )
                                for row in range(cfg.batch_size)
                            ]
                        )
                        for uid in cfg.center_of_mass
                    },
                    stiffness_scale={
                        joint: torch.tensor(
                            [
                                (
                                    physical[row]["joint_stiffness_scale"][joint]
                                    if row < count
                                    else 1.0
                                )
                                for row in range(cfg.batch_size)
                            ]
                        )
                        for joint in cfg.joint_stiffness_scale
                    },
                    damping_scale={
                        joint: torch.tensor(
                            [
                                (
                                    physical[row]["joint_damping_scale"][joint]
                                    if row < count
                                    else 1.0
                                )
                                for row in range(cfg.batch_size)
                            ]
                        )
                        for joint in cfg.joint_damping_scale
                    },
                )

            if cfg.visual_profile is not None:
                from .visual import _visual_profile

                if (
                    hashlib.sha256(
                        Path(cfg.visual_profile_file).read_bytes()
                    ).hexdigest()
                    != report["input_files"]["visual_profile_file"]
                ):
                    raise ValueError("visual profile changed after preflight")
                appearance.enter_context(
                    _visual_profile(env, visual_registry[cfg.visual_profile])
                )

        phase_started = time.monotonic()

        def prepare_and_settle():
            prepare()
            if cfg.settling_steps:
                env.sim.update(step=cfg.settling_steps)
            env.sim.sync_render_state()

        initial_obs, _ = env.reset(
            seed=cfg.seed,
            options={"save_data": False, "episode_prepare": prepare_and_settle},
        )
        initial = capture_state(view)
        physics_before = capture_physics_properties(view)
        if collision is not None:
            collision.prepare(initial)
        self.appearance = appearance
        self.initial, self.initial_obs, self.physics_before = (
            initial,
            initial_obs,
            physics_before,
        )
        self.requests = dict(enumerate(requests))
        self.prepared = {}
        self.preparation_checks = {}
        self.program = env.compile_task_program(environment_cfg.task_program)
        items = []
        for row, request in enumerate(requests):
            recipe = request.recipe
            parameters = {
                "ordinal": recipe.context.ordinal,
                "seed": recipe.context.seed,
                "physics": physical[row],
                "source": cfg.source,
                "recipe_id": recipe.recipe_id,
            }
            for op in recipe.operations:
                if op.name == "motion":
                    parameters["motion_recipe"] = json.loads(op.payload_json)[
                        "recipe_index"
                    ]
                if op.name == "joint_scene":
                    parameters["scene_candidate_id"] = json.loads(op.payload_json)[
                        "candidate_id"
                    ]
            initial_values = [asdict(entity) for entity in initial.batches[row]]
            initial_id = _hash(initial_values)
            case = SceneCase(
                _hash(
                    (
                        recipe.compatibility.key,
                        physical[row],
                        parameters.get("scene_candidate_id"),
                        # Each recipe owns an independently prepared physical
                        # scene, even when its sampled values match another.
                        # Sharing a case would let trajectory-only coverage
                        # discard distinct requested recipes.
                        recipe.recipe_id,
                    )
                ),
                initial_id,
                report["semantic_id"],
                self.program.program_id,
                env.robot.uid,
            )
            session.register_case(case, limits, joint_names=names)
            identity, _ = session.propose(
                case.scene_case_id,
                initial_id,
                source_id=self.program.program_id,
                source_revision=report["recipe_revision"],
                template_id=recipe.recipe_id,
                operator_id="episode",
                attempt_id=request.attempt_id,
            )
            episode_id, commit_id = session.episode_ids(identity)
            record = self.journal.update(
                request,
                status="prepared",
                recipe=parameters,
                case=asdict(case),
                identity=asdict(identity),
                episode_id=episode_id,
                commit_id=commit_id,
                joint_names=names,
                initial_state=initial_values,
            )
            self.prepared[row] = (
                identity,
                episode_id,
                commit_id,
                request.directory,
                record,
            )
            valid = collision is None or collision.initial[row]
            checks = (
                ValidationCheck("preparation.source_configuration", "passed"),
                ValidationCheck(
                    "preparation.initial_state", "passed" if valid else "failed"
                ),
                ValidationCheck(
                    "preparation.settling_steps",
                    "passed",
                    metrics={"steps": cfg.settling_steps},
                ),
            )
            self.preparation_checks[row] = ValidationResult(checks)
            if not valid:
                self.journal.update(request, status="initial_rejected")
            items.append(
                PreparedEpisodeWorkItem(
                    EpisodeCandidate(identity, json.dumps(recipe.to_dict())),
                    self.compatibility.key,
                    row,
                )
            )
        timings["prepare"] += time.monotonic() - phase_started
        return tuple(items)

    def execute_batch(
        self,
        assigned: tuple[ExpansionAssignment, ...],
        *,
        on_started: Callable[[CandidateIdentity], None],
    ) -> tuple[ExpertEpisode | None, ...]:
        """Run Task Program motion bindings and return measured row evidence."""
        cfg, report, root = self.cfg, self.report, self.root
        source_artifacts = self.source_artifacts
        env = self.env
        view = self.view
        collision = self.collision
        profile = self.profile
        motion_profile = self.motion_profile
        motion_recipes = self.motion_recipes
        names = self.names
        limits = self.limits
        timings = self.timings

        initial, initial_obs, physics_before = (
            self.initial,
            self.initial_obs,
            self.physics_before,
        )
        program = self.program
        assignments = {
            assignment.env_id: self.prepared[assignment.env_id]
            for assignment in assigned
        }
        bridges, services, records, spools = {}, {}, {}, {}
        self.bridges, self.services = bridges, services
        phase_started = time.monotonic()
        from .visual import _calibration_batch

        calibrations = _calibration_batch(env) if cfg.capture_media else ()
        for row, (
            identity,
            episode_id,
            commit_id,
            directory,
            record,
        ) in assignments.items():
            ordinal = self.requests[row].recipe.context.ordinal
            if profile is None:
                pass
            else:
                from .real2sim import _Real2SimRow

                service = _Real2SimRow(
                    env,
                    row,
                    profile,
                    output=directory,
                    ordinal=ordinal,
                    attempt=self.requests[row].attempt_id,
                    initial_state_id=identity.initial_state_id,
                    recipe=profile.recipes[ordinal % len(profile.recipes)],
                )
                services[row] = service
                bridges[row] = service.bridge
            scene_fields = {
                (entity.uid, name): tuple(getattr(entity, name))
                for entity in initial.batches[row]
                for name in (
                    "pose",
                    "velocity",
                    "qpos",
                    "qvel",
                    "target_qpos",
                    "target_qvel",
                    "qf",
                )
                if getattr(entity, name)
            }
            records[row] = {
                "scene": {
                    key: [torch.tensor(value, dtype=torch.float32)]
                    for key, value in scene_fields.items()
                },
                "qpos": [env.robot.get_qpos()[row].detach().cpu().clone()],
                "qvel": [env.robot.get_qvel()[row].detach().cpu().clone()],
                "qf": [env.robot.get_qf()[row].detach().cpu().clone()],
                "actions": [],
                "objects": {
                    uid: [
                        env.sim.get_rigid_object(uid)
                        .body_state[row]
                        .detach()
                        .cpu()
                        .clone()
                    ]
                    for uid in env.sim.get_rigid_object_uid_list()
                },
            }
            if cfg.capture_media:
                from .visual import _calibrated_frames, _camera_calibration

                frames = _calibrated_frames(
                    env, initial_obs, row, calibration=calibrations[row]
                )
                if not frames:
                    raise ValueError(
                        "capture_media requires configured RGB/depth sensor observations"
                    )
                spool = MediaSpool(
                    directory / "media",
                    max_buffer_bytes=max(1, cfg.media_buffer_bytes // cfg.batch_size),
                    parent_episode_id=episode_id,
                    calibration=calibrations[row],
                )
                spool.append(0.0, frames)
                spools[row] = spool

        def observe(ids, action, observation):
            self.journal.checkpoint()
            stamp = time.monotonic()
            frame_state = capture_state(view)
            robot_states = [
                next(entity for entity in batch if entity.uid == env.robot.uid)
                for batch in frame_state.batches
            ]
            measured_qpos = torch.tensor(
                [state.qpos for state in robot_states], dtype=torch.float32
            )
            measured_qvel = torch.tensor(
                [state.qvel for state in robot_states], dtype=torch.float32
            )
            measured_qf = torch.tensor(
                [state.qf for state in robot_states], dtype=torch.float32
            )
            measured_objects = {
                uid: torch.tensor(
                    [
                        (
                            *next(e for e in batch if e.uid == uid).pose,
                            *next(e for e in batch if e.uid == uid).velocity,
                        )
                        for batch in frame_state.batches
                    ],
                    dtype=torch.float32,
                )
                for uid in env.sim.get_rigid_object_uid_list()
            }
            calibrations = _calibration_batch(env) if cfg.capture_media else ()
            timings["state_capture"] = (
                timings.get("state_capture", 0.0) + time.monotonic() - stamp
            )
            stamp = time.monotonic()
            if collision is not None:
                collision.sample(ids, action, frame_state)
            timings["sampled_validation"] = (
                timings.get("sampled_validation", 0.0) + time.monotonic() - stamp
            )
            for row in ids:
                record = records[row]
                value = (
                    {"qpos": action.value}
                    if isinstance(action.value, torch.Tensor)
                    else action.value
                )

                def active_command(name):
                    data = value[name][row]
                    if data.shape[-1] == env.robot.dof:
                        data = data[list(env.active_joint_ids)]
                    elif data.shape[-1] != len(env.active_joint_ids):
                        raise ValueError("submitted command joint layout differs")
                    return data.detach().cpu()

                command = active_command("qpos")
                if env.expert_action_spec.joint_command_mode == "position_velocity":
                    command = torch.cat(
                        (
                            command,
                            active_command("qvel"),
                        )
                    )
                entity_states = {
                    entity.uid: entity for entity in frame_state.batches[row]
                }
                for (uid, field), values in record["scene"].items():
                    values.append(
                        torch.tensor(
                            getattr(entity_states[uid], field),
                            dtype=torch.float32,
                        )
                    )
                record["actions"].append(command.clone())
                record["qpos"].append(measured_qpos[row].clone())
                record["qvel"].append(measured_qvel[row].clone())
                record["qf"].append(measured_qf[row].clone())
                for uid, values in record["objects"].items():
                    values.append(measured_objects[uid][row].clone())
                if row in spools:
                    stamp = time.monotonic()
                    spools[row].append(
                        len(record["actions"]) * env.step_dt,
                        _calibrated_frames(
                            env, observation, row, calibration=calibrations[row]
                        ),
                    )
                    timings["media_buffering_encoding"] = (
                        timings.get("media_buffering_encoding", 0.0)
                        + time.monotonic()
                        - stamp
                    )
                size = sum(
                    x.numel() * x.element_size()
                    for key in ("actions", "qpos", "qvel", "qf")
                    for x in record[key][-1:]
                )
                size += sum(
                    x.numel() * x.element_size()
                    for values in record["objects"].values()
                    for x in values[-1:]
                )
                size += sum(
                    values[-1].numel() * values[-1].element_size()
                    for values in record["scene"].values()
                )
                record["bytes"] = record.get("bytes", 0) + size
                if record["bytes"] > cfg.episode_byte_budget:
                    raise BufferError("numeric episode buffer budget exceeded")

        def mark_started(row):
            on_started(assignments[row][0])
            record = assignments[row][4]
            record["status"] = "running"
            _atomic_json(assignments[row][3] / "attempt.json", record)

        timings["source_binding"] = (
            timings.get("source_binding", 0.0) + time.monotonic() - phase_started
        )
        phase_started = time.monotonic()
        executor = TaskProgramBatchExecutor(
            env,
            program,
            row_bridges=bridges if profile is not None else None,
            expansion_profile=motion_profile,
            recipe_indices=(
                tuple(
                    (
                        (
                            self.requests[row].recipe.context.ordinal
                            % len(motion_recipes)
                        )
                        if row in self.requests
                        else 0
                    )
                    for row in range(cfg.batch_size)
                )
                if motion_recipes
                else None
            ),
        )
        result = executor.run(
            tuple(assignments),
            max_steps=cfg.max_steps,
            observe_step=observe,
            on_started=mark_started,
            reject_row=(
                (
                    lambda row: row in collision.results
                    and collision.results[row].status != "passed"
                )
                if collision is not None
                else None
            ),
        )
        timings["execute"] += time.monotonic() - phase_started
        phase_started = time.monotonic()
        physics_after = capture_physics_properties(view)
        episodes = {}
        for result_row, row in enumerate(result.env_ids):
            identity, episode_id, commit_id, directory, record = assignments[row]
            record["program_segments"] = result.segments[result_row]
            record["steps"] = result.lengths[result_row]
            record["reason"] = result.reasons[result_row]
            if not result.accepted[result_row]:
                record["status"] = "rejected"
                _atomic_json(directory / "attempt.json", record)
                episodes[row] = None
                continue
            numeric = records[row]
            positions = torch.stack(numeric["qpos"])
            observations = {
                "joint_positions": positions[:, list(env.active_joint_ids)],
                "full_joint_positions": positions,
                "joint_velocities": torch.stack(numeric["qvel"])[
                    :, list(env.active_joint_ids)
                ],
                "joint_efforts": torch.stack(numeric["qf"])[
                    :, list(env.active_joint_ids)
                ],
            }
            observations.update(
                {
                    f"rigid/{uid}/state": torch.stack(values)
                    for uid, values in numeric["objects"].items()
                }
            )
            scene_keys = []
            for index, ((uid, field), values) in enumerate(numeric["scene"].items()):
                key = f"scene_state_{index}"
                observations[key] = torch.stack(values)
                scene_keys.append({"key": key, "entity_uid": uid, "field": field})
            metadata = {
                "attempt_sequence": self.requests[row].sequence,
                "job_instance_id": self.journal.job_instance_id,
                "recipe_id": self.requests[row].recipe.recipe_id,
                "attempt_number": self.requests[row].attempt_id,
                "recipe_spec": self.requests[row].recipe.to_dict(),
                "compatibility_key": self.compatibility.key,
                "expansion_source_artifacts": source_artifacts,
                "data_type": "sim",
                "runtime_versions": report["runtime_versions"],
                "observation_origin": "simulation",
                "visual_profile": cfg.visual_profile,
                "visual_profile_revision": report["input_files"].get(
                    "visual_profile_file"
                ),
                "recipe": record["recipe"],
                "scene_case": record["case"],
                "joint_names": names,
                "source_episode_uuid": _hash(("task_program", program.program_id)),
                "expert_action_spec": env.expert_action_spec.metadata(
                    step_dt=env.step_dt
                ),
                "initial_state": record["initial_state"],
                "scene_state_fields": scene_keys,
                "physics_backend": env.sim.physics_backend,
                "program_segments": result.segments[result_row],
                "expansion": (
                    result.expansion_records[result_row]
                    if result.expansion_records
                    else ()
                ),
                "segments": result.segments[result_row],
                "instruction": program.instruction,
                "instruction_source": "task_program",
            }
            physics_unchanged = all(
                torch.allclose(
                    physics_before[key][row],
                    physics_after[key][row],
                    atol=1e-6,
                    rtol=1e-6,
                )
                for key in physics_before
            )

            def parameter_value(value):
                if isinstance(value, list):
                    return [parameter_value(item) for item in value]
                return (
                    value if math.isfinite(value) else ("+inf" if value > 0 else "-inf")
                )

            metadata["physics_parameters"] = {
                key: parameter_value(value[row].detach().cpu().tolist())
                for key, value in physics_before.items()
            }
            metadata["physics_fingerprint"] = _hash(metadata["physics_parameters"])
            checks = list(self.preparation_checks[row].checks) + [
                ValidationCheck(
                    "physics_parameters",
                    "passed" if physics_unchanged else "failed",
                    "Measured episode parameter baseline",
                ),
                ValidationCheck(
                    "task_success",
                    "passed",
                    "Canonical Task Program and segment validators completed",
                ),
            ]
            if collision is not None:
                measured = collision.terminal.get(row)
                if measured is None:
                    checks.append(ValidationCheck("task.measured", "unavailable"))
                else:
                    checks.extend(measured.checks)
                checked = collision.results.get(row)
                checks.append(
                    ValidationCheck(
                        "path_collision",
                        (
                            "passed"
                            if checked is not None and checked.status == "passed"
                            else "failed"
                        ),
                        (
                            "Measured control-boundary collision checks"
                            if checked is None
                            else checked.reason
                        ),
                    )
                )
            if row in services:
                request = services[row].qualify(
                    qpos=positions,
                    object_pose=torch.stack(numeric["objects"][profile.object_uid])[
                        :, :7
                    ],
                )
                # Source qualification keeps its legacy labels, while the host
                # owns executed action/state/schema and recipe identities.
                owned = set(metadata) - {"recipe"}
                metadata["motion_source_metadata"] = {
                    key: value
                    for key, value in request.metadata.items()
                    if key in owned
                }
                metadata.update(
                    {
                        key: value
                        for key, value in request.metadata.items()
                        if key not in owned
                    }
                )
                metadata["source_origin"] = request.metadata.get("origin")
                metadata.pop("source_episode_uuid", None)
                metadata["source_episode_uuids"] = sorted(
                    {
                        _hash(
                            (
                                "real2sim",
                                unit.unit.source.source_id,
                                unit.unit.source.source_revision,
                            )
                        )
                        for unit in services[row].prototypes.values()
                    }
                )
                checks.extend(request.validation.checks)
                checks.append(
                    ValidationCheck(
                        "path_collision",
                        "passed",
                        "Source-qualified planned and measured collision checks",
                    )
                )
            if row in spools:
                metadata["media_manifest"] = str(spools[row].finish())
                metadata["media_manifest_sha256"] = hashlib.sha256(
                    Path(metadata["media_manifest"]).read_bytes()
                ).hexdigest()
            from embodichain.lab.sim.motion.expansion.coverage import (
                describe_trajectory,
            )

            descriptor = describe_trajectory(
                observations["joint_positions"],
                torch.arange(len(positions), dtype=torch.float64) * env.step_dt,
                limits,
            )
            metadata["coverage_descriptor"] = {
                "phase_ids": descriptor.phase_ids,
                "geometry": descriptor.geometry.tolist(),
                "timing": descriptor.timing.tolist(),
                "normalization": "joint_limits_arc_length",
            }
            metadata["expansion_recipe"] = record["recipe"]
            metadata["motion_origin"] = (
                "real2sim_prototype" if profile is not None else "native_planning"
            )
            metadata["action_origin"] = "submitted_controller_command"
            metadata["state_origin"] = "measured_simulation"
            episode = ExpertEpisode(
                identity,
                observations,
                torch.stack(numeric["actions"]),
                torch.arange(len(positions), dtype=torch.float64) * env.step_dt,
                (
                    "joint_position_velocity"
                    if env.expert_action_spec.joint_command_mode == "position_velocity"
                    else "joint_position"
                ),
                ValidationResult(tuple(checks)),
                episode_id,
                commit_id,
                metadata=metadata,
            )
            episodes[row] = episode
            record["measured_accepted"] = episode.validation.accepted
            record["validation"] = [
                dict(
                    check_id=check.check_id,
                    status=check.status,
                    detail=check.detail,
                    metrics=dict(check.metrics),
                )
                for check in episode.validation.checks
            ]
            _atomic_json(directory / "attempt.json", record)
        timings["acceptance"] = (
            timings.get("acceptance", 0.0) + time.monotonic() - phase_started
        )
        self.services, self.bridges = services, bridges
        return tuple(episodes[assignment.env_id] for assignment in assigned)

    def close(self) -> None:
        """Release every owned resource even when an earlier cleanup operation fails."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        errors = []
        for name in ("appearance", "collision"):
            resource = getattr(self, name, None)
            setattr(self, name, None)
            if resource is not None:
                try:
                    resource.close()
                except BaseException as error:
                    errors.append(error)
        self.__dict__.pop("services", {}).clear()
        self.__dict__.pop("bridges", {}).clear()
        self.view = self.initial = self.initial_obs = self.physics_before = None
        self.baseline = self.auxiliary_baseline = None
        env = self.env
        self.env = None
        if env is not None:
            try:
                env._active_task_program_bridge = None
                env.close(exit_process=False)
            except BaseException as error:
                errors.append(error)
        try:
            SimulationManager.flush_cleanup_queue()
        except BaseException as error:
            errors.append(error)
        if errors:
            fatal = next(
                (error for error in errors if not isinstance(error, Exception)), None
            )
            if fatal is not None:
                raise fatal
            raise RuntimeError(
                "native host cleanup failed: " + "; ".join(map(str, errors))
            ) from errors[0]
