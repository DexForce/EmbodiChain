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

"""Prepare an entire native cohort and register measured initial conditions."""

from __future__ import annotations
from dataclasses import asdict
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time
import torch
from embodichain.lab.expansion.contracts import EpisodeCandidate
from .contracts import (
    SceneCase,
    ValidationCheck,
    ValidationResult,
)

from embodichain.lab.sim.scene import SceneView, capture_state
from embodichain.lab.sim.scene.physics import (
    capture_physics_properties,
    apply_rigid_physics,
)
from .engine import PreparedEpisodeWorkItem
from .journal import RecipeAttempt
from .session import ExpansionSession

from ._gym_utils import _hash
from ._gym_context import GymHostContext

__all__: list[str] = []


class GymCohortPreparer:
    """Prepare an entire native cohort and register measured initial conditions."""

    def __init__(self, context: GymHostContext) -> None:
        self.context = context

    def prepare_batch(
        self, requests: tuple[RecipeAttempt, ...], session: ExpansionSession
    ) -> tuple[PreparedEpisodeWorkItem, ...]:
        """Prepare one complete World cohort before the scheduler admits its rows."""
        ctx = self.context
        cfg, report, root = ctx.cfg, ctx.report, ctx.root
        environment_cfg = ctx.environment_cfg
        env = ctx.env
        view = ctx.view
        managed = ctx.managed
        baseline = ctx.baseline
        auxiliary_baseline = ctx.auxiliary_baseline
        collision = ctx.collision
        scene_source = ctx.scene_source
        names = ctx.names
        limits = ctx.limits
        visual_registry = ctx.visual_registry
        appearance = ctx.appearance
        timings = ctx.timings

        phase_started = time.monotonic()
        for request in requests:
            request.recipe.validate_host(ctx.supported_operations, ctx.compatibility)
            if (
                request.recipe.recipe_id
                != ctx.recipe(request.recipe.context.ordinal).recipe_id
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
        ctx.appearance = appearance

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
        ctx.appearance = appearance
        ctx.initial, ctx.initial_obs, ctx.physics_before = (
            initial,
            initial_obs,
            physics_before,
        )
        ctx.requests = dict(enumerate(requests))
        ctx.prepared = {}
        ctx.preparation_checks = {}
        ctx.program = env.compile_task_program(environment_cfg.task_program)
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
                ctx.program.program_id,
                env.robot.uid,
            )
            session.register_case(case, limits, joint_names=names)
            identity, _ = session.propose(
                case.scene_case_id,
                initial_id,
                source_id=ctx.program.program_id,
                source_revision=report["recipe_revision"],
                template_id=recipe.recipe_id,
                operator_id="episode",
                attempt_id=request.attempt_id,
            )
            episode_id, commit_id = session.episode_ids(identity)
            record = ctx.journal.update(
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
            ctx.prepared[row] = (
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
            ctx.preparation_checks[row] = ValidationResult(checks)
            if not valid:
                ctx.journal.update(request, status="initial_rejected")
            items.append(
                PreparedEpisodeWorkItem(
                    EpisodeCandidate(identity, json.dumps(recipe.to_dict())),
                    ctx.compatibility.key,
                    row,
                )
            )
        timings["prepare"] += time.monotonic() - phase_started
        return tuple(items)
