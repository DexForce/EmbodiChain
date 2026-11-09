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

"""Execute source bindings through the canonical Task Program batch port."""

from __future__ import annotations
from collections.abc import Callable
import time
from embodichain.data_pipeline.recording.media import _atomic_json
from embodichain.lab.expansion.program import TaskProgramBatchExecutor
from .contracts import CandidateIdentity, ExpertEpisode

from embodichain.lab.sim.scene.physics import capture_physics_properties
from .engine import ExpansionAssignment

from ._gym_context import GymHostContext
from ._gym_frames import GymFrameCollector
from ._gym_episode import GymEpisodeAssembler

__all__: list[str] = []


class GymCohortExecutor:
    """Execute source bindings through the canonical Task Program batch port."""

    def __init__(self, context: GymHostContext) -> None:
        self.context = context

    def execute_batch(
        self,
        assigned: tuple[ExpansionAssignment, ...],
        *,
        on_started: Callable[[CandidateIdentity], None],
    ) -> tuple[ExpertEpisode | None, ...]:
        """Run Task Program motion bindings and return measured row evidence."""
        ctx = self.context
        cfg, env, view, collision, timings = (
            ctx.cfg,
            ctx.env,
            ctx.view,
            ctx.collision,
            ctx.timings,
        )
        profile, motion_profile, motion_recipes, program = (
            ctx.profile,
            ctx.motion_profile,
            ctx.motion_recipes,
            ctx.program,
        )
        assignments = {
            assignment.env_id: ctx.prepared[assignment.env_id]
            for assignment in assigned
        }
        bridges, services = {}, {}
        ctx.bridges, ctx.services = bridges, services
        phase_started = time.monotonic()
        prototype_profile = None
        if cfg.prototype_validation_profile is not None:
            from embodichain.lab.task_program.integrations.simulation.real2sim_collection import (
                load_real2sim_tabletop_profile,
            )

            prototype_profile = load_real2sim_tabletop_profile(
                cfg.prototype_validation_profile
            )
        for row, (
            identity,
            episode_id,
            commit_id,
            directory,
            record,
        ) in assignments.items():
            ordinal = ctx.requests[row].recipe.context.ordinal
            if profile is None:
                from embodichain.lab.task_program.integrations.demo_skill import (
                    _ConfiguredDemoSkillSource,
                )

                if env.task_program_adapter.configured_nominal_sources:
                    from .sources import _OccurrenceSourceRow

                    service = _OccurrenceSourceRow(
                        env,
                        row,
                        program,
                        expansion_profile=motion_profile,
                        candidate_index=(
                            (ordinal % len(motion_recipes)) if motion_recipes else 0
                        ),
                        duration_scale=cfg.demo_skill_duration_scale,
                        velocity_limits=cfg.demo_skill_velocity_limits,
                        acceleration_limits=cfg.demo_skill_acceleration_limits,
                        prototype_profile=prototype_profile,
                    )
                    services[row] = service
                    bridges[row] = service.bridge
            else:
                from .real2sim import _Real2SimRow

                service = _Real2SimRow(
                    env,
                    row,
                    profile,
                    output=directory,
                    ordinal=ordinal,
                    attempt=ctx.requests[row].attempt_id,
                    initial_state_id=identity.initial_state_id,
                    recipe=profile.recipes[ordinal % len(profile.recipes)],
                )
                services[row] = service
                bridges[row] = service.bridge

        collector = GymFrameCollector(ctx, assignments)

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
            row_bridges=bridges or None,
            expansion_profile=motion_profile,
            recipe_indices=(
                tuple(
                    (
                        (ctx.requests[row].recipe.context.ordinal % len(motion_recipes))
                        if row in ctx.requests
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
            observe_step=collector.observe,
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
        physics_after = capture_physics_properties(view)
        episodes = GymEpisodeAssembler(ctx).assemble(
            assignments, collector, services, result, physics_after
        )
        ctx.services, ctx.bridges = services, bridges
        return tuple(episodes[assignment.env_id] for assignment in assigned)
