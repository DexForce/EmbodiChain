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

"""Native single-World collection for independent Task Program workloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json
import hashlib
import time
from typing import Any

from embodichain.data_pipeline.recording.media import _atomic_json
from .configured import ExpansionRunCfg
from .host_extensions import ExpansionRowContext, GymExpansionHostExtensions
from .workloads import (
    ExpansionWorkloadCfg,
    WorkloadAssignment,
    WorkloadExecutionCfg,
    WorkloadProgramExecutor,
    WorkloadRowResult,
)

__all__ = ["GymExpansionWorkload", "run_gym_expansion_workloads"]


@dataclass(frozen=True)
class GymExpansionWorkload:
    """Task-owned declarations for one independently budgeted source group.

    Args:
        workload_id: Unique logical source group, independent of its World row.
        cfg: Deployment, output, evidence limits and original collection budget.
        report: Pinned semantic/execution identities and runtime versions.
        source_artifacts: Hashed configuration/source references for each episode.
        extensions: Scene, recipe and plan callbacks; rendering and validation
            belong to the common extensions passed to the collection function.

    Deployments may select different integration/grasp policies, but must share
    exactly the same physical environment, robot, control cadence and program.
    This facade supports native Task Programs with row-local scene/plan ports;
    configured physical/visual/prototype operators use the cohort job API.
    """

    workload_id: str
    cfg: ExpansionRunCfg
    report: Mapping[str, object]
    source_artifacts: Mapping[str, object]
    extensions: GymExpansionHostExtensions = GymExpansionHostExtensions()


def _preview_snapshot(env: Any) -> dict[str, object]:
    """Freeze actual resolved geometry, including finite JSON parameter encoding."""
    from enum import Enum
    import math

    def encode(value: Any) -> Any:
        if isinstance(value, Enum):
            return encode(value.value)
        if isinstance(value, Mapping):
            return {str(key): encode(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [encode(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            if math.isnan(value):
                raise ValueError("resolved preview configuration contains NaN")
            return "+inf" if value > 0 else "-inf"
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "tolist"):
            return encode(value.tolist())
        return str(value)

    return encode(
        {
            "robot": env.robot.cfg.to_dict(),
            "rigid_object": [
                env.sim.get_rigid_object(uid).cfg.to_dict()
                for uid in env.sim.get_rigid_object_uid_list()
            ],
            "ground_plane_visible": getattr(env.cfg, "ground_plane_visible", True),
        }
    )


class _GymWorkloadSource:
    """Compose existing native admission, frame assembly and persistence owners."""

    def __init__(self, owner: Any, groups: Mapping[str, Any]) -> None:
        from ._gym_frames import GymFrameCollector

        self.owner, self.groups, self.env = owner, groups, owner.env
        self.collector = GymFrameCollector(owner._context, {})
        self.active: dict[int, dict[str, Any]] = {}
        self.activation_snapshot = None
        self.nominal = {
            uid: self.env.sim.get_rigid_object(uid)
            .get_local_pose(to_matrix=True)[0]
            .detach()
            .cpu()
            .tolist()
            for uid in self.env.sim.get_rigid_object_uid_list()
        }

    def prepare(self, assignments: tuple[WorkloadAssignment, ...]) -> None:
        from embodichain.lab.sim.scene import (
            SceneView,
            SceneChangeSet,
            apply_changes_batch,
        )
        from .scene import apply_joint_scene_candidates

        stamp = time.monotonic()
        self.activation_snapshot = None
        ids = tuple(a.env_id for a in assignments)
        view = SceneView(self.env.sim, env_ids=ids)
        # Every candidate starts from the same seeded nominal reset. Original
        # reset events may run, but cannot resample an unowned scene axis.
        apply_changes_batch(
            view,
            {row: SceneChangeSet(self.nominal, tuple(self.nominal)) for row in ids},
        )
        candidates = {}
        for assignment in assignments:
            group = self.groups[assignment.workload_id]
            if group["admitted_at"][0] is None:
                group["admitted_at"][0] = time.monotonic()
            ctx = group["host"]._context
            request = ctx.journal.allocate_recipe(ctx.recipe(assignment.ordinal))
            request.recipe.validate_host(ctx.supported_operations, ctx.compatibility)
            for operation in request.recipe.operations:
                if operation.name not in {
                    "physical_parameters",
                    "joint_scene",
                    "motion",
                }:
                    raise ValueError(
                        "rolling workloads only support row-local scene and motion operations"
                    )
                if operation.name == "physical_parameters" and any(
                    json.loads(operation.payload_json).values()
                ):
                    raise ValueError(
                        "physical parameter changes require the cohort job API"
                    )
            ctx.requests[assignment.env_id] = request
            if ctx.scene_source is not None:
                candidates[assignment.env_id] = ctx.scene_source.candidate(
                    assignment.ordinal
                )
        if candidates:
            apply_joint_scene_candidates(
                SceneView(self.env.sim, env_ids=tuple(candidates)), candidates
            )
        self.owner.timings["prepare"] += time.monotonic() - stamp

    def activate(self, assignment: WorkloadAssignment, observation: object) -> Any:
        from embodichain.lab.sim.scene import capture_state
        from embodichain.lab.sim.scene.physics import capture_physics_properties
        from .contracts import (
            EpisodeCandidate,
            SceneCase,
            ValidationCheck,
            ValidationResult,
        )
        from .engine import PreparedEpisodeWorkItem
        from ._gym_utils import _hash

        row = assignment.env_id
        group = self.groups[assignment.workload_id]
        host, engine = group["host"], group["engine"]
        ctx = host._context
        request = ctx.requests[row]
        if self.activation_snapshot is None:
            self.activation_snapshot = (
                capture_state(ctx.view),
                capture_physics_properties(ctx.view),
            )
        initial, physics = self.activation_snapshot
        ctx.initial, ctx.initial_obs = initial, observation
        ctx.physics_before = physics
        collision = ctx.collision
        if collision is not None:
            collision.prepare(initial, env_ids=(row,))
        values = [asdict(entity) for entity in initial.batches[row]]
        initial_id = _hash(values)
        physical = json.loads(
            next(
                op.payload_json
                for op in request.recipe.operations
                if op.name == "physical_parameters"
            )
        )
        parameters = {
            "ordinal": assignment.ordinal,
            "seed": ctx.cfg.seed,
            "physics": physical,
            "source": ctx.cfg.source,
            "recipe_id": request.recipe.recipe_id,
        }
        for op in request.recipe.operations:
            if op.name == "motion":
                parameters["motion_recipe"] = json.loads(op.payload_json)[
                    "recipe_index"
                ]
            elif op.name == "joint_scene":
                parameters["scene_candidate_id"] = json.loads(op.payload_json)[
                    "candidate_id"
                ]
        case = SceneCase(
            _hash(
                (
                    ctx.compatibility.key,
                    physical,
                    parameters.get("scene_candidate_id"),
                    request.recipe.recipe_id,
                )
            ),
            initial_id,
            ctx.report["semantic_id"],
            ctx.program.program_id,
            self.env.robot.uid,
        )
        session = group["session"]
        session.register_case(case, ctx.limits, joint_names=ctx.names)
        identity, _ = session.propose(
            case.scene_case_id,
            initial_id,
            source_id=ctx.program.program_id,
            source_revision=ctx.report["recipe_revision"],
            template_id=request.recipe.recipe_id,
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
            joint_names=ctx.names,
            initial_state=values,
        )
        ctx.prepared[row] = identity, episode_id, commit_id, request.directory, record
        valid = collision is None or collision.initial[row]
        ctx.preparation_checks[row] = ValidationResult(
            (
                ValidationCheck("preparation.source_configuration", "passed"),
                ValidationCheck(
                    "preparation.initial_state", "passed" if valid else "failed"
                ),
                ValidationCheck(
                    "preparation.settling_steps",
                    "passed",
                    metrics={"steps": ctx.cfg.settling_steps},
                ),
            )
        )
        item = PreparedEpisodeWorkItem(
            EpisodeCandidate(identity, json.dumps(request.recipe.to_dict())),
            ctx.compatibility.key,
            row,
        )
        native = engine.admit_row(item)
        if native is None:
            ctx.journal.update(request, status="initial_rejected")
            return None
        factory = None
        if ctx.extensions.plan_transform_factory is not None:
            factory = ctx.extensions.plan_transform_factory(
                self.env,
                ExpansionRowContext(
                    row,
                    identity,
                    assignment.ordinal,
                    request.attempt_id,
                    request.directory,
                    case,
                    request.recipe,
                ),
            )
            if factory is None:
                raise ValueError("plan_transform_factory returned no factory")
        adapter = group["adapter"].for_env_ids((row,))
        bridge = adapter.create_bridge(
            ctx.program, plan_transform_factory=factory, isolate_policy_sessions=True
        )
        # The frame owner is shared and observes all active rows once per step.
        self.owner._context.initial, self.owner._context.initial_obs = (
            initial,
            observation,
        )
        self.collector.add({row: ctx.prepared[row]})
        self.active[row] = {
            "group": group,
            "native": native,
            "adapter": adapter,
            "factory": factory,
            "physics_before": ctx.physics_before,
        }
        return bridge

    def started(self, assignment: WorkloadAssignment, world_control_step: int) -> None:
        row = assignment.env_id
        state = self.active[row]
        ctx = state["group"]["host"]._context
        state["group"]["engine"].mark_row_started(state["native"])
        ctx.prepared[row][4]["world_control_start"] = world_control_step
        ctx.journal.update(
            ctx.requests[row], status="running", world_control_start=world_control_step
        )

    def observe(
        self,
        assignments: tuple[WorkloadAssignment, ...],
        action: Any,
        observation: object,
    ) -> Sequence[int]:
        ids = tuple(a.env_id for a in assignments)
        self.activation_snapshot = None
        for name in {a.workload_id for a in assignments}:
            self.groups[name]["host"]._context.journal.checkpoint()
        self.collector.observe(ids, action, observation)
        checks = self.owner.collision
        return tuple(
            row
            for row in ids
            if checks is not None
            and row in checks.results
            and checks.results[row].status != "passed"
        )

    def complete(self, result: WorkloadRowResult) -> Any:
        from embodichain.lab.sim.scene.physics import capture_physics_properties
        from ._gym_episode import GymEpisodeAssembler

        row = result.assignment.env_id
        state = self.active.pop(row, None)
        if state is None:
            return None
        group = state["group"]
        ctx = group["host"]._context
        outcome = result.program
        factory = state["factory"]
        if factory is not None:
            records = tuple(factory.records)
            ctx.prepared[row][4]["expansion"] = [r.to_metadata() for r in records]
            portable = []
            ctx.source_artifacts = {
                key: value
                for key, value in ctx.source_artifacts.items()
                if not key.startswith(f"plan_{row}_")
            }
            for index, record in enumerate(records):
                trace = record.to_metadata()
                if "plan_artifact" in trace:
                    key = f"plan_{row}_{index}"
                    ctx.source_artifacts[key] = trace.pop("plan_artifact")
                    trace["plan_artifact_key"] = key
                portable.append(trace)
            outcome = replace(outcome, expansion_records=(tuple(portable),))
            if (
                ctx.extensions.require_plan_transform
                and outcome.accepted[0]
                and not records
            ):
                outcome = replace(
                    outcome,
                    accepted=(False,),
                    reasons=("requested_operator_not_applied",),
                )
        ctx.physics_before = state["physics_before"]
        ctx.prepared[row][4].update(
            world_control_start=result.world_control_start,
            world_control_end=result.world_control_end,
        )
        try:
            episode = GymEpisodeAssembler(ctx).assemble(
                {row: ctx.prepared[row]},
                self.collector,
                {},
                outcome,
                capture_physics_properties(ctx.view),
            )[row]
            if episode is not None:
                episode = replace(
                    episode,
                    metadata={
                        **episode.metadata,
                        "execution": {
                            **self.execution,
                            "workload_id": result.assignment.workload_id,
                            "world_control_start": result.world_control_start,
                            "world_control_end": result.world_control_end,
                        },
                    },
                )
            return group["engine"].complete_row(state["native"], episode)
        finally:
            state["adapter"].close()
            self.collector.discard(row)

    def close(self) -> None:
        for state in self.active.values():
            state["adapter"].close()
        self.active.clear()


def run_gym_expansion_workloads(
    workloads: Sequence[GymExpansionWorkload],
    execution: WorkloadExecutionCfg,
    *,
    extensions: GymExpansionHostExtensions | None = None,
    output_dir: str | Path,
) -> Mapping[str, object]:
    """Collect and export independent groups in one process and one owned World.

    The sole collector uses exactly ``workload_count * batch_size`` physical
    environments. It captures the seeded nominal scene in this same World;
    preparation never creates a reference World. Original per-group quotas,
    attempts, measured acceptance and confirmed receipt accounting are retained.
    One bounded writer thread exports frozen groups while other groups collect.
    Writers receive only artifact paths and never access the live World.

    Args:
        workloads: Nonempty unique source groups, each owning a fresh output.
        execution: W, B, planning and original step/settling budgets.
        extensions: Common collection renderer and measured validator. Per-group
            scene/recipe/plan callbacks belong to each workload's extensions.
        output_dir: Fresh root for execution/reference/progress diagnostics.

    Returns:
        Owned per-workload results and actual process/World/planning evidence.
        Partial quotas remain explicit even when qualified artifacts were saved.
    """
    from .host import GymExpansionHost
    from ._gym_recipe import GymRecipeBuilder
    from ._gym_utils import _hash
    from .engine import ExpansionEngine, ExpansionJobCfg
    from .journal import ExpansionJournal
    from .session import ExpansionSession
    from .session_cfg import ExpansionSessionCfg
    from .persistence import ArtifactEpisodeSink, load_expansion_episode
    from .workflow import _RecordedSink
    from .episode_export import export_expansion_episodes
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.gym.utils.registration import REGISTERED_ENVS
    from embodichain.utils.utility import load_config
    from embodichain.lab.gym.envs._json import json_safe_copy

    if not workloads or len({w.workload_id for w in workloads}) != len(workloads):
        raise ValueError("workloads must be nonempty with unique IDs")
    execution.__post_init__()
    if execution.workload_count > len(workloads):
        raise ValueError("workload_count exceeds available groups")
    common = extensions or GymExpansionHostExtensions()
    if (
        common.scene_source_factory
        or common.recipe_transform
        or common.plan_transform_factory
    ):
        raise ValueError(
            "common extensions may only select rendering and measured validation"
        )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    configs = []
    for workload in workloads:
        cfg = workload.cfg
        cfg.__post_init__()
        for field, digest in workload.report["input_files"].items():
            value = getattr(cfg, field, None)
            if (
                value is None
                or hashlib.sha256(Path(value).read_bytes()).hexdigest() != digest
            ):
                raise ValueError(f"workload input {field} differs from preflight")
        if any(
            getattr(cfg, field) != getattr(workloads[0].cfg, field)
            for field in (
                "capture_media",
                "episode_byte_budget",
                "media_buffer_bytes",
            )
        ):
            raise ValueError("workloads must share media and evidence buffer limits")
        if not workload.workload_id or not isinstance(workload.workload_id, str):
            raise ValueError("workload IDs must be nonempty text")
        if (
            any(
                getattr(cfg, field)
                for field in (
                    "source_profile",
                    "collision_profile",
                    "prototype_validation_profile",
                    "motion_profile",
                    "scene_atlas",
                    "visual_profile",
                    "render_profiles",
                    "mass_scale",
                    "friction",
                    "center_of_mass",
                    "joint_stiffness_scale",
                    "joint_damping_scale",
                )
            )
            or cfg.source != "task_program"
        ):
            raise ValueError(
                "configured physical/visual/prototype operators require the cohort job API"
            )
        if (cfg.seed, cfg.max_steps, cfg.settling_steps) != (
            execution.seed,
            execution.max_steps,
            execution.settling_steps,
        ):
            raise ValueError(
                "workload and execution seeds/step/settling budgets must match"
            )
        if (
            workload.extensions.validator_factory
            or workload.extensions.collection_renderer
        ):
            raise ValueError("rendering and validation must use common extensions")
        raw = load_config(cfg.deployment)
        configs.append(
            (
                raw,
                config_to_cfg(
                    raw,
                    source_path=Path(cfg.deployment),
                    task_program_path_override=cfg.program,
                ),
            )
        )
    first = configs[0][1]

    def physical_key(cfg: Any) -> str:
        return _hash(
            {
                name: (
                    getattr(cfg, name).to_dict()
                    if hasattr(getattr(cfg, name), "to_dict")
                    else getattr(cfg, name)
                )
                for name in (
                    "robot",
                    "sim_cfg",
                    "rigid_object",
                    "background",
                    "target_control_frequency",
                    "sim_steps_per_control",
                    "events",
                    "sensor",
                )
                if hasattr(cfg, name)
            }
        )

    if any(physical_key(cfg) != physical_key(first) for _, cfg in configs[1:]):
        raise ValueError(
            "workloads must share physical environment, robot, sensors and cadence"
        )

    owner = source = None
    groups: dict[str, Any] = {}
    adapters: dict[int, Any] = {}
    futures: dict[str, Any] = {}
    started = time.monotonic()
    writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="expansion-export")
    try:
        cfg = ExpansionRunCfg(
            **{**workloads[0].cfg.to_dict(), "batch_size": execution.num_envs}
        )
        owner = GymExpansionHost(
            cfg,
            workloads[0].report,
            root,
            ExpansionJournal(root),
            workloads[0].source_artifacts,
            extensions=common,
        )
        env = owner.env
        adapters[
            id(REGISTERED_ENVS[configs[0][0]["id"]].task_program_adapter_factory)
        ] = env.task_program_adapter
        # Sampling only row zero retains the original B=1 seeded reference
        # regardless of the collection capacity. All rows use that same scene.
        env.reset(seed=execution.seed, options={"save_data": False, "reset_ids": [0]})
        snapshot = _preview_snapshot(env)
        nominal = {
            uid: env.sim.get_rigid_object(uid)
            .get_local_pose()[0]
            .detach()
            .cpu()
            .tolist()
            for uid in env.sim.get_rigid_object_uid_list()
        }
        _atomic_json(
            root / "reference-scene.json",
            {"seed": execution.seed, "poses_xyzw": nominal},
        )
        for workload, (raw, environment_cfg) in zip(workloads, configs, strict=True):
            path = Path(workload.cfg.output_dir)
            path.mkdir(parents=True, exist_ok=False)
            cfg = ExpansionRunCfg(
                **{**workload.cfg.to_dict(), "batch_size": execution.num_envs}
            )
            report = {
                **deepcopy(dict(workload.report)),
                "resolved_deployment": snapshot,
            }
            journal = ExpansionJournal(path)
            journal.open_run(report, cfg.to_dict(), resume=False)
            _atomic_json(path / "preflight.json", report)
            ctx = replace(
                owner._context,
                cfg=cfg,
                report=report,
                root=path,
                journal=journal,
                source_artifacts=deepcopy(dict(workload.source_artifacts)),
                environment_cfg=environment_cfg,
                requests={},
                prepared={},
                preparation_checks={},
                services={},
                bridges={},
                extensions=workload.extensions,
                scene_source=None,
            )
            if workload.extensions.scene_source_factory is not None:
                ctx.scene_source = workload.extensions.scene_source_factory(env)
            builder = GymRecipeBuilder(ctx)
            transform = workload.extensions.recipe_transform
            ctx.recipe = lambda ordinal, builder=builder, transform=transform: (
                builder.recipe(ordinal)
                if transform is None
                else transform(builder.recipe(ordinal))
            )
            # Non-owning facade delegates existing admission/acceptance methods;
            # only owner above opens/closes the shared environment.
            host = GymExpansionHost.__new__(GymExpansionHost)
            host._context = ctx
            factory = REGISTERED_ENVS[raw["id"]].task_program_adapter_factory
            adapter = adapters.get(id(factory))
            if adapter is None:
                adapter = factory.create_adapter(env)
                adapters[id(factory)] = adapter
            ctx.program = adapter.compile(environment_cfg.task_program)
            if (
                workload.extensions.plan_transform_factory is not None
                and adapter.configured_nominal_sources
            ):
                raise ValueError(
                    "a workload transform cannot replace configured nominal sources"
                )
            if (
                groups
                and ctx.program.program_id
                != next(iter(groups.values()))["host"]._context.program.program_id
            ):
                raise ValueError("workloads must share the same Task Program")
            admitted_at = [None]

            def admission_clock(admitted_at=admitted_at):
                return (
                    0.0 if admitted_at[0] is None else time.monotonic() - admitted_at[0]
                )

            session = ExpansionSession(
                ExpansionSessionCfg.from_mapping(
                    {
                        "collection": {
                            "target_committed_episodes": cfg.target_episodes,
                            "max_rollout_attempts": cfg.max_attempts,
                            "max_proposals": cfg.max_attempts,
                            "max_wall_time_s": cfg.max_wall_time_s,
                        },
                        "execution": {
                            "ready_high_watermark": execution.num_envs,
                            "ready_low_watermark": 0,
                        },
                        "persistence": {
                            "pending_episode_limit": execution.num_envs,
                            "pending_max_bytes": cfg.episode_byte_budget
                            * execution.num_envs,
                        },
                        "validation": {"require_path_collision": True},
                    }
                ),
                episode_namespace=journal.job_instance_id,
                clock=admission_clock,
            )
            sink = ArtifactEpisodeSink(path)
            engine = ExpansionEngine(
                ExpansionJobCfg(
                    batch_size=execution.num_envs,
                    candidate_budget=execution.num_envs,
                    episode_byte_budget=cfg.episode_byte_budget,
                ),
                session=session,
                host=host,
                sink=_RecordedSink(sink, journal),
            )
            groups[workload.workload_id] = {
                "host": host,
                "session": session,
                "sink": sink,
                "engine": engine,
                "adapter": adapter,
                "admitted_at": admitted_at,
            }
            journal.stage("collection", "running")
        source = _GymWorkloadSource(owner, groups)
        specs = tuple(
            ExpansionWorkloadCfg(
                workload_id=w.workload_id,
                target_episodes=w.cfg.target_episodes,
                max_attempts=w.cfg.max_attempts,
                max_wall_time_s=w.cfg.max_wall_time_s,
            )
            for w in workloads
        )

        def export(name: str) -> dict[str, Any]:
            group = groups[name]
            journal = group["host"]._context.journal
            stamp = time.monotonic()
            journal.stage("export_lerobot", "running")
            try:
                exported = export_expansion_episodes(journal.root)
                journal.stage("export_lerobot", "complete")
                return {"lerobot": exported, "export_wall_s": time.monotonic() - stamp}
            except Exception as error:
                journal.stage("export_lerobot", "failed", error=str(error))
                raise

        def done(name: str, counts: Mapping[str, object]) -> None:
            group = groups[name]
            journal = group["host"]._context.journal
            complete = counts["committed"] == group["host"].cfg.target_episodes
            journal.stage("collection", "complete" if complete else "budget_exhausted")
            journal.checkpoint(force=True)
            futures[name] = writer.submit(export, name)
            _atomic_json(
                root / "workload-progress.json",
                {
                    key: dict(value["session"].snapshot()["counts"])
                    for key, value in groups.items()
                },
            )

        executor = WorkloadProgramExecutor(
            env, specs, source, execution, on_workload_done=done
        )
        source.execution = {
            "mode": "single_process_single_world",
            "workload_count": execution.workload_count,
            "batch_size": execution.batch_size,
            "num_envs": execution.num_envs,
            "world": dict(executor.world_info),
        }
        _atomic_json(root / "execution.json", source.execution)
        result = dict(executor.run())
        collection_wall = time.monotonic() - started
        source.close()
        for adapter in adapters.values():
            adapter.close()
        owner.close()
        owner = None
        results = {}
        for name, group in groups.items():
            exported = futures[name].result()
            readback = []
            paths = {
                item["episode_id"]: str(
                    Path(exported["lerobot"]["directory"]) / item["dataset_path"]
                )
                for item in exported["lerobot"]["datasets"]
            }
            for receipt in group["sink"].inspect():
                episode = load_expansion_episode(receipt.storage_id)
                readback.append(
                    {
                        "episode_id": episode.episode_id,
                        "steps": len(episode.actions),
                        "expansion": json_safe_copy(
                            episode.metadata.get("expansion", ()),
                            field_name="expansion readback",
                        ),
                        "lerobot": paths[episode.episode_id],
                    }
                )
            counts = result["counts"][name]
            results[name] = {
                "complete": counts["committed"] == group["host"].cfg.target_episodes,
                "target_episodes": group["host"].cfg.target_episodes,
                "committed_episodes": counts["committed"],
                "attempts": counts["attempts"],
                "readback": readback,
                "directory": str(group["host"].root),
                **exported,
            }
            _atomic_json(group["host"].root / "adapter-readback.json", readback)
        result.update(
            workloads=results,
            collection_wall_s=collection_wall,
            wall_s=time.monotonic() - started,
        )
        _atomic_json(root / "workload-execution.json", result)
        return result
    except BaseException as error:
        for name, group in groups.items():
            if name not in futures:
                group["host"]._context.journal.stage(
                    "collection", "failed", error=str(error)
                )
        raise
    finally:
        writer.shutdown(wait=True, cancel_futures=True)
        if source is not None:
            source.close()
        for adapter in adapters.values():
            adapter.close()
        for group in groups.values():
            group["host"]._context.journal.checkpoint(force=True, state="finished")
        if owner is not None:
            owner.close()
