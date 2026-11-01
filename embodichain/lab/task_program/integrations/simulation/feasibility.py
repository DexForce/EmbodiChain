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

"""Gym-owned joint scene/deployment preparation and measured Task Program trials."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from embodichain.lab.sim.scene.feasibility import (
    CandidateEvaluation,
    FeasibilityBudget,
    FeasibilityBudgetExhausted,
    FeasibilityCheck,
    JointSceneCandidate,
    TaskFeasibilityRequest,
    TaskPhase,
    RobotDeploymentCandidate,
    RobotFeasibilityBackend,
    FCLCollisionChecker,
    UnsupportedFeasibilityError,
    plan_task_mode,
)
from embodichain.lab.sim.scene.feasibility._values import digest, plain
from embodichain.lab.sim.motion._budget import _planning_budget_scope
from embodichain.lab.sim.motion.expansion import (
    CandidateTrajectoryBatch,
    CommitReceipt,
    ExpansionSession,
    ExpertEpisode,
    SceneCase,
    TrajectoryExpansionCfg,
    ValidationCheck,
    ValidationResult,
)
from embodichain.lab.task_program.integrations.scene_expansion import (
    execute_scene_variant,
)
from .scene_expansion import SimulationSceneExpansionHost

__all__ = ["TaskProgramFeasibilityHost", "save_feasibility_episode"]


@contextmanager
def _bounded_simulation_updates(environment: Any, budget: FeasibilityBudget):
    sim = environment.sim
    original = getattr(sim, "update", None)
    if not callable(original):
        yield
        return

    def checked(*args, **kwargs):
        budget.check()
        result = original(*args, **kwargs)
        budget.check()
        return result

    sim.update = checked
    try:
        yield
    finally:
        sim.update = original


def save_feasibility_episode(
    episode: ExpertEpisode, directory: str | Path
) -> CommitReceipt:
    """Atomically persist measured tensors and return an idempotent commit receipt.

    Args:
        episode: Accepted session-owned execution evidence.
        directory: Destination root; filename uses only the generated commit ID.
    Returns:
        Confirmed receipt after atomic file publication; writing errors propagate.
    """
    import os
    import tempfile
    from embodichain.lab.sim.scene.feasibility._values import plain

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    if (
        Path(episode.commit_id).name != episode.commit_id
        or "\\" in episode.commit_id
        or episode.commit_id in (".", "..")
    ):
        raise ValueError("commit_id must be a plain filename component")
    path = root / (episode.commit_id + ".npz")
    payload = {
        "actions": episode.actions.cpu().numpy(),
        "timestamps": episode.timestamps.cpu().numpy(),
        **{
            "observation/" + k: v.cpu().numpy() for k, v in episode.observations.items()
        },
    }
    import json

    payload["metadata"] = np.array(
        json.dumps(
            {
                "identity": plain(episode.identity),
                "validation": plain(episode.validation),
                "metadata": plain(episode.metadata),
            },
            allow_nan=False,
        )
    )
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if set(existing.files) != set(payload) or any(
                not np.array_equal(existing[k], v) for k, v in payload.items()
            ):
                raise ValueError(
                    "Existing commit artifact differs from the accepted episode"
                )
    else:
        fd, temporary = tempfile.mkstemp(dir=root, suffix=".npz")
        try:
            with os.fdopen(fd, "wb") as stream:
                np.savez_compressed(stream, **payload)
            try:
                os.link(temporary, path)
            except FileExistsError:
                with np.load(path, allow_pickle=False) as existing:
                    if set(existing.files) != set(payload) or any(
                        not np.array_equal(existing[k], v) for k, v in payload.items()
                    ):
                        raise ValueError(
                            "Concurrent commit artifact differs from the accepted episode"
                        )
        finally:
            Path(temporary).unlink(missing_ok=True)
    identity = episode.identity
    return CommitReceipt(
        episode.episode_id,
        identity.candidate_id,
        identity.attempt_id,
        str(path.resolve()),
        episode.commit_id,
        identity.scene_case_id,
    )


def _selected_observations(
    observations: object, keys: tuple[str, ...], max_bytes: int
) -> dict[str, torch.Tensor]:
    if not keys:
        return {}
    leaves = {}

    def visit(value, prefix=""):
        if isinstance(value, Mapping) or callable(getattr(value, "items", None)):
            for name, child in value.items():
                visit(child, prefix + str(name) + "/")
        else:
            key = prefix[:-1]
            if key in leaves:
                raise ValueError("Ambiguous flattened observation keys")
            leaves[key] = value

    visit(observations)
    selected = {}
    size = 0
    for key in keys:
        value = leaves.get(key)
        if isinstance(value, np.ndarray):
            value = torch.as_tensor(value)
        if not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[0] != 1:
            raise UnsupportedFeasibilityError(
                f"Required B=1 observation {key!r} is unavailable"
            )
        size += value.numel() * value.element_size()
        if size > max_bytes:
            raise FeasibilityBudgetExhausted("recording_budget_exhausted")
        selected["env/" + key] = value[0].detach().cpu().clone()
    return selected


class TaskProgramFeasibilityHost:
    """Evaluate full task modes through canonical Gym execution and shared sessions.

    Args:
        environment_factory: Construct a B=1 environment from copied config at
            these robot root poses. It owns robot/profile/policy selection.
        initial_validator: Required original semantics and physical-layout checks
            after settling. Check IDs must include ``semantics`` and ``geometry``.
        measured_validator: Required state/event task evidence after execution.
        collision_factory: Build complete physical robot/environment geometry.
        runtime_phase: Resolve current contact/held-object permissions from the
            actual controller action provenance, never a guessed path index.
        configure_mode: Bind the selected legal grasp/assignment mode to runtime.
            Required for multiple modes; single-mode default is runtime policy.
        observe: Optional per-control-step measured event monitor.
        sink: Optional accepted ExpertEpisode writer returning CommitReceipt.
        workspace_proposal: Optional explicit phase/part/group workspace prior settings.
        observation_keys: Required flattened environment observation paths. Saved
            under env/ with T+1 actual reset/step samples; missing channels fail.
        settle_steps: Existing host settling substeps (stability still requires
            explicit initial validation).
    """

    def __init__(
        self,
        *,
        environment_factory: Callable[[RobotDeploymentCandidate], Any],
        initial_validator: Callable[[Any, JointSceneCandidate], ValidationResult],
        measured_validator: Callable[[Any, JointSceneCandidate], ValidationResult],
        collision_factory: Callable[[Any], FCLCollisionChecker],
        runtime_phase: Callable[[Any, Any, JointSceneCandidate], TaskPhase],
        configure_mode: Callable[[Any, JointSceneCandidate], None] | None = None,
        observe: Callable[[Any, Any], None] | None = None,
        sink: Callable[[ExpertEpisode], CommitReceipt] | None = None,
        settle_steps: int = 0,
        observation_keys: Sequence[str] = (),
        workspace_proposal: Mapping[str, object] | None = None,
    ) -> None:
        self._factory, self._initial, self._measured = (
            environment_factory,
            initial_validator,
            measured_validator,
        )
        self._collision_factory, self._runtime_phase = collision_factory, runtime_phase
        self._configure_mode, self._observe, self._sink = configure_mode, observe, sink
        self._settle_steps = settle_steps
        from embodichain.lab.sim.scene.feasibility._values import names

        self._observation_keys = names(observation_keys, "observation_keys")
        self._workspace_settings = (
            None if workspace_proposal is None else dict(workspace_proposal)
        )
        self._proposal_source = None
        self._env = None
        self._owned_environment = None
        self._request = None
        self._session = None
        self.last_episode = None

    def prepare(
        self, request: TaskFeasibilityRequest, budget: FeasibilityBudget
    ) -> None:
        """Bind immutable task semantics and instantiate the existing session budget."""
        budget.check()
        self.close()
        self._request = request
        cfg = TrajectoryExpansionCfg()
        maximum = (
            budget.cfg.max_candidates
            + budget.cfg.max_regions * budget.cfg.validation_samples
        )
        cfg.collection.max_proposals = maximum
        cfg.collection.max_rollout_attempts = budget.cfg.max_rollouts
        cfg.collection.target_committed_episodes = min(maximum, budget.cfg.max_rollouts)
        cfg.collection.max_wall_time_s = budget.cfg.max_wall_time_s
        cfg.augmentation.seed = budget.cfg.seed
        cfg.persistence.pending_max_bytes = 4 * budget.cfg.recording_max_bytes
        self._session = ExpansionSession(cfg, clock=budget.clock)
        self.last_episode = None
        self._proposal_source = None
        if self._workspace_settings is not None:
            from embodichain.lab.sim.scene.feasibility import prepare_feasibility
            from embodichain.lab.sim.scene.feasibility.proposals import (
                RobotWorkspaceProposalSource,
            )

            prepared = prepare_feasibility(request, budget.cfg)
            nominal = prepared.candidate(
                (0.5,) * prepared.dimensions,
                mode_id=request.modes[0].mode_id,
                ordinal=0,
            )
            try:
                self._owned_environment = self._factory(nominal.deployment)
                self._env = getattr(
                    self._owned_environment, "unwrapped", self._owned_environment
                )
                with _bounded_simulation_updates(self._env, budget):
                    self._env.reset(
                        seed=budget.cfg.seed % (2**31), options={"save_data": False}
                    )
                    self._proposal_source = RobotWorkspaceProposalSource(
                        self._env.robot, request, budget, **self._workspace_settings
                    )
            finally:
                self.close()

    @property
    def proposal_metrics(self) -> Mapping[str, str | int] | None:
        """Return the numerical workspace prior identity and usage, if enabled."""
        return None if self._proposal_source is None else self._proposal_source.metrics

    def propose(
        self,
        parameters: Sequence[float],
        mode_id: str,
        ordinal: int,
        budget: FeasibilityBudget,
    ) -> tuple[float, ...]:
        """Supply optional workspace guidance for full-domain exploration draws."""
        return (
            tuple(parameters)
            if self._proposal_source is None
            else self._proposal_source.propose(parameters, mode_id, ordinal, budget)
        )

    @property
    def session_snapshot(self) -> Mapping[str, Any]:
        """Return canonical attempt/receipt accounting without exposing mutable state."""
        return {} if self._session is None else self._session.snapshot()

    def close(self) -> None:
        """Release the owned native environment without committing live recordings."""
        try:
            if self._env is not None:
                self._env.close(exit_process=False)
        finally:
            try:
                if (
                    self._owned_environment is not None
                    and self._owned_environment is not self._env
                ):
                    self._owned_environment.close()
            finally:
                self._env = None
                self._owned_environment = None
                from embodichain.lab.sim import SimulationManager

                SimulationManager.flush_cleanup_queue()

    def _entities(self) -> dict[str, np.ndarray]:
        entities = {
            uid: self._env.sim.get_rigid_object(uid)
            .get_local_pose(to_matrix=True)[0]
            .detach()
            .cpu()
            .double()
            .numpy()
            for uid in self._env.sim.get_rigid_object_uid_list()
        }
        root_pose = getattr(self._env.robot, "get_local_pose", None)
        if callable(root_pose):
            entities[self._env.robot.uid] = (
                root_pose(to_matrix=True)[0].detach().cpu().double().numpy()
            )
        if getattr(self._env.sim, "_default_plane", None) is not None:
            entities["default_plane"] = np.eye(4)
        return entities

    def evaluate(
        self, candidate: JointSceneCandidate, budget: FeasibilityBudget
    ) -> CandidateEvaluation:
        """Prepare, plan, restore and independently measure one exact deployment."""
        if self._request is None or candidate.request_id != self._request.request_id:
            raise ValueError("host must be prepared for this exact task request")
        budget.check()
        self.close()
        checks = []
        identity = None
        released = False
        pending_write = False
        started = False
        snapshot_id = None
        started_at = budget.clock()
        original_reset = None
        simulation_updates = ExitStack()
        try:
            self._owned_environment = self._factory(candidate.deployment)
            env = getattr(self._owned_environment, "unwrapped", self._owned_environment)
            self._env = env
            simulation_updates.enter_context(_bounded_simulation_updates(env, budget))
            expert = getattr(getattr(env, "cfg", None), "expert_trajectory", None)
            if getattr(expert, "joint_command_mode", "position") != "position":
                raise UnsupportedFeasibilityError(
                    "The feasibility session requires position-mode qpos episode labels"
                )
            original_reset = env.reset
            last_initial_observations = {}

            def capture_reset(*args, **kwargs):
                nonlocal last_initial_observations
                result = original_reset(*args, **kwargs)
                raw = result[0] if isinstance(result, tuple) else result
                last_initial_observations = _selected_observations(
                    raw, self._observation_keys, budget.cfg.recording_max_bytes
                )
                return result

            env.reset = capture_reset
            budget.check()
            execution_seed = int(candidate.candidate_id[:15], 16) % (2**31)
            env.reset(seed=execution_seed, options={"save_data": False})
            budget.check()
            if self._configure_mode is not None:
                self._configure_mode(env, candidate)
            elif len(self._request.modes) != 1:
                raise UnsupportedFeasibilityError(
                    "Multiple modes require explicit runtime mode binding"
                )
            mode = next(
                m for m in self._request.modes if m.mode_id == candidate.mode_id
            )

            def initial() -> ValidationResult:
                budget.check()
                result = self._initial(env, candidate)
                if not isinstance(result, ValidationResult):
                    raise TypeError("initial validator must return ValidationResult")
                names = {c.check_id for c in result.checks}
                if not {"semantics", "geometry"} <= names:
                    return ValidationResult(
                        (
                            *result.checks,
                            ValidationCheck(
                                "required_scene_checks",
                                "unavailable",
                                "Both semantics and geometry evidence are required",
                            ),
                        )
                    )
                return result

            host = SimulationSceneExpansionHost(
                env, initial_validator=initial, settle_steps=self._settle_steps
            )
            prepared = host(candidate.scene)
            snapshot = host.initial_state
            snapshot_id = prepared.initial_state_id
            for stage in ("semantics", "geometry"):
                relevant = next(
                    (c for c in prepared.validation.checks if c.check_id == stage), None
                )
                status = (
                    "unverified"
                    if relevant is None or relevant.status in ("unavailable", "not_run")
                    else relevant.status
                )
                checks.append(
                    FeasibilityCheck(
                        stage, status, "Actual settled initial-state checks"
                    )
                )
            if not prepared.validation.accepted:
                unresolved = [
                    c for c in prepared.validation.checks if c.status != "passed"
                ]
                status = (
                    "unverified"
                    if any(c.status in ("unavailable", "not_run") for c in unresolved)
                    else "failed"
                )
                checks.append(
                    FeasibilityCheck(
                        "preparation",
                        status,
                        "; ".join(
                            f"{c.check_id}: {c.status} {c.detail}" for c in unresolved
                        ),
                    )
                )
                return CandidateEvaluation(
                    candidate.candidate_id, tuple(checks), initial_state_id=snapshot_id
                )
            collision = self._collision_factory(env)
            actual_entities = self._entities()
            entities = {**self._request.entity_poses, **actual_entities}
            backend = RobotFeasibilityBackend(
                env.robot,
                collision,
                budget.cfg,
                entities,
                seed=int(candidate.candidate_id[:15], 16),
            )
            dynamic_objects = [
                uid
                for uid in env.sim.get_rigid_object_uid_list()
                if env.sim.get_rigid_object(uid).body_type == "dynamic"
            ]
            initial_phase = replace(mode.phases[0], held_objects=())
            initial_collision, _ = collision.check(
                backend.frame_poses(
                    env.robot.get_qpos()[0].detach().cpu().double().numpy(),
                    initial_phase,
                    entities,
                    measured=True,
                ),
                allowed_contacts=initial_phase.allowed_contacts,
                moving_objects=dynamic_objects,
                all_pairs=True,
            )
            checks.append(
                FeasibilityCheck(
                    "initial_collision",
                    initial_collision.status,
                    initial_collision.reason,
                )
            )
            if initial_collision.status != "passed":
                return CandidateEvaluation(
                    candidate.candidate_id, tuple(checks), initial_state_id=snapshot_id
                )
            plan = plan_task_mode(mode, backend, entities, budget)

            def affected(phase_id=None):
                phase = next((p for p in mode.phases if p.phase_id == phase_id), None)
                uids = set()
                if phase is not None:
                    uids.update(
                        t.frame_uid for t in phase.targets if t.frame_uid is not None
                    )
                    uids.update(h.entity_uid for h in phase.held_objects)
                pair = getattr(collision, "last_pair", None) or ()
                uids.update(
                    b.frame_uid for b in collision.bodies if b.collision_id in pair
                )
                return tuple(
                    v.name
                    for v in self._request.variables
                    if v.kind == "robot" or set(v.members) & uids
                )

            checks.extend(
                (
                    replace(c, affected_variables=affected(c.phase_id))
                    if c.status != "passed"
                    else c
                )
                for c in plan.checks
            )
            if not plan.accepted:
                return CandidateEvaluation(
                    candidate.candidate_id, tuple(checks), initial_state_id=snapshot_id
                )
            restored_preparation = host.restore_initial_state(snapshot)
            snapshot_id = restored_preparation.initial_state_id
            if not restored_preparation.validation.accepted:
                return CandidateEvaluation(
                    candidate.candidate_id,
                    (
                        *checks,
                        FeasibilityCheck(
                            "restore",
                            "failed",
                            "Actual restored state failed validation",
                        ),
                    ),
                    initial_state_id=snapshot_id,
                )
            # The shared session identifies the actual restored execution start.
            case = SceneCase(
                candidate.candidate_id,
                snapshot_id,
                candidate.scene.variant_id,
                self._request.task_id,
                self._request.robot_profile_id,
                self._request.policy_id,
            )
            session = self._session
            session.register_case(
                case,
                torch.tensor(backend.joint_limits),
                joint_names=backend.joint_names,
            )
            identity, _ = session.propose(
                case.scene_case_id,
                snapshot_id,
                source_id="scene_robot_feasibility",
                source_revision=candidate.request_id,
                template_id=candidate.candidate_id,
                operator_id=candidate.mode_id,
            )
            planned = np.concatenate(
                [p if i == 0 else p[1:] for i, p in enumerate(plan.paths)]
            )
            # Timing is only the screening reference. Actual controls below are
            # generated/replanned by the canonical Task Program runtime.
            dt = torch.full((1, len(planned)), float(env.step_dt))
            dt[0, 0] = 0.0
            session.add_planned(
                CandidateTrajectoryBatch(
                    torch.tensor(planned[None], dtype=torch.float32),
                    dt,
                    torch.tensor([len(planned)]),
                    (identity,),
                    backend.joint_names,
                ),
                ValidationResult((ValidationCheck("path_collision", "passed"),)),
            )
            if (
                session.take_ready(
                    case.scene_case_id, snapshot_id, candidate_id=identity.candidate_id
                )
                is None
            ):
                session.release(identity, reason="capacity_unavailable")
                released = True
                return CandidateEvaluation(
                    candidate.candidate_id,
                    (
                        *checks,
                        FeasibilityCheck(
                            "execution", "unverified", "No shared-session capacity"
                        ),
                    ),
                    initial_state_id=snapshot_id,
                )
            samples, actions, object_samples = [], [], []
            extra_observations = {
                key: [value] for key, value in last_initial_observations.items()
            }
            recorded_bytes = 65536 + sum(
                value.numel() * value.element_size()
                for value in last_initial_observations.values()
            )
            collision_passed = True
            first_runtime_collision = None
            minimum_clearance = float("inf")
            original_step = env.step
            self._step_error = None

            def checked_step_impl(action: Any, **kwargs: Any) -> Any:
                nonlocal started, collision_passed, minimum_clearance, recorded_bytes, first_runtime_collision
                budget.check()
                phase = self._runtime_phase(env, action, candidate)
                if not isinstance(phase, TaskPhase):
                    raise TypeError("runtime_phase must return TaskPhase")
                declared_phase = next(
                    (p for p in mode.phases if p.phase_id == phase.phase_id), None
                )
                if declared_phase is None or not set(phase.allowed_contacts) <= set(
                    declared_phase.allowed_contacts
                ):
                    raise ValueError(
                        "Runtime phase cannot add undeclared contact permissions"
                    )
                declared_holders = {
                    (h.entity_uid, h.control_part) for h in declared_phase.held_objects
                }
                if (
                    not {(h.entity_uid, h.control_part) for h in phase.held_objects}
                    <= declared_holders
                ):
                    raise ValueError(
                        "Runtime phase cannot rewrite declared object ownership"
                    )
                if not started:
                    budget.charge("rollouts")
                    samples.append(env.robot.get_qpos()[0].detach().cpu().clone())
                    object_samples.append(self._entities())
                    recorded_bytes += samples[0].numel() * samples[
                        0
                    ].element_size() + sum(v.nbytes for v in object_samples[0].values())
                result = original_step(action, **kwargs)
                if not started:
                    session.mark_rollout_started(identity)
                    started = True
                qpos = env.robot.get_qpos()[0].detach().cpu().clone()
                measured_entities = self._entities()
                # Physical object poses are measured, not projected attachments.
                frames = backend.frame_poses(
                    qpos.double().numpy(),
                    replace(phase, held_objects=()),
                    measured_entities,
                    measured=True,
                )
                report, clearance = collision.check(
                    frames,
                    allowed_contacts=phase.allowed_contacts,
                    moving_objects=tuple(
                        set(dynamic_objects)
                        | {h.entity_uid for h in phase.held_objects}
                    ),
                )
                collision_passed &= report.status == "passed"
                if report.status != "passed" and first_runtime_collision is None:
                    first_runtime_collision = {
                        "control_step": len(actions) + 1,
                        "reason": report.reason,
                        "phase_id": phase.phase_id,
                    }
                minimum_clearance = min(minimum_clearance, clearance)
                raw_observations = result[0] if isinstance(result, tuple) else result
                current_observations = _selected_observations(
                    raw_observations,
                    self._observation_keys,
                    budget.cfg.recording_max_bytes - recorded_bytes,
                )
                recorded_bytes += (
                    2 * qpos.numel() * qpos.element_size()
                    + sum(v.nbytes for v in measured_entities.values())
                    + sum(
                        v.numel() * v.element_size()
                        for v in current_observations.values()
                    )
                )
                if recorded_bytes > budget.cfg.recording_max_bytes:
                    raise FeasibilityBudgetExhausted("recording_budget_exhausted")
                for key, value in current_observations.items():
                    extra_observations[key].append(value)
                samples.append(qpos)
                actions.append(
                    env.robot.get_qpos(target=True)[0].detach().cpu().clone()
                )
                object_samples.append(measured_entities)
                if self._observe is not None:
                    self._observe(env, action)
                budget.check()
                return result

            def checked_step(action: Any, **kwargs: Any) -> Any:
                try:
                    return checked_step_impl(action, **kwargs)
                except BaseException as error:
                    self._step_error = error
                    raise

            def measured() -> ValidationResult:
                budget.check()
                result = self._measured(env, candidate)
                return ValidationResult(
                    (
                        *result.checks,
                        ValidationCheck(
                            "feasibility.runtime_collision",
                            "passed" if collision_passed and started else "failed",
                            "Physical geometry at observed control boundaries",
                        ),
                    )
                )

            env.step = checked_step
            try:
                from embodichain.lab.gym.envs.task_program.bridge import (
                    TaskProgramBridgeError,
                )

                try:
                    with _planning_budget_scope(budget):
                        outcome = execute_scene_variant(
                            env,
                            candidate.scene,
                            prepare_scene=lambda _: restored_preparation,
                            measured_validator=measured,
                        )
                except TaskProgramBridgeError as error:
                    if self._step_error is not None:
                        raise self._step_error from error
                    budget.check()
                    return CandidateEvaluation(
                        candidate.candidate_id,
                        (*checks, FeasibilityCheck("execution", "failed", str(error))),
                        initial_state_id=snapshot_id,
                    )
                if self._step_error is not None:
                    raise self._step_error
            finally:
                env.step = original_step
            budget.check()
            accepted = outcome.accepted and collision_passed and started
            checks.append(
                FeasibilityCheck(
                    "execution",
                    "passed" if accepted else "failed",
                    "Canonical completion, measured task evidence and runtime collision checks",
                )
            )
            committed = False
            evidence = {
                "task_program": outcome.to_metadata(),
                "execution": (
                    None
                    if outcome.execution is None
                    else outcome.execution.to_metadata()
                ),
                "terminal_poses": self._entities(),
                "final_joint_positions": env.robot.get_qpos()[0]
                .detach()
                .cpu()
                .tolist(),
                "first_runtime_collision": first_runtime_collision,
                "request_id": candidate.request_id,
                "execution_seed": execution_seed,
                "wall_time_s": budget.clock() - started_at,
                "control_period_s": float(env.step_dt),
                "simulation_device": str(getattr(env.sim, "device", "unknown")),
                "candidate_id": candidate.candidate_id,
                "deployment_id": candidate.deployment.deployment_id,
                "collision_geometry_id": digest(collision.bodies),
                "runtime_collision_sampling": "control_boundaries",
                "session_candidate_id": identity.candidate_id,
            }
            evidence = plain(evidence)
            if accepted:
                episode_id, commit_id = session.episode_ids(identity)
                observed = {
                    "joint_positions": torch.stack(samples),
                    **{
                        key: torch.stack(values)
                        for key, values in extra_observations.items()
                    },
                }
                for uid in actual_entities:
                    observed["entity/" + uid] = torch.tensor(
                        np.array([row[uid] for row in object_samples]),
                        dtype=torch.float32,
                    )
                episode = ExpertEpisode(
                    identity,
                    observed,
                    torch.stack(actions),
                    torch.arange(len(samples), dtype=torch.float64)
                    * float(env.step_dt),
                    "qpos",
                    ValidationResult(
                        (
                            ValidationCheck("task_success", "passed"),
                            ValidationCheck("path_collision", "passed"),
                        )
                    ),
                    episode_id,
                    commit_id,
                    metadata=evidence,
                )
                self.last_episode = episode
                try:
                    admitted = session.accept_episode(episode)
                except BufferError as error:
                    released = True  # The session releases oversized payloads.
                    raise FeasibilityBudgetExhausted(
                        "recording_budget_exhausted"
                    ) from error
                if admitted:
                    pending_write = True
                    if self._sink is not None:
                        try:
                            receipt = self._sink(episode)
                        except Exception as error:
                            receipt = CommitReceipt(
                                episode_id,
                                identity.candidate_id,
                                identity.attempt_id,
                                "failed",
                                commit_id,
                                identity.scene_case_id,
                                confirmed=False,
                                error=str(error),
                            )
                            session.apply_receipt(receipt)
                            pending_write = False
                            session.release(identity, reason="persistence_failed")
                            released = True
                            if not isinstance(error, OSError):
                                raise
                        if not released:
                            session.apply_receipt(receipt)
                        pending_write = False
                        if not receipt.confirmed and not released:
                            session.release(identity, reason="persistence_failed")
                        committed = receipt.confirmed
                        evidence["storage_id"] = receipt.storage_id
                    else:
                        # No writer was requested; release the reserved coverage
                        # through an explicit failed receipt, never mark committed.
                        session.apply_receipt(
                            CommitReceipt(
                                episode_id,
                                identity.candidate_id,
                                identity.attempt_id,
                                "not_requested",
                                commit_id,
                                identity.scene_case_id,
                                confirmed=False,
                                error="No persistence sink requested",
                            )
                        )
                        pending_write = False
                        session.release(identity, reason="persistence_not_requested")
                    released = True
                else:
                    released = True
            if not released:
                session.release(identity, reason="measured_task_rejected")
                released = True
            return CandidateEvaluation(
                candidate.candidate_id,
                tuple(checks),
                outcome.program_completed,
                accepted,
                snapshot_id,
                0.0 if np.isinf(minimum_clearance) else minimum_clearance,
                len(actions) * float(env.step_dt),
                committed,
                evidence,
            )
        except UnsupportedFeasibilityError as error:
            return CandidateEvaluation(
                candidate.candidate_id,
                (*checks, FeasibilityCheck("capability", "unsupported", str(error))),
                initial_state_id=snapshot_id,
            )
        finally:
            try:
                simulation_updates.close()
                if identity is not None and not released and not pending_write:
                    self._session.release(identity, reason="interrupted_or_failed")
            finally:
                try:
                    if self._env is not None:
                        if original_reset is not None:
                            self._env.reset = original_reset
                        # Consume/discard recordings through their existing owner.
                        self._env.reset(options={"save_data": False})
                finally:
                    self.close()
