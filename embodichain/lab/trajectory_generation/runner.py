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

"""Synchronous fixed-scene qpos collection through qualified motion/host ports."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import MISSING, fields, is_dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import TYPE_CHECKING

import torch

from embodichain import __version__
from embodichain.lab.sim.motion.expansion import (
    CandidateTrajectoryBatch,
    CommitReceipt,
    ExpertEpisode,
    GenerationSession,
    SceneCase,
    TrajectoryGenerationJobCfg,
    TrajectoryTemplate,
    ValidationCheck,
    ValidationResult,
    joint_residual,
    retime,
    validate_motion_limits,
)
from embodichain.utils import configclass

if TYPE_CHECKING:
    from .execution import QposRolloutExecutor
    from .initial_state import FixedSceneHost
    from .integrations.planning import EnvRowMotionPlanner
    from .sinks import LeRobotEpisodeSink

from .integrations.contact import PickUpMotionValidator

__all__ = ["MotionLimitsProfile", "GenerationRunner"]


@configclass
class MotionLimitsProfile:
    """Trusted full-joint velocity and acceleration limits.

    Args:
        velocity_limits: Positive finite speed limits in joint units per second.
        acceleration_limits: Positive finite acceleration limits in joint units
            per second squared, in the same full-joint order.
        profile_id: Registered policy ID used by the generation job.
    """

    velocity_limits: torch.Tensor = MISSING
    acceleration_limits: torch.Tensor = MISSING
    profile_id: str = "robot_execution_limits"

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise ValueError("motion limits profile_id must be a nonempty string")
        for name in ("velocity_limits", "acceleration_limits"):
            value = getattr(self, name)
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 1
                or not value.numel()
                or not value.is_floating_point()
                or not bool(torch.isfinite(value).all())
                or not bool((value > 0).all())
            ):
                raise ValueError(f"{name} must be a positive finite floating vector")
            setattr(
                self, name, value.detach().to(device="cpu", dtype=torch.float64).clone()
            )
        if self.velocity_limits.shape != self.acceleration_limits.shape:
            raise ValueError("motion limit vectors must use the same joint order")


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if is_dataclass(value):
        return {
            field.name: _plain(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


class GenerationRunner:
    """Collect qualified qpos variants with one synchronous owner.

    The caller supplies explicit templates, a qualified initial-state profile,
    motion collision model, actual-observation encoder, and task validator.
    This runner supports handwritten free-motion sources and offline atomic
    PickUp exports in ``env_rows`` / ``full_batch`` / synchronous LeRobot mode.
    PickUp requires one shared full-state/contact validator for planning and
    pure-sim execution. AtomicActionRuntime and contact-aware Gym execution
    remain separate integrations.

    Planning failures do not reset physics. Ready candidates reserve episode
    memory and target quota before restoration/execution. Only actual accepted
    rollouts reach the sink, and only confirmed receipts increase coverage.
    The runner is single-use and closes the supplied host and sink on exit.
    ``generation_report.json`` preserves resolved settings, audit, and outcome.

    Args:
        cfg: Strict job settings matching the supplied integration IDs.
        host: Exclusive owner of the full simulator/Gym batch.
        planner: Environment-row motion validation adapter for this robot.
        executor: Concrete qpos rollout adapter for this host.
        sink: Synchronous, local LeRobot writer with readback confirmation.
        motion_limits: Trusted full-joint speed/acceleration policy.
        max_write_retries: Extra synchronous submissions after a failed receipt.
        clock: Monotonic clock used for the job wall-time budget.
    """

    def __init__(
        self,
        cfg: TrajectoryGenerationJobCfg,
        host: FixedSceneHost,
        planner: EnvRowMotionPlanner | PickUpMotionValidator,
        executor: QposRolloutExecutor,
        sink: LeRobotEpisodeSink,
        *,
        motion_limits: MotionLimitsProfile,
        max_write_retries: int = 1,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = TrajectoryGenerationJobCfg.from_mapping(cfg.to_dict())
        self.host, self.planner, self.executor, self.sink = (
            host,
            planner,
            executor,
            sink,
        )
        self.motion_limits = motion_limits.copy()
        self.clock = clock
        self._ran = False
        if type(max_write_retries) is not int or max_write_retries < 0:
            raise ValueError("max_write_retries must be a nonnegative integer")
        self.max_write_retries = max_write_retries
        contact_planner = isinstance(planner, PickUpMotionValidator)
        if cfg.source.kind != "handwritten" and not contact_planner:
            raise ValueError("this runner requires a handwritten qpos source")
        if contact_planner and executor.contact_validator is not planner:
            raise ValueError(
                "PickUp planning and substep validation must share one validator"
            )
        if (
            not contact_planner
            and getattr(executor, "contact_validator", None) is not None
        ):
            raise ValueError(
                "Contact execution requires complete PickUp planning validation"
            )
        if contact_planner and cfg.augmentation.factors.timing.enabled:
            raise ValueError("PickUp contact timing remains fixed")
        if planner.robot is not host.adapter.robot or executor.host is not host:
            raise ValueError("planner and executor must use the same owned host robot")
        expected = {
            "prepare profile": (cfg.reset.prepare_profile_id, host.profile.profile_id),
            "initial-state tolerances": (
                cfg.reset.initial_state_tolerances_profile_id,
                host.adapter.tolerances_profile_id,
            ),
            "motion limits": (
                cfg.validation.motion_limits_profile_id,
                motion_limits.profile_id,
            ),
            "validator": (cfg.validation.validator_id, executor.validator_id),
            "validation profile": (
                cfg.validation.profile_id,
                executor.validation_profile_id,
            ),
            "sink": (cfg.persistence.sink, "lerobot"),
        }
        for label, (requested, actual) in expected.items():
            if requested != actual:
                raise ValueError(
                    f"{label} ID {requested!r} does not match supplied {actual!r}"
                )
        if len(motion_limits.velocity_limits) != len(host.adapter.robot.joint_names):
            raise ValueError("motion limits must cover every robot joint")
        if not math.isclose(
            executor.control_dt * sink.fps, 1, rel_tol=1e-7, abs_tol=1e-9
        ):
            raise ValueError("sink fps must match the authoritative control period")
        cfg.validate_capabilities(
            source_ids=(cfg.source.source_id,),
            validator_ids=(executor.validator_id,),
            profile_ids=(
                host.profile.profile_id,
                host.adapter.tolerances_profile_id,
                motion_limits.profile_id,
                executor.validation_profile_id,
            ),
            sink_ids=("lerobot",),
            operators=("joint_residual", "retime"),
        )
        self._episode_budget = executor.max_episode_bytes
        if self._episode_budget > min(
            sink.max_episode_bytes, cfg.persistence.pending_max_bytes
        ):
            raise ValueError(
                "executor max_episode_bytes must fit both sink and pending byte limits"
            )

    def _validate_templates(self, cases, templates) -> None:
        if len(cases) != self.host.adapter.sim.num_envs or len(templates) != len(cases):
            raise ValueError("provide one case and template per physical row")
        keys = [(case.scene_case_id, case.initial_state_id) for case in cases]
        if len(set(keys)) != len(keys):
            raise ValueError("per_env_case requires distinct case/initial-state pairs")
        for template in templates:
            if not isinstance(template, TrajectoryTemplate):
                raise TypeError("templates must be explicit TrajectoryTemplate values")
            if (
                template.source_id != self.cfg.source.source_id
                or template.template_id != self.cfg.source.template_id
                or template.validator_id != self.cfg.validation.validator_id
            ):
                raise ValueError(
                    "template source/template/validator IDs do not match the job"
                )
            if template.joint_names != tuple(self.host.adapter.robot.joint_names):
                raise ValueError(
                    "template joint names must match the complete robot order"
                )
            if len(template.positions) < 2:
                raise ValueError("an executable template needs at least two samples")
            factors = self.cfg.augmentation.factors
            needed = []
            if factors.spatial.enabled:
                needed.append(factors.spatial.method)
            if factors.timing.enabled:
                needed.append("retime")
            if set(needed) - set(template.allowed_operators):
                raise ValueError(
                    "job enables operators not allowed by the source template"
                )
            if not factors.timing.enabled and not torch.allclose(
                template.dt[1:].double(),
                torch.full_like(template.dt[1:].double(), self.executor.control_dt),
                atol=1e-8,
                rtol=0,
            ):
                raise ValueError(
                    "template timing must match the host clock or permit retime"
                )

    def _variant(self, template, limits, generator):
        factors = self.cfg.augmentation.factors
        value = template
        metadata = {}
        if factors.spatial.enabled:
            value = joint_residual(
                value,
                joint_limits=limits,
                normalized_scale=factors.spatial.joint_offset_scale,
                generator=generator,
            )
            metadata["joint_offset_scale"] = factors.spatial.joint_offset_scale
        if factors.timing.enabled:
            scales = factors.timing.duration_scales
            scale = scales[int(torch.randint(len(scales), (), generator=generator))]
            # Account for positions and arrival intervals before allocating retime output.
            max_samples = self.cfg.execution.ready_max_bytes // (
                value.positions.element_size() * value.positions.shape[1]
                + value.dt.element_size()
            )
            value = retime(
                value,
                duration_scale=scale,
                control_dt=self.executor.control_dt,
                max_samples=max_samples,
            )
            metadata["duration_scale"] = scale
        return value, metadata

    def _quality(self, positions, times, reference, limits) -> ValidationResult:
        ranges = (limits[:, 1] - limits[:, 0]).to(
            device=positions.device, dtype=torch.float64
        )
        actual_length = float(
            torch.linalg.vector_norm(
                positions.double().diff(dim=0) / ranges, dim=1
            ).sum()
        )
        reference_length = float(
            torch.linalg.vector_norm(
                reference.positions.to(positions.device).double().diff(dim=0) / ranges,
                dim=1,
            ).sum()
        )
        duration = float(times[-1] - times[0])
        reference_duration = float(reference.dt.double().sum())
        length_ok = (
            actual_length
            <= reference_length * self.cfg.validation.path_length_ratio_max + 1e-9
        )
        duration_ok = (
            duration
            <= reference_duration * self.cfg.validation.duration_ratio_max + 1e-9
        )
        metrics = {
            "path_length": actual_length,
            "reference_path_length": reference_length,
            "duration_s": duration,
            "reference_duration_s": reference_duration,
        }
        return ValidationResult(
            (
                ValidationCheck(
                    "motion_quality",
                    "passed" if length_ok and duration_ok else "failed",
                    "actual motion compared with the same-case reference",
                    metrics,
                ),
            )
        )

    def _submit(self, session, episode) -> None:
        for submission_id in range(self.max_write_retries + 1):
            if submission_id:
                episode, submission_id = session.retry_write(episode.commit_id)
            try:
                receipt = self.sink.submit(episode, submission_id=submission_id)
            except Exception as error:
                # This concrete synchronous sink raises input errors before any
                # writes; all write/readback failures instead return a receipt.
                session.apply_receipt(
                    CommitReceipt(
                        episode.episode_id,
                        episode.identity.candidate_id,
                        episode.identity.attempt_id,
                        "unsubmitted",
                        episode.commit_id,
                        episode.identity.scene_case_id,
                        confirmed=False,
                        error=f"{type(error).__name__}: {error}",
                        submission_id=submission_id,
                    )
                )
                raise
            session.apply_receipt(receipt)
            if receipt.confirmed:
                return
        raise RuntimeError(f"Episode persistence failed: {receipt.error}")

    def _write_report(self, report) -> None:
        path = self.sink.root / "generation_report.json"
        payload = json.dumps(
            _plain(report),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode()
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=".generation-report.", delete=False
            ) as stream:
                temporary = stream.name
                stream.write(payload)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)

    def run(
        self,
        cases: Sequence[SceneCase],
        templates: Sequence[TrajectoryTemplate],
        *,
        should_stop: Callable[[], bool] | None = None,
    ) -> Mapping[str, object]:
        """Run one bounded collection job and freeze its final audit.

        Args:
            cases: One caller-provided fixed case per simulator row.
            templates: Explicit full-joint references in matching physical row order.
            should_stop: Optional cancellation predicate checked between commands.

        Returns:
            Counters, coverage, audit, resolved configuration, and ``target_reached``.
            Exhausting a budget returns ``target_reached=False`` and a stop reason.

        Raises:
            ValueError: If source or integration contracts are incompatible.
            RuntimeError: If required validation, restoration, or persistence fails.
        """
        if self._ran:
            raise RuntimeError("GenerationRunner is single-use")
        self._ran = True
        session = GenerationSession(self.cfg, clock=self.clock)
        error = None
        cancelled = False
        cases, templates = tuple(cases), tuple(templates)
        try:
            self._validate_templates(cases, templates)
            binding = self.host.acquire_case(cases)
            snapshots = self.host.snapshots(binding)
            if isinstance(self.planner, PickUpMotionValidator):
                world_ids = updated_ids = set(self.planner.collision_world_entity_ids)
            else:
                world_ids = set(
                    self.planner.motion_generator.collision_world_entity_ids
                )
                updated_ids = set(
                    self.planner.motion_generator.dynamic_collision_entity_ids
                )
            if any(
                set(snapshot.entity_poses) - (world_ids & updated_ids)
                for snapshot in snapshots
            ):
                raise ValueError(
                    "generation collision world must update every rigid object under its physical UID"
                )
            limits = self.host.adapter.robot.get_qpos_limits().detach().clone()
            for case, template, row_limits in zip(cases, templates, limits):
                session.register_case(
                    case, row_limits, joint_names=template.joint_names
                )
            plan_checks = {}
            reference_by_candidate = {}
            ready_by_row = set()
            row_offset = 0
            needs_restore = False

            def stop() -> bool:
                return bool(
                    session.stop_reason or (should_stop is not None and should_stop())
                )

            while not stop():
                order = [
                    (row_offset + offset) % len(cases) for offset in range(len(cases))
                ]
                for row in order:
                    if (
                        row in ready_by_row
                        or stop()
                        or session.snapshot()["counts"]["ready"]
                        >= self.cfg.execution.ready_high_watermark
                    ):
                        continue
                    if (
                        session.snapshot()["counts"]["proposed"]
                        >= self.cfg.collection.max_proposals
                    ):
                        break
                    case, reference = cases[row], templates[row]
                    family = None
                    if not self.cfg.augmentation.factors.spatial.enabled:
                        family = (
                            "geometry_"
                            + hashlib.sha256(
                                json.dumps(
                                    (
                                        case.scene_case_id,
                                        case.initial_state_id,
                                        reference.source_id,
                                        reference.source_revision,
                                        reference.template_id,
                                    )
                                ).encode()
                            ).hexdigest()
                        )
                    identity, generator = session.propose(
                        case.scene_case_id,
                        case.initial_state_id,
                        source_id=reference.source_id,
                        source_revision=reference.source_revision,
                        template_id=reference.template_id,
                        operator_id="configured_factors",
                        geometry_family_id=family,
                    )
                    try:
                        variant, factors = self._variant(
                            reference, limits[row], generator
                        )
                        batch = CandidateTrajectoryBatch(
                            variant.positions.unsqueeze(0),
                            variant.dt.unsqueeze(0),
                            torch.tensor([len(variant.positions)], dtype=torch.int64),
                            (identity,),
                            variant.joint_names,
                            (variant.phases,),
                            (factors,),
                            torch.tensor([row], dtype=torch.int64),
                        )
                        collision = self.planner.validate_qpos(batch, snapshots)[0]
                        dynamics = validate_motion_limits(
                            variant,
                            velocity_limits=self.motion_limits.velocity_limits,
                            acceleration_limits=self.motion_limits.acceleration_limits,
                        )
                        validation = ValidationResult(
                            (*collision.checks, *dynamics.checks)
                        )
                        try:
                            session.add_planned(batch, validation)
                        except ValueError:
                            if any(
                                check.status in {"unavailable", "not_run"}
                                for check in validation.checks
                            ):
                                raise RuntimeError(
                                    f"Required planning capability is unavailable: {validation.checks}"
                                ) from None
                            raise
                    except (ValueError, BufferError) as failure:
                        session.release(
                            identity, reason=f"planning rejected: {failure}"
                        )
                        continue
                    plan_checks[identity.candidate_id] = validation
                    reference_by_candidate[identity.candidate_id] = reference
                    ready_by_row.add(row)
                if stop():
                    break
                selected = [None] * len(cases)
                for row in order:
                    case = cases[row]
                    selected[row] = session.take_ready(
                        case.scene_case_id,
                        case.initial_state_id,
                        episode_byte_budget=self._episode_budget,
                    )
                    if selected[row] is not None:
                        ready_by_row.discard(row)
                assigned = [
                    candidate for candidate in selected if candidate is not None
                ]
                if not assigned:
                    if session.stop_reason:
                        break
                    if (
                        session.snapshot()["counts"]["proposed"]
                        >= self.cfg.collection.max_proposals
                    ):
                        break
                    continue
                if stop():
                    break
                if needs_restore:
                    binding = self.host.restore_initial()
                verified = self.host.verify_initial(binding)
                if not verified.accepted:
                    raise RuntimeError(f"Initial state mismatch: {verified.checks}")
                if stop():
                    break
                ids = {
                    candidate.identities[0].candidate_id: session.episode_ids(
                        candidate.identities[0]
                    )
                    for candidate in assigned
                }
                episodes = self.executor.execute(
                    binding,
                    selected,
                    ids,
                    on_started=session.mark_rollout_started,
                    should_stop=stop,
                )
                needs_restore = True
                if len(episodes) != len(cases):
                    raise ValueError(
                        "executor results must preserve physical row order"
                    )
                for row, (candidate, episode) in enumerate(zip(selected, episodes)):
                    if candidate is None:
                        if episode is not None:
                            raise ValueError(
                                "executor produced evidence for an inactive row"
                            )
                        continue
                    identity = candidate.identities[0]
                    if episode is None:
                        plan_checks.pop(identity.candidate_id, None)
                        reference_by_candidate.pop(identity.candidate_id, None)
                        session.release(
                            identity,
                            reason=self.executor.last_failures.get(
                                identity.candidate_id,
                                "execution produced no complete evidence",
                            ),
                        )
                        continue
                    if episode.identity != identity:
                        raise ValueError(
                            "executor evidence belongs to a different candidate"
                        )
                    reference = reference_by_candidate.pop(identity.candidate_id)
                    qpos = episode.observations["joint_positions"]
                    quality = self._quality(
                        qpos, episode.timestamps, reference, limits[row]
                    )
                    measured = replace(
                        reference,
                        positions=qpos,
                        dt=torch.cat(
                            (episode.timestamps.new_zeros(1), episode.timestamps.diff())
                        ),
                        phases=episode.phases,
                    )
                    dynamic = validate_motion_limits(
                        measured,
                        velocity_limits=self.motion_limits.velocity_limits,
                        acceleration_limits=self.motion_limits.acceleration_limits,
                    )
                    # Revalidate measured joints. Free motion uses the captured
                    # fixed world; PickUp uses actual target poses, with root
                    # and all other objects verified by the executor.
                    actual_snapshots = list(snapshots)
                    actual_snapshots[row] = replace(
                        snapshots[row], joint_positions=qpos[0]
                    )
                    actual_batch = CandidateTrajectoryBatch(
                        qpos.unsqueeze(0),
                        measured.dt.unsqueeze(0),
                        torch.tensor([len(qpos)], dtype=torch.int64),
                        (identity,),
                        candidate.joint_names,
                        (episode.phases,),
                        source_row_indices=torch.tensor([row], dtype=torch.int64),
                    )
                    if isinstance(self.planner, PickUpMotionValidator):
                        actual_collision = self.planner.validate_episode(
                            episode,
                            actual_snapshots[row],
                            row=row,
                        )
                    else:
                        actual_collision = self.planner.validate_qpos(
                            actual_batch,
                            actual_snapshots,
                        )[0]
                    all_checks = (
                        *tuple(
                            replace(check, check_id="planned_" + check.check_id)
                            for check in plan_checks.pop(identity.candidate_id).checks
                        ),
                        *actual_collision.checks,
                        *episode.validation.checks,
                        *quality.checks,
                        *tuple(
                            ValidationCheck(
                                "actual_" + check.check_id,
                                check.status,
                                check.detail,
                                check.metrics,
                            )
                            for check in dynamic.checks
                        ),
                    )
                    accepted = replace(episode, validation=ValidationResult(all_checks))
                    if session.accept_episode(accepted):
                        self._submit(session, accepted)
                row_offset = (order[0] + 1) % len(cases)
            cancelled = (
                should_stop is not None
                and should_stop()
                and session.stop_reason is None
            )
        except BaseException as failure:
            error = failure
        finally:
            try:
                for receipt in self.sink.drain():
                    session.apply_receipt(receipt)
            except BaseException as failure:
                if error is None:
                    error = failure
            for identity, state, _ in session.snapshot()["audit"]:
                if state in {
                    "proposed",
                    "ready",
                    "assigned",
                    "running",
                    "write_failed",
                }:
                    session.release(
                        identity,
                        reason="job stopped" if error is None else "job failed",
                    )
            for close in (self.sink.close, self.host.close):
                try:
                    close()
                except BaseException as failure:
                    if error is None:
                        error = failure
            report = {
                **dict(session.snapshot()),
                "configuration": self.cfg.to_dict(),
                "motion_limits": self.motion_limits,
                "cases": cases,
                "version": __version__,
                "target_reached": session.snapshot()["counts"]["committed"]
                >= self.cfg.collection.target_committed_episodes,
                "cancelled": cancelled,
                "error": None if error is None else f"{type(error).__name__}: {error}",
            }
            try:
                self._write_report(report)
            except BaseException as failure:
                if error is None:
                    error = failure
                else:
                    add_note = getattr(error, "add_note", None)
                    if callable(add_note):
                        add_note(
                            f"Writing generation_report.json also failed: {failure}"
                        )
                    else:
                        error.__context__ = failure
        if error is not None:
            raise error
        return report
