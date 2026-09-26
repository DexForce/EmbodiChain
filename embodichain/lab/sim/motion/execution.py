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

"""Fixed-cadence execution of timed joint trajectories in simulation."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable, Protocol, TYPE_CHECKING, Literal, Sequence

import numpy as np
import torch

from embodichain.compute.trajectory import retime_to_control_grid
from embodichain.utils import configclass

from .expansion.contracts import CommitReceipt, ExpertEpisode, ValidationResult
from .expansion.combined import PhysicalSlotPool
from .expansion.coordinator import CandidateCoordinator, CandidateWorkItem
from .expansion.session import GenerationSession

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager

JointCommandMode = Literal["position", "position_velocity"]

__all__ = [
    "JointTrajectoryPlaybackCfg",
    "play_joint_trajectory",
    "InitialStatePort",
    "MeasuredExecutor",
    "EpisodeSink",
    "LocalArtifactSink",
    "FixedSceneInitialStatePort",
    "SingleSlotOutcome",
    "SingleSlotRunner",
    "MultiSlotRunner",
]


@configclass
class JointTrajectoryPlaybackCfg:
    """Fixed-cadence standalone simulation playback settings.

    Args:
        control_dt: Command period, or ``None`` for one physics period.
        joint_command_mode: Whether to write qpos alone or qpos and qvel.
    """

    control_dt: float | None = None
    """Command period, or ``None`` to use the simulation physics period."""

    joint_command_mode: JointCommandMode = "position"
    """Whether playback writes qpos alone or qpos and qvel targets."""

    def __post_init__(self) -> None:
        if self.control_dt is not None and (
            isinstance(self.control_dt, bool)
            or not isinstance(self.control_dt, (int, float))
            or not math.isfinite(self.control_dt)
            or self.control_dt <= 0
        ):
            raise ValueError("control_dt must be a positive finite number or None.")
        if self.joint_command_mode not in ("position", "position_velocity"):
            raise ValueError(
                "joint_command_mode must be 'position' or 'position_velocity'."
            )


def _resolve_playback_timing(
    sim: SimulationManager,
    control_dt: float | None,
) -> tuple[float, float, int]:
    """Resolve one fixed command period as exact simulator substeps."""
    physics_dt = getattr(getattr(sim, "sim_config", None), "physics_dt", None)
    if (
        isinstance(physics_dt, bool)
        or not isinstance(physics_dt, (int, float))
        or not math.isfinite(physics_dt)
        or physics_dt <= 0
    ):
        raise ValueError("sim.sim_config.physics_dt must be positive and finite.")
    destination_dt = float(physics_dt if control_dt is None else control_dt)
    ratio = destination_dt / float(physics_dt)
    steps = round(ratio)
    tolerance = max(1.0e-9, abs(ratio) * 1.0e-7)
    if steps <= 0 or not math.isclose(
        ratio,
        steps,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ValueError("control_dt must be an integer multiple of physics_dt.")
    return float(physics_dt), destination_dt, steps


def play_joint_trajectory(
    sim: SimulationManager,
    robot: Robot,
    *,
    positions: torch.Tensor,
    dt: torch.Tensor,
    joint_ids: Sequence[int],
    cfg: JointTrajectoryPlaybackCfg | None = None,
) -> None:
    """Play a timed joint path without changing the simulation physics period.

    Planner timing is first quantized to the configured fixed command period.
    Every command interval advances an integer number of unchanged physics
    substeps. Velocity targets are recomputed on that executed grid.

    Args:
        sim: Simulation manager in explicit-update mode.
        robot: Robot receiving joint targets.
        positions: Source positions with shape ``(B, N, D)``.
        dt: Source arrival intervals with shape ``(B, N)``.
        joint_ids: Full-robot joint columns addressed by ``positions``.
        cfg: Playback settings. Defaults to position-only at ``physics_dt``.
    """
    if cfg is None:
        cfg = JointTrajectoryPlaybackCfg()
    if not isinstance(cfg, JointTrajectoryPlaybackCfg):
        raise TypeError("cfg must be a JointTrajectoryPlaybackCfg or None.")
    resolved_joint_ids = list(joint_ids)
    if (
        not resolved_joint_ids
        or any(
            type(joint_id) is not int or joint_id < 0 for joint_id in resolved_joint_ids
        )
        or len(set(resolved_joint_ids)) != len(resolved_joint_ids)
    ):
        raise ValueError("joint_ids must contain unique non-negative integers.")
    if not isinstance(positions, torch.Tensor) or positions.dim() != 3:
        raise ValueError("positions must have shape (B, N, D).")
    if positions.shape[-1] != len(resolved_joint_ids):
        raise ValueError("positions DoF must match joint_ids.")

    physics_dt, control_dt, physics_steps = _resolve_playback_timing(
        sim,
        cfg.control_dt,
    )
    positions, velocities, _, _ = retime_to_control_grid(
        positions,
        dt,
        control_dt,
    )
    for waypoint in range(positions.shape[1] - 1):
        robot.set_qpos(
            positions[:, waypoint],
            joint_ids=resolved_joint_ids,
        )
        if cfg.joint_command_mode == "position_velocity":
            robot.set_qvel(
                velocities[:, waypoint],
                joint_ids=resolved_joint_ids,
            )
        sim.update(physics_dt, physics_steps)

    robot.set_qpos(positions[:, -1], joint_ids=resolved_joint_ids)
    if cfg.joint_command_mode == "position_velocity":
        robot.set_qvel(velocities[:, -1], joint_ids=resolved_joint_ids)


class InitialStatePort(Protocol):
    """Restore and verify one candidate's complete initial state."""

    def restore(self, item: CandidateWorkItem) -> object:
        """Restore a candidate and return an opaque prepared binding."""


class MeasuredExecutor(Protocol):
    """Plan-check and execute one prepared candidate."""

    def validate_plan(self, item: CandidateWorkItem) -> ValidationResult:
        """Return host-owned planning/path validation evidence."""

    def execute(
        self,
        item: CandidateWorkItem,
        prepared: object,
        *,
        episode_id: str,
        commit_id: str,
        on_rollout_started: Callable[[], None],
    ) -> ExpertEpisode:
        """Execute and report the first actual command through the callback."""


class EpisodeSink(Protocol):
    """Persist one accepted episode and return its final receipt."""

    def submit(
        self,
        episode: ExpertEpisode,
        *,
        submission_id: int,
    ) -> CommitReceipt:
        """Persist and confirm one episode synchronously or through a bounded sink."""


def _artifact_json(value: object) -> object:
    """Convert immutable contract metadata into ordinary JSON containers."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("artifact metadata must contain finite numbers")
        return value
    if isinstance(value, Mapping):
        return {str(key): _artifact_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_artifact_json(item) for item in value]
    raise ValueError(f"artifact metadata contains unsupported value {type(value)!r}")


class LocalArtifactSink:
    """Write accepted episodes to an atomic local artifact directory.

    The sink is synchronous and deliberately independent of LeRobot. A
    candidate is first assembled in a sibling temporary directory and then
    renamed into ``candidates/<candidate-id>``. A repeated submission for the
    same episode is idempotent; a different episode cannot overwrite it.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self._root = Path(output_dir)
        self._candidates = self._root / "candidates"
        self._failures = self._root / "failures"
        self._candidates.mkdir(parents=True, exist_ok=True)
        self._failures.mkdir(parents=True, exist_ok=True)

    @property
    def output_dir(self) -> Path:
        """Return the owned artifact root."""
        return self._root

    @staticmethod
    def _identity_payload(episode: ExpertEpisode) -> dict[str, object]:
        identity = episode.identity
        return {
            "candidate_id": identity.candidate_id,
            "attempt_id": identity.attempt_id,
            "scene_case_id": identity.scene_case_id,
            "initial_state_id": identity.initial_state_id,
            "source_id": identity.source_id,
            "source_revision": identity.source_revision,
            "template_id": identity.template_id,
            "geometry_family_id": identity.geometry_family_id,
        }

    @staticmethod
    def _validation_payload(episode: ExpertEpisode) -> dict[str, object]:
        return {
            "accepted": episode.validation.accepted,
            "checks": [
                {
                    "check_id": check.check_id,
                    "status": check.status,
                    "detail": check.detail,
                    "metrics": dict(check.metrics),
                }
                for check in episode.validation.checks
            ],
        }

    def _receipt(
        self,
        episode: ExpertEpisode,
        *,
        submission_id: int,
        candidate_dir: Path,
    ) -> CommitReceipt:
        return CommitReceipt(
            episode_id=episode.episode_id,
            candidate_id=episode.identity.candidate_id,
            attempt_id=episode.identity.attempt_id,
            storage_id=str(candidate_dir),
            commit_id=episode.commit_id,
            scene_case_id=episode.identity.scene_case_id,
            submission_id=submission_id,
        )

    def submit(
        self,
        episode: ExpertEpisode,
        *,
        submission_id: int,
    ) -> CommitReceipt:
        """Atomically persist one accepted episode and return its receipt."""
        if not isinstance(episode, ExpertEpisode):
            raise TypeError("episode must be an ExpertEpisode")
        if type(submission_id) is not int or submission_id < 0:
            raise ValueError("submission_id must be a non-negative integer")
        if not episode.validation.accepted:
            raise ValueError("only accepted episodes may be persisted")

        candidate_id = episode.identity.candidate_id
        if candidate_id in {".", ".."} or Path(candidate_id).name != candidate_id:
            raise ValueError("candidate_id must be a single safe path component")
        candidate_dir = self._candidates / candidate_id
        existing_episode = candidate_dir / "physical_episode.json"
        if candidate_dir.exists():
            if not existing_episode.is_file():
                raise RuntimeError(
                    "candidate artifact exists without a physical episode"
                )
            existing = json.loads(existing_episode.read_text(encoding="utf-8"))
            if existing.get("episode_id") != episode.episode_id:
                raise RuntimeError("candidate artifact belongs to a different episode")
            return self._receipt(
                episode,
                submission_id=submission_id,
                candidate_dir=candidate_dir,
            )

        temporary = Path(
            tempfile.mkdtemp(prefix=f".{candidate_id}.", dir=self._candidates)
        )
        try:
            generation = {
                **self._identity_payload(episode),
                "episode_id": episode.episode_id,
                "commit_id": episode.commit_id,
                "action_representation": episode.action_representation,
                "metadata": _artifact_json(episode.metadata),
            }
            (temporary / "generation.json").write_text(
                json.dumps(generation, allow_nan=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            physical = {
                "episode_id": episode.episode_id,
                "commit_id": episode.commit_id,
                "identity": self._identity_payload(episode),
                "validation": self._validation_payload(episode),
                "timestamps": episode.timestamps.detach().cpu().tolist(),
                "phases": [
                    {
                        "phase_id": phase.phase_id,
                        "start_index": phase.start_index,
                        "stop_index": phase.stop_index,
                        "kind": phase.kind,
                    }
                    for phase in episode.phases
                ],
            }
            (temporary / "physical_episode.json").write_text(
                json.dumps(physical, allow_nan=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            torch.save(
                {
                    "actions": episode.actions.detach().cpu(),
                    "timestamps": episode.timestamps.detach().cpu(),
                },
                temporary / "trajectory.pt",
            )
            observations_dir = temporary / "observations"
            observations_dir.mkdir()
            np.savez_compressed(
                observations_dir / "rgb.npz",
                **{
                    key.replace("/", "__"): value.detach().cpu().numpy()
                    for key, value in episode.observations.items()
                },
            )
            os.replace(temporary, candidate_dir)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return self._receipt(
            episode,
            submission_id=submission_id,
            candidate_dir=candidate_dir,
        )

    def write_manifest(self, manifest: Mapping[str, object]) -> Path:
        """Atomically replace the run manifest and return its path."""
        payload = _artifact_json(manifest)
        if not isinstance(payload, dict):
            raise TypeError("manifest must be a mapping")
        target = self._root / "manifest.json"
        temporary = self._root / ".manifest.json.tmp"
        temporary.write_text(
            json.dumps(payload, allow_nan=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, target)
        return target


class FixedSceneInitialStatePort:
    """Adapt fixed-scene restore/verify methods to :class:`InitialStatePort`."""

    def __init__(self, host: object) -> None:
        if not callable(getattr(host, "restore_initial", None)):
            raise TypeError("host must provide restore_initial()")
        if not callable(getattr(host, "verify_initial", None)):
            raise TypeError("host must provide verify_initial(binding)")
        self._host = host

    def restore(self, item: CandidateWorkItem) -> object:
        """Restore and verify the host-owned fixed initial state."""
        del item
        binding = self._host.restore_initial()
        result = self._host.verify_initial(binding)
        if not getattr(result, "accepted", False):
            raise RuntimeError("fixed-scene initial-state verification failed")
        return binding


@dataclass(frozen=True)
class SingleSlotOutcome:
    """Result of one coordinator-driven physical candidate attempt."""

    status: str
    candidate_id: str | None = None
    receipt: CommitReceipt | None = None
    reason: str | None = None
    episode_id: str | None = None
    commit_id: str | None = None
    submission_id: int | None = None


class SingleSlotRunner:
    """Drive one candidate through host restore, execution, and persistence."""

    def __init__(
        self,
        coordinator: CandidateCoordinator,
        restorer: InitialStatePort,
        executor: MeasuredExecutor,
        sink: EpisodeSink,
        *,
        episode_byte_budget: int,
    ) -> None:
        if not isinstance(coordinator, CandidateCoordinator):
            raise TypeError("coordinator must be a CandidateCoordinator")
        for name, value in (
            ("restorer", restorer),
            ("executor", executor),
            ("sink", sink),
        ):
            if value is None:
                raise TypeError(f"{name} must be supplied")
        if type(episode_byte_budget) is not int or episode_byte_budget <= 0:
            raise ValueError("episode_byte_budget must be a positive integer")
        if episode_byte_budget > coordinator.session.pending_max_bytes:
            raise ValueError(
                "episode_byte_budget must be within session pending_max_bytes"
            )
        self._coordinator = coordinator
        self._restorer = restorer
        self._executor = executor
        self._sink = sink
        self._episode_byte_budget = episode_byte_budget

    @staticmethod
    def _release_if_uncommitted(
        session: GenerationSession,
        item: CandidateWorkItem,
        *,
        reason: str,
    ) -> None:
        """Release a candidate unless its owning operation already did so."""
        try:
            session.release(item.spec.identity, reason=reason)
        except ValueError:
            pass

    def run_next(self) -> SingleSlotOutcome:
        """Execute the next FIFO candidate, or report an empty queue."""
        item = self._coordinator.take_next()
        if item is None:
            return SingleSlotOutcome("no_candidate")
        identity = item.spec.identity
        candidate_id = identity.candidate_id
        session = self._coordinator.session
        try:
            validation = self._executor.validate_plan(item)
            self._coordinator.admit_planned(item, validation)
        except Exception as error:
            self._release_if_uncommitted(
                session,
                item,
                reason="planning_failed",
            )
            return SingleSlotOutcome(
                "planning_rejected",
                candidate_id=candidate_id,
                reason=str(error),
            )

        try:
            ready = session.take_ready(
                identity.scene_case_id,
                identity.initial_state_id,
                episode_byte_budget=self._episode_byte_budget,
                candidate_id=candidate_id,
            )
        except Exception:
            self._release_if_uncommitted(
                session,
                item,
                reason="assignment_failed",
            )
            raise
        if ready is None:
            session.release(identity, reason="execution_capacity_unavailable")
            return SingleSlotOutcome("capacity_unavailable", candidate_id=candidate_id)
        if ready.identities != (identity,):
            session.release(identity, reason="assigned_candidate_mismatch")
            raise RuntimeError("session assigned a different candidate")

        episode_id, commit_id = session.episode_ids(identity)
        started = False

        def on_rollout_started() -> None:
            nonlocal started
            if started:
                raise RuntimeError("rollout start callback may run only once")
            session.mark_rollout_started(identity)
            started = True

        try:
            prepared = self._restorer.restore(item)
            episode = self._executor.execute(
                item,
                prepared,
                episode_id=episode_id,
                commit_id=commit_id,
                on_rollout_started=on_rollout_started,
            )
            if not started:
                raise RuntimeError("executor returned before starting the rollout")
        except Exception:
            self._release_if_uncommitted(
                session,
                item,
                reason="execution_failed",
            )
            raise

        try:
            accepted = session.accept_episode(episode)
        except Exception:
            self._release_if_uncommitted(
                session,
                item,
                reason="measured_admission_failed",
            )
            raise
        if not accepted:
            return SingleSlotOutcome("measured_rejected", candidate_id=candidate_id)

        submission_id = 0
        try:
            receipt = self._sink.submit(
                episode,
                submission_id=submission_id,
            )
        except Exception as error:
            return SingleSlotOutcome(
                "write_pending",
                candidate_id=candidate_id,
                reason=str(error),
                episode_id=episode_id,
                commit_id=commit_id,
                submission_id=submission_id,
            )
        expected_receipt_identity = (
            episode.episode_id,
            identity.candidate_id,
            identity.attempt_id,
            episode.commit_id,
            identity.scene_case_id,
            submission_id,
        )
        receipt_identity = (
            (
                receipt.episode_id,
                receipt.candidate_id,
                receipt.attempt_id,
                receipt.commit_id,
                receipt.scene_case_id,
                receipt.submission_id,
            )
            if isinstance(receipt, CommitReceipt)
            else None
        )
        if receipt_identity != expected_receipt_identity:
            return SingleSlotOutcome(
                "write_pending",
                candidate_id=candidate_id,
                reason="sink receipt does not match submitted episode",
                episode_id=episode_id,
                commit_id=commit_id,
                submission_id=submission_id,
            )
        session.apply_receipt(receipt)
        return SingleSlotOutcome(
            "committed" if receipt.confirmed else "write_failed",
            candidate_id=candidate_id,
            receipt=receipt,
            reason=receipt.error or None,
            episode_id=episode_id,
            commit_id=commit_id,
            submission_id=submission_id,
        )

    def run_until_empty(
        self,
        *,
        max_attempts: int | None = None,
    ) -> tuple[SingleSlotOutcome, ...]:
        """Consume the current bounded queue through the same single slot."""
        if max_attempts is not None and (
            type(max_attempts) is not int or max_attempts < 0
        ):
            raise ValueError("max_attempts must be a non-negative integer or None")
        outcomes: list[SingleSlotOutcome] = []
        while self._coordinator.pending_count:
            if max_attempts is not None and len(outcomes) >= max_attempts:
                raise RuntimeError("single-slot attempt budget exhausted")
            outcomes.append(self.run_next())
        return tuple(outcomes)

    @property
    def coordinator(self) -> CandidateCoordinator:
        """Return the coordinator owned by this runner."""
        return self._coordinator


class MultiSlotRunner:
    """Assign queued candidates to compatible slots around a single host runner.

    The current implementation drains FIFO synchronously; the slot pool and
    exact reservation lifecycle are independent of that execution policy, so a
    host can replace the inner runner with a concurrent implementation later.
    """

    def __init__(
        self,
        runner: SingleSlotRunner,
        slot_pool: PhysicalSlotPool,
    ) -> None:
        if not isinstance(runner, SingleSlotRunner):
            raise TypeError("runner must be a SingleSlotRunner")
        if not isinstance(slot_pool, PhysicalSlotPool):
            raise TypeError("slot_pool must be a PhysicalSlotPool")
        self._runner = runner
        self._slot_pool = slot_pool

    @property
    def slot_pool(self) -> PhysicalSlotPool:
        """Return the host-owned physical slot pool."""
        return self._slot_pool

    def run_next(self) -> SingleSlotOutcome:
        """Reserve a compatible slot, run one candidate, and release it."""
        item = self._runner.coordinator.peek_next()
        if item is None:
            return SingleSlotOutcome("no_candidate")
        candidate_id = item.spec.identity.candidate_id
        try:
            reservation = self._slot_pool.reserve(
                candidate_id,
                item.spec.compatibility_key,
            )
        except BufferError:
            return SingleSlotOutcome(
                "capacity_unavailable",
                candidate_id=candidate_id,
                reason="no compatible physical slot is available",
            )
        try:
            outcome = self._runner.run_next()
            if outcome.candidate_id != candidate_id:
                raise RuntimeError("runner consumed a different reserved candidate")
            return outcome
        finally:
            self._slot_pool.release(reservation)

    def run_until_empty(
        self,
        *,
        max_attempts: int | None = None,
    ) -> tuple[SingleSlotOutcome, ...]:
        """Drain FIFO candidates while respecting the optional attempt bound."""
        if max_attempts is not None and (
            type(max_attempts) is not int or max_attempts < 0
        ):
            raise ValueError("max_attempts must be a non-negative integer or None")
        outcomes: list[SingleSlotOutcome] = []
        while self._runner.coordinator.pending_count:
            if max_attempts is not None and len(outcomes) >= max_attempts:
                raise RuntimeError("multi-slot attempt budget exhausted")
            outcome = self.run_next()
            if outcome.status == "capacity_unavailable":
                break
            outcomes.append(outcome)
        return tuple(outcomes)
