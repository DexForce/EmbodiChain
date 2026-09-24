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

import math
from dataclasses import dataclass
from typing import Callable, Protocol, TYPE_CHECKING, Literal, Sequence

import torch

from embodichain.compute.trajectory import retime_to_control_grid
from embodichain.utils import configclass

from .expansion.contracts import CommitReceipt, ExpertEpisode, ValidationResult
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
    "FixedSceneInitialStatePort",
    "SingleSlotOutcome",
    "SingleSlotRunner",
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
